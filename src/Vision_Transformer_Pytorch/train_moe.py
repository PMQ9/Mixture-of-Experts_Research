import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
import torch.nn as nn
from tqdm import tqdm
import time
import os
import sys
from datetime import datetime
import numpy as np
import csv
from PIL import Image
from torch.cuda.amp import autocast, GradScaler
from torchvision.transforms import RandAugment
import argparse
from dataclasses import fields, asdict
import torch.multiprocessing
import logging

from vision_transformer_moe import VisionTransformer, VisionTransformerConfig, LabelSmoothingCrossEntropy, TrafficSignTrainDataset, TrafficSignTestDataset
from vision_transformer_moe import MetaMoE, MetaGatingNet, CombinedDataset
from log_functions import setup_logging, archive_params, plot_metrics, export_to_onnx
from augmentation_functions import cutmix
from config import (
    DEFAULT_BATCH_SIZE, DEFAULT_EPOCH, DEFAULT_LEARNING_RATE,
    DEFAULT_CUTMIX_ALPHA, DEFAULT_CUTMIX_PROB, DEFAULT_WARMUP_EPOCH, DEFAULT_LABEL_SMOOTHING,
    DEFAULT_TEST_START_EPOCH, DEFAULT_TEST_FREQUENCY,
    NORM_MEAN_R_GTSRB, NORM_MEAN_G_GTSRB, NORM_MEAN_B_GTSRB,
    NORM_STD_R_GTSRB, NORM_STD_G_GTSRB, NORM_STD_B_GTSRB,
    NORM_MEAN_R_PTSD, NORM_MEAN_G_PTSD, NORM_MEAN_B_PTSD,
    NORM_STD_R_PTSD, NORM_STD_G_PTSD, NORM_STD_B_PTSD,
    NORM_MEAN_R_UNIFIED, NORM_MEAN_G_UNIFIED, NORM_MEAN_B_UNIFIED,
    NORM_STD_R_UNIFIED, NORM_STD_G_UNIFIED, NORM_STD_B_UNIFIED,
)
from config import apply_config_overrides

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'artifacts'))
PRETRAINED_MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'artifacts', 'results'))

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
if os.name != 'nt':  # Only for Linux
    torch.multiprocessing.set_sharing_strategy('file_system')  # Prevents hangs

# **************** Argument Parser ****************
parser = argparse.ArgumentParser(description='Train a Vision Transformer with MoE')
parser.add_argument('--dataset', type=str, default='GTSRB', choices=['GTSRB', 'PTSD'], help='Dataset to train')
parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE, help='Batch size for training')
parser.add_argument('--epochs', type=int, default=int(os.getenv('CICD_EPOCH', DEFAULT_EPOCH)), help='Number of epochs to train')
parser.add_argument('--learning_rate', type=float, default=DEFAULT_LEARNING_RATE, help='Learning rate for optimizer')
parser.add_argument('--cutmix_alpha', type=float, default=DEFAULT_CUTMIX_ALPHA, help='Alpha parameter for CutMix')
parser.add_argument('--cutmix_prob', type=float, default=DEFAULT_CUTMIX_PROB, help='Probability of applying CutMix')
parser.add_argument('--test_start_epoch', type=int, default=DEFAULT_TEST_START_EPOCH, help='Epoch to start testing')
parser.add_argument('--test_frequency', type=int, default=DEFAULT_TEST_FREQUENCY, help='Frequency of testing in epochs')
parser.add_argument('--warmup_epochs', type=int, default=DEFAULT_WARMUP_EPOCH, help='Number of warmup epochs')
parser.add_argument('--label_smoothing', type=float, default=DEFAULT_LABEL_SMOOTHING, help='Label smoothing factor')
parser.add_argument('--archive_params', type=bool, default=True, help='Save full training params')
parser.add_argument('--export_onnx', type=bool, default=True, help='Export trained model to ONNX')
parser.add_argument('--meta_moe', action='store_true', help='Train MetaMoE model with pre-trained GTSRB and PTSD experts')
parser.add_argument('--save_state_dict', action='store_true', help='Additionally save state_dict for non-MetaMoE models')
parser.add_argument('--gating_loss_weight', type=float, default=1.0, help='Weight for MetaGatingNet supervision loss')

config_fields = [f.name for f in fields(VisionTransformerConfig)]
help_msg = f"Comma-separated list of config overrides, e.g., 'img_size=48,patch_size=8'. Available parameters: {', '.join(config_fields)}"
parser.add_argument('--config_overrides', type=str, default='', help=help_msg)
args = parser.parse_args()

# **************** Training Params ****************
BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
LEARNING_RATE = args.learning_rate
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CUTMIX_ALPHA = args.cutmix_alpha
CUTMIX_PROB = args.cutmix_prob
TEST_START_EPOCH = args.test_start_epoch
TEST_FREQUENCY = args.test_frequency
WARMUP_EPOCHS = args.warmup_epochs
LABEL_SMOOTHING = args.label_smoothing
GATING_LOSS_WEIGHT = args.gating_loss_weight

def train(model, loader, optimizer, criterion, device, balance_loss_weight=None, default_meta_class=None):
    model.train()
    total_loss = 0
    total_balance_loss = 0
    total_gating_loss = 0
    correct = 0
    gating_correct = 0
    total = 0
    scaler = torch.amp.GradScaler(enabled=True)
    gating_criterion = nn.CrossEntropyLoss()
    
    for batch_idx, batch in enumerate(tqdm(loader, desc="Training")):
        if args.meta_moe:
            data, target, meta_class = batch
        else:
            data, target, meta_class = batch
            meta_class = torch.full(target.size(), default_meta_class, dtype=torch.long, device=device)
        
        data, target, meta_class = data.to(device, non_blocking=True), target.to(device, non_blocking=True), meta_class.to(device, non_blocking=True)
        optimizer.zero_grad()

        apply_cutmix = data.size(0) == BATCH_SIZE and np.random.rand() < CUTMIX_PROB

        with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
            if args.meta_moe:
                output, gates = model(data)
                cls_loss = criterion(output, target)
                gating_loss = gating_criterion(gates, meta_class)
                total_loss_combined = cls_loss + GATING_LOSS_WEIGHT * gating_loss
            else:
                if apply_cutmix:
                    data, target_a, target_b, lam = cutmix(data, target, CUTMIX_ALPHA)
                    output, balance_losses = model(data)
                    loss_a = criterion(output, target_a)
                    loss_b = criterion(output, target_b)
                    cls_loss = lam * loss_a + (1 - lam) * loss_b
                else:
                    output, balance_losses = model(data)
                    cls_loss = criterion(output, target)
            
                # Aggregate balance loss across all MoE blocks
                balance_loss = sum(balance_losses) / len(balance_losses) if isinstance(balance_losses, list) else balance_losses
                total_loss_combined = cls_loss + balance_loss_weight * balance_loss
        
        scaler.scale(total_loss_combined).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += cls_loss.item()
        if not args.meta_moe:
            total_balance_loss += balance_loss.item()
        else:
            total_gating_loss += gating_loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        if not args.meta_moe and apply_cutmix:
            correct += lam * predicted.eq(target_a).sum().item() + (1 - lam) * predicted.eq(target_b).sum().item()
        else:
            correct += predicted.eq(target).sum().item()
        if args.meta_moe:
            _, gating_pred = gates.max(1)
            gating_correct += gating_pred.eq(meta_class).sum().item()
        
    avg_loss = total_loss / len(loader)
    avg_balance_loss = total_balance_loss / len(loader) if not args.meta_moe else 0
    avg_gating_loss = total_gating_loss / len(loader) if args.meta_moe else 0
    accuracy = correct / total
    gating_accuracy = gating_correct / total if args.meta_moe else 0
    
    logger.info(f"Train function returning: loss={avg_loss:.4f}, balance_loss={avg_balance_loss:.4f}, gating_loss={avg_gating_loss:.4f}, accuracy={accuracy:.4f}, gating_accuracy={gating_accuracy:.4f}")
    return avg_loss, avg_balance_loss, avg_gating_loss, accuracy, gating_accuracy

def test(model, loader, optimizer, criterion, device, default_meta_class=None):
    model.eval()
    total_loss = 0
    total_balance_loss = 0
    total_gating_loss = 0
    correct = 0
    gating_correct = 0
    total = 0
    gtsrb_correct = 0
    gtsrb_total = 0
    ptsd_correct = 0
    ptsd_total = 0
    gating_criterion = nn.CrossEntropyLoss()
    inference_times = []
    
    with torch.no_grad():
        for batch_idx, (data, target, meta_class) in enumerate(tqdm(loader, desc="Testing")):
            data, target, meta_class = data.to(device), target.to(device), meta_class.to(device)
            start_time = time.time()
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                if args.meta_moe:
                    output, gates = model(data)
                    loss = criterion(output, target)
                    gating_loss = gating_criterion(gates, meta_class)
                    balance_loss = 0
                else:
                    output, balance_losses = model(data)
                    loss = criterion(output, target)
                    balance_loss = sum(balance_losses) / len(balance_losses) if isinstance(balance_losses, list) else balance_losses
                    gating_loss = 0
            inference_time = time.time() - start_time
            inference_times.append(inference_time / data.size(0))
            total_loss += loss.item()
            if not args.meta_moe:
                total_balance_loss += balance_loss.item()
            else:
                total_gating_loss += gating_loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            if args.meta_moe:
                _, gating_pred = gates.max(1)
                gating_correct += gating_pred.eq(meta_class).sum().item()
                gtsrb_mask = meta_class == 0
                ptsd_mask = meta_class == 1
                if gtsrb_mask.any():
                    avg_W_G = gates[gtsrb_mask, 1].mean().item()
                    print(f"Average W_G for GTSRB samples: {avg_W_G:.4f}")
                    
                    gtsrb_correct += predicted[gtsrb_mask].eq(target[gtsrb_mask]).sum().item()
                    gtsrb_total += gtsrb_mask.sum().item()
                    
                    gtsrb_pred = predicted[gtsrb_mask]
                    num_classes_ptsd = 43
                    gtsrb_misrouted = (gtsrb_pred < num_classes_ptsd).sum().item()
                    gtsrb_misrouted_percentage = (gtsrb_misrouted/gtsrb_total)*100
                    print(f"{gtsrb_misrouted}/{gtsrb_total} GTSRB samples predicted as PTSD classes, percentage: {gtsrb_misrouted_percentage} %")
                    
                if ptsd_mask.any():
                    avg_W_P = gates[ptsd_mask, 1].mean().item()
                    print(f"Average W_P for PTSD samples: {avg_W_P:.4f}")
                       
                    ptsd_correct += predicted[ptsd_mask].eq(target[ptsd_mask]).sum().item()
                    ptsd_total += ptsd_mask.sum().item()
                    
                    ptsd_pred = predicted[ptsd_mask]
                    num_classes_gtsrb = 43
                    ptsd_misrouted = (ptsd_pred < num_classes_gtsrb).sum().item()
                    ptsd_misrouted_percentage = (ptsd_misrouted/ptsd_total)*100
                    print(f"{ptsd_misrouted}/{ptsd_total} PTSD samples predicted as GTSRB classes, percentage: {ptsd_misrouted_percentage} %")
        
    avg_loss = total_loss / len(loader)
    avg_balance_loss = total_balance_loss / len(loader) if not args.meta_moe else 0
    avg_gating_loss = total_gating_loss / len(loader) if args.meta_moe else 0
    accuracy = correct / total
    gating_accuracy = gating_correct / total if args.meta_moe else 0
    gtsrb_accuracy = gtsrb_correct / gtsrb_total if gtsrb_total > 0 and args.meta_moe else 0
    ptsd_accuracy = ptsd_correct / ptsd_total if ptsd_total > 0 and args.meta_moe else 0
    avg_inference_time = sum(inference_times) / len(inference_times) if inference_times else 0
    
    logger.info(f"Test function returning: loss={avg_loss:.4f}, balance_loss={avg_balance_loss:.4f}, gating_loss={avg_gating_loss:.4f}, accuracy={accuracy:.4f}, gating_accuracy={gating_accuracy:.4f}, gtsrb_accuracy={gtsrb_accuracy:.4f}, ptsd_accuracy={ptsd_accuracy:.4f}, avg_inference_time={avg_inference_time:.6f} seconds/image")
    return avg_loss, avg_balance_loss, avg_gating_loss, accuracy, gating_accuracy, gtsrb_accuracy, ptsd_accuracy, avg_inference_time

def main():
    default_meta_class = None
    if args.meta_moe:
        num_classes_gtsrb = 43
        num_classes_ptsd = 43
        total_classes = num_classes_gtsrb + num_classes_ptsd
        normalization_mean = (NORM_MEAN_R_UNIFIED, NORM_MEAN_G_UNIFIED, NORM_MEAN_B_UNIFIED)
        normalization_std = (NORM_STD_R_UNIFIED, NORM_STD_G_UNIFIED, NORM_STD_B_UNIFIED)
        
        # Validate dataset paths
        gtsrb_train_dir = './data/GTSRB/Training'
        gtsrb_train_csv = './data/GTSRB/Training/train_with_meta_class.csv'
        ptsd_train_dir = './data/PTSD/Training'
        ptsd_train_csv = './data/PTSD/Training/train_with_meta_class.csv'
        gtsrb_test_dir = './data/GTSRB/Test'
        gtsrb_test_csv = './data/GTSRB/Test/testset_with_meta_class.csv'
        ptsd_test_dir = './data/PTSD/Test'
        ptsd_test_csv = './data/PTSD/Test/testset_with_meta_class.csv'
        
        for path, desc in [
            (gtsrb_train_dir, "GTSRB training directory"),
            (gtsrb_train_csv, "GTSRB training CSV"),
            (ptsd_train_dir, "PTSD training directory"),
            (ptsd_train_csv, "PTSD training CSV"),
            (gtsrb_test_dir, "GTSRB test directory"),
            (gtsrb_test_csv, "GTSRB test CSV"),
            (ptsd_test_dir, "PTSD test directory"),
            (ptsd_test_csv, "PTSD test CSV"),
        ]:
            if not os.path.exists(path):
                logger.error(f"{desc} not found at {path}")
                raise FileNotFoundError(f"{desc} not found at {path}")
    
    else:
        if args.dataset == 'GTSRB':
            num_classes = 43
            train_dir = './data/GTSRB/Training'
            test_dir = './data/GTSRB/Test'
            csv_file = './data/GTSRB/Test/testset_with_meta_class.csv'
            normalization_mean = (NORM_MEAN_R_GTSRB, NORM_MEAN_G_GTSRB, NORM_MEAN_B_GTSRB)
            normalization_std = (NORM_STD_R_GTSRB, NORM_STD_G_GTSRB, NORM_STD_B_GTSRB)
            default_meta_class = 0
        elif args.dataset == 'PTSD':
            num_classes = 43
            train_dir = './data/PTSD/Training'
            test_dir = './data/PTSD/Test'
            csv_file = './data/PTSD/Test/testset_with_meta_class.csv'
            normalization_mean = (NORM_MEAN_R_PTSD, NORM_MEAN_G_PTSD, NORM_MEAN_B_PTSD)
            normalization_std = (NORM_STD_R_PTSD, NORM_STD_G_PTSD, NORM_STD_B_PTSD)
            default_meta_class = 1
        else:
            raise ValueError(f"Unknown dataset: {args.dataset}")
        
        # Validate non-MetaMoE paths
        for path, desc in [
            (train_dir, f"{args.dataset} training directory"),
            (test_dir, f"{args.dataset} test directory"),
            (csv_file, f"{args.dataset} test CSV"),
        ]:
            if not os.path.exists(path):
                logger.error(f"{desc} not found at {path}")
                raise FileNotFoundError(f"{desc} not found at {path}")
    
    config = VisionTransformerConfig(num_class=num_classes if not args.meta_moe else total_classes)
    apply_config_overrides(config, args.config_overrides)
    print(f"Training with {'MetaMoE' if args.meta_moe else args.dataset} with number of classes: {config.num_class}")
    print(f"Using config: {asdict(config)}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    setup_logging(OUTPUT_DIR)

    transform_train = transforms.Compose([
        transforms.Resize(32), 
        RandAugment(num_ops=2, magnitude=9), # If overfitting decreases but training becomes too slow or unstable, reduce num_ops to 1 or magnitude to 5-7. If overfitting, increase magnitude to 10-12 or num_ops to 3
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(normalization_mean, normalization_std),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.2)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize(normalization_mean, normalization_std),
    ])

    if args.meta_moe:
        gtsrb_train_dataset = TrafficSignTrainDataset(
            root='./data/GTSRB/Training',
            csv_file='./data/GTSRB/Training/train_with_meta_class.csv',
            transform=transform_train
        )
        ptsd_train_dataset = TrafficSignTrainDataset(
            root='./data/PTSD/Training',
            csv_file='./data/PTSD/Training/train_with_meta_class.csv',
            transform=transform_train
        )
        combined_train_dataset = CombinedDataset(gtsrb_train_dataset, ptsd_train_dataset, num_classes_gtsrb)

        gtsrb_test_dataset = TrafficSignTestDataset(
            root='./data/GTSRB/Test', 
            csv_file='./data/GTSRB/Test/testset_with_meta_class.csv', 
            transform=transform_test,
            default_meta_class=0  # GTSRB
        )
        ptsd_test_dataset = TrafficSignTestDataset(
            root='./data/PTSD/Test', 
            csv_file='./data/PTSD/Test/testset_with_meta_class.csv', 
            transform=transform_test,
            default_meta_class=1  # PTSD
        )
        combined_test_dataset = CombinedDataset(gtsrb_test_dataset, ptsd_test_dataset, num_classes_gtsrb)
        
        # Log dataset sizes
        logger.info(f"GTSRB training dataset size: {len(gtsrb_train_dataset)}")
        logger.info(f"PTSD training dataset size: {len(ptsd_train_dataset)}")
        logger.info(f"GTSRB test dataset size: {len(gtsrb_test_dataset)}")
        logger.info(f"PTSD test dataset size: {len(ptsd_test_dataset)}")
        
        if len(gtsrb_train_dataset) == 0 or len(ptsd_train_dataset) == 0:
            logger.error("One or both training datasets are empty. Check CSV files and image paths.")
            raise ValueError("Empty training dataset detected")
        if len(gtsrb_test_dataset) == 0 or len(ptsd_test_dataset) == 0:
            logger.error("One or both test datasets are empty. Check CSV files and image paths.")
            raise ValueError("Empty test dataset detected")
    
    else:
        if args.dataset == 'GTSRB':
            train_dataset = TrafficSignTrainDataset(
                root='./data/GTSRB/Training',
                csv_file='./data/GTSRB/Training/train_with_meta_class.csv',
                transform=transform_train
            )
        elif args.dataset == 'PTSD':
                train_dataset = TrafficSignTrainDataset(
                root='./data/PTSD/Training',
                csv_file='./data/PTSD/Training/train_with_meta_class.csv',
                transform=transform_train
            )
        else:
            raise ValueError(f"Unknown dataset: {args.dataset}")    
                    
        test_dataset = TrafficSignTestDataset(
            root=test_dir,
            csv_file=csv_file,
            transform=transform_test,
            default_meta_class=0 if args.dataset == 'GTSRB' else 1
        )
        
        logger.info(f"{args.dataset} training dataset size: {len(train_dataset)}")
        logger.info(f"{args.dataset} test dataset size: {len(test_dataset)}")
        
        if len(train_dataset) == 0 or len(test_dataset) == 0:
            logger.error(f"{args.dataset} dataset is empty. Check dataset paths and CSV.")
            raise ValueError(f"Empty {args.dataset} dataset detected")

    if os.name == 'nt':
        num_workers_train = min(os.cpu_count(), 8)
        prefetch_factor_train = 4
        persistent_workers_train = num_workers_train > 0
        num_workers_test = 8
        persistent_workers_test = True
    else:  # Linux
        num_workers_train = min(os.cpu_count(), 8)
        prefetch_factor_train = 4
        persistent_workers_train = True
        num_workers_test = 8
        persistent_workers_test = True

    train_loader = DataLoader(
        combined_train_dataset if args.meta_moe else train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers_train,
        pin_memory=True,
        persistent_workers=persistent_workers_train,
        prefetch_factor=prefetch_factor_train
    )

    test_loader = DataLoader(
        combined_test_dataset if args.meta_moe else test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers_test,
        persistent_workers=persistent_workers_test,
        pin_memory=True
    )

    if args.meta_moe:
        # Load GTSRB model as full model
        gtsrb_model = torch.load(
            os.path.join(PRETRAINED_MODEL_DIR, "vit_gtsrb_best.pth"),
            map_location=DEVICE,
            weights_only=False
        )
        if not isinstance(gtsrb_model, VisionTransformer):
            raise RuntimeError(f"vit_gtsrb_best.pth is not a VisionTransformer instance: {type(gtsrb_model)}")
        gtsrb_model = gtsrb_model.to(DEVICE)
        print(f"Loaded vit_gtsrb_best.pth as full VisionTransformer model.")

        # Load PTSD model as full model
        ptsd_model = torch.load(
            os.path.join(PRETRAINED_MODEL_DIR, "vit_ptsd_best.pth"),
            map_location=DEVICE,
            weights_only=False
        )
        if not isinstance(ptsd_model, VisionTransformer):
            raise RuntimeError(f"vit_ptsd_best.pth is not a VisionTransformer instance: {type(ptsd_model)}")
        ptsd_model = ptsd_model.to(DEVICE)
        print(f"Loaded vit_ptsd_best.pth as full VisionTransformer model.")

        # Set to evaluation mode and freeze parameters
        gtsrb_model.eval()
        ptsd_model.eval()
        for param in gtsrb_model.parameters():
            param.requires_grad = False
        for param in ptsd_model.parameters():
            param.requires_grad = False

        # Initialize MetaMoE
        meta_gating_net = MetaGatingNet().to(DEVICE)
        model = MetaMoE(
            gtsrb_model=gtsrb_model,
            ptsd_model=ptsd_model,
            meta_gating_net=meta_gating_net,
            num_classes_gtsrb=num_classes_gtsrb,
            num_classes_ptsd=num_classes_ptsd
        ).to(DEVICE)
        optimizer = optim.AdamW(
            meta_gating_net.parameters(),
            lr=LEARNING_RATE,
            weight_decay=0.05,
            fused=torch.cuda.is_available()
        )
    else:
        model = VisionTransformer(config).to(DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.05, fused=torch.cuda.is_available)
        #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    
    criterion = LabelSmoothingCrossEntropy(smoothing=LABEL_SMOOTHING)
    T_max = EPOCHS
    # if T_max = epoch. Pros: steady and predictable decay, improve convergence stability. Cons: complex models might not explore enough
    # if T_max = 100. Pros: good for exploring, escape local minima
    def lr_lambda(epoch):
        if epoch < WARMUP_EPOCHS:
            return (epoch + 1) / WARMUP_EPOCHS
        else:
            t = epoch - WARMUP_EPOCHS
            return 0.5 * (1 + np.cos(np.pi * (t % T_max) / T_max))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []
    train_balance_losses = []
    test_balance_losses = []
    train_gating_losses = []
    test_gating_losses = []
    train_gating_accs = []
    test_gating_accs = []
    test_gtsrb_accs = []
    test_ptsd_accs = []
    best_acc = 0
    total_training_time = 0
    test_inference_times = []
        
    for epoch in range(EPOCHS):
        start_time = time.time()
        train_results = train(model, train_loader, optimizer, criterion, DEVICE, config.balance_loss_weight if not args.meta_moe else None, default_meta_class)
        if len(train_results) != 5:
            logger.error(f"train function returned {len(train_results)} values, expected 5: {train_results}")
            raise ValueError(f"train function returned incorrect number of values: {len(train_results)}")
        train_loss, train_balance_loss, train_gating_loss, train_acc, train_gating_acc = train_results
        
        test_loss, test_balance_loss, test_gating_loss, test_acc, test_gating_acc, test_gtsrb_acc, test_ptsd_acc, test_inference_time = None, None, None, None, None, None, None, None
        if epoch >= TEST_START_EPOCH:
            if (epoch - TEST_START_EPOCH) % TEST_FREQUENCY == 0:
                test_results = test(model, test_loader, optimizer, criterion, DEVICE, default_meta_class)
                test_loss, test_balance_loss, test_gating_loss, test_acc, test_gating_acc, test_gtsrb_acc, test_ptsd_acc, test_inference_time = test_results
                test_inference_times.append(test_inference_time)
    
        scheduler.step()
        epoch_time = time.time() - start_time
        total_training_time += epoch_time
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        train_balance_losses.append(train_balance_loss)
        train_gating_losses.append(train_gating_loss)
        train_gating_accs.append(train_gating_acc)

        if test_loss is not None:
            test_losses.append(test_loss)
            test_accs.append(test_acc)
            test_balance_losses.append(test_balance_loss)
            test_gating_losses.append(test_gating_loss)
            test_gating_accs.append(test_gating_acc)
            test_gtsrb_accs.append(test_gtsrb_acc)
            test_ptsd_accs.append(test_ptsd_acc)

        print(f"{datetime.now()}")
        print(f"Epoch {epoch+1}/{EPOCHS}:")
        print(f"Train loss: {train_loss:.4f}, Train Balance Loss: {train_balance_loss:.4f}, Train Gating Loss: {train_gating_loss:.4f}, Train Acc: {train_acc:.4f}, Train Gating Acc: {train_gating_acc:.4f}")
        if test_loss is not None:
            print(f"Test loss: {test_loss:.4f}, Test Balance Loss: {test_balance_loss:.4f}, Test Gating Loss: {test_gating_loss:.4f}, Test Acc: {test_acc:.4f}, Test Gating Acc: {test_gating_acc:.4f}, Avg Inference Time: {test_inference_time:.6f} seconds/image")
            if args.meta_moe:
                print(f"Test GTSRB Acc: {test_gtsrb_acc:.4f}, Test PTSD Acc: {test_ptsd_acc:.4f}")
        print(f"Epoch time: {epoch_time:.2f} seconds")

        if test_acc is not None and test_acc > best_acc:
            best_acc = test_acc
            save_path = os.path.join(OUTPUT_DIR, "vit_meta_moe_best.pth" if args.meta_moe else f"vit_{args.dataset.lower()}_best.pth")
            torch.save(model, save_path)
            if not args.meta_moe and args.save_state_dict:
                state_dict_path = os.path.join(OUTPUT_DIR, f"vit_{args.dataset.lower()}_best_state_dict.pth")
                torch.save(model.state_dict(), state_dict_path)         
            print(f"New best accuracy: {best_acc:.4f}")
        print()

        if (epoch + 1) % 5 == 0 or epoch == EPOCHS - 1:
            plot_metrics(
                train_losses, test_losses, train_accs, test_accs,
                train_balance_losses, test_balance_losses,
                train_gating_losses, test_gating_losses,
                train_gating_accs, test_gating_accs,
                test_gtsrb_accs, test_ptsd_accs,
                EPOCHS, TEST_START_EPOCH, TEST_FREQUENCY, OUTPUT_DIR,
                meta_moe=args.meta_moe
            )

    print(f"Training completed. Best Accuracy: {best_acc:.4f}")
    print(f"Total training time: {total_training_time:.2f} seconds")
    print(f"Average time per epoch: {total_training_time/EPOCHS:.2f} seconds")
    if test_inference_times:
        avg_test_inference_time = sum(test_inference_times) / len(test_inference_times)
        print(f"Average inference time per image across test epochs: {avg_test_inference_time:.6f} seconds")
    
   # **************** Export to ONNX and save training params **************** 
    if args.export_onnx:
        best_model_path = os.path.join(OUTPUT_DIR, "vit_meta_moe_best.pth" if args.meta_moe else f"vit_{args.dataset.lower()}_best.pth")
        model = torch.load(best_model_path, map_location=DEVICE, weights_only=False)
        export_to_onnx(model=model, config=config, device=DEVICE, output_dir=OUTPUT_DIR, dataset_name="MetaMoE" if args.meta_moe else args.dataset)
    if args.archive_params:
        archive_params(args, config, OUTPUT_DIR)

if __name__ == '__main__':
    if os.name == 'nt':
        from multiprocessing import freeze_support
        freeze_support()
    main()