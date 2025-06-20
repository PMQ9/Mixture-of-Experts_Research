import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
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
import torch

from vision_transformer_moe import VisionTransformer, VisionTransformerConfig
from vision_transformer_moe import LabelSmoothingCrossEntropy, TrafficSignTestDataset
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
)
from config import apply_config_overrides
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'artifacts'))

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

# **************** Training Functions ****************
def train(model, loader, optimizer, criterion, device, balance_loss_weight):
    model.train()
    total_loss = 0
    total_balance_loss = 0
    correct = 0
    total = 0
    scaler = torch.amp.GradScaler(enabled=True)
    
    for batch_idx, (data, target) in enumerate(tqdm(loader, desc="Training")):
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        optimizer.zero_grad()

        apply_cutmix = data.size(0) == BATCH_SIZE and np.random.rand() < CUTMIX_PROB

        with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
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
        total_balance_loss += balance_loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        if apply_cutmix:
            correct += lam * predicted.eq(target_a).sum().item() + (1 - lam) * predicted.eq(target_b).sum().item()
        else:
            correct += predicted.eq(target).sum().item()
        
    avg_loss = total_loss / len(loader)
    avg_balance_loss = total_balance_loss / len(loader)
    accuracy = correct / total
    return avg_loss, avg_balance_loss, accuracy

# **************** Testing Functions ****************
def test(model, loader, optimizer, criterion, device):
    model.eval()
    total_loss = 0
    total_balance_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tqdm(loader, desc="Testing")):
            data, target = data.to(device), target.to(device)
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                output, balance_losses = model(data)
                loss = criterion(output, target)

                balance_loss = sum(balance_losses) / len(balance_losses) if isinstance(balance_losses, list) else balance_losses

            total_loss += loss.item()
            total_balance_loss += balance_loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
        
    avg_loss = total_loss / len(loader)
    avg_balance_loss = total_balance_loss / len(loader)
    accuracy = correct / total
    return avg_loss, avg_balance_loss, accuracy

# **************** Main Functions ****************
def main():
    if args.dataset == 'GTSRB':
        num_classes = 43
        train_dir = './data/GTSRB/Training'
        test_dir = './data/GTSRB/Test'
        csv_file = './data/GTSRB/Test/GT-final_test.csv'
        normalization_mean = (NORM_MEAN_R_GTSRB, NORM_MEAN_G_GTSRB, NORM_MEAN_B_GTSRB)
        normalization_std = (NORM_STD_R_GTSRB, NORM_STD_G_GTSRB, NORM_STD_B_GTSRB)
    elif args.dataset == 'PTSD':
        num_classes = 43
        train_dir = './data/PTSD/Training'
        test_dir = './data/PTSD/Test'
        csv_file = './data/PTSD/Test/testset_CSV.csv'
        normalization_mean = (NORM_MEAN_R_PTSD, NORM_MEAN_G_PTSD, NORM_MEAN_B_PTSD)
        normalization_std = (NORM_STD_R_PTSD, NORM_STD_G_PTSD, NORM_STD_B_PTSD)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    config = VisionTransformerConfig(num_class = num_classes)
    apply_config_overrides(config, args.config_overrides)
    print(f"Training with dataset: {args.dataset} with number of classes: {num_classes}" )
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

    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Training directory not found at {train_dir}")
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"Test directory not found at {test_dir}")
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"Test CSV file not found at {csv_file}")
    
    if os.name == 'nt':  # Windows
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

    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform_train)
    test_dataset = TrafficSignTestDataset(root=test_dir, csv_file=csv_file, transform=transform_test, class_to_idx=train_dataset.class_to_idx)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers_train,
        pin_memory=True,
        persistent_workers=persistent_workers_train,
        prefetch_factor=prefetch_factor_train
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers_test,
        persistent_workers=persistent_workers_test,
        pin_memory=True
    )

    model = VisionTransformer(config).to(DEVICE)
    criterion = LabelSmoothingCrossEntropy(smoothing=LABEL_SMOOTHING)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.05, fused=torch.cuda.is_available)
    #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
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

    best_acc = 0
    total_training_time = 0
        
    for epoch in range(EPOCHS):
        start_time = time.time()
        train_loss, train_balance_loss, train_acc = train(model, train_loader, optimizer, criterion, DEVICE, balance_loss_weight=config.balance_loss_weight)
        
        test_loss, test_balance_loss, test_acc = None, None, None
        if epoch >= TEST_START_EPOCH:
            if (epoch - TEST_START_EPOCH) % TEST_FREQUENCY == 0:
                test_loss, test_balance_loss, test_acc = test(model, test_loader, optimizer, criterion, DEVICE)
    
        scheduler.step()
        epoch_time = time.time() - start_time
        total_training_time += epoch_time
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        train_balance_losses.append(train_balance_loss)

        if test_loss is not None:
            test_losses.append(test_loss)
            test_accs.append(test_acc)
            test_balance_losses.append(test_balance_loss)

        print(f"{datetime.now()}")
        print(f"Epoch {epoch+1}/{EPOCHS}:")
        print(f"Train loss: {train_loss:.4f}, Train Balance Loss: {train_balance_loss:.4f}, Train Acc: {train_acc:.4f}")
        if test_loss is not None:
            print(f"Test loss: {test_loss:.4f}, Test Balance Loss: {test_balance_loss:.4f}, Test Acc: {test_acc:.4f}")
        print(f"Epoch time: {epoch_time:.2f} seconds")

        if test_acc is not None and test_acc > best_acc:
            best_acc = test_acc
            #torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "vit_gtsrb_best.pth"))
            if args.dataset == 'GTSRB':
                torch.save(model, os.path.join(OUTPUT_DIR, "vit_gtsrb_best.pth"))
            elif args.dataset == 'PTSD':
                torch.save(model, os.path.join(OUTPUT_DIR, "vit_ptsd_best.pth"))
            print(f"New best accuracy: {best_acc:.4f}")
        print()

        if (epoch + 1) % 5 == 0 or epoch == EPOCHS - 1:
            plot_metrics(train_losses, test_losses, train_accs, test_accs, train_balance_losses, test_balance_losses, EPOCHS, TEST_START_EPOCH, TEST_FREQUENCY, OUTPUT_DIR)

    print(f"Training completed. Best Accuracy: {best_acc:.4f}")
    print(f"Total training time: {total_training_time:.2f} seconds")
    print(f"Average time per epoch: {total_training_time/EPOCHS:.2f} seconds")
    
   # **************** Export to ONNX and save training params **************** 
   # By default, DO export ONNX and DO NOT log params
    if args.export_onnx == True:
        if args.dataset == 'GTSRB':
            best_model_path = os.path.join(OUTPUT_DIR, "vit_gtsrb_best.pth")
        elif args.dataset == 'PTSD':
            best_model_path = os.path.join(OUTPUT_DIR, "vit_ptsd_best.pth")
        model = torch.load(best_model_path, map_location=DEVICE, weights_only=False)   
        export_to_onnx(model=model, config=config, device=DEVICE, output_dir=OUTPUT_DIR, dataset_name=args.dataset)
    if args.archive_params == True:
        archive_params(args, config, OUTPUT_DIR)

if __name__ == '__main__':
    if os.name == 'nt':
        from multiprocessing import freeze_support
        freeze_support()
    main()
