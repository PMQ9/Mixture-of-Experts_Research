import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
from tqdm import tqdm
from datetime import datetime
from PIL import Image
from torch.cuda.amp import autocast, GradScaler
from torchvision.transforms import RandAugment
import argparse
from dataclasses import fields, asdict
import torch.multiprocessing
import logging
import timm
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent

from vision_transformer_moe import VisionTransformer, VisionTransformerConfig, LabelSmoothingCrossEntropy, TrafficSignTrainDataset, TrafficSignTestDataset
from vision_transformer_moe import MetaGatingNet, CombinedDataset, MetaMoE
from log_functions import setup_logging, archive_params, plot_metrics, export_to_onnx
from augmentation_functions import cutmix
from visualize_robustness import visualize_robustness
from config import (
    DEFAULT_PARAMS, GTSRB_NORM, PTSD_NORM, TSRD_NORM, BTSD_NORM, ETSD_NORM, CIFAR10_NORM, MNIST_NORM, UNIFIED_NORM
)
from config import apply_config_overrides

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'artifacts'))
PRETRAINED_MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'artifacts', 'results'))

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
if os.name != 'nt':
    torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser(description='Train a Vision Transformer with MoE')
parser.add_argument('--dataset', type=str, default='GTSRB', choices=['GTSRB', 'PTSD', 'TSRD', 'BTSD', 'ETSD', 'CIFAR10', 'MNIST'], help='Dataset to train')
parser.add_argument('--batch_size', type=int, default=DEFAULT_PARAMS['batch_size'], help='Batch size for training')
parser.add_argument('--epochs', type=int, default=int(os.getenv('CICD_EPOCH', DEFAULT_PARAMS['epoch'])), help='Number of epochs to train')
parser.add_argument('--learning_rate', type=float, default=DEFAULT_PARAMS['learning_rate'], help='Learning rate for optimizer')
parser.add_argument('--cutmix_alpha', type=float, default=DEFAULT_PARAMS['cutmix_alpha'], help='Alpha parameter for CutMix')
parser.add_argument('--cutmix_prob', type=float, default=DEFAULT_PARAMS['cutmix_prob'], help='Probability of applying CutMix')
parser.add_argument('--test_start_epoch', type=int, default=DEFAULT_PARAMS['test_start_epoch'], help='Epoch to start testing')
parser.add_argument('--test_frequency', type=int, default=DEFAULT_PARAMS['test_frequency'], help='Frequency of testing in epochs')
parser.add_argument('--warmup_epochs', type=int, default=DEFAULT_PARAMS['warmup_epoch'], help='Number of warmup epochs')
parser.add_argument('--label_smoothing', type=float, default=DEFAULT_PARAMS['label_smoothing'], help='Label smoothing factor')
parser.add_argument('--archive_params', type=bool, default=True, help='Save full training params')
parser.add_argument('--export_onnx', type=bool, default=True, help='Export trained model to ONNX')
parser.add_argument('--meta_moe', action='store_true', help='Train MetaMoE model with pre-trained experts')
parser.add_argument('--save_state_dict', action='store_true', help='Additionally save state_dict for non-MetaMoE models')
parser.add_argument('--gating_loss_weight', type=float, default=1.0, help='Weight for MetaGatingNet supervision loss')
parser.add_argument('--num_meta_experts', type=int, default=2, help='Number of experts to in MetaMoE')
parser.add_argument('--meta_top_k', type=int, default=1, help='Number of top experts to use in MetaMoE')
parser.add_argument('--model_arch', type=str, default='convnext_tiny', choices=['vit_moe', 'resnet50', 'resnet101', 'convnext_tiny', 'efficientnet_b0', 'vit_base'], help='Model architecture to use')
parser.add_argument('--gtsrb_model_path', type=str, default=os.path.join(PRETRAINED_MODEL_DIR, "gtsrb_convnext_tiny_best.pth"), help='Path to pre-trained GTSRB model')
parser.add_argument('--cifar10_model_path', type=str, default=os.path.join(PRETRAINED_MODEL_DIR, "cifar10_convnext_tiny_best.pth"), help='Path to pre-trained CIFAR10 model')
parser.add_argument('--mnist_model_path', type=str, default=os.path.join(PRETRAINED_MODEL_DIR, "mnist_convnext_tiny_best.pth"), help='Path to pre-trained MNIST model')
# parser.add_argument('--tsrd_model_path', type=str, default=os.path.join(PRETRAINED_MODEL_DIR, "tsrd_convnext_tiny_best.pth"), help='Path to pre-trained TSRD model')
# parser.add_argument('--btsd_model_path', type=str, default=os.path.join(PRETRAINED_MODEL_DIR, "btsd_convnext_tiny_best.pth"), help='Path to pre-trained BTSD model')
# parser.add_argument('--etsd_model_path', type=str, default=os.path.join(PRETRAINED_MODEL_DIR, "etsd_convnext_tiny_best.pth"), help='Path to pre-trained ETSD model')
parser.add_argument('--art_attack', action='store_true', help='Initiate Adversarial Robustness Toolbox')
parser.add_argument('--art_attack_mode', type=str, default='PGD', choices=['FGM', 'PGD'], help='Attack mode in ART')
parser.add_argument('--visualize_robustness', action='store_true', help='visualize model switching experts')
parser.add_argument('--adversarial_training', action='store_true', help='Enable adversarial training for individual experts')

config_fields = [f.name for f in fields(VisionTransformerConfig)]
help_msg = f"Comma-separated list of config overrides, e.g., 'img_size=48,patch_size=8'. Available parameters: {', '.join(config_fields)}"
parser.add_argument('--config_overrides', type=str, default='', help=help_msg)
args = parser.parse_args()

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

class ModelWrapper(nn.Module):
    def __init__(self, model, num_classes):
        super().__init__()
        self.model = model
        self.num_classes = num_classes

    def forward(self, x):
        output = self.model(x)
        return output, [torch.tensor(0.0, device=x.device)]

class LogitsWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        output = self.model(x)
        if isinstance(output, tuple):
            return output[0]
        return output

def get_num_classes(model):
    if hasattr(model, 'total_classes'):
        return model.total_classes
    elif isinstance(model, VisionTransformer):
        return model.config.num_class
    elif isinstance(model, ModelWrapper):
        return model.num_classes
    else:
        raise ValueError("Cannot determine the number of classes from the model")

def create_model(model_arch, config):
    if model_arch == 'vit_moe':
        return VisionTransformer(config)
    elif model_arch == 'resnet50':
        model = models.resnet50(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, config.num_class)
        return ModelWrapper(model, config.num_class)
    elif model_arch == 'resnet101':
        model = models.resnet101(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, config.num_class)
        return ModelWrapper(model, config.num_class)
    elif model_arch == 'convnext_tiny':
        model = timm.create_model('convnext_tiny', pretrained=True, num_classes=config.num_class)
        return ModelWrapper(model, config.num_class)
    elif model_arch == 'efficientnet_b0':
        model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=config.num_class)
        return ModelWrapper(model, config.num_class)
    elif model_arch == 'vit_base':
        model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=config.num_class, img_size=config.img_size)
        return ModelWrapper(model, config.num_class)
    else:
        raise ValueError(f"Unknown model architecture: {model_arch}")

def pgd_attack(model, data, target, epsilon=0.1, alpha=0.01, num_iter=10, device=DEVICE):
    model.eval()
    data_adv = data.clone().detach().to(device)
    original_data = data.clone().detach().to(device)
    for _ in range(num_iter):
        data_adv.requires_grad_(True)
        with torch.enable_grad():
            output = model(data_adv)
            if isinstance(output, tuple):
                output = output[0]
            loss = nn.CrossEntropyLoss()(output, target)
            loss.backward()
            grad = data_adv.grad.detach()
        with torch.no_grad():
            data_adv = data_adv + alpha * grad.sign()
            perturbation = data_adv - original_data
            perturbation = torch.clamp(perturbation, min=-epsilon, max=epsilon)
            data_adv = original_data + perturbation
        data_adv = data_adv.detach()
    model.train()
    return data_adv

def train(model, loader, optimizer, criterion, device, balance_loss_weight=None, default_meta_class=None):
    model.train()
    total_loss = total_balance_loss = total_gating_loss = correct = gating_correct = total = 0
    scaler = torch.amp.GradScaler(enabled=True)
    gating_criterion = nn.CrossEntropyLoss()
    total_router_time = total_experts_time = total_post_time = total_total_time = 0.0
    total_images = 0
    
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
                output, gates, router_time, experts_time, post_time, total_time = model.forward_with_timing(data)
                cls_loss = criterion(output, target)
                gating_loss = gating_criterion(gates, meta_class)
                total_loss_combined = cls_loss + GATING_LOSS_WEIGHT * gating_loss
                total_router_time += router_time
                total_experts_time += experts_time
                total_post_time += post_time
                total_total_time += total_time
                total_images += data.size(0)
            else:
                if apply_cutmix:
                    data_cutmix, target_a, target_b, lam = cutmix(data, target, CUTMIX_ALPHA)
                    output, balance_losses = model(data_cutmix)
                    loss_a = criterion(output, target_a)
                    loss_b = criterion(output, target_b)
                    loss_clean = lam * loss_a + (1 - lam) * loss_b
                else:
                    output, balance_losses = model(data)
                    loss_clean = criterion(output, target)
                
                if not args.meta_moe and args.adversarial_training:
                    data_adv = pgd_attack(model, data, target)
                    output_adv, balance_losses_adv = model(data_adv)
                    loss_adv = criterion(output_adv, target)
                    cls_loss = (loss_clean + loss_adv) / 2
                    balance_loss = (sum(balance_losses) + sum(balance_losses_adv)) / (2 * len(balance_losses)) if balance_losses else 0
                else:
                    cls_loss = loss_clean
                    balance_loss = sum(balance_losses) / len(balance_losses) if balance_losses else 0
                
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
    
    if args.meta_moe:
        avg_router_time = total_router_time / total_images if total_images > 0 else 0
        avg_experts_time = total_experts_time / total_images if total_images > 0 else 0
        avg_post_time = total_post_time / total_images if total_images > 0 else 0
        avg_total_time = total_total_time / total_images if total_images > 0 else 0
        return avg_loss, avg_balance_loss, avg_gating_loss, accuracy, gating_accuracy, avg_router_time, avg_experts_time, avg_post_time, avg_total_time
    else:
        return avg_loss, avg_balance_loss, avg_gating_loss, accuracy, gating_accuracy

def test(model, loader, optimizer, criterion, device, default_meta_class=None):
    model.eval()
    total_loss = total_balance_loss = total_gating_loss = correct = gating_correct = total = total_images = 0
    gtsrb_correct = cifar10_correct = gtsrb_total = cifar10_total = mnist_correct = mnist_total = 0
    gating_criterion = nn.CrossEntropyLoss()
    inference_times = []
    total_router_time = total_experts_time = total_post_time = total_total_time = 0.0
    
    with torch.no_grad():
        for batch_idx, (data, target, meta_class) in enumerate(tqdm(loader, desc="Testing")):
            data, target, meta_class = data.to(device), target.to(device), meta_class.to(device)
            if args.meta_moe:
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    output, gates, router_time, experts_time, post_time, total_time = model.forward_with_timing(data)
                    loss = criterion(output, target)
                    gating_loss = gating_criterion(gates, meta_class)
                    balance_loss = 0
                total_router_time += router_time
                total_experts_time += experts_time
                total_post_time += post_time
                total_total_time += total_time
                total_images += data.size(0)
            else:
                start_time = time.time()
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
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
                cifar10_mask = meta_class == 1
                mnist_mask = meta_class == 2
                # tsrd_mask = meta_class == 2
                # btsd_mask = meta_class == 3
                # etsd_mask = meta_class == 4
                if gtsrb_mask.any():
                    gtsrb_correct += predicted[gtsrb_mask].eq(target[gtsrb_mask]).sum().item()
                    gtsrb_total += gtsrb_mask.sum().item()
                if cifar10_mask.any():
                    cifar10_correct += predicted[cifar10_mask].eq(target[cifar10_mask]).sum().item()
                    cifar10_total += cifar10_mask.sum().item()
                if mnist_mask.any():
                    mnist_correct += predicted[mnist_mask].eq(target[mnist_mask]).sum().item()
                    mnist_total += mnist_mask.sum().item()
                # if tsrd_mask.any():
                #     tsrd_correct += predicted[tsrd_mask].eq(target[tsrd_mask]).sum().item()
                #     tsrd_total += tsrd_mask.sum().item()
                # if btsd_mask.any():
                #     btsd_correct += predicted[btsd_mask].eq(target[btsd_mask]).sum().item()
                #     btsd_total += btsd_mask.sum().item()
                # if etsd_mask.any():
                #     etsd_correct += predicted[etsd_mask].eq(target[etsd_mask]).sum().item()
                #     etsd_total += etsd_mask.sum().item()
        
    avg_loss = total_loss / len(loader)
    avg_balance_loss = total_balance_loss / len(loader) if not args.meta_moe else 0
    avg_gating_loss = total_gating_loss / len(loader) if args.meta_moe else 0
    accuracy = correct / total
    gating_accuracy = gating_correct / total if args.meta_moe else 0
    gtsrb_accuracy = gtsrb_correct / gtsrb_total if gtsrb_total > 0 and args.meta_moe else 0
    cifar10_accuracy = cifar10_correct / cifar10_total if cifar10_total > 0 and args.meta_moe else 0
    mnist_accuracy = mnist_correct / mnist_total if mnist_total > 0 and args.meta_moe else 0
    # tsrd_accuracy = tsrd_correct / tsrd_total if tsrd_total > 0 and args.meta_moe else 0
    # btsd_accuracy = btsd_correct / btsd_total if btsd_total > 0 and args.meta_moe else 0
    # etsd_accuracy = etsd_correct / etsd_total if etsd_total > 0 and args.meta_moe else 0
    if args.meta_moe:
        avg_router_time = total_router_time / total_images if total_images > 0 else 0
        avg_experts_time = total_experts_time / total_images if total_images > 0 else 0
        avg_post_time = total_post_time / total_images if total_images > 0 else 0
        avg_total_time = total_total_time / total_images if total_images > 0 else 0
        logger.info(f"Test results: loss={avg_loss:.4f}, gating_loss={avg_gating_loss:.4f}, accuracy={accuracy:.4f}, gating_accuracy={gating_accuracy:.4f}, avg_total_inference_time={avg_total_time:.6f} seconds/image")
        return avg_loss, avg_balance_loss, avg_gating_loss, accuracy, gating_accuracy, gtsrb_accuracy, cifar10_accuracy, mnist_accuracy, avg_router_time, avg_experts_time, avg_post_time, avg_total_time
    else:
        avg_inference_time = sum(inference_times) / len(inference_times) if inference_times else 0
        logger.info(f"Test results: loss={avg_loss:.4f}, balance_loss={avg_balance_loss:.4f}, accuracy={accuracy:.4f}, avg_inference_time={avg_inference_time:.6f} seconds/image")
        return avg_loss, avg_balance_loss, avg_gating_loss, accuracy, gating_accuracy, gtsrb_accuracy, cifar10_accuracy, mnist_accuracy, avg_inference_time

def test_adversarial_robustness(model, test_loader, device, eps=0.1):
    model.eval()
    wrapped_model = LogitsWrapper(model).to(device)
    nb_classes = get_num_classes(model)
    criterion = nn.CrossEntropyLoss()
    classifier = PyTorchClassifier(
        model=wrapped_model,
        loss=criterion,
        input_shape=(3, 32, 32),
        nb_classes=nb_classes,
        device_type='gpu' if torch.cuda.is_available() else 'cpu'
    )
    
    if args.art_attack_mode == 'FGM':
        attack = FastGradientMethod(estimator=classifier, eps=eps)
    elif args.art_attack_mode == 'PGD':
        attack = ProjectedGradientDescent(estimator=classifier, eps=0.1, eps_step=0.01, max_iter=40)
    else:
        print("Attack mode unknown")
    total = correct_clean = correct_adv = 0
    is_meta_moe = hasattr(model, 'meta_gating_net') and hasattr(model, 'experts')
    if is_meta_moe:
        gating_correct_clean = gating_correct_adv = 0
        expert_correct_clean = [0] * model.num_experts
        expert_correct_adv = [0] * model.num_experts
        expert_total = [0] * model.num_experts
    
    for data, target, meta_class in tqdm(test_loader, desc="Adversarial Testing"):
        data, target, meta_class = data.to(device), target.to(device), meta_class.to(device)
        data.requires_grad_(True)  # Enable gradients for input
        
        # Clean data predictions
        with torch.no_grad():
            output_clean = wrapped_model(data)
            pred_clean = output_clean.argmax(dim=1)
            if is_meta_moe:
                gates_clean = model.meta_gating_net(data)
                gates_pred_clean = gates_clean.argmax(dim=1)
        
        # Generate adversarial examples
        x_test_np = data.detach().cpu().numpy()
        x_test_adv = attack.generate(x=x_test_np, y=target.cpu().numpy())
        x_test_adv_torch = torch.from_numpy(x_test_adv).to(device).requires_grad_(True)
        
        # Adversarial data predictions
        with torch.no_grad():
            output_adv = wrapped_model(x_test_adv_torch)
            pred_adv = output_adv.argmax(dim=1)
            if is_meta_moe:
                gates_adv = model.meta_gating_net(x_test_adv_torch)
                gates_pred_adv = gates_adv.argmax(dim=1)
        
        total += target.size(0)
        correct_clean += pred_clean.eq(target).sum().item()
        correct_adv += pred_adv.eq(target).sum().item()
        
        if is_meta_moe:
            gating_correct_clean += gates_pred_clean.eq(meta_class).sum().item()
            gating_correct_adv += gates_pred_adv.eq(meta_class).sum().item()
            
            for i in range(model.num_experts):
                mask = (meta_class == i)
                if mask.any():
                    expert_input_clean = data[mask]
                    expert_input_adv = x_test_adv_torch[mask]
                    expert_target = target[mask] - model.class_offsets[i]
                    
                    expert_model = model.experts[i]
                    expert_output_clean = expert_model(expert_input_clean)
                    if isinstance(expert_output_clean, tuple):
                        expert_output_clean = expert_output_clean[0]
                    expert_pred_clean = expert_output_clean.argmax(dim=1)
                    expert_correct_clean[i] += (expert_pred_clean == expert_target).sum().item()
                    
                    expert_output_adv = expert_model(expert_input_adv)
                    if isinstance(expert_output_adv, tuple):
                        expert_output_adv = expert_output_adv[0]
                    expert_pred_adv = expert_output_adv.argmax(dim=1)
                    expert_correct_adv[i] += (expert_pred_adv == expert_target).sum().item()
                    
                    expert_total[i] += mask.sum().item()
    
    # Calculate and print overall accuracies
    acc_clean = correct_clean / total
    acc_adv = correct_adv / total
    print(f"Clean Accuracy: {acc_clean:.4f}, Adversarial Accuracy: {acc_adv:.4f}")
    
    # MetaMoE-specific metrics
    if is_meta_moe:
        gating_acc_clean = gating_correct_clean / total
        gating_acc_adv = gating_correct_adv / total
        print(f"Clean Gating Accuracy: {gating_acc_clean:.4f}, Adversarial Gating Accuracy: {gating_acc_adv:.4f}")
        
        for i in range(model.num_experts):
            if expert_total[i] > 0:
                expert_acc_clean = expert_correct_clean[i] / expert_total[i]
                expert_acc_adv = expert_correct_adv[i] / expert_total[i]
                print(f"Expert {i}: Clean Acc: {expert_acc_clean:.4f}, Adv Acc: {expert_acc_adv:.4f}")

def main():
    dataset_params = {
        'GTSRB': {
            'num_classes': 43,
            'train_dir': './data/GTSRB/Training',
            'test_dir': './data/GTSRB/Test',
            'csv_file': './data/GTSRB/Test/testset_with_meta_class.csv',
            'normalization_mean': (GTSRB_NORM['mean']),
            'normalization_std': (GTSRB_NORM['std']),
            'default_meta_class': 0
        },
        'CIFAR10': {
            'num_classes': 10,
            'train_dir': './data/CIFAR10/Training',
            'test_dir': './data/CIFAR10/Test',
            'csv_file': './data/CIFAR10/Test/testset_with_meta_class.csv',
            'normalization_mean': (CIFAR10_NORM['mean']),
            'normalization_std': (CIFAR10_NORM['std']),
            'default_meta_class': 1
        },
        # 'TSRD': {
        #     'num_classes': 58,
        #     'train_dir': './data/TSRD/Training',
        #     'test_dir': './data/TSRD/Test',
        #     'csv_file': './data/TSRD/Test/testset_with_meta_class.csv',
        #     'normalization_mean': (TSRD_NORM['mean']),
        #     'normalization_std': (TSRD_NORM['std']),
        #     'default_meta_class': 2
        # }
        # 'BTSD': {
        #     'num_classes': 62,
        #     'train_dir': './data/BTSD/Training',
        #     'test_dir': './data/BTSD/Test',
        #     'csv_file': './data/BTSD/Test/testset_with_meta_class.csv',
        #     'normalization_mean': (BTSD_NORM['mean']),
        #     'normalization_std': (BTSD_NORM['std']),
        #     'default_meta_class': 3
        # }
        # 'ETSD': {
        #     'num_classes': 55,
        #     'train_dir': './data/ETSD/Training',
        #     'test_dir': './data/ETSD/Test',
        #     'csv_file': './data/ETSD/Test/testset_with_meta_class.csv',
        #     'normalization_mean': (ETSD_NORM['mean']),
        #     'normalization_std': (ETSD_NORM['std']),
        #     'default_meta_class': 4
        # }
        'MNIST': {
            'num_classes': 10,
            'train_dir': './data/MNIST/Training',
            'test_dir': './data/MNIST/Test',
            'csv_file': './data/MNIST/Test/testset_with_meta_class.csv',
            'normalization_mean': (MNIST_NORM['mean']),
            'normalization_std': (MNIST_NORM['std']),
            'default_meta_class': 2
        }
    }

    if args.meta_moe:
        datasets = ['GTSRB', 'CIFAR10']
        num_classes_list = [dataset_params[ds]['num_classes'] for ds in datasets]
        total_classes = sum(num_classes_list)
        normalization_mean = (UNIFIED_NORM['mean'])
        normalization_std = (UNIFIED_NORM['std'])
        
        transform_train = transforms.Compose([
            transforms.Resize(32), 
            RandAugment(num_ops=2, magnitude=9),
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
        
        train_datasets = []
        test_datasets = []
        for dataset in datasets:
            params = dataset_params[dataset]
            train_dataset = TrafficSignTrainDataset(
                root=params['train_dir'],
                csv_file=os.path.join(params['train_dir'], 'train_with_meta_class.csv'),
                transform=transform_train
            )
            test_dataset = TrafficSignTestDataset(
                root=params['test_dir'],
                csv_file=params['csv_file'],
                transform=transform_test,
                default_meta_class=params['default_meta_class']
            )
            train_datasets.append(train_dataset)
            test_datasets.append(test_dataset)
            logger.info(f"{dataset} training dataset size: {len(train_dataset)}")
            logger.info(f"{dataset} test dataset size: {len(test_dataset)}")
            if len(train_dataset) == 0 or len(test_dataset) == 0:
                logger.error(f"{dataset} dataset is empty. Check CSV files and image paths.")
                raise ValueError(f"Empty {dataset} dataset detected")
        
        combined_train_dataset = CombinedDataset(
            datasets=train_datasets,
            num_classes_list=num_classes_list
        )
        combined_test_dataset = CombinedDataset(
            datasets=test_datasets,
            num_classes_list=num_classes_list
        )
    else:
        if args.dataset not in dataset_params:
            raise ValueError(f"Unknown dataset: {args.dataset}")
        params = dataset_params[args.dataset]
        num_classes = params['num_classes']
        train_dir = params['train_dir']
        test_dir = params['test_dir']
        csv_file = params['csv_file']
        normalization_mean = params['normalization_mean']
        normalization_std = params['normalization_std']
        default_meta_class = params['default_meta_class']
        
        for path, desc in [
            (train_dir, f"{args.dataset} training directory"),
            (test_dir, f"{args.dataset} test directory"),
            (csv_file, f"{args.dataset} test CSV"),
        ]:
            if not os.path.exists(path):
                logger.error(f"{desc} not found at {path}")
                raise FileNotFoundError(f"{desc} not found at {path}")
        
        transform_train = transforms.Compose([
            transforms.Resize(32), 
            RandAugment(num_ops=2, magnitude=9),
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
        
        train_dataset = TrafficSignTrainDataset(
            root=train_dir,
            csv_file=os.path.join(train_dir, 'train_with_meta_class.csv'),
            transform=transform_train
        )
        test_dataset = TrafficSignTestDataset(
            root=test_dir,
            csv_file=csv_file,
            transform=transform_test,
            default_meta_class=default_meta_class
        )
        
        logger.info(f"{args.dataset} training dataset size: {len(train_dataset)}")
        logger.info(f"{args.dataset} test dataset size: {len(test_dataset)}")
        if len(train_dataset) == 0 or len(test_dataset) == 0:
            logger.error(f"{args.dataset} dataset is empty. Check dataset paths and CSV.")
            raise ValueError(f"Empty {args.dataset} dataset detected")
    
    config = VisionTransformerConfig(num_class=num_classes if not args.meta_moe else total_classes)
    apply_config_overrides(config, args.config_overrides)
    print(f"Training with {'MetaMoE' if args.meta_moe else args.dataset} with number of classes: {config.num_class}")
    if args.meta_moe:
        print(f"Meta_MoE architecture: activate {args.meta_top_k} of {args.num_meta_experts} experts")
    else:
        print(f"Using config: {asdict(config)}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    setup_logging(OUTPUT_DIR)

    if os.name == 'nt':
        num_workers_train = min(os.cpu_count(), 8)
        prefetch_factor_train = 4
        persistent_workers_train = num_workers_train > 0
        num_workers_test = 8
        persistent_workers_test = True
    else:
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
        gtsrb_model = torch.load(args.gtsrb_model_path, map_location=DEVICE, weights_only=False)
        if not isinstance(gtsrb_model, (VisionTransformer, models.ResNet, timm.models.ConvNeXt, ModelWrapper)):
            raise RuntimeError(f"{args.gtsrb_model_path} is not a supported model type")
        if isinstance(gtsrb_model, ModelWrapper):
            gtsrb_model = gtsrb_model.model
        gtsrb_model = gtsrb_model.to(DEVICE)
        print(f"Loaded {args.gtsrb_model_path} as full model. Type: {type(gtsrb_model)}")
        
        cifar10_model = torch.load(args.cifar10_model_path, map_location=DEVICE, weights_only=False)
        if not isinstance(cifar10_model, (VisionTransformer, models.ResNet, timm.models.ConvNeXt, ModelWrapper)):
            raise RuntimeError(f"{args.cifar10_model_path} is not a supported model type")
        if isinstance(cifar10_model, ModelWrapper):
            cifar10_model = cifar10_model.model
        cifar10_model = cifar10_model.to(DEVICE)
        print(f"Loaded {args.cifar10_model_path} as full model. Type: {type(cifar10_model)}")

        mnist_model = torch.load(args.mnist_model_path, map_location=DEVICE, weights_only=False)
        if not isinstance(mnist_model, (VisionTransformer, models.ResNet, timm.models.ConvNeXt, ModelWrapper)):
            raise RuntimeError(f"{args.mnist_model_path} is not a supported model type")
        if isinstance(mnist_model, ModelWrapper):
            mnist_model = mnist_model.model
        mnist_model = mnist_model.to(DEVICE)
        print(f"Loaded {args.mnist_model_path} as full model. Type: {type(mnist_model)}")

        # tsrd_model = torch.load(args.tsrd_model_path, map_location=DEVICE, weights_only=False)
        # if not isinstance(tsrd_model, (VisionTransformer, models.ResNet, timm.models.ConvNeXt, ModelWrapper)):
        #     raise RuntimeError(f"{args.tsrd_model_path} is not a supported model type")
        # if isinstance(tsrd_model, ModelWrapper):
        #     tsrd_model = tsrd_model.model
        # tsrd_model = tsrd_model.to(DEVICE)
        # print(f"Loaded {args.tsrd_model_path} as full model. Type: {type(tsrd_model)}")

        # btsd_model = torch.load(args.btsd_model_path, map_location=DEVICE, weights_only=False)
        # if not isinstance(btsd_model, (VisionTransformer, models.ResNet, timm.models.ConvNeXt, ModelWrapper)):
        #     raise RuntimeError(f"{args.btsd_model_path} is not a supported model type")
        # if isinstance(btsd_model, ModelWrapper):
        #     btsd_model = btsd_model.model
        # btsd_model = btsd_model.to(DEVICE)
        # print(f"Loaded {args.btsd_model_path} as full model. Type: {type(btsd_model)}")
    
        # etsd_model = torch.load(args.etsd_model_path, map_location=DEVICE, weights_only=False)
        # if not isinstance(etsd_model, (VisionTransformer, models.ResNet, timm.models.ConvNeXt, ModelWrapper)):
        #     raise RuntimeError(f"{args.etsd_model_path} is not a supported model type")
        # if isinstance(etsd_model, ModelWrapper):
        #     etsd_model = etsd_model.model
        # etsd_model = etsd_model.to(DEVICE)
        # print(f"Loaded {args.etsd_model_path} as full model. Type: {type(etsd_model)}")

        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 32, 32, device=DEVICE)
            for model, expected_classes, name in [
                (gtsrb_model, dataset_params['GTSRB']['num_classes'], "GTSRB"),
                (cifar10_model, dataset_params['CIFAR10']['num_classes'], "CIFAR10"),
                (mnist_model, dataset_params['CIFAR10']['num_classes'], "MNIST"),
                # (tsrd_model, dataset_params['TSRD']['num_classes'], "TSRD"),
                # (btsd_model, dataset_params['BTSD']['num_classes'], "BTSD"),
                # (etsd_model, dataset_params['ETSD']['num_classes'], "ETSD")
            ]:
                output = model(dummy_input)
                if isinstance(output, tuple):
                    output = output[0]
                if output.shape[1] != expected_classes:
                    raise RuntimeError(f"{name} model output shape {output.shape[1]} does not match expected {expected_classes} classes")
                print(f"{name} model output shape: {output.shape}")

        gtsrb_model.eval()
        cifar10_model.eval()
        mnist_model.eval()
        # tsrd_model.eval()
        # btsd_model.eval()
        # etsd_model.eval()
        for param in gtsrb_model.parameters():
            param.requires_grad = False
        for param in cifar10_model.parameters():
            param.requires_grad = False
        for param in mnist_model.parameters():
            param.requires_grad = False
        # for param in tsrd_model.parameters():
        #     param.requires_grad = False
        # for param in btsd_model.parameters():
        #     param.requires_grad = False
        # for param in etsd_model.parameters():
        #     param.requires_grad = False    

        num_meta_experts = args.num_meta_experts
        meta_gating_net = MetaGatingNet(num_experts=num_meta_experts).to(DEVICE)
        experts = [gtsrb_model, cifar10_model]
        num_classes_list = [dataset_params[ds]['num_classes'] for ds in datasets]
        model = MetaMoE(
            experts=experts,
            num_classes_list=num_classes_list,
            meta_gating_net=meta_gating_net,
            meta_top_k=args.meta_top_k
        ).to(DEVICE)
        optimizer = optim.AdamW(
            meta_gating_net.parameters(),
            lr=LEARNING_RATE,
            weight_decay=0.05,
            fused=torch.cuda.is_available()
        )
    else:
        model = create_model(args.model_arch, config).to(DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.05, fused=torch.cuda.is_available())
    
    criterion = LabelSmoothingCrossEntropy(smoothing=LABEL_SMOOTHING)
    T_max = EPOCHS
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
    test_cifar10_accs = []
    test_mnist_accs = []
    # test_tsrd_accs = []
    # test_btsd_accs = []
    # test_etsd_accs = []
    best_acc = 0
    total_training_time = 0
    test_inference_times = []
        
    for epoch in range(EPOCHS):
        start_time = time.time()
        if args.meta_moe:
            train_results = train(model, train_loader, optimizer, criterion, DEVICE, balance_loss_weight=None)
        else:
            train_results = train(model, train_loader, optimizer, criterion, DEVICE, balance_loss_weight=config.balance_loss_weight, default_meta_class=default_meta_class)
        if args.meta_moe:
            train_loss, train_balance_loss, train_gating_loss, train_acc, train_gating_acc, avg_router_time, avg_experts_time, avg_post_time, avg_total_time = train_results
        else:
            train_loss, train_balance_loss, train_gating_loss, train_acc, train_gating_acc = train_results
        
        test_loss, test_balance_loss, test_gating_loss, test_acc, test_gating_acc, test_gtsrb_acc, test_cifar10_acc, test_mnist_acc, test_inference_time = None, None, None, None, None, None, None, None, None
        if epoch >= TEST_START_EPOCH and (epoch - TEST_START_EPOCH) % TEST_FREQUENCY == 0:
            if args.meta_moe:
                test_results = test(model, test_loader, optimizer, criterion, DEVICE)
            else:
                test_results = test(model, test_loader, optimizer, criterion, DEVICE, default_meta_class=default_meta_class)
            if args.meta_moe or args.meta_moe_fine_tune:
                test_loss, test_balance_loss, test_gating_loss, test_acc, test_gating_acc, test_gtsrb_acc, test_cifar10_acc, test_mnist_acc, avg_router_time_test, avg_experts_time_test, avg_post_time_test, avg_total_time_test = test_results
                test_inference_time = avg_total_time_test
            else:
                test_loss, test_balance_loss, test_gating_loss, test_acc, test_gating_acc, test_gtsrb_acc, test_cifar10_acc, test_mnist_acc, test_inference_time = test_results
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
            test_cifar10_accs.append(test_cifar10_acc)
            test_mnist_accs.append(test_mnist_acc)
            # test_tsrd_accs.append(test_tsrd_acc)
            # test_btsd_accs.append(test_btsd_acc)
            # test_etsd_accs.append(test_etsd_acc)

        print(f"{datetime.now()}")
        print(f"Epoch {epoch+1}/{EPOCHS}:")
        print(f"Train loss: {train_loss:.4f}, Train Balance Loss: {train_balance_loss:.4f}, Train Gating Loss: {train_gating_loss:.4f}, Train Acc: {train_acc:.4f}, Train Gating Acc: {train_gating_acc:.4f}")
        if args.meta_moe:
            print(f"Train Avg Router Time per image: {avg_router_time:.6f} seconds")
            print(f"Train Avg Experts Time per image: {avg_experts_time:.6f} seconds")
            print(f"Train Avg Post-Experts Time per image: {avg_post_time:.6f} seconds")
            print(f"Train Avg Total Inference Time per image: {avg_total_time:.6f} seconds")
        if test_loss is not None:
            print(f"Test loss: {test_loss:.4f}, Test Balance Loss: {test_balance_loss:.4f}, Test Gating Loss: {test_gating_loss:.4f}, Test Acc: {test_acc:.4f}, Test Gating Acc: {test_gating_acc:.4f}")
            if args.meta_moe or args.meta_moe_fine_tune:
                print(f"Test GTSRB Acc: {test_gtsrb_acc:.4f}, Test CIFAR10 Acc: {test_cifar10_acc:.4f}, Test MNIST Acc: {test_mnist_acc:.4f}")
                print(f"Test Avg Router Time per image: {avg_router_time_test:.6f} seconds")
                print(f"Test Avg Experts Time per image: {avg_experts_time_test:.6f} seconds")
                print(f"Test Avg Post-Experts Time per image: {avg_post_time_test:.6f} seconds")
                print(f"Test Avg Total Inference Time per image: {avg_total_time_test:.6f} seconds")
            else:
                print(f"Avg Inference Time per image: {test_inference_time:.6f} seconds")
        print(f"Epoch time: {epoch_time:.2f} seconds")

        if test_acc is not None and test_acc > best_acc:
            best_acc = test_acc
            suffix = "_robust" if args.adversarial_training else "_og"
            save_path = os.path.join(OUTPUT_DIR, f"meta_moe_{args.model_arch}_best.pth" if args.meta_moe else f"{args.dataset.lower()}_{args.model_arch}_best{suffix}.pth")
            torch.save(model, save_path)
            if not args.meta_moe and args.save_state_dict:
                state_dict_path = os.path.join(OUTPUT_DIR, f"{args.model_arch}_{args.dataset.lower()}_best_state_dict{suffix}.pth")
                torch.save(model.state_dict(), state_dict_path)         
            print(f"New best accuracy: {best_acc:.4f}")
        print()

        if (epoch + 1) % 5 == 0 or epoch == EPOCHS - 1:
            plot_metrics(
                train_losses, test_losses, train_accs, test_accs,
                train_balance_losses, test_balance_losses,
                train_gating_losses, test_gating_losses,
                train_gating_accs, test_gating_accs,
                test_gtsrb_accs, test_cifar10_accs,
                EPOCHS, TEST_START_EPOCH, TEST_FREQUENCY, OUTPUT_DIR,
                meta_moe=args.meta_moe
            )

    print(f"Training completed. Best Accuracy: {best_acc:.4f}")
    print(f"Total training time: {total_training_time:.2f} seconds")
    print(f"Average time per epoch: {total_training_time/EPOCHS:.2f} seconds")
    if test_inference_times:
        avg_test_inference_time = sum(test_inference_times) / len(test_inference_times)
        print(f"Average inference time per image across test epochs: {avg_test_inference_time:.6f} seconds")
    
    if args.export_onnx:
        best_model_path = os.path.join(OUTPUT_DIR, f"meta_moe_{args.model_arch}_best.pth" if args.meta_moe else f"{args.dataset.lower()}_{args.model_arch}_best{suffix}.pth")
        model = torch.load(best_model_path, map_location=DEVICE, weights_only=False)
        export_to_onnx(model=model, config=config, device=DEVICE, output_dir=OUTPUT_DIR, dataset_name="MetaMoE" if args.meta_moe else args.dataset, model_arch=args.model_arch)

    if args.art_attack:
        if args.meta_moe:
            best_model_path = os.path.join(OUTPUT_DIR, f"meta_moe_{args.model_arch}_best.pth")
        else:
            best_model_path = os.path.join(OUTPUT_DIR, f"{args.dataset.lower()}_{args.model_arch}_best{suffix}.pth")
        model = torch.load(best_model_path, map_location=DEVICE, weights_only=False)
        model.eval()  # Ensure the model is in evaluation mode
        test_adversarial_robustness(model, test_loader, DEVICE)
    
    if args.meta_moe and args.visualize_robustness:
        # Assuming the model is still in scope and loaded above
        visualize_robustness(model, test_loader, DEVICE, OUTPUT_DIR)
        
    if args.archive_params:
        archive_params(args, config, OUTPUT_DIR)

if __name__ == '__main__':
    if os.name == 'nt':
        from multiprocessing import freeze_support
        freeze_support()
    main()