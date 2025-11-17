import os
import sys
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
import math
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent
from torch.nn import KLDivLoss

from vision_transformer_moe import VisionTransformer, VisionTransformerConfig, LabelSmoothingCrossEntropy, TrafficSignTrainDataset, TrafficSignTestDataset
from vision_transformer_moe import MetaGatingNet, CombinedDataset, MetaMoE
from log_functions import setup_logging, archive_params, plot_metrics, export_to_onnx
from augmentation_functions import cutmix
from visualize_robustness import visualize_robustness
from model_wrapper import ModelWrapper, LogitsWrapper, create_model, get_num_classes
from config import (
    DEFAULT_PARAMS, GTSRB_NORM, CIFAR10_NORM, MNIST_NORM, UNIFIED_NORM
)
from config import apply_config_overrides
from model_utils import resolve_model_path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'artifacts'))
PRETRAINED_MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'artifacts', 'results'))

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# Only set expandable_segments if PyTorch version supports it (2.1+)
if hasattr(torch.version, 'cuda') and torch.version.cuda:
    torch_version = tuple(map(int, torch.__version__.split('.')[:2]))
    if torch_version >= (2, 1):
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
parser.add_argument('--gating_loss_weight', type=float, default=1.0, help='Weight for MetaGatingNet supervision loss')
# training
parser.add_argument('--export_onnx', type=bool, default=True, help='Export trained model to ONNX')
parser.add_argument('--save_state_dict', action='store_true', help='Additionally save state_dict for non-MetaMoE models')
# meta_moe
parser.add_argument('--meta_moe', action='store_true', help='Train MetaMoE model with pre-trained experts')
parser.add_argument('--num_meta_experts', type=int, default=2, help='Number of experts to in MetaMoE')
parser.add_argument('--meta_top_k', type=int, default=1, help='Number of top experts to use in MetaMoE')
parser.add_argument('--fine_tune_meta_moe', action='store_true', help='Enable fine-tuning mode for MetaMoE by adding a new expert')
parser.add_argument('--gating_backbone', type=str, default='ultra_verifiable_cnn', choices=['convnext_tiny', 'convnextv2_femto', 'resnet18', 'efficientnet_b0', 'small_cnn', 'tiny_cnn', 'micro_cnn', 'nnv_cnn', 'ultra_verifiable_cnn'], help='Backbone architecture for the MetaGatingNet router')
parser.add_argument('--adv_gating_train', action='store_true', help='Enable adversarial training for MetaGatingNet router')
parser.add_argument('--gating_epsilon', type=float, default=8.0/255.0, help='Epsilon for adversarial gating training (default: 8/255 to match formal verification)')
# loading pretrained experts
parser.add_argument('--model_arch', type=str, default='ultra_verifiable_cnn', choices=['vit_moe', 'small_cnn', 'tiny_cnn', 'micro_cnn', 'nnv_cnn', 'ultra_verifiable_cnn', 'resnet50', 'resnet101', 'convnext_tiny', 'efficientnet_b0', 'vit_base',
                                                                                'convnextv2_tiny', 'convnext_small', 'convnext_large', 'vit_large'], help='Model architecture to use')
# parser.add_argument('--gtsrb_model_path', type=str, default=os.path.join(PRETRAINED_MODEL_DIR, "gtsrb_*_best.pth"), help='Path or pattern to pre-trained GTSRB model (supports wildcards)')
parser.add_argument('--cifar10_model_path', type=str, default=os.path.join(PRETRAINED_MODEL_DIR, "cifar10_*_best.pth"), help='Path or pattern to pre-trained CIFAR10 model (supports wildcards)')
parser.add_argument('--mnist_model_path', type=str, default=os.path.join(PRETRAINED_MODEL_DIR, "mnist_*_best.pth"), help='Path or pattern to pre-trained MNIST model (supports wildcards)')
# robustness
parser.add_argument('--art_attack', action='store_true', help='Initiate Adversarial Robustness Toolbox')
parser.add_argument('--art_attack_mode', type=str, default='PGD', choices=['FGM', 'PGD'], help='Attack mode in ART')
parser.add_argument('--at_mode', type=str, default='PGD', choices=['PGD', 'TRADES'], help='Attack modes')
parser.add_argument('--visualize_robustness', action='store_true', help='visualize model switching experts')
parser.add_argument('--adv_training', action='store_true', help='Enable adversarial training for individual experts')
parser.add_argument('--trades_beta', type=float, default=6.0, help='Beta for TRADES regularization')

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

if args.dataset == 'GTSRB' or 'CIFAR10':
    pgd_epsilon = DEFAULT_PARAMS['gpd_attack_epsilon_cifar_gtsrb']
    pgd_iter = DEFAULT_PARAMS['gpd_attack_iter']
elif args.dataset == 'MNIST':
    pgd_epsilon = DEFAULT_PARAMS['gpd_attack_epsilon_cifar_gtsrb']
    pgd_iter = DEFAULT_PARAMS['gpd_attack_iter']
else:
    pgd_epsilon = DEFAULT_PARAMS['gpd_attack_epsilon_default']
    pgd_iter = DEFAULT_PARAMS['gpd_attack_iter_default']

def pgd_attack(model, data, target, epsilon=pgd_epsilon, alpha=0.01, num_iter=pgd_iter, device=DEVICE):
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

def pgd_gating_attack(model, data, meta_class, epsilon=8.0/255.0, alpha=0.01, num_iter=10, device=DEVICE):
    model.eval()
    data_adv = data.clone().detach().to(device)
    original_data = data.clone().detach().to(device)
    gating_criterion = nn.CrossEntropyLoss()
    
    for _ in range(num_iter):
        data_adv.requires_grad_(True)
        with torch.enable_grad():
            output_adv, gates_adv = model(data_adv)
            loss = gating_criterion(gates_adv, meta_class)
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
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
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

                if args.adv_gating_train:
                    data_adv = pgd_gating_attack(model, data, meta_class, epsilon=args.gating_epsilon)
                    output_adv, gates_adv = model(data_adv)
                    gating_loss_adv = gating_criterion(gates_adv, meta_class)
                    
                    if args.at_mode == 'trades':
                        kl_loss = KLDivLoss(reduction='batchmean')
                        kl_div = kl_loss(F.log_softmax(gates_adv, dim=1), F.softmax(gates, dim=1))
                        gating_total_adv = gating_loss_adv + args.trades_beta * kl_div
                        total_loss_combined = (total_loss_combined + GATING_LOSS_WEIGHT * gating_total_adv) / 2
                    else:  # PGD
                        total_loss_combined = (total_loss_combined + GATING_LOSS_WEIGHT * gating_loss_adv) / 2
                
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
                
                if not args.meta_moe and args.adv_training:
                    data_adv = pgd_attack(model, data, target)
                    output_adv, balance_losses_adv = model(data_adv)
                    if args.at_mode == 'trades':
                        kl_loss = KLDivLoss(reduction='batchmean')
                        adv_ce_loss = criterion(output_adv, target)
                        kl_div = kl_loss(F.log_softmax(output_adv, dim=1), F.softmax(output, dim=1))  # Note: Use clean output here
                        cls_loss = adv_ce_loss + args.trades_beta * kl_div
                    else:
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
                #gtsrb_mask = meta_class == 0
                cifar10_mask = meta_class == 0
                mnist_mask = meta_class == 1
                # if gtsrb_mask.any():
                #     gtsrb_correct += predicted[gtsrb_mask].eq(target[gtsrb_mask]).sum().item()
                #     gtsrb_total += gtsrb_mask.sum().item()
                if cifar10_mask.any():
                    cifar10_correct += predicted[cifar10_mask].eq(target[cifar10_mask]).sum().item()
                    cifar10_total += cifar10_mask.sum().item()
                if mnist_mask.any():
                    mnist_correct += predicted[mnist_mask].eq(target[mnist_mask]).sum().item()
                    mnist_total += mnist_mask.sum().item()
        
    avg_loss = total_loss / len(loader)
    avg_balance_loss = total_balance_loss / len(loader) if not args.meta_moe else 0
    avg_gating_loss = total_gating_loss / len(loader) if args.meta_moe else 0
    accuracy = correct / total
    gating_accuracy = gating_correct / total if args.meta_moe else 0
    gtsrb_accuracy = gtsrb_correct / gtsrb_total if gtsrb_total > 0 and args.meta_moe else 0
    cifar10_accuracy = cifar10_correct / cifar10_total if cifar10_total > 0 and args.meta_moe else 0
    mnist_accuracy = mnist_correct / mnist_total if mnist_total > 0 and args.meta_moe else 0
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
        # 'GTSRB': {
        #     'num_classes': 43,
        #     'train_dir': './data/GTSRB/Training',
        #     'test_dir': './data/GTSRB/Test',
        #     'csv_file': './data/GTSRB/Test/testset_with_meta_class.csv',
        #     'normalization_mean': (GTSRB_NORM['mean']),
        #     'normalization_std': (GTSRB_NORM['std']),
        #     'default_meta_class': 0
        # },
        'CIFAR10': {
            'num_classes': 10,
            'train_dir': './data/CIFAR10/Training',
            'test_dir': './data/CIFAR10/Test',
            'csv_file': './data/CIFAR10/Test/testset_with_meta_class_modified_for_ab.csv',
            'normalization_mean': (CIFAR10_NORM['mean']),
            'normalization_std': (CIFAR10_NORM['std']),
            'default_meta_class': 0
        },
        'MNIST': {
            'num_classes': 10,
            'train_dir': './data/MNIST/Training',
            'test_dir': './data/MNIST/Test',
            'csv_file': './data/MNIST/Test/testset_with_meta_class_modified_for_ab.csv',
            'normalization_mean': (MNIST_NORM['mean']),
            'normalization_std': (MNIST_NORM['std']),
            'default_meta_class': 1
        }
    }

    if args.meta_moe:
        if args.fine_tune_meta_moe:
            datasets = ['GTSRB', 'CIFAR10', 'MNIST']
            num_meta_experts = 3
        else:
            datasets = ['CIFAR10', 'MNIST']
            num_meta_experts = args.num_meta_experts
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
                csv_file=os.path.join(params['train_dir'], 'train_with_meta_class_modified_for_ab.csv'),
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
            csv_file=os.path.join(train_dir, 'train_with_meta_class_modified_for_ab.csv'),
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
        print(f"Meta_MoE architecture: activate {args.meta_top_k} of {num_meta_experts} experts")
    else:
        print(f"Using config: {asdict(config)}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log_file_handle = setup_logging(OUTPUT_DIR)

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
        if args.fine_tune_meta_moe:
            # Load existing MetaMoE model
            existing_model_path = os.path.join(PRETRAINED_MODEL_DIR, f"meta_moe_convnext_tiny_best.pth")
            existing_model = torch.load(existing_model_path, map_location=DEVICE, weights_only=False)
            if not isinstance(existing_model, MetaMoE):
                raise RuntimeError(f"Loaded model from {existing_model_path} is not a MetaMoE instance")
            
            # Load new MNIST expert
            mnist_model_path = resolve_model_path(args.mnist_model_path)
            mnist_model = torch.load(mnist_model_path, map_location=DEVICE, weights_only=False)
            if isinstance(mnist_model, ModelWrapper):
                mnist_model = mnist_model.model
            mnist_model = mnist_model.to(DEVICE)
            mnist_model.eval()
            for param in mnist_model.parameters():
                param.requires_grad = False
            
            # Expand gating network
            old_gating_net = existing_model.meta_gating_net
            new_gating_net = MetaGatingNet(num_experts=num_meta_experts, backbone=args.gating_backbone).to(DEVICE)
            
            # Updated weight copying to handle potential dim mismatch
            old_fc_linear = old_gating_net.fc[0]  # Linear layer
            new_fc_linear = new_gating_net.fc[0]
            old_feature_dim = old_fc_linear.in_features
            new_feature_dim = new_fc_linear.in_features
            
            with torch.no_grad():
                if old_feature_dim == new_feature_dim:
                    new_fc_linear.weight[:old_fc_linear.out_features] = old_fc_linear.weight
                    new_fc_linear.bias[:old_fc_linear.out_features] = old_fc_linear.bias
                    logger.info("Copied old gating weights successfully.")
                else:
                    logger.warning(f"Feature dimensions mismatch ({old_feature_dim} vs {new_feature_dim}); initializing new weights randomly.") # todo: check what happens if dimension mismatch

                new_weight_slice = new_fc_linear.weight[old_fc_linear.out_features:]                
                nn.init.kaiming_uniform_(new_weight_slice, a=math.sqrt(5))
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(new_fc_linear.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                new_bias_slice = new_fc_linear.bias[old_fc_linear.out_features:]
                nn.init.uniform_(new_bias_slice, -bound, bound)
            
            # Use existing experts + new one
            experts = existing_model.experts + [mnist_model]  # Append MNIST
            
            model = MetaMoE(
                experts=experts,
                num_classes_list=num_classes_list,
                meta_gating_net=new_gating_net,
                meta_top_k=args.meta_top_k
            ).to(DEVICE)
            
            # Optimizer on new gating net only
            optimizer = optim.AdamW(
                new_gating_net.parameters(),
                lr=LEARNING_RATE,
                weight_decay=0.05,
                fused=torch.cuda.is_available()
            )
        # Old code here
        else:
            # gtsrb_model_path = resolve_model_path(args.gtsrb_model_path)
            # gtsrb_model = torch.load(gtsrb_model_path, map_location=DEVICE, weights_only=False)
            # if not isinstance(gtsrb_model, (VisionTransformer, models.ResNet, timm.models.ConvNeXt, ModelWrapper)):
            #     raise RuntimeError(f"{gtsrb_model_path} is not a supported model type")
            # if isinstance(gtsrb_model, ModelWrapper):
            #     gtsrb_model = gtsrb_model.model
            # gtsrb_model = gtsrb_model.to(DEVICE)
            # print(f"Loaded {gtsrb_model_path} as full model. Type: {type(gtsrb_model)}")

            cifar10_model_path = resolve_model_path(args.cifar10_model_path)
            cifar10_model = torch.load(cifar10_model_path, map_location=DEVICE, weights_only=False)
            if not isinstance(cifar10_model, (VisionTransformer, models.ResNet, timm.models.ConvNeXt, ModelWrapper)):
                raise RuntimeError(f"{cifar10_model_path} is not a supported model type")
            if isinstance(cifar10_model, ModelWrapper):
                cifar10_model = cifar10_model.model
            cifar10_model = cifar10_model.to(DEVICE)
            print(f"Loaded {cifar10_model_path} as full model. Type: {type(cifar10_model)}")

            mnist_model_path = resolve_model_path(args.mnist_model_path)
            mnist_model = torch.load(mnist_model_path, map_location=DEVICE, weights_only=False)
            if not isinstance(mnist_model, (VisionTransformer, models.ResNet, timm.models.ConvNeXt, ModelWrapper)):
                raise RuntimeError(f"{mnist_model_path} is not a supported model type")
            if isinstance(mnist_model, ModelWrapper):
                mnist_model = mnist_model.model
            mnist_model = mnist_model.to(DEVICE)
            print(f"Loaded {mnist_model_path} as full model. Type: {type(mnist_model)}")

            with torch.no_grad():
                dummy_input = torch.randn(1, 3, 32, 32, device=DEVICE)
                for model, expected_classes, name in [
                    # (gtsrb_model, dataset_params['GTSRB']['num_classes'], "GTSRB"),
                    (cifar10_model, dataset_params['CIFAR10']['num_classes'], "CIFAR10"),
                    (mnist_model, dataset_params['MNIST']['num_classes'], "MNIST"),
                ]:
                    output = model(dummy_input)
                    if isinstance(output, tuple):
                        output = output[0]
                    if output.shape[1] != expected_classes:
                        raise RuntimeError(f"{name} model output shape {output.shape[1]} does not match expected {expected_classes} classes")
                    print(f"{name} model output shape: {output.shape}")

            # gtsrb_model.eval()
            cifar10_model.eval()
            mnist_model.eval()
            # for param in gtsrb_model.parameters():
            #     param.requires_grad = False
            for param in cifar10_model.parameters():
                param.requires_grad = False
            for param in mnist_model.parameters():
                param.requires_grad = False

            meta_gating_net = MetaGatingNet(num_experts=num_meta_experts, backbone=args.gating_backbone).to(DEVICE)
            experts = [cifar10_model, mnist_model]
            # num_classes_list = [dataset_params[ds]['num_classes'] for ds in datasets]
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
    best_acc = 0
    best_train_acc = 0
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
            if args.meta_moe:
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

        # Track best train accuracy
        if train_acc > best_train_acc:
            best_train_acc = train_acc

        if test_loss is not None:
            test_losses.append(test_loss)
            test_accs.append(test_acc)
            test_balance_losses.append(test_balance_loss)
            test_gating_losses.append(test_gating_loss)
            test_gating_accs.append(test_gating_acc)
            test_gtsrb_accs.append(test_gtsrb_acc)
            test_cifar10_accs.append(test_cifar10_acc)
            test_mnist_accs.append(test_mnist_acc)

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
            if args.meta_moe:
                print(f"Test GTSRB Acc: {test_gtsrb_acc:.4f}, Test CIFAR10 Acc: {test_cifar10_acc:.4f}, Test MNIST Acc: {test_mnist_acc:.4f}")
                print(f"Test Avg Router Time per image: {avg_router_time_test:.6f} seconds")
                print(f"Test Avg Experts Time per image: {avg_experts_time_test:.6f} seconds")
                print(f"Test Avg Post-Experts Time per image: {avg_post_time_test:.6f} seconds")
                print(f"Test Avg Total Inference Time per image: {avg_total_time_test:.6f} seconds")
            else:
                print(f"Avg Inference Time per image: {test_inference_time:.6f} seconds")
        print(f"Epoch time: {epoch_time:.2f} seconds")

        # For MetaMoE, save best model based on GATING accuracy (not classification accuracy)
        # This prevents mode collapse where model routes everything to one expert
        if args.meta_moe:
            save_metric = test_gating_acc if test_gating_acc is not None else None
            metric_name = "gating accuracy"
        else:
            save_metric = test_acc
            metric_name = "accuracy"

        if save_metric is not None and save_metric > best_acc:
            best_acc = save_metric
            # Generate descriptive suffix based on training type
            if args.meta_moe:
                # For MetaMoE routers: use NRT (Non-Robust Training) or RT_epsX (Robust Training with epsilon)
                if args.adv_gating_train:
                    # RT (Robust Training) with epsilon value
                    epsilon_str = f"{args.gating_epsilon:.5f}".rstrip('0').rstrip('.')
                    suffix = f"_RT_eps{epsilon_str}"
                else:
                    # NRT (Non-Robust Training)
                    suffix = "_NRT"
                if args.fine_tune_meta_moe:
                    suffix += "_finetuned"
            else:
                # For individual experts: use _robust or _og
                suffix = "_robust" if args.adv_training else "_og"
            save_path = os.path.join(OUTPUT_DIR, f"meta_moe_{args.model_arch}_best{suffix}.pth" if args.meta_moe else f"{args.dataset.lower()}_{args.model_arch}_best{suffix}.pth")
            torch.save(model, save_path)
            if not args.meta_moe and args.save_state_dict:
                state_dict_path = os.path.join(OUTPUT_DIR, f"{args.model_arch}_{args.dataset.lower()}_best_state_dict{suffix}.pth")
                torch.save(model.state_dict(), state_dict_path)
            print(f"New best {metric_name}: {best_acc:.4f}")
        print()

        if (epoch + 1) % 5 == 0 or epoch == EPOCHS - 1:
            plot_metrics(
                train_losses, test_losses, train_accs, test_accs,
                train_balance_losses, test_balance_losses,
                train_gating_losses, test_gating_losses,
                train_gating_accs, test_gating_accs,
                test_gtsrb_accs, test_cifar10_accs, test_mnist_accs,
                EPOCHS, TEST_START_EPOCH, TEST_FREQUENCY, OUTPUT_DIR,
                meta_moe=args.meta_moe,
                model_arch=args.model_arch,
                adv_training=args.adv_training,
                best_train_acc=best_train_acc,
                best_test_acc=best_acc,
                dataset_name=args.dataset if not args.meta_moe else ''
            )

    print(f"Training completed. Best Accuracy: {best_acc:.4f}")
    print(f"Total training time: {total_training_time:.2f} seconds")
    print(f"Average time per epoch: {total_training_time/EPOCHS:.2f} seconds")
    if test_inference_times:
        avg_test_inference_time = sum(test_inference_times) / len(test_inference_times)
        print(f"Average inference time per image across test epochs: {avg_test_inference_time:.6f} seconds")
    
    if args.export_onnx:
        best_model_path = os.path.join(OUTPUT_DIR, f"meta_moe_{args.model_arch}_best{suffix}.pth" if args.meta_moe else f"{args.dataset.lower()}_{args.model_arch}_best{suffix}.pth")
        model = torch.load(best_model_path, map_location=DEVICE, weights_only=False)
        export_to_onnx(model=model, config=config, device=DEVICE, output_dir=OUTPUT_DIR, dataset_name="MetaMoE" if args.meta_moe else args.dataset, model_arch=args.model_arch)

        # Export to NNV-compatible ONNX for verification (individual experts only)
        if not args.meta_moe and args.model_arch in ['nnv_cnn', 'micro_cnn', 'tiny_cnn', 'small_cnn', 'ultra_verifiable_cnn']:
            import subprocess
            nnv_export_script = os.path.join(os.path.dirname(__file__), '..', 'Formal_Neural_Network_Verification', 'export_to_onnx.py')
            nnv_output_dir = os.path.join(OUTPUT_DIR, 'nnv_models')
            os.makedirs(nnv_output_dir, exist_ok=True)

            print(f"\nExporting to NNV-compatible ONNX format...")
            try:
                result = subprocess.run([
                    sys.executable, nnv_export_script,
                    '--model_path', best_model_path,
                    '--output_dir', nnv_output_dir
                ], capture_output=True, text=True, timeout=60)

                if result.returncode == 0:
                    print(f"NNV ONNX export successful! Output directory: {nnv_output_dir}")
                else:
                    print(f"NNV ONNX export failed: {result.stderr}")
            except Exception as e:
                print(f"Error during NNV ONNX export: {e}")

    if args.art_attack:
        if args.meta_moe:
            best_model_path = os.path.join(OUTPUT_DIR, f"meta_moe_{args.model_arch}_best{suffix}.pth")
        else:
            best_model_path = os.path.join(OUTPUT_DIR, f"{args.dataset.lower()}_{args.model_arch}_best{suffix}.pth")
        model = torch.load(best_model_path, map_location=DEVICE, weights_only=False)
        model.eval()
        test_adversarial_robustness(model, test_loader, DEVICE)
    
    if args.meta_moe and args.visualize_robustness:
        visualize_robustness(model, test_loader, DEVICE, OUTPUT_DIR)
        
    if args.archive_params:
        # Close log file handle and restore stdout before archiving (Windows compatibility)
        if log_file_handle:
            sys.stdout = sys.__stdout__  # Restore original stdout
            log_file_handle.close()  # Close the file handle
        archive_params(args, config, OUTPUT_DIR)

if __name__ == '__main__':
    if os.name == 'nt':
        from multiprocessing import freeze_support
        freeze_support()
    main()