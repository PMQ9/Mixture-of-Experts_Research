import torch
import torch.onnx
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import os
import sys
from datetime import datetime
import numpy as np
import csv
from PIL import Image
from torch.cuda.amp import autocast, GradScaler
from torchvision.transforms import RandAugment

from vision_transformer_moe import VisionTransformer, VisionTransformerConfig, LabelSmoothingCrossEntropy, GTSRBTestDataset

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
torch.set_float32_matmul_precision("high")

# **************** Training Params ****************
BATCH_SIZE = 128
EPOCHS = int(os.getenv('CICD_EPOCH', 400))
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CUTMIX_ALPHA = 1.0
CUTMIX_PROB = 0.5
TEST_START_EPOCH = 50
TEST_FREQUENCY = 2
WARMUP_EPOCHS = 10
LABEL_SMOOTHING = 0.1

NORMALIZATION_MEAN_R_GTSRB = 0.3432482055626116
NORMALIZATION_MEAN_G_GTSRB = 0.31312152061376486
NORMALIZATION_MEAN_B_GTSRB = 0.32248030768500435
NORMALIZATION_STD_R_GTSRB = 0.27380229614172485
NORMALIZATION_STD_G_GTSRB = 0.26033050034131744
NORMALIZATION_STD_B_GTSRB = 0.2660272789537349

NORMALIZAION_MEAN_R_CIFAR10 = 0.4914
NORMALIZAION_MEAN_G_CIFAR10 = 0.4822
NORMALIZAION_MEAN_B_CIFAR10 = 0.4465
NORMALIZATION_STD_R_CIFAR10 = 0.247
NORMALIZATION_STD_G_CIFAR10 = 0.243
NORMALIZATION_STD_B_CIFAR10 = 0.261

# **************** DevOps Params ****************
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'artifacts'))

# **************** DevOps Functions ****************
def setup_logging():
    log_file = os.path.join(OUTPUT_DIR, "training_log.txt")
    class DualOutput:
        def __init__(self, file, terminal):
            self.file = file
            self.terminal = terminal

        def write(self, message):
            self.file.write(message)
            self.terminal.write(message)
            self.file.flush()

        def flush(self):
            self.file.flush()
            self.terminal.flush()

    os.makedirs(OUTPUT_DIR, exist_ok=True)        
    log_file_handle = open(log_file, 'w', buffering=1)
    sys.stdout = sys.stderr = open(log_file, 'w', buffering=1)
    sys.stderr = DualOutput(log_file_handle, sys.__stderr__)
    print(f"Training started at {datetime.now()}\n")
    print(f"Logging to: {log_file}")

# **************** CutMix Function ****************
def cutmix(data, targets, alpha):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets = targets[indices]

    lam = np.random.beta(alpha, alpha)
    
    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
    data[:, :, bbx1:bbx2, bby1:bby2] = shuffled_data[:, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size(-1) * data.size(-2)))
    
    return data, targets, shuffled_targets, lam

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

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

        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
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

def test(model, loader, optimizer, criterion, device):
    model.eval()
    total_loss = 0
    total_balance_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tqdm(loader, desc="Testing")):
            data, target = data.to(device), target.to(device)
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
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

def plot_metrics(train_losses, test_losses, train_accs, test_accs, train_balance_losses, test_balance_losses):
    train_epochs = list(range(EPOCHS))
    test_epochs = list(range(TEST_START_EPOCH, EPOCHS, TEST_FREQUENCY))

    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    fig.suptitle('Training and Testing Metrics', fontsize=16)

    axes[0, 0].plot(train_epochs[:len(train_losses)], train_losses, label='Train Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Classification Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    axes[0, 1].plot(test_epochs[:len(test_losses)], test_losses, label='Test Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title(f'Test Classification Loss (Starting from Epoch {TEST_START_EPOCH})')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    axes[1, 0].plot(train_epochs[:len(train_accs)], train_accs, label='Train Accuracy')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].set_title('Training Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    axes[1, 1].plot(test_epochs[:len(test_accs)], test_accs, label='Test Accuracy')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].set_title(f'Test Accuracy (Starting from Epoch {TEST_START_EPOCH})')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    axes[2, 0].plot(train_epochs[:len(train_balance_losses)], train_balance_losses, label='Train Balance Loss')
    axes[2, 0].set_xlabel('Epoch')
    axes[2, 0].set_ylabel('Balance Loss')
    axes[2, 0].set_title('Training Balance Loss')
    axes[2, 0].legend()
    axes[2, 0].grid(True)

    axes[2, 1].plot(test_epochs[:len(test_balance_losses)], test_balance_losses, label='Test Balance Loss')
    axes[2, 1].set_xlabel('Epoch')
    axes[2, 1].set_ylabel('Balance Loss')
    axes[2, 1].set_title(f'Test Balance Loss (Starting from Epoch {TEST_START_EPOCH})')
    axes[2, 1].legend()
    axes[2, 1].grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(OUTPUT_DIR, "training_metrics.png"))
    plt.close(fig)

# **************** Main Functions ****************
def main():
    config = VisionTransformerConfig(num_class = 43)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    setup_logging()

    transform_train = transforms.Compose([
        transforms.Resize(32), 
        RandAugment(num_ops=2, magnitude=9), # If overfitting decreases but training becomes too slow or unstable, reduce num_ops to 1 or magnitude to 5-7. If overfitting, increase magnitude to 10-12 or num_ops to 3
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((NORMALIZATION_MEAN_R_GTSRB, NORMALIZATION_MEAN_G_GTSRB, NORMALIZATION_MEAN_B_GTSRB), (NORMALIZATION_STD_R_GTSRB, NORMALIZATION_STD_G_GTSRB, NORMALIZATION_STD_B_GTSRB)),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.2)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize((NORMALIZATION_MEAN_R_GTSRB, NORMALIZATION_MEAN_G_GTSRB, NORMALIZATION_MEAN_B_GTSRB), (NORMALIZATION_STD_R_GTSRB, NORMALIZATION_STD_G_GTSRB, NORMALIZATION_STD_B_GTSRB)),
    ])

    train_dir = './src/Vision_Transformer_Pytorch/data/GTSRB/Training'
    test_dir = './src/Vision_Transformer_Pytorch/data/GTSRB/Test'
    csv_file = './src/Vision_Transformer_Pytorch/data/GTSRB/Test/GT-final_test.csv'
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Training directory not found at {train_dir}")
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"Test directory not found at {test_dir}")
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"Test CSV file not found at {csv_file}")
    
    num_workers = min(os.cpu_count(), 8)
    prefetch_factor = 4

    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform_train)
    test_dataset = GTSRBTestDataset(root=test_dir, csv_file=csv_file, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0, prefetch_factor=prefetch_factor)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, persistent_workers=True, pin_memory=True)

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
            torch.save(model, os.path.join(OUTPUT_DIR, "vit_gtsrb_best.pth"))
            print(f"New best accuracy: {best_acc:.4f}")
        print()

        if (epoch + 1) % 5 == 0 or epoch == EPOCHS - 1:
            plot_metrics(train_losses, test_losses, train_accs, test_accs, train_balance_losses, test_balance_losses)

    print(f"Training completed. Best Accuracy: {best_acc:.4f}")
    print(f"Total training time: {total_training_time:.2f} seconds")
    print(f"Average time per epoch: {total_training_time/EPOCHS:.2f} seconds")
    print("\nExporting model to ONNX...")

    best_model_path = os.path.join(OUTPUT_DIR, "vit_gtsrb_best.pth")
    model = torch.load(best_model_path, map_location=DEVICE)
    model.eval()

    class ExpertTracer(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, x):
            out, balance_losses = self.model(x)
            expert_traces = [torch.zeros_like(out) for _ in range(config.num_experts)]
            return (out, *expert_traces)
        
    wrapped_model = ExpertTracer(model).to(DEVICE)
    dummy_input = torch.randn(1, 3, 32, 32).to(DEVICE)  # (batch, channels, height, width)
    onnx_path = os.path.join(OUTPUT_DIR, "vit_gtsrb_best.onnx")
    torch.onnx.export(
        wrapped_model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=14,
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
        },
        verbose=True,
        do_constant_folding=True,
    )
    print(f"ONNX model saved to: {onnx_path}")

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    main()
