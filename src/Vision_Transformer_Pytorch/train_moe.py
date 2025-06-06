import torch
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

from vision_transformer_moe import VisionTransformer, VisionTransformerConfig

# **************** Dataset class for GTSRB ****************
class GTSRBTestDataset(Dataset):
    def __init__(self, root, csv_file, transform=None):
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV file not found at {csv_file}")
        if not os.path.exists(root):
            raise FileNotFoundError(f"Test dataset directory not found at {root}")
        self.root = root
        self.transform = transform
        self.images = []
        self.labels = []
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f, delimiter=';')
            for row in reader:
                self.images.append(row['Filename'])
                self.labels.append(int(row['ClassId']))
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "Images", self.images[idx])
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found at {img_path}")
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# **************** Training Params ****************
BATCH_SIZE = 128
EPOCHS = int(os.getenv('CICD_EPOCH', 350))
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CUTMIX_ALPHA = 1.0
CUTMIX_PROB = 0.5
TEST_START_EPOCH = 100
TEST_FREQUENCY = 2

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
            self.file.flush()  # Ensure immediate write to file

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
    
    # Adjust lambda to exactly match the pixel ratio
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
    
    for batch_idx, (data, target) in enumerate(tqdm(loader, desc="Training")):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        apply_cutmix = data.size(0) == BATCH_SIZE and np.random.rand() < CUTMIX_PROB

        if apply_cutmix:
            data, target_a, target_b, lam = cutmix(data, target, CUTMIX_ALPHA)
            output, balance_losses = model(data)
            cls_loss = lam * criterion(output, target_a) + (1 - lam) * criterion(output, target_b)
        else:
            output, balance_losses = model(data)
            cls_loss = criterion(output, target)
        
        # Aggregate balance loss across all MoE blocks
        balance_loss = sum(balance_losses) / len(balance_losses) if isinstance(balance_losses, list) else balance_losses
        total_loss_combined = cls_loss + balance_loss_weight * balance_loss
        
        total_loss_combined.backward()
        optimizer.step()
        
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
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
    
    # Plot losses
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(test_losses, label='Test Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Test Classification Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracies
    ax2.plot(train_accs, label='Train Accuracy')
    ax2.plot(test_accs, label='Test Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Test Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    # Plot balance losses
    ax3.plot(train_balance_losses, label='Train Balance Loss')
    ax3.plot(test_balance_losses, label='Test Balance Loss')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Balance Loss')
    ax3.set_title('Training and Test Balance Loss')
    ax3.legend()
    ax3.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "training_metrics.png"))
    plt.close(fig)

# **************** Main Functions ****************
def main():
    config = VisionTransformerConfig(num_class = 43)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    setup_logging()

    transform_train = transforms.Compose([
        transforms.Resize(32), 
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)), # adapt this for GTSRB
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.2)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),  # adapt this for GTSRB
    ])

    # Verify dataset paths
    train_dir = './data/GTSRB/Training'
    test_dir = './data/GTSRB/Test'
    csv_file = './data/GTSRB/Test/GT-final_test.csv'
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Training directory not found at {train_dir}")
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"Test directory not found at {test_dir}")
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"Test CSV file not found at {csv_file}")
    
    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform_train)
    test_dataset = GTSRBTestDataset(root=test_dir, csv_file=csv_file, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True, prefetch_factor=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)

    model = VisionTransformer(config).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

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
        
        # Perform testing only after TEST_START_EPOCH and every TEST_FREQUENCY epochs
        test_loss, test_balance_loss, test_acc = None, None, None
        if epoch >= TEST_START_EPOCH:
            if (epoch - TEST_START_EPOCH) % TEST_FREQUENCY == 0:
                test_loss, test_balance_loss, test_acc = test(model, test_loader, optimizer, criterion, DEVICE)
    
        scheduler.step()

        epoch_time = time.time() - start_time
        total_training_time += epoch_time
        
        # Store training metrics
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        train_balance_losses.append(train_balance_loss)

        # Store testing metrics
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
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "vit_gtsrb_best.pth"))
            print(f"New best accuracy: {best_acc:.4f}")
        print()


        # Plot metrics every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == EPOCHS - 1:

            plot_metrics(train_losses, test_losses, train_accs, test_accs, train_balance_losses, test_balance_losses)

    print(f"Training completed. Best Accuracy: {best_acc:.4f}")
    print(f"Total training time: {total_training_time:.2f} seconds")
    print(f"Average time per epoch: {total_training_time/EPOCHS:.2f} seconds")

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    main()
