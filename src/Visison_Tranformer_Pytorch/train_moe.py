import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import os

from vision_transformer_moe import VisionTransformer, VisionTransformerConfig

# Training Params
BATCH_SIZE = 128
EPOCHS = 200
LEARNING_RATE = 3e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DevOps Params
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'artifacts'))

def train(model, loader, optimizer, criterion, device, balance_loss_weight=0.1):
    model.train()
    total_loss = 0
    total_balance_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(tqdm(loader)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
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
        for batch_idx, (data, target) in enumerate(tqdm(loader)):
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

def main():
    config = VisionTransformerConfig
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])

    train_dataset = datasets.CIFAR10(root='.data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root='.data', train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True, prefetch_factor=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)

    model = VisionTransformer(config).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.1)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

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
        test_loss, test_balance_loss, test_acc = test(model, test_loader, optimizer, criterion, DEVICE)
        scheduler.step()

        epoch_time = time.time() - start_time
        total_training_time += epoch_time
        
        # Store metrics for plotting
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        train_balance_losses.append(train_balance_loss)
        test_balance_losses.append(test_balance_loss)

        print(f"Epoch {epoch+1}/{EPOCHS}:")
        print(f"Train loss: {train_loss:.4f}, Train Balance Loss: {train_balance_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Test loss: {test_loss:.4f}, Test Balance Loss: {test_balance_loss:.4f}, Test Acc: {test_acc:.4f}")
        print(f"Epoch time: {epoch_time:.2f} seconds")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "vit_cifar10_best.pth"))
            print(f"New best accuracy: {best_acc:.4f}")
        print()

        # Plot metrics every 5 epochs
        if (epoch + 1) % 2 == 0 or epoch == EPOCHS - 1:
            plot_metrics(train_losses, test_losses, train_accs, test_accs, train_balance_losses, test_balance_losses)

    print(f"Training completed. Best Accuracy: {best_acc:.4f}")
    print(f"Total training time: {total_training_time:.2f} seconds")
    print(f"Average time per epoch: {total_training_time/EPOCHS:.2f} seconds")

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    main()
