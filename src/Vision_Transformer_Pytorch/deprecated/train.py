import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import time

from vision_transformer import VisionTransformer, VisionTransformerConfig

# Training Params
BATCH_SIZE = 128
EPOCHS = 200
LEARNING_RATE = 3e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(tqdm(loader)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    return avg_loss, accuracy

def test(model, loader, optimizer, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tqdm(loader)):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
        
    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    return avg_loss, accuracy

def main():
    config = VisionTransformerConfig

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

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    model = VisionTransformer(config).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_acc = 0
    total_training_time = 0
    
    for epoch in range(EPOCHS):
        start_time = time.time()
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, DEVICE)
        test_loss, test_acc = test(model, test_loader, optimizer, criterion, DEVICE)
        scheduler.step()
        
        epoch_time = time.time() - start_time  # Calculate time taken for this epoch
        total_training_time += epoch_time  # Add to total training time

        print(f"Epoch {epoch+1}/{EPOCHS}:")
        print(f"Train loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Test loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
        print(f"Epoch time: {epoch_time:.2f} seconds")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'vit_cifer10_best.pth')
            print(f"New best accuracy: {best_acc:.4f}")
        print()

    print(f"Training completed. Best Accuracy: {best_acc:.4f}")
    print(f"Total training time: {total_training_time:.2f} seconds")
    print(f"Average time per epoch: {total_training_time/EPOCHS:.2f} seconds")

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    main()
