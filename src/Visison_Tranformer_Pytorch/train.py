import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from vision_transformer import VisionTransformer, VisionTransformerConfig

# Training Params
BATCH_SIZE = 128
EPOCHS = 50
LEARNING_RATE = 3e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model with configurations
config = VisionTransformerConfig

# Data augmentation and normalization
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
])

# Load dataset from CIFAR-10
train_dataset = datasets.CIFAR10(root='.data', train = True, download=True, transform=transform_train)
test_dataset = datasets.CIFAR10(root='.data', train = False, download=True, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

model = VisionTransformer(config).to(DEVICE)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.05)

# learning rate 
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data,target) in enumerate(tqdm(loader)):
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
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data,target) in enumerate(tqdm(loader)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        
        total_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    return avg_loss, accuracy

# Training Loop
best_acc = 0
for epoch in range(EPOCHS):
    train_loss, train_acc = train(model, train_loader, optimizer, criterion, DEVICE)
    test_loss, test_acc = test(model, test_loader, optimizer, criterion, DEVICE)
    scheduler.step()
    
    print(f"Epoch {epoch+1}/{EPOCHS}:")
    print(f"Train loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    print(f"Test loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), 'vit_cifer10_best.pth')
        print(f"New best accuracy: {best_acc:.4f}")
    print()
    
print(f"Training completed. Best Accuracy: {best_acc:.4f}")