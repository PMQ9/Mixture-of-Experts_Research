import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import argparse
import os
from tqdm import tqdm
from vision_transformer_moe import VisionTransformer, TrafficSignTestDataset, LabelSmoothingCrossEntropy, MetaMoE
from config import (
    NORM_MEAN_R_GTSRB, NORM_MEAN_G_GTSRB, NORM_MEAN_B_GTSRB,
    NORM_STD_R_GTSRB, NORM_STD_G_GTSRB, NORM_STD_B_GTSRB,
    NORM_MEAN_R_PTSD, NORM_MEAN_G_PTSD, NORM_MEAN_B_PTSD,
    NORM_STD_R_PTSD, NORM_STD_G_PTSD, NORM_STD_B_PTSD,
    NORM_MEAN_R_UNIFIED, NORM_MEAN_G_UNIFIED, NORM_MEAN_B_UNIFIED,
    NORM_STD_R_UNIFIED, NORM_STD_G_UNIFIED, NORM_STD_B_UNIFIED,
)

# Test commands:
# Self check PTSD: python test_model.py --dataset PTSD --test_csv ./../../data/PTSD/Test/testset_CSV.csv --model_path ./../../artifacts/results/vit_ptsd_best.pth
# Self check GTSRB: python test_model.py --dataset GTSRB --test_csv ./../../data/GTSRB/Test/GT-final_test.csv --model_path ./../../artifacts/results/vit_gtsrb_best.pth

# Test meta_moe on GTSRB: python test_model.py --dataset MetaMoE --test_set GTSRB --model_path ./../../artifacts/results/vit_meta_moe_best.pth
# Test meta_moe on GTSRB: python test_model.py --dataset MetaMoE --test_set PTSD --model_path ./../../artifacts/results/vit_meta_moe_best.pth

def test(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    total_balance_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in tqdm(loader, desc="Testing"):
            data, target = data.to(device), target.to(device)
            result = model(data)
            if isinstance(model, MetaMoE):
                output = result
                balance_losses = []
            else:
                output, balance_losses = result
            loss = criterion(output, target)
            balance_loss = sum(balance_losses) / len(balance_losses) if balance_losses else 0
            
            total_loss += loss.item()
            total_balance_loss += balance_loss
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    avg_loss = total_loss / len(loader)
    avg_balance_loss = total_balance_loss / len(loader)
    accuracy = correct / total
    return avg_loss, avg_balance_loss, accuracy

def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description='Test a Vision Transformer model')
    parser.add_argument('--dataset', type=str, required=True, choices=['GTSRB', 'PTSD', 'MetaMoE'], help='Dataset to test on (GTSRB, PTSD, or MetaMoE)')
    parser.add_argument('--test_set', type=str, choices=['GTSRB', 'PTSD'], help='Test set to use when testing MetaMoE model')
    parser.add_argument('--test_csv', type=str, help='Path to the test CSV file (required for GTSRB and PTSD)')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model file')
    args = parser.parse_args()

    # Validate arguments
    if args.dataset == 'MetaMoE':
        if args.test_set is None:
            parser.error("--test_set is required when --dataset is MetaMoE")
    else:
        if args.test_csv is None:
            parser.error("--test_csv is required when --dataset is GTSRB or PTSD")

    # Device setup
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset-specific settings
    if args.dataset == 'MetaMoE':
        if args.test_set == 'GTSRB':
            test_csv = './../../data/GTSRB/Test/GT-final_test.csv'
            train_dir = './../../data/GTSRB/Training'
            class_to_idx = datasets.ImageFolder(root=train_dir).class_to_idx
            class_to_idx = {str(int(k.lstrip('0')) if k.lstrip('0') else '0'): v for k, v in class_to_idx.items()}
        elif args.test_set == 'PTSD':
            test_csv = './../../data/PTSD/Test/testset_CSV.csv'
            train_dir = './../../data/PTSD/Training'
            ptsd_class_to_idx = datasets.ImageFolder(root=train_dir).class_to_idx
            class_to_idx = {k: v + 43 for k, v in ptsd_class_to_idx.items()}
        test_dir = os.path.dirname(test_csv)
        normalization_mean = (NORM_MEAN_R_UNIFIED, NORM_MEAN_G_UNIFIED, NORM_MEAN_B_UNIFIED)
        normalization_std = (NORM_STD_R_UNIFIED, NORM_STD_G_UNIFIED, NORM_STD_B_UNIFIED)
    else:
        test_csv = args.test_csv
        test_dir = os.path.dirname(test_csv)
        if args.dataset == 'GTSRB':
            train_dir = './../../data/GTSRB/Training'
            normalization_mean = (NORM_MEAN_R_GTSRB, NORM_MEAN_G_GTSRB, NORM_MEAN_B_GTSRB)
            normalization_std = (NORM_STD_R_GTSRB, NORM_STD_G_GTSRB, NORM_STD_B_GTSRB)
        elif args.dataset == 'PTSD':
            train_dir = './../../data/PTSD/Training'
            normalization_mean = (NORM_MEAN_R_PTSD, NORM_MEAN_G_PTSD, NORM_MEAN_B_PTSD)
            normalization_std = (NORM_STD_R_PTSD, NORM_STD_G_PTSD, NORM_STD_B_PTSD)
        class_to_idx = datasets.ImageFolder(root=train_dir).class_to_idx
        if args.dataset == 'GTSRB':
            class_to_idx = {str(int(k.lstrip('0')) if k.lstrip('0') else '0'): v for k, v in class_to_idx.items()}

    # Define test transform
    transform_test = transforms.Compose([
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize(normalization_mean, normalization_std),
    ])

    # Load test dataset
    test_dataset = TrafficSignTestDataset(
        root=test_dir,
        csv_file=test_csv,
        transform=transform_test,
        class_to_idx=class_to_idx
    )

    # Create DataLoader
    test_loader = DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Load model
    model = torch.load(args.model_path, map_location=DEVICE)
    model.to(DEVICE)
    model.eval()

    # Define criterion
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)

    # Run test
    test_loss, test_balance_loss, test_acc = test(model, test_loader, criterion, DEVICE)

    # Print results
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Balance Loss: {test_balance_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

if __name__ == '__main__':
    main()