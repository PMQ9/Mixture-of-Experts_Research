import os
import sys
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse
from tqdm import tqdm

SRC_DIR = os.path.join(os.path.dirname(__file__), '..', 'Vision_Transformer_Pytorch')
sys.path.append(SRC_DIR)

from vision_transformer_moe import CombinedDataset, TrafficSignTestDataset, MetaMoE
from config import GTSRB_NORM, MNIST_NORM, CIFAR10_NORM  # Adjust if using UNIFIED_NORM

ARTIFACTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'artifacts'))
MODEL_PATH = os.path.join(ARTIFACTS_DIR, 'nnv' , 'meta_moe_convnext_tiny_best.pth')  # Example; update to your model
GTSRB_TEST_ROOT = './data/GTSRB/Test'  # Update to actual paths
GTSRB_TEST_CSV = './data/GTSRB/Test/testset_with_meta_class.csv'
MNIST_TEST_ROOT = './data/MNIST/Test'
MNIST_TEST_CSV = './data/MNIST/Test/testset_with_meta_class.csv'
CIFAR10_TEST_ROOT = './data/CIFAR10/Test'
CIFAR10_TEST_CSV = './data/CIFAR10/Test/testset_with_meta_class.csv'
OUTPUT_DIR = os.path.join(ARTIFACTS_DIR, 'sampled_io')
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_test_transforms(norm):
    return transforms.Compose([
        transforms.Resize((32, 32)),  # Assuming 32x32 as in your config; adjust if needed
        transforms.ToTensor(),
        transforms.Normalize(mean=norm['mean'], std=norm['std'])
    ])

def load_combined_test_dataset(batch_size=128):
    # Load individual test datasets (as in your code)
    gtsrb_test = TrafficSignTestDataset(GTSRB_TEST_ROOT, GTSRB_TEST_CSV, get_test_transforms(GTSRB_NORM), default_meta_class=0)
    mnist_test = TrafficSignTestDataset(MNIST_TEST_ROOT, MNIST_TEST_CSV, get_test_transforms(MNIST_NORM), default_meta_class=1)  # Assume CSV for MNIST
    cifar10_test = TrafficSignTestDataset(CIFAR10_TEST_ROOT, CIFAR10_TEST_CSV, get_test_transforms(CIFAR10_NORM), default_meta_class=2)  # Assume CSV

    num_classes_list = [len(set(gtsrb_test.labels)), len(set(mnist_test.labels)), len(set(cifar10_test.labels))]  # Or from config
    combined_test = CombinedDataset([gtsrb_test, mnist_test, cifar10_test], num_classes_list)
    return DataLoader(combined_test, batch_size=batch_size, shuffle=False), num_classes_list

def perturb_input(x, eps=8/255):
    """Generate uniform perturbation within L_inf ball."""
    noise = torch.empty_like(x).uniform_(-eps, eps)
    x_pert = torch.clamp(x + noise, 0, 1)
    return x_pert

def sample_io(model, test_loader, num_train_samples=10000, num_calib_samples=5000, eps=8/255, is_train_set=True):
    model.eval()
    inputs = []
    outputs = []
    count = 0
    max_samples = num_train_samples if is_train_set else num_calib_samples
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Sampling {'train' if is_train_set else 'calib'} set"):
            x, _, _ = [b.to(DEVICE) for b in batch]  # Ignore labels/meta
            for i in range(x.size(0)):
                if count >= max_samples:
                    break
                x_i = x[i].unsqueeze(0)
                x_pert = perturb_input(x_i, eps) if eps > 0 else x_i  # Perturb for robustness set I
                y, _ = model(x_pert)  # Get logits (ignore gates)
                inputs.append(x_pert.cpu())
                outputs.append(y.cpu())
                count += 1
            if count >= max_samples:
                break
    return torch.cat(inputs), torch.cat(outputs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sample inputs/outputs from MetaMoE for surrogate training.')
    parser.add_argument('--model_path', type=str, default=MODEL_PATH, help='Path to trained MetaMoE .pth')
    parser.add_argument('--eps', type=float, default=8/255, help='L_inf perturbation radius')
    args = parser.parse_args()

    # Load model
    model = torch.load(args.model_path, map_location=DEVICE)
    test_loader, num_classes_list = load_combined_test_dataset()

    # Sample train set T (any W' over I; here uniform perturbations)
    inputs_train, outputs_train = sample_io(model, test_loader, eps=args.eps, is_train_set=True)
    torch.save({'inputs': inputs_train, 'outputs': outputs_train}, os.path.join(OUTPUT_DIR, 'train_set.pt'))

    # Sample calibration set (target W over I; reuse loader for simplicity, but can use different dist)
    inputs_calib, outputs_calib = sample_io(model, test_loader, eps=args.eps, is_train_set=False)
    torch.save({'inputs': inputs_calib, 'outputs': outputs_calib}, os.path.join(OUTPUT_DIR, 'calib_set.pt'))

    print(f'Sampled {len(inputs_train)} train and {len(inputs_calib)} calib pairs saved to {OUTPUT_DIR}.')