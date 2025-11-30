import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='Calculate the normalization values of the dataset')
parser.add_argument('--dataset', type=str, default='unified_norm_cifar10_mnist',
                    choices=['GTSRB', 'PTSD', 'CIFAR10', 'MNIST', 'unified_norm_cifar10_mnist', 'unified_norm_gtsrb_ptsd'],
                    help='Dataset to calculate normalization values for')
args = parser.parse_args()

if __name__ == '__main__':
    # Set device to CPU for simplicity and precision in accumulation
    device = torch.device('cpu')

    # Transform for GTSRB (ImageFolder format)
    transform_imagefolder = transforms.Compose([
        transforms.Resize(32),      # Resize the smaller side to 32, preserving aspect ratio
        transforms.CenterCrop(32),  # Crop to 32x32 to ensure uniform size
        transforms.ToTensor()       # Convert to tensor with values in [0, 1]
    ])

    # Transform for CIFAR-10 and MNIST (torchvision datasets, already 32x32 or 28x28)
    transform_torchvision = transforms.Compose([
        transforms.Resize(32),      # Ensure 32x32 for MNIST
        transforms.ToTensor()       # Convert to tensor with values in [0, 1]
    ])

    # Initialize accumulators for sum and sum of squares for each channel (R, G, B)
    sum_c = torch.zeros(3, dtype=torch.float64)
    sum_squares_c = torch.zeros(3, dtype=torch.float64)
    total_pixels = 0

    def process_imagefolder_dataset(root_path, dataset_name):
        """Process datasets in ImageFolder format (e.g., GTSRB)"""
        global sum_c, sum_squares_c, total_pixels

        dataset = datasets.ImageFolder(root_path, transform=transform_imagefolder)
        print(f"Processing {dataset_name} dataset at: {root_path}")
        print(f"Number of images: {len(dataset)}")

        loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4)

        for data, _ in tqdm(loader, desc=f"Processing {dataset_name}"):
            data = data.to(device)  # Shape: (batch_size, 3, 32, 32)
            B = data.size(0)
            sum_c += data.sum(dim=[0, 2, 3]).to(torch.float64)
            sum_squares_c += (data ** 2).sum(dim=[0, 2, 3]).to(torch.float64)
            total_pixels += B * 32 * 32

    def process_cifar10_dataset(root_path):
        """Process CIFAR-10 dataset (torchvision format)"""
        global sum_c, sum_squares_c, total_pixels

        dataset = datasets.CIFAR10(root=root_path, train=True, download=False,
                                    transform=transform_torchvision)
        print(f"Processing CIFAR-10 dataset at: {root_path}")
        print(f"Number of images: {len(dataset)}")

        loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4)

        for data, _ in tqdm(loader, desc="Processing CIFAR-10"):
            data = data.to(device)  # Shape: (batch_size, 3, 32, 32)
            B = data.size(0)
            sum_c += data.sum(dim=[0, 2, 3]).to(torch.float64)
            sum_squares_c += (data ** 2).sum(dim=[0, 2, 3]).to(torch.float64)
            total_pixels += B * 32 * 32

    def process_mnist_dataset(root_path):
        """Process MNIST dataset (torchvision format, grayscale -> RGB)"""
        global sum_c, sum_squares_c, total_pixels

        # MNIST is grayscale, we need to convert to RGB for consistency
        transform_mnist = transforms.Compose([
            transforms.Resize(32),
            transforms.Grayscale(num_output_channels=3),  # Convert grayscale to RGB
            transforms.ToTensor()
        ])

        dataset = datasets.MNIST(root=root_path, train=True, download=False,
                                 transform=transform_mnist)
        print(f"Processing MNIST dataset at: {root_path}")
        print(f"Number of images: {len(dataset)}")

        loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4)

        for data, _ in tqdm(loader, desc="Processing MNIST"):
            data = data.to(device)  # Shape: (batch_size, 3, 32, 32)
            B = data.size(0)
            sum_c += data.sum(dim=[0, 2, 3]).to(torch.float64)
            sum_squares_c += (data ** 2).sum(dim=[0, 2, 3]).to(torch.float64)
            total_pixels += B * 32 * 32

    # Process datasets based on command line argument
    if args.dataset == 'GTSRB':
        process_imagefolder_dataset('data/GTSRB/Training', 'GTSRB')
    elif args.dataset == 'PTSD':
        process_imagefolder_dataset('data/PTSD/Training', 'PTSD')
    elif args.dataset == 'CIFAR10':
        process_cifar10_dataset('data')
    elif args.dataset == 'MNIST':
        process_mnist_dataset('data')
    elif args.dataset == 'unified_norm_cifar10_mnist':
        # Process CIFAR-10 and MNIST for unified normalization
        print("=" * 80)
        print("Calculating UNIFIED normalization values for CIFAR-10 and MNIST")
        print("=" * 80)
        process_cifar10_dataset('data')
        process_mnist_dataset('data')
    elif args.dataset == 'unified_norm_gtsrb_ptsd':
        # Process GTSRB and PTSD for unified normalization
        print("=" * 80)
        print("Calculating UNIFIED normalization values for GTSRB and PTSD")
        print("=" * 80)
        process_imagefolder_dataset('data/GTSRB/Training', 'GTSRB')
        process_imagefolder_dataset('data/PTSD/Training', 'PTSD')
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # Compute mean and standard deviation
    mean_c = sum_c / total_pixels
    variance_c = (sum_squares_c / total_pixels) - (mean_c ** 2)
    std_c = torch.sqrt(variance_c)

    # Print the results
    print("\n" + "=" * 80)
    print("RESULTS:")
    print("=" * 80)
    print(f"Total pixels processed: {total_pixels}")
    print(f"\nMean (R, G, B): {mean_c.tolist()}")
    print(f"Std (R, G, B): {std_c.tolist()}")
    print("\nFor config.py:")
    print(f"NORM_MEAN_R = {mean_c[0].item()}")
    print(f"NORM_MEAN_G = {mean_c[1].item()}")
    print(f"NORM_MEAN_B = {mean_c[2].item()}")
    print(f"NORM_STD_R = {std_c[0].item()}")
    print(f"NORM_STD_G = {std_c[1].item()}")
    print(f"NORM_STD_B = {std_c[2].item()}")
    print("=" * 80)