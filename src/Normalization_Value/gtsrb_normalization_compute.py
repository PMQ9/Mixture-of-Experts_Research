import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='Calculate the normalization values of the dataset')
parser.add_argument('--dataset', type=str, default='GTSRB', choices=['GTSRB', 'PTSD'], help='Dataset to calculate')
args = parser.parse_args()

if __name__ == '__main__':
    # Set device to CPU for simplicity and precision in accumulation
    device = torch.device('cpu')

    if args.dataset == 'GTSRB':
        root = './data/GTSRB/Training'
    elif args.dataset == 'PTSD':
        root = './data/PTSD/Training'
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    transform = transforms.Compose([
        transforms.Resize(32),      # Resize the smaller side to 32, preserving aspect ratio
        transforms.CenterCrop(32),  # Crop to 32x32 to ensure uniform size
        transforms.ToTensor()       # Convert to tensor with values in [0, 1]
    ])

    # Create the dataset
    dataset = datasets.ImageFolder(root, transform=transform)
    print(f"Number of images: {len(dataset)}")

    # Create the DataLoader for batch processing
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4)

    # Initialize accumulators for sum and sum of squares for each channel (R, G, B)
    sum_c = torch.zeros(3, dtype=torch.float64)
    sum_squares_c = torch.zeros(3, dtype=torch.float64)
    total_pixels = 0

    # Iterate through the dataset to compute sums
    for data, _ in tqdm(loader, desc="Processing batches"):
        data = data.to(device)  # Shape: (batch_size, 3, 32, 32)
        B = data.size(0)        # Number of images in the batch
        sum_c += data.sum(dim=[0, 2, 3]).to(torch.float64)         # Sum over batch, height, width
        sum_squares_c += (data ** 2).sum(dim=[0, 2, 3]).to(torch.float64)  # Sum of squares
        total_pixels += B * 32 * 32  # Total pixels processed

    # Compute mean and standard deviation
    mean_c = sum_c / total_pixels
    variance_c = (sum_squares_c / total_pixels) - (mean_c ** 2)
    std_c = torch.sqrt(variance_c)

    # Print the results
    print(f"Mean (R, G, B): {mean_c.tolist()}")
    print(f"Std (R, G, B): {std_c.tolist()}")