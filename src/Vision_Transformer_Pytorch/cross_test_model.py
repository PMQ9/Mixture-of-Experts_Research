import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse
import os
from tqdm import tqdm
import csv
from PIL import Image
from vision_transformer_moe import VisionTransformer, LabelSmoothingCrossEntropy

# python cross_test_model.py --test_dataset PTSD --test_csv ./../../data/PTSD/Test/testset_CSV.csv --model_path ./../../artifacts/results/vit_gtsrb_best.pth --output_csv gtsrb_on_ptsd_predictions.csv
# python cross_test_model.py --test_dataset GTSRB --test_csv ./../../data/GTSRB/Test/GT-final_test.csv --model_path ./../../artifacts/results/vit_ptsd_best.pth --output_csv ptsd_on_gtsrb_predictions.csv

# Simplified dataset class without label mapping
class CrossTestDataset:
    def __init__(self, root, csv_file, transform=None):
        self.root = root
        self.transform = transform
        self.images = []
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f, delimiter=';')
            for row in reader:
                self.images.append(row['Filename'])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "Images", self.images[idx])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

def cross_test(model, loader, device, output_csv):
    model.eval()
    predictions = []
    
    with torch.no_grad(), open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Image', 'Predicted_Class', 'Confidence'])
        
        for data in tqdm(loader, desc="Cross-Testing"):
            data = data.to(device)
            output, _ = model(data)
            probs = torch.softmax(output, dim=1)
            confidences, predicted = probs.max(1)
            
            for img_idx, pred, conf in zip(range(data.size(0)), predicted.tolist(), confidences.tolist()):
                writer.writerow([img_idx, pred, conf])
                predictions.append({'image_idx': img_idx, 'predicted_class': pred, 'confidence': conf})
    
    return predictions

def main():
    parser = argparse.ArgumentParser(description='Cross-test a Vision Transformer model')
    parser.add_argument('--test_dataset', type=str, required=True, choices=['GTSRB', 'PTSD'], help='Dataset to test on')
    parser.add_argument('--test_csv', type=str, required=True, help='Path to the test CSV file')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model file')
    parser.add_argument('--output_csv', type=str, default='cross_test_predictions.csv', help='Path to save prediction CSV')
    args = parser.parse_args()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set normalization based on test dataset
    if args.test_dataset == 'GTSRB':
        from config import NORM_MEAN_R_GTSRB, NORM_MEAN_G_GTSRB, NORM_MEAN_B_GTSRB, NORM_STD_R_GTSRB, NORM_STD_G_GTSRB, NORM_STD_B_GTSRB
        normalization_mean = (NORM_MEAN_R_GTSRB, NORM_MEAN_G_GTSRB, NORM_MEAN_B_GTSRB)
        normalization_std = (NORM_STD_R_GTSRB, NORM_STD_G_GTSRB, NORM_STD_B_GTSRB)
    else:
        from config import NORM_MEAN_R_PTSD, NORM_MEAN_G_PTSD, NORM_MEAN_B_PTSD, NORM_STD_R_PTSD, NORM_STD_G_PTSD, NORM_STD_B_PTSD
        normalization_mean = (NORM_MEAN_R_PTSD, NORM_MEAN_G_PTSD, NORM_MEAN_B_PTSD)
        normalization_std = (NORM_STD_R_PTSD, NORM_STD_G_PTSD, NORM_STD_B_PTSD)

    transform_test = transforms.Compose([
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize(normalization_mean, normalization_std),
    ])

    test_dataset = CrossTestDataset(
        root=os.path.dirname(args.test_csv),
        csv_file=args.test_csv,
        transform=transform_test
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    model = torch.load(args.model_path, map_location=DEVICE, weights_only=False)
    model.to(DEVICE)
    model.eval()

    predictions = cross_test(model, test_loader, DEVICE, args.output_csv)
    print(f"Predictions saved to {args.output_csv}")
    print("Sample predictions:")
    for pred in predictions[:5]:
        print(f"Image {pred['image_idx']}: Predicted Class {pred['predicted_class']}, Confidence {pred['confidence']:.4f}")

if __name__ == '__main__':
    main()