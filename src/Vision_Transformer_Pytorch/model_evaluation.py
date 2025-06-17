import torch
import os
import argparse
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
from vision_transformer_moe import VisionTransformer, VisionTransformerConfig

# **************** Argument Parser ****************
parser = argparse.ArgumentParser(description='Evaluate a Vision Transformer with MoE model')
parser.add_argument('--dataset', type=str, default='GTSRB', choices=['GTSRB', 'PTSD'], help='Model to evaluate')
args = parser.parse_args()

# **************** Params ****************
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# **************** Normalization Values ****************
NORMALIZATION_MEAN_R_GTSRB = 0.3432482055626116
NORMALIZATION_MEAN_G_GTSRB = 0.31312152061376486
NORMALIZATION_MEAN_B_GTSRB = 0.32248030768500435
NORMALIZATION_STD_R_GTSRB = 0.27380229614172485
NORMALIZATION_STD_G_GTSRB = 0.26033050034131744
NORMALIZATION_STD_B_GTSRB = 0.2660272789537349

NORMALIZATION_MEAN_R_PTSD = 0.42227414577051153
NORMALIZATION_MEAN_G_PTSD = 0.40389899174730964
NORMALIZATION_MEAN_B_PTSD = 0.42392441068660547
NORMALIZATION_STD_R_PTSD = 0.2550717671385188
NORMALIZATION_STD_G_PTSD = 0.2273784047793104
NORMALIZATION_STD_B_PTSD = 0.22533597220675006

# **************** Functions ****************
def classify_images_in_folder(folder_path, model_path, dataset=args.dataset):
    if dataset == 'GTSRB':
        normalization_mean = (NORMALIZATION_MEAN_R_GTSRB, NORMALIZATION_MEAN_G_GTSRB, NORMALIZATION_MEAN_B_GTSRB)
        normalization_std = (NORMALIZATION_STD_R_GTSRB, NORMALIZATION_STD_G_GTSRB, NORMALIZATION_STD_B_GTSRB)
        num_classes = 43
    elif dataset == 'PTSD':
        normalization_mean = (NORMALIZATION_MEAN_R_PTSD, NORMALIZATION_MEAN_G_PTSD, NORMALIZATION_MEAN_B_PTSD)
        normalization_std = (NORMALIZATION_STD_R_PTSD, NORMALIZATION_STD_G_PTSD, NORMALIZATION_STD_B_PTSD)
        num_classes = 43
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    config = VisionTransformerConfig(
        num_class = num_classes
        #img_size=32, patch_size=4, in_chans=3, num_class=43, embed_dim=192, depth=9,
        #num_heads=12, mlp_ratio=2.0, qkv_bias=True, drop_rate=0.15, attn_drop_rate=0.1,
        #num_experts=7, top_k=3, balance_loss_weight=1.0, drop_path_rate=0.1, router_weight_reg=0.03
    )

    # Initialize and load the model
    model = VisionTransformer(config)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    # Define preprocessing based on dataset
    if dataset == 'GTSRB':
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize(normalization_mean, normalization_std),
        ])
    elif dataset == 'PTSD':
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize(normalization_mean, normalization_std),
        ])
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    model = model.to(DEVICE)

    # Supported image extensions
    valid_extensions = ('.ppm', '.jpeg', '.jpg', '.png')

    # Store results
    results = []

    # Iterate through files in the folder
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(valid_extensions):
            image_path = os.path.join(folder_path, filename)
            try:
                # Load and preprocess image
                image = Image.open(image_path).convert('RGB')
                input_tensor = transform(image).unsqueeze(0).to(DEVICE)

                # Perform inference
                with torch.no_grad():
                    output, _ = model(input_tensor)
                    probabilities = F.softmax(output, dim=1)  # Convert logits to probabilities
                    confidence, predicted_class = torch.max(probabilities, 1)
                
                # Store result
                results.append({
                    'filename': filename,
                    'predicted_class': predicted_class.item(),
                    'confidence': confidence.item() * 100  # Convert to percentage
                })
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue

    # Print results
    print("\nClassification Results:")
    print("Filename | Predicted Class | Confidence (%)")
    print("-" * 50)
    for result in results:
        print(f"{result['filename']} | {result['predicted_class']} | {result['confidence']:.2f}")

if __name__ == "__main__":
    folder_path = './data/Evaluation'
    if args.dataset == 'GTSRB':
        model_path = './artifacts/results/vit_gtsrb_best.pth'
    elif args.dataset == 'PTSD':
        model_path = './artifacts/results/vit_ptsd_best.pth'
    dataset = args.dataset 

    classify_images_in_folder(folder_path, model_path, dataset)