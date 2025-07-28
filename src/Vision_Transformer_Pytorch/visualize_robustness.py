import os
import torch
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
from scipy import stats

def visualize_robustness(model, test_loader, device, OUTPUT_DIR, num_perturbations=1000, perturbation_scale=0.01):
    model.eval()
    with torch.no_grad():
        min_diff = float('inf')
        best_x = None
        best_output = None
        best_gates = None
        best_target = None
        best_meta_class = None
        best_image_idx = 0
        for batch_idx, (data, target, meta_class) in enumerate(test_loader):
            data = data.to(device)
            for i in range(data.size(0)):
                x = data[i].unsqueeze(0)
                output, gates = model(x)
                gates_np = gates.cpu().numpy().squeeze()
                gates_sorted = np.sort(gates_np)[::-1]
                diff = gates_sorted[0] - gates_sorted[1]
                if diff < min_diff:
                    min_diff = diff
                    best_x = x
                    best_output = output
                    best_gates = gates_np
                    best_target = target[i].item()
                    best_meta_class = meta_class[i].item()
                    best_image_idx = batch_idx * test_loader.batch_size + i
        
        if best_x is None:
            raise ValueError("No suitable input found in the test dataset")

        print(f"Chosen image index: {best_image_idx}, Target class: {best_target}, Meta class: {best_meta_class}, Min gating prob difference: {min_diff:.4f}")
        to_pil = transforms.ToPILImage()
        pil_image = to_pil(best_x.squeeze(0).cpu())
        pil_image.save(os.path.join(OUTPUT_DIR, f"chosen_image_{best_image_idx}.png"))
        
        perturbations = torch.randn(num_perturbations, *best_x.shape[1:], device=device) * perturbation_scale
        perturbed_inputs = best_x + perturbations
        
        dense_outputs = []
        sparse_outputs = []
        gating_decisions = []
        
        num_experts = model.num_experts
        for pert in perturbed_inputs:
            model.meta_top_k = num_experts
            dense_output, _ = model(pert.unsqueeze(0))
            dense_outputs.append(dense_output.cpu().numpy())
            
            model.meta_top_k = 1
            sparse_output, gates_sparse = model(pert.unsqueeze(0))
            sparse_outputs.append(sparse_output.cpu().numpy())
            gating_decisions.append(gates_sparse.cpu().numpy())
        
        dense_outputs = np.array(dense_outputs).squeeze()
        sparse_outputs = np.array(sparse_outputs).squeeze()
        gating_decisions = np.array(gating_decisions).squeeze()
        
        perturbation_magnitudes = torch.norm(perturbations.view(num_perturbations, -1), dim=1).cpu().numpy()
        dense_diff = np.linalg.norm(dense_outputs - best_output.cpu().numpy(), axis=1)
        sparse_diff = np.linalg.norm(sparse_outputs - best_output.cpu().numpy(), axis=1)
        
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.scatter(perturbation_magnitudes, dense_diff, alpha=0.5, s=10, label='Data Points')
        slope, intercept, r_value, _, _ = stats.linregress(perturbation_magnitudes, dense_diff)
        trend_line = intercept + slope * perturbation_magnitudes
        plt.plot(perturbation_magnitudes, trend_line, color='red', linestyle='--', label=f'Trend (R² = {r_value**2:.2f})')
        plt.title(f'Dense MetaMoE (top-k={num_experts}): Output Difference')
        plt.xlabel('Perturbation Magnitude')
        plt.ylabel('Output Difference')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.subplot(1, 2, 2)
        selected_expert = np.argmax(gating_decisions, axis=1)
        for expert in np.unique(selected_expert):
            mask = selected_expert == expert
            plt.scatter(perturbation_magnitudes[mask], sparse_diff[mask], alpha=0.5, s=10, label=f'Expert {expert}')
        plt.title('Sparse MetaMoE (top-k=1): Output Difference')
        plt.xlabel('Perturbation Magnitude')
        plt.ylabel('Output Difference')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'robustness_visualization.png'))
        plt.close()
        plt.figure(figsize=(8, 5))
        plt.scatter(perturbation_magnitudes, selected_expert, alpha=0.5, s=10, label='Data Points')
        plt.title('Sparse MetaMoE (top-k=1): Selected Expert')
        plt.xlabel('Perturbation Magnitude')
        plt.ylabel('Selected Expert Index')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(OUTPUT_DIR, 'gating_decisions.png'))
        plt.close()
        print("Robustness visualization completed. Plots saved in", OUTPUT_DIR)