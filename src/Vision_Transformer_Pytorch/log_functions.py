import os
from datetime import datetime
import shutil
import json
import sys
import torch
import torch.nn as nn
from dataclasses import asdict
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'artifacts'))

# **************** Logging Function ****************
def setup_logging(output_dir):
    log_file = os.path.join(output_dir, "training_log.txt")
    os.makedirs(output_dir, exist_ok=True) 

    class TerminalOutput:
        def __init__(self, file):
            self.file = file
        def write(self, x):
            self.file.write(x)
            self.file.flush()
        def flush(self):
            self.file.flush()
           
    log_file_handle = open(log_file, 'w', buffering=1)
    sys.stdout = TerminalOutput(log_file_handle)
    print(f"Training started at {datetime.now()}\n")
    print(f"Logging to: {log_file}")

# **************** Archive Trained Models for Fine Tuning ****************
def archive_params(args, config, output_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"training_{timestamp}"
    artifacts_dir = os.path.join(output_dir, folder_name)
    os.makedirs(artifacts_dir, exist_ok=True)
    for item in os.listdir(output_dir):
        src = os.path.join(output_dir, item)
        if (
            item == folder_name  # Skip current run's folder (just created)
            or item == "results"  # Skip results folder
            or (os.path.isdir(src) and item.startswith("training_"))  # Skip all training_* folders
        ):
            continue
        dst = os.path.join(artifacts_dir, item)
        shutil.move(src, dst)
    config_log = {
        "training_parameters": vars(args),
        "model_config": asdict(config),
        "timestamp": timestamp
    }
    log_file_path = os.path.join(artifacts_dir, "training_config.json")
    with open(log_file_path, 'w') as f:
        json.dump(config_log, f, indent=4)
    print(f"\nAll artifacts moved to: {artifacts_dir}")
    print(f"Training configuration saved to: {log_file_path}")
    return artifacts_dir

# **************** Plot training metrics ****************
def plot_metrics(train_losses, test_losses, train_accs, test_accs, 
                 train_balance_losses, test_balance_losses,
                 train_gating_losses, test_gating_losses,
                 train_gating_accs, test_gating_accs,
                 test_gtsrb_accs, test_ptsd_accs, test_tsrd_accs,
                 plot_epochs, plot_test_start_epoch, plot_test_freq, output_dir,
                 meta_moe=False):
    train_epochs = list(range(plot_epochs))
    test_epochs = list(range(plot_test_start_epoch, plot_epochs, plot_test_freq))

    if meta_moe:
        fig, axes = plt.subplots(5, 4, figsize=(45, 27))
        fig.suptitle('MetaMoE Training and Testing Metrics', fontsize=16)

        # Classification Loss
        axes[0, 0].plot(train_epochs[:len(train_losses)], train_losses, label='Train Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training Classification Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        axes[0, 1].plot(test_epochs[:len(test_losses)], test_losses, label='Test Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title(f'Test Classification Loss (Starting from Epoch {plot_test_start_epoch})')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # Classification Accuracy
        axes[1, 0].plot(train_epochs[:len(train_accs)], train_accs, label='Train Accuracy')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].set_title('Training Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        axes[1, 1].plot(test_epochs[:len(test_accs)], test_accs, label='Test Accuracy')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].set_title(f'Test Accuracy (Starting from Epoch {plot_test_start_epoch})')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        # Per-Meta-Class Accuracy
        axes[2, 0].plot(test_epochs[:len(test_gtsrb_accs)], test_gtsrb_accs, label='GTSRB Test Accuracy')
        axes[2, 0].set_xlabel('Epoch')
        axes[2, 0].set_ylabel('Accuracy')
        axes[2, 0].set_title(f'GTSRB Test Accuracy (Starting from Epoch {plot_test_start_epoch})')
        axes[2, 0].legend()
        axes[2, 0].grid(True)

        axes[2, 1].plot(test_epochs[:len(test_ptsd_accs)], test_ptsd_accs, label='PTSD Test Accuracy')
        axes[2, 1].set_xlabel('Epoch')
        axes[2, 1].set_ylabel('Accuracy')
        axes[2, 1].set_title(f'PTSD Test Accuracy (Starting from Epoch {plot_test_start_epoch})')
        axes[2, 1].legend()
        axes[2, 1].grid(True)

        axes[2, 2].plot(test_epochs[:len(test_tsrd_accs)], test_tsrd_accs, label='TSRD Test Accuracy')
        axes[2, 2].set_xlabel('Epoch')
        axes[2, 2].set_ylabel('Accuracy')
        axes[2, 2].set_title(f'TSRD Test Accuracy (Starting from Epoch {plot_test_start_epoch})')
        axes[2, 2].legend()
        axes[2, 2].grid(True)

        # axes[2, 3].plot(test_epochs[:len(test_btsd_accs)], test_btsd_accs, label='BTSD Test Accuracy')
        # axes[2, 3].set_xlabel('Epoch')
        # axes[2, 3].set_ylabel('Accuracy')
        # axes[2, 3].set_title(f'BTSD Test Accuracy (Starting from Epoch {plot_test_start_epoch})')
        # axes[2, 3].legend()
        # axes[2, 3].grid(True)

        # axes[2, 4].plot(test_epochs[:len(test_etsd_accs)], test_etsd_accs, label='ETSD Test Accuracy')
        # axes[2, 4].set_xlabel('Epoch')
        # axes[2, 4].set_ylabel('Accuracy')
        # axes[2, 4].set_title(f'ETSD Test Accuracy (Starting from Epoch {plot_test_start_epoch})')
        # axes[2, 4].legend()
        # axes[2, 4].grid(True)

        # Gating Loss
        axes[3, 0].plot(train_epochs[:len(train_gating_losses)], train_gating_losses, label='Train Gating Loss')
        axes[3, 0].set_xlabel('Epoch')
        axes[3, 0].set_ylabel('Loss')
        axes[3, 0].set_title('Training Gating Loss')
        axes[3, 0].legend()
        axes[3, 0].grid(True)

        axes[3, 1].plot(test_epochs[:len(test_gating_losses)], test_gating_losses, label='Test Gating Loss')
        axes[3, 1].set_xlabel('Epoch')
        axes[3, 1].set_ylabel('Loss')
        axes[3, 1].set_title(f'Test Gating Loss (Starting from Epoch {plot_test_start_epoch})')
        axes[3, 1].legend()
        axes[3, 1].grid(True)

        axes[4, 0].plot(train_epochs[:len(train_gating_accs)], train_gating_accs, label='Train Gating Accuracy')
        axes[4, 0].set_xlabel('Epoch')
        axes[4, 0].set_ylabel('Accuracy')
        axes[4, 0].set_title('Training Gating Accuracy')
        axes[4, 0].legend()
        axes[4, 0].grid(True)

        axes[4, 1].plot(test_epochs[:len(test_gating_accs)], test_gating_accs, label='Test Gating Accuracy')
        axes[4, 1].set_xlabel('Epoch')
        axes[4, 1].set_ylabel('Accuracy')
        axes[4, 1].set_title(f'Test Gating Accuracy (Starting from Epoch {plot_test_start_epoch})')
        axes[4, 1].legend()
        axes[4, 1].grid(True)

    else:
        fig, axes = plt.subplots(3, 2, figsize=(15, 18))
        fig.suptitle('Training and Testing Metrics', fontsize=16)

        axes[0, 0].plot(train_epochs[:len(train_losses)], train_losses, label='Train Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training Classification Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        axes[0, 1].plot(test_epochs[:len(test_losses)], test_losses, label='Test Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title(f'Test Classification Loss (Starting from Epoch {plot_test_start_epoch})')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        axes[1, 0].plot(train_epochs[:len(train_accs)], train_accs, label='Train Accuracy')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].set_title('Training Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        axes[1, 1].plot(test_epochs[:len(test_accs)], test_accs, label='Test Accuracy')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].set_title(f'Test Accuracy (Starting from Epoch {plot_test_start_epoch})')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        axes[2, 0].plot(train_epochs[:len(train_balance_losses)], train_balance_losses, label='Train Balance Loss')
        axes[2, 0].set_xlabel('Epoch')
        axes[2, 0].set_ylabel('Balance Loss')
        axes[2, 0].set_title('Training Balance Loss')
        axes[2, 0].legend()
        axes[2, 0].grid(True)

        axes[2, 1].plot(test_epochs[:len(test_balance_losses)], test_balance_losses, label='Test Balance Loss')
        axes[2, 1].set_xlabel('Epoch')
        axes[2, 1].set_ylabel('Balance Loss')
        axes[2, 1].set_title(f'Test Balance Loss (Starting from Epoch {plot_test_start_epoch})')
        axes[2, 1].legend()
        axes[2, 1].grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(output_dir, "training_metrics.png"))
    plt.close(fig)

# **************** Export to ONNX ****************
def export_to_onnx(model, config, device, output_dir, dataset_name, model_arch):
    print("\nExporting model to ONNX...")
    model.eval()
    
    class ExpertTracer(nn.Module):
        def __init__(self, model, is_meta_moe=False):
            super().__init__()
            self.model = model
            self.is_meta_moe = is_meta_moe
            self.num_experts = config.num_experts if not is_meta_moe else 2  # MetaMoE has 2 experts (GTSRB, PTSD)
        def forward(self, x):
            if self.is_meta_moe:
                out, gates = self.model(x)
                expert_traces = [torch.zeros_like(out) for _ in range(self.num_experts)]
                return (out, *expert_traces)
            else:
                out, balance_losses = self.model(x)
                expert_traces = [torch.zeros_like(out) for _ in range(self.num_experts)]
                return (out, *expert_traces)

    from vision_transformer_moe import MetaMoE
    wrapped_model = ExpertTracer(model, is_meta_moe=isinstance(model, MetaMoE)).to(device)
    dummy_input = torch.randn(1, 3, 32, 32).to(device)    
    onnx_path = os.path.join(output_dir, f"{dataset_name.lower()}_{model_arch}.onnx")

    torch.onnx.export(
        wrapped_model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=14,
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
        },
        verbose=False,
        do_constant_folding=True,
    )
    print(f"ONNX model saved to: {onnx_path}")