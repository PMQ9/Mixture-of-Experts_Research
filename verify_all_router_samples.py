"""
Run alpha-beta-CROWN Verification on All MetaMoE Router Samples

This script provides a complete end-to-end verification workflow:
1. [Optional] Exports PyTorch model to ONNX
2. Automatically generates fresh VNNLIB specifications with configurable sample counts
3. Verifies all router samples (MNIST + CIFAR10)
4. Generates a comprehensive verification report

Usage:
    # Default: 10 MNIST + 10 CIFAR10 samples
    python verify_all_router_samples.py

    # Verify a PyTorch model (auto-exports to ONNX)
    python verify_all_router_samples.py \
        --model_path artifacts/training_20251020_123456/meta_moe_ultra_verifiable_cnn_best_og.pth

    # Verify specific ONNX file
    python verify_all_router_samples.py \
        --onnx_path artifacts/abcrown_models/my_router_only.onnx

    # Custom sample counts
    python verify_all_router_samples.py --num_mnist 20 --num_cifar 20

    # Custom epsilon (4/255)
    python verify_all_router_samples.py --epsilon 0.01569

    # Full custom verification
    python verify_all_router_samples.py \
        --model_path artifacts/my_model.pth \
        --num_mnist 50 --num_cifar 50 \
        --epsilon 0.01569 --timeout 120
"""

import subprocess
import sys
from pathlib import Path
import os
import time
import argparse

# Compute project root from script location (robust to working directory)
project_root = Path(__file__).parent.resolve()

# Add project paths
sys.path.insert(0, str(project_root / 'src/Vision_Transformer_Pytorch'))
sys.path.insert(0, str(project_root))

# Import the VNNLIB generation function
from prepare_router_verification import main as prepare_vnnlib


def generate_vnnlib_specs(num_mnist, num_cifar, epsilon, vnnlib_dir, no_random_sampling=False):
    """
    Generate fresh VNNLIB specifications before verification.

    Args:
        num_mnist: Number of MNIST samples
        num_cifar: Number of CIFAR10 samples
        epsilon: L-infinity perturbation bound
        vnnlib_dir: Output directory for VNNLIB files
        no_random_sampling: Disable random sampling (default: random sampling enabled)
    """
    print("="*80)
    print("Step 1: Generating Fresh VNNLIB Specifications")
    print("="*80)

    # Save original sys.argv
    original_argv = sys.argv

    # Set arguments for prepare_router_verification.py
    sys.argv = [
        'prepare_router_verification.py',
        '--num_mnist', str(num_mnist),
        '--num_cifar', str(num_cifar),
        '--epsilon', str(epsilon),
        '--output_dir', str(vnnlib_dir)
    ]

    if no_random_sampling:
        sys.argv.append('--no_random_sampling')

    try:
        # Call the preparation script
        prepare_vnnlib()
    finally:
        # Restore original sys.argv
        sys.argv = original_argv

    print("\nVNNLIB generation complete!")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description='Run alpha-beta-CROWN verification on MetaMoE router'
    )
    parser.add_argument('--num_mnist', type=int, default=10,
                        help='Number of MNIST samples to verify (default: 10)')
    parser.add_argument('--num_cifar', type=int, default=10,
                        help='Number of CIFAR10 samples to verify (default: 10)')
    parser.add_argument('--epsilon', type=float, default=8.0/255.0,
                        help='L-infinity perturbation bound (default: 8/255 to match empirical testing)')
    parser.add_argument('--timeout', type=int, default=60,
                        help='Timeout per sample in seconds (default: 60)')
    parser.add_argument('--no_random_sampling', action='store_true',
                        help='Disable random sampling and use stride sampling instead (default: random sampling enabled)')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to MetaMoE .pth model (will auto-export to ONNX)')
    parser.add_argument('--onnx_path', type=str, default=None,
                        help='Path to router ONNX file (default: auto-detect most recent *router_only.onnx)')

    args = parser.parse_args()

    # Configuration (use project_root for robust path resolution)
    abcrown_dir = (project_root / "modules/alpha-beta-CROWN/complete_verifier").resolve()
    vnnlib_dir = (project_root / "artifacts/vnnlib_specs/router").resolve()

    # Handle PyTorch model export if provided
    if args.model_path:
        model_path = Path(args.model_path)
        if not model_path.exists():
            print(f"\nERROR: Model file not found: {model_path}")
            sys.exit(1)

        print(f"\n{'='*80}")
        print("Exporting PyTorch Model to ONNX")
        print(f"{'='*80}")
        print(f"Model: {model_path}")

        # Check if export script exists
        export_script = project_root / "src/Formal_Neural_Network_Verification/alpha-beta-crown/export_router_to_abcrown.py"
        if not export_script.exists():
            print(f"\nERROR: Export script not found: {export_script}")
            sys.exit(1)

        # Run export script
        result = subprocess.run([
            sys.executable, str(export_script),
            '--model_path', str(model_path),
            '--output_dir', str(project_root / 'artifacts/abcrown_models')
        ])

        if result.returncode != 0:
            print("\nERROR: Router export failed!")
            sys.exit(1)

        print("\nRouter ONNX export complete!\n")

        # The export script creates a file with pattern: *_router_only.onnx
        # Auto-detect it as the most recent one
        onnx_dir = project_root / "artifacts/abcrown_models"
        onnx_files = list(onnx_dir.glob('*router_only.onnx'))
        if onnx_files:
            onnx_model = max(onnx_files, key=lambda p: p.stat().st_mtime).resolve()
            print(f"Using exported ONNX: {onnx_model.name}\n")
        else:
            print("\nERROR: Export succeeded but ONNX file not found!")
            sys.exit(1)

    # Determine ONNX model path
    elif args.onnx_path:
        onnx_model = Path(args.onnx_path).resolve()
        if not onnx_model.exists():
            print(f"\nERROR: ONNX file not found: {onnx_model}")
            print("Please check the path and try again.")
            sys.exit(1)
        print(f"\nUsing specified ONNX model: {onnx_model.name}")
    else:
        # Auto-detect latest ONNX in artifacts/abcrown_models/
        onnx_dir = project_root / "artifacts/abcrown_models"
        if not onnx_dir.exists():
            print(f"\nERROR: ONNX directory not found: {onnx_dir}")
            print("Please export your router model to ONNX first.")
            sys.exit(1)

        onnx_files = list(onnx_dir.glob('*router_only.onnx'))
        if not onnx_files:
            print(f"\nERROR: No router ONNX file found in {onnx_dir}")
            print("\nPlease export your router model first:")
            print("  python src/Formal_Neural_Network_Verification/alpha-beta-crown/export_router_to_abcrown.py \\")
            print("    --model_path <your_meta_moe_model.pth>")
            sys.exit(1)

        # Use the most recently modified ONNX file
        onnx_model = max(onnx_files, key=lambda p: p.stat().st_mtime).resolve()
        print(f"\nAuto-detected ONNX model: {onnx_model.name}")
        print(f"  (most recent in artifacts/abcrown_models/)")

    print("\n" + "="*80)
    print("MetaMoE Router Formal Verification - All Samples")
    print("="*80)
    print(f"\nONNX Model: {onnx_model}")
    print(f"VNNLIB specs: {vnnlib_dir}")
    print(f"MNIST samples: {args.num_mnist}")
    print(f"CIFAR10 samples: {args.num_cifar}")
    print(f"Epsilon: {args.epsilon:.5f} ({args.epsilon*255:.1f}/255)")
    print(f"Timeout per sample: {args.timeout}s")

    # Verify ONNX model exists
    if not onnx_model.exists():
        print(f"\nERROR: ONNX model not found: {onnx_model}")
        print("Please ensure the router model has been exported to ONNX format.")
        sys.exit(1)

    # Generate fresh VNNLIB specifications
    generate_vnnlib_specs(args.num_mnist, args.num_cifar, args.epsilon, vnnlib_dir, args.no_random_sampling)

    # Get all VNNLIB files
    mnist_files = sorted(vnnlib_dir.glob("mnist_*.vnnlib"))
    cifar_files = sorted(vnnlib_dir.glob("cifar10_*.vnnlib"))
    all_files = mnist_files + cifar_files

    print("\n" + "="*80)
    print("Step 2: Running Verification on All Samples")
    print("="*80)
    print(f"\nTotal samples: {len(all_files)}")
    print(f"  MNIST: {len(mnist_files)}")
    print(f"  CIFAR10: {len(cifar_files)}")
    print(f"\nDevice: CUDA (GPU acceleration enabled)")
    print(f"Batch size: 2048")

    # Set up Python path for auto_LiRPA
    auto_lirpa_path = (abcrown_dir.parent / "auto_LiRPA").resolve()
    env = os.environ.copy()

    # Convert to string and use forward slashes (works on both Windows and Unix)
    auto_lirpa_str = str(auto_lirpa_path).replace('\\', '/')

    # Build PYTHONPATH
    pythonpath_parts = [auto_lirpa_str]
    if 'PYTHONPATH' in env:
        pythonpath_parts.append(env['PYTHONPATH'])
    env['PYTHONPATH'] = os.pathsep.join(pythonpath_parts)

    print(f"PYTHONPATH set to: {env['PYTHONPATH']}")

    # Change to alpha-beta-CROWN directory
    original_dir = os.getcwd()
    os.chdir(abcrown_dir)

    # Results tracking
    results = {
        'verified': [],
        'falsified': [],
        'timeout': [],
        'unknown': []
    }

    start_time = time.time()

    # Verify each sample
    for idx, vnnlib_file in enumerate(all_files, 1):
        dataset = "MNIST" if "mnist" in vnnlib_file.name else "CIFAR10"
        sample_id = vnnlib_file.stem

        print("\n" + "="*80)
        print(f"[{idx}/{len(all_files)}] Verifying {dataset} sample: {vnnlib_file.name}")
        print("="*80)

        # Convert ONNX path to relative path from alpha-beta-CROWN/complete_verifier
        onnx_relative = os.path.relpath(onnx_model, abcrown_dir)

        config_content = f"""model:
  onnx_path: {onnx_relative}
  input_shape: [1, 3, 32, 32]

specification:
  vnnlib_path: ../../../artifacts/vnnlib_specs/router/{vnnlib_file.name}

general:
  device: cuda
  conv_mode: patches

solver:
  batch_size: 2048
  alpha-crown:
    iteration: 100
    lr_alpha: 0.1
  beta-crown:
    iteration: 20
    lr_alpha: 0.01
    lr_beta: 0.05

attack:
  pgd_order: skip

bab:
  timeout: {args.timeout}
  max_domains: 100000
  branching:
    method: kfsb
    candidates: 3
  attack:
    enabled: false
"""

        config_file = abcrown_dir / "temp_router_verify_config.yaml"
        with open(config_file, 'w') as f:
            f.write(config_content)

        # Run verification
        cmd = [
            sys.executable,
            "abcrown.py",
            "--config", str(config_file),
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=args.timeout + 60)

            # Check for errors
            if result.returncode != 0:
                results['unknown'].append(sample_id)
                print(f"\n[?] ERROR: alpha-beta-CROWN failed with return code {result.returncode}")
                print(f"STDERR:\n{result.stderr}")  # Print full error for debugging
                print(f"\nSTDOUT:\n{result.stdout[:500]}")  # Print some stdout too
                continue

            # Parse result
            if "unsat" in result.stdout.lower() or "verified with init" in result.stdout.lower():
                results['verified'].append(sample_id)
                status = "VERIFIED"
                symbol = "[+]"
            elif "sat" in result.stdout.lower():
                results['falsified'].append(sample_id)
                status = "FALSIFIED"
                symbol = "[-]"
            elif "timeout" in result.stdout.lower():
                results['timeout'].append(sample_id)
                status = "TIMEOUT"
                symbol = "[?]"
            else:
                results['unknown'].append(sample_id)
                status = "UNKNOWN"
                symbol = "[?]"
                # Debug: print output if unknown
                if idx == 1:  # Print first sample's output for debugging
                    print(f"\n[DEBUG] First sample output:")
                    print(result.stdout[:1000])

            # Extract time if available
            time_str = "N/A"
            for line in result.stdout.split('\n'):
                if "Time:" in line:
                    try:
                        time_val = float(line.split("Time:")[1].strip())
                        time_str = f"{time_val:.2f}s"
                    except:
                        pass

            print(f"\n{symbol} {status} ({time_str})")

        except subprocess.TimeoutExpired:
            results['timeout'].append(sample_id)
            print(f"\n[?] TIMEOUT (>{args.timeout + 60}s)")
        except Exception as e:
            results['unknown'].append(sample_id)
            print(f"\n[?] ERROR: {e}")

        # Clean up temp config
        if config_file.exists():
            config_file.unlink()

    total_time = time.time() - start_time

    # Change back to original directory
    os.chdir(original_dir)

    # Print summary report
    print("\n" + "="*80)
    print("VERIFICATION SUMMARY")
    print("="*80)
    print(f"\nTotal samples: {len(all_files)}")
    print(f"  Verified:   {len(results['verified'])} ({len(results['verified'])/len(all_files)*100:.1f}%)")
    print(f"  Falsified:  {len(results['falsified'])} ({len(results['falsified'])/len(all_files)*100:.1f}%)")
    print(f"  Timeout:    {len(results['timeout'])} ({len(results['timeout'])/len(all_files)*100:.1f}%)")
    print(f"  Unknown:    {len(results['unknown'])} ({len(results['unknown'])/len(all_files)*100:.1f}%)")
    print(f"\nTotal time: {total_time:.2f} seconds")
    print(f"Average time per sample: {total_time/len(all_files):.2f} seconds")

    if results['verified']:
        print(f"\nVerified samples:")
        for sample_id in results['verified']:
            print(f"  - {sample_id}")

    if results['falsified']:
        print(f"\nFalsified samples (adversarial examples found):")
        for sample_id in results['falsified']:
            print(f"  - {sample_id}")

    if results['timeout']:
        print(f"\nTimeout samples:")
        for sample_id in results['timeout']:
            print(f"  - {sample_id}")

    # Save results to file
    results_file = Path("artifacts/router_verification_results.txt")
    with open(results_file, 'w') as f:
        f.write("MetaMoE Router Formal Verification Results\n")
        f.write("="*80 + "\n\n")
        f.write(f"Model: {onnx_model.name}\n")
        f.write(f"Epsilon (L-inf): {args.epsilon:.6f} ({args.epsilon*255:.1f}/255)\n")
        f.write(f"Total samples: {len(all_files)}\n")
        f.write(f"  MNIST: {len(mnist_files)}\n")
        f.write(f"  CIFAR10: {len(cifar_files)}\n\n")
        f.write(f"Verified:   {len(results['verified'])} ({len(results['verified'])/len(all_files)*100:.1f}%)\n")
        f.write(f"Falsified:  {len(results['falsified'])} ({len(results['falsified'])/len(all_files)*100:.1f}%)\n")
        f.write(f"Timeout:    {len(results['timeout'])} ({len(results['timeout'])/len(all_files)*100:.1f}%)\n")
        f.write(f"Unknown:    {len(results['unknown'])} ({len(results['unknown'])/len(all_files)*100:.1f}%)\n\n")
        f.write(f"Total time: {total_time:.2f} seconds\n")
        f.write(f"Average time: {total_time/len(all_files):.2f} seconds per sample\n\n")

        for category, samples in results.items():
            if samples:
                f.write(f"\n{category.upper()}:\n")
                for sample_id in samples:
                    f.write(f"  {sample_id}\n")

    print(f"\nResults saved to: {results_file}")
    print("="*80)


if __name__ == '__main__':
    main()
