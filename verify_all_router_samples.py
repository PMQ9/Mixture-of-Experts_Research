"""
Run alpha-beta-CROWN Verification on All MetaMoE Router Samples

This script:
1. Automatically generates fresh VNNLIB specifications with configurable sample counts
2. Verifies all router samples (MNIST + CIFAR10)
3. Generates a comprehensive verification report

Usage:
    # Default: 10 MNIST + 10 CIFAR10 samples
    python verify_all_router_samples.py

    # Custom sample counts
    python verify_all_router_samples.py --num_mnist 20 --num_cifar 20

    # Custom epsilon
    python verify_all_router_samples.py --epsilon 0.01569  # 4/255
"""

import subprocess
import sys
from pathlib import Path
import os
import time
import argparse

# Add project paths
sys.path.insert(0, 'src/Vision_Transformer_Pytorch')
sys.path.insert(0, '.')

# Import the VNNLIB generation function
from prepare_router_verification import main as prepare_vnnlib


def generate_vnnlib_specs(num_mnist, num_cifar, epsilon, vnnlib_dir):
    """
    Generate fresh VNNLIB specifications before verification.

    Args:
        num_mnist: Number of MNIST samples
        num_cifar: Number of CIFAR10 samples
        epsilon: L-infinity perturbation bound
        vnnlib_dir: Output directory for VNNLIB files
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
    parser.add_argument('--epsilon', type=float, default=2.0/255.0,
                        help='L-infinity perturbation bound (default: 2/255)')
    parser.add_argument('--timeout', type=int, default=60,
                        help='Timeout per sample in seconds (default: 60)')

    args = parser.parse_args()

    # Configuration
    abcrown_dir = Path("modules/alpha-beta-CROWN/complete_verifier")
    onnx_model = Path("artifacts/abcrown_models/meta_moe_ultra_verifiable_cnn_best_og_router_only.onnx").resolve()
    vnnlib_dir = Path("artifacts/vnnlib_specs/router").resolve()

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
    generate_vnnlib_specs(args.num_mnist, args.num_cifar, args.epsilon, vnnlib_dir)

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

    # Set up Python path for auto_LiRPA
    auto_lirpa_path = (abcrown_dir.parent / "auto_LiRPA").resolve()
    env = os.environ.copy()
    if 'PYTHONPATH' in env:
        env['PYTHONPATH'] = f"{auto_lirpa_path}{os.pathsep}{env['PYTHONPATH']}"
    else:
        env['PYTHONPATH'] = str(auto_lirpa_path)

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

        # Create config for this sample
        config_content = f"""model:
  onnx_path: ../../../artifacts/abcrown_models/meta_moe_ultra_verifiable_cnn_best_og_router_only.onnx
  input_shape: [1, 3, 32, 32]

specification:
  vnnlib_path: ../../../artifacts/vnnlib_specs/router/{vnnlib_file.name}

solver:
  batch_size: 256
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

        config_file = Path("temp_router_verify_config.yaml").resolve()
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
