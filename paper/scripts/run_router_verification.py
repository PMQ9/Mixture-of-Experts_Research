"""
Run alpha-beta-CROWN Verification on MetaMoE Router

This script runs formal verification to prove that the router maintains
correct expert selection under adversarial perturbations.

NOTE: This is a legacy script. Use run_router_formal_verification.py at the repository root instead.
"""

import subprocess
import sys
from pathlib import Path
import os

# Repository root (navigate from paper/scripts/ back to root)
repo_root = Path(__file__).parent.parent.parent

# Configuration
abcrown_dir = repo_root / "modules/alpha-beta-CROWN/complete_verifier"
onnx_model = (repo_root / "artifacts/abcrown_models/meta_moe_ultra_verifiable_cnn_best_og_router_only.onnx").resolve()
config_file = (repo_root / "artifacts/router_verification_config.yaml").resolve()
vnnlib_dir = (repo_root / "artifacts/vnnlib_specs/router").resolve()

print("="*80)
print("MetaMoE Router Formal Verification with alpha-beta-CROWN")
print("="*80)
print(f"\nONNX Model: {onnx_model}")
print(f"Config: {config_file}")
print(f"VNNLIB specs: {vnnlib_dir}")
print(f"Number of specs: {len(list(vnnlib_dir.glob('*.vnnlib')))}")

# Verify files exist
if not onnx_model.exists():
    print(f"\nERROR: ONNX model not found: {onnx_model}")
    sys.exit(1)

if not config_file.exists():
    print(f"\nERROR: Config file not found: {config_file}")
    sys.exit(1)

if not vnnlib_dir.exists():
    print(f"\nERROR: VNNLIB directory not found: {vnnlib_dir}")
    sys.exit(1)

# Set up Python path for auto_LiRPA
auto_lirpa_path = (abcrown_dir.parent / "auto_LiRPA").resolve()
env = os.environ.copy()
if 'PYTHONPATH' in env:
    env['PYTHONPATH'] = f"{auto_lirpa_path}{os.pathsep}{env['PYTHONPATH']}"
else:
    env['PYTHONPATH'] = str(auto_lirpa_path)

# Change to alpha-beta-CROWN directory
os.chdir(abcrown_dir)
print(f"\nWorking directory: {os.getcwd()}")
print(f"PYTHONPATH includes: {auto_lirpa_path}")

# Test on a single MNIST sample first
print("\n" + "="*80)
print("Running verification on MNIST sample (should route to expert 1)...")
print("="*80)

# alpha-beta-CROWN uses a YAML config file that specifies the model and vnnlib paths
# We'll verify one sample first to test the setup
cmd = [
    sys.executable,
    "abcrown.py",
    "--config", str(config_file),
]

print(f"\nCommand: {' '.join(cmd)}\n")

result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600)

print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr)

print("\n" + "="*80)
if "unsat" in result.stdout.lower():
    print("VERIFIED: Router is provably robust for this MNIST sample!")
    print("  (No adversarial perturbation within epsilon can change routing)")
elif "sat" in result.stdout.lower():
    print("FALSIFIED: Found adversarial perturbation that changes routing")
elif "timeout" in result.stdout.lower():
    print("TIMEOUT: Verification incomplete within time limit")
else:
    print("UNKNOWN: Could not determine verification result")

print("="*80)
print("\nTo verify all 20 samples, modify this script to loop through all VNNLIB files")
print("or use alpha-beta-CROWN's batch verification mode with instances CSV")
