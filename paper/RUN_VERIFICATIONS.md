# Running All Verifications

## Quick Start

Run all verifications and automatically update `VERIFICATION_RESULTS.md`:

```bash
cd paper
python run_all_verifications.py
```

## Usage

### Run All Verifications (Default)
```bash
python run_all_verifications.py
```
- Runs all 12 expert verifications (4 models × 3 epsilons)
- Runs all 6 router verifications (2 models × 3 epsilons)
- Automatically updates VERIFICATION_RESULTS.md with results

### Skip Experts (Router Only)
```bash
python run_all_verifications.py --skip-experts
```
Runs only router verification (2 models × 3 epsilons = 6 tests)

### Skip Routers (Experts Only)
```bash
python run_all_verifications.py --skip-routers
```
Runs only expert verification (4 models × 3 epsilons = 12 tests)

### Run Specific Models
```bash
python run_all_verifications.py --models E_0_CNN_NAT,MoE_CNN_NAT
```
Runs only E_0_CNN_NAT and MoE_CNN_NAT models with all epsilons

### Dry Run (Preview Commands)
```bash
python run_all_verifications.py --dry-run
```
Prints all commands without executing (useful for testing)

## Models

### Experts
- **E_0_CNN_NAT**: CIFAR-10 expert (non-robust training)
- **E_0_CNN_AT**: CIFAR-10 expert (robust training)
- **E_1_CNN_NAT**: MNIST expert (non-robust training)
- **E_1_CNN_AT**: MNIST expert (robust training)

### Routers
- **MoE_CNN_NAT**: MetaMoE router (non-robust training)
- **MoE_CNN_AT**: MetaMoE router (robust training)

## Epsilons Tested

- **2/255** (0.00784) - Standard robustness
- **4/255** (0.01569) - Moderate robustness
- **8/255** (0.03137) - Aggressive robustness

## What Gets Updated

The script automatically updates `VERIFICATION_RESULTS.md` with:

1. **Verified Count**: Number of samples with proven robustness
2. **Falsified Count**: Number of counterexamples found
3. **Timeout Count**: Number of samples that timed out
4. **Unknown Count**: Number of unresolved samples
5. **Average Time**: Average verification time per sample
6. **Last Run**: Timestamp of verification completion

## Example Output

```
================================================================================
VERIFICATION BATCH RUN
================================================================================
Start time: 2025-11-06 16:30:00
Experts to verify: 4
Routers to verify: 2
Dry run: False

================================================================================
[1/18] Expert test: E_0_CNN_NAT @ ε=2/255
================================================================================
Running expert verification...

[RESULTS]
Verified: 18
Falsified: 0
Timeout: 2
Unknown: 0
Avg Time: 32.3s

✓ Results saved to VERIFICATION_RESULTS.md
```

## Important Notes

1. **Runtime**: Total time depends on model complexity
   - Each expert: ~10-30 minutes (20 samples)
   - Each router: ~15-40 minutes (50 MNIST + 50 CIFAR samples)
   - Full batch: ~8-10 hours

2. **GPU Required**: These verifications require CUDA GPU for reasonable time

3. **File Paths**: Script automatically finds models from:
   - `paper/artifacts/E_0_CNN_NAT/`
   - `paper/artifacts/E_0_CNN_AT/`
   - `paper/artifacts/E_1_CNN_NAT/`
   - `paper/artifacts/E_1_CNN_AT/`
   - `paper/artifacts/MoE_CNN_NAT/`
   - `paper/artifacts/MoE_CNN_AT/`

4. **Results Sync**: Results are written to VERIFICATION_RESULTS.md automatically after each test completes

## Troubleshooting

### Models Not Found
Ensure your artifacts are in the correct path:
```
paper/artifacts/
├── E_0_CNN_NAT/
├── E_0_CNN_AT/
├── E_1_CNN_NAT/
├── E_1_CNN_AT/
├── MoE_CNN_NAT/
└── MoE_CNN_AT/
```
