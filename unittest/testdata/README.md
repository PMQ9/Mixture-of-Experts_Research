# Test Data Directory

This directory contains test data for the MoE research test suite.

## Structure

```
testdata/
├── images/           # Test images for inference
├── baselines/        # Baseline outputs for regression tests
├── models/           # Small test model checkpoints (optional)
└── README.md
```

## Setting Up Test Data

### 1. Test Images

Add sample images to `testdata/images/` for testing. You can:

- Copy sample images from your datasets:
  ```bash
  # Example: Copy a GTSRB test image
  cp ../data/GTSRB/Test/Images/00000.ppm testdata/images/gtsrb_sample.ppm

  # Example: Copy a CIFAR-10 test image
  cp ../data/CIFAR10/Test/airplane_0001.png testdata/images/cifar10_sample.png
  ```

- Create synthetic test images (using Python):
  ```python
  from PIL import Image
  import numpy as np

  # Create a simple test image
  img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
  img.save('testdata/images/test_sample.png')
  ```

### 2. Baselines

Baselines are automatically generated when you run regression tests for the first time.

To manually create baselines:

```bash
# Run tests - baselines will be created automatically
go test -v ./internal/regression
```

Baselines are stored as JSON files in `testdata/baselines/` with names like:
- `{model_name}_{image_name}_baseline.json`

### 3. Test Models (Optional)

If you want to include small test models in version control:

```bash
# Copy a trained model to testdata/models/
cp ../artifacts/results/gtsrb_small_cnn_best.pth testdata/models/
```

## Using Custom Test Data

Modify `config.json` in the unittest root to point to your test data:

```json
{
  "test_images": [
    "testdata/images/gtsrb_sample.ppm",
    "testdata/images/cifar10_sample.png",
    "data/GTSRB/Test/Images/00000.ppm"
  ]
}
```

## Important Notes

- Test images should be representative of your actual datasets
- Keep test images small (224x224 or smaller) for faster testing
- Baselines are specific to model versions - regenerate after model changes
- Do not commit large model files to git (add to .gitignore)
