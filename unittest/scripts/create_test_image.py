"""
Create a sample test image for testing purposes.
"""

import argparse
from pathlib import Path

from PIL import Image
import numpy as np


def create_random_image(output_path, size=224):
    """Create a random RGB image."""
    img_array = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    img.save(output_path)
    print(f"Created random image: {output_path}")


def create_gradient_image(output_path, size=224):
    """Create a gradient image for more interesting test data."""
    x = np.linspace(0, 255, size, dtype=np.uint8)
    y = np.linspace(0, 255, size, dtype=np.uint8)
    xx, yy = np.meshgrid(x, y)

    r = xx
    g = yy
    b = (xx + yy) // 2

    img_array = np.stack([r, g, b], axis=2)
    img = Image.fromarray(img_array)
    img.save(output_path)
    print(f"Created gradient image: {output_path}")


def create_pattern_image(output_path, size=224):
    """Create a checkerboard pattern image."""
    img_array = np.zeros((size, size, 3), dtype=np.uint8)

    block_size = size // 8
    for i in range(8):
        for j in range(8):
            if (i + j) % 2 == 0:
                img_array[i*block_size:(i+1)*block_size,
                         j*block_size:(j+1)*block_size] = [255, 255, 255]
            else:
                img_array[i*block_size:(i+1)*block_size,
                         j*block_size:(j+1)*block_size] = [0, 0, 0]

    img = Image.fromarray(img_array)
    img.save(output_path)
    print(f"Created pattern image: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Create test images')
    parser.add_argument('--output_dir', type=str, default='testdata/images',
                       help='Output directory for images')
    parser.add_argument('--size', type=int, default=224,
                       help='Image size (width and height)')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create different types of test images
    create_random_image(output_dir / 'test_sample.png', args.size)
    create_gradient_image(output_dir / 'test_gradient.png', args.size)
    create_pattern_image(output_dir / 'test_pattern.png', args.size)

    print(f"\nTest images created successfully in {output_dir}")
    print("You can now run the test suite!")


if __name__ == '__main__':
    main()
