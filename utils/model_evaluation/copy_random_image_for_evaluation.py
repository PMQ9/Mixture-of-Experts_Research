import os
import shutil
import random
import argparse
from pathlib import Path

def copy_random_images(total_files, gtsrb_dir, ptsd_dir, output_dir):
    gtsrb_path = Path(gtsrb_dir)
    ptsd_path = Path(ptsd_dir)
    output_path = Path(output_dir)
    if not gtsrb_path.exists():
        raise FileNotFoundError(f"GTSRB directory not found: {gtsrb_dir}")
    if not ptsd_path.exists():
        raise FileNotFoundError(f"PTSD directory not found: {ptsd_dir}")

    output_path.mkdir(parents=True, exist_ok=True)

    gtsrb_files = [f for f in gtsrb_path.iterdir() if f.suffix.lower() == '.ppm']
    ptsd_files = [f for f in ptsd_path.iterdir() if f.suffix.lower() in ('.jpg', '.jpeg')]

    if not gtsrb_files:
        raise FileNotFoundError(f"No .ppm files found in {gtsrb_dir}")
    if not ptsd_files:
        raise FileNotFoundError(f"No .jpg/.jpeg files found in {ptsd_dir}")

    files_per_dataset = total_files // 2
    gtsrb_count = files_per_dataset
    ptsd_count = total_files - gtsrb_count

    if gtsrb_count > len(gtsrb_files):
        raise ValueError(f"Requested {gtsrb_count} GTSRB files, but only {len(gtsrb_files)} available")
    if ptsd_count > len(ptsd_files):
        raise ValueError(f"Requested {ptsd_count} PTSD files, but only {len(ptsd_files)} available")

    selected_gtsrb = random.sample(gtsrb_files, gtsrb_count)
    selected_ptsd = random.sample(ptsd_files, ptsd_count)

    copied_files = []
    for file in selected_gtsrb + selected_ptsd:
        try:
            dest = output_path / file.name
            shutil.copy(file, dest)
            copied_files.append(file.name)
        except Exception as e:
            print(f"Error copying {file.name}: {e}")

    print(f"Copied {len(copied_files)} files to {output_dir}:")
    print(f"- {len(selected_gtsrb)} GTSRB files (.ppm)")
    print(f"- {len(selected_ptsd)} PTSD files (.jpg/.jpeg)")
    return copied_files

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Copy random images from GTSRB and PTSD datasets")
    parser.add_argument('--total_images', type=int, required=True, help="Total number of files to copy (split 50/50)")
    args = parser.parse_args()

    gtsrb_dir = "./data/GTSRB/Test/Images"
    ptsd_dir = "./data/PTSD/Test/Images"
    output_dir = "./data/Evaluation"

    try:
        copied_files = copy_random_images(args.total_images, gtsrb_dir, ptsd_dir, output_dir)
    except Exception as e:
        print(f"Error: {e}")
        exit(1)