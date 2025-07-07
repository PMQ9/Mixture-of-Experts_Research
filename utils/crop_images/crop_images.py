import os
import pandas as pd
from PIL import Image
import shutil
import uuid

# Define paths
csv_file = "annotations.csv"  # Path to your CSV file
image_dir = "images"         # Directory containing original images
backup_dir = "images_backup" # Directory for backup images
cropped_dir = "cropped_images" # Directory for cropped images
output_csv = "cropped_annotations.csv" # Output CSV with updated annotations

# Create directories if they don't exist
os.makedirs(backup_dir, exist_ok=True)
os.makedirs(cropped_dir, exist_ok=True)

# Read the CSV file
df = pd.read_csv(csv_file)

# Function to crop image and return new coordinates
def crop_image(image_path, xmin, ymin, xmax, ymax, output_path):
    with Image.open(image_path) as img:
        # Crop the image using the bounding box
        cropped_img = img.crop((xmin, ymin, xmax, ymax))
        # Save the cropped image
        cropped_img.save(output_path)
        # New dimensions
        new_width = xmax - xmin
        new_height = ymax - ymin
        # New coordinates (relative to cropped image)
        new_xmin = 0
        new_ymin = 0
        new_xmax = new_width
        new_ymax = new_height
        return new_width, new_height, new_xmin, new_ymin, new_xmax, new_ymax

# Process each row in the CSV
new_rows = []
for index, row in df.iterrows():
    filename = row['filename']
    image_path = os.path.join(image_dir, filename)
    
    # Verify image exists
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        continue
    
    # Create backup of original image
    backup_path = os.path.join(backup_dir, filename)
    shutil.copy(image_path, backup_path)
    
    # Generate unique filename for cropped image
    file_ext = os.path.splitext(filename)[1]
    new_filename = f"{uuid.uuid4().hex}{file_ext}"
    cropped_path = os.path.join(cropped_dir, new_filename)
    
    # Crop the image
    new_width, new_height, new_xmin, new_ymin, new_xmax, new_ymax = crop_image(
        image_path,
        row['xmin'],
        row['ymin'],
        row['xmax'],
        row['ymax'],
        cropped_path
    )
    
    # Create new row with updated information
    new_row = row.copy()
    new_row['filename'] = new_filename
    new_row['width'] = new_width
    new_row['height'] = new_height
    new_row['xmin'] = new_xmin
    new_row['ymin'] = new_ymin
    new_row['xmax'] = new_xmax
    new_row['ymax'] = new_ymax
    new_rows.append(new_row)

# Create new DataFrame with updated annotations
new_df = pd.DataFrame(new_rows)

# Save the updated CSV
new_df.to_csv(output_csv, index=False)
print(f"Cropped images saved to {cropped_dir}")
print(f"Backup images saved to {backup_dir}")
print(f"Updated annotations saved to {output_csv}")