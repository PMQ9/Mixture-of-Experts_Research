import pandas as pd

# Define file paths
input_csv = "annotations_new.csv"  # Path to your input CSV file
output_txt = "class_mapping.txt"   # Path for the class-to-integer mapping
output_csv = "annotations_new.csv" # Path to overwrite the input CSV with updated ClassId

# Read the CSV file with semicolon delimiter
df = pd.read_csv(input_csv, sep=';')

# Get unique class names and sort them alphabetically
unique_classes = sorted(df['ClassId'].unique())

# Create mapping of class names to integers (starting from 0)
class_mapping = {class_name: idx for idx, class_name in enumerate(unique_classes)}

# Save the class mapping to a text file
with open(output_txt, 'w') as f:
    for class_name, class_id in class_mapping.items():
        f.write(f"{class_name}: {class_id}\n")

# Replace ClassId values in the DataFrame with integer IDs
df['ClassId'] = df['ClassId'].map(class_mapping)

# Save the updated DataFrame back to the CSV
df.to_csv(output_csv, sep=';', index=False)

print(f"Class mapping saved to {output_txt}")
print(f"Updated CSV saved to {output_csv}")