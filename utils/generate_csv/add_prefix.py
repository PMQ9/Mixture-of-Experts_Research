import pandas as pd

# Define file paths
input_csv = "annotations_new.csv"  # Path to your input CSV file
output_csv = "annotations_new_updated.csv"  # Path to save the updated CSV

# Read the CSV file with semicolon delimiter
df = pd.read_csv(input_csv, sep=';')

# Add 'Images/' to the beginning of each filename
df['Filename'] = 'Images/' + df['Filename']

# Save the updated DataFrame to a new CSV
df.to_csv(output_csv, sep=';', index=False)

print(f"Updated CSV with 'Images/' prepended to filenames saved to {output_csv}")