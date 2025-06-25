import os
import csv

def generate_csv(training_dir, output_file):
    """
    Generate a CSV file for machine learning training data.
    
    Args:
        training_dir (str): Path to the Training directory
        output_file (str): Path for the output CSV file
    """
    with open(output_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=';')
        # Write header
        csv_writer.writerow(['Filename', 'ClassId', 'meta_class'])
        
        # Process each class directory
        for class_dir in sorted(os.listdir(training_dir)):
            class_path = os.path.join(training_dir, class_dir)
            
            if not os.path.isdir(class_path):
                continue
                
            try:
                class_id = int(class_dir)  # Convert folder name to integer
            except ValueError:
                continue  # Skip non-numeric directories
                
            # Process each PPM file in the class directory
            for file in os.listdir(class_path):
                if file.endswith(('.ppm', '.jpg')):
                    # Format: class_dir/filename.ppm
                    filename = f"{class_dir}/{file}"
                    # Write row: Filename;ClassId;meta_class
                    csv_writer.writerow([filename, class_id, 1])

if __name__ == '__main__':
    # Configure paths
    training_directory = './../../data/PTSD/Training'
    output_csv = 'train_with_meta_class.csv'
    
    # Generate the CSV file
    generate_csv(training_directory, output_csv)
    print(f"CSV file generated successfully: {output_csv}")