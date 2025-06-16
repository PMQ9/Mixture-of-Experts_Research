import os

def remove_ptsd_prefix(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg') and filename.startswith('PTSD_'):
            new_filename = filename.replace('PTSD_', '', 1)
            
            old_path = os.path.join(folder_path, filename)
            new_path = os.path.join(folder_path, new_filename)
            
            os.rename(old_path, new_path)
            print(f'Renamed: {filename} -> {new_filename}')

if __name__ == '__main__':
    folder_path = './data/PTSD/Test/Images/'  # Change this to your folder path
    remove_ptsd_prefix(folder_path)
    print("All files renamed successfully!")