import os

# Some minor bugs in the PTSD dataset:
# ,Compulsory Keep Right
# NO Waitin

def remove_ptsd_prefix(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg') and filename.startswith('PTSD_'):
            new_filename = filename.replace('PTSD_', '', 1)
            
            old_path = os.path.join(folder_path, filename)
            new_path = os.path.join(folder_path, new_filename)
            
            os.rename(old_path, new_path)
            print(f'Renamed: {filename} -> {new_filename}')

def rename_numbered_folders(root_path):
    for foldername in os.listdir(root_path):
        if '_' in foldername:
            try:
                parts = foldername.split('_', 1)
                int(parts[0])
                new_foldername = parts[1]
                
                old_path = os.path.join(root_path, foldername)
                new_path = os.path.join(root_path, new_foldername)
                
                os.rename(old_path, new_path)
                print(f'Renamed folder: {foldername} -> {new_foldername}')
            except (ValueError, IndexError):
                continue

def remove_leading_zeros(folders_root_gtsrb):
    for foldername in os.listdir(folders_root_gtsrb):
        if foldername.isdigit():
            try:
                num = int(foldername)
                new_name = str(num)
                if new_name != foldername:
                    old_path = os.path.join(folders_root_gtsrb, foldername)
                    new_path = os.path.join(folders_root_gtsrb, new_name)
                    os.rename(old_path, new_path)
                    print(f'Renamed folder: {foldername} -> {new_name}')
            except ValueError:
                continue

if __name__ == '__main__':
    ptsd_folder = './data/PTSD/Test/Images/' 
    if os.path.exists(ptsd_folder):
        remove_ptsd_prefix(ptsd_folder)
    
    folders_root = './data/PTSD/Training'
    if os.path.exists(folders_root):
        rename_numbered_folders(folders_root)

    folders_root_gtsrb = './data/GTSRB/Training'
    if os.path.exists(folders_root_gtsrb):
        remove_leading_zeros(folders_root_gtsrb)
    
    print("All operations completed successfully!")