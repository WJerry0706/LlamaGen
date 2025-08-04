import os
import shutil
import random

# --- Configuration ---
# Path to the parent folder containing all the class subfolders
source_parent_folder = 'imagenet'

# Path to the new folder where the random samples will be stored
destination_parent_folder = 'imagenet_subset'

# Number of files to randomly pick from each subfolder
num_files_to_pick = 8

# --- Main Script ---
print(f"Starting to create ImageNet subset...")
print(f"Source: {source_parent_folder}")
print(f"Destination: {destination_parent_folder}")
print(f"Picking {num_files_to_pick} files from each subfolder.")

# Get a list of all subfolders (classes) in the source directory
class_folders = [
    d for d in os.listdir(source_parent_folder) 
    if os.path.isdir(os.path.join(source_parent_folder, d))
]

# Process each class folder
for class_folder_name in class_folders:
    source_class_path = os.path.join(source_parent_folder, class_folder_name)
    destination_class_path = os.path.join(destination_parent_folder, class_folder_name)
    
    # Get a list of all files in the current class folder
    files_in_folder = [
        f for f in os.listdir(source_class_path)
        if os.path.isfile(os.path.join(source_class_path, f))
    ]
    
    # Check if there are enough files to pick from
    if len(files_in_folder) < num_files_to_pick:
        print(f"Warning: Not enough files in '{class_folder_name}'. Found {len(files_in_folder)}, but need {num_files_to_pick}. Skipping...")
        continue
    
    # Randomly select the files
    selected_files = random.sample(files_in_folder, num_files_to_pick)
    
    # Create the destination subfolder
    os.makedirs(destination_class_path, exist_ok=True)
    
    # Copy the selected files
    for file_to_copy in selected_files:
        source_file_path = os.path.join(source_class_path, file_to_copy)
        destination_file_path = os.path.join(destination_class_path, file_to_copy)
        
        try:
            shutil.copy2(source_file_path, destination_file_path)
            # print(f"Copied: {file_to_copy}") # Uncomment for verbose output
        except Exception as e:
            print(f"Error copying {file_to_copy}: {e}")
            
    print(f"Processed folder: '{class_folder_name}'. Copied {num_files_to_pick} files.")

print("\nTask complete. The ImageNet subset has been created.")