import csv
import os
import shutil

# --- Configuration ---

# 1. File paths
data_file_path = 'download/test/test/metadata.csv'
prompts_file_path = 'coco_captions.txt'

# 2. Folder paths
# The folder where all your original images are stored.
# Based on your example: "download/test/test/COCO_val2014_000000000074.jpg"
images_source_folder = 'download/test/test' 

# The name of the folder where matching images will be copied.
output_folder = 'selected_images' 


# --- Script ---

# 1. Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)
print(f"✅ Output folder is '{output_folder}'. Matched images will be copied here.")

# 2. Read prompts from the text file
search_prompts = set()
try:
    with open(prompts_file_path, 'r', encoding='utf-8') as p_file:
        search_prompts = {line.strip() for line in p_file if line.strip()}
    print(f"✅ Successfully loaded {len(search_prompts)} prompts from '{prompts_file_path}'.\n")
except FileNotFoundError:
    print(f"🚨 Error: The prompts file '{prompts_file_path}' was not found. Please create it.")
    exit()

# 3. Search data file, find images, and copy them
found_matches = []
copied_count = 0
not_found_count = 0

try:
    with open(data_file_path, mode='r', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        for row in reader:
            if len(row) > 2:
                caption = row[2]
                if caption in search_prompts:
                    image_filename = row[1]
                    
                    # Create the full path to the source image and destination
                    source_path = os.path.join(images_source_folder, image_filename)
                    destination_path = os.path.join(output_folder, image_filename)

                    # Check if the source image exists before copying
                    if os.path.exists(source_path):
                        shutil.copy2(source_path, destination_path)
                        found_matches.append(f"✅ Copied: {image_filename}")
                        copied_count += 1
                    else:
                        found_matches.append(f"⚠️ Warning: Image not found at {source_path}")
                        not_found_count += 1

except FileNotFoundError:
    print(f"🚨 Error: The data file '{data_file_path}' was not found.")
    exit()

# --- 4. Display Final Results ---

print("--- Search and Copy Complete ---")

if found_matches:
    for result in found_matches:
        print(result)
    print(f"\n✨ Summary: Copied {copied_count} images to the '{output_folder}' folder.")
    if not_found_count > 0:
        print(f"Could not find {not_found_count} source image files.")
else:
    print(f"\n❌ No matches found in '{data_file_path}' for the prompts listed in '{prompts_file_path}'.")