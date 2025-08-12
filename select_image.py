import os
import random
import shutil

# --- Configuration ---
SOURCE_DIR = "imagenet"
DEST_DIR = "imagenet_selected"
NUM_FILES = 1000
# Add or remove file extensions as needed (must be lowercase)
EXTENSIONS = ['.jpg', '.jpeg', '.png']
# --- End Configuration ---

def collect_images(source_folder):
    """Recursively finds all images with allowed extensions."""
    image_paths = []
    print(f"Searching for images in '{source_folder}'...")
    for root, _, files in os.walk(source_folder):
        for file in files:
            if os.path.splitext(file)[1].lower() in EXTENSIONS:
                image_paths.append(os.path.join(root, file))
    return image_paths

def copy_random_images(image_list, dest_folder, num_to_copy):
    """Selects and copies a random sample of images."""
    if len(image_list) < num_to_copy:
        print(f"⚠️ Warning: Found only {len(image_list)} images, which is less than the {num_to_copy} requested.")
        num_to_copy = len(image_list)

    # Randomly sample the list of images without replacement
    random_sample = random.sample(image_list, num_to_copy)

    # Create the destination folder if it doesn't exist
    os.makedirs(dest_folder, exist_ok=True)
    print(f"Copying {num_to_copy} random images to '{dest_folder}'...")

    for file_path in random_sample:
        shutil.copy(file_path, dest_folder)

    print(f"✅ Done! Copied {num_to_copy} images.")

if __name__ == "__main__":
    all_images = collect_images(SOURCE_DIR)
    if all_images:
        copy_random_images(all_images, DEST_DIR, NUM_FILES)
    else:
        print("No images found with the specified extensions.")