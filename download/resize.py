import os
from PIL import Image
from tqdm import tqdm

# --- Configuration ---
# Set this to the folder you want to process
INPUT_FOLDER = "cc_images"  # <-- Or your generated images folder

# The target size has been changed to 256x256
TARGET_SIZE = (256, 256)
# ---------------------

OUTPUT_FOLDER = INPUT_FOLDER + "_resized"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

print(f"Resizing images from '{INPUT_FOLDER}' to {TARGET_SIZE}...")
print(f"Resized images will be saved in '{OUTPUT_FOLDER}'")

image_files = [f for f in os.listdir(INPUT_FOLDER) if os.path.isfile(os.path.join(INPUT_FOLDER, f))]

for filename in tqdm(image_files, desc="Resizing"):
    input_path = os.path.join(INPUT_FOLDER, filename)
    output_path = os.path.join(OUTPUT_FOLDER, filename)

    try:
        with Image.open(input_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
                
            resized_img = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
            resized_img.save(output_path, 'JPEG', quality=95)

    except Exception as e:
        print(f"\nCould not process file {filename}. Error: {e}")

print("\nResizing complete.")