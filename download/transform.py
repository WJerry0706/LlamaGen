import os
from pathlib import Path
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

# --- Configuration ---

# 1. Set the folder containing your images (WEBP, PNG, etc.)
input_folder = "conceptual_images"

# 2. Set the folder where the converted JPG images will be saved
output_folder = "conceptual_images"

# 3. Set the background color for images with transparency (e.g., PNGs)
#    You can use color names like "white", "black", or RGB tuples like (255, 255, 255).
background_color = "white"

# 4. Set the quality for the output JPG files (1-100)
#    95 is a good balance between quality and file size.
jpeg_quality = 95

# --- End of Configuration ---


def convert_images_to_jpg():
    """
    Finds all non-JPG images in the input folder, converts them to JPG,
    and saves them in the output folder.
    """
    input_path = Path(input_folder)
    output_path = Path(output_folder)

    # Check if the input directory exists
    if not input_path.is_dir():
        print(f"❌ Error: Input folder not found at '{input_path}'")
        return

    # Create the output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"✅ Output folder is '{output_path}'")

    # Get a list of all files in the input directory
    files_to_process = [f for f in input_path.iterdir() if f.is_file()]
    
    if not files_to_process:
        print("No files found in the input folder.")
        return

    converted_count = 0
    skipped_count = 0

    # Process files with a progress bar
    for file_path in tqdm(files_to_process, desc="Converting images"):
        # Skip files that are already JPG
        if file_path.suffix.lower() in ['.jpg', '.jpeg']:
            skipped_count += 1
            continue

        try:
            # Open the image file
            with Image.open(file_path) as img:
                # Create a new filename with a .jpg extension
                output_filename = f"{file_path.stem}.jpg"
                output_file_path = output_path / output_filename
                
                # Convert the image to RGB mode before saving as JPG
                # This handles transparency by blending it with the background color
                if img.mode in ('RGBA', 'LA', 'P'):
                    # Create a new image with the specified background color
                    background = Image.new('RGB', img.size, background_color)
                    # Paste the original image onto the background, using its alpha channel as a mask
                    background.paste(img, (0, 0), img.getchannel('A') if 'A' in img.getbands() else None)
                    img_to_save = background
                else:
                    # If no transparency, just ensure it's in RGB mode
                    img_to_save = img.convert('RGB')

                # Save the image as a JPG file
                img_to_save.save(output_file_path, 'jpeg', quality=jpeg_quality)
                converted_count += 1

        except (UnidentifiedImageError, IOError):
            # This handles cases where a file is not a valid image format
            print(f"\n⚠️ Skipping non-image file: {file_path.name}")
            skipped_count += 1
        except Exception as e:
            print(f"\n❌ An error occurred with file {file_path.name}: {e}")
            skipped_count += 1

    print("\n--- Conversion Complete ---")
    print(f"✅ Successfully converted: {converted_count} files")
    print(f"⏩ Skipped (already JPG or not an image): {skipped_count} files")


if __name__ == "__main__":
    convert_images_to_jpg()