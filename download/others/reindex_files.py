import os
import sys

# --- Configuration ---
IMAGE_DIR = "images"
PROMPT_FILE = "prompts.txt"
CLEANED_PROMPT_FILE = "prompts_cleaned.txt"
# ---------------------

# Verify that the necessary files and folders exist
if not os.path.isdir(IMAGE_DIR):
    print(f"Error: Image directory '{IMAGE_DIR}' not found.")
    sys.exit()

if not os.path.isfile(PROMPT_FILE):
    print(f"Error: Prompt file '{PROMPT_FILE}' not found.")
    sys.exit()

print("Starting the re-indexing process...")

# --- Step 1: Read the existing prompts file ---
try:
    with open(PROMPT_FILE, 'r', encoding='utf-8') as f:
        prompt_lines = f.readlines()
except Exception as e:
    print(f"Error reading {PROMPT_FILE}: {e}")
    sys.exit()

# --- Step 2: Create a mapping from old index to filename ---
# This is efficient for finding files quickly.
try:
    all_image_files = os.listdir(IMAGE_DIR)
    old_idx_to_filename = {os.path.splitext(f)[0]: f for f in all_image_files}
except Exception as e:
    print(f"Error listing files in {IMAGE_DIR}: {e}")
    sys.exit()


# --- Step 3: Loop through prompts, rename images, and create new prompt list ---
new_prompts = []
files_renamed = 0

for new_index, line in enumerate(prompt_lines):
    line = line.strip()
    if not line:
        continue

    try:
        # Split "original_index: prompt_text"
        old_index_str, prompt_text = line.split(':', 1)
        old_index_str = old_index_str.strip()
        prompt_text = prompt_text.strip()

        # Find the corresponding old filename in our map
        if old_index_str in old_idx_to_filename:
            old_filename = old_idx_to_filename[old_index_str]
            old_filepath = os.path.join(IMAGE_DIR, old_filename)
            
            # Get the file extension (e.g., .jpg, .png)
            extension = os.path.splitext(old_filename)[1]
            
            # Create the new filename and path
            new_filename = f"{new_index}{extension}"
            new_filepath = os.path.join(IMAGE_DIR, new_filename)
            
            # Rename the file
            os.rename(old_filepath, new_filepath)
            
            # Add the newly formatted prompt to our list
            new_prompts.append(f"{new_index}: {prompt_text}")
            files_renamed += 1
        else:
            print(f"Warning: Could not find an image file for prompt with original index '{old_index_str}'")

    except ValueError:
        print(f"Warning: Could not parse line: '{line}' - Skipping.")
    except Exception as e:
        print(f"An error occurred while processing line '{line}': {e}")


# --- Step 4: Write the new, cleaned prompts file ---
try:
    with open(CLEANED_PROMPT_FILE, 'w', encoding='utf-8') as f:
        f.write("\n".join(new_prompts))
except Exception as e:
    print(f"Error writing to {CLEANED_PROMPT_FILE}: {e}")


print("\nRe-indexing complete.")
print(f"Successfully renamed {files_renamed} image files.")
print(f"Cleaned prompts saved to '{CLEANED_PROMPT_FILE}'.")