import os
import requests
from urllib.parse import urlparse

# --- Configuration ---
PROMPTS_FILE = 'download/others/prompts.txt'
SOURCE_FILE = 'download/others/Validation_GCC-1.1.0-Validation.tsv'
OUTPUT_FOLDER = 'downloaded_images'

def download_images():
    """
    Reads prompts, finds corresponding image URLs, and downloads them.
    """
    # 1. Create the output directory if it doesn't exist
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    print(f"Output folder is '{OUTPUT_FOLDER}'")

    # 2. Read the source file and create a lookup map (prompt -> URL)
    source_map = {}
    try:
        with open(SOURCE_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                if '\t' in line:
                    parts = line.strip().split('\t', 1)
                    if len(parts) == 2:
                        prompt, url = parts
                        source_map[prompt.strip()] = url.strip()
    except FileNotFoundError:
        print(f"Error: Source file not found at '{SOURCE_FILE}'. Please create it.")
        return

    # 3. Process the prompts file
    try:
        with open(PROMPTS_FILE, 'r', encoding='utf-8') as f:
            prompts = f.read().strip().splitlines()
        print(f"Found {len(prompts)} prompts to process in '{PROMPTS_FILE}'.")
    except FileNotFoundError:
        print(f"Error: Prompts file not found at '{PROMPTS_FILE}'. Please create it.")
        return

    for new_index, line in enumerate(prompts, 1):
        line = line.strip()
        if not line or ':' not in line:
            continue

        # Extract index and prompt text
        try:
            extracted_index, prompt_text = line.split(':', 1)
            extracted_index = extracted_index.strip()
            prompt_text = prompt_text.strip()
        except ValueError:
            print(f"Warning: Skipping malformed line {new_index}: {line}")
            continue

        # Find the prompt in our source map
        if prompt_text in source_map:
            url = source_map[prompt_text]
            print(f"\nProcessing ({new_index}/{len(prompts)}): Found '{prompt_text}'")

            # Download the image
            try:
                response = requests.get(url, timeout=15)
                # Raise an exception for bad status codes (4xx or 5xx)
                response.raise_for_status()

                # Determine file extension from URL path
                path = urlparse(url).path
                ext = os.path.splitext(path)[1]
                
                # Basic check for a valid-looking extension
                if not ext or len(ext) > 5:
                    ext = '.jpg' # Default fallback

                # Create the new filename and save the file
                filename = f"{new_index}-{extracted_index}{ext}"
                filepath = os.path.join(OUTPUT_FOLDER, filename)
                with open(filepath, 'wb') as img_file:
                    img_file.write(response.content)
                print(f"  -> Successfully saved as '{filepath}'")

            except requests.exceptions.RequestException as e:
                print(f"  -> Error downloading {url}: {e}")

        else:
            print(f"\nWarning: Prompt on line {new_index} not found in source file: '{prompt_text}'")

    print("\n✅ Processing complete.")

if __name__ == "__main__":
    download_images()