import pandas as pd
import requests
import os
from tqdm import tqdm

# --- Configuration ---
CSV_FILE_PATH = "Validation_GCC-1.1.0-Validation.tsv"
OUTPUT_IMAGE_DIR = "images"
OUTPUT_PROMPT_FILE = "prompts.txt"
TARGET_DOWNLOAD_COUNT = 1000
# ---------------------

# Create the output directory if it doesn't exist
os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)

# Try to read the TSV file
try:
    df = pd.read_csv(
        CSV_FILE_PATH,
        header=None,
        names=['prompt', 'url'],
        sep='\t',  # Use '\t' for tab-separated files
        on_bad_lines='warn'
    )
except FileNotFoundError:
    print(f"Error: The file '{CSV_FILE_PATH}' was not found.")
    print("Please make sure your file is in the same directory and has the correct name.")
    exit()

all_prompts = []
successful_downloads = 0

print(f"Starting download. Goal: {TARGET_DOWNLOAD_COUNT} images.")

# Loop through the entire dataframe until we have 1000 successful downloads
for index, row in df.iterrows():
    # If we've reached our target, stop the loop
    if successful_downloads >= TARGET_DOWNLOAD_COUNT:
        print(f"Target of {TARGET_DOWNLOAD_COUNT} successful downloads reached.")
        break

    prompt = row['prompt']
    url = row['url']

    # Skip if URL is not a valid string
    if not isinstance(url, str):
        continue

    try:
        # Get the image data from the URL
        response = requests.get(url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()

        # Determine file extension
        content_type = response.headers.get('content-type')
        if content_type and 'image' in content_type:
            extension = '.' + content_type.split('/')[1].split(';')[0]
        else:
            extension = os.path.splitext(url)[1] or '.jpg'
        
        if extension.lower() not in ['.jpg', '.jpeg', '.png', '.gif', '.webp']:
            extension = '.jpg'

        # Save the image, named by its original index from the file
        image_path = os.path.join(OUTPUT_IMAGE_DIR, f"{index}{extension}")
        with open(image_path, 'wb') as f:
            f.write(response.content)

        # Store the prompt and increment our success counter
        all_prompts.append(f"{index}: {prompt}")
        successful_downloads += 1
        
        # Print progress
        print(f"Success ({successful_downloads}/{TARGET_DOWNLOAD_COUNT}): Downloaded image for index {index}")

    except requests.exceptions.RequestException:
        # Silently skip download errors to keep the output clean, or add a print statement if you want to see them
        # print(f"Skipping index {index} due to download error.")
        pass
    except Exception as e:
        # print(f"An unexpected error occurred for index {index}: {e}")
        pass

# Save all prompts to a single text file
with open(OUTPUT_PROMPT_FILE, 'w', encoding='utf-8') as f:
    f.write("\n".join(all_prompts))

print(f"\nDownload complete.")
print(f"{successful_downloads} images saved in the '{OUTPUT_IMAGE_DIR}/' folder.")
print(f"Prompts for downloaded images saved in '{OUTPUT_PROMPT_FILE}'.")