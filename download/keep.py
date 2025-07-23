import os

# --- Configuration ---
# Set the name of the file you want to modify
FILENAME = "coco_captions.txt" 
LINES_TO_KEEP = 1000
# ---------------------

try:
    # Step 1: Read the first 1000 lines into memory
    print(f"Reading the first {LINES_TO_KEEP} lines from '{FILENAME}'...")
    with open(FILENAME, 'r', encoding='utf-8') as f:
        lines = [f.readline() for _ in range(LINES_TO_KEEP)]

    # Step 2: Re-open the file in write mode (which erases it) and write the lines back
    print(f"Truncating the file and writing back the content...")
    with open(FILENAME, 'w', encoding='utf-8') as f:
        f.writelines(lines)

    print(f"\nSuccessfully truncated '{FILENAME}' to its first {LINES_TO_KEEP} lines.")

except FileNotFoundError:
    print(f"Error: The file '{FILENAME}' was not found.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")