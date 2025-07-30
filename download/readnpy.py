import numpy as np
import os
from collections import defaultdict
import json # Import for JSON output

def analyze_npy_codes(directory_path, start_index, end_index, codebook_size=16384):
    """
    Reads .npy files in a specified range, extracts the first layer of codes,
    and counts the occurrences of each index within a given codebook size.

    Args:
        directory_path (str): The path to the directory containing the .npy files.
                              e.g., 'extracted/imagenet256_codes'
        start_index (int): The starting index of the .npy files (e.g., 0 for '0.npy').
        end_index (int): The ending index of the .npy files (e.g., 34744 for '34744.npy').
        codebook_size (int): The maximum possible index value + 1 (size of the VQ codebook).
                             Defaults to 16384.

    Returns:
        collections.defaultdict: A dictionary where keys are codebook indices (0 to codebook_size-1)
                                 and values are their total counts across all processed files.
    """
    index_counts = defaultdict(int)

    print(f"Starting analysis of .npy files from {start_index}.npy to {end_index}.npy in {directory_path}")
    print(f"Expected codebook size: {codebook_size}")

    for i in range(start_index, end_index + 1):
        file_name = f"{i}.npy"
        file_path = os.path.join(directory_path, file_name)

        try:
            data = np.load(file_path)

            if data.shape != (1, 2, 256):
                 print(f"Warning: Unexpected shape for {file_name}. Expected (1, 2, 256), got {data.shape}. Skipping.")
                 continue

            codes_for_original_image = data[0, 0, :]
            codes_for_original_image = codes_for_original_image.astype(int)

            for code_index in codes_for_original_image:
                if 0 <= code_index < codebook_size:
                    index_counts[code_index] += 1
                else:
                    print(f"Warning: Found index {code_index} in {file_name} which is outside expected codebook range [0, {codebook_size-1}].")

        except FileNotFoundError:
            print(f"Error: File '{file_path}' not found. Skipping.")
        except Exception as e:
            print(f"An error occurred while processing '{file_path}': {e}. Skipping.")

        if (i - start_index + 1) % 1000 == 0 or i == end_index:
            print(f"Processed {i - start_index + 1} of {end_index - start_index + 1} files...")

    print("\n--- Analysis Complete ---")
    return index_counts

def save_counts_to_file(counts_dict, output_file_path, format='txt'):
    """
    Saves the dictionary of codebook index counts to a file.

    Args:
        counts_dict (dict): The dictionary containing codebook index counts.
        output_file_path (str): The path where the output file should be saved.
        format (str): 'txt' for human-readable text file, 'json' for JSON file.
                      Defaults to 'txt'.
    """
    os.makedirs(os.path.dirname(output_file_path) or '.', exist_ok=True) # Ensure output directory exists

    if format == 'txt':
        with open(output_file_path, 'w') as f:
            f.write("Codebook Index Counts:\n")
            f.write("----------------------\n")
            # Sort for consistent output, by index ascending
            sorted_items = sorted(counts_dict.items(), key=lambda item: item[0])
            for index, count in sorted_items:
                f.write(f"Index {index}: {count}\n")
        print(f"Counts saved to plain text file: {output_file_path}")
    elif format == 'json':
        with open(output_file_path, 'w') as f:
            # Convert defaultdict to a regular dict for JSON serialization
            json.dump(dict(counts_dict), f, indent=4)
        print(f"Counts saved to JSON file: {output_file_path}")
    else:
        print(f"Unsupported output format: {format}. Please choose 'txt' or 'json'.")


# --- Example Usage ---

# Define the directory and file range
codes_directory = 'extracted/imagenet256_codes'
start_file_index = 0
end_file_index = 34744 # Or adjust to a smaller number for testing, e.g., 99
vq_codebook_size = 16384

# Define output file paths
output_dir = 'analysis_results'
txt_output_file = os.path.join(output_dir, 'codebook_counts.txt')
json_output_file = os.path.join(output_dir, 'codebook_counts.json')

# Run the analysis to get counts
total_counts = analyze_npy_codes(
    directory_path=codes_directory,
    start_index=start_file_index,
    end_index=end_file_index,
    codebook_size=vq_codebook_size
)

# Save the counts to files
if total_counts:
    # Save as plain text
    save_counts_to_file(total_counts, txt_output_file, format='txt')

    # Save as JSON
    save_counts_to_file(total_counts, json_output_file, format='json')

    print("\n--- Summary ---")
    print(f"Total unique indices found: {len(total_counts)}")
    # Optionally, print top 10 most frequent to console as well
    print("\n--- Top 10 Most Frequent Codebook Indices (Console) ---")
    sorted_counts_console = sorted(total_counts.items(), key=lambda item: item[1], reverse=True)
    for index, count in sorted_counts_console[:10]:
        print(f"Index {index}: {count} occurrences")
else:
    print("No data processed or no counts generated. No output files saved.")