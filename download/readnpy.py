import numpy as np
import os
from collections import defaultdict
import json
import matplotlib.pyplot as plt

def analyze_npy_codes(directory_path, start_index, end_index, codebook_size=16384):
    """
    Reads .npy files in a specified range, extracts the first layer of codes,
    and counts the occurrences of each index within a given codebook size.
    """
    index_counts = defaultdict(int)

    print(f"Starting analysis of .npy files from {start_index}.npy to {end_index}.npy in {directory_path}")
    print(f"Expected codebook size: {codebook_size}")

    if not os.path.isdir(directory_path):
        print(f"Error: Directory '{directory_path}' not found. Please ensure the path is correct.")
        return index_counts

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
            pass
        except Exception as e:
            print(f"An error occurred while processing '{file_path}': {e}. Skipping.")

        if (i - start_index + 1) % 1000 == 0 or i == end_index:
            print(f"Processed {i - start_index + 1} of {end_index - start_index + 1} files...")

    print("\n--- Analysis Complete ---")
    return index_counts

def save_counts_to_file(counts_dict, output_file_path, format='txt'):
    """
    Saves the dictionary of codebook index counts to a file.
    """
    os.makedirs(os.path.dirname(output_file_path) or '.', exist_ok=True)

    if format == 'txt':
        with open(output_file_path, 'w') as f:
            f.write("Codebook Index Counts:\n")
            f.write("----------------------\n")
            sorted_items = sorted(counts_dict.items(), key=lambda item: item[0])
            for index, count in sorted_items:
                f.write(f"Index {index}: {count}\n")
        print(f"Counts saved to plain text file: {output_file_path}")
    elif format == 'json':
        with open(output_file_path, 'w') as f:
            json.dump(dict(counts_dict), f, indent=4)
        print(f"Counts saved to JSON file: {output_file_path}")
    else:
        print(f"Unsupported output format: {format}. Please choose 'txt' or 'json'.")

def plot_and_save_counts_by_index(counts_dict, output_image_path):
    """
    Plots the distribution of codebook indices in numerical order (not by frequency)
    using a LINEAR scale for the y-axis.
    """
    if not counts_dict:
        print("No data to plot. Skipping image generation.")
        return

    indices = sorted(counts_dict.keys())
    counts = [counts_dict[index] for index in indices]
    
    os.makedirs(os.path.dirname(output_image_path) or '.', exist_ok=True)

    plt.figure(figsize=(20, 8))
    plt.bar(indices, counts, color='skyblue')
    
    # plt.yscale('log') # This line has been removed as requested.
    
    plt.xlabel('Codebook Index')
    plt.ylabel('Count') # Label updated to reflect linear scale
    plt.title('Distribution of All Codebook Index Frequencies (Sorted by Index)')
    plt.xticks([]) 
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    try:
        plt.savefig(output_image_path)
        print(f"Plot saved to: {output_image_path}")
    except Exception as e:
        print(f"Error saving plot to '{output_image_path}': {e}")

    plt.close()

def plot_and_save_counts_by_frequency(counts_dict, output_image_path):
    """
    Plots the distribution of codebook indices sorted by frequency in descending order
    using a LINEAR scale for the y-axis.
    """
    if not counts_dict:
        print("No data to plot. Skipping image generation.")
        return

    sorted_items = sorted(counts_dict.items(), key=lambda item: item[1], reverse=True)
    counts = [item[1] for item in sorted_items]

    os.makedirs(os.path.dirname(output_image_path) or '.', exist_ok=True)

    plt.figure(figsize=(20, 8))
    plt.bar(range(len(counts)), counts, color='lightgreen')
    
    # plt.yscale('log') # This line has been removed as requested.
    
    plt.xlabel('Codebook Index (Ranked by Frequency)')
    plt.ylabel('Count') # Label updated to reflect linear scale
    plt.title('Distribution of All Codebook Index Frequencies (Sorted by Count)')
    plt.xticks([])
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    try:
        plt.savefig(output_image_path)
        print(f"Plot saved to: {output_image_path}")
    except Exception as e:
        print(f"Error saving plot to '{output_image_path}': {e}")

    plt.close()

def plot_low_frequency_histogram(counts_dict, output_image_path, num_bins=50):
    """
    Selects the 98% of indices with the lowest frequencies and plots a
    histogram of their occurrence counts.
    """
    if not counts_dict:
        print("No data to plot. Skipping low-frequency histogram generation.")
        return
        
    sorted_items = sorted(counts_dict.items(), key=lambda item: item[1])
    num_unique_indices = len(sorted_items)
    cutoff_index = int(num_unique_indices * 0.98)
    low_freq_counts = [item[1] for item in sorted_items[:cutoff_index]]

    if not low_freq_counts:
        print("Not enough data to generate a low-frequency histogram.")
        return

    os.makedirs(os.path.dirname(output_image_path) or '.', exist_ok=True)
    
    plt.figure(figsize=(20, 8))
    plt.hist(low_freq_counts, bins=num_bins, color='coral', edgecolor='black')
    
    plt.xlabel('Number of Occurrences (for a single token)')
    plt.ylabel('Number of Tokens in Bin')
    plt.title('Histogram of Counts for the 98% Least Frequent Tokens')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    try:
        plt.savefig(output_image_path)
        print(f"Low-frequency histogram saved to: {output_image_path}")
    except Exception as e:
        print(f"Error saving histogram to '{output_image_path}': {e}")
        
    plt.close()


# --- Main execution block ---
codes_directory = 'laion_huge/laion_images_256_codes'
start_file_index = 0
end_file_index = 14000
vq_codebook_size = 16384

output_dir = 'analysis_results/laion_huge'
txt_output_file = os.path.join(output_dir, 'codebook_counts.txt')
image_output_file_by_index = os.path.join(output_dir, 'codebook_distribution_by_index_linear.png') # Renamed file
image_output_file_by_frequency = os.path.join(output_dir, 'codebook_distribution_by_frequency_linear.png') # Renamed file
low_freq_histogram_output_file = os.path.join(output_dir, 'low_frequency_histogram.png')

# Run the analysis to get counts
total_counts = analyze_npy_codes(
    directory_path=codes_directory,
    start_index=start_file_index,
    end_index=end_file_index,
    codebook_size=vq_codebook_size
)

# Process and save results only if data was processed
if total_counts:
    save_counts_to_file(total_counts, txt_output_file, format='txt')
    
    plot_and_save_counts_by_index(total_counts, image_output_file_by_index)
    plot_and_save_counts_by_frequency(total_counts, image_output_file_by_frequency)
    plot_low_frequency_histogram(total_counts, low_freq_histogram_output_file)

    print("\n--- Summary ---")
    print(f"Total unique indices found: {len(total_counts)}")
    
    total_sum_counts = sum(total_counts.values())
    average_count = total_sum_counts / len(total_counts) if len(total_counts) > 0 else 0

    sorted_counts = sorted(total_counts.items(), key=lambda item: item[1], reverse=True)
    top_10_indices = sorted_counts[:10]
    last_10_indices = sorted_counts[-10:]

    print("\n--- Distribution Information ---")
    print(f"Average count per index: {average_count:.2f}")

    print("\n--- Top 10 Most Frequent Codebook Indices ---")
    for index, count in top_10_indices:
        print(f"Index {index}: {count} occurrences")

    print("\n--- Last 10 Least Frequent Codebook Indices ---")
    if last_10_indices:
        for index, count in last_10_indices:
            print(f"Index {index}: {count} occurrences")
    else:
        print("Not enough unique indices to display the last 10.")
else:
    print("\nNo data processed. Cannot provide distribution information.")