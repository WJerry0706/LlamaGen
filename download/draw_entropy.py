import matplotlib.pyplot as plt
import numpy as np
import os

# --- Configuration ---
# The base paths where your temperature-specific folders are located.
base_path_coco = "output_entro/256_coco_captions.txt_{temp}/entropy_results.txt"
base_path_conceptual = "output_entro/256_conceptual.txt_{temp}/entropy_results.txt"

# --- Data Reading and Processing ---
# Lists for the COCO Captions dataset
temperatures_coco = []
entropies_coco = []

# Lists for the Conceptual Captions dataset
temperatures_conceptual = []
entropies_conceptual = []

# Generate a list of temperatures to check, from 0.1 to 4.0 with a step of 0.1
temp_values_to_check = np.arange(0.1, 4.1, 0.1)

print("Searching for entropy data files for both datasets...")

# Helper function to read, parse, and scale the entropy file
def read_entropy_from_file(file_path):
    """Reads a single entropy value from a given file path and divides it by 256."""
    try:
        with open(file_path, 'r') as f:
            line = f.readline()
            if "Average Entropy:" in line:
                # Extract the value, convert to float, and divide by 256
                entropy_val = float(line.split("Average Entropy:")[1].strip())
                return entropy_val / 256
    except (IOError, ValueError) as e:
        print(f"Could not read or parse file {file_path}: {e}")
    return None

# Loop through each temperature value to find and read the corresponding files
for temp in temp_values_to_check:
    temp_str = f"{temp:.1f}"

    # --- Process COCO data ---
    file_path_c = base_path_coco.format(temp=temp_str)
    if os.path.exists(file_path_c):
        entropy_val = read_entropy_from_file(file_path_c)
        if entropy_val is not None:
            temperatures_coco.append(temp)
            entropies_coco.append(entropy_val)
            print(f"Successfully read COCO data for temperature: {temp_str}")

    # --- Process Conceptual data ---
    file_path_n = base_path_conceptual.format(temp=temp_str)
    if os.path.exists(file_path_n):
        entropy_val = read_entropy_from_file(file_path_n)
        if entropy_val is not None:
            temperatures_conceptual.append(temp)
            entropies_conceptual.append(entropy_val)
            print(f"Successfully read Conceptual data for temperature: {temp_str}")


# --- Plotting ---
# Check if any data was successfully read before trying to plot
if (temperatures_coco and entropies_coco) or (temperatures_conceptual and entropies_conceptual):
    print("\nData found. Generating plot...")
    plt.figure(figsize=(12, 7))

    # Plot COCO data if it exists
    if temperatures_coco and entropies_coco:
        plt.plot(temperatures_coco, entropies_coco, marker='o', linestyle='-', color='b', label='COCO Captions')

    # Plot Conceptual data if it exists
    if temperatures_conceptual and entropies_conceptual:
        plt.plot(temperatures_conceptual, entropies_conceptual, marker='s', linestyle='--', color='r', label='Conceptual Captions')

    # Add labels and a title for clarity
    plt.title('Average Model Entropy vs. Sampling Temperature', fontsize=16)
    plt.xlabel('Temperature', fontsize=12)
    # Update the Y-axis label to reflect the division
    plt.ylabel('Average Entropy / 256', fontsize=12)

    # Add a grid and a legend
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(fontsize=12)

    # Save the plot to a file
    output_filename = 'entropy_vs_temperature_scaled_comparison.png'
    plt.savefig(output_filename)

    print(f"Plot successfully saved as {output_filename}")
    # To display the plot in environments that support it:
    # plt.show()
else:
    # Inform the user if no data was found
    print("\nNo data was found to plot. Please ensure the file paths are correct and the files contain the expected data.")