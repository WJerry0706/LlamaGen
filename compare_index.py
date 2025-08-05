import os
import re
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def parse_frequency_file(filepath):
    """
    Reads a text file with "Index X: Y" format and returns a dictionary
    of index-to-frequency mappings.

    Args:
        filepath (str): The path to the input text file.

    Returns:
        dict: A dictionary where keys are the integer indices and values are
              their integer counts. Returns an empty dictionary if file not found.
    """
    if not os.path.exists(filepath):
        print(f"Error: File not found at '{filepath}'")
        return {}
    
    freq_dict = {}
    line_pattern = re.compile(r'Index\s+(\d+):\s+(\d+)')

    with open(filepath, 'r') as f:
        for line in f:
            # print(line)
            match = line_pattern.search(line)
            if match:
                index = int(match.group(1))
                count = int(match.group(2))
                # print(count)
                freq_dict[index] = count

    # print(freq_dict)
    return freq_dict

def plot_frequency_comparison(freq_dict1, freq_dict2, output_image_path, regression_results, source1_name="Source 1", source2_name="Source 2"):
    """
    Generates and saves a scatter plot with a linear regression line.

    Args:
        freq_dict1 (dict): Frequency data from the first source.
        freq_dict2 (dict): Frequency data from the second source.
        output_image_path (str): Path to save the output plot image.
        regression_results (dict): A dictionary containing 'slope', 'intercept', and 'r2'.
        source1_name (str): Name of the first data source for plot labels.
        source2_name (str): Name of the second data source for plot labels.
    """
    if not freq_dict1 or not freq_dict2:
        print("One or both frequency dictionaries are empty. Cannot generate plot.")
        return

    all_indices = sorted(list(set(freq_dict1.keys()) | set(freq_dict2.keys())))
    x_values = [freq_dict1.get(index, 0) for index in all_indices]
    y_values = [freq_dict2.get(index, 0) for index in all_indices]

    os.makedirs(os.path.dirname(output_image_path) or '.', exist_ok=True)
    
    plt.figure(figsize=(12, 12))
    plt.scatter(x_values, y_values, alpha=0.6, s=50, edgecolors='k', c='royalblue', label='Codebook Indices')
    
    max_val = max(max(x_values), max(y_values))
    
    # Plot y=x reference line
    plt.plot([0, max_val], [0, max_val], color='red', linestyle='--', alpha=0.8, label='y = x (Equal Frequency)')
    
    # Plot the linear regression line
    slope = regression_results['slope']
    intercept = regression_results['intercept']
    r2 = regression_results['r2']
    
    regression_x = np.array([0, max_val])
    regression_y = slope * regression_x + intercept
    
    plt.plot(regression_x, regression_y, color='green', linewidth=2, 
             label=f'Linear Regression\ny={slope:.2f}x + {intercept:.2f}\n$R^2$={r2:.3f}')
    
    plt.xlabel(f'Frequency in training dataset', fontsize=12)
    plt.ylabel(f'Frequency in inference dataset', fontsize=12)
    plt.title('Linear Regression of Codebook Index Frequencies', fontsize=16)
    
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.xlim(0, max_val * 1.05)
    plt.ylim(0, max_val * 1.05)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()

    try:
        plt.savefig(output_image_path)
        print(f"\nComparison plot with regression line saved to: {output_image_path}")
    except Exception as e:
        print(f"Error saving plot: {e}")
        
    plt.close()

file1_path = "analysis_results/imagenet/codebook_counts.txt"
file2_path = "analysis_results/meissonic/codebook_counts.txt"
output_plot_file = "analysis_results/cross_dataset.png"

# 1. Parse the two files into dictionaries
freq1 = parse_frequency_file(file1_path)
freq2 = parse_frequency_file(file2_path)

# print(freq1)
# print(freq2)

if freq1 and freq2:
    # 2. Prepare data for scikit-learn
    all_indices = sorted(list(set(freq1.keys()) | set(freq2.keys())))
    x_values = np.array([freq1.get(index, 0) for index in all_indices]).reshape(-1, 1)
    y_values = np.array([freq2.get(index, 0) for index in all_indices])

    # 3. Perform Linear Regression
    model = LinearRegression()
    model.fit(x_values, y_values)

    # 4. Get and print results
    slope = model.coef_[0]
    intercept = model.intercept_
    y_pred = model.predict(x_values)
    r2 = r2_score(y_values, y_pred)

    print("\n--- Linear Regression Results ---")
    print(f"Slope (m): {slope:.4f}")
    print(f"Intercept (c): {intercept:.4f}")
    print(f"R-squared: {r2:.4f}")

    regression_results = {'slope': slope, 'intercept': intercept, 'r2': r2}

    # 5. Generate and save the plot with the regression line
    plot_frequency_comparison(
        freq1, 
        freq2, 
        output_plot_file,
        regression_results,
        source1_name="File 1", 
        source2_name="File 2"
    )