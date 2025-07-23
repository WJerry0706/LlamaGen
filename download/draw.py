import os
import re
import numpy as np
import matplotlib.pyplot as plt

# --- Configuration ---
# Directories where your result files are stored
fid_results_dir = 'results_fid_conceptual'
is_results_dir = 'results_is_conceptual'

# Output file name for the plot
output_plot_file = 'conceptual.png'
# --- End of Configuration ---


def parse_score_file(filepath):
    """
    Parses a result file to extract the mean score and standard deviation.
    Handles two formats:
    1. "Score Name: 14.2255 ± 2.0766" (for IS)
    2. "FID score (...): 80.4526"      (for FID)
    """
    if not os.path.exists(filepath):
        return np.nan, np.nan

    with open(filepath, 'r') as f:
        content = f.read()
        
        match_with_std = re.search(r":\s*([\d.]+)\s*±\s*([\d.]+)", content)
        if match_with_std:
            mean_score = float(match_with_std.group(1))
            std_dev = float(match_with_std.group(2))
            return mean_score, std_dev
            
        match_mean_only = re.search(r":\s*([\d.]+)", content)
        if match_mean_only:
            mean_score = float(match_mean_only.group(1))
            return mean_score, 0.0
            
    return np.nan, np.nan


# 1. Define the range of temperatures to plot
temperatures = np.arange(0.1, 2.1, 0.1)

# 2. Load the data from the text files
fid_scores, fid_errors = [], []
is_scores, is_errors = [], []

print("Reading score files...")
for temp in temperatures:
    temp_str = f"{temp:.1f}"
    fid_file = os.path.join(fid_results_dir, f'fid_score_{temp_str}.txt')
    is_file = os.path.join(is_results_dir, f'is_score_{temp_str}.txt')
    
    mean, std = parse_score_file(fid_file)
    fid_scores.append(mean)
    fid_errors.append(std)
    
    mean, std = parse_score_file(is_file)
    is_scores.append(mean)
    is_errors.append(std)

# 3. Create the plot
print("Generating plot...")
fig, ax1 = plt.subplots(figsize=(12, 7))

color1 = 'tab:blue'
ax1.set_xlabel('Temperature', fontsize=14)
ax1.set_ylabel('FID Score (Lower is Better)', color=color1, fontsize=14)
ax1.plot(temperatures, fid_scores, 'o-', color=color1, label='FID Score')
ax1.tick_params(axis='y', labelcolor=color1)

ax2 = ax1.twinx()
color2 = 'tab:red'
ax2.set_ylabel('Inception Score (Higher is Better)', color=color2, fontsize=14)
ax2.plot(temperatures, is_scores, 's--', color=color2, label='Inception Score')
ax2.tick_params(axis='y', labelcolor=color2)

# 4. Final plot adjustments
plt.title('Model Performance vs. Sampling Temperature', fontsize=16)
ax1.grid(True, linestyle='--', alpha=0.6)

lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='best')

fig.tight_layout()

# 5. Save the plot to a file
plt.savefig(output_plot_file, dpi=300)

# The following line is commented out to prevent the plot from showing on screen
# plt.show()

print(f"✅ Plot saved to {output_plot_file}")