import torch
import re

# === Configuration ===
txt_file = "analysis_results/imagenet/codebook_counts.txt"   # input file path
pt_file = "codebook_probs/imagenet.pt"      # output file path
num_bins = 16384                    # total indices

# === Step 1: Read text file ===
with open(txt_file, "r") as f:
    content = f.read()

# === Step 2: Parse counts ===
# Matches: "Index <number>: <count>"
pattern = r"Index\s+(\d+):\s+(\d+)"
matches = re.findall(pattern, content)

counts = {int(idx): int(cnt) for idx, cnt in matches}

# === Step 3: Create counts tensor ===
count_tensor = torch.zeros(num_bins, dtype=torch.float64)
for idx, c in counts.items():
    if 0 <= idx < num_bins:
        count_tensor[idx] = c

# === Step 4: Compute probabilities ===
total = count_tensor.sum()
if total == 0:
    raise ValueError("No counts found; total is zero.")

prob_tensor = (count_tensor / total).to(torch.float32)

# === Step 5: Save to .pt file ===
torch.save(prob_tensor, pt_file)

print(f"Saved probabilities to '{pt_file}' with shape {prob_tensor.shape}")
print(f"Probabilities sum to: {prob_tensor.sum().item():.6f}")
print(f"Non-zero entries: {(prob_tensor > 0).sum().item()}")
