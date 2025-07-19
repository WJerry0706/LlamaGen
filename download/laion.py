import pandas as pd
from datasets import load_dataset

NUM_SAMPLES = 1_000_000
DATASET_NAME = "laion/laion-coco"
OUTPUT_FILENAME = "laion-coco-1M-metadata.tsv"

print(f"Streaming metadata for the first {NUM_SAMPLES:,} samples from {DATASET_NAME}...")

# Load the dataset in streaming mode
ds = load_dataset(DATASET_NAME, streaming=True)

# Take the first N samples from the 'train' split
ds_subset = ds['train'].take(NUM_SAMPLES)

# Convert the subset to a pandas DataFrame
df = pd.DataFrame(ds_subset)

# Keep only the URL and TEXT columns needed by img2dataset
df_to_save = df[['URL', 'TEXT']]

# Save to a tab-separated file (.tsv)
df_to_save.to_csv(OUTPUT_FILENAME, sep='\t', header=False, index=False)

print(f"Successfully saved metadata for {len(df_to_save):,} pairs to '{OUTPUT_FILENAME}'")