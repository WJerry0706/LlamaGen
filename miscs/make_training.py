# create_my_parquet_dataset.py
import pathlib
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
import os

# --- Configuration ---
IMAGENET_TRAIN_DIR = "imagenet_subset"
OUTPUT_DIR = "ImageNet-Parquet"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Discover Files ---
print("Discovering images...")
p = pathlib.Path(IMAGENET_TRAIN_DIR)
image_files = list(p.glob('**/*.JPEG'))
num_total_images = len(image_files)
print(f"Found {num_total_images} images.")

# --- Process in Batches ---
batch_size = 512
num_batches = (num_total_images + batch_size - 1) // batch_size
print(f"Creating {num_batches} Parquet files...")

# --- Define the Schema for MyParquetDataset ---
schema = pa.schema([
    pa.field('task2', pa.string()),
    pa.field('image', pa.struct([pa.field('bytes', pa.binary())]))
])

for i in tqdm(range(num_batches), desc="Creating Parquet Files"):
    start_index = i * batch_size
    end_index = min((i + 1) * batch_size, num_total_images)
    batch_paths = image_files[start_index:end_index]

    processed_data = []
    for image_path in batch_paths:
        try:
            image_bytes = image_path.read_bytes()
            processed_data.append({'task2': "", 'image': {'bytes': image_bytes}})
        except Exception as e:
            print(f"Warning: Skipping {image_path}. Error: {e}")

    if not processed_data: continue

    table = pa.Table.from_pylist(processed_data, schema=schema)
    output_filename = f"train-{i:05d}-of-{num_batches:05d}.parquet"
    pq.write_table(table, os.path.join(OUTPUT_DIR, output_filename), compression='snappy')

print("Dataset creation complete!")