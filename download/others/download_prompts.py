# Import the function to load datasets
from datasets import load_dataset

# 1. Load the Parti Prompts dataset from Hugging Face
#    This will download the data to a local cache.
print("Downloading dataset...")
dataset = load_dataset("nateraw/parti-prompts")
print("Download complete.")

# 2. Extract the list of prompts from the 'Prompt' column
#    The dataset has one split, named 'train'.
prompts = dataset['train']['Prompt']

# 3. Write all the prompts to a text file
output_filename = "parti_prompts.txt"
with open(output_filename, "w", encoding="utf-8") as f:
    for prompt in prompts:
        # Write each prompt followed by a newline character
        f.write(prompt + "\n")

print(f"\nSuccessfully saved {len(prompts)} prompts to {output_filename}")