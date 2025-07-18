import os
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Define the model name from the Hugging Face Hub
model_name = "google/flan-t5-xl"
# Define the name for your output folder
output_dir = "t5-ckpt"

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

print(f"Loading model and tokenizer for '{model_name}'...")
print("This may take a while...")

# Load the pretrained model and tokenizer
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)
print("Model and tokenizer loaded successfully.")


print(f"\nSaving model and tokenizer to '{output_dir}'...")
# Save the model's weights and configuration file
model.save_pretrained(output_dir)

# Save the tokenizer's files
tokenizer.save_pretrained(output_dir)

print(f"All files have been saved to the '{output_dir}' folder.")