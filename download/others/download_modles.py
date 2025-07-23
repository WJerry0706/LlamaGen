from transformers import T5Tokenizer, T5ForConditionalGeneration

model_name = "google/flan-t5-xxl"

print("Loading tokenizer...")
tokenizer = T5Tokenizer.from_pretrained(model_name)

print("Loading model... This will download ~44 GB and may take a very long time.")
# device_map="auto" and load_in_8bit=True are essential for managing this huge model
model = T5ForConditionalGeneration.from_pretrained(
    model_name,
    device_map="auto",
    load_in_8bit=True
)

print("\nFLAN-T5-XXL has been downloaded and loaded successfully!")