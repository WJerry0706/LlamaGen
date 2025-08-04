import re

# === Configuration ===
input_file = "download/others/Validation_GCC-1.1.0-Validation.tsv"       # Your input .txt file
output_file = "coco.txt"        # Where to save cleaned captions

# === Process the file ===
with open(input_file, 'r', encoding='utf-8') as infile, \
     open(output_file, 'w', encoding='utf-8') as outfile:
    
    for line in infile:
        # Remove the tab + URL from the end
        clean_line = re.sub(r'\s*\t?https?://\S+$', '', line.strip())
        if clean_line:
            outfile.write(clean_line + '\n')

print(f"✅ Cleaned captions written to: {output_file}")
