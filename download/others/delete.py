# Define the input and output filenames
input_filename = "prompts_cleaned.txt"
output_filename = "cleaned_prompts.txt"

cleaned_lines = []

try:
    # Open the input file to read the original prompts
    with open(input_filename, 'r', encoding='utf-8') as f:
        for line in f:
            # Strip any leading/trailing whitespace from the line
            line = line.strip()
            if not line:
                continue

            # Find the position of the first colon
            try:
                colon_index = line.index(':')
                # Take the part of the string after the first colon
                cleaned_line = line[colon_index + 1:].strip()
                cleaned_lines.append(cleaned_line)
            except ValueError:
                # If a line has no colon, add it as is
                cleaned_lines.append(line)

    # Write the cleaned lines to the output file
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write("\n".join(cleaned_lines))

    print(f"Successfully processed the file.")
    print(f"Cleaned prompts have been saved to '{output_filename}'")

except FileNotFoundError:
    print(f"Error: The file '{input_filename}' was not found.")
    print("Please create the file and add your text to it.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")