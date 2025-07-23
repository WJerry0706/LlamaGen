#!/bin/bash

# --- Configuration ---
# The Python script to execute
PYTHON_SCRIPT="evaluation/FID.py"

# The base directory for all model outputs
BASE_INPUT_DIR="output_entro"

# The constant reference directory
REFERENCE_DIR="coco_images"

# The directory where all result files will be saved
RESULTS_DIR="results_fid_coco"
# --- End of Configuration ---

# Exit immediately if a command exits with a non-zero status.
set -e

# Create the results directory if it doesn't exist
mkdir -p "$RESULTS_DIR"

echo "🚀 Starting FID score calculation for all scales..."
echo "==================================================="

# Loop through scales from 0.1 to 2.0 with a step of 0.1
for scale in $(seq 0.1 0.1 4.0)
do
    # Construct the full path for the current input folder
    INPUT_FOLDER="${BASE_INPUT_DIR}/256_coco_captions.txt_${scale}/samples"
    
    # Construct the name for the output file
    OUTPUT_FILE="${RESULTS_DIR}/fid_score_${scale}.txt"

    echo ""
    echo "Processing scale: ${scale}"
    
    # Check if the input directory exists before running the script
    if [ -d "$INPUT_FOLDER" ]; then
        echo "-> Input:  ${INPUT_FOLDER}"
        echo "-> Output: ${OUTPUT_FILE}"
        
        # Execute the Python script with the correct parameters
        python "$PYTHON_SCRIPT" \
            --generated_dir "$INPUT_FOLDER" \
            --real_dir "$REFERENCE_DIR" \
            --output_file "$OUTPUT_FILE"
    else
        echo "⚠️  Warning: Directory not found, skipping. Path: ${INPUT_FOLDER}"
    fi
done

echo ""
echo "==================================================="
echo "✅ All calculations are complete."
echo "Results are saved in the '${RESULTS_DIR}' directory."