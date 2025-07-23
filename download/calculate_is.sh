#!/bin/bash

# --- Configuration ---
PYTHON_SCRIPT="evaluation/IS.py"
BASE_INPUT_DIR="output_entro"
RESULTS_DIR="results_is_coco"
# The NUM_IMAGES variable is no longer needed.
# --- End of Configuration ---

set -e
mkdir -p "$RESULTS_DIR"

echo "🚀 Starting Inception Score calculation for all scales..."
echo "========================================================"

for scale in $(seq 0.1 0.1 4.0)
do
    INPUT_FOLDER="${BASE_INPUT_DIR}/256_coco_captions.txt_${scale}/samples"
    OUTPUT_FILE="${RESULTS_DIR}/is_score_${scale}.txt"

    echo ""
    echo "Processing scale: ${scale}"
    
    if [ -d "$INPUT_FOLDER" ]; then
        echo "-> Input:  ${INPUT_FOLDER}"
        echo "-> Output: ${OUTPUT_FILE}"
        
        # Execute the Python script without the --num_images argument
        python "$PYTHON_SCRIPT" \
            --input_dir "$INPUT_FOLDER" \
            --output_file "$OUTPUT_FILE"
    else
        echo "⚠️  Warning: Directory not found, skipping. Path: ${INPUT_FOLDER}"
    fi
done

echo ""
echo "========================================================"
echo "✅ All IS calculations are complete."
echo "Results are saved in the '${RESULTS_DIR}' directory."