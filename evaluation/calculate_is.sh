#!/bin/bash

# --- Configuration ---
PYTHON_SCRIPT="evaluation/IS.py"
# The root directory containing your 'roco_*' folders
BASE_INPUT_DIR="output_adjust"
# Directory where the result files will be saved
RESULTS_DIR="evaluation_results/is"
# --- End of Configuration ---

set -e
mkdir -p "$RESULTS_DIR"

echo "🚀 Starting Inception Score calculation for all 'roco_*' folders..."
echo "=================================================================="

# Loop through all directories in BASE_INPUT_DIR that start with 'roco_'
for INPUT_FOLDER in "${BASE_INPUT_DIR}"/*; do
    # Check if the item found is actually a directory
    if [ -d "$INPUT_FOLDER" ]; then
        # Get the base name of the folder (e.g., "roco_v1_images")
        FOLDER_BASENAME=$(basename "$INPUT_FOLDER")
        OUTPUT_FILE="${RESULTS_DIR}/is_score_${FOLDER_BASENAME}.txt"

        echo ""
        echo "Processing folder: ${FOLDER_BASENAME}"
        echo "-> Input:  ${INPUT_FOLDER}"
        echo "-> Output: ${OUTPUT_FILE}"

        # Execute the Python script
        python "$PYTHON_SCRIPT" \
            --input_dir "$INPUT_FOLDER" \
            --output_file "$OUTPUT_FILE"
    else
        # This message will show if no folders match the 'roco_*' pattern
        echo "⚠️  Warning: No directories found matching the pattern '${BASE_INPUT_DIR}/roco_*'"
        break
    fi
done

echo ""
echo "=================================================================="
echo "✅ All IS calculations are complete."
echo "Results are saved in the '${RESULTS_DIR}' directory."