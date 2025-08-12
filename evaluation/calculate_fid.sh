#!/bin/bash

# --- Configuration ---
# Path to your new FID calculation script
PYTHON_SCRIPT="evaluation/FID.py"

# The root directory containing your generated image folders (e.g., 'meissonic_v1', 'meissonic_v2')
BASE_GENERATED_DIR="output_adjust"

# The root directory containing your real reference image folders (e.g., 'meissonic', 'another_dataset')
BASE_REAL_DIR="/home/jieyu/Meissonic"

# Directory where the result files will be saved
RESULTS_DIR="evaluation_results/fid"

# FID calculation parameters from your Python script
RESOLUTION=1024
FEATURE_EXTRACTOR="inception" # 'inception' or 'clip'
# --- End of Configuration ---

set -e
mkdir -p "$RESULTS_DIR"

echo "🚀 Starting FID score calculation for all 'meissonic_*' folders..."
echo "======================================================================"

# Loop through all directories in BASE_GENERATED_DIR that start with 'meissonic_'
for GENERATED_FOLDER in "${BASE_GENERATED_DIR}"/*; do
    # Check if the item found is actually a directory
    if [ -d "$GENERATED_FOLDER" ]; then
        # Get the base name of the generated folder (e.g., "meissonic_v1_run1")
        GENERATED_BASENAME=$(basename "$GENERATED_FOLDER")
        
        # --- LOGIC: Determine the real folder name ---
        # Takes the part of the name before the first underscore (e.g., "meissonic")
        REAL_FOLDER_NAME="${GENERATED_BASENAME%%_*}"
        REAL_FOLDER_PATH="${BASE_REAL_DIR}/${REAL_FOLDER_NAME}"
        REAL_FOLDER_PATH="imagenet"

        # --- Sanity Check: Ensure the real folder exists before proceeding ---
        if [ ! -d "$REAL_FOLDER_PATH" ]; then
            echo ""
            echo "⚠️  Warning: Skipping '${GENERATED_BASENAME}'. Corresponding real folder not found at: ${REAL_FOLDER_PATH}"
            continue # Skip to the next folder
        fi

        OUTPUT_FILE="${RESULTS_DIR}/fid_score_${GENERATED_BASENAME}.txt"

        echo ""
        echo "Processing: ${GENERATED_BASENAME}"
        echo "-> Generated: ${GENERATED_FOLDER}"
        echo "-> Real:      ${REAL_FOLDER_PATH}"
        echo "-> Output:    ${OUTPUT_FILE}"

        # Execute the Python script with all required arguments
        python "$PYTHON_SCRIPT" \
            --generated_dir "$GENERATED_FOLDER" \
            --real_dir "$REAL_FOLDER_PATH" \
            --output_file "$OUTPUT_FILE" \
            --resolution "$RESOLUTION" \
            --feature_extractor "$FEATURE_EXTRACTOR"
    fi
done

echo ""
echo "======================================================================"
echo "✅ All FID calculations are complete."
echo "Results are saved in the '${RESULTS_DIR}' directory."