#!/bin/bash

# This script runs the Python generation script in parallel on two GPUs.
# It processes the list of temperatures in pairs, assigning one to each GPU.

# Exit immediately if a command fails.
set -e

# --- Configuration ---
PYTHON_MODULE="autoregressive.sample.sample_c2i_multi"
NUM_SAMPLES_PER_CLASS=8
NUM_CLASSES=1000
BASE_OUTPUT_PATH="/data/Work_dir/meissonic_output"
TEMPERATURES=(0.3 0.7 1 1.5 2 3 4 6 8)

# <<< MODIFIED SECTION >>>
# Choose which two GPUs to run the parallel tasks on.
# For example, to use the third and fourth GPUs, set GPU1=2 and GPU2=3.
GPU1=2
GPU2=3


# --- Main Execution Logic ---
echo "🚀 Starting PARALLEL batch image generation on GPU $GPU1 and GPU $GPU2..."
mkdir -p "$BASE_OUTPUT_PATH"

# We use a C-style loop to iterate through the temperatures array two at a time.
for (( i=0; i<${#TEMPERATURES[@]}; i+=2 )); do
    # Get the first temperature for this pair
    temp1=${TEMPERATURES[i]}
    
    # Get the second temperature for this pair (if it exists)
    temp2=${TEMPERATURES[i+1]}

    echo "----------------------------------------------------------"
    echo "🔥 Starting new batch..."
    echo "----------------------------------------------------------"

    # --- Task 1 (on GPU set by GPU1 variable) ---
    echo "Assigning Temperature $temp1 to GPU $GPU1."
    OUTPUT_DIR_1="${BASE_OUTPUT_PATH}/temperature/llamagen_${temp1}"
    
    # Run the first process in the background, assigned to the selected GPU
    CUDA_VISIBLE_DEVICES="$GPU1" python -m "$PYTHON_MODULE" --from-fsdp \
        --temperature "$temp1" \
        --output-dir "$OUTPUT_DIR_1" \
        --num-samples-per-class "$NUM_SAMPLES_PER_CLASS" \
        --num-classes "$NUM_CLASSES" \
        --batch-size 16 &

    # --- Task 2 (on GPU set by GPU2 variable) ---
    # Check if a second temperature exists in this pair (for odd-numbered lists)
    if [ -n "$temp2" ]; then
        echo "Assigning Temperature $temp2 to GPU $GPU2."
        OUTPUT_DIR_2="${BASE_OUTPUT_PATH}/temperature/llamagen_${temp2}"
        
        # Run the second process in the background, assigned to the selected GPU
        CUDA_VISIBLE_DEVICES="$GPU2" python -m "$PYTHON_MODULE" --from-fsdp \
            --temperature "$temp2" \
            --output-dir "$OUTPUT_DIR_2" \
            --num-samples-per-class "$NUM_SAMPLES_PER_CLASS" \
            --num-classes "$NUM_CLASSES" \
            --batch-size 16 &
    fi

    # --- Wait for Both Tasks to Finish ---
    echo "Waiting for jobs (T=$temp1 on GPU $GPU1, T=$temp2 on GPU $GPU2) to complete..."
    wait
    echo "✅ Batch complete."

done

echo ""
echo "----------------------------------------------------------"
echo "🎉 All parallel image generation tasks are complete!"
echo "----------------------------------------------------------"