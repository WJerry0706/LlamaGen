#!/usr/bin/env bash
set -euo pipefail

# --- Configuration ---
# Name of your Python script
PYTHON_SCRIPT="main.py"

# <<< SET THE DIFFERENT TAU VALUES YOU WANT TO TEST HERE >>>
# The script will loop through each value in this array.
TAUS=(0.0 0.01 0.02 0.04 0.07 0.1 0.15 0.2 0.25 0.3 0.4 0.5 0.75 1.0)

# ----- Model & Path Configuration -----
GPT_MODEL="GPT-XXL"
GPT_CKPT="./pretrained_models/c2i_XXL_384.pt"
PRIORS_PATH="codebook_probs/imagenet.pt"
VQ_MODEL="VQ-16"
VQ_CKPT="./pretrained_models/vq_ds16_c2i.pt"

# ----- GPT Model Settings -----
GPT_TYPE="c2i"
FROM_FSDP=0             # Set to 1 to add the --from-fsdp flag
PRECISION="bf16"
COMPILE_MODEL=0         # Set to 1 to add the --compile flag

# ----- VQ & Image Settings -----
CODEBOOK_SIZE=16384
CODEBOOK_EMBED_DIM=8
IMAGE_SIZE=384
DOWNSAMPLE_SIZE=16
NUM_CLASSES=1000

# ----- Generation Settings -----
CFG_SCALE=4.0
TOP_K=2000
TEMPERATURE=1.0
TOP_P=1.0
NUM_SAMPLES_PER_CLASS=2
BATCH_SIZE=8

# --- Script Execution ---

echo "Starting generation script..."
echo "Will test for ${#TAUS[@]} tau values: ${TAUS[*]}"
echo ""

# Loop through each specified tau value
for tau in "${TAUS[@]}"; do
    echo "========================================="
    echo "===== Running generation for tau = $tau ====="
    echo "========================================="

    # Handle optional boolean flags
    optional_flags=()
    if [[ "$COMPILE_MODEL" == "1" ]]; then
        optional_flags+=(--compile)
    fi
    if [[ "$FROM_FSDP" == "1" ]]; then
        optional_flags+=(--from-fsdp)
    fi

    # Run the Python script with the current tau value and all other settings
    python -m autoregressive.sample.sample_c2i_multi --from-fsdp \
        --tau "$tau" \

    echo "--- Finished for tau = $tau ---"
    echo ""
done

echo "All generation runs are complete."