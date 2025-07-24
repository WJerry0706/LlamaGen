#!/bin/bash

# ==============================================================================
# Bash script to run image generation with a range of temperature values
# across multiple prompt collections.
#
# This script will loop through each specified prompt file. For each file,
# it will generate a new random seed and then perform a full temperature sweep.
#
# Prerequisites:
# 1. Required Python packages are installed (torch, torchvision, etc.).
# 2. Pre-trained models are located in the './pretrained_models/' directory.
# 3. Prompt files ('conceptual.txt', 'coco_captions.txt') exist.
# ==============================================================================

# --- Configuration ---
# Add all your prompt collections to this array
PROMPT_FILES=("conceptual.txt" "coco_captions.txt")

# Temperature sweep settings
START_TEMP=0.1
END_TEMP=1.0
STEP_TEMP=0.1

# Other settings
BATCH_SIZE=16           # Adjust based on your VRAM

# --- Script Body ---
echo "🚀 Starting generation for multiple prompt collections..."
echo "========================================================"

# Loop through each prompt file in the array
for prompts_file in "${PROMPT_FILES[@]}"
do
  echo ""
  echo "********************************************************"
  echo "➡️  Processing prompts from: $prompts_file"

  # Generate a new random seed for this entire prompt collection's sweep
  SEED=$RANDOM
  echo "🌱 Using random seed for this set: $SEED"
  echo "********************************************************"

  # Generate a sequence of temperatures and loop through them
  for temp in $(seq $START_TEMP $STEP_TEMP $END_TEMP)
  do
    echo ""
    echo "--- Running generation with temperature: $temp ---"
    
    # Check if the prompt file exists before running
    if [ ! -f "$prompts_file" ]; then
        echo "⚠️  Warning: Prompt file '$prompts_file' not found. Skipping."
        continue
    fi

    # Execute the Python script with the current prompt file and temperature
    python -m autoregressive.sample.sample_t2i \
      --prompts "$prompts_file" \
      --temperature "$temp" \
      --seed "$SEED" 

    echo "--- Finished run for temperature: $temp ---"
  done
done

echo ""
echo "========================================================"
echo "✅ All experiments complete."
echo "Generated images are saved in the 'output/' directory."