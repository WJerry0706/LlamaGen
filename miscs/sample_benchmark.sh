#!/bin/bash

# This script runs the image generation experiment for multiple temperature settings.

# --- Configuration ---
PROMPTS_FILE="benchmark.txt"
TEMPERATURES=(1.0 0.1) # Add more temperatures to this list if needed

# --- Run Experiments ---
echo "Starting generation experiments..."

for TEMP in "${TEMPERATURES[@]}"
do
  echo ""
  echo "================================================="
  echo "  Running with Temperature = $TEMP"
  echo "================================================="
  
  python -m autoregressive.sample.sample_t2i_multi \
    --prompts "$PROMPTS_FILE" \
    --temperature "$TEMP"
    
  echo "Finished run with Temperature = $TEMP"
done

echo ""
echo "================================================="
echo "✅ All experiments are complete."
echo "================================================="