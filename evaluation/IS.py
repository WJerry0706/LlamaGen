import os
import argparse
from torch_fidelity import calculate_metrics

# --- Configuration is handled by command-line arguments ---

def compute_inception_score(images_path, num_samples):
    """
    Calculates the Inception Score (IS) for a directory of images.
    """
    print(f"Calculating Inception Score (IS) using {num_samples} found images...")
    print("This may take a while. ⏳")
    
    # The torch-fidelity library handles all the complex parts
    metrics_dict = calculate_metrics(
        input1=images_path,
        isc=True,                # Enable Inception Score calculation
        isc_splits=10,           # Standard number of splits for IS
        input1_max_samples=num_samples
    )
    
    return metrics_dict


if __name__ == "__main__":
    # --- 1. Set up Argument Parser ---
    parser = argparse.ArgumentParser(description="Calculate Inception Score for a directory of images.")
    
    parser.add_argument("--input_dir", type=str, required=True, 
                        help="Path to the folder with your generated images.")

    parser.add_argument("--output_file", type=str, required=True, 
                        help="Path to the file to save the final IS score.")

    args = parser.parse_args()

    # --- 2. Automatically Count Images in the Input Directory ---
    if not os.path.isdir(args.input_dir):
        print(f"❌ Error: Input directory not found: {args.input_dir}")
        exit()
    
    try:
        # Define common image file extensions
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
        
        # List files and count only the ones with image extensions
        image_files = [f for f in os.listdir(args.input_dir) if os.path.splitext(f)[1].lower() in image_extensions]
        num_images = len(image_files)

        if num_images == 0:
            print(f"❌ Error: No valid image files found in {args.input_dir}")
            exit()

    except OSError as e:
        print(f"❌ Error accessing directory {args.input_dir}: {e}")
        exit()

    # --- 3. Calculate Inception Score ---
    results = compute_inception_score(args.input_dir, num_images)

    # --- 4. Display and Save Results ---
    if results:
        is_mean = results['inception_score_mean']
        is_std = results['inception_score_std']
        
        result_string = f"Inception Score: {is_mean:.4f} ± {is_std:.4f}"
        
        print("\n✅ IS calculation complete.")
        print(result_string)

        try:
            os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
            with open(args.output_file, 'w') as f:
                f.write(result_string + "\n")
            print(f"✅ Result saved to: {args.output_file}")
        except IOError as e:
            print(f"❌ Error: Could not write to file {args.output_file}. Reason: {e}")