import os
import torch
import argparse  # Imported for command-line arguments
import numpy as np
from PIL import Image
from cleanfid import fid
from torchvision.transforms import functional as F

# --- All configuration is now handled by command-line arguments ---

class CenterCropLongEdge(object):
    """
    Crops an image to a square by cropping the long edge.
    """
    def __call__(self, img):
        short_edge = min(img.size)
        return F.center_crop(img, (short_edge, short_edge))

    def __repr__(self):
        return self.__class__.__name__

@torch.no_grad()
def compute_fid_score(gt_dir, fake_dir, resize_size, model_name):
    """
    Calculates the FID score using the clean-fid library with custom preprocessing.
    """
    center_crop_trsf = CenterCropLongEdge()
    def custom_image_transform(image_np):
        image_pil = Image.fromarray(image_np)
        image_pil = center_crop_trsf(image_pil)
        if resize_size is not None:
            image_pil = image_pil.resize((resize_size, resize_size), Image.LANCZOS)
        return np.array(image_pil)

    if model_name == "inception":
        fid_model_name = "inception_v3"
    elif model_name == "clip":
        fid_model_name = "clip_vit_b_32"
    else:
        raise ValueError(f"Unknown feature extractor: {model_name}")

    score = fid.compute_fid(
        gt_dir,
        fake_dir,
        model_name=fid_model_name,
        custom_image_tranform=custom_image_transform
    )
    return score


if __name__ == "__main__":
    # --- 1. Set up Argument Parser ---
    parser = argparse.ArgumentParser(description="Calculate FID score between two image directories.")
    
    parser.add_argument("--generated_dir", type=str, required=True, 
                        help="Path to the folder with your generated images.")
                        
    parser.add_argument("--real_dir", type=str, required=True, 
                        help="Path to the folder with your real reference images.")

    parser.add_argument("--output_file", type=str, required=True, 
                        help="Path to the file to save the final FID score.")
                        
    parser.add_argument("--resolution", type=int, default=1024,
                        help="Resolution to resize images to before FID calculation (e.g., 256).")
                        
    parser.add_argument("--feature_extractor", type=str, default="inception", choices=["inception", "clip"],
                        help="Feature extractor model to use ('inception' or 'clip').")

    args = parser.parse_args()

    # --- 2. Validate Paths ---
    if not os.path.isdir(args.generated_dir):
        print(f"❌ Error: Generated images path does not exist: {args.generated_dir}")
        exit()

    if not os.path.isdir(args.real_dir):
        print(f"❌ Error: Real images path does not exist: {args.real_dir}")
        exit()

    # --- 3. Calculate FID Score ---
    print(f"Comparing folders:\n- Generated: {args.generated_dir}\n- Real: {args.real_dir}")
    print("Calculating FID score... This may take a while. ⏳")

    fid_score = compute_fid_score(
        gt_dir=args.real_dir,
        fake_dir=args.generated_dir,
        resize_size=args.resolution,
        model_name=args.feature_extractor
    )
    
    # --- 4. Display and Save Results ---
    result_string = f"FID score ({args.feature_extractor} @ {args.resolution}px): {fid_score:.4f}"
    
    print("\n✅ FID calculation complete.")
    print(result_string)

    try:
        with open(args.output_file, 'w') as f:
            f.write(result_string + "\n")
        print(f"✅ Result saved to: {args.output_file}")
    except IOError as e:
        print(f"❌ Error: Could not write to file {args.output_file}. Reason: {e}")