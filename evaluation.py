import os

# --- Configuration ---
# 1. Path to the folder with your generated images
generated_images_path = "output/temperature=2.0/samples"

# 2. Path to the folder with your real reference images
real_images_path = "real_images" 
# --- End of Configuration ---


# Check if the generated images folder exists
if not os.path.isdir(generated_images_path):
    print(f"Error: Generated images path does not exist: {generated_images_path}")
    exit()

# Check if the real images folder exists
if not os.path.isdir(real_images_path):
    print(f"Error: Real images path does not exist: {real_images_path}")
    print("Please create this folder and fill it with real-world images for comparison.")
    exit()

print("Calculating FID score... This may take a while.")

# Construct and run the command-line tool from the pytorch-fid library
command = f"python -m pytorch_fid {real_images_path} {generated_images_path}"

# Execute the command
os.system(command)

print("FID calculation complete.")