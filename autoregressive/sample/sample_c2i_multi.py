# Modified from:
#   DiT:  https://github.com/facebookresearch/DiT/blob/main/sample.py
import re
import torch
import random # Import random for setting different seeds
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)
from torchvision.utils import save_image

import time
import argparse
from tokenizer.tokenizer_image.vq_model import VQ_models
from autoregressive.models.gpt import GPT_models
from autoregressive.models.generate import generate


def main(args):
    # Setup PyTorch:
    # We will set seeds dynamically inside the loop
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # create and load model
    vq_model = VQ_models[args.vq_model](
        codebook_size=args.codebook_size,
        codebook_embed_dim=args.codebook_embed_dim)
    vq_model.to(device)
    vq_model.eval()
    checkpoint = torch.load(args.vq_ckpt, map_location="cpu")
    vq_model.load_state_dict(checkpoint["model"])
    del checkpoint
    print(f"image tokenizer is loaded")

    # create and load gpt model
    precision = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.precision]
    latent_size = args.image_size // args.downsample_size
    gpt_model = GPT_models[args.gpt_model](
        vocab_size=args.codebook_size,
        block_size=latent_size ** 2,
        num_classes=args.num_classes,
        cls_token_num=args.cls_token_num,
        model_type=args.gpt_type,
    ).to(device=device, dtype=precision)
    
    checkpoint = torch.load(args.gpt_ckpt, map_location="cpu")
    if args.from_fsdp: # fspd
        model_weight = checkpoint
    elif "model" in checkpoint:  # ddp
        model_weight = checkpoint["model"]
    elif "module" in checkpoint: # deepspeed
        model_weight = checkpoint["module"]
    elif "state_dict" in checkpoint:
        model_weight = checkpoint["state_dict"]
    else:
        raise Exception("please check model weight, maybe add --from-fsdp to run command")
    gpt_model.load_state_dict(model_weight, strict=False)
    gpt_model.eval()
    del checkpoint
    print(f"gpt model is loaded")

    if args.compile:
        print(f"compiling the model...")
        gpt_model = torch.compile(
            gpt_model,
            mode="reduce-overhead",
            fullgraph=True
        ) # requires PyTorch 2.0 (optional)
    else:
        print(f"no need to compile model in demo") 

    output_class_samples = {}

    # Iterate through all ImageNet classes
    for class_id in range(args.num_classes): # Assuming ImageNet classes are 0 to num_classes-1
        print(f"Generating samples for class: {class_id}")
        # Generate N samples for each class
        for i in range(args.num_samples_per_class):
            # Set a totally random seed for each sample
            current_seed = random.randint(0, 2**32 - 1)
            torch.manual_seed(current_seed)
            print(f"  Generating sample {i+1}/{args.num_samples_per_class} with seed: {current_seed}")

            c_indices = torch.tensor([class_id], device=device)
            qzshape = [1, args.codebook_embed_dim, latent_size, latent_size] # Generate one sample at a time

            t1 = time.time()
            # The fix for the AttributeError: 'tuple' object has no attribute 'long'
            # Assuming 'generate' returns a tuple and the actual indices are the first element.
            generated_output = generate(
                gpt_model, c_indices, latent_size ** 2,
                cfg_scale=args.cfg_scale, cfg_interval=args.cfg_interval,
                temperature=args.temperature, top_k=args.top_k,
                top_p=args.top_p, sample_logits=True, 
            )
            index_sample = generated_output[0].long() # Access the first element and then convert to long()

            sampling_time = time.time() - t1
            print(f"  gpt sampling takes about {sampling_time:.2f} seconds.")    
            
            t2 = time.time()
            samples = vq_model.decode_code(index_sample, qzshape) # output value is between [-1, 1]
            decoder_time = time.time() - t2
            print(f"  decoder takes about {decoder_time:.2f} seconds.")

            # Store the sample in the dictionary
            sample_name = f"class{class_id}_{i+1}"
            output_class_samples[sample_name] = samples.cpu() # Move to CPU to store in dictionary

            # Optional: Save individual images to verify
            import os
            
            # Corrected line: Use an f-string for os.makedirs to properly include class_id
            output_dir_for_class = f"output_class/class{class_id}"
            os.makedirs(output_dir_for_class, exist_ok=True) 
            
            # Use the correctly formed output directory
            image_save_path = f"{output_dir_for_class}/{sample_name}.png"
            save_image(samples, image_save_path, normalize=True, value_range=(-1, 1))
            print(f"  Sample saved to {image_save_path}")
    
    print("\nGeneration complete. All samples are stored in 'output_class_samples' dictionary.")
    # You can now save output_class_samples to a file (e.g., using torch.save or pickle)
    # This will save a dictionary where keys are "classX_Y" and values are the generated image tensors.
    # import os
    # os.makedirs("generated_data", exist_ok=True)
    # torch.save(output_class_samples, "generated_data/imagenet_generated_samples.pt")
    # print("Dictionary saved to generated_data/imagenet_generated_samples.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpt-model", type=str, choices=list(GPT_models.keys()), default="GPT-XXL")
    parser.add_argument("--gpt-ckpt", type=str, default="./pretrained_models/c2i_XXL_384.pt")
    parser.add_argument("--gpt-type", type=str, choices=['c2i', 't2i'], default="c2i", help="class-conditional or text-conditional")
    parser.add_argument("--from-fsdp", action='store_true')
    parser.add_argument("--cls-token-num", type=int, default=1, help="max token number of condition input")
    parser.add_argument("--precision", type=str, default='bf16', choices=["none", "fp16", "bf16"]) 
    parser.add_argument("--compile", action='store_true', default=False)
    parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()), default="VQ-16")
    parser.add_argument("--vq-ckpt", type=str, default="./pretrained_models/vq_ds16_c2i.pt", help="ckpt path for vq model")
    parser.add_argument("--codebook-size", type=int, default=16384, help="codebook size for vector quantization")
    parser.add_argument("--codebook-embed-dim", type=int, default=8, help="codebook dimension for vector quantization")
    parser.add_argument("--image-size", type=int, choices=[256, 384, 512], default=384)
    parser.add_argument("--downsample-size", type=int, choices=[8, 16], default=16)
    parser.add_argument("--num-classes", type=int, default=1000) # ImageNet has 1000 classes
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--cfg-interval", type=float, default=-1)
    parser.add_argument("--seed", type=int, default=0, help="Initial seed (not used for individual sample generation)")
    parser.add_argument("--top-k", type=int, default=2000,help="top-k value to sample with")
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature value to sample with")
    parser.add_argument("--top-p", type=float, default=1.0, help="top-p value to sample with")
    parser.add_argument("--num-samples-per-class", type=int, default=10, help="Number of samples to generate per class")
    args = parser.parse_args()
    main(args)