import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)     # disable default parameter init for faster speed
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)  # disable default parameter init for faster speed
from torchvision.utils import save_image

import os
import time
import argparse
import re
from datetime import datetime
from tokenizer.tokenizer_image.vq_model import VQ_models
from language.t5 import T5Embedder
from autoregressive.models.gpt import GPT_models
from autoregressive.models.generate import generate
import json
from tqdm import tqdm
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def sanitize_filename(text):
    """
    Cleans a string to be a valid, short filename.
    """
    # Remove non-alphanumeric characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Replace spaces with underscores and shorten
    return text.replace(' ', '_')[:100]


def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # # MODIFICATION: Create a timestamped output directory
    # timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # output_path = os.path.join("output", timestamp)
    # os.makedirs(output_path, exist_ok=True)
    # print(f"Saving generated images to: {output_path}")

    folder_name = f"temperature={args.temperature}"
    output_path = os.path.join("output", folder_name)
    os.makedirs(output_path, exist_ok=True)
    print(f"Saving generated images to: {output_path}")

    # Save the configuration (args) to a JSON file
    config_path = os.path.join(output_path, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        # Convert the args namespace to a dictionary and save it
        json.dump(vars(args), f, indent=4)
    print(f"Configuration saved to: {config_path}")

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
        block_size=latent_size ** 2,
        cls_token_num=args.cls_token_num,
        model_type=args.gpt_type,
    ).to(device=device, dtype=precision)

    checkpoint = torch.load(args.gpt_ckpt, map_location="cpu")
 
    if "model" in checkpoint:  # ddp
        model_weight = checkpoint["model"]
    elif "module" in checkpoint: # deepspeed
        model_weight = checkpoint["module"]
    elif "state_dict" in checkpoint:
        model_weight = checkpoint["state_dict"]
    else:
        raise Exception("please check model weight")
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
    
    assert os.path.exists(args.t5_path)
    t5_model = T5Embedder(
        device=device, 
        local_cache=True, 
        cache_dir=args.t5_path, 
        dir_or_name=args.t5_model_type,
        torch_dtype=precision,
        model_max_length=args.t5_feature_max_len,
    )

    # prompts = [
    # "A portrait photo of a kangaroo wearing an orange hoodie and blue sunglasses standing on the grassin front of the Sydney Opera House holding a sign on the chest that says Welcome Friends!",
    # "A blue Porsche 356 parked in front of a yellow brick wall.",
    # "A photo of an astronaut riding a horse in the forest. There is a river in front of them with water lilies.",
    # "A map of the United States made out of sushi. It is on a table next to a glass of red wine."
    # ]

    filename = "download/cleaned_prompts.txt"
    try:
        with open(filename, "r", encoding="utf-8") as f:
            prompts = [line.strip() for line in f]
        
        print(f"Successfully loaded {len(prompts)} prompts from '{filename}'.")

    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
        print("Please make sure the text file is in the same directory as your Python script.")
        prompts = []

    if not prompts:
        print("No prompts to process. Exiting.")
        return
        
    all_samples = []
    total_sampling_time = 0
    
    num_batches = (len(prompts) + args.batch_size - 1) // args.batch_size
    
    for i in tqdm(range(0, len(prompts), args.batch_size), desc="Processing Batches"):
        prompt_batch = prompts[i:i + args.batch_size]
        print(f"\n--- Processing Batch {i // args.batch_size + 1}/{num_batches} (size: {len(prompt_batch)}) ---")

        caption_embs, emb_masks = t5_model.get_text_embeddings(prompt_batch)

        if not args.no_left_padding:
            new_emb_masks = torch.flip(emb_masks, dims=[-1])
            new_caption_embs = []
            for idx, (caption_emb, emb_mask) in enumerate(zip(caption_embs, emb_masks)):
                valid_num = int(emb_mask.sum().item())
                new_caption_emb = torch.cat([caption_emb[valid_num:], caption_emb[:valid_num]])
                new_caption_embs.append(new_caption_emb)
            new_caption_embs = torch.stack(new_caption_embs)
        else:
            new_caption_embs, new_emb_masks = caption_embs, emb_masks
        
        c_indices = new_caption_embs * new_emb_masks[:,:, None]
        c_emb_masks = new_emb_masks

        qzshape = [len(c_indices), args.codebook_embed_dim, latent_size, latent_size]
        t1 = time.time()
        index_sample = generate(
            gpt_model, c_indices, latent_size ** 2, 
            c_emb_masks, 
            cfg_scale=args.cfg_scale,
            temperature=args.temperature, top_k=args.top_k,
            top_p=args.top_p, sample_logits=True, 
        )
        sampling_time = time.time() - t1
        total_sampling_time += sampling_time
        print(f"Batch sampling took about {sampling_time:.2f} seconds.")    
        
        samples = vq_model.decode_code(index_sample, qzshape)
        
        # This code goes inside your 'for i in range(...)' loop

        samples_path = os.path.join(output_path, "samples")
        os.makedirs(samples_path, exist_ok=True)

        for j, sample in enumerate(samples):
            global_idx = i + j
            prompt_text = prompt_batch[j]
            sanitized_prompt = sanitize_filename(prompt_text)
            
            # Create the specific filename for this image
            img_filename = f"{global_idx:04d}_{sanitized_prompt}.png"
            
            # Join the 'samples' directory path and the filename to create the full path
            img_path = os.path.join(samples_path, img_filename)
            
            # Save the image using the correct full path
            save_image(sample, img_path, normalize=True, value_range=(-1, 1))

        print(f"Saved {len(samples)} individual images to '{samples_path}'.")

        all_samples.append(samples)

    final_samples = torch.cat(all_samples, dim=0)
    
    print("\n--- All Batches Processed ---")
    print(f"Total sampling time: {total_sampling_time:.2f} seconds.")

    # MODIFICATION: Save a grid of the first 4 images
    if len(final_samples) > 0:
        num_grid_images = min(4, len(final_samples))
        grid_samples = final_samples[:num_grid_images]
        grid_path = os.path.join(output_path, "grid_first_4.png")
        
        # Make a 2x2 grid if we have 4 images, otherwise a single row
        grid_nrow = 2 if num_grid_images == 4 else num_grid_images
        
        save_image(grid_samples, grid_path, nrow=grid_nrow, normalize=True, value_range=(-1, 1))
        print(f"Saved a grid of the first {num_grid_images} images to '{grid_path}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--t5-path", type=str, default='pretrained_models/t5-ckpt')
    parser.add_argument("--t5-model-type", type=str, default='flan-t5-xl')
    parser.add_argument("--t5-feature-max-len", type=int, default=120)
    parser.add_argument("--t5-feature-dim", type=int, default=2048)
    parser.add_argument("--no-left-padding", action='store_true', default=False)
    parser.add_argument("--gpt-model", type=str, choices=list(GPT_models.keys()), default="GPT-XL")
    parser.add_argument("--gpt-ckpt", type=str, default="./pretrained_models/t2i_XL_stage1_256.pt")
    parser.add_argument("--gpt-type", type=str, choices=['c2i', 't2i'], default="t2i", help="class->image or text->image")  
    parser.add_argument("--cls-token-num", type=int, default=120, help="max token number of condition input")
    parser.add_argument("--precision", type=str, default='bf16', choices=["none", "fp16", "bf16"]) 
    parser.add_argument("--compile", action='store_true', default=False)
    parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()), default="VQ-16")
    parser.add_argument("--vq-ckpt", type=str, default="./pretrained_models/vq_ds16_t2i.pt", help="ckpt path for vq model")
    parser.add_argument("--codebook-size", type=int, default=16384, help="codebook size for vector quantization")
    parser.add_argument("--codebook-embed-dim", type=int, default=8, help="codebook dimension for vector quantization")
    parser.add_argument("--image-size", type=int, choices=[256, 384, 512], default=256)
    parser.add_argument("--downsample-size", type=int, choices=[8, 16], default=16)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=1000, help="top-k value to sample with")
    parser.add_argument("--temperature", type=float, default=2.0, help="temperature value to sample with")
    parser.add_argument("--top-p", type=float, default=1.0, help="top-p value to sample with")
    
    parser.add_argument("--batch-size", type=int, default=16, help="Number of prompts to process at once.")
    
    args = parser.parse_args()
    main(args)