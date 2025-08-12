# Modified from:
#   DiT:  https://github.com/facebookresearch/DiT/blob/main/sample.py
import os
import re
import time
import argparse
import random

import torch
from torchvision.utils import save_image

from tokenizer.tokenizer_image.vq_model import VQ_models
from autoregressive.models.gpt import GPT_models
from autoregressive.models.generate import generate

# PyTorch perf
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)


def main(args):
    # Setup PyTorch:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ----- image tokenizer (VQ) -----
    vq_model = VQ_models[args.vq_model](
        codebook_size=args.codebook_size,
        codebook_embed_dim=args.codebook_embed_dim
    ).to(device)
    vq_model.eval()
    checkpoint = torch.load(args.vq_ckpt, map_location="cpu")
    vq_model.load_state_dict(checkpoint["model"])
    del checkpoint
    print("image tokenizer is loaded")

    # ----- GPT model -----
    precision = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.precision]
    latent_size = args.image_size // args.downsample_size
    T = latent_size ** 2

    gpt_model = GPT_models[args.gpt_model](
        vocab_size=args.codebook_size,
        block_size=T,
        num_classes=args.num_classes,
        cls_token_num=args.cls_token_num,
        model_type=args.gpt_type,
    ).to(device=device, dtype=precision)

    checkpoint = torch.load(args.gpt_ckpt, map_location="cpu")
    if args.from_fsdp:
        model_weight = checkpoint
    elif "model" in checkpoint:
        model_weight = checkpoint["model"]
    elif "module" in checkpoint:
        model_weight = checkpoint["module"]
    elif "state_dict" in checkpoint:
        model_weight = checkpoint["state_dict"]
    else:
        raise Exception("please check model weight, maybe add --from-fsdp to run command")
    gpt_model.load_state_dict(model_weight, strict=False)
    gpt_model.eval()
    del checkpoint
    print("gpt model is loaded")

    if args.compile:
        print("compiling the model...")
        gpt_model = torch.compile(gpt_model, mode="reduce-overhead", fullgraph=True)
    else:
        print("no need to compile model in demo")

    output_class_samples = {}
    os.makedirs("output_adjust", exist_ok=True)

    # ===== Batched generation =====
    for class_id in range(args.num_classes):
        print(f"Generating samples for class: {class_id}")
        out_dir = f"output_adjust/{args.tau}"
        os.makedirs(out_dir, exist_ok=True)

        remaining = args.num_samples_per_class
        produced = 0

        while remaining > 0:
            bs = min(args.batch_size, remaining)

            # different randomness per batch
            torch.manual_seed(random.randint(0, 2**32 - 1))

            # class indices for the batch: shape [bs]
            c_indices = torch.full((bs,), class_id, device=device, dtype=torch.long)

            t1 = time.time()
            tau = args.tau
            priors_path = args.priors_path
            # generate() is expected to support batched c_indices and return (indices, ...)
            generated_output = generate(
                gpt_model, c_indices, T,
                cfg_scale=args.cfg_scale, cfg_interval=args.cfg_interval,
                temperature=args.temperature, top_k=args.top_k,
                top_p=args.top_p, sample_logits=True,
                tau=tau, priors_path=priors_path
            )
            index_sample = generated_output[0].long()  # [bs, T]
            sampling_time = time.time() - t1
            print(f"  gpt sampling took {sampling_time:.2f}s for batch size {bs}")

            # decode VQ codes → images
            qzshape = [bs, args.codebook_embed_dim, latent_size, latent_size]
            t2 = time.time()
            samples = vq_model.decode_code(index_sample, qzshape)  # [-1,1], [bs, C, H, W]
            decoder_time = time.time() - t2
            print(f"  decoder took {decoder_time:.2f}s for batch size {bs}")

            # save & collect
            for i in range(bs):
                k = produced + i
                sample_name = f"class{class_id}_{k+1}"
                image_save_path = f"{out_dir}/{sample_name}.png"
                save_image(samples[i:i+1], image_save_path, normalize=True, value_range=(-1, 1))
                output_class_samples[sample_name] = samples[i].detach().cpu()
                print(f"  Saved {image_save_path}")

            produced += bs
            remaining -= bs

    print("\nGeneration complete. All samples are stored in 'output_class_samples' dictionary.")
    # Example to persist:
    # os.makedirs("generated_data", exist_ok=True)
    # torch.save(output_class_samples, "generated_data/imagenet_generated_samples.pt")
    # print("Dictionary saved to generated_data/imagenet_generated_samples.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpt-model", type=str, choices=list(GPT_models.keys()), default="GPT-XXL")
    parser.add_argument("--gpt-ckpt", type=str, default="./pretrained_models/c2i_XXL_384.pt")
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--priors_path", type=str, default="codebook_probs/imagenet.pt")
    parser.add_argument("--gpt-type", type=str, choices=['c2i', 't2i'], default="c2i",
                        help="class-conditional or text-conditional")
    parser.add_argument("--from-fsdp", action='store_true')
    parser.add_argument("--cls-token-num", type=int, default=1, help="max token number of condition input")
    parser.add_argument("--precision", type=str, default='bf16', choices=["none", "fp16", "bf16"])
    parser.add_argument("--compile", action='store_true', default=False)

    parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()), default="VQ-16")
    parser.add_argument("--vq-ckpt", type=str, default="./pretrained_models/vq_ds16_c2i.pt",
                        help="ckpt path for vq model")
    parser.add_argument("--codebook-size", type=int, default=16384, help="codebook size for vector quantization")
    parser.add_argument("--codebook-embed-dim", type=int, default=8,
                        help="codebook dimension for vector quantization")

    parser.add_argument("--image-size", type=int, choices=[256, 384, 512], default=384)
    parser.add_argument("--downsample-size", type=int, choices=[8, 16], default=16)
    parser.add_argument("--num-classes", type=int, default=1000)

    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--cfg-interval", type=float, default=-1)
    parser.add_argument("--seed", type=int, default=0,
                        help="Initial seed (unused per-image; batching uses random seeds per batch)")
    parser.add_argument("--top-k", type=int, default=2000, help="top-k value to sample with")
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature value to sample with")
    parser.add_argument("--top-p", type=float, default=1.0, help="top-p value to sample with")

    parser.add_argument("--num-samples-per-class", type=int, default=2,
                        help="Number of samples to generate per class")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for batched sampling")

    args = parser.parse_args()
    main(args)
