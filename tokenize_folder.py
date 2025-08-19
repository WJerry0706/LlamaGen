# Modified from:
#   fast-DiT: https://github.com/chuanyangjin/fast-DiT/blob/main/extract_features.py
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from PIL import Image
import numpy as np
import argparse
import os

from utils.distributed import init_distributed_mode
from dataset.augmentation import center_crop_arr
# from dataset.build import build_dataset  <- 我们不再需要这个
from tokenizer.tokenizer_image.vq_model import VQ_models

#################################################################################
#                      新增的自定义Dataset类 (START)                          #
#################################################################################
class FlatImageFolderDataset(Dataset):
    """
    一个自定义的数据集类，用于加载扁平文件夹结构中的所有图像。
    它会直接读取 `root` 目录下的所有图片，忽略任何子目录。
    """
    IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')

    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.samples = self._find_images(root)
        if not self.samples:
            raise FileNotFoundError(f"在目录 {root} 下没有找到任何支持的图片文件")

    def _find_images(self, dir_path):
        paths = []
        for filename in sorted(os.listdir(dir_path)):
            if filename.lower().endswith(self.IMG_EXTENSIONS):
                path = os.path.join(dir_path, filename)
                if os.path.isfile(path):
                    paths.append(path)
        return paths

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path = self.samples[index]
        try:
            with Image.open(path) as img:
                img = img.convert("RGB")
        except Exception as e:
            print(f"警告：无法加载图片 {path}，跳过。错误: {e}")
            # 返回一个占位符，或者处理下一个
            return self.__getitem__((index + 1) % len(self))

        if self.transform:
            img = self.transform(img)
        
        # 因为没有类别文件夹，我们为所有图片返回一个虚拟标签 0
        label = 0
        return img, torch.tensor(label)

#################################################################################
#                      新增的自定义Dataset类 (END)                            #
#################################################################################

#################################################################################
#                                  主函数循环                                 #
#################################################################################
def main(args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    # Setup DDP:
    if not args.debug:
        init_distributed_mode(args)
        rank = dist.get_rank()
        device = rank % torch.cuda.device_count()
        seed = args.global_seed * dist.get_world_size() + rank
        torch.manual_seed(seed)
        torch.cuda.set_device(device)
        print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")
    else:
        device = 'cuda'
        rank = 0
    
    # Setup a feature folder:
    if args.debug or rank == 0:
        os.makedirs(args.code_path, exist_ok=True)
        # 文件夹名称现在更通用
        codes_dir = os.path.join(args.code_path, f'{os.path.basename(args.data_path)}_{args.image_size}_codes')
        labels_dir = os.path.join(args.code_path, f'{os.path.basename(args.data_path)}_{args.image_size}_labels')
        os.makedirs(codes_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)
        print(f"代码将保存到: {codes_dir}")
        print(f"标签将保存到: {labels_dir}")

    # create and load model
    vq_model = VQ_models[args.vq_model](
        codebook_size=args.codebook_size,
        codebook_embed_dim=args.codebook_embed_dim)
    vq_model.to(device)
    vq_model.eval()
    checkpoint = torch.load(args.vq_ckpt, map_location="cpu")
    vq_model.load_state_dict(checkpoint["model"])
    del checkpoint

    # Setup data:
    if args.ten_crop:
        crop_size = int(args.image_size * args.crop_range)
        transform = transforms.Compose([
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, crop_size)),
            transforms.TenCrop(args.image_size), # this is a tuple of PIL Images
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])), # returns a 4D tensor
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
    else:
        crop_size = args.image_size 
        transform = transforms.Compose([
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, crop_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
    
    # ==================== 主要修改点 ====================
    # 使用我们新的 FlatImageFolderDataset 类替换原来的 build_dataset
    # dataset = build_dataset(args, transform=transform) # 旧代码
    dataset = FlatImageFolderDataset(root=args.data_path, transform=transform) # 新代码
    print(f"从 {args.data_path} 中找到 {len(dataset)} 张图片。")
    # ====================================================

    if not args.debug:
        sampler = DistributedSampler(
            dataset,
            num_replicas=dist.get_world_size(),
            rank=rank,
            shuffle=False,
            seed=args.global_seed
        )
    else:
        sampler = None
    loader = DataLoader(
        dataset,
        batch_size=1, # important!
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )

    total = 0
    print("开始处理图片...")
    for x, y in loader:
        x = x.to(device)
        if args.ten_crop:
            x_all = x.flatten(0, 1)
            num_aug = 10
        else:
            x_flip = torch.flip(x, dims=[-1])
            x_all = torch.cat([x, x_flip])
            num_aug = 2
        y = y.to(device)
        with torch.no_grad():
            _, _, [_, _, indices] = vq_model.encode(x_all)
        codes = indices.reshape(x.shape[0], num_aug, -1)

        x = codes.detach().cpu().numpy()    # (1, num_aug, args.image_size//16 * args.image_size//16)
        train_steps = rank + total
        np.save(os.path.join(codes_dir, f'{train_steps}.npy'), x)

        y = y.detach().cpu().numpy()    # (1,)
        np.save(os.path.join(labels_dir, f'{train_steps}.npy'), y)
        
        if not args.debug:
            total += dist.get_world_size()
        else:
            total += 1
        
        if rank == 0 and total % 100 == 0:
            print(f"已处理 {total}/{len(loader) * (dist.get_world_size() if not args.debug else 1)} 张图片")

    print("所有图片处理完毕。")
    if not args.debug:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True, help="包含图片的文件夹路径 (扁平结构)")
    parser.add_argument("--code-path", type=str, default='output_codes', required=False, help="保存npy文件的输出文件夹")
    parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()), default="VQ-16")
    parser.add_argument("--vq-ckpt", type=str, default='pretrained_models/vq_ds16_c2i.pt', required=False, help="vq model的ckpt路径")
    parser.add_argument("--codebook-size", type=int, default=16384)
    parser.add_argument("--codebook-embed-dim", type=int, default=8)
    # --dataset 参数不再需要，但保留以防其他地方使用
    # parser.add_argument("--dataset", type=str, default='imagenet') 
    parser.add_argument("--image-size", type=int, choices=[256, 384, 448, 512], default=256)
    parser.add_argument("--ten-crop", action='store_true')
    parser.add_argument("--crop-range", type=float, default=1.1)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--debug", action='store_true')
    args = parser.parse_args()
    main(args)