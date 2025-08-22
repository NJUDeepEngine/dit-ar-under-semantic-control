# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained DiT.
"""
import torch
import pdb
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusion import create_diffusion,from_patch_seq_last,to_patch_seq_single,to_patch_seq_all
from diffusers.models import AutoencoderKL
from download import find_model
from models import DiT_models
import argparse
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
from PIL import Image
import numpy as np

def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
        # Setup DDP:
    dist.init_process_group("nccl")
    #pdb.set_trace()
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    
    x = torch.load("/data3/lmy/ss_ksu-check/011-DiT-L-4/8000/x_t_all.pt", map_location="cpu")

    # 取出 x 和 y
    y = torch.load("/data3/lmy/ss_ksu-check/011-DiT-L-4/8000/y.pt", map_location="cpu")  # (B,)
    x=x.to(device)
    y=y.to(device)
    #x=x[:,:1]
    #x_t_all=to_patch_seq_all(x,args.patch_size)
    x = x.to(device)
    x=to_patch_seq_all(x,args.patch_size)
    #pdb.set_trace()
    #class_labels = [10,100,200,300,400,500,600,700,800,900]
    #y = torch.tensor(class_labels, device=device)
    #pdb.set_trace()
    
    if args.ckpt is None:
        assert args.model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000

    # Load model:
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    ).to(device)
    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()  # important!
    # directly use ddim
    diffusion = create_diffusion(str(args.num_sampling_steps))
    
    #vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    # Labels to condition the model with (feel free to change):
    #class_labels = [0]

    # Create sampling noise:
    #n = len(class_labels)
    #z = torch.randn(n, 4, latent_size, latent_size, device=device)
    #x=to_patch_seq_single(z,args.patch_size)
    num_patches=(latent_size//args.patch_size)**2
    max_gen_len=num_patches*args.max_step
    #y = torch.tensor(class_labels, device=device)

    # Setup classifier-free guidance:
    #z = torch.cat([z, z], 0)
    #z_start=to_patch_seq_single(x,args.patch_size)
    z_start=x.clone()
    print(z_start.shape)
    #pdb.set_trace()
    
    #pdb.set_trace()
    #y_null = torch.tensor([1000] * n, device=device)
    #y = torch.cat([y, y_null], 0)
    #model_kwargs = dict(y=y, is_training=False,cfg_scale=args.cfg_scale)
    model_kwargs = dict(y=y, return_last=True)
    # Sample images:
    #samples = diffusion.p_sample_loop(
    #    model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
    #)
    #pdb.set_trace()
    #samples = diffusion.ar_sample_test(model.forward,z_start.shape,z_start,max_gen_len=max_gen_len,clip_denoised=False,model_kwargs=model_kwargs,device=device,num_patch=num_patches,vae_path="/data0/lmy/dit-ar-under-semantic-control/vae")
    samples = diffusion.ar_sample_test(model.forward,z_start.shape,z_start,max_gen_len=max_gen_len,clip_denoised=False,model_kwargs=model_kwargs,device=device,num_patch=num_patches,vae_path="/data0/lmy/dit-ar-under-semantic-control/vae")
   
    #samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
    #pdb.set_trace()
    samples = from_patch_seq_last(samples, patch_size=args.patch_size, patch_num=num_patches, out_channels=args.out_channels)
    samples = vae.decode(samples / 0.18215).sample

    # Save and display images:
    save_image(samples, "sample.png", nrow=4, normalize=True, value_range=(-1, 1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    parser.add_argument("--patch-size",type=int,default=4)
    parser.add_argument("--max-step",type=int,default=20)
    parser.add_argument("--out-channels",type=int,default=4)
    #parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--global-batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--global-seed", type=int, default=0)
    args = parser.parse_args()
    main(args)
