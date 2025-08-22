# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""
import torch
import wandb
import pdb
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os
import json
from torchvision.utils import save_image

from models import DiT_models
from diffusion import create_diffusion,from_patch_seq_last, ImageConverter
from diffusers.models import AutoencoderKL


#################################################################################
#                             Training Helper Functions                         #
#################################################################################

import random

def training_scheduler(train_step, total_steps):
    """
    训练调度函数：根据当前步数，抽签决定 full-seq / KSU / rollout K
    
    Args:
        train_step (int): 当前训练步
        total_steps (int): 总训练步数 (如 16500)
    
    Returns:
        dict: {
            "mode": "fullseq" or "ksu",
            "K": int (仅 KSU 时有意义),
            "enable_rollout": bool
        }
    """
    p_ss, p_ss_near = 0.0, 0.0
    # ===== 阶段划分 =====
    ratio = train_step / total_steps
    if ratio < 0.3:          # 前30%
        fullseq_prob = 0.9
        ksu_prob = 0.1
        K = 4
        ksu_batch_prob = 0.5 + 0.2 * (ratio / 0.3)  # 0.5 -> 0.7
    elif ratio < 0.7:        # 中间40%
        fullseq_prob = 0.5
        ksu_prob = 0.5
        K = 8
        ksu_batch_prob = 0.7 + 0.2 * ((ratio - 0.3) / 0.4)  # 0.7 -> 0.9
    else:                    # 最后30%
        fullseq_prob = 0.3
        ksu_prob = 0.7
        K = 16
        ksu_batch_prob = 0.9 + 0.1 * ((ratio - 0.7) / 0.3)  # 0.9 -> 1.0
    
    # ===== 第一次抽签：决定模式 =====
    if random.random() < fullseq_prob:
        return {"mode": "fullseq", "K": 0, "enable_rollout": False,"p_ss":p_ss,"p_ss_near":p_ss_near}
    
    # ===== 第二次抽签：进入 KSU，决定是否 rollout =====
    enable_rollout = random.random() < ksu_batch_prob

    ss_end_step = int(0.3 * total_steps)   # 前 30% 步数内线性涨满
    ss_p_end    = 0.4
    p_ss = linear_schedule(
        step=train_step,
        start_step=0,
        end_step=ss_end_step,
        start_val=0.0,
        end_val=ss_p_end
    )
    near_frame_boost = 1.25
    p_ss_near = min(0.5, p_ss * near_frame_boost)

    if enable_rollout:
        return {"mode": "ksu", "K": K, "enable_rollout": True,"p_ss":p_ss,"p_ss_near":p_ss_near}
    else:
        return {"mode": "ksu", "K": 1, "enable_rollout": False,"p_ss":p_ss,"p_ss_near":p_ss_near}


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def check_trainable_parameters(model, logger=None):
    """
    Check and log which parameters are trainable in the model.
    """
    total_params = 0
    trainable_params = 0
    frozen_params = 0
    
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
        else:
            frozen_params += param.numel()
            if logger:
                logger.info(f"Frozen parameter: {name} (shape: {param.shape})")
    
    if logger:
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Frozen parameters: {frozen_params:,}")
        logger.info(f"Trainable ratio: {trainable_params/total_params*100:.2f}%")
    
    return trainable_params, frozen_params, total_params




def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


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


#################################################################################
#                                  Training Loop                                #
#################################################################################
def linear_schedule(step, start_step, end_step, start_val, end_val):
    if end_step <= start_step:
        return end_val
    if step <= start_step:
        return start_val
    if step >= end_step:
        return end_val
    r = (step - start_step) / float(end_step - start_step)
    return start_val + r * (end_val - start_val)

def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    dist.init_process_group("nccl")
    #pdb.set_trace()
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
        with open(f"{experiment_dir}/training_args.json", 'w') as f:
            json.dump(vars(args), f, indent=4)
        custom_logger_setting = {
            "folder": experiment_dir,
            "log_every": args.detailed_log_every,
            "pic_print": args.detailed_log_pic_print,
            "middle_vars_print": args.detailed_log_middle_vars_print,
            "loss_analysis": args.detailed_log_loss_analysis
        }
    else:
        logger = create_logger(None)
        custom_logger_setting = None
    
    wandb_run = None
    if args.use_wandb and dist.get_rank() == 0:
        init_kwargs = dict(
            project=args.wandb_project,
            entity=args.wandb_team,
            name=args.wandb_name,
            config=vars(args),
        )
        if args.wandb_id is not None:
            init_kwargs.update(id=args.wandb_id, resume=args.wandb_resume)

        wandb_run = wandb.init(**init_kwargs)
         
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    )
    #ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    #requires_grad(ema, False)
    
    custom_training_settings = {
        "fixed_sequence": args.fixed_sequence,  
        "use_real_target": args.use_real_target,
        "predict_eos_patch": args.predict_eos_patch,
        "loss_type": args.loss_type
    }
    

    vae = AutoencoderKL.from_pretrained(args.vae_path).to(device)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    image_converter = ImageConverter(h=latent_size, w=latent_size, patch_size=model.patch_size, vae=vae)
    model = DDP(model.to(device), device_ids=[device])
    diffusion = create_diffusion(timestep_respacing="",diffusion_steps=1000,image_converter=image_converter, training_settings = custom_training_settings, use_kl = args.loss_type == "KL")
    # vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-mse").to(device)
    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    
    # 检查模型参数的可训练性
    logger.info("=== DiT Model Parameter Analysis ===")
    check_trainable_parameters(model.module, logger)  # 使用model.module来访问原始模型
    
    # 确保所有参数都是可训练的
    requires_grad(model, True)
    logger.info("=== After ensuring all parameters are trainable ===")
    check_trainable_parameters(model.module, logger)

    requires_grad(vae, False)


    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)
    
    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()
    start_epoch = 0

    # ===== Resume from checkpoint (optional) =====
    if args.resume_ckpt is not None:
        map_loc = {"cuda:%d" % 0: "cuda:%d" % device}  # 兼容跨卡加载
        ckpt = torch.load(args.resume_ckpt, map_location=lambda storage, loc: storage.cuda(device))
        model.module.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])

        # 恢复步数/epoch（如果老 ckpt 没存，就从文件名猜一把）
        train_steps = int(ckpt.get("train_steps", 0))
        start_epoch = int(ckpt.get("epoch", 0))
        if args.resume_epoch is not None:
            start_epoch = int(args.resume_epoch)
        if args.resume_step is not None:
            train_steps = int(args.resume_step)

        if train_steps == 0:
            # 例如 checkpoints/0001000.pt -> 1000
            try:
                import re
                fname = os.path.basename(args.resume_ckpt)
                m = re.search(r"(\d+)\.pt$", fname)
                if m:
                    train_steps = int(m.group(1))
            except Exception:
                pass

        if dist.get_rank() == 0:
            logger.info(f"Resumed from {args.resume_ckpt} (epoch={start_epoch}, train_steps={train_steps})")
            if wandb_run is not None:
                wandb.run.summary["resumed_from_ckpt"] = args.resume_ckpt
                wandb.run.summary["resumed_from_epoch"] = start_epoch
                wandb.run.summary["resumed_from_step"] = train_steps
    if wandb_run is not None:
        # 用一个统一的 step 作为横轴
        wandb.define_metric("global_step")
        wandb.define_metric("train/*", step_metric="global_step")
        wandb.define_metric("val/*",   step_metric="global_step")


    # Setup data:
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])

    dataset = ImageFolder(args.data_path, transform=transform)

    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")
    #update_ema(ema, model.module, decay=0)
    # Prepare models for training:
    model.train()
    #ema.eval()
    print("new ss ksu")
    # Variables for monitoring/logging purposes:
    total_steps=len(dataset)//args.global_batch_size * args.epochs
    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            with torch.no_grad():
                latent = vae.encode(x).latent_dist.sample().mul_(0.18215)
            rand_t = args.rand_t
            t = torch.full((latent.shape[0],), rand_t, device=device, dtype=torch.long)
            if args.use_ss:
                decision = training_scheduler(train_steps, total_steps)
                mode=decision["mode"]
                k=decision["K"]
                enable_rollout=decision["enable_rollout"]
                p_ss = decision["p_ss"]
                p_ss_near = decision["p_ss_near"]
                return_last= False
                model_kwargs = dict(y=y,return_last=return_last) 
                # 计算当前batch的损失
                opt.zero_grad(set_to_none=True)
                if mode=="fullseq":
                    loss_dict = diffusion.training_losses(model, latent, t, custom_logger_setting, vae, model_kwargs)
                elif mode=="ksu":
                    ss_settings = {
                        "ksu":{
                            "k":k,
                            "enable_rollout":enable_rollout,
                            "num_t0":6
                        },
                        "ss": {
                            "p": float(p_ss),
                            "p_near": float(p_ss_near),
                        }
                    }

                    loss_dict = diffusion.ss_training_losses(model, latent, t, custom_logger_setting, vae, model_kwargs,ss_settings)
            else:
                loss_dict = diffusion.training_losses(model, latent, t, custom_logger_setting, vae, model_kwargs)
            loss = loss_dict["loss"].mean()
            loss.backward()
            opt.step()
            #update_ema(ema, model.module)
            # 记录损失值
            running_loss += loss.detach().item()
            log_steps += 1
            train_steps += 1

            if train_steps % args.log_every == 0:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)

                # 对所有进程减少损失历史
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")

                # W&B 记录（可选）
                if dist.get_rank() == 0 and wandb_run is not None:
                    wandb.log({
                        "train/loss_avg": float(avg_loss),
                        "train/steps_per_sec": float(steps_per_sec),
                        "global_step": train_steps
                    }, step=train_steps)

                # 重置监控变量
                running_loss = 0
                log_steps = 0
                start_time = time()

            # 保存模型检查点（可选）
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args,
                        "train_steps": train_steps,
                        "epoch": epoch
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                    if wandb_run is not None:
                        wandb.run.summary["last_ckpt_path"] = checkpoint_path
                dist.barrier()

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...
    if wandb_run is not None and dist.get_rank() == 0:
        wandb.run.summary["final_train_steps"] = train_steps
        wandb.finish()
    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-S/8")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--global-batch-size", type=int, default=8)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--ckpt-every", type=int, default=100)
    parser.add_argument("--use-wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="your_project")
    parser.add_argument("--wandb-team",type=str,default="nju-ics")
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--wandb-id", type=str, default=None, help="Set to resume the same W&B run id")
    parser.add_argument("--wandb-resume", type=str, choices=["never","allow","must"], default="never")   
    parser.add_argument("--resume-ckpt", type=str, default=None, help="Path to a checkpoint to resume from")
    parser.add_argument("--resume-epoch", type=int, default=None, help="Resume from a specific epoch")
    parser.add_argument("--resume-step", type=int, default=None, help="Resume from a specific step")
    parser.add_argument("--description", type=str, default=None, help="Describe what the turn is training for")
    parser.add_argument("--vae-path", type=str, default="stabilityai/sd-vae-ft-ema", help="Describe where vae model from")
    parser.add_argument("--detailed-log-every", type=int, default=100)
    parser.add_argument("--detailed-log-pic-print", action="store_true")
    parser.add_argument("--detailed-log-middle-vars-print", action="store_true")
    parser.add_argument("--detailed-log-loss-analysis", action="store_true")
    parser.add_argument("--fixed-sequence", action="store_true")
    parser.add_argument("--use-real-target", action="store_true")
    parser.add_argument("--predict-eos-patch", action="store_true")
    parser.add_argument("--rand-t", type=int, default=20)
    parser.add_argument("--loss-type", type=str, default="MSE")
    parser.add_argument("--use_ss",action="store_true")
    parser.add_argument("--ksu_k_max", type=int, default=16)
    parser.add_argument("--ss_near_boost", type=float, default=1.25)
    args = parser.parse_args()
    main(args)