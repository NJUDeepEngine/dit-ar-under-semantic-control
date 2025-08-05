# Modified from OpenAI's diffusion repos
#     GLIDE: https://github.com/openai/glide-text2im/blob/main/glide_text2im/gaussian_diffusion.py
#     ADM:   https://github.com/openai/guided-diffusion/blob/main/guided_diffusion
#     IDDPM: https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py

import torch as th
import numpy as np


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL divergence between two gaussians.
    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, th.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for th.exp().
    logvar1, logvar2 = [
        x if isinstance(x, th.Tensor) else th.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + th.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * th.exp(-logvar2)
    )


def approx_standard_normal_cdf(x):
    """
    A fast approximation of the cumulative distribution function of the
    standard normal.
    """
    return 0.5 * (1.0 + th.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * th.pow(x, 3))))


def continuous_gaussian_log_likelihood(x, *, means, log_scales):
    """
    Compute the log-likelihood of a continuous Gaussian distribution.
    :param x: the targets
    :param means: the Gaussian mean Tensor.
    :param log_scales: the Gaussian log stddev Tensor.
    :return: a tensor like x of log probabilities (in nats).
    """
    centered_x = x - means
    inv_stdv = th.exp(-log_scales)
    normalized_x = centered_x * inv_stdv
    log_probs = th.distributions.Normal(th.zeros_like(x), th.ones_like(x)).log_prob(normalized_x)
    return log_probs


def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    """
    Compute the log-likelihood of a Gaussian distribution discretizing to a
    given image.
    :param x: the target images. It is assumed that this was uint8 values,
              rescaled to the range [-1, 1].
    :param means: the Gaussian mean Tensor.
    :param log_scales: the Gaussian log stddev Tensor.
    :return: a tensor like x of log probabilities (in nats).
    """
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = th.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = th.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = th.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = th.where(
        x < -0.999,
        log_cdf_plus,
        th.where(x > 0.999, log_one_minus_cdf_min, th.log(cdf_delta.clamp(min=1e-12))),
    )
    assert log_probs.shape == x.shape
    return log_probs

# Dataset 或 dataloader 里
# 原始数据: x (B, T, C, H, W)

def to_patch_seq(x, patch_size):
    B, T, C, H, W = x.shape
    assert H % patch_size == 0 and W % patch_size == 0
    h, w = H // patch_size, W // patch_size

    x = x.reshape(B, T, C, h, patch_size, w, patch_size)  # 分 patch
    x = x.permute(0, 1, 3, 5, 2, 4, 6)  # B, T, h, w, C, p, p
    x = x.reshape(B, T * h * w, C, patch_size, patch_size)  # B, T*Num_Patch, C, P, P
    return x

def from_patch_seq(patch_seq, eos_token, patch_size, patch_num, out_channels):
    """
    Args:
        patch_seq: (B, L, C, P, P)
        eos_token: (C, P, P)
        patch_size: int
        patch_num: 每张图的 patch 数（int），必须是平方数（即 patch_num = h * w）
        out_channels: 通道数 C
    Returns:
        imgs: (B, C, H, W)，按有效 patch 解码的图像
    """
    B, L, C, P, _ = patch_seq.shape
    assert patch_num <= L
    assert P == patch_size
    assert C == out_channels

    h = w = int(patch_num ** 0.5)
    assert h * w == patch_num, "patch_num 必须是平方数"

    eos_mask = (patch_seq == eos_token.view(1, 1, C, P, P)).all(dim=(2, 3, 4))  # (B, L)
    eos_pos = eos_mask.float().argmax(dim=1)  # (B,)
    no_eos = (eos_mask.sum(dim=1) == 0)
    eos_pos[no_eos] = patch_seq.shape[1]

    imgs = []
    for i in range(B):
        valid_end = eos_pos[i].item()
        valid_start = max(0, valid_end - patch_num)
        x = patch_seq[i, valid_start:valid_end]  # 向前回溯 patch_num 个

        # Padding 使长度为 patch_num
        valid_len = valid_end - valid_start
        pad_len = patch_num - valid_len
        if pad_len > 0:
            pad = th.zeros(pad_len, C, P, P, device=x.device, dtype=x.dtype)
            x = th.cat([pad, x], dim=0)  # pad 在前面补齐

        x = x.view(h, w, C, P, P)        # (h, w, C, P, P)
        x = x.permute(2, 0, 3, 1, 4)     # (C, h, P, w, P)
        x = x.reshape(C, h * P, w * P)   # (C, H, W)
        imgs.append(x)

    imgs = th.stack(imgs, dim=0)  # (B, C, H, W)
    return imgs

def timesteps_padding(t, mapper = None) -> dict:
    """
    对每张图像，根据其当前时间步 t_i，构建从 1000 步压缩到 t_i+1 的映射表，
    并将 [t_i, ..., 0] 映射回原始 1000 步空间下的时间戳。

    Args:
        t (Tensor): shape (B,) 当前 batch 每张图像的时间步数（压缩后）
        mapper: 映射方式

    Returns:
        dict:
            - list_timesteps: List[List[int]]，每张图像的原始时间戳序列（降序）
            - patched_timesteps: Tensor (B, max_t+1)，右对齐，左边填0
    """
    batch_size = t.shape[0]
    device = t.device
    max_t = t.max().item()

    list_timesteps = []
    patched_timesteps = []

    if mapper is None:
        mapper = lambda x: x

    for i in range(batch_size):
        t_i = t[i].item()

        mapped = [mapper(t_i)[j] for j in range(t_i, -1, -1)]  # 反向

        list_timesteps.append(mapped)

        # 3. 右对齐，左边补0，使所有样本同长度
        pad_len = max_t + 1 - len(mapped)
        padded = [-1] * pad_len + mapped
        patched_timesteps.append(padded)

    patched_timesteps = th.tensor(patched_timesteps, dtype=th.long, device=device)

    return {
        "list_timesteps": list_timesteps,
        "patched_timesteps": patched_timesteps
    }