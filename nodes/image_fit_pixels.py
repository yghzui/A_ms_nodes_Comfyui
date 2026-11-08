# -*- coding: utf-8 -*-
# Created time : 2026/11/08
# File   : image_fit_pixels.py
# Description : 按百万像素等比缩放（含遮罩、空输入安全）—— 官方「缩放图像（像素）」的替代
"""官方 `ImageScaleToTotalPixels` 有两个问题：
1. 输入为空（上游没选图）时直接报错，不能用于「槽位可能为空」的参考图链路；
2. 只处理 IMAGE，不处理 MASK，图1 的透明通道遮罩会跟图错位。

本节点按目标百万像素等比缩放，IMAGE 与 MASK 用同一算法同步缩放；
输入为空时输出 None + 全 0 遮罩，下游（文本编码器 / 拼接 / 羽化）都能安全跳过。
"""

import math

import torch

import comfy.utils

_EMPTY_MASK_SIDE = 64


def _is_empty(value):
    """上游 LoadImageBatchAdvanced 空路径返回 [None]，这里统一判空。"""
    if value is None:
        return True
    if isinstance(value, (list, tuple)):
        return len(value) == 0 or all(item is None for item in value)
    return False


def _as_batch(value):
    """把列表/单张统一成 [B, H, W, C]。"""
    if isinstance(value, (list, tuple)):
        frames = [item.unsqueeze(0) if item.dim() == 3 else item for item in value]
        return torch.cat(frames, dim=0)
    if value.dim() == 3:
        return value.unsqueeze(0)
    return value


def _resize_image(image, width, height, method):
    return comfy.utils.common_upscale(image.movedim(-1, 1), width, height, method, "disabled").movedim(1, -1)


def _resize_mask(mask, width, height, method):
    """遮罩按图像同一算法缩放，并**还原成输入时的维度**。

    ComfyUI 的 MASK 有 [H,W] 和 [B,H,W] 两种形态（列表输入时会被逐项映射，
    传进来的就是 [H,W]），这里必须按原维度返回，否则下游会把 4 维当成 5 维来解包。
    """
    original_dim = mask.dim()
    if original_dim == 2:
        data = mask.unsqueeze(0).unsqueeze(0)
    elif original_dim == 3:
        data = mask.unsqueeze(1)
    else:
        data = mask
    out = comfy.utils.common_upscale(data, width, height, method, "disabled")
    if original_dim == 2:
        out = out.squeeze(0).squeeze(0)
    elif original_dim == 3:
        out = out.squeeze(1)
    return out.clamp(0.0, 1.0)


class ImageFitPixelsMs:
    """按目标百万像素等比缩放图像与遮罩；空输入不报错。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "method": (["lanczos", "bicubic", "bilinear", "area", "nearest-exact"], {
                    "default": "lanczos",
                    "tooltip": "缩放算法，图像与遮罩用同一算法。",
                }),
                "megapixels": ("FLOAT", {
                    "default": 1.0, "min": 0.05, "max": 16.0, "step": 0.05,
                    "tooltip": "目标百万像素（1.0 ≈ 1024×1024）。按比例自动缩放，可能放大也可能缩小。",
                }),
                "multiple": ("INT", {
                    "default": 32, "min": 8, "max": 256, "step": 8,
                    "tooltip": "输出宽高对齐到该数值的倍数。",
                }),
                "enabled": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "关闭时原样透传（仍会处理空输入）。",
                }),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "可为空；为空时输出 None，下游会自动跳过。"}),
                "mask": ("MASK", {"tooltip": "跟随图像同步缩放；为空时输出全 0 遮罩。"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "fit"
    CATEGORY = "A_my_nodes/image"
    DESCRIPTION = (
        "按百万像素等比缩放（含遮罩）。与官方「缩放图像（像素）」的区别："
        "支持遮罩同步缩放，且输入为空时不报错。"
    )

    def fit(self, method="lanczos", megapixels=1.0, multiple=32, enabled=True, image=None, mask=None):
        if _is_empty(image):
            return (None, torch.zeros((1, _EMPTY_MASK_SIDE, _EMPTY_MASK_SIDE)))

        batch = _as_batch(image)
        if isinstance(mask, (list, tuple)):
            mask = next((item for item in mask if item is not None), None)

        if not enabled:
            if _is_empty(mask):
                mask = torch.zeros((batch.shape[0], batch.shape[1], batch.shape[2]),
                                   dtype=batch.dtype, device=batch.device)
            elif mask.dim() == 2:
                mask = mask.unsqueeze(0)
            return (batch, mask)

        height, width = int(batch.shape[1]), int(batch.shape[2])
        scale = math.sqrt(float(megapixels) * 1_000_000.0 / float(width * height))
        step = max(8, int(multiple))
        target_width = max(step, int(round(width * scale / step)) * step)
        target_height = max(step, int(round(height * scale / step)) * step)

        if (target_width, target_height) == (width, height):
            result = batch
        else:
            result = _resize_image(batch, target_width, target_height, method)

        if _is_empty(mask):
            resized_mask = torch.zeros((result.shape[0], target_height, target_width),
                                       dtype=result.dtype, device=result.device)
        else:
            resized_mask = _resize_mask(mask, target_width, target_height, method)
            if resized_mask.dim() == 2:
                resized_mask = resized_mask.unsqueeze(0)

        print(f"[ImageFitPixelsMs] {width}x{height} -> {target_width}x{target_height} "
              f"({megapixels}MP, {method}, enabled={enabled})")
        return (result, resized_mask)


NODE_CLASS_MAPPINGS = {"ImageFitPixelsMs": ImageFitPixelsMs}
NODE_DISPLAY_NAME_MAPPINGS = {"ImageFitPixelsMs": "按百万像素缩放（含遮罩）by ms"}
