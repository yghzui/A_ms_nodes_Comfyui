# -*- coding: utf-8 -*-
# Created time : 2026/11/08
# Auther : ygh
# File   : qwen_image21_align.py
# Description : Qwen-Image 2.1 宽高对齐（zero shift），移植自 BlueBox 的 Cut (Qwen-Image 2.1) 思路
"""Qwen-Image 2.1 对潜在尺寸很敏感：如果直接把图1的原始宽高喂给采样，
模型内部会把参考图重新缩放到自己的面积基准上，尺度一变就等于平移，输出和参考对不齐。

这里的做法是：
1. 按图像面积推一个基准 resolution（或手动给），对齐到 32 的倍数；
2. 在「宽×高 ≈ resolution²」的白名单里挑一个比例最接近输入、且面积接近、比例接近标准比例的尺寸；
3. 校验这个尺寸是缩放的**不动点**（按 resolution 再缩一次仍是它自己），保证模型内部缩放是恒等变换；
4. 把图像（以及遮罩）实际裁剪/补边/拉伸到该尺寸。

这样宽度、高度、resolution 三者自洽，参考图与潜空间逐像素对应，不再出现偏移。
"""

import math

import torch

_MIN_SIDE = 128
_MAX_SIDE = 4096
_STEP = 32
_MIN_RESOLUTION = 128
_MAX_RESOLUTION = 4096
_RESOLUTIONS = list(range(_MIN_RESOLUTION, _MAX_RESOLUTION + 1, _STEP))

# 常见画幅比例，用来在候选里偏向「标准比例」，避免选出 1023:1024 这种别扭尺寸
_STANDARD_RATIOS = (1.0, 4 / 3, 3 / 4, 16 / 9, 9 / 16, 3 / 2, 2 / 3)

_MODES = ("pad32", "stretch", "crop", "pad")
_RESOLUTION_MODES = ("default", "manual", "custom")


def _snap_resolution(value):
    """把基准分辨率对齐到 32 的倍数并夹在允许区间内。"""
    snapped = int(round(float(value) / _STEP)) * _STEP
    snapped = max(_MIN_RESOLUTION, min(_MAX_RESOLUTION, snapped))
    return snapped if snapped in _RESOLUTIONS else min(_RESOLUTIONS, key=lambda item: abs(item - snapped))


def _ratio_distance(left, right):
    if left <= 0 or right <= 0:
        return float("inf")
    return abs(math.log(left) - math.log(right))


def _score_candidate(width, height, product, resolution, input_ratio):
    ratio = width / height
    distance_input = _ratio_distance(ratio, input_ratio)
    distance_standard = min(_ratio_distance(ratio, item) for item in _STANDARD_RATIOS)
    target_area = float(resolution) * float(resolution)
    distance_area = abs(product * _STEP * _STEP - target_area) / target_area
    return distance_input + distance_standard * 0.6 + distance_area * 0.4


def _generate_whitelist(resolution, tol_ratio=1.0):
    """列一组候选尺寸：(宽, 高, 宽高块数乘积)，面积都落在 resolution² 附近。"""
    blocks = resolution // _STEP
    if blocks < 1:
        return []
    target = blocks * blocks
    tolerance = max(1, int(blocks * tol_ratio))
    pairs = []
    seen = set()
    for width_blocks in range(1, blocks * 3):
        for height_blocks in range(1, blocks * 3):
            product = width_blocks * height_blocks
            if abs(product - target) > tolerance:
                continue
            width, height = width_blocks * _STEP, height_blocks * _STEP
            if width > _MAX_SIDE or height > _MAX_SIDE:
                continue
            if width < _MIN_SIDE or height < _MIN_SIDE:
                continue
            if (width, height) in seen:
                continue
            seen.add((width, height))
            pairs.append((width, height, product))
    return pairs


def _fixed_point_of(width, height, resolution):
    """按 resolution 归一化后重新取整，看尺寸是否仍是自己。"""
    scale = math.sqrt((float(resolution) * float(resolution)) / (float(width) * float(height)))
    new_width = int(round(width * scale / _STEP)) * _STEP
    new_height = int(round(height * scale / _STEP)) * _STEP
    new_width = max(_MIN_SIDE, min(_MAX_SIDE, new_width))
    new_height = max(_MIN_SIDE, min(_MAX_SIDE, new_height))
    return new_width, new_height


def _pick_target_size(width, height, resolution, tol_ratio=1.0):
    input_ratio = width / float(height)
    whitelist = _generate_whitelist(resolution, tol_ratio)
    if not whitelist:
        raise ValueError(f"resolution={resolution} 无法生成候选尺寸")
    best, best_score = (width, height), float("inf")
    for candidate_width, candidate_height, product in whitelist:
        score = _score_candidate(candidate_width, candidate_height, product, resolution, input_ratio)
        if score < best_score:
            best_score = score
            best = (candidate_width, candidate_height)
    return best, whitelist


def _snap_to_fixed_point(target_width, target_height, resolution, tol_ratio, input_width, input_height):
    """优先选不动点；若首选不是不动点，退回白名单里误差最小的不动点。"""
    if _fixed_point_of(target_width, target_height, resolution) == (target_width, target_height):
        return target_width, target_height
    input_ratio = input_width / float(input_height)
    fallback, best_error = None, float("inf")
    for width, height, _ in _generate_whitelist(resolution, tol_ratio):
        if _fixed_point_of(width, height, resolution) != (width, height):
            continue
        error = _ratio_distance(width / float(height), input_ratio)
        if error < best_error:
            best_error, fallback = error, (width, height)
    return fallback or (target_width, target_height)


def _prepare(tensor, is_mask):
    """统一成 [B, H, W, C] 并记录原始维度，供还原时用。"""
    original_dim = tensor.dim()
    if is_mask:
        if original_dim == 2:
            tensor = tensor.unsqueeze(0)
            original_dim = 3
        tensor = tensor.unsqueeze(-1)
    elif original_dim == 3:
        tensor = tensor.unsqueeze(0)
    return tensor, original_dim


def _restore(tensor, original_dim, is_mask):
    if is_mask:
        tensor = tensor.squeeze(-1)
        if original_dim == 2:
            tensor = tensor.squeeze(0)
    elif original_dim == 3:
        tensor = tensor.squeeze(0)
    return tensor


def _resize(image, width, height):
    data = image.permute(0, 3, 1, 2)
    data = torch.nn.functional.interpolate(data, size=(height, width), mode="bilinear", align_corners=False)
    return data.permute(0, 2, 3, 1)


def _apply_geometry(tensor, target_width, target_height, mode, pad_value=0.0, is_mask=False):
    """图像与遮罩共用同一套几何变换，保证两者严格对齐。"""
    tensor, original_dim = _prepare(tensor, is_mask)
    batch, height, width, channels = tensor.shape

    if mode == "stretch":
        out = _resize(tensor, target_width, target_height)
    elif mode == "pad":
        scale = min(target_width / float(width), target_height / float(height))
        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))
        resized = _resize(tensor, new_width, new_height)
        out = torch.full((batch, target_height, target_width, channels), pad_value,
                         dtype=resized.dtype, device=resized.device)
        top = (target_height - new_height) // 2
        left = (target_width - new_width) // 2
        out[:, top:top + new_height, left:left + new_width, :] = resized
    else:  # crop
        scale = max(target_width / float(width), target_height / float(height))
        new_width = max(target_width, int(round(width * scale)))
        new_height = max(target_height, int(round(height * scale)))
        resized = _resize(tensor, new_width, new_height)
        top = (new_height - target_height) // 2
        left = (new_width - target_width) // 2
        out = resized[:, top:top + target_height, left:left + target_width, :]

    return _restore(out, original_dim, is_mask)


def pad_offset(target_side, source_side):
    """补边时单侧偏移。复原节点必须用同一个公式，否则裁回来会偏。"""
    return max(0, (int(target_side) - int(source_side)) // 2)


def _pad_center(tensor, target_width, target_height, is_mask=False):
    """只补边、不缩放：内容按原像素 1:1 贴到目标尺寸正中。"""
    tensor, original_dim = _prepare(tensor, is_mask)
    batch, height, width, channels = tensor.shape
    if (width, height) == (target_width, target_height):
        return _restore(tensor, original_dim, is_mask)

    out = torch.zeros((batch, target_height, target_width, channels),
                      dtype=tensor.dtype, device=tensor.device)
    left = pad_offset(target_width, width)
    top = pad_offset(target_height, height)
    out[:, top:top + height, left:left + width, :] = tensor[:, :, :, :]
    return _restore(out, original_dim, is_mask)


def _ceil_step(value):
    return max(_STEP, int(math.ceil(float(value) / _STEP)) * _STEP)


_CUSTOM_STEP = 16


def _ceil_step16(value):
    """空潜空间实际按 16 取整（实测 900->896、912->912），这里向上取整，不缩图、不裁图。"""
    return max(_CUSTOM_STEP, int(math.ceil(float(value) / _CUSTOM_STEP)) * _CUSTOM_STEP)


def _is_empty(value):
    """LoadImageBatchAdvanced 空路径返回 [None]，这里统一判空。"""
    if value is None:
        return True
    if isinstance(value, (list, tuple)):
        return len(value) == 0 or all(item is None for item in value)
    return False


class QwenImage21AlignSize:
    """把参考图对齐到 Qwen-Image 2.1 的稳定尺寸，并输出配套的宽 / 高 / resolution。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "要对齐的参考图，通常接第 1 张参考图。"}),
                "resolution_mode": (list(_RESOLUTION_MODES), {
                    "default": "default",
                    "tooltip": "default=只补边到 32 的倍数（不缩放，resolution 传 0），配合复原节点可保证输出宽高与输入完全一致；manual=按下面的基准分辨率缩放到稳定尺寸；custom=用指定宽高出空潜空间。",
                }),
                "resolution_value": ("INT", {
                    "default": 1024, "min": _MIN_RESOLUTION, "max": _MAX_RESOLUTION, "step": _STEP,
                    "tooltip": "仅 manual 模式生效。",
                }),
                "mode": (list(_MODES), {
                    "default": "pad32",
                    "tooltip": "仅 manual 模式生效：pad32=只补边不缩放（default 模式固定用这个）；stretch=拉伸；crop=居中裁切；pad=等比缩放后补黑边。",
                }),
                "custom_width": ("INT", {
                    "default": 1024, "min": _MIN_SIDE, "max": _MAX_SIDE, "step": _CUSTOM_STEP,
                    "tooltip": "custom 模式下的画布宽，向上取整到 16 的倍数（模型潜空间是 16 倍下采样）；参考图不会被缩放或裁切。",
                }),
                "custom_height": ("INT", {
                    "default": 1024, "min": _MIN_SIDE, "max": _MAX_SIDE, "step": _CUSTOM_STEP,
                    "tooltip": "custom 模式下的画布高，规则同 custom_width。",
                }),
            },
            "optional": {
                "mask": ("MASK", {"tooltip": "随图像做同一套变换，保证遮罩不会错位。"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT", "INT", "*", "BOOLEAN")
    RETURN_NAMES = ("image", "mask", "width", "height", "resolution", "mode", "restore")
    FUNCTION = "align"
    CATEGORY = "A_my_nodes/image"
    DESCRIPTION = (
        "Qwen-Image 2.1 宽高对齐：custom 模式把画布宽高向上取整到 8 的倍数并输出 mode=empty"
        "（直接接潜空间来源的 mode）；其余模式输出 mode=auto。没有参考图时也回 mode=empty 并给零遮罩。"
    )

    def align(self, image, resolution_mode="default", resolution_value=1024, mode="pad32",
              custom_width=1024, custom_height=1024, mask=None):
        # 没有参考图（纯文生图）：给一块空遮罩，图输出 None（文本编码器会跳过 None）
        if _is_empty(image):
            target_width = _ceil_step16(custom_width)
            target_height = _ceil_step16(custom_height)
            zero_mask = torch.zeros((1, target_height, target_width), dtype=torch.float32)
            return (None, zero_mask, target_width, target_height, 0, "empty", False)

        if isinstance(image, (list, tuple)):
            image = next(item for item in image if item is not None)
        if image.dim() == 3:
            image = image.unsqueeze(0)
        _, height, width, _ = image.shape

        if resolution_mode == "custom":
            # 自定义宽高 = 空潜空间出图：只把画布宽高对齐到 8 的倍数，图原样透传
            target_width = _ceil_step16(custom_width)
            target_height = _ceil_step16(custom_height)
            if _is_empty(mask):
                aligned_mask = torch.zeros((image.shape[0], target_height, target_width),
                                           dtype=image.dtype, device=image.device)
            else:
                aligned_mask = mask.unsqueeze(0) if mask.dim() == 2 else mask
            return (image, aligned_mask, target_width, target_height, 0, "empty", False)

        if resolution_mode == "default":
            # 只补边不缩放：宽高向上取整到 32 的倍数，内容保持 1:1（零重采样）。
            # resolution 传 0，编码器就按参考图自身尺寸取整到 32，706/688 这类尺寸不会再被缩放一次。
            target_width = _ceil_step(width)
            target_height = _ceil_step(height)
            resolution = 0
            if target_width * target_height > 4 * 1024 * 1024:
                print(f"[QwenImage21AlignSize] 输入 {width}x{height} 会按原尺寸出图（补边到 "
                      f"{target_width}x{target_height}，约 {(target_width * target_height) / 1e6:.1f}MP），"
                      f"显存和耗时都会明显上升，建议先在前面缩到 1536 左右。")
            aligned_image = _pad_center(image, target_width, target_height)
            if _is_empty(mask):
                aligned_mask = torch.zeros((aligned_image.shape[0], target_height, target_width),
                                           dtype=aligned_image.dtype, device=aligned_image.device)
            else:
                aligned_mask = _pad_center(mask, target_width, target_height, is_mask=True)
                if aligned_mask.dim() == 2:
                    aligned_mask = aligned_mask.unsqueeze(0)
            return (aligned_image, aligned_mask, target_width, target_height, resolution, "auto", True)

        # manual：按给定基准分辨率缩放到稳定尺寸（会重采样，尺寸因此可能和输入不同）
        resolution = _snap_resolution(resolution_value)
        target_width, target_height = _pick_target_size(width, height, resolution, tol_ratio=1.0)[0]
        target_width, target_height = _snap_to_fixed_point(
            target_width, target_height, resolution, 1.0, width, height)
        use_mode = "stretch" if mode == "pad32" else mode

        aligned_image = _apply_geometry(image, target_width, target_height, use_mode, pad_value=0.0)

        if _is_empty(mask):
            aligned_mask = torch.zeros((aligned_image.shape[0], target_height, target_width),
                                       dtype=aligned_image.dtype, device=aligned_image.device)
        else:
            aligned_mask = _apply_geometry(mask, target_width, target_height, use_mode,
                                           pad_value=0.0, is_mask=True)
            if aligned_mask.dim() == 2:
                aligned_mask = aligned_mask.unsqueeze(0)

        return (aligned_image, aligned_mask, target_width, target_height, resolution, "auto", True)


class QwenImage21RestoreSize:
    """把生成结果精确裁回参考图的原始宽高（与补边用同一套偏移公式），实现「输入输出同尺寸」。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "生成结果（解码后的图）。"}),
                "reference": ("IMAGE", {"tooltip": "原始参考图（通常接第 1 张参考图，未经对齐）。"}),
            },
            "optional": {
                "enable": ("BOOLEAN", {
                    "default": True, "forceInput": True,
                    "tooltip": "接「宽高对齐」的 restore 输出：auto（潜空间来自图像）为 True 才裁回；"
                               "custom / 无图时为 False，直接原样透传。",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "restore"
    CATEGORY = "A_my_nodes/image"
    DESCRIPTION = (
        "尺寸复原：把生成结果按参考图的宽高居中裁切/缩放回原始尺寸。"
        "enable=False（custom 空潜空间模式）或没有参考图时原样透传。"
    )

    def restore(self, image, reference, enable=True):
        # 开关关闭（custom 空潜空间）或没有参考图：不裁，原样输出
        if not enable or _is_empty(reference):
            return (image,)
        if image.dim() == 3:
            image = image.unsqueeze(0)
        if reference.dim() == 3:
            reference = reference.unsqueeze(0)

        target_height, target_width = int(reference.shape[1]), int(reference.shape[2])
        _, height, width, _ = image.shape

        if (width, height) == (target_width, target_height):
            return (image,)

        if width >= target_width and height >= target_height:
            left = pad_offset(width, target_width)
            top = pad_offset(height, target_height)
            cropped = image[:, top:top + target_height, left:left + target_width, :]
            return (cropped.contiguous(),)

        # 生成结果比参考图小：说明潜空间尺寸不是由这张图推出来的，只能缩放兜底
        print(f"[QwenImage21RestoreSize] 生成 {width}x{height} 小于参考 {target_width}x{target_height}，"
              f"改用缩放还原")
        return (_resize(image, target_width, target_height),)


NODE_CLASS_MAPPINGS = {"QwenImage21AlignSize": QwenImage21AlignSize,
                       "QwenImage21RestoreSize": QwenImage21RestoreSize}
NODE_DISPLAY_NAME_MAPPINGS = {"QwenImage21AlignSize": "Qwen2.1 宽高对齐（zero shift）by ms",
                              "QwenImage21RestoreSize": "Qwen2.1 尺寸复原 by ms"}
