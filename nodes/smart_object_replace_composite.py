import cv2
import numpy as np
import torch


def _to_numpy_mask(mask, batch_index, batch_size):
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().float().numpy()

    if mask.ndim == 2:
        return np.clip(mask, 0.0, 1.0).astype(np.float32)

    if mask.ndim == 3:
        if mask.shape[0] == batch_size:
            return np.clip(mask[batch_index], 0.0, 1.0).astype(np.float32)
        return np.clip(mask[0], 0.0, 1.0).astype(np.float32)

    raise ValueError(f"Unsupported mask shape: {getattr(mask, 'shape', None)}")


def _resize_mask_if_needed(mask, width, height):
    if mask.shape == (height, width):
        return mask
    return cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)


def _dilate_mask(mask, radius):
    if radius <= 0:
        return mask

    kernel_size = radius * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    output = cv2.dilate(mask.astype(np.float32), kernel)
    return np.clip(output, 0.0, 1.0)


def _erode_mask(mask, radius):
    if radius <= 0:
        return mask

    kernel_size = radius * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    output = cv2.erode(mask.astype(np.float32), kernel)
    return np.clip(output, 0.0, 1.0)


def _blur_mask(mask, radius):
    if radius <= 0:
        return mask

    kernel_size = radius * 2 + 1
    output = cv2.GaussianBlur(mask.astype(np.float32), (kernel_size, kernel_size), 0)
    return np.clip(output, 0.0, 1.0)


def _distance_weight(edit_core, safe_bg, edit_mask):
    edit_bin = (edit_core > 0.5).astype(np.uint8)
    safe_bin = (safe_bg > 0.5).astype(np.uint8)
    edit_mask_bin = (edit_mask > 0.5).astype(np.uint8)

    if edit_bin.sum() == 0:
        return np.ones_like(edit_mask, dtype=np.float32)

    if safe_bin.sum() == 0:
        weight = np.ones_like(edit_mask, dtype=np.float32)
        weight[edit_mask_bin == 1] = 0.0
        return weight

    d_edit = cv2.distanceTransform(1 - edit_bin, cv2.DIST_L2, 5)
    d_safe = cv2.distanceTransform(1 - safe_bin, cv2.DIST_L2, 5)

    weight_inside = d_edit / (d_edit + d_safe + 1e-6)
    weight_inside[edit_bin == 1] = 0.0
    weight_inside[safe_bin == 1] = 1.0

    weight = np.ones_like(edit_mask, dtype=np.float32)
    weight[edit_mask_bin == 1] = weight_inside[edit_mask_bin == 1]
    return np.clip(weight, 0.0, 1.0)


def _match_color_mean_std(original_rgb, edited_rgb, ref_mask, apply_mask, eps=1e-6):
    ref_region = ref_mask > 0.5
    if ref_region.sum() < 64:
        return edited_rgb

    original_ref = original_rgb[ref_region]
    edited_ref = edited_rgb[ref_region]

    original_mean = original_ref.mean(axis=0)
    edited_mean = edited_ref.mean(axis=0)
    original_std = original_ref.std(axis=0)
    edited_std = edited_ref.std(axis=0)

    corrected = (edited_rgb - edited_mean) / (edited_std + eps) * original_std + original_mean
    corrected = np.clip(corrected, 0.0, 1.0)

    alpha = np.clip(apply_mask, 0.0, 1.0)[..., None]
    output = edited_rgb * (1.0 - alpha) + corrected * alpha
    return np.clip(output, 0.0, 1.0)


class SmartObjectReplaceComposite:
    CATEGORY = "A_my_nodes/Image"
    FUNCTION = "composite"
    RETURN_TYPES = ("IMAGE", "MASK", "MASK", "MASK", "MASK")
    RETURN_NAMES = (
        "result_image",
        "weight_original",
        "safe_bg_mask",
        "edit_core_mask",
        "transition_mask",
    )
    DESCRIPTION = (
        "基于总编辑遮罩、原对象遮罩和新对象遮罩做智能回贴合成。"
        "节点会尽量回贴原图背景，在对象附近保留编辑结果，并输出调试遮罩方便观察融合区域。"
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_image": ("IMAGE", {
                    "tooltip": "原图 O。用于提供回贴背景和颜色参考。"
                }),
                "edited_image": ("IMAGE", {
                    "tooltip": "局部编辑后的图 E。对象附近区域优先保留该图内容。"
                }),
                "mask_A": ("MASK", {
                    "tooltip": "总编辑遮罩 A。定义允许发生融合的整体区域。"
                }),
                "mask_A_P": ("MASK", {
                    "tooltip": "原对象遮罩 A_P。通常是原人物或原物体分割结果。"
                }),
                "mask_B_P": ("MASK", {
                    "tooltip": "新对象遮罩 B_P。通常是替换后新人物或新物体分割结果。"
                }),
                "edit_expand": ("INT", {
                    "default": 24,
                    "min": 0,
                    "max": 256,
                    "step": 1,
                    "tooltip": "对 A_P 与 B_P 的联合区域做外扩，得到必须使用编辑图的核心区。"
                }),
                "transition_width": ("INT", {
                    "default": 64,
                    "min": 1,
                    "max": 512,
                    "step": 1,
                    "tooltip": "编辑核心区到安全背景区之间的过渡宽度。值越大，融合越柔和。"
                }),
                "final_blur": ("INT", {
                    "default": 3,
                    "min": 0,
                    "max": 64,
                    "step": 1,
                    "tooltip": "对最终原图权重做轻微平滑，减少硬边。"
                }),
                "color_match": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "是否用安全背景区估计色差，并对编辑图做局部颜色匹配。"
                }),
                "protect_new_object": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "开启后尽量保护新对象核心颜色，避免颜色匹配过度影响主体。"
                }),
                "new_object_protect_erode": ("INT", {
                    "default": 6,
                    "min": 0,
                    "max": 64,
                    "step": 1,
                    "tooltip": "新对象保护区腐蚀半径。值越大，颜色匹配越不容易进入主体内部。"
                }),
            }
        }

    @staticmethod
    def _ensure_image_batch(image, name):
        if not isinstance(image, torch.Tensor):
            raise ValueError(f"{name} must be a torch.Tensor")
        if image.ndim != 4:
            raise ValueError(f"{name} must be in [B, H, W, C] format, got shape {tuple(image.shape)}")
        if image.shape[-1] < 3:
            raise ValueError(f"{name} must contain at least 3 channels, got {image.shape[-1]}")
        return image[..., :3]

    def composite(
        self,
        original_image,
        edited_image,
        mask_A,
        mask_A_P,
        mask_B_P,
        edit_expand,
        transition_width,
        final_blur,
        color_match,
        protect_new_object,
        new_object_protect_erode,
    ):
        original_image = self._ensure_image_batch(original_image, "original_image")
        edited_image = self._ensure_image_batch(edited_image, "edited_image")

        if original_image.shape[:3] != edited_image.shape[:3]:
            raise ValueError(
                "original_image and edited_image must share the same batch, height and width"
            )

        original_batch = original_image.detach().cpu().float().numpy()
        edited_batch = edited_image.detach().cpu().float().numpy()
        batch_size, height, width, _ = original_batch.shape

        results = []
        weight_outputs = []
        safe_outputs = []
        edit_outputs = []
        transition_outputs = []

        for batch_index in range(batch_size):
            original_rgb = original_batch[batch_index]
            edited_rgb = edited_batch[batch_index]

            mask_a = _to_numpy_mask(mask_A, batch_index, batch_size)
            mask_ap = _to_numpy_mask(mask_A_P, batch_index, batch_size)
            mask_bp = _to_numpy_mask(mask_B_P, batch_index, batch_size)

            mask_a = np.clip(_resize_mask_if_needed(mask_a, width, height), 0.0, 1.0)
            mask_ap = np.clip(_resize_mask_if_needed(mask_ap, width, height), 0.0, 1.0)
            mask_bp = np.clip(_resize_mask_if_needed(mask_bp, width, height), 0.0, 1.0)

            mask_a_bin = (mask_a > 0.5).astype(np.float32)
            mask_ap_bin = (mask_ap > 0.5).astype(np.float32)
            mask_bp_bin = (mask_bp > 0.5).astype(np.float32)

            union_mask = np.maximum(mask_ap_bin, mask_bp_bin)

            edit_core = _dilate_mask(union_mask, edit_expand) * mask_a_bin
            edit_core = np.clip(edit_core, 0.0, 1.0)

            safe_radius = edit_expand + transition_width
            unsafe_bg = _dilate_mask(union_mask, safe_radius)
            safe_bg = mask_a_bin * (1.0 - unsafe_bg)
            safe_bg = np.clip(safe_bg, 0.0, 1.0)

            weight_original = _distance_weight(edit_core, safe_bg, mask_a_bin)

            if final_blur > 0:
                weight_original = _blur_mask(weight_original, final_blur)

            weight_original[edit_core > 0.5] = 0.0
            weight_original[safe_bg > 0.5] = 1.0
            weight_original[mask_a_bin < 0.5] = 1.0
            weight_original = np.clip(weight_original, 0.0, 1.0)

            edited_used = edited_rgb.copy()
            if color_match:
                if protect_new_object:
                    protected_new_object = _erode_mask(mask_bp_bin, new_object_protect_erode)
                    color_apply = mask_a_bin * (1.0 - protected_new_object)
                else:
                    color_apply = mask_a_bin

                color_apply = _blur_mask(color_apply, 8)
                color_apply = np.clip(color_apply, 0.0, 1.0)
                edited_used = _match_color_mean_std(
                    original_rgb,
                    edited_rgb,
                    ref_mask=safe_bg,
                    apply_mask=color_apply,
                )

            weight_rgb = weight_original[..., None]
            result = original_rgb * weight_rgb + edited_used * (1.0 - weight_rgb)
            result = np.clip(result, 0.0, 1.0).astype(np.float32)

            transition = mask_a_bin * (1.0 - edit_core) * (1.0 - safe_bg)
            transition = np.clip(transition, 0.0, 1.0).astype(np.float32)

            results.append(result)
            weight_outputs.append(weight_original.astype(np.float32))
            safe_outputs.append(safe_bg.astype(np.float32))
            edit_outputs.append(edit_core.astype(np.float32))
            transition_outputs.append(transition)

        device = original_image.device
        dtype = original_image.dtype

        result_tensor = torch.from_numpy(np.stack(results, axis=0)).to(device=device, dtype=dtype)
        weight_tensor = torch.from_numpy(np.stack(weight_outputs, axis=0)).to(device=device, dtype=dtype)
        safe_tensor = torch.from_numpy(np.stack(safe_outputs, axis=0)).to(device=device, dtype=dtype)
        edit_tensor = torch.from_numpy(np.stack(edit_outputs, axis=0)).to(device=device, dtype=dtype)
        transition_tensor = torch.from_numpy(np.stack(transition_outputs, axis=0)).to(device=device, dtype=dtype)

        return (
            result_tensor,
            weight_tensor,
            safe_tensor,
            edit_tensor,
            transition_tensor,
        )
