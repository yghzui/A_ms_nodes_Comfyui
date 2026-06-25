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


def _prepare_optional_mask(mask, batch_index, batch_size, width, height):
    if mask is None:
        return None

    try:
        prepared_mask = _to_numpy_mask(mask, batch_index, batch_size)
    except Exception:
        return None

    return np.clip(_resize_mask_if_needed(prepared_mask, width, height), 0.0, 1.0)


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
        "original_image_weight",
        "safe_background_mask",
        "edited_priority_mask",
        "blend_transition_mask",
    )
    OUTPUT_TOOLTIPS = (
        "最终智能融合后的输出图像。",
        "原图参与融合的权重图。值越高，越偏向使用 original_image。",
        "确认可安全回贴 original_image 背景的区域遮罩。",
        "必须优先保留 edited_image 内容的核心区域遮罩。",
        "安全背景区与编辑优先区之间的混合过渡区域遮罩。",
    )
    DESCRIPTION = (
        "基于编辑区域遮罩、原主体遮罩和新主体遮罩做智能回贴合成。"
        "支持主体遮罩缺省时的降级逻辑，并输出调试遮罩方便观察融合区域。"
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
                "edit_region_mask": ("MASK", {
                    "tooltip": "编辑区域总遮罩。定义允许发生融合和回贴的整体区域。"
                }),
                "edit_expand": ("INT", {
                    "default": 24,
                    "min": 0,
                    "max": 256,
                    "step": 1,
                    "tooltip": "对主体联合区域做外扩，得到必须使用编辑图的核心区。"
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
            },
            "optional": {
                "original_subject_mask": ("MASK", {
                    "tooltip": "可选，原图主体遮罩。只有它存在时，节点会按原主体边界附近优先保留编辑图。"
                }),
                "edited_subject_mask": ("MASK", {
                    "tooltip": "可选，编辑结果中的新主体遮罩。只有它存在时，节点会按新主体边界附近优先保留编辑图，并优先用它保护新主体颜色。"
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
        edit_region_mask,
        edit_expand,
        transition_width,
        final_blur,
        color_match,
        protect_new_object,
        new_object_protect_erode,
        original_subject_mask=None,
        edited_subject_mask=None,
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

            edit_region_mask_array = _to_numpy_mask(edit_region_mask, batch_index, batch_size)
            original_subject_mask_array = _prepare_optional_mask(
                original_subject_mask, batch_index, batch_size, width, height
            )
            edited_subject_mask_array = _prepare_optional_mask(
                edited_subject_mask, batch_index, batch_size, width, height
            )

            edit_region_mask_array = np.clip(
                _resize_mask_if_needed(edit_region_mask_array, width, height), 0.0, 1.0
            )
            edit_region_mask_binary = (edit_region_mask_array > 0.5).astype(np.float32)
            original_subject_mask_binary = (
                (original_subject_mask_array > 0.5).astype(np.float32)
                if original_subject_mask_array is not None
                else None
            )
            edited_subject_mask_binary = (
                (edited_subject_mask_array > 0.5).astype(np.float32)
                if edited_subject_mask_array is not None
                else None
            )

            subject_masks = []
            if original_subject_mask_binary is not None:
                subject_masks.append(original_subject_mask_binary)
            if edited_subject_mask_binary is not None:
                subject_masks.append(edited_subject_mask_binary)

            if subject_masks:
                union_mask = np.maximum.reduce(subject_masks)

                edit_core = _dilate_mask(union_mask, edit_expand) * edit_region_mask_binary
                edit_core = np.clip(edit_core, 0.0, 1.0)

                safe_radius = edit_expand + transition_width
                unsafe_bg = _dilate_mask(union_mask, safe_radius)
                safe_bg = edit_region_mask_binary * (1.0 - unsafe_bg)
                safe_bg = np.clip(safe_bg, 0.0, 1.0)

                weight_original = _distance_weight(edit_core, safe_bg, edit_region_mask_binary)

                if final_blur > 0:
                    weight_original = _blur_mask(weight_original, final_blur)

                weight_original[edit_core > 0.5] = 0.0
                weight_original[safe_bg > 0.5] = 1.0
                weight_original[edit_region_mask_binary < 0.5] = 1.0
                weight_original = np.clip(weight_original, 0.0, 1.0)
            else:
                # 没有主体遮罩时，无法可靠区分背景和主体，保守地在编辑区域内完全采用编辑图。
                union_mask = edit_region_mask_binary
                edit_core = edit_region_mask_binary.copy()
                safe_bg = np.zeros_like(edit_region_mask_binary, dtype=np.float32)
                weight_original = np.ones_like(edit_region_mask_binary, dtype=np.float32)
                weight_original[edit_region_mask_binary > 0.5] = 0.0

            edited_used = edited_rgb.copy()
            if color_match:
                if protect_new_object and edited_subject_mask_binary is not None:
                    protected_new_object = _erode_mask(
                        edited_subject_mask_binary, new_object_protect_erode
                    )
                    color_apply = edit_region_mask_binary * (1.0 - protected_new_object)
                else:
                    color_apply = edit_region_mask_binary

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

            transition = edit_region_mask_binary * (1.0 - edit_core) * (1.0 - safe_bg)
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
