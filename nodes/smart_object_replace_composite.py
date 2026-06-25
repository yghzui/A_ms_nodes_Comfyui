import cv2
import numpy as np
import torch
from time import perf_counter


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


def _compute_roi_bounds(mask_binary, padding, width, height):
    ys, xs = np.where(mask_binary > 0.5)
    if ys.size == 0 or xs.size == 0:
        return None

    y_min = max(int(ys.min()) - padding, 0)
    y_max = min(int(ys.max()) + padding + 1, height)
    x_min = max(int(xs.min()) - padding, 0)
    x_max = min(int(xs.max()) + padding + 1, width)
    return y_min, y_max, x_min, x_max


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

    # 当 ROI 较大时降采样计算距离变换，减少计算量
    h, w = edit_bin.shape
    if h * w > 40000:
        small_h, small_w = max(1, h // 2), max(1, w // 2)
        small_edit = cv2.resize(edit_bin, (small_w, small_h), interpolation=cv2.INTER_NEAREST)
        small_safe = cv2.resize(safe_bin, (small_w, small_h), interpolation=cv2.INTER_NEAREST)
        d_edit = cv2.distanceTransform(1 - small_edit, cv2.DIST_L2, 5)
        d_safe = cv2.distanceTransform(1 - small_safe, cv2.DIST_L2, 5)
        d_edit = cv2.resize(d_edit, (w, h), interpolation=cv2.INTER_LINEAR)
        d_safe = cv2.resize(d_safe, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
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

    apply_region = apply_mask > eps
    if apply_region.sum() == 0:
        return edited_rgb

    original_ref = original_rgb[ref_region]
    edited_ref = edited_rgb[ref_region]

    original_mean = original_ref.mean(axis=0)
    edited_mean = edited_ref.mean(axis=0)
    original_std = original_ref.std(axis=0)
    edited_std = edited_ref.std(axis=0)

    output = edited_rgb.copy()
    alpha = np.clip(apply_mask[apply_region], 0.0, 1.0)[:, None]
    edited_pixels = edited_rgb[apply_region]
    corrected_pixels = (edited_pixels - edited_mean) / (edited_std + eps) * original_std + original_mean
    np.clip(corrected_pixels, 0.0, 1.0, out=corrected_pixels)
    output[apply_region] = edited_pixels * (1.0 - alpha) + corrected_pixels * alpha
    np.clip(output, 0.0, 1.0, out=output)
    return output


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
                "edit_region_mask": ("MASK", {
                    "tooltip": "可选，编辑区域总遮罩。存在时节点只在该区域附近做 ROI 融合；未连接时直接输出 edited_image。"
                }),
                "edited_subject_mask": ("MASK", {
                    "tooltip": "可选，编辑结果中的新主体遮罩。只有它存在时，节点会按新主体边界附近优先保留编辑图，并优先用它保护新主体颜色。"
                }),
                "original_subject_mask": ("MASK", {
                    "tooltip": "可选，原图主体遮罩。只有它存在时，节点会按原主体边界附近优先保留编辑图。"
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
        edit_expand,
        transition_width,
        final_blur,
        color_match,
        protect_new_object,
        new_object_protect_erode,
        edit_region_mask=None,
        edited_subject_mask=None,
        original_subject_mask=None,
    ):
        node_start_time = perf_counter()
        original_image = self._ensure_image_batch(original_image, "original_image")
        edited_image = self._ensure_image_batch(edited_image, "edited_image")

        if original_image.shape[:3] != edited_image.shape[:3]:
            raise ValueError(
                "original_image and edited_image must share the same batch, height and width"
            )

        if edit_region_mask is None:
            print("[SmartObjectReplaceComposite] bypass: edit_region_mask is not connected")
            return (edited_image.clamp(0.0, 1.0), None, None, None, None)

        batch_size, height, width, _ = original_image.shape
        device = original_image.device
        dtype = original_image.dtype

        # 批量 GPU→CPU 传输，避免循环内重复传输
        original_cpu = original_image.detach().cpu().float().numpy()
        edited_cpu = edited_image.detach().cpu().float().numpy()

        # 预分配输出 tensor，循环内只填充 ROI
        result_tensor = original_image.clone()
        weight_tensor = torch.ones((batch_size, height, width), device=device, dtype=dtype)
        safe_tensor = torch.zeros((batch_size, height, width), device=device, dtype=dtype)
        edit_tensor = torch.zeros((batch_size, height, width), device=device, dtype=dtype)
        transition_tensor = torch.zeros((batch_size, height, width), device=device, dtype=dtype)

        has_any_roi = False

        for batch_index in range(batch_size):
            batch_start_time = perf_counter()

            stage_start = perf_counter()
            edit_region_mask_array = _to_numpy_mask(edit_region_mask, batch_index, batch_size)
            edited_subject_mask_array = _prepare_optional_mask(
                edited_subject_mask, batch_index, batch_size, width, height
            )
            original_subject_mask_array = _prepare_optional_mask(
                original_subject_mask, batch_index, batch_size, width, height
            )

            edit_region_mask_array = _resize_mask_if_needed(edit_region_mask_array, width, height)
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

            roi_padding = max(
                edit_expand + transition_width,
                final_blur,
                new_object_protect_erode if protect_new_object else 0,
                8 if color_match else 0,
            )
            roi_bounds = _compute_roi_bounds(
                edit_region_mask_binary, roi_padding, width, height
            )
            mask_prepare_ms = (perf_counter() - stage_start) * 1000.0

            if roi_bounds is None:
                # roi 为空时，result_tensor 已预填充 original_image，无需修改
                batch_total_ms = (perf_counter() - batch_start_time) * 1000.0
                print(
                    f"[SmartObjectReplaceComposite] batch={batch_index} roi=empty "
                    f"mask_prepare={mask_prepare_ms:.1f}ms total={batch_total_ms:.1f}ms"
                )
                continue

            has_any_roi = True

            y_min, y_max, x_min, x_max = roi_bounds
            roi_slice = np.s_[y_min:y_max, x_min:x_max]
            roi_height = y_max - y_min
            roi_width = x_max - x_min

            stage_start = perf_counter()

            # 使用预传输的 CPU 数组，避免重复 GPU→CPU 传输
            original_rgb = original_cpu[batch_index, y_min:y_max, x_min:x_max, :3]
            edited_rgb = edited_cpu[batch_index, y_min:y_max, x_min:x_max, :3]

            edit_region_mask_binary = edit_region_mask_binary[roi_slice]
            if original_subject_mask_binary is not None:
                original_subject_mask_binary = original_subject_mask_binary[roi_slice]
            if edited_subject_mask_binary is not None:
                edited_subject_mask_binary = edited_subject_mask_binary[roi_slice]
            roi_extract_ms = (perf_counter() - stage_start) * 1000.0

            stage_start = perf_counter()
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
            mask_compute_ms = (perf_counter() - stage_start) * 1000.0

            edited_used = edited_rgb.copy()
            stage_start = perf_counter()
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
            color_match_ms = (perf_counter() - stage_start) * 1000.0

            stage_start = perf_counter()
            weight_rgb = weight_original[..., None]
            result = original_rgb * weight_rgb + edited_used * (1.0 - weight_rgb)
            result = np.clip(result, 0.0, 1.0).astype(np.float32)

            transition = edit_region_mask_binary * (1.0 - edit_core) * (1.0 - safe_bg)
            transition = np.clip(transition, 0.0, 1.0).astype(np.float32)
            blend_ms = (perf_counter() - stage_start) * 1000.0

            stage_start = perf_counter()
            # 直接写入预分配的 tensor，避免重复创建和 stack
            result_tensor[batch_index, y_min:y_max, x_min:x_max, :3] = torch.from_numpy(result).to(
                device=device, dtype=dtype
            )
            weight_tensor[batch_index, y_min:y_max, x_min:x_max] = torch.from_numpy(weight_original).to(
                device=device, dtype=dtype
            )
            safe_tensor[batch_index, y_min:y_max, x_min:x_max] = torch.from_numpy(safe_bg).to(
                device=device, dtype=dtype
            )
            edit_tensor[batch_index, y_min:y_max, x_min:x_max] = torch.from_numpy(edit_core).to(
                device=device, dtype=dtype
            )
            transition_tensor[batch_index, y_min:y_max, x_min:x_max] = torch.from_numpy(transition).to(
                device=device, dtype=dtype
            )
            assemble_ms = (perf_counter() - stage_start) * 1000.0

            batch_total_ms = (perf_counter() - batch_start_time) * 1000.0
            print(
                f"[SmartObjectReplaceComposite] batch={batch_index} "
                f"roi={roi_width}x{roi_height} "
                f"mask_prepare={mask_prepare_ms:.1f}ms "
                f"roi_extract={roi_extract_ms:.1f}ms "
                f"mask_compute={mask_compute_ms:.1f}ms "
                f"color_match={color_match_ms:.1f}ms "
                f"blend={blend_ms:.1f}ms "
                f"assemble={assemble_ms:.1f}ms "
                f"total={batch_total_ms:.1f}ms"
            )

        if not has_any_roi:
            return (edited_image.clamp(0.0, 1.0), None, None, None, None)

        return (
            result_tensor,
            weight_tensor,
            safe_tensor,
            edit_tensor,
            transition_tensor,
        )
