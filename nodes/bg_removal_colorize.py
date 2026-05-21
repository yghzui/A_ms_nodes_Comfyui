import folder_paths
import torch

from .image_mix import parse_color

try:
    from comfy.bg_removal_model import load as load_background_removal_model
    BG_REMOVAL_IMPORT_ERROR = None
except Exception as exc:
    load_background_removal_model = None
    BG_REMOVAL_IMPORT_ERROR = exc


UNAVAILABLE_MODEL_OPTION = "WARNING: background removal unavailable"
NO_MODEL_OPTION = "WARNING: no background removal models found"


class BackgroundRemovalColorize:
    CATEGORY = "A_my_nodes/Image"
    FUNCTION = "process"
    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "IMAGE")
    RETURN_NAMES = ("background_replaced", "mask", "subject_fill", "rgba_image")
    DESCRIPTION = (
        "使用 ComfyUI 原生 background_removal 模型生成前景遮罩，并同时输出背景换色图、"
        "mask、主体纯色填充图，以及把 mask 写入 alpha 通道的 RGBA 图。"
    )

    _warning_printed = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "要执行抠图的输入图像。若输入包含 alpha，只会使用其 RGB 部分参与抠图。"
                }),
                "enable_bg_removal": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "是否执行抠图。关闭时直接返回原图到第一个输出，其余三个输出返回 None。"
                }),
                "bg_removal_name": (cls._get_model_options(), {
                    "tooltip": "背景移除模型。列表获取方式与 ComfyUI 原生 Load Background Removal Model 节点一致。"
                }),
                "background_color": ("STRING", {
                    "default": "#FFFFFF",
                    "multiline": False,
                    "tooltip": "背景替换颜色，支持 #RRGGBB 或 R,G,B 格式。前端会把该输入显示为可点击的实时色块。"
                }),
                "fill_color": ("STRING", {
                    "default": "#000000",
                    "multiline": False,
                    "tooltip": "主体填充颜色，支持 #RRGGBB 或 R,G,B 格式。前端会把该输入显示为可点击的实时色块。"
                }),
            }
        }

    @classmethod
    def _warn_unavailable_once(cls):
        if cls._warning_printed:
            return
        detail = f": {BG_REMOVAL_IMPORT_ERROR}" if BG_REMOVAL_IMPORT_ERROR is not None else ""
        print(f"[A_my_nodes] 警告：background_removal 不可用{detail}")
        cls._warning_printed = True

    @classmethod
    def _get_model_options(cls):
        if load_background_removal_model is None:
            cls._warn_unavailable_once()
            return [UNAVAILABLE_MODEL_OPTION]

        files = folder_paths.get_filename_list("background_removal")
        if not files:
            return [NO_MODEL_OPTION]
        return sorted(files)

    @staticmethod
    def _ensure_rgb(image: torch.Tensor) -> torch.Tensor:
        if image.shape[-1] < 3:
            raise ValueError(f"输入图像通道数不足，当前为 {image.shape[-1]}，至少需要 3 个通道。")
        return image[..., :3]

    @staticmethod
    def _normalize_mask(mask: torch.Tensor) -> torch.Tensor:
        if mask.ndim == 4:
            if mask.shape[1] == 1:
                mask = mask[:, 0]
            elif mask.shape[-1] == 1:
                mask = mask[..., 0]
            else:
                mask = mask[:, 0]
        elif mask.ndim != 3:
            raise ValueError(f"背景移除模型返回了不支持的 mask 维度: {tuple(mask.shape)}")

        return mask.clamp(0.0, 1.0)

    @staticmethod
    def _solid_color_image_like(image_rgb: torch.Tensor, color: str) -> torch.Tensor:
        r, g, b = parse_color(color)
        color_image = torch.empty_like(image_rgb)
        color_image[..., 0] = r
        color_image[..., 1] = g
        color_image[..., 2] = b
        return color_image

    def process(self, image, enable_bg_removal, bg_removal_name, background_color, fill_color):
        image_rgb = self._ensure_rgb(image)
        if not enable_bg_removal:
            empty_mask = torch.zeros(
                (image_rgb.shape[0], image_rgb.shape[1], image_rgb.shape[2]),
                dtype=image_rgb.dtype,
                device=image_rgb.device,
            )
            return (
                image_rgb.clamp(0.0, 1.0),
                empty_mask,
                image_rgb.clamp(0.0, 1.0),
                image_rgb.clamp(0.0, 1.0),
            )

        if load_background_removal_model is None:
            self._warn_unavailable_once()
            raise RuntimeError("background_removal 模块不可用，当前节点无法执行。")

        if bg_removal_name == UNAVAILABLE_MODEL_OPTION:
            raise RuntimeError("background_removal 模块不可用，当前节点无法执行。")
        if bg_removal_name == NO_MODEL_OPTION:
            raise RuntimeError("未找到 background_removal 模型文件，请先放入对应模型后再使用。")

        model_path = folder_paths.get_full_path_or_raise("background_removal", bg_removal_name)
        bg_model = load_background_removal_model(model_path)
        if bg_model is None:
            raise RuntimeError("背景移除模型文件无效，无法加载为可用的 background removal 模型。")

        mask = self._normalize_mask(bg_model.encode_image(image_rgb))
        mask_4d = mask.unsqueeze(-1)

        background_image = self._solid_color_image_like(image_rgb, background_color)
        fill_image = self._solid_color_image_like(image_rgb, fill_color)

        background_replaced = image_rgb * mask_4d + background_image * (1.0 - mask_4d)
        subject_fill = image_rgb * (1.0 - mask_4d) + fill_image * mask_4d
        rgba_image = torch.cat((image_rgb, mask_4d), dim=-1)

        return (
            background_replaced.clamp(0.0, 1.0),
            mask,
            subject_fill.clamp(0.0, 1.0),
            rgba_image.clamp(0.0, 1.0),
        )
