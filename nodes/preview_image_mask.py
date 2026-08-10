import torch
import nodes
from comfy_extras.nodes_mask import MaskPreview

from .any_batch_accumulator import ANY_TYPE


class AnyImageMaskPreview:
    def __init__(self):
        self.image_preview = nodes.PreviewImage()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "data": (ANY_TYPE, {"tooltip": "支持 ComfyUI IMAGE 或 MASK；无效数据不会显示预览。"}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = (ANY_TYPE,)
    RETURN_NAMES = ("data",)
    FUNCTION = "preview"
    OUTPUT_NODE = True
    CATEGORY = "A_my_nodes/image"
    DESCRIPTION = "自动识别 IMAGE 或 MASK 并调用 ComfyUI 官方预览；空值、空张量、全零、NaN 或 Inf 数据不显示。"

    def _is_valid(self, data):
        if not isinstance(data, torch.Tensor) or data.numel() == 0:
            return False
        if not data.any():
            return False
        if torch.isnan(data).any() or torch.isinf(data).any():
            return False
        return True

    def preview(self, data=None, prompt=None, extra_pnginfo=None):
        if not self._is_valid(data):
            return {"ui": {"images": []}, "result": (data,)}

        if data.ndim == 4:
            return self.image_preview.save_images(data, "ComfyUI", prompt, extra_pnginfo)
        if data.ndim in (2, 3):
            return MaskPreview.execute(data)

        return {"ui": {"images": []}, "result": (data,)}
