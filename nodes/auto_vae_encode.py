import torch
from nodes import VAEEncode, VAEEncodeForInpaint

class AutoVAEEncode:
    """
    自动 VAE 编码节点。
    根据是否传入了有效的 mask，自动在官方的 VAEEncode 和 VAEEncodeForInpaint 之间切换。
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pixels": ("IMAGE", ),
                "vae": ("VAE", ),
                "grow_mask_by": ("INT", {"default": 6, "min": 0, "max": 64, "step": 1}),
            },
            "optional": {
                "mask": ("MASK", ),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "encode"
    CATEGORY = "latent/inpaint"

    def encode(self, pixels, vae, grow_mask_by=6, mask=None):
        # 检查是否传入了 mask，并且 mask 中是否包含大于 0 的有效区域
        if mask is not None and torch.any(mask > 0):
            # 有效遮罩：使用 inpaint 专用编码，这会包含 noise_mask 以供采样器使用
            encoder = VAEEncodeForInpaint()
            return encoder.encode(vae=vae, pixels=pixels, mask=mask, grow_mask_by=grow_mask_by)
        else:
            # 无效遮罩或未传入：退退回使用普通 VAE 编码
            encoder = VAEEncode()
            return encoder.encode(vae=vae, pixels=pixels)
