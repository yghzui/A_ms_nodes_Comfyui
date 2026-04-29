import torch
from nodes import VAEEncode, VAEEncodeForInpaint
import comfy.utils
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
            print("🚀使用 inpaint 专用编码")
            encoder = VAEEncodeForInpaint()
            return encoder.encode(vae=vae, pixels=pixels, mask=mask, grow_mask_by=grow_mask_by)
        else:
            # 无效遮罩或未传入：退退回使用普通 VAE 编码
            print("🚀使用普通 VAE 编码")
            encoder = VAEEncode()
            return encoder.encode(vae=vae, pixels=pixels)
class FluxLatentMaskBinder:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "apply_mask"
    CATEGORY = "latent/advanced"
    DESCRIPTION = "Binds a mask directly to a latent's noise_mask property without altering the original pixels, ideal for Flux inpainting."

    def apply_mask(self, samples, mask=None):
        # 1. 拦截：如果未传入 mask，直接原样返回
        if mask is None:
            return (samples,)
            
        # 2. 拦截：如果 mask 存在但全是黑的（无效，没有任何需要重绘的地方），直接原样返回
        # 使用 .max() 检查是最快的方法
        if mask.max() <= 0:
            return (samples,)

        # 3. 获取目标尺寸
        # 从输入的 latent 中直接提取目标高宽。
        # latent["samples"] 的形状通常是 [Batch, Channels, Height, Width]
        latent_tensor = samples["samples"]
        latent_height = latent_tensor.shape[2]
        latent_width = latent_tensor.shape[3]

        # 4. 调整 Mask 维度以适配缩放函数
        # comfy.utils.common_upscale 需要 [B, C, H, W] 的 4D 张量
        if mask.dim() == 2:
            # 形状 [H, W] -> 变成 [1, 1, H, W]
            mask_samples = mask.unsqueeze(0).unsqueeze(0)
        elif mask.dim() == 3:
            # 形状 [B, H, W] -> 变成 [B, 1, H, W]
            mask_samples = mask.unsqueeze(1)
        else:
            # 异常维度保护
            return (samples,)

        # 5. 面积插值缩放
        # 使用 "area" (面积插值) 缩小 mask 到 latent 尺寸，这是处理 mask 最稳定的降采样方式
        m = comfy.utils.common_upscale(mask_samples, latent_width, latent_height, "area", "center")
        
        # 6. 还原维度
        # 缩放完后去掉我们为了适配计算加上的 Channel 维度，恢复到 [B, H, W]
        noise_mask = m.squeeze(1)

        # 7. 安全绑定并输出
        # 浅拷贝字典，防止修改污染上游节点的数据
        new_latent = samples.copy()
        new_latent["noise_mask"] = noise_mask

        return (new_latent,)
