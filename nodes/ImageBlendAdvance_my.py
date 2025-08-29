import torch
import numpy as np
from PIL import Image, ImageDraw
import sys
import os

# 添加必要的类型转换函数
def pil2tensor(image):
    """将PIL图像转换为tensor"""
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def tensor2pil(image):
    """将tensor转换为PIL图像"""
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def image2mask(image):
    """将图像转换为遮罩"""
    if image.mode != 'L':
        image = image.convert('L')
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def mask2image(mask):
    """将遮罩转换为图像"""
    return Image.fromarray(np.clip(255. * mask.cpu().numpy().squeeze(), 0, 255).astype(np.uint8), mode='L')

def RGB2RGBA(image, mask):
    """将RGB图像和遮罩合并为RGBA图像"""
    (R, G, B) = image.convert('RGB').split()
    return Image.merge('RGBA', (R, G, B, mask.convert('L')))

def __rotate_expand(image, angle, SSAA=0, method="lanczos"):
    """旋转并扩展图像"""
    if angle == 0.0 or angle == 360.0:
        return image
    
    resize_sampler = Image.LANCZOS
    rotate_sampler = Image.BICUBIC
    if method == "bicubic":
        resize_sampler = Image.BICUBIC
        rotate_sampler = Image.BICUBIC
    elif method == "hamming":
        resize_sampler = Image.HAMMING
        rotate_sampler = Image.BILINEAR
    elif method == "bilinear":
        resize_sampler = Image.BILINEAR
        rotate_sampler = Image.BILINEAR
    elif method == "box":
        resize_sampler = Image.BOX
        rotate_sampler = Image.NEAREST
    elif method == "nearest":
        resize_sampler = Image.NEAREST
        rotate_sampler = Image.NEAREST
    
    if SSAA > 1:
        width, height = image.size
        img_us_scaled = image.resize((width * SSAA, height * SSAA), resize_sampler)
        img_rotated = img_us_scaled.rotate(angle, rotate_sampler, expand=True, fillcolor=(0, 0, 0, 0))
        img_down_scaled = img_rotated.resize((img_rotated.width // SSAA, img_rotated.height // SSAA), resize_sampler)
        return img_down_scaled
    else:
        return image.rotate(angle, rotate_sampler, expand=True, fillcolor=(0, 0, 0, 0))

def image_rotate_extend_with_alpha(image, angle, alpha=None, method="lanczos", SSAA=0):
    """带alpha通道的图像旋转扩展"""
    _image = __rotate_expand(image.convert('RGB'), angle, SSAA, method)
    if alpha is not None:
        _alpha = __rotate_expand(alpha.convert('RGB'), angle, SSAA, method)
        ret_image = RGB2RGBA(_image, _alpha)
    else:
        ret_image = _image
        _alpha = Image.new('L', _image.size, 255)
    return (_image, _alpha.convert('L'), ret_image)

def chop_image_v2(image, mask, invert_mask, blend_mode, opacity):
    """图像混合处理"""
    # 简化的混合模式实现
    if blend_mode == "normal":
        # 正常混合模式
        if invert_mask:
            # 反转遮罩
            mask_array = np.array(mask)
            mask_array = 255 - mask_array
            mask = Image.fromarray(mask_array)
        
        # 应用透明度
        if opacity < 100:
            mask_array = np.array(mask).astype(np.float32)
            mask_array = mask_array * (opacity / 100.0)
            mask = Image.fromarray(mask_array.astype(np.uint8))
        
        return RGB2RGBA(image, mask)
    else:
        # 其他混合模式暂时使用正常模式
        return chop_image_v2(image, mask, invert_mask, "normal", opacity)

# 混合模式列表
chop_mode_v2 = [
    "normal", "dissolve", "darken", "multiply", "color burn", "linear burn", "darker color",
    "lighten", "screen", "color dodge", "linear dodge(add)", "lighter color", "dodge",
    "overlay", "soft light", "hard light", "vivid light", "linear light", "pin light",
    "hard mix", "difference", "exclusion", "subtract", "divide", "hue", "saturation",
    "color", "luminosity", "grain extract", "grain merge"
]

class ImageBlendAdvanceMy:
    """增强版图像混合节点 - 基于ComfyUI_LayerStyle的ImageBlendAdvanceV3，新增背景遮罩功能"""
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "layer_image": ("IMAGE",),
                "invert_mask": (["False", "True"],),
                "blend_mode": (chop_mode_v2,),
                "opacity": ("INT", {"default": 100, "min": 0, "max": 100, "step": 1}),
                "x_percent": ("INT", {"default": 50, "min": -999, "max": 999, "step": 1}),
                "y_percent": ("INT", {"default": 50, "min": -999, "max": 999, "step": 1}),
                "mirror": (["None", "horizontal", "vertical", "both"],),
                "scale": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 100.0, "step": 0.01}),
                "aspect_ratio": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 100.0, "step": 0.01}),
                "rotate": ("FLOAT", {"default": 0, "min": -999, "max": 999, "step": 0.1}),
                "transform_method": (["lanczos", "bicubic", "hamming", "bilinear", "box", "nearest"],),
                "anti_aliasing": ("INT", {"default": 0, "min": 0, "max": 16, "step": 1}),
                # 新增背景遮罩参数
                "background_mask": ("MASK",),
                "invert_background_mask": (["False", "True"],),
            },
            "optional": {
                "background_image": ("IMAGE",),
                "layer_mask": ("MASK",),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "image_blend_advance_my"
    CATEGORY = "😺dzNodes/LayerUtility"
    
    def image_blend_advance_my(self, layer_image, invert_mask, blend_mode, opacity,
                              x_percent, y_percent, mirror, scale, aspect_ratio, rotate,
                              transform_method, anti_aliasing, background_mask, invert_background_mask,
                              background_image=None, layer_mask=None):
        
        # 处理背景图像
        if background_image is not None:
            background = tensor2pil(background_image)
        else:
            background = tensor2pil(layer_image).convert('RGB')
        
        # 处理图层图像和遮罩
        layer = tensor2pil(layer_image).convert('RGB')
        if layer_mask is not None:
            mask = mask2image(layer_mask)
        else:
            mask = Image.new('L', layer.size, 255)
        
        # 处理背景遮罩
        bg_mask = mask2image(background_mask)
        if invert_background_mask == "True":
            bg_mask_array = np.array(bg_mask)
            bg_mask_array = 255 - bg_mask_array
            bg_mask = Image.fromarray(bg_mask_array)
        
        # 调整图层尺寸以匹配背景
        if layer.size != background.size:
            layer = layer.resize(background.size, Image.LANCZOS)
            mask = mask.resize(background.size, Image.LANCZOS)
        
        # 处理镜像
        if mirror == "horizontal":
            layer = layer.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        elif mirror == "vertical":
            layer = layer.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
        elif mirror == "both":
            layer = layer.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.FLIP_TOP_BOTTOM)
        
        # 处理缩放和宽高比
        if scale != 1.0 or aspect_ratio != 1.0:
            new_width = int(layer.width * scale * aspect_ratio)
            new_height = int(layer.height * scale)
            layer = layer.resize((new_width, new_height), Image.LANCZOS)
            mask = mask.resize((new_width, new_height), Image.LANCZOS)
        
        # 处理旋转
        if rotate != 0:
            layer, mask, _ = image_rotate_extend_with_alpha(layer, rotate, mask, transform_method, anti_aliasing)
        
        # 处理位置
        if layer.size != background.size or x_percent != 50 or y_percent != 50:
            # 创建新的图层和遮罩，尺寸与背景相同
            new_layer = Image.new('RGB', background.size, (0, 0, 0))
            new_mask = Image.new('L', background.size, 0)
            
            # 计算位置
            x = int((background.width - layer.width) * x_percent / 100)
            y = int((background.height - layer.height) * y_percent / 100)
            
            # 粘贴图层和遮罩
            new_layer.paste(layer, (x, y))
            new_mask.paste(mask, (x, y))
            
            layer = new_layer
            mask = new_mask
        
        # 图层合成
        invert_mask_bool = invert_mask == "True"
        composite_image = chop_image_v2(layer, mask, invert_mask_bool, blend_mode, opacity)
        
        # 背景合成 - 新增功能：在背景遮罩为黑色的区域显示原始背景
        final_image = Image.new('RGB', background.size, (0, 0, 0))
        final_mask = Image.new('L', background.size, 0)
        
        # 将composite_image转换为RGB用于合成
        if composite_image.mode == 'RGBA':
            comp_rgb = composite_image.convert('RGB')
            comp_alpha = composite_image.split()[3]
        else:
            comp_rgb = composite_image.convert('RGB')
            comp_alpha = mask
        
        # 确保所有图像尺寸一致
        if bg_mask.size != background.size:
            bg_mask = bg_mask.resize(background.size, Image.LANCZOS)
        if comp_rgb.size != background.size:
            comp_rgb = comp_rgb.resize(background.size, Image.LANCZOS)
        if comp_alpha.size != background.size:
            comp_alpha = comp_alpha.resize(background.size, Image.LANCZOS)
        
        # 使用背景遮罩进行合成
        bg_mask_array = np.array(bg_mask).astype(np.float32) / 255.0
        comp_alpha_array = np.array(comp_alpha).astype(np.float32) / 255.0
        
        # 在背景遮罩为白色的区域使用合成图像，黑色区域使用原始背景
        final_image_array = np.array(background).astype(np.float32)
        comp_rgb_array = np.array(comp_rgb).astype(np.float32)
        
        # 应用背景遮罩
        for c in range(3):
            final_image_array[:, :, c] = (
                bg_mask_array * comp_rgb_array[:, :, c] + 
                (1 - bg_mask_array) * final_image_array[:, :, c]
            )
        
        final_image = Image.fromarray(final_image_array.astype(np.uint8))
        
        # 更新最终遮罩：结合图层遮罩和背景遮罩
        final_mask_array = bg_mask_array * comp_alpha_array
        final_mask = Image.fromarray((final_mask_array * 255).astype(np.uint8), mode='L')
        
        return (pil2tensor(final_image), image2mask(final_mask))

# 节点映射
NODE_CLASS_MAPPINGS = {
    "ImageBlendAdvanceMy": ImageBlendAdvanceMy
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageBlendAdvanceMy": "ImageBlend Advance My"
}