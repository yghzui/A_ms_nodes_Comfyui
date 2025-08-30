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

def chop_image_v2(background_image, layer_image, blend_mode, opacity):
    """图像混合处理 - 与原函数签名一致"""
    # 简化的混合模式实现，主要支持normal模式
    if blend_mode == "normal":
        # 将图像转换为RGBA格式
        bg = background_image.convert('RGBA')
        layer = layer_image.convert('RGBA')
        
        # 应用透明度
        if opacity < 100:
            # 调整图层的alpha通道
            r, g, b, a = layer.split()
            a = a.point(lambda x: int(x * opacity / 100))
            layer = Image.merge('RGBA', (r, g, b, a))
        
        # 使用PIL的alpha合成
        result = Image.alpha_composite(bg, layer)
        return result.convert('RGB')
    else:
        # 其他混合模式暂时使用正常模式
        return chop_image_v2(background_image, layer_image, "normal", opacity)

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
        self.NODE_NAME = 'ImageBlendAdvanceMy'
    
    @classmethod
    def INPUT_TYPES(cls):
        mirror_mode = ['None', 'horizontal', 'vertical']
        method_mode = ['lanczos', 'bicubic', 'hamming', 'bilinear', 'box', 'nearest']
        return {
            "required": {
                "background_image": ("IMAGE",),  # 背景图排在第一位且必须存在
                "layer_image": ("IMAGE",),
                "invert_mask": ("BOOLEAN", {"default": True}),  # 修正为BOOLEAN类型
                "blend_mode": (chop_mode_v2,),
                "opacity": ("INT", {"default": 100, "min": 0, "max": 100, "step": 1}),
                "x_percent": ("FLOAT", {"default": 50, "min": -999, "max": 999, "step": 0.01}),  # 修正为FLOAT类型
                "y_percent": ("FLOAT", {"default": 50, "min": -999, "max": 999, "step": 0.01}),  # 修正为FLOAT类型
                "mirror": (mirror_mode,),  # 移除both选项，与原函数保持一致
                "scale": ("FLOAT", {"default": 1, "min": 0.01, "max": 100, "step": 0.01}),
                "aspect_ratio": ("FLOAT", {"default": 1, "min": 0.01, "max": 100, "step": 0.01}),
                "rotate": ("FLOAT", {"default": 0, "min": -999999, "max": 999999, "step": 0.01}),
                "transform_method": (method_mode,),
                "anti_aliasing": ("INT", {"default": 0, "min": 0, "max": 16, "step": 1}),
            },
            "optional": {
                "layer_mask": ("MASK",),
                "background_mask": ("MASK",),  # 移到optional中
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "image_blend_advance_my"
    CATEGORY = "😺dzNodes/LayerUtility"
    
    def image_blend_advance_my(self, background_image, layer_image, invert_mask, blend_mode, opacity,
                              x_percent, y_percent, mirror, scale, aspect_ratio, rotate,
                              transform_method, anti_aliasing, layer_mask=None, background_mask=None):
        
        # 背景图像必须存在且排在第一位
        _canvas = tensor2pil(background_image).convert('RGBA')
        _layer = tensor2pil(layer_image)
        
        # 处理图层遮罩
        if layer_mask is not None:
            if invert_mask:
                _mask = tensor2pil(1 - layer_mask).convert('L')
            else:
                _mask = tensor2pil(layer_mask).convert('L')
        else:
            # 如果没有提供遮罩，从图层的alpha通道提取或创建白色遮罩
            if _layer.mode == 'RGBA':
                _mask = _layer.split()[-1]
            else:
                _mask = Image.new('L', _layer.size, 'white')
        
        # 确保遮罩尺寸匹配
        if _mask.size != _layer.size:
            _mask = Image.new('L', _layer.size, 'white')
            print(f"Warning: {self.NODE_NAME} mask mismatch, dropped!")
        
        # 记录原始图层尺寸
        orig_layer_width = _layer.width
        orig_layer_height = _layer.height
        _mask = _mask.convert("RGBA")
        
        # 计算目标尺寸
        target_layer_width = int(orig_layer_width * scale)
        target_layer_height = int(orig_layer_height * scale * aspect_ratio)
        
        # 处理镜像
        if mirror == 'horizontal':
            _layer = _layer.transpose(Image.FLIP_LEFT_RIGHT)
            _mask = _mask.transpose(Image.FLIP_LEFT_RIGHT)
        elif mirror == 'vertical':
            _layer = _layer.transpose(Image.FLIP_TOP_BOTTOM)
            _mask = _mask.transpose(Image.FLIP_TOP_BOTTOM)
        
        # 缩放
        _layer = _layer.resize((target_layer_width, target_layer_height))
        _mask = _mask.resize((target_layer_width, target_layer_height))
        
        # 旋转
        _layer, _mask, _ = image_rotate_extend_with_alpha(_layer, rotate, _mask, transform_method, anti_aliasing)
        
        # 计算位置
        x = int(_canvas.width * x_percent / 100 - _layer.width / 2)
        y = int(_canvas.height * y_percent / 100 - _layer.height / 2)
        
        # 合成图层 - 按照原函数逻辑
        import copy
        _comp = copy.copy(_canvas)
        _compmask = Image.new("RGBA", _comp.size, color='black')
        _comp.paste(_layer, (x, y))
        _compmask.paste(_mask, (x, y))
        _compmask = _compmask.convert('L')
        
        # 应用混合模式和透明度 - 修正参数顺序
        _comp = chop_image_v2(_canvas, _comp, blend_mode, opacity)
        
        # 如果有背景遮罩，应用背景遮罩逻辑
        if background_mask is not None:
            bg_mask = tensor2pil(background_mask).convert('L')
            if bg_mask.size != _canvas.size:
                bg_mask = bg_mask.resize(_canvas.size, Image.LANCZOS)
            
            # 背景遮罩区域：白色区域使用合成结果，黑色区域保持原背景
            bg_mask_array = np.array(bg_mask).astype(np.float32) / 255.0
            canvas_array = np.array(_canvas.convert('RGB')).astype(np.float32)
            comp_array = np.array(_comp.convert('RGB')).astype(np.float32)
            
            final_image_array = np.zeros_like(canvas_array)
            for c in range(3):
                final_image_array[:, :, c] = (
                    bg_mask_array * comp_array[:, :, c] + 
                    (1 - bg_mask_array) * canvas_array[:, :, c]
                )
            
            _canvas = Image.fromarray(final_image_array.astype(np.uint8))
            
            # 最终遮罩结合图层遮罩和背景遮罩
            comp_mask_array = np.array(_compmask).astype(np.float32) / 255.0
            final_mask_array = bg_mask_array * comp_mask_array
            _compmask = Image.fromarray((final_mask_array * 255).astype(np.uint8), mode='L')
        else:
            # 没有背景遮罩时，直接合成到背景
            _canvas.paste(_comp, mask=_compmask)
        
        print(f"{self.NODE_NAME} Processed 1 image.")
        # 修正mask维度：确保输出为(n,h,w)格式，符合ComfyUI标准
        mask_output = image2mask(_compmask)
        if mask_output.dim() == 4 and mask_output.shape[1] == 1:
            # 如果是(1,1,h,w)格式，去掉多余的通道维度，保持(1,h,w)
            mask_output = mask_output.squeeze(1)
        return (pil2tensor(_canvas), mask_output)

# 节点映射
NODE_CLASS_MAPPINGS = {
    "ImageBlendAdvanceMy": ImageBlendAdvanceMy
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageBlendAdvanceMy": "ImageBlend Advance My"
}