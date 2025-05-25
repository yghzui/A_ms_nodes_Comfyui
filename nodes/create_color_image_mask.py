import torch
import numpy as np
import re

class CreateColorImageAndMask:
    def __init__(self):
        self.color = "#FF0000"  # 默认红色
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "color": ("STRING", {"default": "#FF0000", "multiline": False}),
                "invert_mask": ("BOOLEAN", {"default": False}),
                "default_n": ("INT", {"default": 1, "min": 1, "max": 64}),
                "default_h": ("INT", {"default": 512, "min": 64, "max": 4096}),
                "default_w": ("INT", {"default": 512, "min": 64, "max": 4096}),
            },
            "optional": {
                "reference_image": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "generate"
    CATEGORY = "image/color"

    def parse_color(self, color_str):
        """解析颜色字符串，支持hex和dec格式"""
        # 移除所有空白字符
        color_str = ''.join(color_str.split())
        
        # 检查是否为hex格式 (#RRGGBB 或 RRGGBB)
        if color_str.startswith('#'):
            color_str = color_str[1:]
        if re.match(r'^[0-9A-Fa-f]{6}$', color_str):
            return tuple(int(color_str[i:i+2], 16) for i in (0, 2, 4))
        
        # 检查是否为dec格式 (r,g,b 或 r g b)
        dec_match = re.findall(r'\d+', color_str)
        if len(dec_match) == 3:
            rgb = [min(255, max(0, int(x))) for x in dec_match]
            return tuple(rgb)
            
        # 如果都不匹配，返回默认红色
        return (255, 0, 0)

    def generate(self, color, invert_mask, default_n, default_h, default_w, reference_image=None):
        if reference_image is not None:
            n, h, w, c = reference_image.shape
        else:
            n, h, w = default_n, default_h, default_w
            c = 3
        
        # 解析颜色并归一化
        rgb = self.parse_color(color)
        rgb_normalized = [x/255.0 for x in rgb]
        
        # 创建指定颜色的图像
        color_image = torch.ones((n, h, w, c), dtype=torch.float32)
        for i in range(3):
            color_image[..., i] *= rgb_normalized[i]
            
        # 创建mask，根据invert_mask决定是白色还是黑色
        mask_value = 0.0 if invert_mask else 1.0
        white_mask = torch.ones((n, h, w), dtype=torch.float32) * mask_value
        
        return (color_image, white_mask)

