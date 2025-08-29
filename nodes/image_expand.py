# -*- coding: utf-8 -*-
# Created time : 2024/12/19
# Auther : AI Assistant
# File   : image_expand.py
# Description : 图像扩展节点，支持四边扩展并生成遮罩

import torch
import torch.nn.functional as F
import re

class ImageExpand:
    """
    图像扩展节点
    对输入图像的四个边进行指定颜色的外扩，同时生成扩展区域的遮罩
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),  # 输入图像 (n,h,w,c)
                "color": ("STRING", {
                    "default": "#000000",
                    "multiline": False,
                    "tooltip": "扩展颜色，支持HEX格式(#FF0000)或RGB格式(255,0,0)"
                }),
                "left": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 2048,
                    "step": 1,
                    "tooltip": "左边扩展像素数"
                }),
                "right": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 2048,
                    "step": 1,
                    "tooltip": "右边扩展像素数"
                }),
                "top": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 2048,
                    "step": 1,
                    "tooltip": "上边扩展像素数"
                }),
                "bottom": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 2048,
                    "step": 1,
                    "tooltip": "下边扩展像素数"
                }),
            },
            "optional": {
                "mask": ("MASK", {
                    "tooltip": "可选的原始遮罩，如果提供则与扩展遮罩合并"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK", "MASK")
    RETURN_NAMES = ("expanded_image", "expansion_mask", "combined_mask")
    FUNCTION = "expand_image"
    CATEGORY = "image/transform"
    
    def parse_color(self, color_str):
        """
        解析颜色字符串，支持HEX和RGB格式
        
        Args:
            color_str: 颜色字符串，如"#FF0000"或"255,0,0"
            
        Returns:
            tuple: (r, g, b) 归一化的RGB值 (0-1范围)
        """
        color_str = color_str.strip()
        
        # 处理HEX格式
        if color_str.startswith('#'):
            hex_color = color_str[1:]
            if len(hex_color) == 6:
                r = int(hex_color[0:2], 16) / 255.0
                g = int(hex_color[2:4], 16) / 255.0
                b = int(hex_color[4:6], 16) / 255.0
                return (r, g, b)
            elif len(hex_color) == 3:
                r = int(hex_color[0], 16) / 15.0
                g = int(hex_color[1], 16) / 15.0
                b = int(hex_color[2], 16) / 15.0
                return (r, g, b)
        
        # 处理RGB格式
        # 匹配 "255,0,0" 或 "(255,0,0)" 或 "255 0 0" 等格式
        rgb_match = re.findall(r'\d+', color_str)
        if len(rgb_match) >= 3:
            r = int(rgb_match[0]) / 255.0
            g = int(rgb_match[1]) / 255.0
            b = int(rgb_match[2]) / 255.0
            return (r, g, b)
        
        # 默认返回黑色
        print(f"警告: 无法解析颜色 '{color_str}'，使用默认黑色")
        return (0.0, 0.0, 0.0)
    
    def expand_image(self, image, color, left, right, top, bottom, mask=None):
        """
        扩展图像并生成遮罩
        
        Args:
            image: 输入图像张量 (n,h,w,c)
            color: 扩展颜色字符串
            left, right, top, bottom: 各边扩展像素数
            mask: 可选的原始遮罩
            
        Returns:
            tuple: (扩展后图像, 扩展遮罩, 合并遮罩)
        """
        # 解析颜色
        r, g, b = self.parse_color(color)
        
        # 获取原始图像尺寸
        n, h, w, c = image.shape
        
        # 计算扩展后的尺寸
        new_h = h + top + bottom
        new_w = w + left + right
        
        # 创建扩展后的图像张量，填充指定颜色
        expanded_image = torch.zeros((n, new_h, new_w, c), dtype=image.dtype, device=image.device)
        
        # 填充背景颜色
        if c >= 3:  # RGB或RGBA
            expanded_image[:, :, :, 0] = r  # R通道
            expanded_image[:, :, :, 1] = g  # G通道
            expanded_image[:, :, :, 2] = b  # B通道
            if c == 4:  # RGBA，Alpha通道设为1
                expanded_image[:, :, :, 3] = 1.0
        else:  # 灰度图
            # 使用RGB的平均值作为灰度值
            gray_value = (r + g + b) / 3.0
            expanded_image[:, :, :, 0] = gray_value
        
        # 将原始图像放置在正确位置
        expanded_image[:, top:top+h, left:left+w, :] = image
        
        # 创建扩展区域的遮罩
        expansion_mask = torch.zeros((n, new_h, new_w), dtype=torch.float32, device=image.device)
        
        # 标记扩展区域为1
        if top > 0:
            expansion_mask[:, :top, :] = 1.0  # 上边扩展区域
        if bottom > 0:
            expansion_mask[:, top+h:, :] = 1.0  # 下边扩展区域
        if left > 0:
            expansion_mask[:, :, :left] = 1.0  # 左边扩展区域
        if right > 0:
            expansion_mask[:, :, left+w:] = 1.0  # 右边扩展区域
        
        # 处理合并遮罩
        combined_mask = expansion_mask.clone()
        
        if mask is not None:
            # 确保原始遮罩尺寸匹配
            if mask.shape[-2:] == (h, w):  # 检查高度和宽度
                # 如果原始遮罩的批次维度不匹配，进行调整
                if mask.shape[0] != n:
                    if mask.shape[0] == 1:
                        # 如果原始遮罩只有一个批次，复制到所有批次
                        mask = mask.repeat(n, 1, 1)
                    else:
                        # 如果批次数不匹配且不为1，取第一个
                        mask = mask[:1].repeat(n, 1, 1)
                
                # 将原始遮罩放置在合并遮罩的对应位置
                combined_mask[:, top:top+h, left:left+w] = torch.maximum(
                    combined_mask[:, top:top+h, left:left+w], 
                    mask
                )
            else:
                print(f"警告: 原始遮罩尺寸 {mask.shape[-2:]} 与图像尺寸 {(h, w)} 不匹配，忽略原始遮罩")
        
        return (expanded_image, expansion_mask, combined_mask)