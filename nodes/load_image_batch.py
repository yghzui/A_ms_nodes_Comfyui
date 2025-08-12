import os
import torch
import numpy as np
from PIL import Image
import folder_paths

class LoadImageBatchAdvanced:
    """
    一个高级的图像批量加载节点，功能如下：
    1. 提供一个按钮，用于打开文件选择对话框并支持多选。
    2. 在节点上显示所选图像的缩略图列表。
    3. 点击缩略图可以放大预览。
    4. 输出包含原始尺寸图像的列表 (IMAGE)、它们对应的路径列表 (STRING) 和mask列表 (MASK)。
    5. 可选择是否对mask进行归一化处理（0-1）。
    6. 新增：可选择是否将透明通道应用到图像（将alpha乘到RGB）。
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                # 这个隐藏的输入字段将由前端的JS代码填充
                "image_paths": ("STRING", {"default": "", "multiline": False, "widget": "hidden"}),
                # 添加遮罩归一化选项
                "normalize_mask": ("BOOLEAN", {"default": True, "label": "归一化遮罩"}),
                # 新增：是否将透明通道应用到图像（将alpha乘到RGB）
                "apply_alpha_to_image": ("BOOLEAN", {"default": False, "label": "应用透明到图像"}),
            },
            "optional": {
                # 增加一个刷新开关，当用户重新选择相同文件时也能强制刷新
                "trigger": ("INT", {"default": 0, "min": 0, "max": 0xffffffffff, "widget": "hidden"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING",)
    RETURN_NAMES = ("image", "mask", "image_paths",)
    OUTPUT_IS_LIST = (True, True, True,)
    FUNCTION = "load_images"
    CATEGORY = "A_my_nodes/Image"

    def load_images(self, image_paths, normalize_mask=True, apply_alpha_to_image=False, trigger=0):
        if not image_paths:
            # 如果没有图像路径，返回三个空列表
            return ([], [], [],)

        paths = image_paths.split(',')
        input_dir = folder_paths.get_input_directory()
        
        image_list = []
        mask_list = []
        path_list = []

        for path in paths:
            path = path.strip()
            if not path:
                continue
            
            image_path = os.path.join(input_dir, path)
            if not os.path.exists(image_path):
                print(f"警告: 文件不存在 {image_path}, 已跳过。")
                continue
            
            try:
                img = Image.open(image_path)

                # 处理特殊模式（参考原始实现）
                if img.mode == 'I':
                    # 将 32-bit 整型图近似归一化到 0..1
                    img = img.point(lambda i: i * (1 / 255))

                # 透明通道与调色板透明
                mask_np = None
                has_alpha = False
                if 'A' in img.getbands():
                    # 直接从 alpha 读取
                    alpha = img.getchannel('A')
                    mask_np = np.array(alpha).astype(np.float32) / 255.0
                    has_alpha = True
                elif img.mode == 'P' and 'transparency' in img.info:
                    # 调色板带透明，转 RGBA 后取 alpha
                    rgba = img.convert('RGBA')
                    alpha = rgba.getchannel('A')
                    mask_np = np.array(alpha).astype(np.float32) / 255.0
                    img = rgba  # 后续从 RGBA 转 RGB
                    has_alpha = True

                # 始终输出 RGB 图像（与 Comfy IMAGE 类型一致）
                rgb_img = img.convert("RGB")

                # 生成遮罩：如果没有透明信息则输出全零遮罩
                if mask_np is None:
                    mask_np = np.zeros((rgb_img.height, rgb_img.width), dtype=np.float32)
                else:
                    # 与原生一致：mask = 1 - alpha
                    mask_np = 1.0 - mask_np
                    if normalize_mask:
                        # 仍保持 0..1 区间
                        mask_np = np.clip(mask_np, 0.0, 1.0)

                # 处理RGB图像（图像始终归一化）
                image_np = np.array(rgb_img).astype(np.float32) / 255.0

                # 可选：将透明通道应用到图像（将 alpha 乘到 RGB，上面 mask = 1-alpha 不变）
                if apply_alpha_to_image and has_alpha:
                    # 上面 mask_np = 1 - alpha，因此 alpha = 1 - mask
                    alpha_np = 1.0 - mask_np
                    if alpha_np.ndim == 2:
                        alpha_np = alpha_np[:, :, None]
                    image_np = image_np * alpha_np

                image_tensor = torch.from_numpy(image_np)
                mask_tensor = torch.from_numpy(mask_np)
                
                # 添加batch维度
                image_list.append(image_tensor.unsqueeze(0))
                mask_list.append(mask_tensor.unsqueeze(0))
                path_list.append(os.path.join(input_dir, path))  # 保存有效的路径
            except Exception as e:
                print(f"错误: 加载文件失败 {image_path}, 原因: {e}")
        
        # 返回图像列表、mask列表和路径列表
        return (image_list, mask_list, path_list)

# 注意: NODE_CLASS_MAPPINGS 和 NODE_DISPLAY_NAME_MAPPINGS
# 将在 __init__.py 文件中进行管理，以避免冲突和保持代码整洁。
