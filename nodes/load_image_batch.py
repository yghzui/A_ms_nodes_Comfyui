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
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                # 这个隐藏的输入字段将由前端的JS代码填充
                "image_paths": ("STRING", {"default": "", "multiline": False, "widget": "hidden"}),
                # 添加遮罩归一化选项
                "normalize_mask": ("BOOLEAN", {"default": True, "label": "归一化遮罩"}),
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

    def load_images(self, image_paths, normalize_mask=True, trigger=0):
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
                # 检查是否有alpha通道
                if img.mode == 'RGBA':
                    # 分离RGB和alpha通道
                    rgb_img = img.convert("RGB")
                    alpha = img.split()[-1]  # 获取alpha通道
                    # 将alpha转换为numpy数组，根据选项决定是否归一化
                    mask_np = np.array(alpha).astype(np.float32)
                    if normalize_mask:
                        mask_np = mask_np / 255.0
                else:
                    rgb_img = img.convert("RGB")
                    # 创建全黑mask（表示无遮罩）
                    mask_np = np.zeros((img.height, img.width), dtype=np.float32)
                
                # 处理RGB图像（图像始终归一化）
                image_np = np.array(rgb_img).astype(np.float32) / 255.0
                image_tensor = torch.from_numpy(image_np)
                
                # 处理mask
                mask_tensor = torch.from_numpy(mask_np)
                
                # 添加batch维度
                image_list.append(image_tensor.unsqueeze(0))
                mask_list.append(mask_tensor.unsqueeze(0))
                path_list.append(path)  # 保存有效的路径
            except Exception as e:
                print(f"错误: 加载文件失败 {image_path}, 原因: {e}")
        
        # 返回图像列表、mask列表和路径列表
        return (image_list, mask_list, path_list)

# 注意: NODE_CLASS_MAPPINGS 和 NODE_DISPLAY_NAME_MAPPINGS
# 将在 __init__.py 文件中进行管理，以避免冲突和保持代码整洁。
