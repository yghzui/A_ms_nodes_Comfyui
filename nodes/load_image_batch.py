import os
import torch
import numpy as np
from PIL import Image
import folder_paths

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

try:
    import cv2
except ImportError:
    cv2 = None

def load_image_with_fallback(image_path):
    """尝试使用 PIL 加载图片，如果失败则尝试使用 OpenCV"""
    try:
        img = Image.open(image_path)
        return img, False
    except Exception as pil_error:
        if cv2 is not None:
            try:
                # OpenCV 读取的是 BGR 格式
                cv_img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
                if cv_img is not None:
                    # 检查通道数
                    if len(cv_img.shape) == 3:
                        if cv_img.shape[2] == 4:
                            # BGRA -> RGBA
                            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGRA2RGBA)
                            img = Image.fromarray(cv_img, 'RGBA')
                        else:
                            # BGR -> RGB
                            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                            img = Image.fromarray(cv_img, 'RGB')
                    else:
                        # 灰度图
                        img = Image.fromarray(cv_img)
                    print(f"ℹ️ [LoadImageBatch] PIL 加载失败，已通过 OpenCV 成功重试: {image_path}")
                    return img, True
            except Exception as cv_error:
                print(f"❌ [LoadImageBatch] PIL 和 OpenCV 加载均失败: {image_path}, PIL: {pil_error}, CV: {cv_error}")
        else:
            print(f"❌ [LoadImageBatch] PIL 加载失败且 OpenCV 未安装: {image_path}, 原因: {pil_error}")
        raise pil_error

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
                "image_path_use": ("STRING", {"default": "", "multiline": False}),
                "reuse_mask": ("BOOLEAN", {"default": False, "label": "遮罩复用(同尺寸复用首个[input])"}),

            },
            "optional": {
                "batch_manager": ("MY_BATCH_MANAGER",),
            },
            "hidden": {
                # 添加遮罩归一化选项
                "normalize_mask": ("BOOLEAN", {"default": True, "label": "归一化遮罩"}),
                # 新增：是否将透明通道应用到图像（将alpha乘到RGB）
                "apply_alpha_to_image": ("BOOLEAN", {"default": True, "label": "应用透明到图像,避免透明通道显示异常"}),

            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "INT", "STRING",)
    RETURN_NAMES = ("image", "mask", "image_paths", "image_count", "paths_string",)
    OUTPUT_IS_LIST = (True, True, True, False, False,)
    FUNCTION = "load_images"
    CATEGORY = "A_my_nodes/Image"

    def load_images(self, image_paths, image_path_use="", normalize_mask=True, apply_alpha_to_image=False, reuse_mask=False, batch_manager=None):
        use_str = (image_path_use or '').strip() or (image_paths or '').strip()
        if not use_str:
            return ([], [], [], 0, "")

        raw_paths = [p.strip() for p in use_str.split(',') if p.strip()]
        if batch_manager is not None and len(raw_paths) > 0:
            batch_manager.total_count = len(raw_paths)
            batch_idx = batch_manager.current_index
            if batch_idx >= len(raw_paths):
                batch_idx = batch_idx % len(raw_paths)
            raw_paths = [raw_paths[batch_idx]]
        input_dir = folder_paths.get_input_directory()
        
        image_list = []
        mask_list = []
        path_list = []

        size_to_edited_mask = {}

        if reuse_mask:
            for raw_path in raw_paths:
                path_no_suffix = raw_path
                if path_no_suffix.endswith(" [input]"):
                    path_no_suffix = path_no_suffix[:-8]
                else:
                    continue

                if not path_no_suffix:
                    continue

                image_path = os.path.join(input_dir, path_no_suffix)
                if not os.path.exists(image_path):
                    continue

                try:
                    img, _ = load_image_with_fallback(image_path)
                    if img.mode == 'I':
                        img = img.point(lambda i: i * (1 / 255))

                    mask_np = None
                    if 'A' in img.getbands():
                        alpha = img.getchannel('A')
                        mask_np = np.array(alpha).astype(np.float32) / 255.0
                    elif img.mode == 'P' and 'transparency' in img.info:
                        rgba = img.convert('RGBA')
                        alpha = rgba.getchannel('A')
                        mask_np = np.array(alpha).astype(np.float32) / 255.0
                        img = rgba

                    rgb_img = img.convert("RGB")
                    width, height = rgb_img.size

                    if mask_np is None:
                        mask_np = np.zeros((height, width), dtype=np.float32)
                    else:
                        mask_np = 1.0 - mask_np
                        if normalize_mask:
                            mask_np = np.clip(mask_np, 0.0, 1.0)

                    size_key = (width, height)
                    if size_key not in size_to_edited_mask:
                        size_to_edited_mask[size_key] = mask_np
                except Exception as e:
                    print(f"错误: 预扫描遮罩失败 {image_path}, 原因: {e}")

        for idx, raw_path in enumerate(raw_paths):
            path_no_suffix = raw_path
            if path_no_suffix.endswith(" [input]"):
                path_no_suffix = path_no_suffix[:-8]

            if not path_no_suffix:
                continue

            image_path = os.path.join(input_dir, path_no_suffix)
            if not os.path.exists(image_path):
                print(f"警告: 文件不存在 {image_path}, 已跳过。")
                continue
            
            try:
                img, _ = load_image_with_fallback(image_path)
            except Exception as e:
                # 已经在 load_image_with_fallback 中打印了详细错误
                continue

            try:
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

                width, height = rgb_img.size

                # 生成遮罩：如果没有透明信息则输出全零遮罩
                if mask_np is None:
                    mask_np = np.zeros((height, width), dtype=np.float32)
                else:
                    # 与原生一致：mask = 1 - alpha
                    mask_np = 1.0 - mask_np
                    if normalize_mask:
                        # 仍保持 0..1 区间
                        mask_np = np.clip(mask_np, 0.0, 1.0)

                if reuse_mask and size_to_edited_mask:
                    size_key = (width, height)
                    if size_key in size_to_edited_mask:
                        mask_np = size_to_edited_mask[size_key]

                # 处理RGB图像（图像始终归一化）
                image_np = np.array(rgb_img).astype(np.float32) / 255.0

                # 可选：将透明通道应用到图像（将alpha乘到RGB，上面 mask = 1-alpha 不变）
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
                path_list.append(os.path.join(input_dir, path_no_suffix))
            except Exception as e:
                print(f"错误: 加载文件失败 {image_path}, 原因: {e}")
        
        # 计算图片数量
        image_count = len(image_list)
        
        # 将路径列表转换为字符串（用逗号分隔）
        paths_string = ','.join(path_list) if path_list else ""
        
        # 返回图像列表、mask列表、路径列表、图片数量和路径字符串
        return (image_list, mask_list, path_list, image_count, paths_string)

class LoadImageByIndex:
    """
    根据索引从路径字符串中加载指定的图像
    接收LoadImageBatchAdvanced的paths_string输出和索引，返回对应的单张图像
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "paths_string": ("STRING", {"default": "", "multiline": False}),
                "index": ("INT", {"default": 0, "min": 0, "max": 999999}),
                # 添加遮罩归一化选项
                "normalize_mask": ("BOOLEAN", {"default": True, "label": "归一化遮罩"}),
                # 是否将透明通道应用到图像
                "apply_alpha_to_image": ("BOOLEAN", {"default": False, "label": "应用透明到图像"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING",)
    RETURN_NAMES = ("image", "mask", "image_path",)
    OUTPUT_IS_LIST = (False, False, False,)
    FUNCTION = "load_image_by_index"
    CATEGORY = "A_my_nodes/Image"

    def load_image_by_index(self, paths_string, index, normalize_mask=True, apply_alpha_to_image=False):
        if not paths_string:
            # 如果没有路径字符串，返回空的tensor
            empty_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            empty_mask = torch.zeros((1, 64, 64), dtype=torch.float32)
            return (empty_image, empty_mask, "")

        # 将路径字符串转换为路径列表
        paths = [path.strip() for path in paths_string.split(',') if path.strip()]
        
        if not paths or index >= len(paths) or index < 0:
            # 索引超出范围，返回空的tensor
            empty_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            empty_mask = torch.zeros((1, 64, 64), dtype=torch.float32)
            return (empty_image, empty_mask, "")
        
        # 获取指定索引的路径
        target_path = paths[index]
        
        # 去除可能存在的 [input] 后缀
        if target_path.endswith(" [input]"):
            target_path = target_path[:-8]
        
        if not os.path.exists(target_path):
            print(f"警告: 文件不存在 {target_path}")
            empty_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            empty_mask = torch.zeros((1, 64, 64), dtype=torch.float32)
            return (empty_image, empty_mask, target_path)
        
        try:
            img, _ = load_image_with_fallback(target_path)
        except Exception as e:
            # 返回空张量，防止下游节点崩溃
            empty_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            empty_mask = torch.zeros((1, 64, 64), dtype=torch.float32)
            return (empty_image, empty_mask, target_path)

        try:
            # 处理特殊模式（与LoadImageBatchAdvanced保持一致）
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

            # 可选：将透明通道应用到图像
            if apply_alpha_to_image and has_alpha:
                # 上面 mask_np = 1 - alpha，因此 alpha = 1 - mask
                alpha_np = 1.0 - mask_np
                if alpha_np.ndim == 2:
                    alpha_np = alpha_np[:, :, None]
                image_np = image_np * alpha_np

            image_tensor = torch.from_numpy(image_np)
            mask_tensor = torch.from_numpy(mask_np)
            
            # 添加batch维度
            image_tensor = image_tensor.unsqueeze(0)
            mask_tensor = mask_tensor.unsqueeze(0)
            
            return (image_tensor, mask_tensor, target_path)
            
        except Exception as e:
            print(f"错误: 加载文件失败 {target_path}, 原因: {e}")
            empty_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            empty_mask = torch.zeros((1, 64, 64), dtype=torch.float32)
            return (empty_image, empty_mask, target_path)

# 注意: NODE_CLASS_MAPPINGS 和 NODE_DISPLAY_NAME_MAPPINGS
# 将在 __init__.py 文件中进行管理，以避免冲突和保持代码整洁。
