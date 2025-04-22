# -*- coding: utf-8 -*-
# Created time : 2025/04/22 23:11 
# Auther : ygh
# File   : resize_image_by_person.py
# Description :
import folder_paths
import os
import cv2
import torch
import numpy as np
import math
import logging
from ultralytics import YOLO
from custom_nodes.A_my_nodes.nodes.image_nodes import img2tensor, tensor2img

comfyui_model_path=folder_paths.models_dir
person_yolov8m_seg_path=os.path.join(comfyui_model_path,"ultralytics","segm","person_yolov8m-seg.pt")

MAX_RESOLUTION = 8192

# 添加测试函数
def person_detection(image_path, confidence=0.5):
    """
    测试人物检测并在图像上绘制边界框
    Args:
        image_path: 图像路径
        confidence: 检测置信度
    Returns:
        保存标注后的图像到同目录下的 "_detected.jpg" 文件
    """
    # 确保模型存在
    if not os.path.exists(person_yolov8m_seg_path):
        print(f"找不到人物检测模型: {person_yolov8m_seg_path}")
        return
    
    # 加载模型
    model = YOLO(person_yolov8m_seg_path)
    
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法加载图像: {image_path}")
        return
    
    # 检测人物
    results = model(img, conf=confidence, classes=0)  # 仅检测人物类别(0)
    
    if len(results) == 0 or len(results[0].boxes) == 0:
        print("未检测到人物")
        return
    
    # 获取所有人物的边界框
    all_boxes = results[0].boxes.xyxy.cpu().numpy()
    if len(all_boxes) == 0:
        print("未检测到人物边界框")
        return
    
    # 按照x坐标（左到右）排序
    sorted_boxes = sorted(all_boxes, key=lambda box: box[0])
    
    # 绘制边界框
    img_draw = img.copy()
    colors = [
        (0, 255, 0),    # 绿色
        (255, 0, 0),    # 蓝色 (OpenCV中为BGR顺序)
        (0, 0, 255),    # 红色
        (255, 255, 0),  # 青色
        (255, 0, 255),  # 品红色
        (0, 255, 255),  # 黄色
    ]
    
    box_thickness = 3
    font_scale = 1.0
    
    print(f"检测到 {len(sorted_boxes)} 个人物:")
    for idx, box in enumerate(sorted_boxes):
        color = colors[idx % len(colors)]
        x_min, y_min, x_max, y_max = map(int, box)
        
        # 绘制矩形
        cv2.rectangle(img_draw, (x_min, y_min), (x_max, y_max), color, box_thickness)
        
        # 显示索引编号
        text = f"Person {idx}"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, box_thickness)[0]
        cv2.rectangle(img_draw, (x_min, y_min - text_size[1] - 10), (x_min + text_size[0], y_min), color, -1)
        cv2.putText(img_draw, text, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), box_thickness)
        
        print(f"  Person {idx}: 位置 [{x_min}, {y_min}, {x_max}, {y_max}]")
    
    # 保存结果图像
    output_path = os.path.splitext(image_path)[0] + "_detected.jpg"
    cv2.imwrite(output_path, img_draw)
    print(f"已保存检测结果到: {output_path}")

# # 执行测试
# # 如果直接运行此脚本，则自动执行测试
# if __name__ == "__main__":
#     test_image_path = r"D:\AI\comfyui\input\mmexport1743693085119.jpg"
#     if os.path.exists(test_image_path):
#         print(f"正在测试图像: {test_image_path}")
#         person_detection(test_image_path, confidence=0.25)
#     else:
#         print(f"测试图像不存在: {test_image_path}")

class ResizeImageByPerson:
    @classmethod
    def INPUT_TYPES(s):
        scale_to_list = ['longest', 'None', 'shortest', 'width', 'height', 'total_pixel(kilo pixel)']
        interpolation_modes = ["nearest-exact", "bilinear", "area", "bicubic"]
        return {
            "required": {
                "images": ("IMAGE",),  # 输入图像张量
                "masks": ("MASK",),    # 输入遮罩张量
                "crop_by_person": ("BOOLEAN", {
                    "default": False, 
                    "tooltip": "是否使用person_yolov8m-seg模型检测人物，并在缩放前裁剪到仅包含人物区域"
                }),
                "person_indices": ("STRING", {
                    "default": "0", 
                    "tooltip": "要处理的人物索引，从左到右排序。输入0表示最左边人物；-1表示所有人物；多个索引用逗号分隔，如'0,1'"
                }),
                "merge_output": ("BOOLEAN", {
                    "default": False, 
                    "tooltip": "当选择多个人物时，是否合并输出为一个包含所有选中人物的最小边界框"
                }),
                "person_confidence": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.1,
                    "max": 1.0, 
                    "step": 0.05,
                    "tooltip": "人物检测置信度阈值"
                }),
                "padding_percent": ("INT", {
                    "default": 10,
                    "min": 0,
                    "max": 100,
                    "step": 1,
                    "tooltip": "人物裁剪边界框扩展百分比"
                }),
                "resize": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "是否使用指定的宽高调整图像尺寸(优先级高于scale_to_side)"
                }),
                "width": ("INT", {
                    "default": 512, 
                    "min": 0, 
                    "max": MAX_RESOLUTION, 
                    "step": 8,
                    "tooltip": "调整后的图像宽度，0表示保持原始宽度"
                }),
                "height": ("INT", {
                    "default": 512, 
                    "min": 0, 
                    "max": MAX_RESOLUTION, 
                    "step": 8,
                    "tooltip": "调整后的图像高度，0表示保持原始高度"
                }),
                "keep_proportion": ("BOOLEAN", {
                    "default": False, 
                    "tooltip": "在resize=True时，是否保持图像原始比例（会根据width/height计算最终尺寸）"
                }),
                "scale_to_side": (scale_to_list, {
                    "default": "None",
                    "tooltip": "在resize=False时，按照哪个边进行缩放"
                }),
                "scale_to_length": ("INT", {
                    "default": 1024, 
                    "min": 4, 
                    "max": 999999, 
                    "step": 1,
                    "tooltip": "缩放边的目标长度"
                }),
                "divisible_by": ("INT", {
                    "default": 2,
                    "min": 0,
                    "max": 512,
                    "step": 1,
                    "tooltip": "调整图像尺寸，使其可以被此数整除"
                }),
                "interpolation": (interpolation_modes, {
                    "default": "bilinear",
                    "tooltip": "图像缩放时使用的插值方法"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT")
    RETURN_NAMES = ("image", "mask", "person_count")
    FUNCTION = "resize_images_and_masks"
    CATEGORY = "My_node/image"

    def __init__(self):
        self.person_model = None

    def load_person_model(self):
        if self.person_model is None:
            if not os.path.exists(person_yolov8m_seg_path):
                raise FileNotFoundError(f"找不到人物检测模型: {person_yolov8m_seg_path}")
            self.person_model = YOLO(person_yolov8m_seg_path)
        return self.person_model

    def get_all_person_bboxes(self, img_np, confidence):
        """检测图像中的人物并返回按照从左到右排序的边界框列表"""
        # 记录输入图像形状
        logging.info(f"get_all_person_bboxes输入图像形状: {img_np.shape}")
        
        # 确保img_np形状正确 - YOLO模型需要[H, W, C]格式
        if len(img_np.shape) == 4:  # 如果形状是[N, H, W, C]
            img_np = img_np[0]  # 取第一个图像
            logging.info(f"转换后的图像形状: {img_np.shape}")
            
        model = self.load_person_model()
        results = model(img_np, conf=confidence, classes=0)  # 仅检测人物类别(0)
        
        if not results or len(results) == 0:
            print("未检测到人物")
            return []
        
        # 获取所有人物的边界框
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            print("未检测到人物边界框")
            return []
            
        # 获取边界框坐标
        all_boxes = boxes.xyxy.cpu().numpy()
        if len(all_boxes) == 0:
            print("未检测到人物边界框")
            return []
            
        # 按照x坐标（左到右）排序
        sorted_boxes = sorted(all_boxes, key=lambda box: box[0])
        
        return sorted_boxes

    def get_merged_bbox(self, boxes):
        """合并多个边界框为一个最小包含所有边界框的边界框"""
        if not boxes or len(boxes) == 0:
            return None
        
        x_min = min(box[0] for box in boxes)
        y_min = min(box[1] for box in boxes)
        x_max = max(box[2] for box in boxes)
        y_max = max(box[3] for box in boxes)
        
        return [x_min, y_min, x_max, y_max]

    def parse_indices(self, indices_str, max_idx):
        """解析用户输入的索引字符串"""
        try:
            # 如果输入-1，返回所有索引
            if indices_str.strip() == "-1":
                return list(range(max_idx + 1))
            
            # 解析逗号分隔的索引
            indices = []
            for idx in indices_str.split(','):
                idx = idx.strip()
                if idx:
                    idx_int = int(idx)
                    if 0 <= idx_int <= max_idx:
                        indices.append(idx_int)
            
            # 如果没有有效索引，默认返回第一个（索引0）
            if not indices and max_idx >= 0:
                return [0]
                
            return indices
        except ValueError:
            # 如果解析失败，返回默认索引0
            if max_idx >= 0:
                return [0]
            return []

    def resize_images_and_masks(self, images, masks, crop_by_person, person_indices, merge_output, 
                                person_confidence, padding_percent, resize, width, height, 
                                keep_proportion, scale_to_side, scale_to_length, divisible_by, interpolation):
        """
        主处理函数，按照以下逻辑顺序处理图像：
        1. 首先根据crop_by_person参数决定是否进行人物裁剪，以及如何裁剪
        2. 然后将裁剪后的图像传递给process_single_image进行尺寸调整
           - 如果resize=True，优先使用width和height参数
           - 如果resize=False但scale_to_side不是"None"，则使用scale_to_side和scale_to_length参数
        """
        output_images = []
        output_masks = []
        
        # 设置插值方法
        if interpolation == "nearest-exact":
            interp_method = cv2.INTER_NEAREST_EXACT
        elif interpolation == "bilinear":
            interp_method = cv2.INTER_LINEAR
        elif interpolation == "area":
            interp_method = cv2.INTER_AREA
        elif interpolation == "bicubic":
            interp_method = cv2.INTER_CUBIC
        else:
            interp_method = cv2.INTER_LINEAR
            
        # 遮罩应该始终使用最近邻插值以保持边缘清晰
        mask_interp_method = cv2.INTER_NEAREST
        
        total_person_count = 0

        # 打印输入图像的形状，确认格式
        logging.info(f"输入图像形状: {images.shape}")  # 应该是 [N, H, W, C]

        for img_idx, (img, mask) in enumerate(zip(images, masks)):
            # 在ComfyUI中，图像格式为[H, W, C]
            # 确保img是正确的形状 (不需要额外的批次维度)
            img_np = img.cpu().numpy()  # 直接从tensor转为numpy，保留原始维度
            mask_np = mask.cpu().numpy()
            
            # 记录图像形状，帮助调试
            logging.info(f"图像{img_idx}形状: {img_np.shape}")
            
            # 处理裁剪部分 - 步骤1：根据是否需要裁剪人物来确定处理的图像
            # 正确获取图像的高度和宽度，基于实际形状
            if len(img_np.shape) == 4:  # 如果形状是[N, H, W, C]
                h, w = img_np.shape[1:3]  # 获取高度和宽度
                # 需要将[N, H, W, C]转换为[H, W, C]供后续处理
                img_np = img_np[0]  # 取第一个图像
            else:  # 如果形状是[H, W, C]
                h, w = img_np.shape[0:2]
                
            # 同样处理mask
            if len(mask_np.shape) == 3:  # 如果形状是[N, H, W]
                mask_np = mask_np[0]  # 取第一个mask
            
            # 创建一个图像列表和掩码列表，用于存储需要处理的图像
            # 如果不需要裁剪或没有检测到人物，它只会包含原始图像
            # 如果需要裁剪且检测到人物，它会包含裁剪后的图像
            images_to_process = []
            masks_to_process = []
            
            if crop_by_person:
                # 获取所有人物的边界框
                # 对于YOLO检测，需要转换为BGR格式
                img_np_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR) if img_np.shape[2] == 3 else img_np
                all_person_boxes = self.get_all_person_bboxes(img_np_bgr, person_confidence)
                total_person_count = len(all_person_boxes)
                
                if total_person_count > 0:
                    # 解析用户提供的索引
                    valid_indices = self.parse_indices(person_indices, total_person_count - 1)
                    
                    # 根据索引选择边界框
                    selected_boxes = [all_person_boxes[idx] for idx in valid_indices if idx < total_person_count]
                    
                    if selected_boxes:
                        # 如果需要合并输出或只有一个边界框
                        if merge_output or len(selected_boxes) == 1:
                            bbox = self.get_merged_bbox(selected_boxes)
                            
                            if bbox is not None:
                                x_min, y_min, x_max, y_max = bbox
                                
                                # 计算填充范围
                                padding_w = (x_max - x_min) * (padding_percent / 100)
                                padding_h = (y_max - y_min) * (padding_percent / 100)
                                
                                # 应用填充，同时确保边界不会超出图像
                                x_min = max(0, x_min - padding_w)
                                y_min = max(0, y_min - padding_h)
                                x_max = min(w, x_max + padding_w)
                                y_max = min(h, y_max + padding_h)
                                
                                # 裁剪图像和遮罩
                                crop_img = img_np[int(y_min):int(y_max), int(x_min):int(x_max)]
                                crop_mask = mask_np[int(y_min):int(y_max), int(x_min):int(x_max)]
                                
                                # 添加到处理列表
                                images_to_process.append(crop_img)
                                masks_to_process.append(crop_mask)
                        else:
                            # 处理多个独立的边界框
                            for box in selected_boxes:
                                x_min, y_min, x_max, y_max = box
                                
                                # 计算填充范围
                                padding_w = (x_max - x_min) * (padding_percent / 100)
                                padding_h = (y_max - y_min) * (padding_percent / 100)
                                
                                # 应用填充，同时确保边界不会超出图像
                                x_min = max(0, x_min - padding_w)
                                y_min = max(0, y_min - padding_h)
                                x_max = min(w, x_max + padding_w)
                                y_max = min(h, y_max + padding_h)
                                
                                # 裁剪图像和遮罩
                                crop_img = img_np[int(y_min):int(y_max), int(x_min):int(x_max)]
                                crop_mask = mask_np[int(y_min):int(y_max), int(x_min):int(x_max)]
                                
                                # 添加到处理列表
                                images_to_process.append(crop_img)
                                masks_to_process.append(crop_mask)
                    else:
                        # 没有选中任何有效的边界框，处理整个图像
                        images_to_process.append(img_np)
                        masks_to_process.append(mask_np)
                else:
                    # 没有检测到人物，处理整个图像
                    images_to_process.append(img_np)
                    masks_to_process.append(mask_np)
            else:
                # 不裁剪，直接处理整个图像
                images_to_process.append(img_np)
                masks_to_process.append(mask_np)
            
            # 步骤2：缩放处理 - 对每个裁剪后的图像应用尺寸调整
            for proc_img, proc_mask in zip(images_to_process, masks_to_process):
                # 调用process_single_image处理缩放逻辑
                processed_img, processed_mask = self.process_single_image(
                    proc_img, proc_mask, resize, width, height, scale_to_side, 
                    scale_to_length, keep_proportion, divisible_by, 
                    interp_method, mask_interp_method
                )
                
                # 收集处理后的图像和掩码
                output_images.append(processed_img)
                output_masks.append(processed_mask)

        # 如果没有处理任何图像，返回原始图像
        if not output_images:
            logging.info(f"没有处理任何图像，返回原始图像: {images.shape}, {masks.shape}")
            return images, masks, torch.tensor([total_person_count], dtype=torch.int32)
        
        # 处理输出
        if len(output_images) == 1:
            # 如果只有一个图像，直接返回而不需要拼接
            result_img = output_images[0]
            result_mask = output_masks[0]
            logging.info(f"单图像输出形状: {result_img.shape}, {result_mask.shape}")
            return result_img, result_mask, torch.tensor([total_person_count], dtype=torch.int32)
        else:
            # 如果有多个图像，确保它们有相同的形状，然后拼接
            # 过滤掉形状不一致的图像
            first_img_shape = output_images[0].shape[1:]  # 忽略批次维度
            first_mask_shape = output_masks[0].shape[1:]  # 忽略批次维度
            
            filtered_images = []
            filtered_masks = []
            for img, mask in zip(output_images, output_masks):
                if img.shape[1:] == first_img_shape and mask.shape[1:] == first_mask_shape:
                    filtered_images.append(img)
                    filtered_masks.append(mask)
            
            if not filtered_images:
                # 如果没有有效图像，返回原始图像
                logging.info(f"没有有效的过滤图像，返回原始图像: {images.shape}, {masks.shape}")
                return images, masks, torch.tensor([total_person_count], dtype=torch.int32)
            
            # 拼接所有有效的图像和掩码
            final_images = torch.cat(filtered_images, dim=0)
            final_masks = torch.cat(filtered_masks, dim=0)
            
            logging.info(f"多图像拼接后输出形状: {final_images.shape}, {final_masks.shape}")
            return final_images, final_masks, torch.tensor([total_person_count], dtype=torch.int32)

    def process_single_image(self, img_np, mask_np, resize, width, height, scale_to_side, 
                            scale_to_length, keep_proportion, divisible_by, 
                            interp_method, mask_interp_method):
        """处理单个图像的缩放逻辑，按照明确的优先级：
        1. 如果resize=True, 则使用width和height参数(keep_proportion可能会影响)
        2. 如果resize=False但scale_to_side不是"None", 则使用scale_to_side和scale_to_length
        3. 如果都不使用，保持原始尺寸
        
        输入:
            img_np: numpy格式的图像，格式为[H, W, C]
            mask_np: numpy格式的掩码，格式为[H, W]
            
        输出:
            img_tensor: tensor格式的图像，格式为[1, H, W, C]
            mask_tensor: tensor格式的掩码，格式为[1, H, W]
        """
        # 记录输入形状，帮助调试
        logging.info(f"process_single_image输入图像形状: {img_np.shape}")
        if hasattr(mask_np, 'shape'):
            logging.info(f"process_single_image输入掩码形状: {mask_np.shape}")
            
        # 确保img_np形状正确
        if len(img_np.shape) == 4:  # 如果形状是[N, H, W, C]
            img_np = img_np[0]  # 取第一个图像
            
        # 确保mask_np形状正确
        if len(mask_np.shape) == 3:  # 如果形状是[N, H, W]
            mask_np = mask_np[0]  # 取第一个mask
            
        h, w = img_np.shape[0:2]
        target_width, target_height = w, h  # 默认保持原始尺寸
        
        # 第一优先级：使用resize参数
        if resize:
            # 如果resize开启，以width和height为基准
            target_width = width if width > 0 else w
            target_height = height if height > 0 else h
            
            # 应用keep_proportion（如果启用）
            if keep_proportion:
                ratio = min(target_width / w, target_height / h)
                target_width = round(w * ratio)
                target_height = round(h * ratio)
                
            # 确保尺寸可被divisible_by整除
            if divisible_by > 1:
                target_width = target_width - (target_width % divisible_by)
                target_height = target_height - (target_height % divisible_by)
        
        # 第二优先级：如果未resize但使用scale_to_side
        elif scale_to_side != 'None':
            ratio = w / h  # 原始宽高比
            
            if scale_to_side == 'longest':
                # 按照长边缩放
                if w >= h:  # 宽边更长
                    target_width = scale_to_length
                    target_height = int(target_width / ratio)
                else:  # 高边更长
                    target_height = scale_to_length
                    target_width = int(target_height * ratio)
            
            elif scale_to_side == 'shortest':
                # 按照短边缩放
                if w < h:  # 宽边更短
                    target_width = scale_to_length
                    target_height = int(target_width / ratio)
                else:  # 高边更短
                    target_height = scale_to_length
                    target_width = int(target_height * ratio)
            
            elif scale_to_side == 'width':
                # 强制按照宽度缩放
                target_width = scale_to_length
                target_height = int(target_width / ratio)
            
            elif scale_to_side == 'height':
                # 强制按照高度缩放
                target_height = scale_to_length
                target_width = int(target_height * ratio)
            
            elif scale_to_side == 'total_pixel(kilo pixel)':
                # 按照总像素数缩放（千像素）
                target_width = int(math.sqrt(ratio * scale_to_length * 1000))
                target_height = int(target_width / ratio)
                
            # 确保尺寸可被divisible_by整除
            if divisible_by > 1:
                target_width = target_width - (target_width % divisible_by)
                target_height = target_height - (target_height % divisible_by)
        
        # 确保尺寸至少为1x1
        target_width = max(1, target_width)
        target_height = max(1, target_height)

        # 调整图像大小
        img_resized = cv2.resize(img_np, (target_width, target_height), interpolation=interp_method)
        mask_resized = cv2.resize(mask_np, (target_width, target_height), interpolation=mask_interp_method)

        # 在ComfyUI中，图像格式应该是[B, H, W, C]
        # 将numpy图像转换为tensor，并确保格式正确
        if img_resized.dtype == np.uint8:
            # 如果是uint8，需要归一化为0-1
            img_tensor = torch.from_numpy(img_resized.astype(np.float32) / 255.0)
        else:
            # 如果已经是float类型，确保是float32
            img_tensor = torch.from_numpy(img_resized.astype(np.float32))
        
        # 确保图像是[B, H, W, C]格式
        if len(img_tensor.shape) == 3:  # [H, W, C]
            img_tensor = img_tensor.unsqueeze(0)  # 添加批次维度 [1, H, W, C]
        
        # 对于掩码，ComfyUI期望格式为[B, H, W]
        if mask_resized.dtype == np.uint8:
            mask_tensor = torch.from_numpy(mask_resized.astype(np.float32) / 255.0)
        else:
            mask_tensor = torch.from_numpy(mask_resized.astype(np.float32))
            
        if len(mask_tensor.shape) == 2:  # [H, W]
            mask_tensor = mask_tensor.unsqueeze(0)  # 添加批次维度 [1, H, W]
        elif len(mask_tensor.shape) == 3 and mask_tensor.shape[2] == 1:  # [H, W, 1]
            mask_tensor = mask_tensor.squeeze(-1).unsqueeze(0)  # 转换为 [1, H, W]
            
        return img_tensor, mask_tensor
