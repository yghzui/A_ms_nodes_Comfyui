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
                "resize": ("BOOLEAN", {"default": False}),
                "width": ("INT", {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
                "height": ("INT", {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
                "scale_to_side": (scale_to_list,),
                "scale_to_length": ("INT", {"default": 1024, "min": 4, "max": 999999, "step": 1}),
                "keep_proportion": ("BOOLEAN", {"default": False, "tooltip": "保持比例"}),
                "divisible_by": ("INT", {
                    "default": 2,
                    "min": 0,
                    "max": 512,
                    "step": 1,
                    "tooltip": "调整图像尺寸，使其可以被此数整除。"
                }),
                "interpolation": (interpolation_modes, {
                    "default": "bilinear",
                    "tooltip": "图像缩放时使用的插值方法"
                }),
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

    def resize_images_and_masks(self, images, masks, resize, width, height, scale_to_side, scale_to_length, 
                                keep_proportion, divisible_by, interpolation, crop_by_person, 
                                person_indices, merge_output, person_confidence, padding_percent):
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

        for img, mask in zip(images, masks):
            img_np = tensor2img(img)
            mask_np = mask.cpu().numpy()
            
            # 处理裁剪部分
            h, w = img_np.shape[0:2]
            orig_h, orig_w = h, w
            
            if crop_by_person:
                # 获取所有人物的边界框
                all_person_boxes = self.get_all_person_bboxes(img_np, person_confidence)
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
                                img_np = img_np[int(y_min):int(y_max), int(x_min):int(x_max)]
                                mask_np = mask_np[int(y_min):int(y_max), int(x_min):int(x_max)]
                                
                                # 处理缩放部分并添加到输出
                                processed_img, processed_mask = self.process_single_image(
                                    img_np, mask_np, resize, width, height, scale_to_side, 
                                    scale_to_length, keep_proportion, divisible_by, 
                                    interp_method, mask_interp_method
                                )
                                
                                output_images.append(processed_img)
                                output_masks.append(processed_mask)
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
                                
                                # 处理缩放部分并添加到输出
                                processed_img, processed_mask = self.process_single_image(
                                    crop_img, crop_mask, resize, width, height, scale_to_side, 
                                    scale_to_length, keep_proportion, divisible_by, 
                                    interp_method, mask_interp_method
                                )
                                
                                output_images.append(processed_img)
                                output_masks.append(processed_mask)
                    else:
                        # 没有选中任何有效的边界框，处理整个图像
                        processed_img, processed_mask = self.process_single_image(
                            img_np, mask_np, resize, width, height, scale_to_side, 
                            scale_to_length, keep_proportion, divisible_by, 
                            interp_method, mask_interp_method
                        )
                        
                        output_images.append(processed_img)
                        output_masks.append(processed_mask)
                else:
                    # 没有检测到人物，处理整个图像
                    processed_img, processed_mask = self.process_single_image(
                        img_np, mask_np, resize, width, height, scale_to_side, 
                        scale_to_length, keep_proportion, divisible_by, 
                        interp_method, mask_interp_method
                    )
                    
                    output_images.append(processed_img)
                    output_masks.append(processed_mask)
            else:
                # 不裁剪，直接处理整个图像
                processed_img, processed_mask = self.process_single_image(
                    img_np, mask_np, resize, width, height, scale_to_side, 
                    scale_to_length, keep_proportion, divisible_by, 
                    interp_method, mask_interp_method
                )
                
                output_images.append(processed_img)
                output_masks.append(processed_mask)

        if not output_images:
            # 如果没有输出，返回原始图像
            return images, masks, torch.tensor([total_person_count], dtype=torch.int32)
        
        # 确保所有图像和遮罩都是相同尺寸，才能进行拼接
        # 创建一个批次的图像和遮罩，而不是在维度0上拼接
        output_images_tensor = torch.stack(output_images, dim=0)
        output_masks_tensor = torch.stack(output_masks, dim=0)
            
        return output_images_tensor, output_masks_tensor, torch.tensor([total_person_count], dtype=torch.int32)

    def process_single_image(self, img_np, mask_np, resize, width, height, scale_to_side, 
                            scale_to_length, keep_proportion, divisible_by, 
                            interp_method, mask_interp_method):
        """处理单个图像的缩放逻辑"""
        h, w = img_np.shape[0:2]
        
        # 处理缩放部分
        if resize:
            if keep_proportion:
                ratio = min(width / w, height / h)
                width = round(w * ratio)
                height = round(h * ratio)
            else:
                if width == 0:
                    width = w
                if height == 0:
                    height = h

            if divisible_by > 1:
                width = width - (width % divisible_by)
                height = height - (height % divisible_by)
        else:
            width, height = w, h

        ratio = width / height
        if ratio > 1:
            if scale_to_side == 'longest':
                target_width = scale_to_length
                target_height = int(target_width / ratio)
            elif scale_to_side == 'shortest':
                target_height = scale_to_length
                target_width = int(target_height * ratio)
            elif scale_to_side == 'width':
                target_width = scale_to_length
                target_height = int(target_width / ratio)
            elif scale_to_side == 'height':
                target_height = scale_to_length
                target_width = int(target_height * ratio)
            elif scale_to_side == 'total_pixel(kilo pixel)':
                target_width = math.sqrt(ratio * scale_to_length * 1000)
                target_height = target_width / ratio
                target_width = int(target_width)
                target_height = int(target_height)
            else:
                target_width = width
                target_height = int(target_width / ratio)
        else:
            if scale_to_side == 'longest':
                target_height = scale_to_length
                target_width = int(target_height * ratio)
            elif scale_to_side == 'shortest':
                target_width = scale_to_length
                target_height = int(target_width / ratio)
            elif scale_to_side == 'width':
                target_width = scale_to_length
                target_height = int(target_width / ratio)
            elif scale_to_side == 'height':
                target_height = scale_to_length
                target_width = int(target_height * ratio)
            elif scale_to_side == 'total_pixel(kilo pixel)':
                target_width = math.sqrt(ratio * scale_to_length * 1000)
                target_height = target_width / ratio
                target_width = int(target_width)
                target_height = int(target_height)
            else:
                target_height = height
                target_width = int(target_height * ratio)

        # 确保尺寸至少为1x1
        target_width = max(1, target_width)
        target_height = max(1, target_height)

        img_resized = cv2.resize(img_np, (target_width, target_height), interpolation=interp_method)
        mask_resized = cv2.resize(mask_np, (target_width, target_height), interpolation=mask_interp_method)

        # 将numpy数组转回tensor，确保维度正确
        img_tensor = img2tensor(img_resized)
        mask_tensor = torch.from_numpy(mask_resized).float().unsqueeze(0)  # 添加批次维度

        return img_tensor, mask_tensor
