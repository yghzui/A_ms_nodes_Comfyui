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
import hashlib

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
                "crop_by_person": ("BOOLEAN", {
                    "default": False, 
                    "tooltip": "是否使用person_yolov8m-seg模型检测人物，并在缩放前裁剪到仅包含人物区域"
                }),
                "use_largest_person": ("BOOLEAN", {
                    "default": False, 
                    "tooltip": "是否只处理检测到的最大人物框（面积最大）"
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
            },
            "optional": {
                "masks": ("MASK", {
                    "tooltip": "输入遮罩张量，可选。如果不提供，将创建全黑的遮罩"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "STRING")
    RETURN_NAMES = ("image", "mask", "person_count", "crop_info")
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
        
        # 确保是三通道彩色图像
        if len(img_np.shape) != 3 or img_np.shape[2] != 3:
            logging.error(f"图像不是三通道彩色图像: {img_np.shape}")
            return []
        
        # 记录图像范围，帮助判断是否需要归一化
        img_min = np.min(img_np)
        img_max = np.max(img_np)
        logging.info(f"图像数值范围: min={img_min}, max={img_max}")
        
        # 如果图像是float类型且范围在[0,1]之间，转换为uint8
        if img_np.dtype == np.float32 or img_np.dtype == np.float64:
            if img_max <= 1.0:
                img_np = (img_np * 255).astype(np.uint8)
                logging.info(f"将[0,1]浮点图像转换为uint8: {img_np.shape}, {img_np.dtype}")
        
        # 确保图像是BGR格式 (YOLO需要BGR格式)
        # 注意: 在调用该函数前，图像已经被转换为BGR格式了，这里做最后的确认
        # 如果图像是从ComfyUI获取的，它通常是RGB格式
        logging.info(f"确认图像已经是BGR格式")
            
        try:
            model = self.load_person_model()
            # 打印图像的一些基本信息，帮助诊断
            logging.info(f"送入YOLO的图像信息: 形状={img_np.shape}, 类型={img_np.dtype}, 范围=[{np.min(img_np)},{np.max(img_np)}]")
            results = model(img_np, conf=confidence, classes=0)  # 仅检测人物类别(0)
            
            if not results or len(results) == 0:
                logging.warning("YOLO未检测到任何结果")
                return []
            
            # 获取所有人物的边界框
            boxes = results[0].boxes
            if boxes is None or len(boxes) == 0:
                logging.warning("未检测到人物边界框")
                return []
                
            # 获取边界框坐标
            all_boxes = boxes.xyxy.cpu().numpy()
            if len(all_boxes) == 0:
                logging.warning("转换后未获得有效边界框")
                return []
                
            # 按照x坐标（左到右）排序
            sorted_boxes = sorted(all_boxes, key=lambda box: box[0])
            logging.info(f"检测到{len(sorted_boxes)}个人物边界框")
            
            return sorted_boxes
        except Exception as e:
            logging.error(f"人物检测过程中发生错误: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            return []

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

    def resize_images_and_masks(self, images, crop_by_person, use_largest_person, person_indices, merge_output, 
                                person_confidence, padding_percent, resize, width, height, 
                                keep_proportion, scale_to_side, scale_to_length, divisible_by, interpolation,
                                masks=None):
        """
        主处理函数，按照以下逻辑顺序处理图像：
        1. 首先根据crop_by_person参数决定是否进行人物裁剪，以及如何裁剪
           - 如果use_largest_person=True，则只处理面积最大的人物框
           - 否则，根据person_indices选择要处理的人物
        2. 然后将裁剪后的图像传递给process_single_image进行尺寸调整
           - 如果resize=True，优先使用width和height参数
           - 如果resize=False但scale_to_side不是"None"，则使用scale_to_side和scale_to_length参数
        
        参数:
            images: 输入图像张量
            masks: 输入掩码张量，可选。如果不提供，将创建全黑的掩码
            crop_by_person: 是否使用人物检测进行裁剪
            use_largest_person: 是否只处理面积最大的人物框
            person_indices: 要处理的人物索引字符串
            merge_output: 处理多个人物时是否合并输出
            其他参数: 控制缩放和输出格式的参数
            
        返回:
            处理后的图像、掩码和检测到的人物数量
        """
        output_images = []
        output_masks = []
        crop_boxes = []  # 新增：用于存储裁剪框信息
        
        # 如果未提供masks，创建与images相同批次的全黑mask
        if masks is None:
            logging.info("未提供masks，创建全黑mask")
            # 创建一个与images相同批次和尺寸的全黑mask
            # masks格式为[batch, height, width]
            masks = torch.zeros((images.shape[0], images.shape[1], images.shape[2]), 
                                dtype=torch.float32, device=images.device)
        
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
        logging.info(f"输入遮罩形状: {masks.shape}")  # 应该是 [N, H, W]

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
                # YOLO模型需要BGR格式图像，但ComfyUI中通常是RGB格式
                logging.info(f"原始图像形状和类型: {img_np.shape}, {img_np.dtype}")
                
                # 确保图像是正确的数值范围
                if img_np.dtype == np.float32 or img_np.dtype == np.float64:
                    if np.max(img_np) <= 1.0:
                        img_for_detection = (img_np * 255).astype(np.uint8)
                    else:
                        img_for_detection = img_np.astype(np.uint8)
                else:
                    img_for_detection = img_np
                
                # 确保图像是BGR格式
                if img_for_detection.shape[2] == 3:  # 确保有3个通道
                    img_for_detection = cv2.cvtColor(img_for_detection, cv2.COLOR_RGB2BGR)
                    logging.info(f"转换RGB到BGR完成")
                else:
                    logging.warning(f"图像不是3通道，无法进行RGB到BGR转换: {img_for_detection.shape}")
                
                try:
                    all_person_boxes = self.get_all_person_bboxes(img_for_detection, person_confidence)
                    total_person_count = len(all_person_boxes)
                    logging.info(f"检测到{total_person_count}个人物")
                except Exception as e:
                    logging.error(f"人物检测失败: {str(e)}")
                    all_person_boxes = []
                    total_person_count = 0
                
                if total_person_count > 0:
                    try:
                        # 如果启用了只处理最大人物框选项
                        if use_largest_person and total_person_count > 1:
                            # 计算每个边界框的面积
                            areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in all_person_boxes]
                            # 找出面积最大的边界框索引
                            largest_idx = areas.index(max(areas))
                            logging.info(f"选择面积最大的人物框 #{largest_idx}, 面积: {areas[largest_idx]}像素")
                            # 只保留最大的人物框
                            all_person_boxes = [all_person_boxes[largest_idx]]
                            # 更新有效索引为单个索引0
                            valid_indices = [0]
                        else:
                            # 解析用户提供的索引
                            valid_indices = self.parse_indices(person_indices, total_person_count - 1)
                        
                        # 根据索引选择边界框
                        selected_boxes = [all_person_boxes[idx] for idx in valid_indices if idx < total_person_count]
                        
                        if selected_boxes:
                            # 根据resize或scale_to_side计算目标尺寸
                            target_width, target_height = self.calculate_target_size(
                                w, h, resize, width, height, scale_to_side, 
                                scale_to_length, keep_proportion, divisible_by
                            )
                            logging.info(f"计算得到的目标尺寸: {target_width}x{target_height}")
                            
                            # 目标宽高比
                            target_ratio = target_width / target_height
                            
                            # 如果需要合并输出或只有一个边界框
                            if merge_output or len(selected_boxes) == 1:
                                bbox = self.get_merged_bbox(selected_boxes)
                                
                                if bbox is not None:
                                    # 使用智能扩展边界框函数
                                    x_min, y_min, x_max, y_max = self.extend_bbox_to_target_ratio(
                                        bbox, target_ratio, w, h, padding_percent
                                    )
                                    
                                    # 裁剪图像和遮罩
                                    crop_img = img_np[int(y_min):int(y_max), int(x_min):int(x_max)]
                                    crop_mask = mask_np[int(y_min):int(y_max), int(x_min):int(x_max)] if mask_np is not None else None
                                    
                                    # 记录裁剪框信息
                                    crop_info = f"[{int(x_min)},{int(y_min)},{int(x_max)},{int(y_max)}]"
                                    crop_boxes.append(crop_info)
                                    
                                    logging.info(f"智能扩展后的人物区域: {crop_img.shape}, 实际比例: {crop_img.shape[1]/crop_img.shape[0]}, 目标比例: {target_ratio}")
                                    
                                    # 添加到处理列表
                                    images_to_process.append(crop_img)
                                    masks_to_process.append(crop_mask)
                                else:
                                    logging.warning("无法获取合并边界框，使用原始图像")
                                    images_to_process.append(img_np)
                                    masks_to_process.append(mask_np)
                                    # 使用原图尺寸，位置从[0,0]开始
                                    crop_boxes.append(f"[0,0,{w},{h}]")
                            else:
                                # 处理多个独立的边界框
                                for box_idx, box in enumerate(selected_boxes):
                                    # 使用智能扩展边界框函数
                                    x_min, y_min, x_max, y_max = self.extend_bbox_to_target_ratio(
                                        box, target_ratio, w, h, padding_percent
                                    )
                                    
                                    # 裁剪图像和遮罩
                                    crop_img = img_np[int(y_min):int(y_max), int(x_min):int(x_max)]
                                    crop_mask = mask_np[int(y_min):int(y_max), int(x_min):int(x_max)] if mask_np is not None else None
                                    
                                    # 记录裁剪框信息
                                    crop_info = f"[{int(x_min)},{int(y_min)},{int(x_max)},{int(y_max)}]"
                                    crop_boxes.append(crop_info)
                                    
                                    logging.info(f"智能扩展后的人物{box_idx}: {crop_img.shape}, 实际比例: {crop_img.shape[1]/crop_img.shape[0]}, 目标比例: {target_ratio}")
                                    
                                    # 添加到处理列表
                                    images_to_process.append(crop_img)
                                    masks_to_process.append(crop_mask)
                        else:
                            # 没有选中任何有效的边界框，处理整个图像
                            logging.info("没有选中有效边界框，处理整个图像")
                            images_to_process.append(img_np)
                            masks_to_process.append(mask_np)
                            # 使用原图尺寸，位置从[0,0]开始
                            crop_boxes.append(f"[0,0,{w},{h}]")
                    except Exception as e:
                        logging.error(f"边界框处理失败: {str(e)}")
                        images_to_process.append(img_np)
                        masks_to_process.append(mask_np)
                        # 使用原图尺寸，位置从[0,0]开始
                        crop_boxes.append(f"[0,0,{w},{h}]")
                else:
                    # 没有检测到人物，处理整个图像
                    logging.info("未检测到人物，处理整个图像")
                    images_to_process.append(img_np)
                    masks_to_process.append(mask_np)
                    # 使用原图尺寸，位置从[0,0]开始
                    crop_boxes.append(f"[0,0,{w},{h}]")
            else:
                # 不裁剪，直接处理整个图像
                logging.info("不进行人物裁剪，处理整个图像")
                images_to_process.append(img_np)
                masks_to_process.append(mask_np)
                # 使用原图尺寸，位置从[0,0]开始
                crop_boxes.append(f"[0,0,{w},{h}]")
            
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
            # 使用原图尺寸，位置从[0,0]开始
            return images, masks, total_person_count, f"[0,0,{w},{h}]"
        
        # 处理输出
        if len(output_images) == 1:
            # 如果只有一个图像，直接返回而不需要拼接
            result_img = output_images[0]
            result_mask = output_masks[0]
            # 如果没有裁剪框信息，使用原图尺寸
            crop_info = crop_boxes[0] if crop_boxes else f"[0,0,{w},{h}]"
            logging.info(f"单图像输出形状: {result_img.shape}, {result_mask.shape}")
            return result_img, result_mask, total_person_count, crop_info
        else:
            # 如果有多个图像，确保它们有相同的形状，然后拼接
            # 过滤掉形状不一致的图像
            first_img_shape = output_images[0].shape[1:]  # 忽略批次维度
            first_mask_shape = output_masks[0].shape[1:]  # 忽略批次维度
            
            filtered_images = []
            filtered_masks = []
            filtered_crop_boxes = []
            for i, (img, mask) in enumerate(zip(output_images, output_masks)):
                if img.shape[1:] == first_img_shape and mask.shape[1:] == first_mask_shape:
                    filtered_images.append(img)
                    filtered_masks.append(mask)
                    if i < len(crop_boxes):
                        filtered_crop_boxes.append(crop_boxes[i])
            
            if not filtered_images:
                # 如果没有有效图像，返回原始图像
                logging.info(f"没有有效的过滤图像，返回原始图像: {images.shape}, {masks.shape}")
                # 使用原图尺寸，位置从[0,0]开始
                return images, masks, total_person_count, f"[0,0,{w},{h}]"
            
            # 拼接所有有效的图像和掩码
            final_images = torch.cat(filtered_images, dim=0)
            final_masks = torch.cat(filtered_masks, dim=0)
            
            # 合并裁剪框信息为字符串
            crop_info = ", ".join(filtered_crop_boxes) if filtered_crop_boxes else f"[0,0,{w},{h}]"
            
            logging.info(f"多图像拼接后输出形状: {final_images.shape}, {final_masks.shape}")
            return final_images, final_masks, total_person_count, crop_info

    # @classmethod
    # def IS_CHANGED(s, images, crop_by_person, use_largest_person, person_indices, merge_output,
    #                             person_confidence, padding_percent, resize, width, height,
    #                             keep_proportion, scale_to_side, scale_to_length, divisible_by, interpolation,
    #                             masks=None):
    #     # 计算输入的哈希值，确保只有在输入变化时才重新计算
    #     m = hashlib.sha256()

    #     # 对图像进行哈希
    #     images_flat = images.reshape(-1).numpy().tobytes()
    #     m.update(images_flat[:1024])  # 只使用部分数据做哈希，避免计算过重

    #     # 如果存在masks，也对它进行哈希
    #     if masks is not None:
    #         masks_flat = masks.reshape(-1).numpy().tobytes()
    #         m.update(masks_flat[:1024])

    #     # 将其他参数也加入哈希计算
    #     m.update(str(crop_by_person).encode())
    #     m.update(str(use_largest_person).encode())
    #     m.update(str(person_indices).encode())
    #     m.update(str(merge_output).encode())
    #     m.update(str(person_confidence).encode())
    #     m.update(str(padding_percent).encode())
    #     m.update(str(resize).encode())
    #     m.update(str(width).encode())
    #     m.update(str(height).encode())
    #     m.update(str(keep_proportion).encode())
    #     m.update(str(scale_to_side).encode())
    #     m.update(str(scale_to_length).encode())
    #     m.update(str(divisible_by).encode())
    #     m.update(str(interpolation).encode())

    #     return m.digest().hex()


    def process_single_image(self, img_np, mask_np, resize, width, height, scale_to_side, 
                            scale_to_length, keep_proportion, divisible_by, 
                            interp_method, mask_interp_method):
        """处理单个图像的缩放逻辑
        
        输入:
            img_np: numpy格式的图像，格式为[H, W, C]
            mask_np: numpy格式的掩码，格式为[H, W]，可以为None
            
        输出:
            img_tensor: tensor格式的图像，格式为[1, H, W, C]
            mask_tensor: tensor格式的掩码，格式为[1, H, W]
        """
        # 记录输入形状，帮助调试
        logging.info(f"process_single_image输入图像形状: {img_np.shape}")
        if mask_np is None:
            logging.info("process_single_image: mask_np为None，将创建全黑mask")
            # 创建一个与图像相同尺寸的全黑mask
            mask_np = np.zeros((img_np.shape[0], img_np.shape[1]), dtype=np.uint8)
        elif hasattr(mask_np, 'shape'):
            logging.info(f"process_single_image输入掩码形状: {mask_np.shape}")
            
        # 确保img_np形状正确
        if len(img_np.shape) == 4:  # 如果形状是[N, H, W, C]
            img_np = img_np[0]  # 取第一个图像
            
        # 确保mask_np形状正确
        if mask_np is not None and len(mask_np.shape) == 3:  # 如果形状是[N, H, W]
            mask_np = mask_np[0]  # 取第一个mask
            
        # 检查输入图像有效性
        if img_np.size == 0 or (mask_np is not None and hasattr(mask_np, 'size') and mask_np.size == 0):
            logging.error("输入图像或掩码为空")
            # 创建一个最小的有效图像和掩码
            dummy_img = np.zeros((64, 64, 3), dtype=np.uint8)
            dummy_mask = np.zeros((64, 64), dtype=np.uint8)
            # 返回dummy图像的tensor
            return (torch.from_numpy(dummy_img).float() / 255.0).unsqueeze(0), torch.from_numpy(dummy_mask).float().unsqueeze(0)
            
        h, w = img_np.shape[0:2]
        logging.info(f"原始尺寸: {w}x{h}, 是否调整大小: {resize}")
        
        # 计算目标尺寸
        target_width, target_height = self.calculate_target_size(
            w, h, resize, width, height, scale_to_side, 
            scale_to_length, keep_proportion, divisible_by
        )
        
        # 如果预处理已经将图像调整到接近目标比例，现在只需进行较小的缩放
        current_ratio = w / h
        target_ratio = target_width / target_height
        
        logging.info(f"当前比例: {current_ratio:.4f}, 目标比例: {target_ratio:.4f}, 比例差异: {abs(current_ratio - target_ratio):.4f}")
        
        # 如果当前比例已经非常接近目标比例（误差小于5%），可以直接缩放
        # 否则，可能需要轻微的裁剪或填充来达到精确的目标比例
        if abs(current_ratio - target_ratio) > 0.05:
            logging.info("当前比例与目标比例差异较大，调整比例...")
            # 创建一个正确比例的画布
            if current_ratio > target_ratio:
                # 宽度过大，计算新的宽度
                new_width = int(h * target_ratio)
                crop_x_offset = (w - new_width) // 2
                img_np = img_np[:, crop_x_offset:crop_x_offset+new_width]
                mask_np = mask_np[:, crop_x_offset:crop_x_offset+new_width]
                logging.info(f"裁剪宽度到新尺寸: {img_np.shape}")
            else:
                # 高度过大，计算新的高度
                new_height = int(w / target_ratio)
                crop_y_offset = (h - new_height) // 2
                img_np = img_np[crop_y_offset:crop_y_offset+new_height, :]
                mask_np = mask_np[crop_y_offset:crop_y_offset+new_height, :]
                logging.info(f"裁剪高度到新尺寸: {img_np.shape}")

        # 调整图像大小
        try:
            logging.info(f"最终调整尺寸: {target_width}x{target_height}")
            img_resized = cv2.resize(img_np, (target_width, target_height), interpolation=interp_method)
            mask_resized = cv2.resize(mask_np, (target_width, target_height), interpolation=mask_interp_method)
            logging.info(f"调整后的图像形状: {img_resized.shape}")
        except Exception as e:
            logging.error(f"调整尺寸失败: {str(e)}")
            # 如果调整失败，尝试使用更安全的方法
            try:
                img_resized = np.zeros((target_height, target_width, 3), dtype=img_np.dtype)
                mask_resized = np.zeros((target_height, target_width), dtype=mask_np.dtype)
                logging.warning(f"创建了空白图像代替调整失败的图像: {img_resized.shape}")
            except Exception as e2:
                logging.error(f"创建空白图像也失败: {str(e2)}")
                # 使用原始图像
                img_resized = img_np
                mask_resized = mask_np

        # 在ComfyUI中，图像格式应该是[B, H, W, C]
        # 将numpy图像转换为tensor，并确保格式正确
        try:
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
                
            logging.info(f"处理后的图像形状: {img_tensor.shape}, 掩码形状: {mask_tensor.shape}")
            return img_tensor, mask_tensor
        except Exception as e:
            logging.error(f"转换为tensor失败: {str(e)}")
            # 创建一个最小的有效图像和掩码
            dummy_img = np.zeros((64, 64, 3), dtype=np.uint8)
            dummy_mask = np.zeros((64, 64), dtype=np.uint8)
            # 返回dummy图像的tensor
            return (torch.from_numpy(dummy_img).float() / 255.0).unsqueeze(0), torch.from_numpy(dummy_mask).float().unsqueeze(0)

    def calculate_target_size(self, w, h, resize, width, height, scale_to_side, 
                            scale_to_length, keep_proportion, divisible_by):
        """计算目标尺寸"""
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
                old_tw, old_th = target_width, target_height
                target_width = target_width - (target_width % divisible_by)
                target_height = target_height - (target_height % divisible_by)
                if old_tw != target_width or old_th != target_height:
                    logging.info(f"调整为可被{divisible_by}整除的尺寸: {target_width}x{target_height}")
        
        # 第二优先级：如果未resize但使用scale_to_side
        elif scale_to_side != 'None':
            logging.info(f"使用scale_to_side={scale_to_side}, scale_to_length={scale_to_length}")
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
                old_tw, old_th = target_width, target_height
                target_width = target_width - (target_width % divisible_by)
                target_height = target_height - (target_height % divisible_by)
                if old_tw != target_width or old_th != target_height:
                    logging.info(f"调整为可被{divisible_by}整除的尺寸: {target_width}x{target_height}")
                    
            logging.info(f"按{scale_to_side}缩放后的目标尺寸: {target_width}x{target_height}")
        
        # 确保尺寸至少为1x1
        target_width = max(1, target_width)
        target_height = max(1, target_height)

        return target_width, target_height

    def extend_bbox_to_target_ratio(self, bbox, target_ratio, w, h, padding_percent):
        """扩展边界框到目标比例
        
        Args:
            bbox: 原始边界框 [x_min, y_min, x_max, y_max]
            target_ratio: 目标宽高比 (width/height)
            w, h: 原图宽高
            padding_percent: 基础填充百分比
            
        Returns:
            扩展后的边界框 [x_min, y_min, x_max, y_max]
        """
        x_min, y_min, x_max, y_max = bbox
        
        # 记录原始边界框
        bbox_width = x_max - x_min
        bbox_height = y_max - y_min
        orig_ratio = bbox_width / bbox_height
        
        logging.info(f"原始边界框: 宽={bbox_width}, 高={bbox_height}, 比例={orig_ratio:.4f}, 目标比例={target_ratio:.4f}")
        
        # 首先应用基础填充，保持原有宽高比
        padding_w = (x_max - x_min) * (padding_percent / 100)
        padding_h = (y_max - y_min) * (padding_percent / 100)
        
        x_min_padded = max(0, x_min - padding_w)
        y_min_padded = max(0, y_min - padding_h)
        x_max_padded = min(w, x_max + padding_w)
        y_max_padded = min(h, y_max + padding_h)
        
        # 基础填充后的尺寸
        bbox_width_padded = x_max_padded - x_min_padded
        bbox_height_padded = y_max_padded - y_min_padded
        padded_ratio = bbox_width_padded / bbox_height_padded
        
        logging.info(f"基础填充后: 宽={bbox_width_padded}, 高={bbox_height_padded}, 比例={padded_ratio:.4f}")
        
        # 计算向目标比例扩展需要的额外像素
        if abs(padded_ratio - target_ratio) < 0.01:
            # 如果已经非常接近目标比例，不需要额外调整
            logging.info("基础填充后已接近目标比例，无需额外调整")
            return [x_min_padded, y_min_padded, x_max_padded, y_max_padded]
            
        # 调整比例：选择最小的改动来达到目标比例
        if padded_ratio < target_ratio:
            # 当前太窄，需要横向扩展
            # 计算目标宽度
            target_width = bbox_height_padded * target_ratio
            width_diff = target_width - bbox_width_padded
            
            # 计算左右各需添加的像素数
            add_left = width_diff / 2
            add_right = width_diff / 2
            
            # 检查是否超出图像边界
            if x_min_padded - add_left < 0:
                # 左边超出边界，将多余的分配给右边
                add_right += (add_left - x_min_padded)
                add_left = x_min_padded
            
            if x_max_padded + add_right > w:
                # 右边超出边界，将多余的分配给左边
                add_left += (add_right - (w - x_max_padded))
                add_right = w - x_max_padded
                
                # 再次检查左边是否超出
                if x_min_padded - add_left < 0:
                    add_left = x_min_padded
            
            # 应用横向扩展
            x_min_final = x_min_padded - add_left
            x_max_final = x_max_padded + add_right
            y_min_final = y_min_padded
            y_max_final = y_max_padded
            
            logging.info(f"横向扩展: 左侧添加={add_left:.1f}px, 右侧添加={add_right:.1f}px")
        else:
            # 当前太宽，需要纵向扩展
            # 计算目标高度
            target_height = bbox_width_padded / target_ratio
            height_diff = target_height - bbox_height_padded
            
            # 计算上下各需添加的像素数
            add_top = height_diff / 2
            add_bottom = height_diff / 2
            
            # 检查是否超出图像边界
            if y_min_padded - add_top < 0:
                # 上边超出边界，将多余的分配给下边
                add_bottom += (add_top - y_min_padded)
                add_top = y_min_padded
            
            if y_max_padded + add_bottom > h:
                # 下边超出边界，将多余的分配给上边
                add_top += (add_bottom - (h - y_max_padded))
                add_bottom = h - y_max_padded
                
                # 再次检查上边是否超出
                if y_min_padded - add_top < 0:
                    add_top = y_min_padded
            
            # 应用纵向扩展
            x_min_final = x_min_padded
            x_max_final = x_max_padded
            y_min_final = y_min_padded - add_top
            y_max_final = y_max_padded + add_bottom
            
            logging.info(f"纵向扩展: 上方添加={add_top:.1f}px, 下方添加={add_bottom:.1f}px")
        
        # 确保最终坐标在有效范围内
        x_min_final = max(0, x_min_final)
        y_min_final = max(0, y_min_final)
        x_max_final = min(w, x_max_final)
        y_max_final = min(h, y_max_final)
        
        # 计算最终比例
        final_width = x_max_final - x_min_final
        final_height = y_max_final - y_min_final
        final_ratio = final_width / final_height
        
        logging.info(f"最终边界框: 宽={final_width}, 高={final_height}, 比例={final_ratio:.4f}")
        
        return [x_min_final, y_min_final, x_max_final, y_max_final]

    def resize_by_detection(self, img_np, detections_dict, resize, width, height, 
                         scale_to_side, scale_to_length, keep_proportion, divisible_by,
                         interp_method, mask_interp_method, margin_percent,
                         select_num=None, min_scale=0.8, max_scale=1.2):
        """基于人物检测结果进行裁剪和缩放
        
        输入:
            img_np: 图像数组
            detections_dict: 检测结果字典
            resize等参数: 缩放参数
            margin_percent: 边界框周围的额外空间（百分比）
            select_num: 选择的索引，如果为None则使用所有检测
            min_scale, max_scale: 缩放范围
            
        输出:
            results: 处理后的图像和掩码列表
        """
        # 如果检测结果为空，返回None
        if not detections_dict:
            return None
        
        # 获取检测到的边界框和索引
        bboxes = detections_dict['bboxes']
        indices = detections_dict['indices']
        logging.info(f"检测到{len(bboxes)}个人物，indices={indices}")
        
        # 如果启用了索引选择但没有匹配的索引，返回None
        if select_num is not None and len(indices) > 0 and select_num not in indices:
            logging.info(f"select_num={select_num}不在检测到的indices={indices}中")
            return None
        
        # 计算目标尺寸
        orig_w, orig_h = img_np.shape[1], img_np.shape[0]
        target_width, target_height = self.calculate_target_size(
            orig_w, orig_h, resize, width, height, scale_to_side, 
            scale_to_length, keep_proportion, divisible_by
        )
        logging.info(f"目标尺寸: {target_width}x{target_height}, 比例: {target_width/target_height:.4f}")
        
        # 为目标尺寸创建空白掩码
        output_masks = []
        output_images = []
        
        # 目标宽高比
        target_ratio = target_width / target_height
        
        # 遍历每个检测到的边界框
        for i, bbox in enumerate(bboxes):
            bbox_index = indices[i] if i < len(indices) else i
            # 如果指定了select_num且不匹配当前索引，则跳过
            if select_num is not None and select_num != bbox_index:
                continue
                
            logging.info(f"处理人物 #{bbox_index}, 原始边界框: {bbox}")
            
            # 扩展边界框以适应目标比例，使用正确的参数顺序
            extended_bbox = self.extend_bbox_to_target_ratio(
                bbox, target_ratio, orig_w, orig_h, margin_percent
            )
            logging.info(f"扩展后的边界框: {extended_bbox}")
            
            # 解包边界框坐标
            x1, y1, x2, y2 = extended_bbox
            
            # 确保边界框在图像范围内
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(orig_w, x2)
            y2 = min(orig_h, y2)
            
            # 裁剪图像
            cropped_img = img_np[int(y1):int(y2), int(x1):int(x2)]
            
            if cropped_img.size == 0:
                logging.warning(f"裁剪区域无效: {x1},{y1},{x2},{y2}, 跳过此区域")
                continue
                
            logging.info(f"裁剪区域: {x1},{y1},{x2},{y2}, 裁剪后尺寸: {cropped_img.shape}")
            
            # 创建裁剪区域的掩码
            mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
            cv2.rectangle(mask, (int(x1), int(y1)), (int(x2), int(y2)), 255, -1)
            cropped_mask = mask[int(y1):int(y2), int(x1):int(x2)]
            
            # 处理裁剪后的图像和掩码
            processed_img, processed_mask = self.process_single_image(
                cropped_img, cropped_mask, resize, width, height, 
                scale_to_side, scale_to_length, keep_proportion, divisible_by,
                interp_method, mask_interp_method
            )
            
            output_images.append(processed_img)
            output_masks.append(processed_mask)
            
            # 如果只需要处理一个指定的索引，处理完就可以退出循环
            if select_num is not None:
                break
                
        if not output_images:
            logging.warning("没有有效的裁剪结果")
            return None
            
        # 将所有结果合并为一个列表返回
        results = []
        for img, mask in zip(output_images, output_masks):
            results.append((img, mask))
            
        logging.info(f"resize_by_detection完成，生成了{len(results)}个结果")
        return results
