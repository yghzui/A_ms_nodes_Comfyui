# -*- coding: utf-8 -*-
# Created time : 2025/04/22 23:11 
# Auther : ygh
# File   : crop_person.py
# Description :
import folder_paths
import os
import cv2
import torch
import numpy as np
import math
import logging
import onnxruntime as ort
from custom_nodes.A_my_nodes.nodes.image_nodes import img2tensor, tensor2img
import hashlib

# 定义模型路径
comfyui_model_path=folder_paths.models_dir
yolov10m_path=os.path.join(comfyui_model_path,"detection","yolov10m.onnx")


YOLO_MODEL_PATH = yolov10m_path
MAX_RESOLUTION = 8192

# 移植自 test_yolov10m.py 的辅助函数
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)

def preprocess(img, input_shape):
    # Letterbox
    img_letterbox, ratio, (dw, dh) = letterbox(img, new_shape=input_shape, auto=False)
    
    # HWC to CHW, BGR to RGB
    # 注意：这里假设输入img是BGR格式（如果是OpenCV读取或已转换）
    blob = cv2.cvtColor(img_letterbox, cv2.COLOR_BGR2RGB)
    blob = blob.transpose((2, 0, 1))
    blob = np.ascontiguousarray(blob)
    
    # 0-255 to 0-1
    blob = blob.astype(np.float32) / 255.0
    
    # Add batch dimension
    blob = blob[np.newaxis, ...]
    return blob, ratio, (dw, dh)

# 添加测试函数 (更新为使用ONNX)
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
    if not os.path.exists(YOLO_MODEL_PATH):
        print(f"找不到人物检测模型: {YOLO_MODEL_PATH}")
        return
    
    # 加载模型
    try:
        session = ort.InferenceSession(YOLO_MODEL_PATH, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法加载图像: {image_path}")
        return
        
    # 获取输入信息
    model_inputs = session.get_inputs()
    input_names = [inp.name for inp in model_inputs]
    input_shape = (640, 640)
    if len(model_inputs[0].shape) == 4:
        h, w = model_inputs[0].shape[2], model_inputs[0].shape[3]
        if isinstance(h, int) and isinstance(w, int):
            input_shape = (h, w)
            
    # 预处理
    blob, ratio, (dw, dh) = preprocess(img, input_shape)
    
    # 推理
    output_names = [out.name for out in session.get_outputs()]
    outputs = session.run(output_names, {input_names[0]: blob})
    detections = outputs[0]
    
    if len(detections.shape) == 3:
        detections = detections[0]
    
    # 收集结果
    sorted_boxes = []
    for det in detections:
        if len(det) >= 6:
            x1, y1, x2, y2, score, cls = det[:6]
            if int(cls) == 0 and score > confidence:
                # 还原坐标
                x1 -= dw
                y1 -= dh
                x2 -= dw
                y2 -= dh
                
                x1 /= ratio[0]
                y1 /= ratio[1]
                x2 /= ratio[0]
                y2 /= ratio[1]
                
                # Clip
                h_img, w_img = img.shape[:2]
                x1 = max(0, min(x1, w_img))
                y1 = max(0, min(y1, h_img))
                x2 = max(0, min(x2, w_img))
                y2 = max(0, min(y2, h_img))
                
                sorted_boxes.append([x1, y1, x2, y2])
                
    # 按照x坐标排序
    sorted_boxes.sort(key=lambda x: x[0])
    
    if not sorted_boxes:
        print("未检测到人物")
        return

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

class CropPerson:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),  # 输入图像张量
                "crop_by_person": ("BOOLEAN", {
                    "default": False, 
                    "tooltip": "是否使用YOLOv10m ONNX模型检测人物，并裁剪到仅包含人物区域"
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
            },
            "optional": {
                "masks": ("MASK", {
                    "tooltip": "输入遮罩张量，可选。如果不提供，将创建全黑的遮罩"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "STRING")
    RETURN_NAMES = ("image", "mask", "person_count", "crop_info")
    FUNCTION = "crop_images_and_masks"
    CATEGORY = "My_node/image"

    def __init__(self):
        self.person_model = None

    def load_person_model(self):
        if self.person_model is None:
            if not os.path.exists(YOLO_MODEL_PATH):
                raise FileNotFoundError(f"找不到人物检测模型: {YOLO_MODEL_PATH}")
            try:
                self.person_model = ort.InferenceSession(YOLO_MODEL_PATH, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            except Exception as e:
                logging.error(f"加载ONNX模型失败: {e}")
                raise e
        return self.person_model

    def get_all_person_bboxes(self, img_np, confidence):
        """检测图像中的人物并返回按照从左到右排序的边界框列表"""
        # 记录输入图像形状
        logging.info(f"get_all_person_bboxes输入图像形状: {img_np.shape}")
        
        # 确保img_np形状正确
        if len(img_np.shape) == 4:  # 如果形状是[N, H, W, C]
            img_np = img_np[0]  # 取第一个图像
            logging.info(f"转换后的图像形状: {img_np.shape}")
        
        # 确保是三通道彩色图像
        if len(img_np.shape) != 3 or img_np.shape[2] != 3:
            logging.error(f"图像不是三通道彩色图像: {img_np.shape}")
            return []
        
        # 记录图像范围
        img_min = np.min(img_np)
        img_max = np.max(img_np)
        logging.info(f"图像数值范围: min={img_min}, max={img_max}")
        
        # 如果图像是float类型且范围在[0,1]之间，转换为uint8
        if img_np.dtype == np.float32 or img_np.dtype == np.float64:
            if img_max <= 1.0:
                img_np = (img_np * 255).astype(np.uint8)
                logging.info(f"将[0,1]浮点图像转换为uint8: {img_np.shape}, {img_np.dtype}")
        
        # 注意：外部调用者(resize_images_and_masks)已经将RGB转为了BGR
        # 所以这里img_np是BGR格式，可以直接传给preprocess（它会转回RGB，符合test_yolov10m.py逻辑）
        logging.info(f"确认图像应该是BGR格式")
            
        try:
            session = self.load_person_model()
            
            # 获取模型输入尺寸
            model_inputs = session.get_inputs()
            input_names = [inp.name for inp in model_inputs]
            input_shape = (640, 640)
            if len(model_inputs[0].shape) == 4:
                h, w = model_inputs[0].shape[2], model_inputs[0].shape[3]
                if isinstance(h, int) and isinstance(w, int):
                    input_shape = (h, w)
            
            # 预处理
            blob, ratio, (dw, dh) = preprocess(img_np, input_shape)
            
            # 推理
            output_names = [out.name for out in session.get_outputs()]
            outputs = session.run(output_names, {input_names[0]: blob})
            detections = outputs[0]
            
            if len(detections.shape) == 3:
                detections = detections[0]
            
            # 收集结果
            all_boxes = []
            for det in detections:
                if len(det) >= 6:
                    x1, y1, x2, y2, score, cls = det[:6]
                    if int(cls) == 0 and score > confidence:
                        # 还原坐标
                        x1 -= dw
                        y1 -= dh
                        x2 -= dw
                        y2 -= dh
                        
                        x1 /= ratio[0]
                        y1 /= ratio[1]
                        x2 /= ratio[0]
                        y2 /= ratio[1]
                        
                        # Clip
                        h_img, w_img = img_np.shape[:2]
                        x1 = max(0, min(x1, w_img))
                        y1 = max(0, min(y1, h_img))
                        x2 = max(0, min(x2, w_img))
                        y2 = max(0, min(y2, h_img))
                        
                        all_boxes.append([x1, y1, x2, y2])

            if not all_boxes:
                logging.warning("未检测到人物")
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

    def crop_images_and_masks(self, images, crop_by_person, use_largest_person, person_indices, merge_output, 
                                person_confidence, padding_percent, masks=None):
        """
        主处理函数，逻辑：
        1. 根据crop_by_person参数决定是否进行人物裁剪
        2. 如果裁剪，根据use_largest_person或person_indices选择要处理的人物
        3. 应用padding_percent扩展边界框
        4. 裁剪图像和mask
        
        参数:
            images: 输入图像张量
            masks: 输入掩码张量，可选
            crop_by_person: 是否使用人物检测进行裁剪
            use_largest_person: 是否只处理面积最大的人物框
            person_indices: 要处理的人物索引字符串
            merge_output: 处理多个人物时是否合并输出
            person_confidence: 检测置信度
            padding_percent: 边界框扩展百分比
            
        返回:
            裁剪后的图像、掩码和检测到的人物数量
        """
        output_images = []
        output_masks = []
        crop_boxes = []  # 存储裁剪框信息
        
        # 如果未提供masks，创建与images相同批次的全黑mask
        # 修复mask创建逻辑：确保包含width维度
        if masks is None:
            logging.info("未提供masks，创建全黑mask")
            # masks格式为[batch, height, width]
            # 确保images是[N, H, W, C]
            masks = torch.zeros((images.shape[0], images.shape[1], images.shape[2]), 
                                dtype=torch.float32, device=images.device)
        
        total_person_count = 0

        # 打印输入图像的形状，确认格式
        logging.info(f"输入图像形状: {images.shape}")  # 应该是 [N, H, W, C]
        logging.info(f"输入遮罩形状: {masks.shape}")  # 应该是 [N, H, W]

        for img_idx, (img, mask) in enumerate(zip(images, masks)):
            # 在ComfyUI中，图像格式为[H, W, C]
            img_np = img.cpu().numpy()  # 直接从tensor转为numpy，保留原始维度
            mask_np = mask.cpu().numpy()
            
            logging.info(f"图像{img_idx}形状: {img_np.shape}")
            
            # 获取图像的高度和宽度
            if len(img_np.shape) == 4:  # 如果形状是[N, H, W, C]
                h, w = img_np.shape[1:3]
                img_np = img_np[0]  # 取第一个图像
            else:  # 如果形状是[H, W, C]
                h, w = img_np.shape[0:2]
                
            # 同样处理mask
            if len(mask_np.shape) == 3:  # 如果形状是[N, H, W]
                mask_np = mask_np[0]  # 取第一个mask
            
            images_to_process = []
            masks_to_process = []
            
            if crop_by_person:
                # 转换图像格式进行检测
                if img_np.dtype == np.float32 or img_np.dtype == np.float64:
                    if np.max(img_np) <= 1.0:
                        img_for_detection = (img_np * 255).astype(np.uint8)
                    else:
                        img_for_detection = img_np.astype(np.uint8)
                else:
                    img_for_detection = img_np
                
                # RGB to BGR
                if img_for_detection.shape[2] == 3:
                    img_for_detection = cv2.cvtColor(img_for_detection, cv2.COLOR_RGB2BGR)
                
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
                        # 选择要处理的边界框
                        if use_largest_person and total_person_count > 1:
                            areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in all_person_boxes]
                            largest_idx = areas.index(max(areas))
                            all_person_boxes = [all_person_boxes[largest_idx]]
                            valid_indices = [0]
                        else:
                            valid_indices = self.parse_indices(person_indices, total_person_count - 1)
                        
                        selected_boxes = [all_person_boxes[idx] for idx in valid_indices if idx < total_person_count]
                        
                        if selected_boxes:
                            # 如果需要合并输出或只有一个边界框
                            if merge_output or len(selected_boxes) == 1:
                                bbox = self.get_merged_bbox(selected_boxes)
                                
                                if bbox is not None:
                                    # 扩展边界框 (Padding)
                                    x_min, y_min, x_max, y_max = self.extend_bbox(bbox, w, h, padding_percent)
                                    
                                    # 裁剪图像和遮罩 (确保mask和image使用相同的裁剪区域)
                                    crop_img = img_np[int(y_min):int(y_max), int(x_min):int(x_max)]
                                    crop_mask = mask_np[int(y_min):int(y_max), int(x_min):int(x_max)] if mask_np is not None else None
                                    
                                    crop_info = f"[{int(x_min)},{int(y_min)},{int(x_max)},{int(y_max)}]"
                                    crop_boxes.append(crop_info)
                                    
                                    images_to_process.append(crop_img)
                                    masks_to_process.append(crop_mask)
                                else:
                                    logging.warning("无法获取合并边界框，使用原始图像")
                                    images_to_process.append(img_np)
                                    masks_to_process.append(mask_np)
                                    crop_boxes.append(f"[0,0,{w},{h}]")
                            else:
                                # 处理多个独立的边界框
                                for box in selected_boxes:
                                    x_min, y_min, x_max, y_max = self.extend_bbox(box, w, h, padding_percent)
                                    
                                    crop_img = img_np[int(y_min):int(y_max), int(x_min):int(x_max)]
                                    crop_mask = mask_np[int(y_min):int(y_max), int(x_min):int(x_max)] if mask_np is not None else None
                                    
                                    crop_info = f"[{int(x_min)},{int(y_min)},{int(x_max)},{int(y_max)}]"
                                    crop_boxes.append(crop_info)
                                    
                                    images_to_process.append(crop_img)
                                    masks_to_process.append(crop_mask)
                        else:
                            logging.info("没有选中有效边界框，处理整个图像")
                            images_to_process.append(img_np)
                            masks_to_process.append(mask_np)
                            crop_boxes.append(f"[0,0,{w},{h}]")
                    except Exception as e:
                        logging.error(f"边界框处理失败: {str(e)}")
                        images_to_process.append(img_np)
                        masks_to_process.append(mask_np)
                        crop_boxes.append(f"[0,0,{w},{h}]")
                else:
                    logging.info("未检测到人物，处理整个图像")
                    images_to_process.append(img_np)
                    masks_to_process.append(mask_np)
                    crop_boxes.append(f"[0,0,{w},{h}]")
            else:
                logging.info("不进行人物裁剪，处理整个图像")
                images_to_process.append(img_np)
                masks_to_process.append(mask_np)
                crop_boxes.append(f"[0,0,{w},{h}]")
            
            # 格式转换
            for proc_img, proc_mask in zip(images_to_process, masks_to_process):
                processed_img, processed_mask = self.process_single_image(proc_img, proc_mask)
                output_images.append(processed_img)
                output_masks.append(processed_mask)

        # 记录生成的crop_boxes
        logging.info(f"生成的crop_boxes列表: {crop_boxes}")

        # 处理输出
        if not output_images:
            logging.warning("没有输出图像，返回原图")
            crop_info_str = f"[0,0,{w},{h}]"
            return images, masks, total_person_count, crop_info_str
        
        if len(output_images) == 1:
            result_img = output_images[0]
            result_mask = output_masks[0]
            crop_info = crop_boxes[0] if crop_boxes else f"[0,0,{w},{h}]"
            logging.info(f"返回单个图像 crop_info: {crop_info}")
            return result_img, result_mask, total_person_count, crop_info
        else:
            # 过滤形状不一致的图像
            first_img_shape = output_images[0].shape[1:]
            first_mask_shape = output_masks[0].shape[1:]
            
            filtered_images = []
            filtered_masks = []
            filtered_crop_boxes = []
            
            for i, (img, mask) in enumerate(zip(output_images, output_masks)):
                # 检查形状是否一致 (除了batch维度)
                if img.shape[1:] == first_img_shape and mask.shape[1:] == first_mask_shape:
                    filtered_images.append(img)
                    filtered_masks.append(mask)
                    if i < len(crop_boxes):
                        filtered_crop_boxes.append(crop_boxes[i])
                else:
                    logging.warning(f"图像 {i} 形状 {img.shape} 与首个图像 {first_img_shape} 不一致，已跳过")
            
            if not filtered_images:
                logging.warning("过滤后没有有效图像，返回原图")
                return images, masks, total_person_count, f"[0,0,{w},{h}]"
            
            final_images = torch.cat(filtered_images, dim=0)
            final_masks = torch.cat(filtered_masks, dim=0)
            crop_info = ", ".join(filtered_crop_boxes) if filtered_crop_boxes else f"[0,0,{w},{h}]"
            
            logging.info(f"返回合并图像 crop_info: {crop_info}")
            return final_images, final_masks, total_person_count, crop_info

    def process_single_image(self, img_np, mask_np):
        """处理单个图像：仅进行格式转换"""
        if mask_np is None:
            mask_np = np.zeros((img_np.shape[0], img_np.shape[1]), dtype=np.uint8)
            
        # 确保形状正确
        if len(img_np.shape) == 4: img_np = img_np[0]
        if len(mask_np.shape) == 3: mask_np = mask_np[0]
            
        # 检查有效性
        if img_np.size == 0:
            dummy_img = np.zeros((64, 64, 3), dtype=np.uint8)
            dummy_mask = np.zeros((64, 64), dtype=np.uint8)
            return (torch.from_numpy(dummy_img).float() / 255.0).unsqueeze(0), torch.from_numpy(dummy_mask).float().unsqueeze(0)

        # 转换为tensor
        try:
            if img_np.dtype == np.uint8:
                img_tensor = torch.from_numpy(img_np.astype(np.float32) / 255.0)
            else:
                img_tensor = torch.from_numpy(img_np.astype(np.float32))
            
            if len(img_tensor.shape) == 3:
                img_tensor = img_tensor.unsqueeze(0)
            
            if mask_np.dtype == np.uint8:
                mask_tensor = torch.from_numpy(mask_np.astype(np.float32) / 255.0)
            else:
                mask_tensor = torch.from_numpy(mask_np.astype(np.float32))
                
            if len(mask_tensor.shape) == 2:
                mask_tensor = mask_tensor.unsqueeze(0)
            elif len(mask_tensor.shape) == 3 and mask_tensor.shape[2] == 1:
                mask_tensor = mask_tensor.squeeze(-1).unsqueeze(0)
                
            return img_tensor, mask_tensor
        except Exception as e:
            logging.error(f"转换为tensor失败: {str(e)}")
            dummy_img = np.zeros((64, 64, 3), dtype=np.uint8)
            dummy_mask = np.zeros((64, 64), dtype=np.uint8)
            return (torch.from_numpy(dummy_img).float() / 255.0).unsqueeze(0), torch.from_numpy(dummy_mask).float().unsqueeze(0)

    def extend_bbox(self, bbox, w, h, padding_percent):
        """扩展边界框 (Padding)"""
        x_min, y_min, x_max, y_max = bbox
        
        # 计算宽高
        bbox_width = x_max - x_min
        bbox_height = y_max - y_min
        
        # 计算padding像素
        padding_w = bbox_width * (padding_percent / 100)
        padding_h = bbox_height * (padding_percent / 100)
        
        # 应用padding
        x_min_final = max(0, x_min - padding_w)
        y_min_final = max(0, y_min - padding_h)
        x_max_final = min(w, x_max + padding_w)
        y_max_final = min(h, y_max + padding_h)
        
        logging.info(f"原始边界框: [{x_min},{y_min},{x_max},{y_max}]")
        logging.info(f"扩展后边界框: [{x_min_final},{y_min_final},{x_max_final},{y_max_final}]")
        
        return [x_min_final, y_min_final, x_max_final, y_max_final]

class CropInfoToNumbers:
    """将裁剪信息字符串转换为具体的数值输出"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "crop_info": ("STRING", {"default": "[0,0,0,0]"}),
                "index": ("INT", {
                    "default": 0, 
                    "min": 0,
                    "max": 999,
                    "step": 1,
                    "tooltip": "当有多个裁剪框时，选择要输出的索引。如果索引超出范围，将使用最后一个框的值"
                }),
            },
        }
    
    RETURN_TYPES = ("INT", "INT", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("x_min", "y_min", "x_max", "y_max", "width", "height")
    FUNCTION = "convert_crop_info"
    CATEGORY = "My_node/image"

    def convert_crop_info(self, crop_info: str, index: int = 0):
        """
        将裁剪信息字符串转换为具体的数值
        
        Args:
            crop_info: 裁剪信息字符串，格式为 "[x1,y1,x2,y2]" 或 "[x1,y1,x2,y2], [x3,y3,x4,y4], ..."
            index: 要获取的裁剪框索引
            
        Returns:
            tuple: (x_min, y_min, x_max, y_max, width, height)
        """
        try:
            # 移除所有空格
            crop_info = crop_info.replace(" ", "")
            
            # 分割多个裁剪框
            boxes = crop_info.split(",")
            box_list = []
            current_box = []
            
            # 解析裁剪框字符串
            for part in boxes:
                # 检查是否包含左括号
                if "[" in part:
                    # 开始新的裁剪框
                    current_box = []
                    # 清理并添加数字
                    num = part.replace("[", "").replace("]", "")
                    if num:
                        current_box.append(int(num))
                # 检查是否包含右括号
                elif "]" in part:
                    # 清理并添加最后一个数字
                    num = part.replace("[", "").replace("]", "")
                    if num:
                        current_box.append(int(num))
                    # 如果收集到4个数字，添加到框列表
                    if len(current_box) == 4:
                        box_list.append(current_box)
                    current_box = []
                else:
                    # 添加中间的数字
                    num = part.replace("[", "").replace("]", "")
                    if num:
                        current_box.append(int(num))
            
            # 如果没有解析到任何有效的框，返回默认值
            if not box_list:
                logging.warning(f"无法解析裁剪信息: {crop_info}")
                return 0, 0, 0, 0, 0, 0
            
            # 如果索引超出范围，使用最后一个框
            if index >= len(box_list):
                logging.info(f"索引 {index} 超出范围，使用最后一个框")
                selected_box = box_list[-1]
            else:
                selected_box = box_list[index]
            
            # 计算宽度和高度
            x_min, y_min, x_max, y_max = selected_box
            width = x_max - x_min
            height = y_max - y_min
            
            # 返回所有值，包括计算出的宽度和高度
            return x_min, y_min, x_max, y_max, width, height
            
        except Exception as e:
            logging.error(f"解析裁剪信息时出错: {str(e)}")
            return 0, 0, 0, 0, 0, 0
