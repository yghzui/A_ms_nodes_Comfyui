import math
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from functools import reduce
import cv2

import torch
import numpy as np
from PIL import Image
import onnx
import onnxruntime as ort
from tqdm import tqdm  # 添加tqdm进度条支持

import folder_paths


def get_a_person_mask_generator_model_path() -> str:
    # 使用相对路径，基于当前节点目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_file_path = os.path.join(current_dir, "..", "..", "models", "face_person", "selfie_multiclass_256x256.onnx")
    model_file_path = os.path.normpath(model_file_path)

    if not os.path.exists(model_file_path):
        print(f"ONNX model not found at {model_file_path}")
        print("Please ensure the ONNX model is available at the specified path")
        raise FileNotFoundError(f"ONNX model not found: {model_file_path}")

    return model_file_path


class APersonMaskGeneratorMs:
    # 类级别的模型缓存
    _ort_session = None
    _model_path = None

    def __init__(self):
        # download the model if we need it
        get_a_person_mask_generator_model_path()

    @classmethod
    def _get_or_create_ort_session(cls):
        """获取或创建ONNX Runtime会话实例（单例模式）"""
        current_model_path = get_a_person_mask_generator_model_path()
        
        # 如果模型路径改变或者session不存在，重新创建
        if cls._ort_session is None or cls._model_path != current_model_path:
            print("正在初始化ONNX Runtime模型...")
            
            # 直接加载原始ONNX模型，不做任何修改
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            cls._ort_session = ort.InferenceSession(current_model_path, providers=providers)
            cls._model_path = current_model_path
            
            # 打印模型信息
            input_info = cls._ort_session.get_inputs()[0]
            output_info = cls._ort_session.get_outputs()[0]
            print(f"ONNX Runtime模型初始化完成")
            print(f"输入: {input_info.name} - {input_info.shape} ({input_info.type})")
            print(f"输出: {output_info.name} - {output_info.shape} ({output_info.type})")
        
        return cls._ort_session

    @classmethod
    def _cleanup_ort_session(cls):
        """清理ONNX Runtime会话资源"""
        if cls._ort_session is not None:
            cls._ort_session = None
            cls._model_path = None
            print("ONNX Runtime模型资源已清理")

    @classmethod
    def INPUT_TYPES(cls):
        false_widget = (
            "BOOLEAN",
            {"default": False, "label_on": "enabled", "label_off": "disabled"},
        )
        true_widget = (
            "BOOLEAN",
            {"default": True, "label_on": "enabled", "label_off": "disabled"},
        )

        return {
            "required": {
                "images": ("IMAGE",),
            },
            "optional": {
                "face_mask": true_widget,
                "background_mask": false_widget,
                "hair_mask": false_widget,
                "body_mask": false_widget,
                "clothes_mask": false_widget,
                "confidence": (
                    "FLOAT",
                    {"default": 0.40, "min": 0.01, "max": 1.0, "step": 0.01},
                ),
                "refine_mask": true_widget,
            },
        }

    CATEGORY = "A_my_nodes/person_and_face"
    # 预先定义所有可能的输出：合并遮罩 + 各个区域的单独遮罩
    RETURN_TYPES = ("MASK", "MASK", "MASK", "MASK", "MASK", "MASK")
    RETURN_NAMES = ("merged_mask", "face_mask", "background_mask", "hair_mask", "body_mask", "clothes_mask")

    FUNCTION = "generate_mask"

    def preprocess_image_for_onnx(self, image: Image, target_size=(256, 256)) -> np.ndarray:
        """预处理图像用于ONNX推理，优化为PyTorch张量操作"""
        # 转换PIL图像为PyTorch张量
        image_tensor = torch.from_numpy(np.array(image)).float()
        
        # 确保是RGB格式
        if image_tensor.shape[-1] == 4:  # RGBA
            image_tensor = image_tensor[:, :, :3]  # 去掉alpha通道
        elif len(image_tensor.shape) == 2:  # 灰度图
            image_tensor = image_tensor.unsqueeze(-1).repeat(1, 1, 3)
        
        # 归一化到[0,1]范围
        image_tensor = image_tensor / 255.0
        
        # 调整图像大小到模型输入尺寸，使用PyTorch插值
        # 转换为 (C, H, W) 格式用于插值
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
        
        # 使用双线性插值调整尺寸
        resized_tensor = torch.nn.functional.interpolate(
            image_tensor, 
            size=target_size, 
            mode='bilinear', 
            align_corners=False
        )
        
        # 转换回 (H, W, C) 格式并添加batch维度
        resized_tensor = resized_tensor.squeeze(0).permute(1, 2, 0)  # (H, W, C)
        resized_tensor = resized_tensor.unsqueeze(0)  # (1, H, W, C)
        
        # 转换为numpy数组用于ONNX推理
        return resized_tensor.cpu().numpy()

    def postprocess_onnx_output(self, output: np.ndarray, original_size: tuple) -> torch.Tensor:
        """后处理ONNX输出，使用优化的处理逻辑"""
        # 导入优化的后处理器
        try:
            from fixed_optimized_postprocessing import FixedOptimizedPostProcessor
            processor = FixedOptimizedPostProcessor()
            return processor.postprocess_onnx_output_minimal(output, original_size)
        except ImportError:
            # 回退到原始实现
            return self._fallback_postprocess(output, original_size)
    
    def _fallback_postprocess(self, output: np.ndarray, original_size: tuple) -> torch.Tensor:
        """回退的后处理实现"""
        # output shape: (256, 256, 6) 单张图像
        # 需要调整回原始图像尺寸
        
        # 转换为PyTorch张量进行更精确的数值计算
        output_tensor = torch.from_numpy(output).float()
        
        # 基于测试结果，使用softmax激活函数在通道维度上，效果最接近MediaPipe
        # softmax_channel激活函数与MediaPipe的差异仅为0.000172，是最佳选择
        softmax_output = torch.nn.functional.softmax(output_tensor, dim=2)
        
        # 使用PyTorch的插值进行高质量的尺寸调整
        # 转换为 (C, H, W) 格式用于插值
        softmax_output = softmax_output.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
        
        # 使用双线性插值调整到原始尺寸，mode='bilinear'比LANCZOS更适合概率值
        resized_output = torch.nn.functional.interpolate(
            softmax_output, 
            size=original_size[::-1],  # (height, width)
            mode='bilinear', 
            align_corners=False
        )
        
        # 转换回 (H, W, C) 格式
        resized_output = resized_output.squeeze(0).permute(1, 2, 0)  # (H, W, C)
        
        return resized_output

    def get_bbox_for_mask(self, mask_tensor: torch.Tensor):
        """从tensor格式的遮罩获取边界框"""
        # 将tensor转换为numpy数组
        mask_np = mask_tensor.cpu().numpy()
        
        # 找到非零像素的位置
        nonzero_indices = np.nonzero(mask_np)
        
        if len(nonzero_indices[0]) == 0:
            return None
        
        # 获取边界框坐标
        min_y, max_y = nonzero_indices[0].min(), nonzero_indices[0].max()
        min_x, max_x = nonzero_indices[1].min(), nonzero_indices[1].max()
        
        bbox_width = max_x - min_x
        bbox_height = max_y - min_y
        
        # 扩展边界框20%
        bbox_padding_x = round(bbox_width * 0.2)
        bbox_padding_y = round(bbox_height * 0.2)
        
        # 确保边界框在图像范围内
        height, width = mask_np.shape
        left = max(0, min_x - bbox_padding_x)
        upper = max(0, min_y - bbox_padding_y)
        right = min(width, max_x + bbox_padding_x)
        lower = min(height, max_y + bbox_padding_y)
        
        return (left, upper, right, lower)

    def __get_mask(
            self,
            image: Image,
            ort_session,
            face_mask: bool,
            background_mask: bool,
            hair_mask: bool,
            body_mask: bool,
            clothes_mask: bool,
            confidence: float,
            refine_mask: bool,
            original_size=None,
            onnx_output=None
    ) -> tuple[Image, dict]:
        """
        使用ONNX模型生成遮罩，优化为PyTorch张量操作
        """
        # 执行推理
        if onnx_output is None:
            # 预处理图像
            input_tensor = self.preprocess_image_for_onnx(image)
            
            # 执行推理
            outputs = ort_session.run(None, {ort_session.get_inputs()[0].name: input_tensor})
            onnx_output = outputs[0]  # 形状: [1, 256, 256, 6]
        
        # 如果onnx_output是4维的（批量输出），取第一个
        if len(onnx_output.shape) == 4:
            onnx_output = onnx_output[0]  # 从 (1, 256, 256, 6) 变为 (256, 256, 6)
        
        # 后处理输出到原始尺寸，现在返回PyTorch张量
        if original_size is None:
            original_size = image.size
        
        processed_output = self.postprocess_onnx_output(onnx_output, original_size)
        
        # 使用PyTorch张量操作进行遮罩生成
        individual_masks = {}
        masks_for_merge = []
        
        # 初始化所有可能的遮罩键为None，确保字典完整性
        individual_masks['background'] = None
        individual_masks['hair'] = None
        individual_masks['body'] = None
        individual_masks['face'] = None
        individual_masks['clothes'] = None
        
        # 处理背景遮罩 (通道0)
        background_tensor = (processed_output[:, :, 0] > confidence).float()
        individual_masks['background'] = background_tensor
        if background_mask:
            masks_for_merge.append(background_tensor)
        
        # 处理头发遮罩 (通道1)
        hair_tensor = (processed_output[:, :, 1] > confidence).float()
        individual_masks['hair'] = hair_tensor
        if hair_mask:
            masks_for_merge.append(hair_tensor)
        
        # 处理身体遮罩 (通道2)
        body_tensor = (processed_output[:, :, 2] > confidence).float()
        individual_masks['body'] = body_tensor
        if body_mask:
            masks_for_merge.append(body_tensor)
        
        # 处理脸部遮罩 (通道3)
        face_tensor = (processed_output[:, :, 3] > confidence).float()
        individual_masks['face'] = face_tensor
        if face_mask:
            masks_for_merge.append(face_tensor)
        
        # 处理衣服遮罩 (通道4)
        clothes_tensor = (processed_output[:, :, 4] > confidence).float()
        individual_masks['clothes'] = clothes_tensor
        if clothes_mask:
            masks_for_merge.append(clothes_tensor)
        
        # 使用逻辑OR合并选中的遮罩（更适合二值遮罩）
        if len(masks_for_merge) == 0:
            merged_mask_tensor = torch.zeros_like(processed_output[:, :, 0])
        else:
            # 使用逻辑OR操作合并遮罩，比maximum更适合二值遮罩
            merged_mask_tensor = torch.stack(masks_for_merge, dim=0).any(dim=0).float()
        
        # 直接返回tensor格式的合并遮罩
        mask_image = merged_mask_tensor
        
        # 精细化遮罩处理
        if refine_mask:
            bbox = self.get_bbox_for_mask(mask_tensor=mask_image)
            if bbox is not None:
                cropped_image_pil = image.crop(bbox)
                
                cropped_mask_image, cropped_individual_masks = self.__get_mask(
                    image=cropped_image_pil,
                    ort_session=ort_session,
                    face_mask=face_mask,
                    background_mask=background_mask,
                    hair_mask=hair_mask,
                    body_mask=body_mask,
                    clothes_mask=clothes_mask,
                    confidence=confidence,
                    refine_mask=False,
                )
                
                # 更新合并遮罩 - 使用tensor操作
                height, width = mask_image.shape
                updated_mask_tensor = torch.zeros((height, width), dtype=torch.float32)
                crop_h, crop_w = cropped_mask_image.shape
                updated_mask_tensor[bbox[1]:bbox[1]+crop_h, bbox[0]:bbox[0]+crop_w] = cropped_mask_image
                mask_image = updated_mask_tensor
                
                # 更新各个区域的遮罩 - 使用tensor操作
                for key, cropped_mask in cropped_individual_masks.items():
                    if cropped_mask is not None:
                        updated_individual_tensor = torch.zeros((height, width), dtype=torch.float32)
                        crop_h, crop_w = cropped_mask.shape
                        updated_individual_tensor[bbox[1]:bbox[1]+crop_h, bbox[0]:bbox[0]+crop_w] = cropped_mask
                        individual_masks[key] = updated_individual_tensor

        return mask_image, individual_masks

    def _preprocess_images_batch_for_onnx(self, images_tensor, target_size=(256, 256)) -> tuple[np.ndarray, tuple]:
        """批量预处理images张量用于ONNX推理，优化为直接张量操作"""
        # 转换整个批次的tensor到numpy，一次性操作
        img_arrays = 255.0 * images_tensor.cpu().numpy()  # (n, h, w, c)
        
        # 记录原始尺寸 (width, height) - 所有图像尺寸相同
        original_size = (img_arrays.shape[2], img_arrays.shape[1])  # (width, height)
        
        # 确保是RGB格式 - 批量操作，直接张量截取
        if img_arrays.shape[-1] == 4:  # RGBA
            img_arrays = img_arrays[:, :, :, :3]  # 只取RGB通道，批量截取
        elif img_arrays.shape[-1] == 3:  # RGB
            pass  # 保持不变
        
        # 归一化到[0,1]范围
        img_arrays = img_arrays / 255.0
        
        # 批量调整图像大小到模型输入尺寸
        # 转换为 (n, c, h, w) 格式用于插值
        img_tensor = torch.from_numpy(img_arrays).float().permute(0, 3, 1, 2)  # (n, c, h, w)
        
        # 使用双线性插值批量调整尺寸
        resized_tensor = torch.nn.functional.interpolate(
            img_tensor, 
            size=target_size, 
            mode='bilinear', 
            align_corners=False
        )
        
        # 转换回 (n, h, w, c) 格式用于ONNX推理
        resized_tensor = resized_tensor.permute(0, 2, 3, 1)  # (n, h, w, c)
        
        # 转换为numpy数组用于ONNX推理
        return resized_tensor.cpu().numpy(), original_size

    def _postprocess_onnx_outputs_batch(self, outputs: np.ndarray, original_size: tuple) -> list[torch.Tensor]:
        """批量后处理ONNX输出，优化为批量操作"""
        batch_size = outputs.shape[0]
        
        # 转换为PyTorch张量进行批量处理
        output_tensor = torch.from_numpy(outputs).float()  # (n, 256, 256, 6)
        
        # 批量应用softmax激活函数
        softmax_output = torch.nn.functional.softmax(output_tensor, dim=3)  # (n, 256, 256, 6)
        
        # 批量调整到原始尺寸
        # 转换为 (n, c, h, w) 格式用于插值
        softmax_output = softmax_output.permute(0, 3, 1, 2)  # (n, 6, 256, 256)
        
        # 使用双线性插值批量调整到原始尺寸
        resized_output = torch.nn.functional.interpolate(
            softmax_output, 
            size=original_size[::-1],  # (height, width)
            mode='bilinear', 
            align_corners=False
        )
        
        # 转换回 (n, h, w, c) 格式
        resized_output = resized_output.permute(0, 2, 3, 1)  # (n, h, w, 6)
        
        # 分离为单独的张量列表
        processed_outputs = [resized_output[i] for i in range(batch_size)]
        
        return processed_outputs



    def generate_mask(
            self,
            images,
            face_mask: bool,
            background_mask: bool,
            hair_mask: bool,
            body_mask: bool,
            clothes_mask: bool,
            confidence: float,
            refine_mask: bool,
    ):
        """Create a segmentation mask from an image

        Args:
            image (torch.Tensor): The image to create the mask for.
            face_mask (bool): create a mask for the background.
            background_mask (bool): create a mask for the hair.
            hair_mask (bool): create a mask for the body .
            body_mask (bool): create a mask for the face.
            clothes_mask (bool): create a mask for the clothes.
            confidence (float): how confident the model is that the detected item is there.
            break_image_into_tiles ("none" or "auto"): break large images into tiles to improve detection.

        Returns:
            torch.Tensor: The segmentation masks.
        """
        # 使用缓存的ONNX session实例，避免重复加载模型
        session = self._get_or_create_ort_session()

        print(f"正在进行优化推理，总共 {len(images)} 张图像")
        
        # 批量预处理：直接对整个张量进行操作，避免逐张转换
        preprocessed_images, original_size = self._preprocess_images_batch_for_onnx(images)
        
        # 由于ONNX模型只支持单张推理，需要循环处理，但优化了预处理和后处理
        print("正在进行ONNX推理...")
        input_name = session.get_inputs()[0].name
        onnx_outputs = []
        
        for i in tqdm(range(len(preprocessed_images)), desc="ONNX推理", unit="张"):
            # 单张推理
            single_input = preprocessed_images[i:i+1]  # 保持4D形状 [1, 256, 256, 3]
            output = session.run(None, {input_name: single_input})
            onnx_outputs.append(output[0])
            
            # 调试输出：检查ONNX推理结果
            if i == 0:  # 只打印第一张图像的调试信息
                print(f"ONNX输出形状: {output[0].shape}")
                print(f"ONNX输出范围: [{output[0].min():.4f}, {output[0].max():.4f}]")
                print(f"各通道最大值: {[output[0][0, :, :, ch].max() for ch in range(output[0].shape[-1])]}")
        
        # 批量后处理：将所有输出合并后一起处理
        print("正在进行批量后处理...")
        if onnx_outputs:
            # 合并所有输出为一个批次
            batch_outputs = np.concatenate(onnx_outputs, axis=0)  # (n, 256, 256, 6)
            processed_outputs = self._postprocess_onnx_outputs_batch(batch_outputs, original_size)
        else:
            processed_outputs = []
        
        # 批量生成遮罩 - 优化版本
        print("正在批量转换区域遮罩...")
        
        if not processed_outputs:
            # 如果没有处理结果，返回空的tensor
            batch_size = len(images)
            if batch_size > 0:
                image_height, image_width = images[0].shape[:2]
            else:
                image_height, image_width = 256, 256
            
            empty_mask = torch.zeros((batch_size, image_height, image_width), dtype=torch.float32)
            return (empty_mask, empty_mask, empty_mask, empty_mask, empty_mask, empty_mask)
        
        # 将所有processed_outputs堆叠为一个批次tensor
        processed_outputs_stack = torch.stack(processed_outputs, dim=0)  # (n, h, w, 6)
        
        # 通道索引映射 (与ONNX模型输出对应)
        channel_indices = {
            'background': 0,
            'hair': 1,
            'body': 2,
            'face': 3,
            'clothes': 4
        }
        
        # 批量阈值处理所有通道
        all_masks_tensor = (processed_outputs_stack > confidence).float()  # (n, h, w, 6)
        
        # 提取各个通道的mask tensor (始终输出实际结果)
        face_masks = all_masks_tensor[:, :, :, channel_indices['face']]
        background_masks = all_masks_tensor[:, :, :, channel_indices['background']]
        hair_masks = all_masks_tensor[:, :, :, channel_indices['hair']]
        body_masks = all_masks_tensor[:, :, :, channel_indices['body']]
        clothes_masks = all_masks_tensor[:, :, :, channel_indices['clothes']]
        
        # 根据用户选择提取需要的通道并合并
        selected_channels = []
        if background_mask: selected_channels.append(0)
        if hair_mask: selected_channels.append(1)
        if body_mask: selected_channels.append(2)
        if face_mask: selected_channels.append(3)
        if clothes_mask: selected_channels.append(4)
        
        if len(selected_channels) == 0:
            # 如果没有选择任何通道，创建全零mask
            merged_masks = torch.zeros_like(processed_outputs_stack[:, :, :, 0])
        else:
            # 批量合并选中的通道
            selected_masks = all_masks_tensor[:, :, :, selected_channels]  # (n, h, w, selected_count)
            merged_masks = selected_masks.any(dim=3).float()  # (n, h, w)
        
        print(f"优化推理完成，处理了 {len(images)} 张图像")
        
        # 始终返回6个固定的遮罩，按照RETURN_NAMES的顺序
        # ("merged_mask", "face_mask", "background_mask", "hair_mask", "body_mask", "clothes_mask")
        return (merged_masks, face_masks, background_masks, hair_masks, body_masks, clothes_masks)

    def __del__(self):
        try:
            self._cleanup_ort_session()
        except:
            pass  # 忽略清理过程中的任何错误

    @classmethod
    def cleanup_resources(cls):
        """手动清理类级别的模型资源"""
        cls._cleanup_ort_session()

# 注册清理函数，在模块卸载时自动清理资源
# import atexit
# atexit.register(APersonMaskGeneratorMs.cleanup_resources)
      
