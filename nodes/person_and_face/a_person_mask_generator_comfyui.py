import math
import os
import sys

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
    model_folder_name = "mediapipe"
    model_name = "selfie_multiclass_256x256.onnx"

    model_folder_path = os.path.join(folder_paths.models_dir, model_folder_name)
    model_file_path = os.path.join(model_folder_path, model_name)

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
        """后处理ONNX输出，使用PyTorch张量操作提升质量"""
        # output shape: (256, 256, 6) 单张图像
        # 需要调整回原始图像尺寸
        
        # 转换为PyTorch张量进行更精确的数值计算
        output_tensor = torch.from_numpy(output).float()
        
        # 应用sigmoid激活函数，使用PyTorch的稳定实现
        # 使用torch.sigmoid比手动计算更稳定，避免数值溢出
        sigmoid_output = torch.sigmoid(output_tensor)
        
        # 使用PyTorch的插值进行高质量的尺寸调整
        # 转换为 (C, H, W) 格式用于插值
        sigmoid_output = sigmoid_output.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
        
        # 使用双线性插值调整到原始尺寸，mode='bilinear'比LANCZOS更适合概率值
        resized_output = torch.nn.functional.interpolate(
            sigmoid_output, 
            size=original_size[::-1],  # (height, width)
            mode='bilinear', 
            align_corners=False
        )
        
        # 转换回 (H, W, C) 格式
        resized_output = resized_output.squeeze(0).permute(1, 2, 0)  # (H, W, C)
        
        return resized_output

    def get_bbox_for_mask(self, mask_image: Image):
        # Convert the image to grayscale
        grayscale = mask_image.convert("L")

        # Create a binary mask where non-black pixels are white (255)
        mask_for_bbox = grayscale.point(lambda p: 255 if p > 0 else 0)

        # Get the bounding box of the non-black areas
        bbox = mask_for_bbox.getbbox()

        if bbox != None:
            left = bbox[0]
            upper = bbox[1]
            right = bbox[2]
            lower = bbox[3]

            bbox_width = right - left
            bbox_height = lower - upper

            # expand the box by 20% in each direction if possible
            bbox_padding_x = round(bbox_width * 0.2)
            bbox_padding_y = round(bbox_height * 0.2)

            # left, upper, right, lower
            bbox = (
                # left
                left - bbox_padding_x if left > bbox_padding_x else 0,
                # upper
                upper - bbox_padding_y if upper > bbox_padding_y else 0,
                # right
                right + bbox_padding_x if right < grayscale.width - bbox_padding_x else grayscale.width,
                # lower
                lower + bbox_padding_y if lower < grayscale.height - bbox_padding_y else grayscale.height,
            )

        return bbox

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
        # 如果没有提供预计算的ONNX输出，则进行推理
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
        if background_mask:
            mask_tensor = (processed_output[:, :, 0] > confidence).float()
            individual_masks['background'] = self._tensor_to_pil_mask(mask_tensor)
            masks_for_merge.append(mask_tensor)
        
        # 处理头发遮罩 (通道1)
        if hair_mask:
            mask_tensor = (processed_output[:, :, 1] > confidence).float()
            individual_masks['hair'] = self._tensor_to_pil_mask(mask_tensor)
            masks_for_merge.append(mask_tensor)
        
        # 处理身体遮罩 (通道2)
        if body_mask:
            mask_tensor = (processed_output[:, :, 2] > confidence).float()
            individual_masks['body'] = self._tensor_to_pil_mask(mask_tensor)
            masks_for_merge.append(mask_tensor)
        
        # 处理脸部遮罩 (通道3)
        if face_mask:
            mask_tensor = (processed_output[:, :, 3] > confidence).float()
            individual_masks['face'] = self._tensor_to_pil_mask(mask_tensor)
            masks_for_merge.append(mask_tensor)
        
        # 处理衣服遮罩 (通道4)
        if clothes_mask:
            mask_tensor = (processed_output[:, :, 4] > confidence).float()
            individual_masks['clothes'] = self._tensor_to_pil_mask(mask_tensor)
            masks_for_merge.append(mask_tensor)
        
        # 使用逻辑OR合并选中的遮罩（更适合二值遮罩）
        if len(masks_for_merge) == 0:
            merged_mask_tensor = torch.zeros_like(processed_output[:, :, 0])
        else:
            # 使用逻辑OR操作合并遮罩，比maximum更适合二值遮罩
            merged_mask_tensor = torch.stack(masks_for_merge, dim=0).any(dim=0).float()
        
        # 创建合并后的遮罩图像
        mask_image = self._tensor_to_pil_mask(merged_mask_tensor)
        
        # 精细化遮罩处理
        if refine_mask:
            bbox = self.get_bbox_for_mask(mask_image=mask_image)
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
                
                # 更新合并遮罩
                updated_mask_image = Image.new('RGBA', image.size, (0, 0, 0))
                updated_mask_image.paste(cropped_mask_image, bbox)
                mask_image = updated_mask_image
                
                # 更新各个区域的遮罩
                for key, cropped_mask in cropped_individual_masks.items():
                    if cropped_mask is not None:
                        updated_individual_mask = Image.new('RGBA', image.size, (0, 0, 0))
                        updated_individual_mask.paste(cropped_mask, bbox)
                        individual_masks[key] = updated_individual_mask

        return mask_image, individual_masks

    def _tensor_to_pil_mask(self, mask_tensor: torch.Tensor) -> Image:
        """将PyTorch张量转换为PIL遮罩图像"""
        # 将0-1的浮点值转换为0-255的整数值
        mask_array = (mask_tensor.cpu().numpy() * 255).astype(np.uint8)
        
        # 创建RGBA遮罩：白色为前景，黑色为背景
        mask_rgba = np.zeros((*mask_array.shape, 4), dtype=np.uint8)
        mask_rgba[:, :, :3] = mask_array[..., np.newaxis]  # RGB通道
        mask_rgba[:, :, 3] = mask_array  # Alpha通道
        
        return Image.fromarray(mask_rgba)

    def get_mask_images(
            self,
            images, # tensors
            face_mask: bool,
            background_mask: bool,
            hair_mask: bool,
            body_mask: bool,
            clothes_mask: bool,
            confidence: float,
            refine_mask: bool,
    ) -> tuple[list[Image], list[dict]]:
        # 使用缓存的ONNX session实例，避免重复加载模型
        session = self._get_or_create_ort_session()

        mask_images: list[Image] = []
        individual_masks_list: list[dict] = []

        # 循环推理每张图像
        print(f"正在进行循环推理，总共 {len(images)} 张图像")
        
        for i, tensor_image in enumerate(tqdm(images, desc="处理图像", unit="张")):
            # Convert the Tensor to a PIL image
            img_array = 255.0 * tensor_image.cpu().numpy()
            
            # 记录原始尺寸
            original_size = (img_array.shape[1], img_array.shape[0])  # (width, height)
            
            # 确保是RGB格式
            if img_array.shape[-1] == 4:  # RGBA
                img_array = img_array[:, :, :3]  # 只取RGB通道
            elif img_array.shape[-1] == 3:  # RGB
                pass  # 保持不变
            
            image_pil = Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))
            
            # 单张图像推理
            mask_image, individual_masks = self.__get_mask(
                image=image_pil,
                ort_session=session,
                face_mask=face_mask,
                background_mask=background_mask,
                hair_mask=hair_mask,
                body_mask=body_mask,
                clothes_mask=clothes_mask,
                confidence=confidence,
                refine_mask=refine_mask,
                original_size=original_size,
                onnx_output=None  # 让__get_mask方法自己进行推理
            )
            mask_images.append(mask_image)
            individual_masks_list.append(individual_masks)

        print(f"循环推理完成，处理了 {len(mask_images)} 张图像")
        return mask_images, individual_masks_list

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

        mask_images, individual_masks_list = self.get_mask_images(
            images=images,
            face_mask=face_mask,
            background_mask=background_mask,
            hair_mask=hair_mask,
            body_mask=body_mask,
            clothes_mask=clothes_mask,
            confidence=confidence,
            refine_mask=refine_mask,
        )

        # 转换合并遮罩为tensor
        merged_tensor_masks = []
        # 为遮罩转换添加进度条
        for mask_image in tqdm(mask_images, desc="转换遮罩格式", unit="个"):
            tensor_mask = mask_image.convert("RGB")
            tensor_mask = np.array(tensor_mask).astype(np.float32) / 255.0
            tensor_mask = torch.from_numpy(tensor_mask)[None,]
            tensor_mask = tensor_mask.squeeze(3)[..., 0]
            merged_tensor_masks.append(tensor_mask)

        # 获取图像尺寸用于创建纯黑遮罩
        batch_size = len(images)
        if batch_size > 0:
            image_height, image_width = images[0].shape[:2]
        else:
            image_height, image_width = 256, 256  # 默认尺寸

        # 创建纯黑遮罩的函数
        def create_black_mask():
            """创建纯黑遮罩tensor"""
            black_mask = torch.zeros((batch_size, image_height, image_width), dtype=torch.float32)
            return black_mask

        # 转换各个区域遮罩为tensor的函数
        def convert_mask_to_tensor(mask_image):
            if mask_image is None:
                return None
            tensor_mask = mask_image.convert("RGB")
            tensor_mask = np.array(tensor_mask).astype(np.float32) / 255.0
            tensor_mask = torch.from_numpy(tensor_mask)[None,]
            tensor_mask = tensor_mask.squeeze(3)[..., 0]
            return tensor_mask

        # 收集各个区域的tensor遮罩
        face_tensor_masks = []
        background_tensor_masks = []
        hair_tensor_masks = []
        body_tensor_masks = []
        clothes_tensor_masks = []

        # 为各个区域遮罩处理添加进度条
        for individual_masks in tqdm(individual_masks_list, desc="处理区域遮罩", unit="组"):
            # 安全地获取各个区域的遮罩，如果不存在则使用None
            face_mask_tensor = convert_mask_to_tensor(individual_masks.get('face'))
            background_mask_tensor = convert_mask_to_tensor(individual_masks.get('background'))
            hair_mask_tensor = convert_mask_to_tensor(individual_masks.get('hair'))
            body_mask_tensor = convert_mask_to_tensor(individual_masks.get('body'))
            clothes_mask_tensor = convert_mask_to_tensor(individual_masks.get('clothes'))
            
            face_tensor_masks.append(face_mask_tensor)
            background_tensor_masks.append(background_mask_tensor)
            hair_tensor_masks.append(hair_mask_tensor)
            body_tensor_masks.append(body_mask_tensor)
            clothes_tensor_masks.append(clothes_mask_tensor)

        # 合并遮罩
        merged_masks = torch.cat(merged_tensor_masks, dim=0) if merged_tensor_masks else create_black_mask()

        # 处理各个区域的遮罩：如果启用则使用实际遮罩，否则使用纯黑遮罩
        # 按照RETURN_NAMES的固定顺序：face_mask, background_mask, hair_mask, body_mask, clothes_mask
        
        # Face遮罩
        if face_mask and any(m is not None for m in face_tensor_masks):
            face_masks = torch.cat([m for m in face_tensor_masks if m is not None], dim=0)
        else:
            face_masks = create_black_mask()
        
        # Background遮罩
        if background_mask and any(m is not None for m in background_tensor_masks):
            background_masks = torch.cat([m for m in background_tensor_masks if m is not None], dim=0)
        else:
            background_masks = create_black_mask()
        
        # Hair遮罩
        if hair_mask and any(m is not None for m in hair_tensor_masks):
            hair_masks = torch.cat([m for m in hair_tensor_masks if m is not None], dim=0)
        else:
            hair_masks = create_black_mask()
        
        # Body遮罩
        if body_mask and any(m is not None for m in body_tensor_masks):
            body_masks = torch.cat([m for m in body_tensor_masks if m is not None], dim=0)
        else:
            body_masks = create_black_mask()
        
        # Clothes遮罩
        if clothes_mask and any(m is not None for m in clothes_tensor_masks):
            clothes_masks = torch.cat([m for m in clothes_tensor_masks if m is not None], dim=0)
        else:
            clothes_masks = create_black_mask()

        # 始终返回6个固定的遮罩，按照RETURN_NAMES的顺序
        # ("merged_mask", "face_mask", "background_mask", "hair_mask", "body_mask", "clothes_mask")
        return (merged_masks, face_masks, background_masks, hair_masks, body_masks, clothes_masks)

    def __del__(self):
        """析构函数：清理模型资源"""
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
      
