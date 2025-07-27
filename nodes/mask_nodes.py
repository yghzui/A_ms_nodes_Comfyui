# -*- coding: utf-8 -*-
# Created time : 2024/10/01 16:45 
# Auther : ygh
# File   : mask_nodes.py
# Description :
# import numpy as np
# import time
# import torch
# from comfy import model_management
# from custom_nodes.facerestore_cf.facelib.utils.face_restoration_helper import FaceRestoreHelper
# from custom_nodes.facerestore_cf.facelib.detection.retinaface import retinaface
# from torchvision.transforms.functional import normalize
# from torchvision.utils import make_grid
# from comfy_extras.chainner_models import model_loading
# import torch.nn.functional as F
# import os
# import re
# import json
# import hashlib
# import cv2
# from PIL import ImageGrab, ImageDraw, ImageFont, Image, ImageSequence, ImageOps
# from comfy.cli_args import args
# from comfy.utils import ProgressBar, common_upscale
# import folder_paths
# from nodes import MAX_RESOLUTION
import warnings
from .segment_anything_func import *
from .imagefunc import *
from scipy import ndimage
# from custom_nodes.A_my_nodes.nodes.get_result_text_p import run_script_with_subprocess, get_list_from_txt
import cv2
import numpy as np
import torch
from custom_nodes.A_my_nodes.nodes.face_without_glasses import infer_and_get_mask

warnings.filterwarnings("ignore")


class CreateTextMask:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",),
                             "device": ([-1, 0], {"default": -1, "tooltip": "-1:cpu, 0:gpu"}),
                             }}

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("bbox_image", "mask")
    FUNCTION = "crop_text"
    CATEGORY = "My_node/mask"

    # @classmethod
    # def IS_CHANGED(s, image, device):
    #     # 计算输入的哈希值，确保只有在输入变化时才重新计算
    #     m = hashlib.sha256()
        
    #     # 对图像进行哈希
    #     images_flat = image.reshape(-1).numpy().tobytes()
    #     m.update(images_flat[:1024])  # 只使用部分数据做哈希，避免计算过重
        
    #     # 将其他参数也加入哈希计算
    #     m.update(str(device).encode())
        
    #     return m.digest().hex()

    def crop_text(self, image, device):
        if device == -1:
            use_gpu = False
        else:
            use_gpu = True
        image_np = image.squeeze(0).detach().cpu().numpy()  # 去掉 batch 维度，并转换为 NumPy 数组
        # concurrent_dir = os.path.dirname(os.path.dirname(__file__))
        # image_np_save_path = os.path.join(concurrent_dir, "temp", "image_np_save.jpg")
        # cv2.imwrite(image_np_save_path, image_np)
        # save_text_path = os.path.join(concurrent_dir, "temp", r"result_ocr_img_numpy.txt")
        # run_script_with_subprocess(image_np_save_path, save_text_path, use_gpu)
        # if os.path.exists(save_text_path):
        #     result_list = get_numpy_from_txt(save_text_path)

        # if os.path.exists(image_np_save_path):
        # 假设 ComfyUI 的张量像素值范围是 [0, 1]，将其转换为 [0, 255]，并将类型转换为 uint8
        image_np = (image_np * 255).astype(np.uint8)
        width, height = image_np.shape[1], image_np.shape[0]
        # new_bbox_list = []
        # 创建一个与图像相同大小的空白蒙版，初始化为黑色
        mask = np.zeros((height, width), dtype=np.uint8)
        # for i in range(len(result_list)):
        #     # 将这些点转换为所需的形状（-1, 1, 2）以便在图像上绘制
        #     points = result_list[i]
        #     points = points.reshape((-1, 1, 2))
        #
        #     # 使用 OpenCV 画出多边形，参数 (图像, 点, 是否闭合, 颜色, 线条宽度)
        #     cv2.polylines(image_np, [points], isClosed=True, color=(0, 255, 0), thickness=2)
        #     # 填充多边形的内部为白色
        #     cv2.fillPoly(mask, [points], color=(255, 255, 255))
        # 可选：将蒙版应用到原图像，生成框内区域的图片
        image_np_bgra = image_np
        # masked_image = cv2.bitwise_and(image_np_bgr, image_np_bgr, mask=mask)
        masked_image = torch.tensor(image_np_bgra).unsqueeze(0)  # 添加 batch 维度
        masked_image = masked_image.float() / 255.0  # 如果你想标准化为 [0, 1] 的范围
        mask_out = torch.tensor(mask).unsqueeze(0)  # 变为 [1, height, width, 1]
        # # 2. 将蒙版和应用蒙版的图像从 NumPy 数组转换为 PyTorch 张量
        # # (height, width, channels) -> (1, height, width, channels)
        # mask_tensor = torch.tensor(mask_3ch).unsqueeze(0)  # 添加 batch 维度
        # 3. 确保数据类型为 float32，如果需要 (比如应用到神经网络)
        mask_out = mask_out.float() / 255.0  # 如果你想标准化为 [0, 1] 的范围
        # 4. 输出四维张量的形状
        # print("mask_tensor", mask_out.shape)  # 输出: torch.Size([1, height, width, channels])
        # print(masked_image.shape)  # 输出: torch.Size([1, height, width, channels])
        return (masked_image, mask_out)


class TextMaskMy:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "model": (
                    ['DB_IC15_resnet18.onnx', 'DB_IC15_resnet50.onnx', 'DB_TD500_resnet18.onnx',
                     'DB_TD500_resnet50.onnx'],
                    {"default": "DB_TD500_resnet50.onnx", "tooltip": "选择检测模型"}),
            },

        }

    CATEGORY = "My_node/example"
    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "example_func"
    
    # @classmethod
    # def IS_CHANGED(s, image, model):
    #     # 计算输入的哈希值，确保只有在输入变化时才重新计算
    #     m = hashlib.sha256()
        
    #     # 对图像进行哈希
    #     images_flat = image.reshape(-1).numpy().tobytes()
    #     m.update(images_flat[:1024])  # 只使用部分数据做哈希，避免计算过重
        
    #     # 将其他参数也加入哈希计算
    #     m.update(str(model).encode())
        
    #     return m.digest().hex()

    def example_func(self, image, model):
        start_time = time.time()
        image_np = image.squeeze(0).detach().cpu().numpy()
        # 假设 ComfyUI 的张量像素值范围是 [0, 1]，将其转换为 [0, 255]，并将类型转换为 uint8
        image_np = (image_np * 255).astype(np.uint8)
        # concurrent_dir = os.path.dirname(os.path.dirname(__file__))
        # image_np_save_path = os.path.join(concurrent_dir, "temp", "image_np_save.jpg")
        # cv2.imwrite(image_np_save_path, image_np)
        width, height = image_np.shape[1], image_np.shape[0]
        # 创建一个与图像相同大小的空白蒙版，初始化为黑色
        mask = np.zeros((height, width), dtype=np.uint8)

        # 1. 加载预训练模型的权重和配置文件
        dir_node = os.path.dirname(os.path.dirname(__file__))
        model_dir = os.path.join(dir_node, "models", "text_det_model")
        model_weights = os.path.join(model_dir, model)
        model = cv2.dnn_TextDetectionModel_DB(model_weights)
        # 2. 设置模型输入的预处理参数
        model.setInputParams(
            scale=1.0 / 255.0,
            size=(736, 736),
            mean=(122.67891434, 116.66876762, 104.00698793),
            swapRB=True,
        )
        # 3. 读取图像
        # image = cv2.imread(image_np_save_path)
        # 4. 执行文本检测
        boxes, scores = model.detect(image_np)
        # 5. 可视化检测结果
        for box in boxes:
            points = box.astype(np.int32)
            cv2.polylines(image_np, [points], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.fillPoly(mask, [points], color=(255, 255, 255))
        image_np_bgra = image_np
        masked_image = torch.tensor(image_np_bgra).unsqueeze(0)  # 添加 batch 维度
        masked_image = masked_image.float() / 255.0  # 如果你想标准化为 [0, 1] 的范围
        mask_out = torch.tensor(mask).unsqueeze(0)  # 变为 [1, height, width, 1]
        mask_out = mask_out.float() / 255.0  # 如果你想标准化为 [0, 1] 的范围
        print(f"TextMaskMy cost time: {time.time() - start_time} s")
        return (masked_image, mask_out)


previous_dino_model = ""


class GroundingDinoGetBbox:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "grounding_dino_model": (list_groundingdino_model(),),
                "threshold": ("FLOAT", {"default": 0.3, "min": 0, "max": 1.0, "step": 0.01}),
                "prompt": ("STRING", {"default": "subject"}),
            },
        }

    CATEGORY = "My_node/mask"

    RETURN_TYPES = ("IMAGE", "STRING")
    FUNCTION = "example_func"

    # @classmethod
    # def IS_CHANGED(s, image, grounding_dino_model, threshold, prompt):
    #     # 计算输入的哈希值，确保只有在输入变化时才重新计算
    #     m = hashlib.sha256()
        
    #     # 对图像进行哈希
    #     images_flat = image.reshape(-1).numpy().tobytes()
    #     m.update(images_flat[:1024])  # 只使用部分数据做哈希，避免计算过重
        
    #     # 将其他参数也加入哈希计算
    #     m.update(str(grounding_dino_model).encode())
    #     m.update(str(threshold).encode())
    #     m.update(str(prompt).encode())
        
    #     return m.digest().hex()

    def example_func(self, image, grounding_dino_model, threshold, prompt):
        global DINO_MODEL
        global previous_dino_model

        if previous_dino_model != grounding_dino_model:
            DINO_MODEL = load_groundingdino_model(grounding_dino_model)
            previous_dino_model = grounding_dino_model
        # DINO_MODEL = load_groundingdino_model(grounding_dino_model)
        start_time = time.time()
        image_ = image.squeeze(0).detach().cpu()
        image_np = image_.numpy()
        i = pil2tensor(tensor2pil(image_).convert('RGB'))
        _image = tensor2pil(i).convert('RGBA')
        # 假设 ComfyUI 的张量像素值范围是 [0, 1]，将其转换为 [0, 255]，并将类型转换为 uint8
        image_np = (image_np * 255).astype(np.uint8)
        # concurrent_dir = os.path.dirname(os.path.dirname(__file__))
        # image_np_save_path = os.path.join(concurrent_dir, "temp", "image_np_save.jpg")
        # cv2.imwrite(image_np_save_path, image_np)
        # width, height = image_np.shape[1], image_np.shape[0]
        boxes = groundingdino_predict(DINO_MODEL, _image, prompt, threshold)
        # 转化为list
        boxes = boxes.tolist()
        str_bbox_list = []
        if len(boxes) != 0:
            for box in boxes:
                print(box)
                print(type(box))
                x_y_dict = {"x": int((box[0] + box[2]) / 2), "y": int((box[1] + box[3]) / 2)}
                str_bbox_list.append(x_y_dict)
                # 使用 cv2.rectangle 在图像上绘制矩形
                cv2.rectangle(image_np, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color=(255, 0, 0),
                              thickness=5)
                cv2.circle(image_np, (int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)), 8, (0, 0, 255), -1)

        # else:
        #     str_bbox_list = [{"x": 0, "y": 0}]

        # boxes = boxes.astype(np.int32)
        print(boxes)
        # print(boxes.shape)
        print(len(boxes))
        # for box in boxes:
        #     points = box.astype(np.int32)
        #     cv2.polylines(image_np, [points], isClosed=True, color=(0, 255, 0), thickness=2)
        # image_np_bgra = image_np
        masked_image = torch.tensor(image_np).unsqueeze(0)  # 添加 batch 维度
        masked_image = masked_image.float() / 255.0  # 如果你想标准化为 [0, 1] 的范围
        # mask_out = torch.tensor(mask).unsqueeze(0)  # 变为 [1, height, width, 1]
        # mask_out = mask_out.float() / 255.0  # 如果你想标准化为 [0, 1] 的范围
        print(f"GroundingDinoGetBbox cost time: {time.time() - start_time} s")
        return (masked_image, str(str_bbox_list))
        # return (image,)


def fill_region_mask(mask_tensor):
    n, h, w = mask_tensor.shape
    result = []

    for i in range(n):
        mask = mask_tensor[i]

        # 1. 找到所有白色区域
        white_region = mask == 1

        # 2. 通过腐蚀找到边界
        eroded = ndimage.binary_erosion(white_region).astype(int)

        # 3. 填充白色区域内部的黑色区域
        filled = ndimage.binary_fill_holes(eroded).astype(int)

        # 4. 只要是被白色完全包围的黑色区域就变为白色
        # result[i] = np.where(filled == 1, 1, mask)
        # 将修改后的 mask 重新放回 tensor
        # 4. 只要是被白色完全包围的黑色区域就变为白色
        mask = np.where(filled == 1, 1, mask)
        result.append(torch.tensor(mask, dtype=torch.float32))
    result = torch.stack(result, dim=0)

    return result


"""遮罩相加"""


def add_masks(dilation_erosion, *masks):
    if not masks:
        return None

    result_mask = masks[0].cpu()  # 初始化为第0个mask
    for mask in masks[1:]:
        mask = mask.cpu()
        cv2_mask = np.array(mask) * 255
        # 如果cv2_mask为纯黑，跳过
        if cv2_mask.sum() == 0:
            continue
        # Check if result_mask and cv2_mask have the same shape
        if result_mask.shape == cv2_mask.shape:
            cv2_result_mask = cv2.add(np.array(result_mask) * 255, cv2_mask)
            # Clamp the result after each addition
            result_mask = torch.clamp(result_mask / 255.0, min=0, max=1)
        # else:
        #     # If shapes are incompatible, skip this mask
        #     continue

    # result_mask = torch.clamp(result_mask / 255.0, min=0, max=1)

    # Convert to numpy for OpenCV operations
    cv2_result_mask = (result_mask.cpu().numpy() * 255)

    # Ensure the mask is 2D
    if cv2_result_mask.ndim > 2:
        cv2_result_mask = cv2_result_mask.squeeze()

    # Create a kernel for dilation/erosion
    kernel_size = max(1, abs(dilation_erosion))  # Ensure kernel size is at least 1
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    if dilation_erosion > 0:
        cv2_result_mask = cv2.dilate(cv2_result_mask, kernel, iterations=1)
    elif dilation_erosion < 0:
        cv2_result_mask = cv2.erode(cv2_result_mask, kernel, iterations=1)

    # Convert back to torch tensor
    result_mask = torch.from_numpy(cv2_result_mask / 255.0).float()

    return result_mask


class MaskAdd:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask_count": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
                "dilation_erosion": ("INT", {"default": 0, "min": -50, "max": 50, "step": 1}),
            },
            "optional": {
                **{f"mask_{i}": ("MASK",) for i in range(1, 11)},
            }
        }

    CATEGORY = "My_node/mask"
    RETURN_TYPES = ("MASK",)
    FUNCTION = "add_multiple_masks"
    
    # @classmethod
    # def IS_CHANGED(s, mask_count, dilation_erosion, **kwargs):
    #     # 计算输入的哈希值，确保只有在输入变化时才重新计算
    #     m = hashlib.sha256()
        
    #     # 将参数加入哈希计算
    #     m.update(str(mask_count).encode())
    #     m.update(str(dilation_erosion).encode())
        
    #     # 对每个mask进行哈希
    #     for i in range(1, mask_count + 1):
    #         mask = kwargs.get(f"mask_{i}")
    #         if mask is not None:
    #             mask_flat = mask.reshape(-1).numpy().tobytes()
    #             m.update(mask_flat[:1024])  # 只使用部分数据做哈希，避免计算过重
        
    #     return m.digest().hex()

    def add_multiple_masks(self, mask_count, dilation_erosion, **kwargs):
        masks = [kwargs.get(f"mask_{i}") for i in range(1, mask_count + 1) if kwargs.get(f"mask_{i}") is not None]
        result_mask = add_masks(dilation_erosion, *masks)
        return (result_mask,)


"""遮罩相减"""


def process_masks(dilation_erosion_1, fill_region_mask_1, dilation_erosion_2, fill_region_mask_2,
                  dilation_erosion_result, fill_region_mask_result, n, *masks):
    if not masks or len(masks) < 1:
        return None, None, None

    # Sum the first n masks with dilation/erosion
    sum_first_n = masks[0].cpu()
    for mask in masks[1:n]:
        if mask is not None:
            mask = mask.cpu()
            cv2_mask = np.array(mask) * 255
            # 如果cv2_mask为纯黑，跳过
            if cv2_mask.sum() == 0:
                continue
            # Check if sum_first_n and cv2_mask have the same shape
            if sum_first_n.shape == cv2_mask.shape:
                cv2_sum_first_n = cv2.add(np.array(sum_first_n) * 255, cv2_mask)
                sum_first_n = torch.from_numpy(cv2_sum_first_n)
                sum_first_n = torch.clamp(sum_first_n / 255.0, min=0, max=1)

    if dilation_erosion_1 != 0:
        # 对 sum_first_n 应用 dilation/erosion
        kernel_size_1 = max(1, abs(dilation_erosion_1))  # Ensure kernel size is at least 1
        kernel_1 = np.ones((kernel_size_1, kernel_size_1), np.uint8)
        # 对每个样本进行 dilation/erosion 操作
        for i in range(sum_first_n.shape[0]):
            if dilation_erosion_1 > 0:
                dilated = cv2.dilate(np.array(sum_first_n[i]) * 255, kernel_1, iterations=1)
            elif dilation_erosion_1 < 0:
                dilated = cv2.erode(np.array(sum_first_n[i]) * 255, kernel_1, iterations=1)
            else:
                dilated = np.array(sum_first_n[i])  # 如果 dilation_erosion_1 为 0，保持原样
            sum_first_n[i] = torch.from_numpy(dilated / 255.0).float()  # 转换为张量并赋值

    if fill_region_mask_1:
        sum_first_n = fill_region_mask(sum_first_n)
    # Sum the masks from n to 12 with dilation/erosion
    sum_n_to_12 = torch.zeros_like(masks[0]).cpu()
    for mask in masks[n:12]:
        if mask is not None:
            mask = mask.cpu()
            cv2_mask = np.array(mask) * 255
            # 如果cv2_mask为纯黑，跳过
            if cv2_mask.sum() == 0:
                continue
            # Check if sum_n_to_12 and cv2_mask have the same shape
            if sum_n_to_12.shape == cv2_mask.shape:
                cv2_sum_n_to_12 = cv2.add(np.array(sum_n_to_12) * 255, cv2_mask)
                sum_n_to_12 = torch.from_numpy(cv2_sum_n_to_12)
                sum_n_to_12 = torch.clamp(sum_n_to_12 / 255.0, min=0, max=1)

    if dilation_erosion_2 != 0:
        # 对 sum_n_to_12 应用 dilation/erosion
        kernel_size_2 = max(1, abs(dilation_erosion_2))  # Ensure kernel size is at least 1
        kernel_2 = np.ones((kernel_size_2, kernel_size_2), np.uint8)
        # 对每个样本进行 dilation/erosion 操作
        for i in range(sum_n_to_12.shape[0]):
            if dilation_erosion_2 > 0:
                dilated = cv2.dilate(np.array(sum_n_to_12[i]) * 255, kernel_2, iterations=1)
            elif dilation_erosion_2 < 0:
                dilated = cv2.erode(np.array(sum_n_to_12[i]) * 255, kernel_2, iterations=1)
            else:
                dilated = np.array(sum_n_to_12[i])  # 如果 dilation_erosion_2 为 0，保持原样
            sum_n_to_12[i] = torch.from_numpy(dilated / 255.0).float()  # 转换为张量并赋值

    # If no masks in n to 12, use a black mask
    if sum_n_to_12.sum() == 0:
        sum_n_to_12 = torch.zeros_like(sum_first_n).cpu()
    if fill_region_mask_2:
        sum_n_to_12 = fill_region_mask(sum_n_to_12)
    # Subtract the two sums using cv2.subtract
    if sum_first_n.shape == sum_n_to_12.shape:
        cv2_result_mask = cv2.subtract(np.array(sum_first_n) * 255, np.array(sum_n_to_12) * 255)
        cv2_result_mask = torch.from_numpy(cv2_result_mask)
        cv2_result_mask = torch.clamp(cv2_result_mask / 255.0, min=0, max=1)
    else:
        cv2_result_mask = sum_first_n

    if dilation_erosion_result != 0:
        kernel_size_result = max(1, abs(dilation_erosion_result))  # Ensure kernel size is at least 1
        kernel_result = np.ones((kernel_size_result, kernel_size_result), np.uint8)
        # 对每个样本进行 dilation/erosion 操作
        for i in range(cv2_result_mask.shape[0]):
            if dilation_erosion_result > 0:
                dilated = cv2.dilate(np.array(cv2_result_mask[i]) * 255, kernel_result, iterations=1)
            elif dilation_erosion_result < 0:
                dilated = cv2.erode(np.array(cv2_result_mask[i]) * 255, kernel_result, iterations=1)
            else:
                dilated = np.array(cv2_result_mask[i])  # 如果 dilation_erosion_result 为 0，保持原样
            cv2_result_mask[i] = torch.from_numpy(dilated / 255.0).float()  # 转换为张量并赋值
    if fill_region_mask_result:
        cv2_result_mask = fill_region_mask(cv2_result_mask)

    # Ensure the mask is 2D
    if sum_first_n.ndim < 3:
        sum_first_n = sum_first_n.squeeze(0)
    if sum_n_to_12.ndim < 3:
        sum_n_to_12 = sum_n_to_12.squeeze(0)
    if cv2_result_mask.ndim < 3:
        cv2_result_mask = cv2_result_mask.squeeze(0)

    return sum_first_n, sum_n_to_12, cv2_result_mask


class MaskSubtract:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "n": ("INT", {"default": 2, "min": 2, "max": 12, "step": 1}),
                "dilation_erosion_1": ("INT", {"default": 0, "min": -50, "max": 50, "step": 1}),
                "fill_region_mask_1": ("BOOLEAN", {"default": False, "tooltips": "是否填充第一个遮罩"}),
                "dilation_erosion_2": ("INT", {"default": 0, "min": -50, "max": 50, "step": 1}),
                "fill_region_mask_2": ("BOOLEAN", {"default": False, "tooltips": "是否填充第二个遮罩"}),
                "dilation_erosion_result": ("INT", {"default": 0, "min": -50, "max": 50, "step": 1}),
                "fill_region_mask_result": ("BOOLEAN", {"default": False, "tooltips": "是否填充结果遮罩"}),

            },
            "optional": {
                **{f"mask_{i}": ("MASK",) for i in range(1, 13)},
            }
        }

    CATEGORY = "My_node/mask"
    RETURN_TYPES = ("MASK", "MASK", "MASK")
    RETURN_NAMES = ("sum_first_n", "sum_n_to_12", "result_mask")
    FUNCTION = "subtract_multiple_masks"

    # @classmethod
    # def IS_CHANGED(s, n, dilation_erosion_1, fill_region_mask_1, dilation_erosion_2, fill_region_mask_2,
    #              dilation_erosion_result, fill_region_mask_result, **kwargs):
    #     # 计算输入的哈希值，确保只有在输入变化时才重新计算
    #     m = hashlib.sha256()
        
    #     # 将参数加入哈希计算
    #     m.update(str(n).encode())
    #     m.update(str(dilation_erosion_1).encode())
    #     m.update(str(fill_region_mask_1).encode())
    #     m.update(str(dilation_erosion_2).encode())
    #     m.update(str(fill_region_mask_2).encode())
    #     m.update(str(dilation_erosion_result).encode())
    #     m.update(str(fill_region_mask_result).encode())
        
    #     # 对每个mask进行哈希
    #     for i in range(1, 13):
    #         mask = kwargs.get(f"mask_{i}")
    #         if mask is not None:
    #             mask_flat = mask.reshape(-1).numpy().tobytes()
    #             m.update(mask_flat[:1024])  # 只使用部分数据做哈希，避免计算过重
        
    #     return m.digest().hex()

    def subtract_multiple_masks(self, n, dilation_erosion_1, fill_region_mask_1, dilation_erosion_2, fill_region_mask_2,
                                dilation_erosion_result, fill_region_mask_result, **kwargs):
        # Collect all masks, including None, to maintain order
        masks = [kwargs.get(f"mask_{i}") for i in range(1, 13)]
        result_masks = process_masks(dilation_erosion_1, fill_region_mask_1, dilation_erosion_2, fill_region_mask_2,
                                     dilation_erosion_result, fill_region_mask_result, n, *masks)
        return result_masks


"""遮罩重叠"""


class MaskOverlap:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask_1": ("MASK",),
                "mask_2": ("MASK",),
                "threshold": ("INT", {"default": 10, "min": 0, "max": 255, "step": 1}),
                "tolerance": ("FLOAT", {"default": 0.05, "min": 0, "max": 1.0, "step": 0.01}),
            },
        }

    CATEGORY = "My_node/mask"
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask_2",)
    FUNCTION = "calculate_overlap"
    
    # @classmethod
    # def IS_CHANGED(s, mask_1, mask_2, threshold, tolerance):
    #     # 计算输入的哈希值，确保只有在输入变化时才重新计算
    #     m = hashlib.sha256()
        
    #     # 对mask进行哈希
    #     if mask_1 is not None:
    #         mask_1_flat = mask_1.reshape(-1).numpy().tobytes()
    #         m.update(mask_1_flat[:1024])  # 只使用部分数据做哈希，避免计算过重
        
    #     if mask_2 is not None:
    #         mask_2_flat = mask_2.reshape(-1).numpy().tobytes()
    #         m.update(mask_2_flat[:1024])
        
    #     # 将其他参数也加入哈希计算
    #     m.update(str(threshold).encode())
    #     m.update(str(tolerance).encode())
        
    #     return m.digest().hex()

    def calculate_overlap(self, mask_1, mask_2, threshold, tolerance):
        # 将mask转换为NumPy数组
        mask_1_np = mask_1.cpu().numpy() * 255
        mask_2_np = mask_2.cpu().numpy() * 255

        # 计算第一个遮罩中大于阈值的像素数量
        count_1 = np.sum(mask_1_np > threshold)

        # 计算重合点数量
        overlap_count = np.sum((mask_1_np > threshold) & (mask_2_np > threshold))
        print("重合个数", overlap_count)
        print("总数", count_1)

        # 计算重合率
        overlap_ratio = overlap_count / count_1 if count_1 > 0 else 0

        # 判断重合率是否大于容差
        if overlap_ratio > tolerance:
            # 返回一个与mask_2相同维度的纯黑mask
            black_mask = torch.zeros_like(mask_2)
            return (black_mask,)
        else:
            return (mask_2,)


class PasteMasksMy:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "base_image": ("IMAGE",),  # 用于生成纯黑底图
                "mask_images": ("MASK",),  # 输入的遮罩图像，形如(3, 720, 720)
                "squares_info": ("STRING",),  # 输入的位置信息，形如str(squares_info)
            }
        }

    RETURN_TYPES = ("MASK",)  # 返回合成后的遮罩
    FUNCTION = "paste_masks"
    CATEGORY = "My_node/mask"
    
    # @classmethod
    # def IS_CHANGED(s, base_image, mask_images, squares_info):
    #     # 计算输入的哈希值，确保只有在输入变化时才重新计算
    #     m = hashlib.sha256()
        
    #     # 对图像进行哈希
    #     base_image_flat = base_image.reshape(-1).numpy().tobytes()
    #     m.update(base_image_flat[:1024])  # 只使用部分数据做哈希，避免计算过重
        
    #     # 对masks进行哈希
    #     mask_images_flat = mask_images.reshape(-1).numpy().tobytes()
    #     m.update(mask_images_flat[:1024])
        
    #     # 将其他参数也加入哈希计算
    #     m.update(str(squares_info).encode())
        
    #     return m.digest().hex()

    def paste_masks(self, base_image, mask_images, squares_info):
        # 将squares_info从字符串转换为列表
        squares_info = eval(squares_info)  # 使用eval函数

        # 获取批次大小
        batch_size = base_image.shape[0]

        # 创建一个与base_image相同大小的纯黑遮罩
        result_mask = torch.zeros((batch_size, base_image.shape[1], base_image.shape[2]), dtype=torch.float32)

        cur_index = 0

        for i, mask_info in enumerate(squares_info):
            for mask in mask_info:
                x, y, size = mask  # 获取x, y坐标和大小
                mask_image = mask_images[cur_index]  # 假设mask_images是一个包含多个遮罩的列表

                # 将mask_image转换为numpy数组
                mask_image_np = (mask_image.cpu().numpy() * 255).astype(np.uint8)  # 转换为uint8格式
                mask_image_resized = cv2.resize(mask_image_np, (size, size))  # 调整遮罩图像大小

                # 将调整大小的遮罩图像转换为PyTorch张量
                mask_tensor = torch.from_numpy(mask_image_resized).float() / 255.0

                # 确保目标区域的大小与mask_tensor匹配
                target_height = min(size, result_mask.shape[1] - y)
                target_width = min(size, result_mask.shape[2] - x)
                mask_tensor = mask_tensor[:target_height, :target_width]

                # 只粘贴非黑色区域
                non_black_area = mask_tensor > 0
                result_mask[i, y:y + target_height, x:x + target_width][non_black_area] = mask_tensor[non_black_area]

                cur_index += 1
        # 检查result_mask是不是n,h,w的维度
        if len(result_mask.shape) != 3:
            result_mask = result_mask.unsqueeze(0)

        return (result_mask,)  # 返回合成后的遮罩


def resize_tensor_torch(tensor, target_shape=(256, 256)):
    """
    将形状为 (N, H, W, C) 的张量调整为 (N, 256, 256, C)

    :param tensor: 输入张量，形状为 (N, H, W, C)
    :param target_shape: 目标形状 (height, width)，默认为 (256, 256)
    :return: 调整后的张量，形状为 (N, 256, 256, C)
    """
    N, H, W, C = tensor.shape
    target_height, target_width = target_shape
    tensor = tensor.permute(0, 3, 1, 2)  # 调整为 (N, C, H, W)
    resized_tensor = F.interpolate(tensor, size=(target_height, target_width), mode='bilinear', align_corners=False)
    resized_tensor = resized_tensor.permute(0, 2, 3, 1)  # 调整回 (N, H, W, C)
    return resized_tensor


def resize_mask_tensor_torch(tensor, target_shape=(256, 256)):
    """
    将形状为 (N, 256, 256) 的张量调整为 (N, H, W)

    :param tensor: 输入张量，形状为 (N, 256, 256)
    :param target_shape: 目标形状 (height, width)，默认为 (256, 256)
    :return: 调整后的张量，形状为 (N, H, W)
    """
    N, H, W = tensor.shape
    target_height, target_width = target_shape
    resized_tensor = F.interpolate(tensor.unsqueeze(1), size=(target_height, target_width), mode='bilinear',
                                   align_corners=False)
    resized_tensor = resized_tensor.squeeze(1)  # 移除增加的维度
    return resized_tensor


class RemoveGlassesFaceMask:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),  # 输入图像，形状为 (n, h, w, c)
            }
        }

    CATEGORY = "My_node/mask"
    RETURN_TYPES = ("MASK",)
    FUNCTION = "get_glasses_mask"
    
    # @classmethod
    # def IS_CHANGED(s, image):
    #     # 计算输入的哈希值，确保只有在输入变化时才重新计算
    #     m = hashlib.sha256()
        
    #     # 对图像进行哈希
    #     images_flat = image.reshape(-1).numpy().tobytes()
    #     m.update(images_flat[:1024])  # 只使用部分数据做哈希，避免计算过重
        
    #     return m.digest().hex()

    def get_glasses_mask(self, image):
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        cur_dir = os.path.dirname(cur_dir)
        model_path = os.path.join(cur_dir, r"models\face_remove_glasses\dfl_xseg.onnx")
        h, w = image.shape[1], image.shape[2]
        image = resize_tensor_torch(image)
        # 转化为np
        image = image.numpy()
        mask_image = infer_and_get_mask(model_path, image)
        # mask_image_1 = mask_image[0]
        # mask_image_2 = np.squeeze(mask_image_1, axis=-1)
        # print(mask_image[0].shape)
        # print(mask_image[0][0].shape)
        # mask_image_3 = torch.from_numpy(mask_image_2).float()
        # mask_image_4 = resize_mask_tensor_torch(mask_image_3, (h, w))
        last_mask = []
        for i in range(image.shape[0]):
            occlusion_mask = mask_image[0][i].transpose(0, 1, 2).clip(0, 1).astype(np.float32)
            occlusion_mask = cv2.resize(occlusion_mask, (h, w))
            occlusion_mask = (cv2.GaussianBlur(occlusion_mask.clip(0, 1), (0, 0), 2).clip(0.5, 1) - 0.5) * 2
            last_mask.append(torch.tensor(occlusion_mask, dtype=torch.float32))
        last_mask_tensor = torch.stack(last_mask, dim=0)

        return (last_mask_tensor,)


class AdjustMaskValues:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "masks": ("MASK",),  # 输入的遮罩，形状为 (n, h, w)
                "value": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),  # 要设置的值
            }
        }

    CATEGORY = "My_node/mask"
    RETURN_TYPES = ("MASK",)
    FUNCTION = "adjust_values"
    
    # @classmethod
    # def IS_CHANGED(s, masks, value):
    #     # 计算输入的哈希值，确保只有在输入变化时才重新计算
    #     m = hashlib.sha256()
        
    #     # 对masks进行哈希
    #     if masks is not None:
    #         masks_flat = masks.reshape(-1).numpy().tobytes()
    #         m.update(masks_flat[:1024])  # 只使用部分数据做哈希，避免计算过重
        
    #     # 将其他参数也加入哈希计算
    #     m.update(str(value).encode())
        
    #     return m.digest().hex()

    def adjust_values(self, masks, value):
        # 确保值在0到1之间
        value = max(0.0, min(1.0, value))

        # 创建一个新的遮罩张量
        adjusted_masks = masks.clone()
        print(f"---------------------------------masks shape: {adjusted_masks.shape}")
        for i in range(masks.shape[0]):
            adjusted_masks[i][masks[i] != 0] = value

        # 将遮罩中为1的值设置为指定的值
        # adjusted_masks[masks != 0] = value
        print(f"---------------------------------masks new shape: {adjusted_masks.shape}")

        return (adjusted_masks,)


class NormalizeMask:
    """
    一个用于归一化遮罩（Mask）的节点。
    可以将遮罩的值进行二值化（0和1或0和255）或范围归一化（0到1或0到255），并可选择输出的数据精度。
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),
                "normalization": (["To 0 and 1", "To 0-1 range", "To 0 and 255", "To 0-255 range"], 
                                 {"default": "To 0-1 range", "tooltip": "选择归一化方法"}),
                "precision": (["float32", "float16", "bfloat16"], {"default": "float32", "tooltip": "选择输出精度"}),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "normalize_mask"
    CATEGORY = "My_node/mask"

    def normalize_mask(self, mask, normalization, precision):
        # 映射精度字符串到torch.dtype
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        target_dtype = dtype_map.get(precision, torch.float32)

        # 获取输入张量的设备
        device = mask.device

        if normalization == "To 0 and 1":
            # 二值化处理：将所有非零值设为1，零值保持为0
            normalized_mask = (mask > 0).to(device=device, dtype=target_dtype)
        elif normalization == "To 0 and 255":
            # 二值化处理：将所有非零值设为255，零值保持为0
            normalized_mask = (mask > 0).to(device=device, dtype=target_dtype) * 255.0
        elif normalization == "To 0-1 range":
            # 范围归一化：将每个mask的值缩放到[0, 1]范围
            # 在同一设备上创建
            normalized_mask = torch.zeros_like(mask, device=device)
            for i in range(mask.shape[0]):
                single_mask = mask[i]
                min_val = single_mask.min()
                max_val = single_mask.max()

                if max_val > min_val:
                    # 执行归一化 (x - min) / (max - min)
                    normalized_mask[i] = (single_mask - min_val) / (max_val - min_val)
                else:
                    # 如果所有像素值都相同，则进行二值化处理
                    normalized_mask[i] = (single_mask > 0).float()

            normalized_mask = normalized_mask.to(device=device, dtype=target_dtype)
        elif normalization == "To 0-255 range":
            # 范围归一化：将每个mask的值缩放到[0, 255]范围
            normalized_mask = torch.zeros_like(mask, device=device)
            for i in range(mask.shape[0]):
                single_mask = mask[i]
                min_val = single_mask.min()
                max_val = single_mask.max()

                if max_val > min_val:
                    # 执行归一化 (x - min) / (max - min) * 255
                    normalized_mask[i] = (single_mask - min_val) / (max_val - min_val) * 255.0
                else:
                    # 如果所有像素值都相同，则进行二值化处理
                    normalized_mask[i] = (single_mask > 0).float() * 255.0

            normalized_mask = normalized_mask.to(device=device, dtype=target_dtype)
        else:
            # 默认情况，不应发生
            normalized_mask = mask.to(device=device, dtype=target_dtype)

        return (normalized_mask,)


class AnalyzeMask:
    """
    一个用于分析遮罩（Mask）类型的节点。
    它可以判断遮罩是二值型（只包含0和1或0和255）还是范围型，并显示其值的范围。
    支持批量分析多张遮罩。
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),
            }
        }

    OUTPUT_NODE = True
    RETURN_TYPES = ()
    FUNCTION = "analyze"
    CATEGORY = "My_node/mask"

    def analyze(self, mask):
        if mask is None or mask.numel() == 0:
            result_string = "空遮罩或无效遮罩"
            return {"ui": {"text": [result_string]}, "result": result_string}

        # 确保mask是3D张量 (n,h,w)
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)  # 添加batch维度

        # 获取整体形状信息
        mask_shape = mask.shape
        shape_info = f"遮罩整体形状: {mask_shape}"
        
        results = [shape_info]  # 添加形状信息作为第一行
        
        for i in range(mask.shape[0]):
            single_mask = mask[i]
            
            # 获取单个遮罩的形状
            single_shape = single_mask.shape
            
            # 获取最小值和最大值
            min_val = single_mask.min().item()
            max_val = single_mask.max().item()

            # 获取所有唯一值以进行判断
            unique_vals = torch.unique(single_mask)

            # 检查是否为0-1二值型
            is_binary_01 = all(torch.isclose(v, torch.tensor(0.0, device=mask.device)) or 
                            torch.isclose(v, torch.tensor(1.0, device=mask.device)) for v in unique_vals)
            
            # 检查是否为0-255二值型
            is_binary_0255 = all(torch.isclose(v, torch.tensor(0.0, device=mask.device)) or 
                              torch.isclose(v, torch.tensor(255.0, device=mask.device)) for v in unique_vals)

            if is_binary_01:
                result = f"遮罩 #{i+1}: 二值型遮罩 (0-1). \n形状: {single_shape}, \n范围: ({min_val:.4f}, {max_val:.4f})"
            elif is_binary_0255:
                result = f"遮罩 #{i+1}: 二值型遮罩 (0-255). \n形状: {single_shape}, \n范围: ({min_val:.4f}, {max_val:.4f})"
            elif 0 <= min_val <= 1 and 0 <= max_val <= 1:
                result = f"遮罩 #{i+1}: 范围型遮罩 (0-1). \n形状: {single_shape}, \n范围: [{min_val:.4f}, {max_val:.4f}]"
            elif 0 <= min_val <= 255 and 0 <= max_val <= 255:
                result = f"遮罩 #{i+1}: 范围型遮罩 (0-255). \n形状: {single_shape}, \n范围: [{min_val:.4f}, {max_val:.4f}]"
            else:
                result = f"遮罩 #{i+1}: 未知范围型遮罩. \n形状: {single_shape}, \n范围: [{min_val:.4f}, {max_val:.4f}]"
            
            results.append(result)

        # 将所有结果用换行符连接
        final_result = "\n".join(results)
        return {"ui": {"text": [final_result]}, "result": final_result}
