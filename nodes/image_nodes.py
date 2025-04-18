# -*- coding: utf-8 -*-
# Created time : 2024/09/12 22:36 
# Auther : ygh
# File   : image_nodes.py
# Description :
import numpy as np
# import time
import torch
from comfy import model_management
# from custom_nodes.facerestore_cf.facelib.utils.face_restoration_helper import FaceRestoreHelper
# from custom_nodes.facerestore_cf.facelib.detection.retinaface import retinaface
from torchvision.transforms.functional import normalize
from torchvision.utils import make_grid
# from comfy_extras.chainner_models import model_loading
# import torch.nn.functional as F
# import random
import math
import os
# import re
# import json
import hashlib
import cv2
import logging
# try:
#     import cv2
# except:
#     print("OpenCV not installed")
#     pass
from PIL import ImageGrab, ImageDraw, ImageFont, Image, ImageSequence, ImageOps
# from comfy.cli_args import args
# from comfy.utils import ProgressBar, common_upscale
import folder_paths
from nodes import MAX_RESOLUTION
import warnings

# from custom_nodes.A_my_nodes.nodes.get_result_text_p import run_script_with_subprocess, get_numpy_from_txt

warnings.filterwarnings("ignore")


def img2tensor(imgs, bgr2rgb=True, float32=True):
    """Numpy array to tensor.

    Args:
        imgs (list[ndarray] | ndarray): Input images.
        bgr2rgb (bool): Whether to change bgr to rgb.
        float32 (bool): Whether to change to float32.

    Returns:
        list[tensor] | tensor: Tensor images. If returned results only have
            one element, just return tensor.
    """

    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            if img.dtype == 'float64':
                img = img.astype('float32')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)


def tensor2img(tensor, rgb2bgr=True, out_type=np.uint8, min_max=(0, 1)):
    """Convert torch Tensors into image numpy arrays.

    After clamping to [min, max], values will be normalized to [0, 1].

    Args:
        tensor (Tensor or list[Tensor]): Accept shapes:
            1) 4D mini-batch Tensor of shape (B x 3/1 x H x W);
            2) 3D Tensor of shape (3/1 x H x W);
            3) 2D Tensor of shape (H x W).
            Tensor channel should be in RGB order.
        rgb2bgr (bool): Whether to change rgb to bgr.
        out_type (numpy type): output types. If ``np.uint8``, transform outputs
            to uint8 type with range [0, 255]; otherwise, float type with
            range [0, 1]. Default: ``np.uint8``.
        min_max (tuple[int]): min and max values for clamp.

    Returns:
        (Tensor or list): 3D ndarray of shape (H x W x C) OR 2D ndarray of
        shape (H x W). The channel order is BGR.
    """
    if not (torch.is_tensor(tensor) or (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError(f'tensor or list of tensors expected, got {type(tensor)}')

    if torch.is_tensor(tensor):
        tensor = [tensor]
    result = []
    for _tensor in tensor:
        _tensor = _tensor.squeeze(0).float().detach().cpu().clamp_(*min_max)
        _tensor = (_tensor - min_max[0]) / (min_max[1] - min_max[0])

        n_dim = _tensor.dim()
        if n_dim == 4:
            img_np = make_grid(_tensor, nrow=int(math.sqrt(_tensor.size(0))), normalize=False).numpy()
            img_np = img_np.transpose(1, 2, 0)
            if rgb2bgr:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 3:
            img_np = _tensor.numpy()
            img_np = img_np.transpose(1, 2, 0)
            if img_np.shape[2] == 1:  # gray image
                img_np = np.squeeze(img_np, axis=2)
            else:
                if rgb2bgr:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 2:
            img_np = _tensor.numpy()
        else:
            raise TypeError('Only support 4D, 3D or 2D tensor. ' f'But received with dimension: {n_dim}')
        if out_type == np.uint8:
            # Unlike MATLAB, numpy.unit8() WILL NOT round by default.
            img_np = (img_np * 255.0).round()
        img_np = img_np.astype(out_type)
        result.append(img_np)
    if len(result) == 1:
        result = result[0]
    return result


class LoadAndResizeImageMy:
    _color_channels = ["alpha", "red", "green", "blue"]

    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        scale_to_list = ['longest', 'None', 'shortest', 'width', 'height', 'total_pixel(kilo pixel)']
        return {"required":
            {
                "image": (sorted(files), {"image_upload": True}),
                "resize": ("BOOLEAN", {"default": False}),
                "width": ("INT", {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 8, }),
                "height": ("INT", {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 8, }),
                "repeat": ("INT", {"default": 1, "min": 1, "max": 1, "step": 1, }),
                "scale_to_side": (scale_to_list,),  # 是否按长边缩放
                "scale_to_length": ("INT", {"default": 1024, "min": 4, "max": 999999, "step": 1}),
                "keep_proportion": ("BOOLEAN", {"default": False, "tooltip": "保持比例"}),
                "divisible_by": ("INT", {"default": 2, "min": 0, "max": 512, "step": 1, }),
                "mask_channel": (s._color_channels,),
            },
        }


    CATEGORY = "My_node/image"
    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("image", "mask", "width", "height", "image_dir", "image_name", "image_ext")
    FUNCTION = "load_image"

    def load_image(self, image, resize, width, height, repeat,
                   keep_proportion, divisible_by, mask_channel, scale_to_side, scale_to_length):
        image_path = folder_paths.get_annotated_filepath(image)
        image_dir = os.path.dirname(image_path)
        image_full_name = os.path.basename(image_path)
        image_name, image_ext = os.path.splitext(image_full_name)

        import node_helpers
        img = node_helpers.pillow(Image.open, image_path)

        output_images = []
        output_masks = []
        w, h = None, None

        excluded_formats = ['MPO']

        W, H = img.size
        if resize:
            if keep_proportion:
                ratio = min(width / W, height / H)
                width = round(W * ratio)
                height = round(H * ratio)
            else:
                if width == 0:
                    width = W
                if height == 0:
                    height = H

            if divisible_by > 1:
                width = width - (width % divisible_by)
                height = height - (height % divisible_by)
        else:
            width, height = W, H
        ratio = width / height
        # calculate target width and height
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

        # img=img.resize((target_width, target_height), Image.Resampling.BILINEAR)
        for i in ImageSequence.Iterator(img):
            i = node_helpers.pillow(ImageOps.exif_transpose, i)

            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            image = i.convert("RGB")

            if len(output_images) == 0:
                w = image.size[0]
                h = image.size[1]

            if image.size[0] != w or image.size[1] != h:
                continue
            if resize:
                image = image.resize((width, height), Image.Resampling.BILINEAR)

            image = image.resize((target_width, target_height), Image.Resampling.BILINEAR)
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            mask = None
            c = mask_channel[0].upper()
            if c in i.getbands():
                if resize:
                    i = i.resize((width, height), Image.Resampling.BILINEAR)
                i=i.resize((target_width, target_height), Image.Resampling.BILINEAR)
                mask = np.array(i.getchannel(c)).astype(np.float32) / 255.0
                mask = torch.from_numpy(mask)
                if c == 'A':
                    mask = 1. - mask
            else:
                mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")

            output_images.append(image)
            output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1 and img.format not in excluded_formats:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]
            if repeat > 1:
                output_image = output_image.repeat(repeat, 1, 1, 1)
                output_mask = output_mask.repeat(repeat, 1, 1)

        return (output_image, output_mask, target_width, target_height, image_dir, image_name, image_ext)

    @classmethod
    def IS_CHANGED(s, image):
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, image):
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)

        return True

class ResizeImagesAndMasks:
    @classmethod
    def INPUT_TYPES(s):
        scale_to_list = ['longest', 'None', 'shortest', 'width', 'height', 'total_pixel(kilo pixel)']
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
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "resize_images_and_masks"

    def resize_images_and_masks(self, images, masks, resize, width, height, scale_to_side, scale_to_length, keep_proportion, divisible_by):
        output_images = []
        output_masks = []

        for img, mask in zip(images, masks):
            img_np = img.cpu().numpy()
            mask_np = mask.cpu().numpy()
            import logging

            logging.info(f"img_np.shape: {img_np.shape}")
            h, w = img_np.shape[0:2]
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

            img_resized = cv2.resize(img_np, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
            mask_resized = cv2.resize(mask_np, (target_width, target_height), interpolation=cv2.INTER_NEAREST)

            output_images.append(torch.from_numpy(img_resized).float())
            output_masks.append(torch.from_numpy(mask_resized).float())

        return torch.stack(output_images), torch.stack(output_masks)
# class CropFaceMy:
#     @classmethod
#     def INPUT_TYPES(s):
#         return {"required": {"image": ("IMAGE",),
#                              "facedetection": (["retinaface_resnet50", "retinaface_mobile0.25", "YOLOv5l", "YOLOv5n"],),
#                              "output_size": ("INT", {"default": 512, "min": 256, "max": 1024, "step": 2})
#                              }}

#     RETURN_TYPES = ("IMAGE","MASK","STRING")

#     FUNCTION = "crop_face"

#     CATEGORY = "My_node/image"

#     def __init__(self):
#         self.face_helper = None

#     def crop_face(self, image, facedetection, output_size):
#         device = model_management.get_torch_device()
#         if self.face_helper is None:
#             self.face_helper = FaceRestoreHelper(1, face_size=512, crop_ratio=(1, 1), det_model=facedetection,
#                                                  save_ext='png', use_parse=True, device=device)

#         image_np = 255. * image.cpu().numpy()

#         total_images = image_np.shape[0]
#         out_images = np.ndarray(shape=(total_images, output_size, output_size, 3))
#         next_idx = 0

#         # 获取输入样本的宽高
#         height, width = image.shape[1:3]  # 形状为 (2, 960, 720, 3)，获取宽高

#         # 创建一个与样本数量相同的纯黑张量
#         mask = torch.zeros((total_images, height, width), dtype=torch.float32)
#         squares_info = []  # 用于记录正方形的位置信息

#         for i in range(total_images):

#             cur_image_np = image_np[i, :, :, ::-1]

#             original_resolution = cur_image_np.shape[0:2]

#             if self.face_helper is None:
#                 return image

#             self.face_helper.clean_all()
#             self.face_helper.read_image(cur_image_np)
#             self.face_helper.get_face_landmarks_5(only_center_face=False, resize=640, eye_dist_threshold=5)
#             self.face_helper.align_warp_face()
#             print("面部信息:",self.face_helper.all_landmarks_5)
#             print("面部框:",self.face_helper.det_faces)
#             faces_found = len(self.face_helper.cropped_faces)
#             # print("self.face_helper.cropped_faces:",self.face_helper.cropped_faces)
#             if faces_found == 0:
#                 next_idx += 1  # output black image for no face
#             if out_images.shape[0] < next_idx + faces_found:
#                 print(out_images.shape)
#                 print((next_idx + faces_found, output_size, output_size, 3))
#                 print('aaaaa')
#                 out_images = np.resize(out_images, (next_idx + faces_found, output_size, output_size, 3))
#                 print(out_images.shape)
#             for j in range(faces_found):
#                 cropped_face_1 = self.face_helper.cropped_faces[j]
#                 cropped_face_2 = img2tensor(cropped_face_1 / 255., bgr2rgb=True, float32=True)
#                 normalize(cropped_face_2, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
#                 cropped_face_3 = cropped_face_2.unsqueeze(0).to(device)
#                 cropped_face_4 = tensor2img(cropped_face_3, rgb2bgr=True, min_max=(-1, 1)).astype('uint8')
#                 cropped_face_5 = cv2.cvtColor(cropped_face_4, cv2.COLOR_BGR2RGB)
#                 out_images[next_idx] = cv2.resize(cropped_face_5, (output_size, output_size))  # 调整输出图像大小
#                 next_idx += 1

#                 # 计算面部框的坐标
#                 bbox = self.face_helper.det_faces[j]
#                 x1, y1, x2, y2 = bbox[:4]  # 获取面部框的坐标
#                 center_x = (x1 + x2) / 2
#                 center_y = (y1 + y2) / 2
#                 side_length = min(x2 - x1, y2 - y1)  # 计算正方形的边长

#                 # 计算正方形的边长和中心点
#                 square_size = int(side_length)
#                 square_x = int(center_x - square_size // 2)
#                 square_y = int(center_y - square_size // 2)

#                 # 将正方形区域填充为白色
#                 mask[i, square_y:square_y + square_size, square_x:square_x + square_size] = 1.0

#                 # 记录正方形的信息
#                 if len(squares_info) <= i:
#                     squares_info.append([])
#                 squares_info[i].append([square_x, square_y, square_size])

#         cropped_face_6 = np.array(out_images).astype(np.float32) / 255.0
#         cropped_face_7 = torch.from_numpy(cropped_face_6)

#         return (cropped_face_7, mask, str(squares_info))  # 返回张量、mask和正方形信息

class CropFaceMy:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",),
                             "det_thresh": ("FLOAT", {"default": 0.5, "min": 0, "max": 1.0, "step": 0.01}),
                             "scale_factor": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 5.0, "step": 0.01}),
                             "device": ([-1, 0], {"default": -1, "tooltip": "-1:cpu, 0:gpu"}),
                             "det_size": ("INT", {"default": 512, "min": 256, "max": 1024, "step": 2}),
                             "output_size": ("INT", {"default": 512, "min": 256, "max": 1024, "step": 2}),
                             }}

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")

    FUNCTION = "crop_face"

    CATEGORY = "My_node/image"

    def __init__(self):
        self.face_helper = None

    def crop_face(self, image, det_thresh, scale_factor, device, det_size, output_size):

        if device < 0:
            provider = "CPU"
        else:
            provider = "CUDA"
        print("provider", provider)
        print("det_size", det_size)
        print("det_thresh", det_thresh)
        device = model_management.get_torch_device()
        model = insightface_loader(provider=provider, name='buffalo_l', det_thresh=det_thresh, size=det_size)
        image_np = 255. * image.cpu().numpy()

        total_images = image_np.shape[0]
        out_images = []

        # 获取输入样本的宽高
        height, width = image.shape[1:3]  # 形状为 (2, 960, 720, 3)，获取宽高

        # 创建一个与样本数量相同的纯黑张量
        mask = torch.zeros((total_images, height, width), dtype=torch.float32)
        squares_info = []  # 用于记录正方形的位置信息
        
        # 创建存储单个人脸遮罩的列表
        individual_masks = []
        
        # 创建用于存储眼睛关键点坐标的列表
        original_eye_points_list = []  # 原始图像上的眼睛关键点
        cropped_eye_points_list = []   # 裁剪图像上的眼睛关键点
        
        # 创建用于绘制关键点的图像副本
        original_images_with_points = []
        cropped_images_with_points = []
        
        image_np_new = image_np.copy()
        for i in range(total_images):
            # 为每个图像创建一个空的眼睛关键点列表
            original_eye_points_per_image = []
            cropped_eye_points_per_image = []
            
            # 为当前图像创建副本用于绘制关键点
            original_image_with_points = image_np_new[i].copy()
            
            while len(squares_info) <= i:  # 确保列表长度足够
                squares_info.append([])  # 添加空列表

            cur_image_np = image_np_new[i, :, :, ::-1]  # 将RGB转换为BGR

            # 检查图像是否为空
            if cur_image_np is None or cur_image_np.size == 0:
                print(f"Image at index {i} is empty or not loaded correctly.")
                continue

            faces = model.get(cur_image_np)
            faces_found = len(faces)
            for j in range(faces_found):
                bbox = faces[j].bbox
                x1, y1, x2, y2 = bbox[:4]  # 获取面部框的坐标
                
                # 获取关键点信息
                kps = faces[j].kps  # 获取关键点坐标
                # insightface的关键点格式为[左眼，右眼，鼻尖，左嘴角，右嘴角]
                left_eye = kps[0]  # 左眼中心坐标
                right_eye = kps[1]  # 右眼中心坐标
                
                # 存储原始图像中的眼睛关键点
                original_eye_point = {
                    "left_eye": left_eye.tolist(),
                    "right_eye": right_eye.tolist()
                }
                original_eye_points_per_image.append(original_eye_point)
                
                # 在原始图像上绘制这些点
                cv2.circle(original_image_with_points, (int(left_eye[0]), int(left_eye[1])), 5, (0, 0, 255), -1)  # 左眼用红色
                cv2.circle(original_image_with_points, (int(right_eye[0]), int(right_eye[1])), 5, (0, 255, 0), -1)  # 右眼用绿色

                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                # 根据scale_factor调整面部框的大小,
                # 如果x1,y1超出边界则取最小值0,0,
                # 如果x2,y2超出边界则取最大值width,height

                x1 = max(0, int(center_x - (x2 - x1) * scale_factor / 2))
                y1 = max(0, int(center_y - (y2 - y1) * scale_factor / 2))
                x2 = min(width, int(center_x + (x2 - x1) * scale_factor / 2))
                y2 = min(height, int(center_y + (y2 - y1) * scale_factor / 2))
                side_length = min(x2 - x1, y2 - y1)  # 计算正方形的边长

                # 计算正方形的边长和中心点
                square_size = int(side_length)
                if int(center_x - square_size // 2) <= 0:
                    square_x = 0
                elif int(center_x - square_size // 2) >= width:
                    square_x = width - 1
                else:
                    square_x = int(center_x - square_size // 2)
                if int(center_y - square_size // 2) <= 0:
                    square_y = 0
                elif int(center_y - square_size // 2) >= height:
                    square_y = height - 1
                else:
                    square_y = int(center_y - square_size // 2)
                    # square_size = min(int(side_length))
                square_size = min(square_size, width - square_x, height - square_y)

                # 将正方形区域填充为白色    
                mask[i, square_y:int(square_y + square_size), square_x:int(square_x + square_size)] = 1.0
                
                # 创建单独的人脸遮罩
                individual_mask = torch.zeros((total_images, height, width), dtype=torch.float32)
                individual_mask[i, square_y:int(square_y + square_size), square_x:int(square_x + square_size)] = 1.0
                individual_masks.append(individual_mask)

                # 记录正方形的信息
                squares_info[i].append([square_x, square_y, square_size])
                # 裁剪正方形面部
                cropped_face_1 = cur_image_np[square_y:int(square_y + square_size),
                                 square_x:int(square_x + square_size)]

                # 检查裁剪的面部图像是否为空
                if cropped_face_1 is None or cropped_face_1.size == 0:
                    print(f"Cropped face at index {i}, face {j} is empty.")
                    continue
                
                # 计算裁剪图像中眼睛关键点的新坐标
                # 从原始图像坐标转换为裁剪图像坐标
                def transform_point(point, square_x, square_y, output_size, square_size):
                    # 首先减去裁剪起点坐标，获得在裁剪图像中的相对坐标
                    relative_x = point[0] - square_x
                    relative_y = point[1] - square_y
                    # 然后根据输出尺寸进行缩放
                    scaled_x = relative_x * (output_size / square_size)
                    scaled_y = relative_y * (output_size / square_size)
                    return [scaled_x, scaled_y]
                
                # 转换眼睛中心点到裁剪图像坐标系
                cropped_left_eye = transform_point(left_eye, square_x, square_y, output_size, square_size)
                cropped_right_eye = transform_point(right_eye, square_x, square_y, output_size, square_size)
                
                # 存储裁剪图像中的眼睛关键点
                cropped_eye_point = {
                    "left_eye": cropped_left_eye,
                    "right_eye": cropped_right_eye
                }
                cropped_eye_points_per_image.append(cropped_eye_point)

                cropped_face_2 = img2tensor(cropped_face_1 / 255., bgr2rgb=True, float32=True)
                normalize(cropped_face_2, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
                cropped_face_3 = cropped_face_2.unsqueeze(0).to(device)
                cropped_face_4 = tensor2img(cropped_face_3, rgb2bgr=True, min_max=(-1, 1)).astype('uint8')
                cropped_face_5 = cv2.cvtColor(cropped_face_4, cv2.COLOR_BGR2RGB)  # 将BGR转换为RGB 例如(78,78,3)
                resized_face = cv2.resize(cropped_face_5, (output_size, output_size))  # 调整输出图像大小
                out_images.append(resized_face)
                
                # 为裁剪后的图像创建副本用于绘制关键点
                cropped_image_with_points = resized_face.copy()
                # 在裁剪图像上绘制关键点
                cv2.circle(cropped_image_with_points, (int(cropped_left_eye[0]), int(cropped_left_eye[1])), 5, (0, 0, 255), -1)  # 左眼用红色
                cv2.circle(cropped_image_with_points, (int(cropped_right_eye[0]), int(cropped_right_eye[1])), 5, (0, 255, 0), -1)  # 右眼用绿色
                cropped_images_with_points.append(cropped_image_with_points)
            
            # 将当前图像的眼睛关键点添加到列表中
            original_eye_points_list.append(original_eye_points_per_image)
            cropped_eye_points_list.append(cropped_eye_points_per_image)
            
            # 将标记了关键点的原始图像添加到列表中
            original_images_with_points.append(original_image_with_points)

        # 合并所有单独的人脸遮罩
        if individual_masks:
            stacked_masks = torch.cat(individual_masks, dim=0)
        else:
            # 如果没有找到人脸，创建一个空张量
            stacked_masks = torch.zeros((1, height, width), dtype=torch.float32)
        
        # 处理裁剪面部图像
        cropped_face_6 = np.array(out_images).astype(np.float32) / 255.0
        cropped_face_7 = torch.from_numpy(cropped_face_6)
        if cropped_face_7.ndim == 3:
            cropped_face_7 = cropped_face_7.unsqueeze(0)

        # 处理带有关键点的原始图像
        original_images_tensor = torch.from_numpy(np.array(original_images_with_points).astype(np.float32) / 255.0)
        
        # 处理带有关键点的裁剪图像
        if cropped_images_with_points:
            cropped_images_tensor = torch.from_numpy(np.array(cropped_images_with_points).astype(np.float32) / 255.0)
            if cropped_images_tensor.ndim == 3:
                cropped_images_tensor = cropped_images_tensor.unsqueeze(0)
        else:
            # 如果没有找到人脸，创建一个空张量
            cropped_images_tensor = torch.zeros((1, output_size, output_size, 3), dtype=torch.float32)

        # 返回所有需要的数据
        return (cropped_face_7, mask, str(squares_info), stacked_masks, 
                str(original_eye_points_list), str(cropped_eye_points_list),
                original_images_tensor, cropped_images_tensor)  # 返回张量、总mask、正方形信息、各个人脸遮罩和眼睛关键点信息


def insightface_loader(provider="CPU", name='buffalo_l', det_thresh=0.5, size=640):
    try:
        from insightface.app import FaceAnalysis
    except ImportError as e:
        raise Exception(e)
    path = os.path.join(folder_paths.models_dir, "insightface")
    model = FaceAnalysis(name=name, root=path, providers=[provider + 'ExecutionProvider', ])
    model.prepare(ctx_id=0, det_thresh=det_thresh, det_size=(size, size))
    return model


def cv_imread(file_path):
    """
    读取含中文路径的图片并返回cv2图像对象。

    :param file_path: 图片的完整路径，包括中文字符
    :return: cv2图像对象
    """
    # 从文件中读取字节
    img_array = np.fromfile(file_path, dtype=np.uint8)

    # 解码字节为图像
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    # 返回图像
    return img


class CreateBboxMask:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",),
                             "det_thresh": ("FLOAT", {"default": 0.5, "min": 0, "max": 1.0, "step": 0.01}),
                             "device": ([-1, 0], {"default": -1, "tooltip": "-1:cpu, 0:gpu"}),
                             "size": ("INT", {"default": 640, "min": 256, "max": 1024, "step": 2}),
                             "expand_left": ("FLOAT", {"default": 0.5, "min": 0.5, "max": 20.0, "step": 0.05}),
                             "expand_right": ("FLOAT", {"default": 0.5, "min": 0.5, "max": 20.0, "step": 0.05}),
                             "expand_top": ("FLOAT", {"default": 0.5, "min": 0.5, "max": 20.0, "step": 0.05}),
                             "expand_bottom": ("FLOAT", {"default": 0.5, "min": 0.5, "max": 20.0, "step": 0.05}),

                             }}

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "crop_face"

    CATEGORY = "My_node/mask"

    def __init__(self):
        self.face_helper = None

    def crop_face(self, image, det_thresh, device, size, expand_left, expand_right, expand_top, expand_bottom):
        if device < 0:
            provider = "CPU"
        else:
            provider = "CUDA"
        print("provider", provider)
        print("size", size)
        print("det_thresh", det_thresh)
        model = insightface_loader(provider=provider, name='buffalo_l', det_thresh=det_thresh, size=size)
        image_np = image.squeeze(0).detach().cpu().numpy()  # 去掉 batch 维度，并转换为 NumPy 数组

        # 假设 ComfyUI 的张量像素值范围是 [0, 1]，将其转换为 [0, 255]，并将类型转换为 uint8
        image_np = (image_np * 255).astype(np.uint8)

        # 如果通道顺序是 RGB，但 cv2 需要 BGR 格式，则需要转换通道顺序
        image_np_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        # image_np_bgr = cv_imread(r"D:\AI\SD\test\huangguangningnvpengyou.jpg")
        faces = model.get(image_np_bgr)
        width, height = image_np_bgr.shape[1], image_np_bgr.shape[0]
        # new_bbox_list = []
        # 创建一个与图像相同大小的空白蒙版，初始化为黑色
        mask = np.zeros((height, width), dtype=np.uint8)
        # cropped_faces = []  # 新增列表用于存储裁剪后的面部图像

        for i in range(len(faces)):
            bbox = faces[i].bbox
            # 计算矩形框的宽度和高度
            bbox_width = bbox[2] - bbox[0]
            bbox_height = bbox[3] - bbox[1]

            # 计算中心点坐标
            bbox_center_x = (bbox[0] + bbox[2]) / 2
            bbox_center_y = (bbox[1] + bbox[3]) / 2

            # 计算每个方向的增加量
            left_increase = bbox_width * expand_left
            right_increase = bbox_width * expand_right
            top_increase = bbox_height * expand_top
            bottom_increase = bbox_height * expand_bottom

            # 计算新的左上角和右下角坐标
            new_x1 = max(0, bbox_center_x - left_increase)
            new_y1 = max(0, bbox_center_y - top_increase)
            new_x2 = min(width, bbox_center_x + right_increase)
            new_y2 = min(height, bbox_center_y + bottom_increase)

            # 在蒙版上绘制白色矩形 (255 表示白色)
            # cv2.rectangle(mask, (new_x1, new_y1), (new_x2, new_y2), color=(255, 255, 255), thickness=-1)
            mask[int(new_y1):int(new_y2), int(new_x1):int(new_x2)] = 255

            # # 裁剪面部图像并添加到列表中
            # cropped_face = image_np_bgr[int(new_y1):int(new_y2), int(new_x1):int(new_x2)]
            # cropped_faces.append(cropped_face)

        # 将裁剪后的面部图像列表转换为 3D 张量
        # cropped_faces_tensor = torch.tensor(np.array(cropped_faces)).permute(0, 3, 1, 2)  # 转换为 [n, channels, height, width]

        # 可选：将蒙版应用到原图像，生成框内区域的图片
        image_np_bgra = cv2.cvtColor(image_np_bgr, cv2.COLOR_BGR2RGBA)
        # 创建一个 Alpha 通道，Alpha 值为 255 的区域表示不透明，0 表示完全透明
        alpha_channel = np.ones(image_np_bgr.shape[:2], dtype=image_np_bgr.dtype) * 255  # 初始化为全不透明
        # 根据 mask 更新 Alpha 通道，mask为0的地方变成透明（Alpha=0）
        alpha_channel[mask == 0] = 0
        # 将 Alpha 通道添加到 BGRA 图像的最后一维
        image_np_bgra[:, :, 3] = alpha_channel
        # masked_image = cv2.bitwise_and(image_np_bgr, image_np_bgr, mask=mask)
        masked_image_tensor = torch.tensor(image_np_bgra).unsqueeze(0)  # 添加 batch 维度
        masked_image_tensor = masked_image_tensor.float() / 255.0  # 如果你想标准化为 [0, 1] 的范围
        mask_tensor = torch.tensor(mask).unsqueeze(0)  # 变为 [1, height, width, 1]
        # mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        #
        # # 2. 将蒙版和应用蒙版的图像从 NumPy 数组转换为 PyTorch 张量
        # # (height, width, channels) -> (1, height, width, channels)
        # mask_tensor = torch.tensor(mask_3ch).unsqueeze(0)  # 添加 batch 维度
        # 3. 确保数据类型为 float32，如果需要 (比如应用到神经网络)
        mask_tensor = mask_tensor.float() / 255.0  # 如果你想标准化为 [0, 1] 的范围
        # 4. 输出四维张量的形状
        print("mask_tensor", mask_tensor.shape)  # 输出: torch.Size([1, height, width, channels])
        print(masked_image_tensor.shape)  # 输出: torch.Size([1, height, width, channels])

        return (masked_image_tensor, mask_tensor)  # 修改返回值以包含裁剪后的面部图像张量


class PasteFacesMy:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "base_image": ("IMAGE",),  # 第一个输入，与CropFaceMy的image相同
                "face_images": ("IMAGE",),  # 第二个输入，形如(3, 720, 720, 3)
                "squares_info": ("STRING",),  # 第三个输入，CropFaceMy输出的str(squares_info)
            }
        }

    RETURN_TYPES = ("IMAGE",)  # 返回合成后的图像
    FUNCTION = "paste_faces"
    CATEGORY = "My_node/image"

    def paste_faces(self, base_image, face_images, squares_info):
        # 将squares_info从字符串转换为列表
        squares_info = eval(squares_info)  # 使用eval函数

        result_image = base_image.clone()
        cur_index = 0

        for i, face_info in enumerate(squares_info):
            for face in face_info:
                x, y, size = face  # 获取x, y坐标和大小

                face_image = face_images[cur_index]  # 假设face_images是一个包含多个图像的列表

                # 将face_image转换为numpy数组
                face_image_np = (face_image.cpu().numpy() * 255).astype(np.uint8)  # 转换为uint8格式
                face_image_resized = cv2.resize(face_image_np, (size, size))  # 调整人脸图像大小
                # 确保目标区域的大小与face_image_resized匹配
                target_height = min(size, result_image.shape[1] - y)
                target_width = min(size, result_image.shape[2] - x)
                print("size", size)
                print("result_image.shape[1]-y", result_image.shape[1] - y)
                print("result_image.shape[2]-x", result_image.shape[2] - x)
                print("target_height", target_height)
                print("target_width", target_width)
                face_image_resized = face_image_resized[:target_height, :target_width]
                if face_image_resized.shape[2] == 4:
                    print("存在透明通道")
                    # 分离RGB和Alpha通道
                    face_rgb = face_image_resized[:, :, :3]
                    face_alpha = face_image_resized[:, :, 3] / 255.0  # 归一化Alpha通道

                    # 将调整大小的人脸图像转换为PyTorch张量
                    face_rgb_tensor = torch.from_numpy(face_rgb).permute(2, 0, 1).float() / 255.0
                    face_alpha_tensor = torch.from_numpy(face_alpha).unsqueeze(0).float()

                    # 获取当前区域的基图像
                    base_region = result_image[i, y:y + target_height, x:x + target_width].float().permute(2, 0, 1)

                    # 混合图像
                    blended_region = face_rgb_tensor * face_alpha_tensor + base_region * (1 - face_alpha_tensor)
                    blended_region = blended_region.permute(1, 2, 0)
                    print(blended_region.shape)
                    # display_images(blended_region)
                    # 将混合后的图像粘贴回基图像
                    result_image[i, y:y + target_height, x:x + target_width] = blended_region
                else:
                    # 没有透明通道，直接粘贴
                    face_rgb_tensor = torch.from_numpy(face_image_resized).permute(2, 0, 1).float() / 255.0
                    face_rgb_tensor = face_rgb_tensor.permute(1, 2, 0)
                    result_image[i, y:y + target_height, x:x + target_width] = face_rgb_tensor

                cur_index += 1

        return (result_image,)  # 返回合成后的图像


class GenerateWhiteTensor:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_str": ("STRING",),
                "size": ("INT", {"default": 512, "min": 256, "max": 2048, "step": 32}),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "generate_tensor"
    CATEGORY = "My_node/tensor"

    def generate_tensor(self, input_str, size):
        # 将输入字符串转换为列表
        input_list = eval(input_str)  # 注意：eval可能存在安全隐患，确保输入是可信的

        # 获取最内层列表的数量
        n = sum(len(inner_list) for inner_list in input_list)

        # 生成一个纯黑的张量
        black_tensor = torch.ones((n, size, size), dtype=torch.float32)

        return (black_tensor,)


def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))


def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def tensor2rgba(t: torch.Tensor) -> torch.Tensor:
    size = t.size()
    if (len(size) < 4):
        return t.unsqueeze(3).repeat(1, 1, 1, 4)
    elif size[3] == 1:
        return t.repeat(1, 1, 1, 4)
    elif size[3] == 3:
        alpha_tensor = torch.ones((size[0], size[1], size[2], 1))
        return torch.cat((t, alpha_tensor), dim=3)
    else:
        return t


class MyLoadImageListPlus:

    @classmethod
    def INPUT_TYPES(s):
        scale_to_list = ['longest', 'None', 'shortest', 'width', 'height', 'total_pixel(kilo pixel)']
        return {"required": {"input_folder": ("STRING", {"default": "", }),
                             "start_index": ("INT", {"default": 0, "min": 0, "max": 99999}),
                             "max_images": ("INT", {"default": 1, "min": 1, "max": 99999}),
                             "scale_to_side": (scale_to_list,),  # 是否按长边缩放
                             "scale_to_length": ("INT", {"default": 1024, "min": 4, "max": 999999, "step": 1}),
                             },
                #    "optional": {"input_path": ("STRING", {"default": '', "multiline": False}),
                #    }
                }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "STRING", "STRING", "INT", "INT", "INT")
    RETURN_NAMES = ("IMAGE", "MASK", "index", "filename", "filename_prefix", "width", "height", "list_length",)
    OUTPUT_IS_LIST = (True, True, True, True, True, True, True, False)
    FUNCTION = "make_list"
    CATEGORY = "My_node/image"

    def make_list(self, start_index, max_images, input_folder, scale_to_side, scale_to_length):
        if not input_folder:
            return None

        in_path = input_folder

        file_list = [f for f in os.listdir(in_path) if
                     f.lower().endswith(('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'))]

        image_list = []
        mask_list = []
        index_list = []
        filename_list = []
        width_list = []
        height_list = []
        prefix_list = []
        exif_list = []

        # Ensure start_index is within the bounds of the list
        start_index = max(0, min(start_index, len(file_list) - 1))

        # Calculate the end index based on max_rows
        end_index = min(start_index + max_images, len(file_list) - 1)

        for num in range(start_index, end_index):
            filename = file_list[num]
            print("filename", filename)
            # 文件名前缀
            prefix = os.path.splitext(filename)[0]
            img_path = os.path.join(in_path, filename)
            print("img_path", img_path)
            img = Image.open(img_path)

            width, height = img.size
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
            img = img.resize((target_width, target_height), Image.LANCZOS)
            width_list.append(target_width)
            height_list.append(target_height)
            prefix_list.append(prefix)

            image_list.append(pil2tensor(img.convert("RGB")))

            tensor_img = pil2tensor(img)
            mask_list.append(tensor2rgba(tensor_img)[:, :, :, 0])

            # Populate the image index
            index_list.append(num)

            # Populate the filename_list
            filename_list.append(filename)

        if not image_list:
            # Handle the case where the list is empty
            print(" No images found.")
            return None
        list_length = end_index - start_index

        return (image_list, mask_list, index_list, filename_list, prefix_list, width_list, height_list, list_length,)


class CropFaceMyDetailed:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",),
                             "det_thresh": ("FLOAT", {"default": 0.5, "min": 0, "max": 1.0, "step": 0.01}),
                             "scale_factor": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 5.0, "step": 0.01}),
                             "device": ([-1, 0], {"default": -1, "tooltip": "-1:cpu, 0:gpu"}),
                             "det_size": ("INT", {"default": 512, "min": 256, "max": 1024, "step": 2}),
                             "output_size": ("INT", {"default": 512, "min": 256, "max": 1024, "step": 2}),
                             }}

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "MASK", "STRING", "STRING", "IMAGE", "IMAGE")
    RETURN_NAMES = ("image", "mask", "squares_info", "individual_masks", "original_eye_points", "cropped_eye_points", "original_image_with_points", "cropped_image_with_points")

    FUNCTION = "crop_face"

    CATEGORY = "My_node/image"

    def __init__(self):
        self.face_helper = None

    def crop_face(self, image, det_thresh, scale_factor, device, det_size, output_size):

        if device < 0:
            provider = "CPU"
        else:
            provider = "CUDA"
        print("provider", provider)
        print("det_size", det_size)
        print("det_thresh", det_thresh)
        device = model_management.get_torch_device()
        model = insightface_loader(provider=provider, name='buffalo_l', det_thresh=det_thresh, size=det_size)
        image_np = 255. * image.cpu().numpy()

        total_images = image_np.shape[0]
        out_images = []

        # 获取输入样本的宽高
        height, width = image.shape[1:3]  # 形状为 (2, 960, 720, 3)，获取宽高

        # 创建一个与样本数量相同的纯黑张量
        mask = torch.zeros((total_images, height, width), dtype=torch.float32)
        squares_info = []  # 用于记录正方形的位置信息
        
        # 创建存储单个人脸遮罩的列表
        individual_masks = []
        
        # 创建用于存储眼睛关键点坐标的列表
        original_eye_points_list = []  # 原始图像上的眼睛关键点
        cropped_eye_points_list = []   # 裁剪图像上的眼睛关键点
        
        # 创建用于绘制关键点的图像副本
        original_images_with_points = []
        cropped_images_with_points = []
        
        image_np_new = image_np.copy()
        for i in range(total_images):
            # 为每个图像创建一个空的眼睛关键点列表
            original_eye_points_per_image = []
            cropped_eye_points_per_image = []
            
            # 为当前图像创建副本用于绘制关键点
            original_image_with_points = image_np_new[i].copy()
            
            while len(squares_info) <= i:  # 确保列表长度足够
                squares_info.append([])  # 添加空列表

            cur_image_np = image_np_new[i, :, :, ::-1]  # 将RGB转换为BGR

            # 检查图像是否为空
            if cur_image_np is None or cur_image_np.size == 0:
                print(f"Image at index {i} is empty or not loaded correctly.")
                continue

            faces = model.get(cur_image_np)
            faces_found = len(faces)
            for j in range(faces_found):
                bbox = faces[j].bbox
                x1, y1, x2, y2 = bbox[:4]  # 获取面部框的坐标
                
                # 获取关键点信息
                kps = faces[j].kps  # 获取关键点坐标
                # insightface的关键点格式为[左眼，右眼，鼻尖，左嘴角，右嘴角]
                left_eye = kps[0]  # 左眼中心坐标
                right_eye = kps[1]  # 右眼中心坐标
                
                # 存储原始图像中的眼睛关键点
                original_eye_point = {
                    "left_eye": left_eye.tolist(),
                    "right_eye": right_eye.tolist()
                }
                original_eye_points_per_image.append(original_eye_point)
                
                # 在原始图像上绘制这些点
                cv2.circle(original_image_with_points, (int(left_eye[0]), int(left_eye[1])), 5, (0, 0, 255), -1)  # 左眼用红色
                cv2.circle(original_image_with_points, (int(right_eye[0]), int(right_eye[1])), 5, (0, 255, 0), -1)  # 右眼用绿色

                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                # 根据scale_factor调整面部框的大小,
                # 如果x1,y1超出边界则取最小值0,0,
                # 如果x2,y2超出边界则取最大值width,height

                x1 = max(0, int(center_x - (x2 - x1) * scale_factor / 2))
                y1 = max(0, int(center_y - (y2 - y1) * scale_factor / 2))
                x2 = min(width, int(center_x + (x2 - x1) * scale_factor / 2))
                y2 = min(height, int(center_y + (y2 - y1) * scale_factor / 2))
                side_length = min(x2 - x1, y2 - y1)  # 计算正方形的边长

                # 计算正方形的边长和中心点
                square_size = int(side_length)
                if int(center_x - square_size // 2) <= 0:
                    square_x = 0
                elif int(center_x - square_size // 2) >= width:
                    square_x = width - 1
                else:
                    square_x = int(center_x - square_size // 2)
                if int(center_y - square_size // 2) <= 0:
                    square_y = 0
                elif int(center_y - square_size // 2) >= height:
                    square_y = height - 1
                else:
                    square_y = int(center_y - square_size // 2)
                    # square_size = min(int(side_length))
                square_size = min(square_size, width - square_x, height - square_y)

                # 将正方形区域填充为白色
                mask[i, square_y:int(square_y + square_size), square_x:int(square_x + square_size)] = 1.0
                
                # 创建单独的人脸遮罩
                individual_mask = torch.zeros((total_images, height, width), dtype=torch.float32)
                individual_mask[i, square_y:int(square_y + square_size), square_x:int(square_x + square_size)] = 1.0
                individual_masks.append(individual_mask)

                # 记录正方形的信息
                squares_info[i].append([square_x, square_y, square_size])
                # 裁剪正方形面部
                cropped_face_1 = cur_image_np[square_y:int(square_y + square_size),
                                 square_x:int(square_x + square_size)]

                # 检查裁剪的面部图像是否为空
                if cropped_face_1 is None or cropped_face_1.size == 0:
                    print(f"Cropped face at index {i}, face {j} is empty.")
                    continue
                
                # 计算裁剪图像中眼睛关键点的新坐标
                # 从原始图像坐标转换为裁剪图像坐标
                def transform_point(point, square_x, square_y, output_size, square_size):
                    # 首先减去裁剪起点坐标，获得在裁剪图像中的相对坐标
                    relative_x = point[0] - square_x
                    relative_y = point[1] - square_y
                    # 然后根据输出尺寸进行缩放
                    scaled_x = relative_x * (output_size / square_size)
                    scaled_y = relative_y * (output_size / square_size)
                    return [scaled_x, scaled_y]
                
                # 转换眼睛中心点到裁剪图像坐标系
                cropped_left_eye = transform_point(left_eye, square_x, square_y, output_size, square_size)
                cropped_right_eye = transform_point(right_eye, square_x, square_y, output_size, square_size)
                
                # 存储裁剪图像中的眼睛关键点
                cropped_eye_point = {
                    "left_eye": cropped_left_eye,
                    "right_eye": cropped_right_eye
                }
                cropped_eye_points_per_image.append(cropped_eye_point)

                cropped_face_2 = img2tensor(cropped_face_1 / 255., bgr2rgb=True, float32=True)
                normalize(cropped_face_2, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
                cropped_face_3 = cropped_face_2.unsqueeze(0).to(device)
                cropped_face_4 = tensor2img(cropped_face_3, rgb2bgr=True, min_max=(-1, 1)).astype('uint8')
                cropped_face_5 = cv2.cvtColor(cropped_face_4, cv2.COLOR_BGR2RGB)  # 将BGR转换为RGB 例如(78,78,3)
                resized_face = cv2.resize(cropped_face_5, (output_size, output_size))  # 调整输出图像大小
                out_images.append(resized_face)
                
                # 为裁剪后的图像创建副本用于绘制关键点
                cropped_image_with_points = resized_face.copy()
                # 在裁剪图像上绘制关键点
                cv2.circle(cropped_image_with_points, (int(cropped_left_eye[0]), int(cropped_left_eye[1])), 5, (0, 0, 255), -1)  # 左眼用红色
                cv2.circle(cropped_image_with_points, (int(cropped_right_eye[0]), int(cropped_right_eye[1])), 5, (0, 255, 0), -1)  # 右眼用绿色
                cropped_images_with_points.append(cropped_image_with_points)
            
            # 将当前图像的眼睛关键点添加到列表中
            original_eye_points_list.append(original_eye_points_per_image)
            cropped_eye_points_list.append(cropped_eye_points_per_image)
            
            # 将标记了关键点的原始图像添加到列表中
            original_images_with_points.append(original_image_with_points)

        # 合并所有单独的人脸遮罩
        if individual_masks:
            stacked_masks = torch.cat(individual_masks, dim=0)
        else:
            # 如果没有找到人脸，创建一个空张量
            stacked_masks = torch.zeros((1, height, width), dtype=torch.float32)
        
        # 处理裁剪面部图像
        cropped_face_6 = np.array(out_images).astype(np.float32) / 255.0
        cropped_face_7 = torch.from_numpy(cropped_face_6)
        if cropped_face_7.ndim == 3:
            cropped_face_7 = cropped_face_7.unsqueeze(0)

        # 处理带有关键点的原始图像
        original_images_tensor = torch.from_numpy(np.array(original_images_with_points).astype(np.float32) / 255.0)
        
        # 处理带有关键点的裁剪图像
        if cropped_images_with_points:
            cropped_images_tensor = torch.from_numpy(np.array(cropped_images_with_points).astype(np.float32) / 255.0)
            if cropped_images_tensor.ndim == 3:
                cropped_images_tensor = cropped_images_tensor.unsqueeze(0)
        else:
            # 如果没有找到人脸，创建一个空张量
            cropped_images_tensor = torch.zeros((1, output_size, output_size, 3), dtype=torch.float32)

        # 返回所有需要的数据
        return (cropped_face_7, mask, str(squares_info), stacked_masks, 
                str(original_eye_points_list), str(cropped_eye_points_list),
                original_images_tensor, cropped_images_tensor)  # 返回张量、总mask、正方形信息、各个人脸遮罩和眼睛关键点信息


class PasteFacesAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "background_image": ("IMAGE",),  # 包含人脸的背景图像
                "face_image": ("IMAGE",),  # 可能包含透明通道的人脸图像
                "original_eye_points": ("STRING",),  # 背景图像中人脸的眼睛关键点
                "cropped_eye_points": ("STRING",),  # 人脸图像中的眼睛关键点
                "background_face_indices": ("STRING", {"default": "-1"}),  # 要替换的背景人脸索引，-1表示所有人脸
                "paste_face_indices": ("STRING", {"default": "0"}),  # 用于粘贴的人脸索引
                "enable_rotation": ("BOOLEAN", {"default": True}),  # 是否根据眼睛坐标旋转人脸
                "blend_alpha": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}),  # 混合透明度
                "scale_adjustment": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.01}),  # 缩放调整系数
                "debug_mode": ("BOOLEAN", {"default": True}),  # 是否打印详细调试信息
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("result_image",)
    FUNCTION = "paste_faces"
    CATEGORY = "My_node/image"

    def paste_faces(self, background_image, face_image, original_eye_points, cropped_eye_points, 
                   background_face_indices, paste_face_indices, enable_rotation, blend_alpha, scale_adjustment, debug_mode):
        """
        高级人脸粘贴函数，根据眼睛关键点对齐和融合人脸
        
        Args:
            background_image: 背景图像，可能包含多个人脸
            face_image: 替换用的人脸图像，可能包含透明通道
            original_eye_points: 背景图像中人脸的眼睛坐标 (字符串格式)
            cropped_eye_points: 人脸图像中的眼睛坐标 (字符串格式)
            background_face_indices: 要替换的背景人脸索引，-1表示所有人脸
            paste_face_indices: 用于粘贴的人脸索引
            enable_rotation: 是否根据眼睛坐标旋转人脸
            blend_alpha: 混合透明度
            scale_adjustment: 缩放调整系数
            debug_mode: 是否打印详细调试信息
        """
        import cv2
        import numpy as np
        import math
        import copy
        
        # 调试输出函数
        def debug_print(msg):
            if debug_mode:
                print(f"[PasteFacesAdvanced] {msg}")
        
        # 输出输入参数的形状和类型
        debug_print(f"背景图像形状: {background_image.shape}, 类型: {background_image.dtype}")
        debug_print(f"人脸图像形状: {face_image.shape}, 类型: {face_image.dtype}")
        
        # 转换眼睛坐标字符串为Python对象
        original_eyes = eval(original_eye_points)
        cropped_eyes = eval(cropped_eye_points)
        
        debug_print(f"原始眼睛坐标: {original_eyes}")
        debug_print(f"裁剪眼睛坐标: {cropped_eyes}")
        
        # 转换背景和人脸图像为numpy数组 (复制以避免修改原始数据)
        background_np = background_image.cpu().numpy().copy()
        face_np = face_image.cpu().numpy().copy()
        
        # 解析背景人脸索引
        if background_face_indices == "-1":
            # 处理所有背景人脸
            bg_indices = []
            for i in range(len(original_eyes)):
                for j in range(len(original_eyes[i])):
                    bg_indices.append((i, j))
        else:
            # 处理指定的背景人脸索引
            bg_indices_list = background_face_indices.split(',')
            bg_indices = []
            for idx in bg_indices_list:
                idx = idx.strip()
                if idx and idx.isdigit():
                    bg_indices.append((0, int(idx)))  # 假设图像索引为0，只使用人脸索引
        
        # 解析粘贴人脸索引
        if paste_face_indices == "0":
            # 默认使用第一个人脸
            paste_indices = [0] * len(bg_indices)
        else:
            paste_indices_list = paste_face_indices.split(',')
            paste_indices = []
            for idx in paste_indices_list:
                idx = idx.strip()
                if idx and idx.isdigit():
                    paste_indices.append(int(idx))
            
            # 如果粘贴人脸索引少于背景人脸索引，则重复使用最后一个索引
            if len(paste_indices) < len(bg_indices):
                last_idx = paste_indices[-1] if paste_indices else 0
                paste_indices.extend([last_idx] * (len(bg_indices) - len(paste_indices)))
        
        debug_print(f"背景替换人脸图像索引: {bg_indices}")
        debug_print(f"粘贴替换人脸图像索引: {paste_indices}")
        
        # 创建结果图像的副本
        result_np = copy.deepcopy(background_np)
        
        # 为每个背景人脸应用替换
        for i, ((bg_img_idx, bg_face_idx), paste_idx) in enumerate(zip(bg_indices, paste_indices)):
            try:
                # 检查索引是否有效
                if (bg_img_idx >= len(original_eyes) or 
                    bg_face_idx >= len(original_eyes[bg_img_idx])):
                    debug_print(f"无效的背景索引: 背景图像索引 {bg_img_idx}, 背景人脸索引 {bg_face_idx}")
                    continue
                
                if paste_idx >= len(face_np):
                    debug_print(f"无效的粘贴索引: 粘贴人脸索引 {paste_idx}, 可用人脸数量 {len(face_np)}")
                    continue
                
                if len(cropped_eyes) == 0 or len(cropped_eyes[0]) == 0:
                    debug_print(f"裁剪眼睛坐标为空")
                    continue
                
                if paste_idx >= len(cropped_eyes[0]):
                    debug_print(f"无效的裁剪眼睛索引: 粘贴人脸索引 {paste_idx}, 可用眼睛坐标数量 {len(cropped_eyes[0])}")
                    continue
                
                # 获取背景图像中的眼睛坐标
                bg_left_eye = np.array(original_eyes[bg_img_idx][bg_face_idx]["left_eye"])
                bg_right_eye = np.array(original_eyes[bg_img_idx][bg_face_idx]["right_eye"])
                
                # 获取粘贴图像中的眼睛坐标
                paste_left_eye = np.array(cropped_eyes[0][paste_idx]["left_eye"])
                paste_right_eye = np.array(cropped_eyes[0][paste_idx]["right_eye"])
                
                debug_print(f"背景左眼坐标: {bg_left_eye}, 右眼坐标: {bg_right_eye}")
                debug_print(f"粘贴左眼坐标: {paste_left_eye}, 右眼坐标: {paste_right_eye}")
                
                # 计算眼睛中心点
                bg_eye_center = (bg_left_eye + bg_right_eye) / 2
                paste_eye_center = (paste_left_eye + paste_right_eye) / 2
                
                debug_print(f"背景眼睛中心点: {bg_eye_center}")
                debug_print(f"粘贴眼睛中心点: {paste_eye_center}")
                
                # 计算眼睛间距
                bg_eye_distance = np.linalg.norm(bg_right_eye - bg_left_eye)
                paste_eye_distance = np.linalg.norm(paste_right_eye - paste_left_eye)
                
                if paste_eye_distance == 0:
                    debug_print(f"粘贴图像眼睛间距为0，跳过")
                    continue
                
                # 计算缩放比例
                scale = (bg_eye_distance / paste_eye_distance) * scale_adjustment
                debug_print(f"缩放比例: {scale} (背景眼距: {bg_eye_distance}, 粘贴眼距: {paste_eye_distance})")
                
                # 计算旋转角度（如果启用）
                angle = 0
                if enable_rotation:
                    bg_eye_angle = math.atan2(bg_right_eye[1] - bg_left_eye[1], 
                                            bg_right_eye[0] - bg_left_eye[0])
                    paste_eye_angle = math.atan2(paste_right_eye[1] - paste_left_eye[1], 
                                                paste_right_eye[0] - paste_left_eye[0])
                    angle = (bg_eye_angle - paste_eye_angle) * 180 / math.pi
                    debug_print(f"旋转角度: {angle} 度")
                
                # 获取粘贴的人脸图像
                paste_face = face_np[paste_idx].copy()
                h, w = paste_face.shape[0:2]
                debug_print(f"粘贴人脸形状: {paste_face.shape}")
                
                # 获取背景图像尺寸
                bg_h, bg_w = background_np[bg_img_idx].shape[0:2]
                debug_print(f"背景图像尺寸: {bg_w} x {bg_h}")
                
                # 使用更直接的方法计算变换矩阵，确保眼睛位置完全对齐
                # 用于直接计算眼睛对齐的三点仿射变换
                src_points = np.array([
                    paste_left_eye,  # 左眼
                    paste_right_eye,  # 右眼
                    paste_eye_center + np.array([0, 30])  # 中点下方的点，提供垂直方向的参考
                ], dtype=np.float32)
                
                dst_points = np.array([
                    bg_left_eye,  # 目标左眼
                    bg_right_eye,  # 目标右眼
                    bg_eye_center + np.array([0, 30 * scale])  # 中点下方的点，考虑缩放
                ], dtype=np.float32)
                
                # 计算仿射变换矩阵
                M_affine = cv2.getAffineTransform(src_points, dst_points)
                debug_print(f"仿射变换矩阵: \n{M_affine}")
                
                # 确保图像数据类型正确(uint8)以便OpenCV处理
                paste_face_uint8 = (paste_face * 255).astype(np.uint8) if paste_face.dtype == np.float32 or paste_face.dtype == np.float64 else paste_face
                
                # 应用仿射变换
                warped_face = cv2.warpAffine(paste_face_uint8, M_affine, (bg_w, bg_h), 
                                            borderMode=cv2.BORDER_CONSTANT, 
                                            borderValue=(0, 0, 0, 0))
                
                debug_print(f"变换后人脸形状: {warped_face.shape}")
                
                # 创建一个遮罩用于混合
                mask = np.zeros((bg_h, bg_w), dtype=np.float32)
                
                # 如果人脸图像有透明通道，使用它作为遮罩
                if paste_face.shape[2] == 4:
                    debug_print(f"使用透明通道作为遮罩")
                    alpha_channel = paste_face_uint8[:, :, 3]
                    alpha_warped = cv2.warpAffine(alpha_channel, M_affine, (bg_w, bg_h))
                    mask = alpha_warped.astype(np.float32) / 255.0 * blend_alpha
                else:
                    debug_print(f"创建基于颜色的简单遮罩")
                    # 创建一个基于颜色的简单遮罩
                    # 先转换为灰度图
                    if warped_face.shape[2] >= 3:  # 确保有足够的通道
                        gray_face = cv2.cvtColor(warped_face[:, :, :3], cv2.COLOR_RGB2GRAY)
                        _, mask_binary = cv2.threshold(gray_face, 1, 255, cv2.THRESH_BINARY)
                        mask = mask_binary.astype(np.float32) / 255.0 * blend_alpha
                
                # 应用遮罩进行混合
                if len(warped_face.shape) == 3 and warped_face.shape[2] >= 3:
                    # 验证变换后的眼睛位置
                    if debug_mode:
                        # 创建一个测试图像用于验证
                        test_img = result_np[bg_img_idx].copy() * 255
                        # 在背景图像上标记目标眼睛位置
                        cv2.circle(test_img, (int(bg_left_eye[0]), int(bg_left_eye[1])), 5, (255, 0, 0), -1)
                        cv2.circle(test_img, (int(bg_right_eye[0]), int(bg_right_eye[1])), 5, (0, 255, 0), -1)
                        cv2.circle(test_img, (int(bg_eye_center[0]), int(bg_eye_center[1])), 5, (0, 0, 255), -1)
                        debug_print(f"标记了背景眼睛位置的测试图像已创建")
                    
                    for c in range(3):  # 仅处理RGB通道
                        if c < result_np[bg_img_idx].shape[2]:  # 确保结果图像有足够的通道
                            # 将mask扩展为3D以便广播
                            mask_3d = mask[:, :, np.newaxis] if len(mask.shape) == 2 else mask
                            # 应用混合公式: result = background * (1 - mask) + foreground * mask
                            result_np[bg_img_idx, :, :, c] = (
                                result_np[bg_img_idx, :, :, c] * (1 - mask_3d[:, :, 0]) + 
                                warped_face[:, :, c].astype(np.float32) / 255.0 * mask_3d[:, :, 0]
                            )
                else:
                    debug_print(f"警告: 变换后的人脸形状不正确: {warped_face.shape}")
            
            except Exception as e:
                debug_print(f"处理人脸 {i} 时出错: {str(e)}")
                import traceback
                debug_print(traceback.format_exc())
        
        # 转换回PyTorch张量前确保数据类型和值范围正确
        # 将值范围限制在[0, 1]
        result_np = np.clip(result_np, 0.0, 1.0).astype(np.float32)
        # 转换回PyTorch张量
        result_tensor = torch.from_numpy(result_np)
        
        return (result_tensor,)
