import numpy as np
import time
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import io
import base64
import random
import math
import os
import re
import json
from PIL.PngImagePlugin import PngInfo

try:
    import cv2
except:
    print("OpenCV not installed")
    pass
from PIL import ImageGrab, ImageDraw, ImageFont, Image, ImageSequence, ImageOps

from nodes import MAX_RESOLUTION, SaveImage
from comfy_extras.nodes_mask import ImageCompositeMasked
from comfy.cli_args import args
from comfy.utils import ProgressBar, common_upscale
import folder_paths

# import model_management

script_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import torch  # Make sure you have PyTorch installed

import matplotlib.pyplot as plt


def load_images(folder, image_load_cap, start_index):
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Folder '{folder} cannot be found.'")
    dir_files = os.listdir(folder)
    if len(dir_files) == 0:
        raise FileNotFoundError(f"No files in directory '{folder}'.")

    # Filter files by extension
    valid_extensions = ['.jpg', '.jpeg', '.png', '.webp']
    dir_files = [f for f in dir_files if any(f.lower().endswith(ext) for ext in valid_extensions)]

    dir_files = sorted(dir_files)
    dir_files = [os.path.join(folder, x) for x in dir_files]

    # start at start_index
    dir_files = dir_files[start_index:]

    images = []
    masks = []
    image_path_list = []

    limit_images = False
    if image_load_cap > 0:
        limit_images = True
    image_count = 0

    has_non_empty_mask = False

    for image_path in dir_files:
        if os.path.isdir(image_path) and os.path.ex:
            continue
        if limit_images and image_count >= image_load_cap:
            break
        i = Image.open(image_path)
        i = ImageOps.exif_transpose(i)
        image = i.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        if 'A' in i.getbands():
            mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
            mask = 1. - torch.from_numpy(mask)
            has_non_empty_mask = True
        else:
            mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
        images.append(image)
        masks.append(mask)
        image_path_list.append(image_path)
        image_count += 1

    if len(images) == 1:
        return (images[0], masks[0], 1)

    elif len(images) > 1:
        image1 = images[0]
        mask1 = None

        for image2 in images[1:]:
            if image1.shape[1:] != image2.shape[1:]:
                image2 = common_upscale(image2.movedim(-1, 1), image1.shape[2], image1.shape[1], "bilinear",
                                        "center").movedim(1, -1)
            image1 = torch.cat((image1, image2), dim=0)

        for mask2 in masks[1:]:
            if has_non_empty_mask:
                if image1.shape[1:3] != mask2.shape:
                    mask2 = torch.nn.functional.interpolate(mask2.unsqueeze(0).unsqueeze(0),
                                                            size=(image1.shape[2], image1.shape[1]),
                                                            mode='bilinear', align_corners=False)
                    mask2 = mask2.squeeze(0)
                else:
                    mask2 = mask2.unsqueeze(0)
            else:
                mask2 = mask2.unsqueeze(0)

            if mask1 is None:
                mask1 = mask2
            else:
                mask1 = torch.cat((mask1, mask2), dim=0)

        return (image1, mask1, len(images), image_path_list)


def load_images_with_alpha(folder, image_load_cap, start_index):
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Folder '{folder} cannot be found.'")
    dir_files = os.listdir(folder)
    if len(dir_files) == 0:
        raise FileNotFoundError(f"No files in directory '{folder}'.")

    # Filter files by extension
    valid_extensions = ['.jpg', '.jpeg', '.png', '.webp']
    dir_files = [f for f in dir_files if any(f.lower().endswith(ext) for ext in valid_extensions)]

    dir_files = sorted(dir_files)
    dir_files = [os.path.join(folder, x) for x in dir_files]

    # start at start_index
    dir_files = dir_files[start_index:]

    images = []
    masks = []

    limit_images = False
    if image_load_cap > 0:
        limit_images = True
    image_count = 0

    for image_path in dir_files:
        if os.path.isdir(image_path):
            continue
        if limit_images and image_count >= image_load_cap:
            break
        i = Image.open(image_path)
        i = ImageOps.exif_transpose(i)
        image = i.convert("RGBA")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        images.append(image)

        # Extract alpha channel or create a black mask if no alpha channel
        alpha_channel = image[:, :, :, 3] if image.shape[3] == 4 else torch.zeros((1, image.shape[1], image.shape[2]),
                                                                                  dtype=torch.float32)
        masks.append(alpha_channel)

        image_count += 1

    if len(images) == 0:
        raise ValueError("No valid images found.")

    # Concatenate images along the batch dimension
    image_tensor = torch.cat(images, dim=0)
    mask_tensor = torch.cat(masks, dim=0)

    return image_tensor, mask_tensor


def display_images(image_tensor):
    a = """
    输入格式:
    输入rgb图片: torch.Size([图片张数, 高, 宽, 通道])
    输入rgba图片: torch.Size([4, 512, 512, 4])
    输入mask图片: torch.Size([4, 512, 512])
    """
    print(a)
    num_images = image_tensor.shape[0]

    # 动态调整每行显示的图片数量
    if num_images < 3:
        cols = num_images
        row_w = 5 * num_images
    else:
        cols = 4
        row_w = 5 * cols

    rows = math.ceil(num_images / cols)  # 计算行数
    fig, axes = plt.subplots(rows, cols, figsize=(row_w, 5 * rows))

    # 处理单个子图的情况
    if num_images == 1:
        axes = [axes]
    else:
        axes = axes.flatten()  # 将axes展平以便于迭代

    for i in range(num_images):
        image = image_tensor[i].numpy()

        # 根据图像的维度选择显示方式
        print("image.ndim", image.ndim)
        if image.ndim == 3 and image.shape[2] == 4:
            # RGBA 图像
            axes[i].imshow(image)
        elif image.ndim == 3 and image.shape[2] == 3:
            # RGB 图像
            axes[i].imshow(image)
        elif image.ndim == 2:
            # 遮罩图像
            axes[i].imshow(image, cmap='gray')
        else:
            raise ValueError("不支持的图像格式")

        axes[i].axis('off')  # 不显示坐标轴

    # 隐藏多余的子图
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()


def paste_faces(base_image, face_images, squares_info):
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

            if face_image_resized.shape[2] == 4:
                print("存在透明通道")
                # 分离RGB和Alpha通道
                face_rgb = face_image_resized[:, :, :3]
                face_alpha = face_image_resized[:, :, 3] / 255.0  # 归一化Alpha通道

                # 将调整大小的人脸图像转换为PyTorch张量
                face_rgb_tensor = torch.from_numpy(face_rgb).permute(2, 0, 1).float() / 255.0
                face_alpha_tensor = torch.from_numpy(face_alpha).unsqueeze(0).float()

                # 获取当前区域的基图像
                base_region = result_image[i, y:y + size, x:x + size].float().permute(2, 0, 1)

                # 混合图像
                blended_region = face_rgb_tensor * face_alpha_tensor + base_region * (1 - face_alpha_tensor)
                blended_region = blended_region.permute(1, 2, 0)
                print(blended_region.shape)
                # display_images(blended_region)
                # 将混合后的图像粘贴回基图像
                result_image[i, y:y + size, x:x + size] = blended_region
            else:
                # 没有透明通道，直接粘贴
                face_rgb_tensor = torch.from_numpy(face_image_resized).permute(2, 0, 1).float() / 255.0
                face_rgb_tensor = face_rgb_tensor.permute(1, 2, 0)
                result_image[i, y:y + size, x:x + size] = face_rgb_tensor

            cur_index += 1

    return result_image  # 返回合成后的图像


# PIL to Mask
def pil2mask(image):
    image_np = np.array(image.convert("L")).astype(np.float32) / 255.0
    mask = torch.from_numpy(image_np)
    return 1.0 - mask


def fill_region(masks):
    """
    将输入的 n, h, w 的 n 个 mask 里白色区域内部的黑色填充成白色。

    参数:
    masks (torch.Tensor): 形状为 (n, h, w) 的二值掩码张量，其中 0 表示黑色，255 表示白色。

    返回:
    filled_masks (torch.Tensor): 填充后的掩码张量，形状为 (n, h, w)。
    """
    n, h, w = masks.shape
    filled_masks = []

    for i in range(n):
        mask = masks[i].cpu().numpy().astype(np.uint8)

        # 确保掩码是二值的
        _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        # 使用 OpenCV 的 floodFill 函数填充内部黑色区域
        im_floodfill = binary_mask.copy()
        h, w = binary_mask.shape[:2]
        mask_floodfill = np.zeros((h + 2, w + 2), np.uint8)

        # 从图像的边界开始填充
        for y in range(h):
            cv2.floodFill(im_floodfill, mask_floodfill, (0, y), 255)
            cv2.floodFill(im_floodfill, mask_floodfill, (w - 1, y), 255)
        for x in range(w):
            cv2.floodFill(im_floodfill, mask_floodfill, (x, 0), 255)
            cv2.floodFill(im_floodfill, mask_floodfill, (x, h - 1), 255)

        # 将填充后的区域与原掩码进行逻辑或操作
        filled_mask = binary_mask | cv2.bitwise_not(im_floodfill)

        # 将填充后的掩码转换回 PyTorch 张量
        filled_mask_tensor = torch.from_numpy(filled_mask).unsqueeze(0)
        filled_masks.append(filled_mask_tensor)

    # 将所有填充后的掩码堆叠成一个张量
    filled_masks = torch.cat(filled_masks, dim=0)

    return filled_masks


import numpy as np
from scipy import ndimage


def process_mask(mask_tensor):
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


if __name__ == "__main__":
    # folder_rgb = r"D:\AI\SD\input\图片\rgb_3_1"
    folder_rgba = r"D:\AI\SD\input\图片\rgb_3_2"
    # folder_alpha = r"D:\AI\SD\input\图片\rgba_4"
    image_load_cap = 0
    start_index = 0
    # # 如果load_images输出四个参数
    # try:
    #     # 尝试解包4个返回值（多张图片的情况）
    #     image1, mask1, len_images, image_path_list = load_images(folder_rgb, image_load_cap, start_index)
    #     print("多张图片模式")
    # except ValueError:
    #     # 如果失败，说明是3个返回值（单张图片的情况）
    #     image1, mask1, len_images = load_images(folder_rgb, image_load_cap, start_index)
    #     image_path_list = None
    # print("单张图片模式")
    try:
        image1_1, mask1_1, len_images_1, image_path_list_1 = load_images(folder_rgba, image_load_cap, start_index)
        # print("多张图片模式")
    except ValueError:
        image1_1, mask1_1, len_images_1 = load_images(folder_rgba, image_load_cap, start_index)
    #
    # print("输入rgb图片:", image1.shape)
    # image2, mask2 = load_images_with_alpha(folder_alpha, image_load_cap, start_index)
    # print("输入rgba图片:", image2.shape)
    # # image_3 = paste_faces(image1, image1_1, "[[[388, 0, 304], [0, 47, 383], [287, 311, 356], [589, 179, 351]]]")
    # # display_images(image_3)
    # # display_images(image1)
    # print("mask1_1.shape", mask1_1.shape)
    # print("mask1_1.dim", mask1_1.dim)
    # display_images(image2)
    # display_images(mask2)

    # 填充白色遮罩内部的黑色
    # 生成一些示例数据
    # masks = torch.randint(0, 2, (4, 512, 512)).mul(255).to(torch.uint8)
    #
    # # 填充掩码
    # filled_masks = process_mask(mask2)

    # # 打印原始掩码和填充后掩码的形状
    # print(f"Original masks shape: {mask2.shape}")
    # print(f"Filled masks shape: {filled_masks.shape}")
    # # 合并成8,512,512
    # display_images(torch.cat([mask2, filled_masks], dim=0))
    # display_images(image1_1)
    from face_without_glasses import infer_and_get_mask


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


    model_path = r"D:\AI\comfyui\ComfyUI-aki-v1.3\custom_nodes\A_my_nodes\models\face_remove_glasses\dfl_xseg.onnx"
    h, w = image1_1.shape[1], image1_1.shape[2]
    image1_1 = resize_tensor_torch(image1_1)
    # 转化为np
    image1_1 = image1_1.cpu().numpy()
    mask_image = infer_and_get_mask(model_path, image1_1)
    mask_image_1 = mask_image[0]
    mask_image_2 = np.squeeze(mask_image_1, axis=-1)
    # print(mask_image[0].shape)
    # print(mask_image[0][0].shape)
    mask_image_3 = torch.from_numpy(mask_image_2 * 255)
    mask_image_4 = resize_mask_tensor_torch(mask_image_3, (h, w))
    # mask_image = torch.from_numpy(mask_image)
    last_mask = []
    for i in range(image1_1.shape[0]):
        occlusion_mask = mask_image[0][i].transpose(0, 1, 2).clip(0, 1).astype(np.float32)
        occlusion_mask = cv2.resize(occlusion_mask, (h, w))
        occlusion_mask = (cv2.GaussianBlur(occlusion_mask.clip(0, 1), (0, 0), 5).clip(0.5, 1) - 0.5) * 2
        last_mask.append(torch.tensor(occlusion_mask, dtype=torch.float32))
    last_mask_tensor = torch.stack(last_mask, dim=0)
    display_images(last_mask_tensor)
