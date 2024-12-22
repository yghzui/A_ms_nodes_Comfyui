# -*- coding: utf-8 -*-
# Created time : 2024/12/22 19:31 
# Auther : ygh
# File   : 测试.py
# Description :
import os
import cv2
import torch
import numpy as np
from PIL import Image
from a_person_face_landmark_mask_generator_comfyui_by_my import APersonFaceLandmarkMaskGenerator
import matplotlib.pyplot as plt
import math
import mediapipe as mp
# 读取本地图像并转换为指定的 torch.Tensor 格式
def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)[None,]  # [1, 3, H, W]
    return image_tensor

# 定义 display_images 函数
def display_images(image_tensor):
    a = """
    输入格式:
    输入rgb图片: torch.Size([图片张数, 通道, 高, 宽])
    输入rgba图片: torch.Size([4, 4, 512, 512])
    输入mask图片: torch.Size([图片张数, 高, 宽])
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
        if image.ndim == 3 and image.shape[0] == 4:
            # RGBA 图像
            axes[i].imshow(image.transpose(1, 2, 0))
        elif image.ndim == 3 and image.shape[0] == 3:
            # RGB 图像
            axes[i].imshow(image.transpose(1, 2, 0))
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

# 测试代码
if __name__ == "__main__":
    # 读取本地图像
    image_path = r"D:\AI\SD\face_id\caiwenjie.png"  # 替换为你的图像路径
    image_tensor = load_image(image_path)

    # 创建掩码生成器实例
    mask_generator = APersonFaceLandmarkMaskGenerator()

    # 生成掩码
    # image_tensor = mp.Image.create_from_file(image_path)
    masks, = mask_generator.generate_mask(
        images=image_tensor,
        face=False,
        left_eyebrow=False,
        right_eyebrow=False,
        left_eye=True,
        right_eye=True,
        left_pupil=False,
        right_pupil=False,
        lips=True,
        nose=True,
        number_of_faces=1,
        confidence=0.40,
    )


    # 显示生成的掩码
    display_images(masks)
