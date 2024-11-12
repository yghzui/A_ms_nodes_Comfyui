# -*- coding: utf-8 -*-
# Created time : 2024/11/12 22:02 
# Auther : ygh
# File   : face_without_glasses.py
# Description :
import cv2
import numpy as np
import onnxruntime as ort
import time


# 加载 ONNX 模型
def load_onnx_model(model_path: str):
    session = ort.InferenceSession(model_path)
    return session


# 读取并预处理图片
def preprocess_image(image_path: str, input_shape: tuple):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    # 调整图片大小
    resized_image = cv2.resize(image, (input_shape[1], input_shape[2]))

    # 转换为浮点数并归一化
    input_image = resized_image.astype(np.float32) / 255.0

    # 增加批次维度
    input_image = np.expand_dims(input_image, axis=0)

    return input_image


# 进行推理并获取输出
def infer_and_get_mask(model_path, input_image):
    session = load_onnx_model(model_path)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # 开始计时
    start_time = time.time()

    # 进行推理
    mask = session.run([output_name], {input_name: input_image})

    # 结束计时
    end_time = time.time()
    inference_time = end_time - start_time
    print(f"Inference time: {inference_time:.4f} seconds")
    return mask
