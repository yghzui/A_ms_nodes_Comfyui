# -*- coding: utf-8 -*-
# Created time : 2024/10/01 15:22
# Auther : ygh
# File   : get_result_text_p.py
# Description :
import base64
import os
import subprocess
import numpy as np


def numpy_to_base64(numpy_img):
    # 将 numpy 图像数组编码为 JPEG 格式
    _, buffer = cv2.imencode(".jpg", numpy_img)
    # 将编码的图片数据转换为 base64 字符串
    img_base64 = base64.b64encode(buffer).decode("utf-8")
    return img_base64


def str_to_array(s):
    """将字符串转换为NumPy数组"""
    arr = np.array(eval(s))
    return arr.astype(int)


def str_to_list(s):
    lst = eval(s)
    return lst


def get_list_from_txt(txt_path):
    with open(txt_path, "r") as f:
        result = f.read()
    if "None" in result:
        return None
    # # 删除text文件
    # if os.path.exists(txt_path):
    #     os.remove(txt_path)
    return str_to_list(result)


def get_numpy_from_txt(txt_path):
    with open(txt_path, "r") as f:
        result = f.read()
        return str_to_array(result)


def run_script_with_subprocess(img_path, save_text_path, use_gpu=False):
    # 构建命令行参数
    command = [
        r"D:\software\anaconda3\envs\paddleocr310\python.exe",
        r"D:\Anaconda_projet\PaddleOCR\ocr_example.py",
        "--img_path",
        img_path,
        "--save_text_path",
        save_text_path,
        "--use_gpu",
        str(use_gpu),
    ]

    # 运行子进程并捕获输出
    subprocess.run(command, capture_output=True, text=True)


#
# print(sys.executable)
if __name__ == "__main__":
    # 示例数据
    import cv2

    img_path = r"D:\Anaconda_projet\PaddleOCR\ppocr_img\img_m_leg\3c387d15d13459353a51ecb2ecd54482c144690d9d427dd7ac3519165d3d2bee.0.PNG"
    img = cv2.imread(img_path)
    img_numpy = img  # np.array([[1, 2, 3], [4, 5, 6]])
    save_text_path = r"temp/result_ocr_img_numpy.txt"
    use_gpu = False

    # 运行脚本
    # 运行脚本
    # run_script_with_subprocess(img_path, save_text_path, use_gpu)
    result_list = get_list_from_txt(save_text_path)
    img = cv2.imread(img_path)
    result_list = result_list[0]
    print(result_list)
    print(len(result_list))

    print(img.shape)
    width, height = img.shape[1], img.shape[0]

    # # 创建一个与图像相同大小的空白蒙版，初始化为黑色
    mask = np.zeros((height, width), dtype=np.uint8)
    for i in range(len(result_list)):
        # 将这些点转换为所需的形状（-1, 1, 2）以便在图像上绘制
        points = np.array(result_list[i], dtype=np.int32)
        # points = points.reshape((-1, 1, 2))

        # 使用 OpenCV 画出多边形，参数 (图像, 点, 是否闭合, 颜色, 线条宽度)
        cv2.polylines(img, [points], isClosed=True, color=(0, 255, 0), thickness=2)
        # 填充多边形的内部为白色
        cv2.fillPoly(mask, [points], color=(255, 255, 255))

    # 显示图像
    cv2.imshow("Image with Rectangle", img)
    cv2.imshow("Mask", mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
