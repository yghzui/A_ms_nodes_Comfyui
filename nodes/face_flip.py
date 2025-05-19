import os
import sys
import cv2
import torch
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import folder_paths
from typing import List, Tuple, Dict, Any, Optional, Union
from PIL import Image, ImageDraw, ImageFont


def draw_eye_landmarks(image: np.ndarray, landmarks, mode: str) -> np.ndarray:
    """
    在图像上绘制关键点
    
    Args:
        image: 输入图像
        landmarks: mediapipe 检测到的关键点
        mode: 检测模式，'eyes' 或 'nose_eyes'
        
    Returns:
        标注了关键点的图像
    """
    img = image.copy()
    h, w = img.shape[:2]
    
    # 定义关键点和颜色
    if mode == 'eyes':
        points = {
            "right_eye_left": (33, (0, 255, 0)),     # 右眼左角 - 绿色
            "right_eye_right": (133, (0, 255, 0)),   # 右眼右角 - 绿色
            "right_pupil": (468, (255, 0, 0)),       # 右眼瞳孔 - 红色
            "left_eye_left": (362, (0, 0, 255)),     # 左眼左角 - 蓝色
            "left_eye_right": (263, (0, 0, 255)),    # 左眼右角 - 蓝色
            "left_pupil": (473, (255, 0, 0)),        # 左眼瞳孔 - 红色
        }
    else:  # nose_eyes mode
        points = {
            "nose_tip": (4, (255, 0, 0)),         # 鼻尖 - 红色
            "right_eye_right": (133, (0, 255, 0)), # 右眼右角 - 绿色
            "left_eye_left": (362, (0, 0, 255)),   # 左眼左角 - 蓝色
        }
    
    # 绘制每个关键点
    for _, (point_idx, color) in points.items():
        point = landmarks.landmark[point_idx]
        x = int(point.x * w)
        y = int(point.y * h)
        # 绘制点
        cv2.circle(img, (x, y), 4, color, -1)  # 稍微加大点的大小
        cv2.circle(img, (x, y), 6, color, 2)   # 添加边框使点更明显
    
    return img

def detect_face_orientation(image: np.ndarray, detection_threshold: float = 0.5, mode: str = 'eyes') -> Tuple[Optional[str], Optional[Any]]:
    """
    使用MediaPipe检测人脸朝向
    
    Args:
        image: 输入的BGR格式图像
        detection_threshold: 人脸检测阈值
        mode: 检测模式，'eyes' 使用瞳孔位置，'nose_eyes' 使用鼻尖和外眼角距离
        
    Returns:
        人脸朝向，"left"或"right"，如果没有检测到脸则返回None
        检测到的关键点
    """
    # 使用MediaPipe face_mesh
    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        refine_landmarks=True,
        max_num_faces=5,  # 检测多个人脸
        min_detection_confidence=detection_threshold,  # 人脸检测置信度阈值
    ) as face_mesh:
        # 将BGR图像转换为RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 处理图像
        results = face_mesh.process(rgb_image)
        
        # 如果没有检测到人脸，返回None
        if not results.multi_face_landmarks:
            print("[FaceFlip] 未检测到人脸")
            return None, None
        
        # 获取所有检测到的人脸
        face_landmarks = results.multi_face_landmarks
        print(f"[FaceFlip] 检测到 {len(face_landmarks)} 个人脸")
        
        # 使用第一个检测到的人脸（通常是最大/最清晰的）
        landmarks = face_landmarks[0]
        
        if mode == 'nose_eyes':
            try:
                # 获取鼻尖和外眼角的关键点
                nose_tip = landmarks.landmark[4]      # 鼻尖
                right_eye = landmarks.landmark[133]   # 右眼右角
                left_eye = landmarks.landmark[362]    # 左眼左角
                
                # 计算鼻尖到两个外眼角的距离
                dist_to_right = ((nose_tip.x - right_eye.x)**2 + 
                               (nose_tip.y - right_eye.y)**2)**0.5
                dist_to_left = ((nose_tip.x - left_eye.x)**2 + 
                              (nose_tip.y - left_eye.y)**2)**0.5
                
                # 比较距离判断朝向
                if dist_to_left > dist_to_right:
                    print(f"[FaceFlip] 鼻尖到左眼距离 {dist_to_left:.2f} > 到右眼距离 {dist_to_right:.2f}，判断为朝右")
                    return "right", landmarks
                else:
                    print(f"[FaceFlip] 鼻尖到左眼距离 {dist_to_left:.2f} < 到右眼距离 {dist_to_right:.2f}，判断为朝左")
                    return "left", landmarks
                    
            except (IndexError, AttributeError) as e:
                print(f"[FaceFlip] 关键点检测错误: {e}")
                return None, None
        
        else:  # eyes mode
            try:
                # 右眼的关键点 (33: 右眼左角, 133: 右眼右角)
                right_eye_left = landmarks.landmark[33]
                right_eye_right = landmarks.landmark[133]
                # 右眼的瞳孔中心点 (468: 右眼瞳孔)
                right_pupil = landmarks.landmark[468]
                
                # 左眼的关键点 (362: 左眼左角, 263: 左眼右角)
                left_eye_left = landmarks.landmark[362]
                left_eye_right = landmarks.landmark[263]
                # 左眼的瞳孔中心点 (473: 左眼瞳孔)
                left_pupil = landmarks.landmark[473]
            except (IndexError, AttributeError) as e:
                print(f"[FaceFlip] 关键点检测错误: {e}")
                return None, None
            
            # 计算瞳孔距离眼角的距离，用于判断脸的朝向
            orientation_left = "unknown"
            orientation_right = "unknown"
            
            # 检查左眼
            valid_left = True
            try:
                # 计算左眼瞳孔距左眼角和右眼角的距离
                dist_to_left = ((left_pupil.x - left_eye_left.x)**2 + 
                                (left_pupil.y - left_eye_left.y)**2)**0.5
                dist_to_right = ((left_pupil.x - left_eye_right.x)**2 + 
                                (left_pupil.y - left_eye_right.y)**2)**0.5
                
                if dist_to_left > dist_to_right:
                    orientation_left = "left"
                    print(f"[FaceFlip] 左眼: 瞳孔距左眼角 {dist_to_left:.2f} > 距右眼角 {dist_to_right:.2f}，判断为朝向左侧")
                else:
                    orientation_left = "right"
                    print(f"[FaceFlip] 左眼: 瞳孔距左眼角 {dist_to_left:.2f} < 距右眼角 {dist_to_right:.2f}，判断为朝向右侧")
            except (AttributeError, TypeError) as e:
                print(f"[FaceFlip] 左眼距离计算错误: {e}")
                valid_left = False
            
            # 检查右眼
            valid_right = True
            try:
                # 计算右眼瞳孔距左眼角和右眼角的距离
                dist_to_left = ((right_pupil.x - right_eye_left.x)**2 + 
                               (right_pupil.y - right_eye_left.y)**2)**0.5
                dist_to_right = ((right_pupil.x - right_eye_right.x)**2 + 
                                (right_pupil.y - right_eye_right.y)**2)**0.5
                
                if dist_to_left > dist_to_right:
                    orientation_right = "left"
                    print(f"[FaceFlip] 右眼: 瞳孔距左眼角 {dist_to_left:.2f} > 距右眼角 {dist_to_right:.2f}，判断为朝向左侧")
                else:
                    orientation_right = "right"
                    print(f"[FaceFlip] 右眼: 瞳孔距左眼角 {dist_to_left:.2f} < 距右眼角 {dist_to_right:.2f}，判断为朝向右侧")
            except (AttributeError, TypeError) as e:
                print(f"[FaceFlip] 右眼距离计算错误: {e}")
                valid_right = False
            
            # 根据检测结果确定最终朝向
            final_orientation = None
            if valid_left and valid_right:
                # 两只眼睛都可用，判断结果一致则使用该结果
                if orientation_left == orientation_right:
                    final_orientation = orientation_left
                    print(f"[FaceFlip] 左右眼判断结果一致: {final_orientation}")
                else:
                    # 如果结果不一致，取左眼的结果（通常左眼更准确）
                    final_orientation = orientation_left
                    print(f"[FaceFlip] 左右眼判断结果不一致，优先使用左眼结果: {final_orientation}")
            elif valid_left:
                # 只有左眼可用
                final_orientation = orientation_left
                print(f"[FaceFlip] 只有左眼可用，使用左眼判断结果: {final_orientation}")
            elif valid_right:
                # 只有右眼可用
                final_orientation = orientation_right
                print(f"[FaceFlip] 只有右眼可用，使用右眼判断结果: {final_orientation}")
            else:
                # 都不可用
                print("[FaceFlip] 眼部关键点无法用于判断朝向")
                return None, None
            
            return final_orientation, landmarks


class FaceFlip:
    """
    FaceFlip节点: 用于根据人脸朝向或手动设置翻转图像
    
    功能:
    1. 自动检测人脸朝向，确保人脸朝向特定方向
    2. 支持手动水平和垂直翻转
    3. 支持批量处理多张图像
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),  # 输入图像张量，格式为n,h,w,c
                "auto_flip_face": ("BOOLEAN", {"default": True, "tooltip": "根据人脸朝向自动翻转图像"}),
                "target_orientation": (["left", "right"], {"default": "left", "tooltip": "设置人脸希望朝向的方向"}),
                "horizontal_flip": ("BOOLEAN", {"default": False, "tooltip": "手动水平翻转图像"}),
                "vertical_flip": ("BOOLEAN", {"default": False, "tooltip": "手动垂直翻转图像"}),
                "detection_threshold": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 0.9, "step": 0.05, "tooltip": "人脸检测阈值，值越低检测越宽松"}),
                "detection_mode": (["eyes", "nose_eyes"], {"default": "eyes", "tooltip": "检测模式：eyes-使用瞳孔位置，nose_eyes-使用鼻尖和外眼角距离"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("flipped_image", "landmarks_image")
    FUNCTION = "flip_images"
    CATEGORY = "My_node/image"

    def flip_images(self, image, auto_flip_face, target_orientation, horizontal_flip, vertical_flip, detection_threshold, detection_mode):
        """
        执行图像翻转处理
        
        Args:
            image: 输入图像张量 (n,h,w,c)
            auto_flip_face: 是否根据人脸朝向自动翻转
            target_orientation: 目标朝向 ('left'或'right')
            horizontal_flip: 是否手动水平翻转
            vertical_flip: 是否手动垂直翻转
            detection_threshold: 人脸检测阈值
            detection_mode: 检测模式
            
        Returns:
            处理后的图像张量 (n,h,w,c)
            带关键点标注的图像张量 (n,h,w,c)
        """
        # 输入是(n,h,w,c)格式的张量，转换为numpy处理
        batch_size = image.shape[0]
        output_images = []
        output_landmarks_images = []
        
        print(f"[FaceFlip] 开始处理 {batch_size} 张图像")
        print(f"[FaceFlip] 参数: 自动翻转={auto_flip_face}, 目标朝向={target_orientation}, 水平翻转={horizontal_flip}, 垂直翻转={vertical_flip}, 检测模式={detection_mode}")

        for i in range(batch_size):
            # 处理每张图像
            print(f"[FaceFlip] 处理第 {i+1}/{batch_size} 张图像")
            img = image[i].cpu().numpy()  # 转换为numpy数组
            # 转换为OpenCV格式 (BGR)，便于人脸检测
            img_cv = (img * 255).astype(np.uint8)
            # 默认不翻转
            do_horizontal_flip = horizontal_flip
            
            # 创建带关键点的图像副本
            landmarks_img = img.copy()
            landmarks_img_cv = img_cv.copy()
            
            if auto_flip_face:
                # 检测人脸朝向
                face_orientation, landmarks = detect_face_orientation(img_cv, detection_threshold, detection_mode)
                
                if face_orientation is not None:
                    # 如果检测到人脸，并且朝向与目标方向不符，进行水平翻转
                    if face_orientation != target_orientation:
                        do_horizontal_flip = not do_horizontal_flip  # 翻转状态取反
                        print(f"[FaceFlip] 检测到人脸朝向 {face_orientation}，与目标朝向 {target_orientation} 不符，需要翻转")
                    else:
                        print(f"[FaceFlip] 检测到人脸朝向 {face_orientation}，与目标朝向 {target_orientation} 相符，无需翻转")
                    
                    # 在landmarks图像上绘制关键点
                    if landmarks is not None:
                        landmarks_img_cv = draw_eye_landmarks(landmarks_img_cv, landmarks, detection_mode)
                        landmarks_img = landmarks_img_cv.astype(np.float32) / 255.0
                else:
                    print("[FaceFlip] 未能确定人脸朝向，跳过自动翻转")
            
            # 根据最终确定的状态进行翻转
            if do_horizontal_flip:
                print("[FaceFlip] 执行水平翻转")
                img = np.flip(img, axis=1).copy()  # 添加.copy()确保数组连续
                landmarks_img = np.flip(landmarks_img, axis=1).copy()
            
            if vertical_flip:
                print("[FaceFlip] 执行垂直翻转")
                img = np.flip(img, axis=0).copy()  # 添加.copy()确保数组连续
                landmarks_img = np.flip(landmarks_img, axis=0).copy()
            
            # 确保数组是连续的
            if not img.flags['C_CONTIGUOUS']:
                img = np.ascontiguousarray(img)
            if not landmarks_img.flags['C_CONTIGUOUS']:
                landmarks_img = np.ascontiguousarray(landmarks_img)
            
            # 添加到输出列表
            output_images.append(torch.from_numpy(img))
            output_landmarks_images.append(torch.from_numpy(landmarks_img))
        
        print("[FaceFlip] 处理完成")
        # 将所有处理后的图像合并为批次
        return (torch.stack(output_images), torch.stack(output_landmarks_images))


if __name__ == "__main__":
    # 测试代码
    import cv2
    import torch
    import os
    
    # 读取测试图片
    image_path = r"D:\AI\SD\input\小伙伴\A未分类\mmexport1747579574354.jpg" # 请确保这个路径存在
    
    # 处理中文路径
    img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print(f"无法读取图片: {image_path}")
        exit(1)
    
    # 转换为 RGB 并归一化到 0-1
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    
    # 转换为 torch tensor 并添加 batch 维度
    img_tensor = torch.from_numpy(img)[None, ...]  # shape: (1, H, W, 3)
    
    # 创建 FaceFlip 实例并处理图片
    face_flip = FaceFlip()
    flipped_image, landmarks_image = face_flip.flip_images(
        image=img_tensor,
        auto_flip_face=True,
        target_orientation="right",
        horizontal_flip=False,
        vertical_flip=False,
        detection_threshold=0.3,
        detection_mode="nose_eyes"  # 使用新的检测模式
    )
    
    # 转换回 OpenCV 格式并保存
    result_img = (flipped_image[0].numpy() * 255).astype(np.uint8)
    result_img = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
    
    landmarks_img = (landmarks_image[0].numpy() * 255).astype(np.uint8)
    landmarks_img = cv2.cvtColor(landmarks_img, cv2.COLOR_RGB2BGR)
    
    # 获取原始图片的目录和文件名
    dir_path = os.path.dirname(image_path)
    file_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # 保存原图和结果图片
    original_save_path = os.path.join(dir_path, f"{file_name}_original.jpg")
    result_save_path = os.path.join(dir_path, f"{file_name}_result.jpg")
    landmarks_save_path = os.path.join(dir_path, f"{file_name}_landmarks.jpg")
    
    # 使用 imencode 和 fromfile 来保存带中文路径的图片
    cv2.imencode('.jpg', img * 255)[1].tofile(original_save_path)
    cv2.imencode('.jpg', result_img)[1].tofile(result_save_path)
    cv2.imencode('.jpg', landmarks_img)[1].tofile(landmarks_save_path)
    
    print(f"原图已保存至: {original_save_path}")
    print(f"结果图已保存至: {result_save_path}")
    print(f"关键点图已保存至: {landmarks_save_path}")




