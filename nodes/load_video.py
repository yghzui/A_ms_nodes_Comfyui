import os
import cv2
import numpy as np
import torch
from typing import List, Tuple
try:
    from moviepy import VideoFileClip
except ImportError:
    from moviepy.editor import VideoFileClip

class LoadVideoFromFolder:
    """
    从文件夹中加载视频文件的节点，保持原始宽高
    当target_fps=0时使用视频原始帧率
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": ("STRING", {"default": ""}),  # 视频文件夹路径
                "target_fps": ("INT", {"default": 0, "min": 0, "max": 120}),  # 目标帧率，0表示使用原始帧率
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "STRING", "INT", "AUDIO")  # 返回图像列表、帧率列表、文件名列表、帧数列表、音频列表
    RETURN_NAMES = ("images", "fps_list", "filenames", "frame_counts", "audio_list")
    FUNCTION = "load_videos"
    CATEGORY = "A_my_nodes"
    OUTPUT_IS_LIST = (True, True, True, True, True)

    def load_videos(self, folder_path: str, target_fps: int) -> Tuple[List[torch.Tensor], List[int], List[str], List[int], List[np.ndarray]]:
        """
        加载文件夹中的所有视频文件，保持原始宽高
        
        Args:
            folder_path: 视频文件夹路径
            target_fps: 目标帧率，0表示使用原始帧率
            
        Returns:
            Tuple[List[torch.Tensor], List[int], List[str], List[int], List[np.ndarray]]: 
            - 图像张量列表 (每个张量形状为[n,h,w,c])
            - 帧率列表
            - 文件名列表
            - 帧数列表
            - 音频数据列表
        """
        if not os.path.exists(folder_path):
            raise ValueError(f"文件夹路径不存在: {folder_path}")
            
        # 支持的视频格式
        video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv')
        
        # 存储结果
        all_frames = []
        fps_list = []
        filenames = []
        frame_counts = []  # 存储每个视频的帧数
        audio_list = []    # 存储每个视频的音频
        
        # 遍历文件夹中的视频文件
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(video_extensions):
                video_path = os.path.join(folder_path, filename)
                
                # 打开视频文件
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    print(f"无法打开视频文件: {video_path}")
                    continue
                    
                # 获取原始视频信息
                orig_fps = int(cap.get(cv2.CAP_PROP_FPS))
                
                # 如果target_fps为0，使用原始帧率
                current_fps = orig_fps if target_fps == 0 else target_fps
                frame_interval = max(1, round(orig_fps / current_fps)) if target_fps > 0 else 1
                
                frames = []
                frame_count = 0
                total_frames = 0  # 记录实际保存的帧数
                
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                        
                    # 按目标帧率采样
                    if frame_count % frame_interval == 0:
                        # BGR转RGB，保持原始宽高
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frames.append(frame)
                        total_frames += 1
                    
                    frame_count += 1
                
                cap.release()
                
                if frames:  # 如果成功读取了帧
                    # 转换为torch张量
                    frames_tensor = torch.from_numpy(np.stack(frames)).float() / 255.0
                    all_frames.append(frames_tensor)
                    fps_list.append(current_fps)  # 使用实际使用的帧率
                    filenames.append(filename)
                    frame_counts.append(total_frames)  # 添加实际帧数
                    
                    # 提取音频
                    try:
                        video = VideoFileClip(video_path)
                        if video.audio is not None:
                            # 获取音频数据
                            audio_data = video.audio.to_soundarray()
                            # 如果是立体声，转换为单声道
                            if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
                                audio_data = np.mean(audio_data, axis=1)
                            audio_list.append(audio_data)
                        else:
                            # 如果没有音频，添加空数组
                            audio_list.append(np.array([]))
                        video.close()
                    except Exception as e:
                        print(f"提取音频失败: {e}")
                        audio_list.append(np.array([]))
        
        if not all_frames:
            raise ValueError(f"在文件夹中没有找到有效的视频文件: {folder_path}")
            
        return (all_frames, fps_list, filenames, frame_counts, audio_list)
