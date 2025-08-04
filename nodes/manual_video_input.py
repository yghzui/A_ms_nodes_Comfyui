import os
from typing import List, Tuple, Any
from folder_paths import get_output_directory

class ManualVideoInput:
    """手动输入视频文件名节点 - 用于测试ShowResultLast节点"""
    
    def __init__(self):
        self.output_dir = get_output_directory()
        self.type = "ManualVideoInput"
        self.description = "手动输入视频文件名，转换为ShowResultLast节点可接受的格式"
        self.category = "显示工具"
        self.output_node = False
        self.return_type = "VHS_FILENAMES"
    
    RETURN_TYPES = ("VHS_FILENAMES",)
    RETURN_NAMES = ("filenames",)
    FUNCTION = "execute"
    OUTPUT_NODE = False
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_files": ("STRING", {
                    "multiline": True, 
                    "default": "视频1.mp4\n视频2.mp4\n视频3.mp4",
                    "placeholder": "请输入视频文件名，每行一个，例如：\n视频1.mp4\n视频2.mp4\n视频3.mp4"
                }),
                "use_relative_path": ("BOOLEAN", {"default": True}),
                "add_audio_suffix": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "custom_folder": ("STRING", {
                    "default": "测试",
                    "placeholder": "自定义文件夹名称（可选）"
                }),
            }
        }
    
    def parse_video_files(self, video_files_str: str) -> List[str]:
        """解析视频文件名字符串"""
        if not video_files_str.strip():
            return []
        
        # 按行分割，去除空行和空白字符
        lines = [line.strip() for line in video_files_str.strip().split('\n')]
        video_files = [line for line in lines if line and line.lower().endswith('.mp4')]
        
        return video_files
    
    def execute(self, video_files, use_relative_path: bool = True, add_audio_suffix: bool = False, custom_folder: str = "测试"):
        """执行节点逻辑"""
        print(f"ManualVideoInput: 接收到视频文件列表: {video_files}")
        
        # 解析视频文件名
        parsed_files = self.parse_video_files(video_files)
        print(f"ManualVideoInput: 解析出 {len(parsed_files)} 个视频文件")
        
        if not parsed_files:
            print("ManualVideoInput: 没有找到有效的MP4文件")
            return (["未找到MP4文件"],)
        
        # 构建完整的文件路径
        result_files = []
        for filename in parsed_files:
            # 确保文件名有.mp4扩展名
            if not filename.lower().endswith('.mp4'):
                filename += '.mp4'
            
            if use_relative_path:
                # 使用相对路径格式
                relative_path = os.path.join(custom_folder, filename)
                result_files.append(relative_path)
            else:
                # 使用绝对路径格式
                absolute_path = os.path.join(self.output_dir, custom_folder, filename)
                result_files.append(absolute_path)
            
            # 如果启用音频后缀，添加带音频的版本
            if add_audio_suffix:
                base_name = os.path.splitext(filename)[0]
                audio_filename = f"{base_name}-audio.mp4"
                
                if use_relative_path:
                    audio_relative_path = os.path.join(custom_folder, audio_filename)
                    result_files.append(audio_relative_path)
                else:
                    audio_absolute_path = os.path.join(self.output_dir, custom_folder, audio_filename)
                    result_files.append(audio_absolute_path)
        
        print(f"ManualVideoInput: 生成 {len(result_files)} 个文件路径")
        print(f"ManualVideoInput: 文件路径列表: {result_files}")
        
        # 返回VHS_FILENAMES格式的数据
        return (result_files,) 