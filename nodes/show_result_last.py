import os
import json
from typing import List, Tuple, Any
from folder_paths import get_output_directory
class ShowResultLast:
    """显示结果节点 - 解析MP4文件并显示在文本框中"""
    
    def __init__(self):
        self.output_dir = get_output_directory()
        self.type = "ShowResultLast"
        self.description = "接收多个文件路径，解析出MP4文件并显示在只读文本框中"
        self.category = "显示工具"
        self.output_node = True
        self.return_type = "STRING"
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "execute"
    OUTPUT_NODE = True
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "Filenames": ("VHS_FILENAMES", {"multiline": True, "default": ""}),
            },
            "optional": {
                "show_all_files": ("BOOLEAN", {"default": False}),
            }
        }
    
    def parse_file_paths(self, file_paths_str: str) -> List[str]:
        """解析文件路径字符串，提取MP4文件"""
        mp4_files = []
        
        try:
            # 尝试解析JSON格式
            if file_paths_str.strip().startswith('['):
                data = json.loads(file_paths_str)
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, list) and len(item) >= 2:
                            # 处理 [true, ["file1.png", "file2.mp4"]] 格式
                            if isinstance(item[1], list):
                                for file_path in item[1]:
                                    if isinstance(file_path, str) and file_path.lower().endswith('.mp4'):
                                        mp4_files.append(file_path)
                            # 处理单个文件路径
                            elif isinstance(item[1], str) and item[1].lower().endswith('.mp4'):
                                mp4_files.append(item[1])
                        # 处理单个文件路径
                        elif isinstance(item, str) and item.lower().endswith('.mp4'):
                            mp4_files.append(item)
            else:
                # 处理普通文本格式，按行分割
                lines = file_paths_str.strip().split('\n')
                for line in lines:
                    line = line.strip()
                    if line and line.lower().endswith('.mp4'):
                        mp4_files.append(line)
                        
        except json.JSONDecodeError:
            # JSON解析失败，按普通文本处理
            lines = file_paths_str.strip().split('\n')
            for line in lines:
                line = line.strip()
                if line and line.lower().endswith('.mp4'):
                    mp4_files.append(line)
        
        return mp4_files
    
    def execute(self, Filenames, show_all_files: bool = False):
        """执行节点逻辑"""
        print(f"ShowResultLast: 接收到文件路径数据: {Filenames}")
        
        mp4_files = []
        
        def extractMp4Files(data):
            """递归提取MP4文件路径"""
            if isinstance(data, dict):
                for k in data:
                    extractMp4Files(data[k])
            elif isinstance(data, list):
                for i in range(len(data)):
                    extractMp4Files(data[i])
            elif isinstance(data, tuple):
                # 处理 (True, [...]) 格式
                if len(data) >= 2 and isinstance(data[1], list):
                    for item in data[1]:
                        extractMp4Files(item)
                else:
                    for item in data:
                        extractMp4Files(item)
            elif isinstance(data, str) and data.lower().endswith('.mp4'):
                mp4_files.append(data)
        
        # 开始提取MP4文件
        extractMp4Files(Filenames)
        
        print(f"ShowResultLast: 解析出 {len(mp4_files)} 个MP4文件")
        
        # 构建显示文本列表
        if mp4_files:
            # 将所有MP4文件路径合并为一个字符串
            display_text = "找到的MP4文件:\n"
            for i, mp4_file in enumerate(mp4_files, 1):
                display_text += f"{i}. {mp4_file}\n"
        else:
            display_text = "未找到MP4文件"
        
        # 如果启用显示所有文件，也显示原始数据
        if show_all_files:
            display_text += f"\n原始数据:\n{Filenames}"
        
        print(f"ShowResultLast: 显示文本: {display_text}")
        return_data = []
        if mp4_files:
            for mp4_file in mp4_files:
                relative_path = os.path.relpath(mp4_file, self.output_dir)
                return_data.append(relative_path)
        # 返回UI更新数据，让前端能够接收
        return {
            "ui": {
                "text": return_data  # 作为一个元素的列表返回
            }
        }
