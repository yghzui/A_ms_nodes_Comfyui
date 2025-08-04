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
    
    def prioritize_audio_mp4(self, mp4_files: List[str]) -> List[str]:
        """优先选择带音频的MP4文件，如果存在带-audio后缀的文件，则只保留音频版本"""
        if not mp4_files:
            return []
        
        # 按文件名分组，找出对应的音频版本
        file_groups = {}
        
        for mp4_file in mp4_files:
            # 获取文件名（不含扩展名）
            file_name = os.path.splitext(os.path.basename(mp4_file))[0]
            
            # 检查是否是音频版本
            is_audio_version = file_name.endswith('-audio')
            
            # 获取基础文件名（去掉-audio后缀）
            base_name = file_name[:-6] if is_audio_version else file_name
            
            # 按基础文件名分组
            if base_name not in file_groups:
                file_groups[base_name] = {'normal': None, 'audio': None}
            
            if is_audio_version:
                file_groups[base_name]['audio'] = mp4_file
            else:
                file_groups[base_name]['normal'] = mp4_file
        
        # 构建最终的文件列表，优先选择音频版本
        final_files = []
        for base_name, versions in file_groups.items():
            if versions['audio']:
                # 如果存在音频版本，优先选择音频版本
                final_files.append(versions['audio'])
                print(f"选择音频版本: {os.path.basename(versions['audio'])} (基础名: {base_name})")
            elif versions['normal']:
                # 如果没有音频版本，选择普通版本
                final_files.append(versions['normal'])
                print(f"选择普通版本: {os.path.basename(versions['normal'])} (基础名: {base_name})")
        
        return final_files
    
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
        
        # 优先选择带音频的MP4文件
        filtered_mp4_files = self.prioritize_audio_mp4(mp4_files)
        
        print(f"ShowResultLast: 过滤后剩余 {len(filtered_mp4_files)} 个MP4文件")
        
        # 构建显示文本列表
        if filtered_mp4_files:
            # 将所有MP4文件路径合并为一个字符串
            display_text = "找到的MP4文件:\n"
            for i, mp4_file in enumerate(filtered_mp4_files, 1):
                display_text += f"{i}. {mp4_file}\n"
        else:
            display_text = "未找到MP4文件"
        
        # 如果启用显示所有文件，也显示原始数据
        if show_all_files:
            display_text += f"\n原始数据:\n{Filenames}"
        
        print(f"ShowResultLast: 显示文本: {display_text}")
        return_data = []
        if filtered_mp4_files:
            for mp4_file in filtered_mp4_files:
                relative_path = os.path.relpath(mp4_file, self.output_dir)
                return_data.append(relative_path)
        # 返回UI更新数据，让前端能够接收
        return {
            "ui": {
                "text": return_data  # 作为一个元素的列表返回
            }
        }
