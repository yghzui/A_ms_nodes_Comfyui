import os
import json
import comfy.model_management
from typing import List, Tuple, Any
from folder_paths import get_output_directory
from .batch_utils import requeue_workflow_unchecked
try:
    from comfy_execution.graph import ExecutionBlocker
except ImportError:
    from comfy_execution.graph_utils import ExecutionBlocker
class ShowResultLast:
    """显示结果节点 - 解析MP4文件并显示在文本框中"""
    
    def __init__(self):
        self.output_dir = get_output_directory()
        self.type = "ShowResultLast"
        self.description = "接收多个文件路径，解析出MP4文件并显示在只读文本框中"
        self.category = "显示工具"
        self.output_node = True
        self.return_type = "STRING"
        self.display_results = []
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "execute"
    OUTPUT_NODE = True
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "Filenames": ("VHS_FILENAMES", {"multiline": True, "default": ""}),
                "show_all_files": ("BOOLEAN", {"default": False}),
                "display_count": ("INT", {"default": -1, "min": -1, "max": 1000, "step": 1, "tooltip": "界面保留的最后视频数量，-1 表示不限制，0 按 1 处理；连接 batch_manager 时忽略此项"}),
                "path_cache": ("STRING", {"default": "", "multiline": False}),
                "batch_manager": ("MY_BATCH_MANAGER",),
            }
        }

    def restore_path_cache(self, path_cache):
        if not path_cache:
            return []
        try:
            cache_data = json.loads(path_cache)
        except (TypeError, json.JSONDecodeError):
            return []
        if isinstance(cache_data, dict):
            cached_paths = cache_data.get("source_paths", [])
        elif isinstance(cache_data, list):
            cached_paths = list(reversed(cache_data))
        else:
            return []
        restored_paths = []
        for path in cached_paths:
            if isinstance(path, str) and path.lower().endswith('.mp4'):
                restored_paths.append(path if os.path.isabs(path) else os.path.join(self.output_dir, path))
        return self.deduplicate_paths(restored_paths)

    def is_manual_cache_update(self, path_cache):
        if not path_cache:
            return False
        try:
            cache_data = json.loads(path_cache)
        except (TypeError, json.JSONDecodeError):
            return False
        return isinstance(cache_data, dict) and bool(cache_data.get("manual_update"))

    def path_key(self, path):
        resolved_path = path if os.path.isabs(path) else os.path.join(self.output_dir, path)
        return os.path.normcase(os.path.abspath(resolved_path))

    def deduplicate_paths(self, paths):
        unique_paths = []
        known_paths = set()
        for path in paths:
            key = self.path_key(path)
            if key not in known_paths:
                unique_paths.append(path)
                known_paths.add(key)
        return unique_paths

    def append_new_results(self, new_paths):
        known_paths = {self.path_key(path) for path in self.display_results}
        for path in new_paths:
            key = self.path_key(path)
            if key not in known_paths:
                self.display_results.append(path)
                known_paths.add(key)
    
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
    
    def execute(self, Filenames=None, show_all_files: bool = False, batch_manager=None, display_count: int = -1, path_cache: str = ""):
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
        
        # --- Batch Manager Logic Start ---
        final_files_to_show = filtered_mp4_files
        status_text = ""
        
        if batch_manager:
            self.display_results = []
            # Add current results to manager
            batch_manager.results.extend(filtered_mp4_files)
            
            # Increment index
            batch_manager.current_index += 1
            print(f"ShowResultLast: Batch Progress {batch_manager.current_index}/{batch_manager.total_count}")
            
            if batch_manager.current_index < batch_manager.total_count:
                # Need to continue loop
                print("ShowResultLast: Triggering requeue and stopping current execution...")
                requeue_workflow_unchecked()
                status_text = f"正在处理第 {batch_manager.current_index} / {batch_manager.total_count} 个任务...\n"
                
                # Stop downstream nodes gracefully using ExecutionBlocker
                print("ShowResultLast: Requeue triggered, returning ExecutionBlocker.")
                return {
                    "ui": {"text": []},
                    "result": (ExecutionBlocker(None),)
                }
            else:
                # Loop finished
                print("ShowResultLast: Batch processing complete. Showing all results.")
                final_files_to_show = batch_manager.results
                status_text = "批量处理完成！所有结果如下：\n"
                # Reset manager state
                batch_manager.is_running = False
                # Clear results to avoid memory leaks
                batch_manager.results = []
        else:
            if self.is_manual_cache_update(path_cache):
                self.display_results = self.restore_path_cache(path_cache)
            elif not self.display_results:
                self.display_results = self.restore_path_cache(path_cache)
            self.append_new_results(filtered_mp4_files)
            if display_count == 0:
                display_count = 1
            if display_count > 0:
                self.display_results = self.display_results[-display_count:]
            final_files_to_show = self.display_results.copy()
        # --- Batch Manager Logic End ---

        final_files_to_show = self.deduplicate_paths(final_files_to_show)
        final_files_to_show.reverse()
        
        # 构建显示文本列表
        if final_files_to_show:
            # 将所有MP4文件路径合并为一个字符串
            display_text = status_text + "找到的MP4文件:\n"
            for i, mp4_file in enumerate(final_files_to_show, 1):
                display_text += f"{i}. {mp4_file}\n"
        else:
            display_text = status_text + ("" if status_text else "未找到MP4文件")
        
        # 如果启用显示所有文件，也显示原始数据
        if show_all_files:
            display_text += f"\n原始数据:\n{Filenames}"
        
        print(f"ShowResultLast: 显示文本: {display_text}")
        return_data = []
        if final_files_to_show:
            for mp4_file in final_files_to_show:
                try:
                    relative_path = os.path.relpath(mp4_file, self.output_dir)
                    return_data.append(relative_path)
                except Exception as e:
                    print(f"Error calculating relpath for {mp4_file}: {e}")
                    
        # 返回UI更新数据，让前端能够接收
        return {
            "ui": {
                "text": return_data,  # 作为一个元素的列表返回
                "path_cache": [json.dumps({"source_paths": list(reversed(final_files_to_show)), "display_paths": return_data}, ensure_ascii=False)],
            },
            "result": (display_text,)
        }
