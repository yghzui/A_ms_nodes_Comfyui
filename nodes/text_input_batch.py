# -*- coding: utf-8 -*-
# Created by My: 批量字符串输入节点（后端）
# 说明：
# - 提供一个可选的 index 输入（INT），用于选择返回第几个字符串
# - 字符串列表通过前端动态控件聚合为 JSON 写入 strings_json（隐藏/内部容器）
# - 输出：字符串列表（JSON 字符串）与根据 index 选中的字符串；当 index 无效或越界时回退到第一个；当列表为空返回空串
# - 新增：支持标题编辑和启用状态管理，输出包含标题、内容、启用状态的字典

import json


class TextInputBatch:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # 选中的索引作为必填输入口；保留小部件以便直接输入（不强制为连接端口）
                "index": ("INT", {"default": 0, "min": 0, "max": 100000000, "tooltip": "返回第几个字符串（从0开始）。越界则回退到第一个"}),
            },
            "optional": {
                "columns": ("INT", {"default": 2, "min": 1, "max": 8, "tooltip": "布局列数（1-8）"}),
                "batch_manager": ("MY_BATCH_MANAGER",),
            },
            "hidden": {
                # 字符串列表容器（前端写入，后端解析）；隐藏以避免未加载前端脚本时出现可见输入框
                "strings_json": ("STRING", {"default": "[]"}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "INT", "STRING", "STRING")
    RETURN_NAMES = ("strings", "selected", "selected_title", "count", "dict_output", "dict_output_list")
    OUTPUT_IS_LIST = (True, False, False, False, False, True)
    FUNCTION = "aggregate_strings"
    CATEGORY = "A_my_nodes/text"

    def aggregate_strings(self, index=0, strings_json="[]", columns=2, batch_manager=None):
        # 健壮解析 JSON
        try:
            data = json.loads(strings_json) if isinstance(strings_json, str) else []
        except Exception:
            data = []

        # --- Batch Manager 逻辑 (提前到数据构建前) ---
        # 预先计算 current_index，因为在构建 entries 时需要用到它
        
        # 1. 找出所有已启用的索引
        enabled_indices = []
        if isinstance(data, list):
            for i, item in enumerate(data):
                is_enabled = True
                if isinstance(item, dict):
                    is_enabled = item.get("enabled", True)
                if is_enabled:
                    enabled_indices.append(i)
        
        # 2. 根据启用列表计算 current_index
        if batch_manager is not None:
            # 如果有启用的条目，仅在启用条目中循环
            if len(enabled_indices) > 0:
                batch_manager.total_count = len(enabled_indices)
                # 获取在 enabled_indices 中的相对索引
                batch_idx = batch_manager.current_index
                # 映射回实际索引 (防止越界)
                if batch_idx >= len(enabled_indices):
                    batch_idx = batch_idx % len(enabled_indices)
                current_index = enabled_indices[batch_idx]
                print(f"TextInputBatch: BatchManager filtered index {batch_idx}/{len(enabled_indices)} -> Real index {current_index}")
            else:
                # 如果没有启用的条目，回退到默认行为（处理总数为0的情况）
                batch_manager.total_count = 1 
                current_index = 0
        else:
            try:
                current_index = int(index) if index is not None else 0
            except Exception:
                current_index = 0
        # ------------------------

        # {{ AURA-X: Modify - 修改启用状态逻辑，确保只有当前选中索引对应项目启用. }}
        # 处理新的数据结构，支持标题和启用状态
        strings_list = []
        dict_output_list=[]
        titles_dict = {}
        entries = []
        
        # 构建 entries 列表
        if isinstance(data, list):
            for i, item in enumerate(data):
                if isinstance(item, dict) and "title" in item and "content" in item:
                    title = item.get("title", f"prompt_{i}")
                    content = str(item.get("content", ""))
                    # 默认启用，除非明确设置为False
                    enabled = item.get("enabled", True)
                else:
                    content = str(item) if item is not None else ""
                    title = f"prompt_{i}"
                    enabled = True
                
                # 如果是当前选中的索引，强制视为启用（虽然前端也做了限制，后端双重保障）
                if i == current_index:
                    enabled = True
                    
                entries.append({"index": i, "title": title, "content": content, "enabled": enabled})
        
        # 确保 current_index 在有效范围内
        if len(entries) > 0:
            current_index = max(0, min(current_index, len(entries) - 1))
            
        # 筛选启用的条目用于列表输出
        filtered_entries = [e for e in entries if e["enabled"]]
        
        # 获取选中项（基于原始索引）
        selected = ""
        selected_title = ""
        if len(entries) > 0 and current_index < len(entries):
            selected = entries[current_index]["content"]
            selected_title = entries[current_index]["title"]
        
        # 如果选中项为空且过滤列表不为空，回退到过滤列表的第一个（兜底逻辑，通常不需要）
        if selected == "" and len(filtered_entries) > 0:
             # 注意：这里的逻辑可能需要根据实际需求调整。
             # 如果选中的确实是空字符串，应该返回空。
             # 但为了保持兼容性，如果selected为空，尝试取第一个启用的非空？
             # 原有逻辑：if selected_title == "": selected = filtered_entries[0]...
             # 我们保持简单：selected就是selected。
             pass

        # 生成输出列表
        for entry in filtered_entries:
            strings_list.append(entry["content"])
            # dict_output_list 包含所有启用的项
            # 这里的enable属性是指是否被当前index选中
            is_selected = (entry["index"] == current_index)
            titles_dict[entry["title"]] = {"prompt": entry["content"], "enable": is_selected}
            dict_output_list.append(json.dumps({entry["title"]: entry["content"]}, ensure_ascii=False))

        # 返回：完整列表(JSON字符串) 与 选中项
        try:
            strings_out = json.dumps(strings_list, ensure_ascii=False)
        except Exception:
            # 兜底，防止非常规字符导致失败
            strings_out = "[]"
        
        # 将strings_out转换为列表
        list_out = []
        try:
            list_out = json.loads(strings_out)
        except Exception:
            list_out = []
        
        # 生成字典输出
        try:
            dict_output = json.dumps(titles_dict, ensure_ascii=False)
        except Exception:
            dict_output = "{}"
        
        # 返回：完整列表(列表)、选中项、选中标题.数量、字典输出、字典输出列表
        return (list_out, selected, selected_title, len(list_out), dict_output, dict_output_list)


class TextDictSplitter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dict_item": ("STRING", {"forceInput": True, "tooltip": "输入的单个字典项(JSON字符串)"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("key", "value")
    FUNCTION = "split_dict"
    CATEGORY = "A_my_nodes/text"

    def split_dict(self, dict_item):
        key = ""
        value = ""
        try:
            # 尝试解析 JSON
            data = json.loads(dict_item)
            if isinstance(data, dict) and len(data) > 0:
                # 获取第一个键值对
                key = list(data.keys())[0]
                value = str(data[key])
        except Exception:
            # 如果解析失败，返回空
            pass
        return (key, value)
