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
            "optional": {},
            "hidden": {
                # 字符串列表容器（前端写入，后端解析）；隐藏以避免未加载前端脚本时出现可见输入框
                "strings_json": ("STRING", {"default": "[]"}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "INT", "STRING", "STRING")
    RETURN_NAMES = ("strings", "selected", "count", "dict_output", "selected_title")
    OUTPUT_IS_LIST = (True, False, False, False, False)
    FUNCTION = "aggregate_strings"
    CATEGORY = "A_my_nodes/text"

    def aggregate_strings(self, index=0, strings_json="[]"):
        # 健壮解析 JSON
        try:
            data = json.loads(strings_json) if isinstance(strings_json, str) else []
        except Exception:
            data = []

        # {{ AURA-X: Modify - 修改启用状态逻辑，确保只有当前选中索引对应项目启用. }}
        # 处理新的数据结构，支持标题和启用状态
        strings_list = []
        titles_dict = {}
        
        # 获取当前选中索引
        try:
            current_index = int(index) if index is not None else 0
        except Exception:
            current_index = 0
        
        if isinstance(data, list):
            # 兼容旧版本数据结构
            for i, item in enumerate(data):
                if isinstance(item, dict) and "title" in item and "content" in item:
                    # 新版本数据结构
                    title = item.get("title", f"prompt_{i}")
                    content = str(item.get("content", ""))
                    # 启用状态：只有当前选中索引对应的项目为True
                    enabled = (i == current_index)
                    strings_list.append(content)
                    titles_dict[title] = {"prompt": content, "enable": enabled}
                else:
                    # 旧版本数据结构，自动生成标题
                    content = str(item) if item is not None else ""
                    title = f"prompt_{i}"
                    # 启用状态：只有当前选中索引对应的项目为True
                    enabled = (i == current_index)
                    strings_list.append(content)
                    titles_dict[title] = {"prompt": content, "enable": enabled}

        # 计算选中值和选中标题
        selected = ""
        selected_title = ""
        if len(strings_list) > 0:
            # 确保索引在有效范围内
            if current_index < 0 or current_index >= len(strings_list):
                current_index = 0
            selected = strings_list[current_index]
            # 获取对应的标题
            title_keys = list(titles_dict.keys())
            if current_index < len(title_keys):
                selected_title = title_keys[current_index]

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
        
        # 返回：完整列表(列表)、选中项、数量、字典输出、选中标题
        return (list_out, selected, len(list_out), dict_output, selected_title)
