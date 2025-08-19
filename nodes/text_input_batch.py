# -*- coding: utf-8 -*-
# Created by My: 批量字符串输入节点（后端）
# 说明：
# - 提供一个可选的 index 输入（INT），用于选择返回第几个字符串
# - 字符串列表通过前端动态控件聚合为 JSON 写入 strings_json（隐藏/内部容器）
# - 输出：字符串列表（JSON 字符串）与根据 index 选中的字符串；当 index 无效或越界时回退到第一个；当列表为空返回空串

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

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("strings", "selected")
    FUNCTION = "aggregate_strings"
    CATEGORY = "A_my_nodes/text"

    def aggregate_strings(self, index=0, strings_json="[]"):
        # 健壮解析 JSON
        try:
            data = json.loads(strings_json) if isinstance(strings_json, str) else []
        except Exception:
            data = []

        # 仅保留字符串类型，并保持顺序
        strings_list = [str(x) for x in data if isinstance(x, (str, int, float)) or x is None]
        # 将 None 转为空串，数字转字符串
        strings_list = ["" if x is None else str(x) for x in strings_list]

        # 计算选中值
        selected = ""
        if len(strings_list) > 0:
            try:
                i = int(index) if index is not None else 0
            except Exception:
                i = 0
            if i < 0 or i >= len(strings_list):
                i = 0
            selected = strings_list[i]

        # 返回：完整列表(JSON字符串) 与 选中项
        try:
            strings_out = json.dumps(strings_list, ensure_ascii=False)
        except Exception:
            # 兜底，防止非常规字符导致失败
            strings_out = "[]"

        return (strings_out, selected)
