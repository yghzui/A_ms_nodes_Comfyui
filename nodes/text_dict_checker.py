# -*- coding: utf-8 -*-
# Created by My: 文本字典检查节点
# 说明：
# - 输入一个字典（JSON字符串）和一个字符串
# - 判断字符串是否在字典的key中
# - 如果不在返回false，如果在根据enable字段返回对应值

import json


class TextDictChecker:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dict_input": ("STRING", {"default": "{}", "tooltip": "输入包含标题、内容、启用状态的字典"}),
                "key_to_check": ("STRING", {"default": "", "tooltip": "要检查的字符串key"}),
                "check_mode": (["absolute", "start_with", "contains"], {"default": "absolute", "tooltip": "匹配模式"}),
            },
        }

    RETURN_TYPES = ("BOOLEAN", "STRING", "BOOLEAN")
    RETURN_NAMES = ("key_exists", "prompt_content", "is_enabled")
    FUNCTION = "check_dict_key"
    CATEGORY = "A_my_nodes/text"

    def check_dict_key(self, dict_input="{}", key_to_check="", check_mode="absolute"):
        # 解析字典输入
        try:
            data_dict = json.loads(dict_input) if isinstance(dict_input, str) else {}
        except Exception:
            data_dict = {}

        # 空key直接返回False
        if not isinstance(key_to_check, str) or key_to_check == "":
            return (False, "", False)

        # 根据匹配模式查找key
        matched_key = None
        if check_mode == "absolute":
            if key_to_check in data_dict:
                matched_key = key_to_check
        elif check_mode == "start_with":
            for k in data_dict.keys():
                if isinstance(k, str) and k.startswith(key_to_check):
                    matched_key = k
                    break
        elif check_mode == "contains":
            for k in data_dict.keys():
                if isinstance(k, str) and key_to_check in k:
                    matched_key = k
                    break
        else:
            if key_to_check in data_dict:
                matched_key = key_to_check

        if matched_key is None:
            return (False, "", False)

        # key存在，获取对应的值
        item = data_dict[matched_key]
        
        # 确保item是字典格式
        if not isinstance(item, dict):
            return (True, str(item), True)
        
        # 获取prompt内容和enable状态
        prompt_content = item.get("prompt", "")
        is_enabled = item.get("enable", True)
        
        return (True, prompt_content, is_enabled)


