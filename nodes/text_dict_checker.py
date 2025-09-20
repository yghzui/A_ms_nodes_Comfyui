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
            },
        }

    RETURN_TYPES = ("BOOLEAN", "STRING", "BOOLEAN")
    RETURN_NAMES = ("key_exists", "prompt_content", "is_enabled")
    FUNCTION = "check_dict_key"
    CATEGORY = "A_my_nodes/text"

    def check_dict_key(self, dict_input="{}", key_to_check=""):
        # 解析字典输入
        try:
            data_dict = json.loads(dict_input) if isinstance(dict_input, str) else {}
        except Exception:
            data_dict = {}

        # 检查key是否存在
        key_exists = key_to_check in data_dict
        
        if not key_exists:
            # key不存在，返回False
            return (False, "", False)
        
        # key存在，获取对应的值
        item = data_dict[key_to_check]
        
        # 确保item是字典格式
        if not isinstance(item, dict):
            return (True, str(item), True)
        
        # 获取prompt内容和enable状态
        prompt_content = item.get("prompt", "")
        is_enabled = item.get("enable", True)
        
        return (key_exists, prompt_content, is_enabled)


# 注册节点
NODE_CLASS_MAPPINGS = {
    "TextDictChecker": TextDictChecker
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TextDictChecker": "Text Dict Checker"
}