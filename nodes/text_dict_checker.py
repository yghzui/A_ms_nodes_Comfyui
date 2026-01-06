# -*- coding: utf-8 -*-
# Created by My: 文本字典检查节点
# 说明：
# - 输入一个字典（JSON字符串）或字符串（直接检查内容）和一个字符串
# - 判断字符串是否在字典的key中，或是否与输入字符串匹配
# - 如果不在返回false，如果在根据enable字段返回对应值

import json
import re


class TextDictChecker:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dict_input": ("STRING", {"default": "{}", "tooltip": "输入包含标题、内容、启用状态的字典，或直接输入字符串"}),
                "key_to_check": ("STRING", {"default": "", "multiline": True, "tooltip": "要检查的字符串key"}),
                "check_mode": (["absolute", "start_with", "contains", "regex"], {"default": "absolute", "tooltip": "匹配模式"}),
            },
        }

    RETURN_TYPES = ("BOOLEAN", "STRING", "BOOLEAN")
    RETURN_NAMES = ("key_exists", "prompt_content", "is_enabled")
    FUNCTION = "check_dict_key"
    CATEGORY = "A_my_nodes/text"

    def check_dict_key(self, dict_input="{}", key_to_check="", check_mode="absolute"):
        # 1. 尝试解析 JSON 字典
        is_json_dict = False
        data_dict = {}
        
        try:
            parsed = json.loads(dict_input) if isinstance(dict_input, str) else {}
            if isinstance(parsed, dict):
                data_dict = parsed
                is_json_dict = True
        except Exception:
            pass

        if not isinstance(key_to_check, str) or key_to_check == "":
            return {"ui": {"text": ["False"]}, "result": (False, "", False)}

        # 2. 如果是字典，执行原有逻辑
        if is_json_dict:
            matched_keys = []
            if check_mode == "absolute":
                if key_to_check in data_dict:
                    matched_keys = [key_to_check]
            elif check_mode == "start_with":
                matched_keys = [k for k in data_dict.keys() if isinstance(k, str) and k.startswith(key_to_check)]
            elif check_mode == "contains":
                matched_keys = [k for k in data_dict.keys() if isinstance(k, str) and key_to_check in k]
            elif check_mode == "regex":
                try:
                    pattern = re.compile(key_to_check)
                    matched_keys = [k for k in data_dict.keys() if isinstance(k, str) and pattern.search(k)]
                except re.error:
                    print(f"TextDictChecker: Invalid regex pattern '{key_to_check}'")
                    matched_keys = []
            else:
                if key_to_check in data_dict:
                    matched_keys = [key_to_check]

            if not matched_keys:
                return {"ui": {"text": ["False"]}, "result": (False, "", False)}

            chosen_key = None
            first_key = matched_keys[0]
            for k in matched_keys:
                item = data_dict[k]
                if isinstance(item, dict):
                    if item.get("enable", True):
                        chosen_key = k
                        break
                else:
                    chosen_key = k
                    break
            if chosen_key is None:
                chosen_key = first_key

            item = data_dict[chosen_key]
            if not isinstance(item, dict):
                return {"ui": {"text": ["True"]}, "result": (True, str(item), True)}
            prompt_content = item.get("prompt", "")
            is_enabled = item.get("enable", True)
            ui_text = "True" if is_enabled else "False"
            return {"ui": {"text": [ui_text]}, "result": (True, prompt_content, is_enabled)}
        
        # 3. 如果不是字典（普通字符串），执行字符串匹配逻辑
        else:
            input_str = str(dict_input)
            matched = False
            
            # 支持多个关键字，以分号分隔
            # 如果是正则模式，不进行分割，直接将整个字符串作为正则模式
            if check_mode == "regex":
                keys = [key_to_check]
            else:
                keys = [k.strip() for k in key_to_check.split(';') if k.strip()]
            
            for key in keys:
                if check_mode == "absolute":
                    if input_str == key:
                        matched = True
                        break
                elif check_mode == "start_with":
                    if input_str.startswith(key):
                        matched = True
                        break
                elif check_mode == "contains":
                    if key in input_str:
                        matched = True
                        break
                elif check_mode == "regex":
                    try:
                        if re.search(key, input_str):
                            matched = True
                            break
                    except re.error:
                        print(f"TextDictChecker: Invalid regex pattern '{key}'")
                else:
                    if input_str == key:
                        matched = True
                        break
            
            if matched:
                return {"ui": {"text": ["True"]}, "result": (True, input_str, True)}
            else:
                return {"ui": {"text": ["False"]}, "result": (False, "", False)}
