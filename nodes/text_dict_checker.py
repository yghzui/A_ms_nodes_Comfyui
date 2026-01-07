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
                "check_mode": (["absolute", "start_with", "contains", "regex", "absolute_invert", "start_with_invert", "contains_invert", "regex_invert"], {"default": "absolute", "tooltip": "匹配模式"}),
            },
        }

    RETURN_TYPES = ("BOOLEAN", "STRING", "BOOLEAN", "STRING")
    RETURN_NAMES = ("key_exists", "prompt_content", "is_enabled", "dict_input")
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
            return {"ui": {"text": ["False"]}, "result": (False, "", False, dict_input)}

        # Determine if it is an inverted mode
        is_invert = check_mode.endswith("_invert")
        base_mode = check_mode.replace("_invert", "")

        # 2. 如果是字典，执行原有逻辑
        if is_json_dict:
            matched_keys = []
            
            # Support multiple keys for dictionary mode too (was not fully supported in previous implementation if using semicolon?)
            # Previous implementation seemed to only split for string matching, not dictionary keys except maybe regex?
            # Wait, looking at previous code, dictionary logic:
            # if check_mode == "absolute": if key_to_check in data_dict...
            # It treated key_to_check as a SINGLE key.
            # Now we should support multiple keys as well.
            
            if base_mode == "regex":
                check_keys = [key_to_check]
            else:
                check_keys = [k.strip() for k in key_to_check.split(';') if k.strip()]
            
            all_dict_keys = [k for k in data_dict.keys() if isinstance(k, str)]

            for check_key in check_keys:
                if base_mode == "absolute":
                    if check_key in data_dict:
                        matched_keys.append(check_key)
                elif base_mode == "start_with":
                    matched_keys.extend([k for k in all_dict_keys if k.startswith(check_key)])
                elif base_mode == "contains":
                    matched_keys.extend([k for k in all_dict_keys if check_key in k])
                elif base_mode == "regex":
                    try:
                        pattern = re.compile(check_key)
                        matched_keys.extend([k for k in all_dict_keys if pattern.search(k)])
                    except re.error:
                        print(f"TextDictChecker: Invalid regex pattern '{check_key}'")
            
            matched_keys = list(set(matched_keys))

            # Invert Logic
            if is_invert:
                if matched_keys:
                    # Match found -> False
                    return {"ui": {"text": ["False"]}, "result": (False, "", False, dict_input)}
                else:
                    # No match found -> True
                    return {"ui": {"text": ["True"]}, "result": (True, str(dict_input), True, dict_input)}
            else:
                # Normal Logic
                if not matched_keys:
                    return {"ui": {"text": ["False"]}, "result": (False, "", False, dict_input)}

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
                    return {"ui": {"text": ["True"]}, "result": (True, str(item), True, dict_input)}
                prompt_content = item.get("prompt", "")
                is_enabled = item.get("enable", True)
                ui_text = "True" if is_enabled else "False"
                return {"ui": {"text": [ui_text]}, "result": (True, prompt_content, is_enabled, dict_input)}
        
        # 3. 如果不是字典（普通字符串），执行字符串匹配逻辑
        else:
            input_str = str(dict_input)
            matched = False
            
            # 支持多个关键字，以分号分隔
            if base_mode == "regex":
                keys = [key_to_check]
            else:
                keys = [k.strip() for k in key_to_check.split(';') if k.strip()]
            
            for key in keys:
                if base_mode == "absolute":
                    if input_str == key:
                        matched = True
                        break
                elif base_mode == "start_with":
                    if input_str.startswith(key):
                        matched = True
                        break
                elif base_mode == "contains":
                    if key in input_str:
                        matched = True
                        break
                elif base_mode == "regex":
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
            
            # Apply Invert Logic
            if is_invert:
                final_state = not matched
            else:
                final_state = matched

            if final_state:
                return {"ui": {"text": ["True"]}, "result": (True, input_str, True, dict_input)}
            else:
                return {"ui": {"text": ["False"]}, "result": (False, "", False, dict_input)}
