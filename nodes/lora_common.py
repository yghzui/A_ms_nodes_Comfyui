import json
import re


class LoraCondition:
    @staticmethod
    def condition_inputs():
        return {
            "enable_mode": (["Auto", True, False], {"default": "Auto"}),
            "dict_input": ("STRING", {"default": "{}", "tooltip": "输入包含标题、内容、启用状态的字典，或直接输入字符串"}),
            "key_to_check": ("STRING", {"default": "judgment word", "multiline": True, "tooltip": "要检查的字符串key"}),
            "check_mode": (["absolute", "start_with", "contains", "regex", "absolute_invert", "start_with_invert", "contains_invert", "regex_invert"], {"default": "contains", "tooltip": "匹配模式"}),
        }

    def check_dict_key_logic(self, dict_input="{}", key_to_check="judgment word", check_mode="contains"):
        """
        Reused logic from TextDictChecker to determine if enabled.
        Returns: (is_enabled, prompt_content, ui_text)
        """
        # 1. Try parsing JSON
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
            return False, "", "False"

        # Determine if it is an inverted mode
        is_invert = check_mode.endswith("_invert")
        base_mode = check_mode.replace("_invert", "")

        # 2. Dictionary logic
        if is_json_dict:
            matched_keys = []

            # Support multiple keys (semicolon separated) unless regex
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
                        print(f"LoraCondition: Invalid regex pattern '{check_key}'")

            # Remove duplicates
            matched_keys = list(set(matched_keys))

            # Invert Logic for Dictionary Mode:
            # If invert is True:
            #   - If matches found -> return False (Disabled)
            #   - If NO matches found -> return True (Enabled) - But what content?
            # Wait, standard logic:
            #   - If matches found -> Check 'enable' field of the matched item.
            #   - If NO matches found -> return False.

            # Invert Logic Definition:
            #   - If matches found -> Force False (Disabled).
            #   - If NO matches found -> Force True (Enabled).
            #   However, if no matches found, we don't have a "value" to return (prompt_content).
            #   So usually invert means: "If match found, result is FALSE. If match NOT found, result is TRUE".
            #   But for dictionary, we need to return content.
            #   Let's stick to simple logic:
            #   Normal: Match -> True (if item.enable=True)
            #   Invert: Match -> False (regardless of item.enable). No Match -> True?
            #   Let's assume Invert means: If the condition is met (key exists), we disable.

            if is_invert:
                if matched_keys:
                    # Found keys matching the condition -> Disabled
                    return False, "", "False"
                else:
                    # No keys found matching the condition -> Enabled
                    # But what to return as content? Just return input string or empty?
                    return True, str(dict_input), "True"
            else:
                # Normal Mode
                if not matched_keys:
                    return False, "", "False"

                chosen_key = None
                first_key = matched_keys[0]

                # Priority: Find first item that has enable=True
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
                    return True, str(item), "True"

                prompt_content = item.get("prompt", "")
                is_enabled = item.get("enable", True)
                ui_text = "True" if is_enabled else "False"
                return is_enabled, prompt_content, ui_text

        # 3. String matching logic
        else:
            input_str = str(dict_input)
            matched = False

            if base_mode == "regex":
                keys = [key_to_check]
            else:
                keys = [k.strip() for k in key_to_check.split(';') if k.strip()]

            for key in keys:
                if base_mode == "absolute":
                    if input_str == key: matched = True; break
                elif base_mode == "start_with":
                    if input_str.startswith(key): matched = True; break
                elif base_mode == "contains":
                    if key in input_str: matched = True; break
                elif base_mode == "regex":
                    try:
                        if re.search(key, input_str): matched = True; break
                    except re.error:
                        print(f"LoraCondition: Invalid regex pattern '{key}'")

            # Apply Invert Logic
            if is_invert:
                final_state = not matched
            else:
                final_state = matched

            if final_state:
                return True, input_str, "True"
            else:
                return False, "", "False"

    def resolve_enable_mode(self, enable_mode, dict_input, key_to_check, check_mode):
        is_active = False
        ui_text_status = "False"

        if enable_mode is True or enable_mode in ["True", "true", "Force True"]:
            is_active = True
            ui_text_status = "True"
        elif enable_mode is False or enable_mode in ["False", "false", "Force False"]:
            is_active = False
            ui_text_status = "False"
        else: # Auto
            is_active, _, ui_text_status = self.check_dict_key_logic(dict_input, key_to_check, check_mode)

        return is_active, ui_text_status
