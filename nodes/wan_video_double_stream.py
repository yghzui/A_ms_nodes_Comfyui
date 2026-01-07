import json
import re
import comfy.sd
import comfy.utils
from nodes import *

class WanVideoDoubleStream:
    def __init__(self):
        self.loaded_lora = None

    INPUT_TYPES = classmethod(lambda cls: {
        "required": {
            "enable_mode": (["Auto", "Force True", "Force False"], {"default": "Auto"}),
            "dict_input": ("STRING", {"default": "{}", "tooltip": "输入包含标题、内容、启用状态的字典，或直接输入字符串"}),
            "key_to_check": ("STRING", {"default": "wan_video", "multiline": True, "tooltip": "要检查的字符串key"}),
            "check_mode": (["absolute", "start_with", "contains", "regex"], {"default": "absolute", "tooltip": "匹配模式"}),
        },
        "optional": {
            "model_high": ("MODEL",),
            "model_low": ("MODEL",),
            "prev_lora_high": ("WANVIDLORA",),
            "prev_lora_low": ("WANVIDLORA",),
            "blocks_high": ("SELECTEDBLOCKS",),
            "blocks_low": ("SELECTEDBLOCKS",),
            "loras_info_high": ("STRING", {"default": "[]", "multiline": False, "hidden": True}),
            "loras_info_low": ("STRING", {"default": "[]", "multiline": False, "hidden": True}),
        }
    })

    RETURN_TYPES = ("MODEL", "WANVIDLORA", "MODEL", "WANVIDLORA")
    RETURN_NAMES = ("model_high", "prev_lora_high", "model_low", "prev_lora_low")
    FUNCTION = "process"
    CATEGORY = "A_my_nodes/video"

    def check_dict_key_logic(self, dict_input="{}", key_to_check="", check_mode="absolute"):
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

        # 2. Dictionary logic
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
                    print(f"WanVideoDoubleStream: Invalid regex pattern '{key_to_check}'")
                    matched_keys = []
            else:
                if key_to_check in data_dict:
                    matched_keys = [key_to_check]

            if not matched_keys:
                return False, "", "False"

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
                return True, str(item), "True"
            
            prompt_content = item.get("prompt", "")
            is_enabled = item.get("enable", True)
            ui_text = "True" if is_enabled else "False"
            return is_enabled, prompt_content, ui_text
        
        # 3. String matching logic
        else:
            input_str = str(dict_input)
            matched = False
            
            if check_mode == "regex":
                keys = [key_to_check]
            else:
                keys = [k.strip() for k in key_to_check.split(';') if k.strip()]
            
            for key in keys:
                if check_mode == "absolute":
                    if input_str == key: matched = True; break
                elif check_mode == "start_with":
                    if input_str.startswith(key): matched = True; break
                elif check_mode == "contains":
                    if key in input_str: matched = True; break
                elif check_mode == "regex":
                    try:
                        if re.search(key, input_str): matched = True; break
                    except re.error:
                        print(f"WanVideoDoubleStream: Invalid regex pattern '{key}'")
            
            if matched:
                return True, input_str, "True"
            else:
                return False, "", "False"

    def process_single_stream(self, model, prev_lora, blocks, loras_info, stream_name):
        """
        Processes a single stream of LoRA loading.
        """
        if prev_lora is None:
            prev_lora = []
            
        current_loras = []
        low_mem_load = False
        merge_loras = True
        
        try:
            loras_data_raw = json.loads(loras_info)
        except Exception as e:
            print(f"[{stream_name}] Error parsing loras_info: {e}")
            loras_data_raw = []

        # Handle different data structures
        if isinstance(loras_data_raw, dict):
            if "loras" in loras_data_raw:
                current_loras_list = loras_data_raw["loras"]
            else:
                current_loras_list = []
            
            # Parse settings
            settings = loras_data_raw.get("settings", {})
            if isinstance(settings, dict):
                # value2 is Low Mem, value3 is Merge
                low_mem_load = settings.get("value2", False)
                merge_loras = settings.get("value3", True)
                
        elif isinstance(loras_data_raw, list):
            current_loras_list = loras_data_raw
        else:
            current_loras_list = []

        # Filter enabled LoRAs
        for lora in current_loras_list:
            if lora.get("on", False):
                current_loras.append(lora)
        
        # If not merging, force low_mem to false as per logic in load_lora_merge.py
        if not merge_loras:
            low_mem_load = False
            
        print(f"[{stream_name}] Settings: low_mem_load={'✅' if low_mem_load else '❌'}, merge_loras={'✅' if merge_loras else '❌'}")
        
        if not current_loras:
            return (model, prev_lora)

        # Output model to be modified (if provided)
        output_model = model

        # Load LoRAs
        for lora in current_loras:
            lora_name = lora.get("lora")
            strength = lora.get("strength", 1.0)
            
            if not lora_name or lora_name == "None" or strength == 0.0:
                continue

            # --- Part 1: Apply LoRA to model (if model is provided) ---
            if output_model is not None:
                try:
                    output_model, _ = nodes.LoraLoader().load_lora(output_model, None, lora_name, strength, 0)
                    print(f"[{stream_name}] 💡Applied LoRA to model: {lora_name}")
                except Exception as e:
                    print(f"[{stream_name}] ❌ Failed to apply LoRA to model: {lora_name}, {e}")
                    # Continue to collect for WanVideo even if model application fails? 
                    # Usually yes, as they might be used separately.

            # --- Part 2: Collect LoRA for WANVIDLORA output ---
            try:
                lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
                prev_lora.append({
                    "path": lora_path,
                    "strength": round(strength, 4) if not isinstance(strength, list) else strength,
                    "name": os.path.splitext(lora_name)[0],
                    "blocks": blocks.get("selected_blocks", {}) if blocks else {},
                    "layer_filter": blocks.get("layer_filter", "") if blocks else "",
                    "low_mem_load": low_mem_load,
                    "merge_loras": merge_loras,
                })
                print(f"[{stream_name}] 💡Collected LoRA for WanVideo: {lora_name}")
            except Exception as e:
                print(f"[{stream_name}] ❌ Failed to find LoRA path: {lora_name}, {e}")

        return (output_model, prev_lora)

    def process(self, enable_mode, dict_input, key_to_check, check_mode, 
                model_high=None, model_low=None, 
                prev_lora_high=None, prev_lora_low=None,
                blocks_high={}, blocks_low={},
                loras_info_high="[]", loras_info_low="[]"):
        
        # 1. Determine Enable State
        is_active = False
        ui_text_status = "False"
        
        if enable_mode == "Force True":
            is_active = True
            ui_text_status = "Force True"
            print(f"UI Status Force True")
        elif enable_mode == "Force False":
            is_active = False
            ui_text_status = "Force False"
            print(f"UI Status Force False")
        else: # Auto
            is_active, _, ui_text_status = self.check_dict_key_logic(dict_input, key_to_check, check_mode)
            print(f"Auto Enable State: {is_active}, UI Status: {ui_text_status}")

        # UI Feedback
        ui_response = {"ui": {"text": [ui_text_status]}}

        # 2. Process based on state
        if not is_active:
            # Pass through without modification
            return {
                "ui": {"text": [ui_text_status]},
                "result": (model_high, prev_lora_high, model_low, prev_lora_low)
            }

        # 3. Process High Stream
        out_model_high, out_lora_high = self.process_single_stream(
            model_high, prev_lora_high, blocks_high, loras_info_high, "High"
        )

        # 4. Process Low Stream
        out_model_low, out_lora_low = self.process_single_stream(
            model_low, prev_lora_low, blocks_low, loras_info_low, "Low"
        )

        # Note: ComfyUI expects return values to match RETURN_TYPES. 
        # The dictionary in the first element is for UI update (frontend).
        # We need to ensure we return the tuple correctly.
        # Actually, standard nodes return a tuple of values. 
        # To return UI update, we return a dictionary with "ui" key AND "result" key containing the tuple.
        
        return {
            "ui": {"text": [ui_text_status]}, 
            "result": (out_model_high, out_lora_high, out_model_low, out_lora_low)
        }
