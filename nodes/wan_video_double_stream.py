import json
import comfy.sd
import comfy.utils
import nodes
from nodes import *
from .lora_common import LoraCondition

class WanVideoDoubleStream(LoraCondition):
    def __init__(self):
        self.loaded_lora = None

    INPUT_TYPES = classmethod(lambda cls: {
        "required": cls.condition_inputs(),
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

    RETURN_TYPES = ("MODEL", "MODEL", "WANVIDLORA", "WANVIDLORA", "STRING")
    RETURN_NAMES = ("model_high", "model_low", "prev_lora_high", "prev_lora_low", "dict_input")
    FUNCTION = "process"
    CATEGORY = "A_my_nodes/video"

    def process_single_stream(self, model, prev_lora, blocks, loras_info, stream_name):
        """
        Processes a single stream of LoRA loading.
        """
        has_model_input = model is not None
        has_prev_lora_input = prev_lora is not None
        should_collect_wanvideo = has_prev_lora_input or not has_model_input
        print(f"[{stream_name}] model is {'✅' if has_model_input else '❌'}.")
        # Create a copy of the list to prevent modifying the original list in place
        # This is crucial when High and Low streams share the same input list
        prev_lora = list(prev_lora) if has_prev_lora_input and prev_lora else []
            
        current_loras = []
        low_mem_load = False
        merge_loras = False
        
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
            if should_collect_wanvideo:
                settings = loras_data_raw.get("settings", {})
                if isinstance(settings, dict):
                    # value2 is Low Mem, value3 is Merge
                    low_mem_load = settings.get("value2", False)
                    merge_loras = settings.get("value3", False)
                
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

        if should_collect_wanvideo and not has_model_input:
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
            if not should_collect_wanvideo:
                continue
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
                if not has_model_input:
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
        is_active, ui_text_status = self.resolve_enable_mode(enable_mode, dict_input, key_to_check, check_mode)

        # UI Feedback
        ui_response = {"ui": {"text": [ui_text_status]}}

        # 2. Process based on state
        if not is_active:
            # Pass through without modification
            return {
                "ui": {"text": [ui_text_status]},
                "result": (model_high, model_low, prev_lora_high, prev_lora_low, dict_input)
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
            "result": (out_model_high, out_model_low, out_lora_high, out_lora_low, dict_input)
        }
