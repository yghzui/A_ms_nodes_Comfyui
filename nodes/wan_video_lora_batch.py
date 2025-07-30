import folder_paths
import os

class WanVideoLoraBatch:
    """
    WanVideo批量收集多个LoRA，参考LoadLoraBatch的UI但使用WanVideoLoraSelectMulti的输入输出格式。
    前端参数结构为LORA_x（如LORA_1, LORA_2...），每个为dict: {on, lora, strength}
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "prev_lora": ("WANVIDLORA", {"default": None, "tooltip": "For loading multiple LoRAs"}),
                "blocks": ("SELECTEDBLOCKS",),
                "low_mem_load": ("BOOLEAN", {"default": False, "tooltip": "Load the LORA model with less VRAM usage, slower loading. No effect if merge_loras is False"}),
                "merge_loras": ("BOOLEAN", {"default": True, "tooltip": "Merge LoRAs into the model, otherwise they are loaded on the fly. Always enabled for GGUF and scaled fp8 models. This affects ALL LoRAs, not just the current one"}),
                # 这里的key由前端动态生成，如LORA_1, LORA_2...
                # 每个value是dict: {"on": bool, "lora": str, "strength": float}
            },
            "hidden": {},
        }

    RETURN_TYPES = ("WANVIDLORA",)
    RETURN_NAMES = ("lora",)
    FUNCTION = "collect_loras"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "批量收集多个WanVideo LoRA，参考LoadLoraBatch的UI但使用WanVideo格式"

    def collect_loras(self, blocks={}, prev_lora=None, low_mem_load=False, merge_loras=True, **kwargs):
        """
        遍历所有LORA_x参数，收集LoRA列表。
        """
        if not merge_loras:
            low_mem_load = False  # Unmerged LoRAs don't need low_mem_load
            
        loras_list = list(prev_lora) if prev_lora else []
        
        for key, value in kwargs.items():
            # 只处理LORA_x参数，且value为dict
            if key.startswith("LORA_") and isinstance(value, dict):
                enabled = value.get("on", True)
                lora_name = value.get("lora", "None")
                strength = value.get("strength", 1.0)
                
                if enabled and lora_name and lora_name != "None":
                    s = round(strength, 4)
                    if s == 0.0:
                        continue
                        
                    print(f"[WanVideoLoraBatch] 收集LoRA: {lora_name}, 强度: {s}")
                    
                    try:
                        lora_path = folder_paths.get_full_path("loras", lora_name)
                        loras_list.append({
                            "path": lora_path,
                            "strength": s,
                            "name": lora_name.split(".")[0],
                            "blocks": blocks.get("selected_blocks", {}),
                            "layer_filter": blocks.get("layer_filter", ""),
                            "low_mem_load": low_mem_load,
                            "merge_loras": merge_loras,
                        })
                    except Exception as e:
                        print(f"[WanVideoLoraBatch] 加载LoRA失败 {lora_name}: {e}")
                        continue
        
        return (loras_list,) 