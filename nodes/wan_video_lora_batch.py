import folder_paths
import os

# 定义FlexibleOptionalInputType类，参考rgthree-comfy的实现
class AnyType(str):
    """A special class that is always equal in not equal comparisons."""
    def __ne__(self, __value: object) -> bool:
        return False

class FlexibleOptionalInputType(dict):
    """A special class to make flexible nodes that pass data to our python handlers."""
    
    def __init__(self, type, data=None):
        self.type = type
        self.data = data
        if self.data is not None:
            for k, v in self.data.items():
                self[k] = v

    def __getitem__(self, key):
        if self.data is not None and key in self.data:
            val = self.data[key]
            return val
        return (self.type,)

    def __contains__(self, key):
        return True

any_type = AnyType("*")

class WanVideoLoraBatch:
    """
    WanVideo批量收集多个LoRA，参考LoadLoraBatch的UI但使用WanVideoLoraSelectMulti的输入输出格式。
    前端参数结构为LORA_x（如LORA_1, LORA_2...），每个为dict: {on, lora, strength}
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": FlexibleOptionalInputType(type=any_type, data={
                "prev_lora": ("WANVIDLORA", {"default": None, "tooltip": "For loading multiple LoRAs"}),
                "blocks": ("SELECTEDBLOCKS",),
                # low_mem_load和merge_loras现在由前端控件管理
                # 这里的key由前端动态生成，如LORA_1, LORA_2...
                # 每个value是dict: {"on": bool, "lora": str, "strength": float}
            }),
            "hidden": {},
        }

    RETURN_TYPES = ("WANVIDLORA",)
    RETURN_NAMES = ("lora",)
    FUNCTION = "collect_loras"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "批量收集多个WanVideo LoRA，参考LoadLoraBatch的UI但使用WanVideo格式"

    def collect_loras(self, blocks={}, prev_lora=None, **kwargs):
        """
        遍历所有LORA_x参数，收集LoRA列表。
        """
        # 从kwargs中获取low_mem_load和merge_loras的值
        # 检查是否有settings控件的数据
        settings_data = kwargs.get("settings", {})
        if isinstance(settings_data, dict):
            low_mem_load = settings_data.get("value1", False)
            merge_loras = settings_data.get("value2", True)
        else:
            # 兼容旧格式
            low_mem_load = kwargs.get("low_mem_load", False)
            merge_loras = kwargs.get("merge_loras", True)
        
        print(f"low_mem_load: {low_mem_load}")
        print(f"merge_loras: {merge_loras}")
        print(f"[WanVideoLoraBatch] 开始收集LoRA，参数数量: {len(kwargs)}")
        print(f"[WanVideoLoraBatch] 所有kwargs参数: {list(kwargs.keys())}")
        
        if not merge_loras:
            low_mem_load = False  # Unmerged LoRAs don't need low_mem_load
            
        loras_list = list(prev_lora) if prev_lora else []
        
        for key, value in kwargs.items():
            print(f"[WanVideoLoraBatch] 处理参数: {key} = {value} (类型: {type(value)})")
            
            # 只处理LORA_x参数，且value为dict
            if key.startswith("LORA_") and isinstance(value, dict):
                print(f"[WanVideoLoraBatch] 发现LoRA参数: {key}")
                enabled = value.get("on", True)
                lora_name = value.get("lora", "None")
                strength = value.get("strength", 1.0)
                
                print(f"[WanVideoLoraBatch] LoRA详情 - 启用: {enabled}, 名称: {lora_name}, 强度: {strength}")
                
                if enabled and lora_name and lora_name != "None":
                    s = round(strength, 4)
                    if s == 0.0:
                        print(f"[WanVideoLoraBatch] 跳过强度为0的LoRA: {lora_name}")
                        continue
                        
                    print(f"[WanVideoLoraBatch] 收集LoRA: {lora_name}, 强度: {s}")
                    
                    try:
                        lora_path = folder_paths.get_full_path("loras", lora_name)
                        print(f"[WanVideoLoraBatch] LoRA路径: {lora_path}")
                        loras_list.append({
                            "path": lora_path,
                            "strength": s,
                            "name": lora_name.split(".")[0],
                            "blocks": blocks.get("selected_blocks", {}),
                            "layer_filter": blocks.get("layer_filter", ""),
                            "low_mem_load": low_mem_load,
                            "merge_loras": merge_loras,
                        })
                        print(f"[WanVideoLoraBatch] 成功添加LoRA到列表，当前列表长度: {len(loras_list)}")
                    except Exception as e:
                        print(f"[WanVideoLoraBatch] 加载LoRA失败 {lora_name}: {e}")
                        continue
            else:
                print(f"[WanVideoLoraBatch] 跳过非LoRA参数: {key}")
        
        print(f"[WanVideoLoraBatch] 最终收集到 {len(loras_list)} 个LoRA")
        return (loras_list,) 