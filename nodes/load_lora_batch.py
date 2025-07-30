
import folder_paths
import nodes

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

class LoadLoraBatch:
    """
    批量加载多个LoRA，只作用于model，不涉及clip。
    前端参数结构为LORA_x（如LORA_1, LORA_2...），每个为dict: {on, lora, strength}
    """
    @classmethod
    def INPUT_TYPES(cls):
        # 不预设LORA_1、LORA_2，由前端动态生成
        return {
            "required": {
                "model": ("MODEL",),
            },
            "optional": FlexibleOptionalInputType(type=any_type, data={
                # 这里的key由前端动态生成，如LORA_1, LORA_2...
                # 每个value是dict: {"on": bool, "lora": str, "strength": float}
            }),
            "hidden": {},
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_loras"
    CATEGORY = "A_my_nodes/loaders"

    def load_loras(self, model, **kwargs):
        """
        遍历所有LORA_x参数，批量应用到model。
        """
        print(f"[LoadLoraBatch] 开始加载LoRA，参数数量: {len(kwargs)}")
        print(f"[LoadLoraBatch] 所有kwargs参数: {list(kwargs.keys())}")
        
        current_model = model
        for key, value in kwargs.items():
            print(f"[LoadLoraBatch] 处理参数: {key} = {value} (类型: {type(value)})")
            
            # 只处理LORA_x参数，且value为dict
            if key.startswith("LORA_") and isinstance(value, dict):
                print(f"[LoadLoraBatch] 发现LoRA参数: {key}")
                enabled = value.get("on", True)
                lora_name = value.get("lora", "None")
                strength = value.get("strength", 1.0)
                
                print(f"[LoadLoraBatch] LoRA详情 - 启用: {enabled}, 名称: {lora_name}, 强度: {strength}")
                
                if enabled and lora_name != "None":
                    print(f"[LoadLoraBatch] 应用LoRA: {lora_name}, 强度: {strength}")
                    current_model, _ = nodes.LoraLoader().load_lora(current_model, None, lora_name, strength, 0)
                else:
                    print(f"[LoadLoraBatch] 跳过LoRA: {lora_name} (未启用或名称为None)")
            else:
                print(f"[LoadLoraBatch] 跳过非LoRA参数: {key}")
        
        print(f"[LoadLoraBatch] 完成LoRA加载")
        return (current_model,)




