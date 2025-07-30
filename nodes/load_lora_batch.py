
import folder_paths
import nodes

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
            "optional": {
                # 这里的key由前端动态生成，如LORA_1, LORA_2...
                # 每个value是dict: {"on": bool, "lora": str, "strength": float}
            },
            "hidden": {},
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_loras"
    CATEGORY = "A_my_nodes/loaders"

    def load_loras(self, model, **kwargs):
        """
        遍历所有LORA_x参数，批量应用到model。
        """
        current_model = model
        for key, value in kwargs.items():
            # 只处理LORA_x参数，且value为dict
            if key.startswith("LORA_") and isinstance(value, dict):
                enabled = value.get("on", True)
                lora_name = value.get("lora", "None")
                strength = value.get("strength", 1.0)
                if enabled and lora_name != "None":
                    print(f"[LoadLoraBatch] 应用LoRA: {lora_name}, 强度: {strength}")
                    current_model, _ = nodes.LoraLoader().load_lora(current_model, None, lora_name, strength, 0)
        return (current_model,)




