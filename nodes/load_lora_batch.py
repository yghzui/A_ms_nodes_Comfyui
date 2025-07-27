
import folder_paths
import nodes
import json

class LoadLoraBatch:
    """
    批量加载多个LoRA的节点,使用与模板相同的原理处理所有LoRA。
    """
    def __init__(self):
        self.lora_loader = nodes.LoraLoader()

    @classmethod
    def INPUT_TYPES(s):
        # 获取所有可用的LoRA文件列表
        loras = ["None"] + folder_paths.get_filename_list("loras")
        
        return {
            "required": {
                "model": ("MODEL",),
            },
            "optional": {
                # 添加一个模板LoRA选项(索引0),这样JavaScript前端可以获取到LoRA列表
                "lora_name_0": (loras,),
                "strength_model_0": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "enabled_0": ("BOOLEAN", {"default": True}),
            },
            "hidden": {
                # 从JS前端接收LoRA配置的JSON字符串
                "lora_json": ("STRING", {"default": "[]"}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_loras"
    CATEGORY = "A_my_nodes/loaders"

    def load_loras(self, model, lora_json="[]", **kwargs):
        """应用所有LoRA到模型"""
        current_model = model
        
        # # 首先应用索引0的LoRA(如果启用且不为"None")
        # if enabled_0 and lora_name_0 != "None":
        #     print(f"[LoadLoraBatch] 应用LoRA(索引0): {lora_name_0}, 强度: {strength_model_0}")
        #     current_model, _ = self.lora_loader.load_lora(current_model, None, lora_name_0, strength_model_0, 0)
        
        try:
            # 解析LoRA配置
            lora_configs = json.loads(lora_json)
            
            # 应用所有启用的LoRA
            for config in lora_configs:
                enabled = config.get("enabled", False)
                lora_name = config.get("lora_name", "None")
                strength_model = config.get("strength_model", 1.0)
                
                if enabled and lora_name != "None":
                    print(f"[LoadLoraBatch] 应用LoRA: {lora_name}, 强度: {strength_model}")
                    current_model, _ = self.lora_loader.load_lora(current_model, None, lora_name, strength_model, 0)
                    
        except json.JSONDecodeError as e:
            print(f"[LoadLoraBatch] 警告: 解析LoRA数据失败: {e}")
        
        return (current_model,)




