
import folder_paths
import nodes
import json

class LoadLoraBatch:
    """
    批量加载多个LoRA的节点,通过在节点上直接放置控件提供友好的用户界面。
    Python端只创建一个固定的模板,所有UI控件由前端JavaScript动态创建。
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
                # 创建一个固定的模板,供UI使用
                "lora_name": (loras,),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "enabled": ("BOOLEAN", {"default": True}),
            },
            "hidden": {
                # 这个隐藏输入会从JS前端接收一个包含所有LoRA配置的JSON字符串
                "lora_batch_data": ("STRING", {"default": "[]"}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_loras"
    CATEGORY = "A_my_nodes/loaders"

    def apply_loras(self, model, lora_name, strength_model, enabled, lora_batch_data="[]"):
        """
        应用所有启用的LoRA到模型
        """
        current_model = model
        
        # 首先应用主模板中的LoRA
        if enabled and lora_name != "None":
            print(f"[LoadLoraBatch] 应用主LoRA: {lora_name}, 强度: {strength_model}")
            current_model, _ = self.lora_loader.load_lora(current_model, None, lora_name, strength_model, 0)
        
        try:
            # 解析从前端传来的额外LoRA配置
            additional_loras = json.loads(lora_batch_data)
            
            # 应用所有额外的LoRA
            for lora_config in additional_loras:
                lora_enabled = lora_config.get("enabled", False)
                lora_name = lora_config.get("lora_name", "None")
                lora_strength = lora_config.get("strength_model", 1.0)
                
                if lora_enabled and lora_name != "None":
                    print(f"[LoadLoraBatch] 应用额外LoRA: {lora_name}, 强度: {lora_strength}")
                    current_model, _ = self.lora_loader.load_lora(current_model, None, lora_name, lora_strength, 0)
                    
        except json.JSONDecodeError as e:
            print(f"[LoadLoraBatch] 警告: 解析额外LoRA数据失败: {e}")
        
        return (current_model,)




