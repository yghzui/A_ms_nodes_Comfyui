
import folder_paths
import nodes  # 导入原始nodes模块

class LoadLoraBatch:
    """
    一个可以批量加载多个LoRA的节点，并能动态调整UI。
    本节点通过调用原生的LoraLoader来加载LoRA，以保持兼容性。
    此版本只接受并输出MODEL。
    """
    MAX_LORAS = 10  # 定义最大LoRA数量

    def __init__(self):
        # 实例化一个原生的LoraLoader，以便调用它的加载方法和利用其缓存机制
        self.lora_loader = nodes.LoraLoader()

    @classmethod
    def INPUT_TYPES(s):
        loras = ["None"] + folder_paths.get_filename_list("loras")
        
        required_inputs = {
            "model": ("MODEL",),
            "clip": ("CLIP",)
        }
        
        optional_inputs = {}
        for i in range(1, s.MAX_LORAS + 1):
            optional_inputs[f"enabled_{i}"] = ("BOOLEAN", {"default": True, "label_on": "启用", "label_off": "禁用"})
            optional_inputs[f"lora_name_{i}"] = (loras,)
            optional_inputs[f"strength_model_{i}"] = ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01})

        return {"required": required_inputs, "optional": optional_inputs}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_lora_batch"
    CATEGORY = "A_my_nodes/loaders"

    def load_lora_batch(self, model, **kwargs):
        """
        按顺序加载并应用所有启用的LoRA。
        """
        current_model = model

        for i in range(1, self.MAX_LORAS + 1):
            enabled = kwargs.get(f"enabled_{i}", True)
            lora_name = kwargs.get(f"lora_name_{i}", "None")
            strength_model = kwargs.get(f"strength_model_{i}", 1.0)
            
            if enabled and lora_name != "None":
                # 调用原生LoraLoader的load_lora方法，但只传入model，模拟LoraLoaderModelOnly的行为
                print(f"正在通过原生LoraLoader加载 LoRA: {lora_name}, model_strength: {strength_model}")
                # load_lora需要一个clip参数，我们传入None。strength_clip设为0。
                # 它返回(model, clip)，我们只关心model
                current_model, _ = self.lora_loader.load_lora(current_model, None, lora_name, strength_model, 0)
        
        return (current_model,)




