class AspectRatioAdjuster:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_width": ("INT", {"default": 1024, "min": 64, "max": 8192}),
                "input_height": ("INT", {"default": 768, "min": 64, "max": 8192}),
                "output_width": ("INT", {"default": 512, "min": 64, "max": 8192}),
                "output_height": ("INT", {"default": 768, "min": 64, "max": 8192}),
                "enable_ratio_adjustment": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    FUNCTION = "adjust_aspect_ratio"
    CATEGORY = "A_my_nodes/math"

    def adjust_aspect_ratio(self, input_width, input_height, output_width, output_height, enable_ratio_adjustment):
        if not enable_ratio_adjustment:
            return (output_width, output_height)
        
        # 计算输入和输出的宽高比
        input_ratio = input_width / input_height
        output_ratio = output_width / output_height
        
        # 如果输入是宽图(ratio > 1)，输出也应该是宽图
        # 如果输入是长图(ratio < 1)，输出也应该是长图
        if (input_ratio > 1 and output_ratio < 1) or (input_ratio < 1 and output_ratio > 1):
            # 交换输出的宽和高
            output_width, output_height = output_height, output_width
            
        return (output_width, output_height)

class I2VConfigureNode:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "output_width": ("INT", {"default": 512, "min": 64, "max": 8192}),
                "output_height": ("INT", {"default": 768, "min": 64, "max": 8192}),
                "length": ("INT", {"default": 33, "min": 1, "max": 100}),
                "steps": ("INT", {"default": 12, "min": 1, "max": 100}),
                "batch_size": ("INT", {"default": 12, "min": 1, "max": 100}),
                "enable_ratio_adjustment": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("INT", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("width", "height", "length", "steps", "batch_size")
    FUNCTION = "adjust_i2v_config"
    CATEGORY = "A_my_nodes/math"

    def adjust_i2v_config(self, images, output_width, output_height, length, steps, batch_size, enable_ratio_adjustment):
        # 从图像张量中获取尺寸信息 (n,h,w,c)
        _, input_height, input_width, _ = images.shape
        
        if not enable_ratio_adjustment:
            return (output_width, output_height, length, steps, batch_size)
        
        # 计算输入和输出的宽高比
        input_ratio = input_width / input_height
        output_ratio = output_width / output_height
        
        # 如果输入是宽图(ratio > 1)，输出也应该是宽图
        # 如果输入是长图(ratio < 1)，输出也应该是长图
        if (input_ratio > 1 and output_ratio < 1) or (input_ratio < 1 and output_ratio > 1):
            # 交换输出的宽和高
            output_width, output_height = output_height, output_width
            
        return (output_width, output_height, length, steps, batch_size)

