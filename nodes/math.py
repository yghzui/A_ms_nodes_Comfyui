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
                "enable_ratio_adjustment": ("BOOLEAN", {"default": True, "tooltip": "自动交换输出的宽高以匹配输入图像的宽高比(横向/纵向)"}),
                "output_width": ("INT", {"default": 512, "min": 64, "max": 8192, "tooltip": "输出视频的宽度"}),
                "output_height": ("INT", {"default": 768, "min": 64, "max": 8192, "tooltip": "输出视频的高度"}),
                "length": ("INT", {"default": 33, "min": 1, "max": 8192, "tooltip": "生成视频的总帧数。如果启用了'使用秒数控制长度'，此项将被覆盖"}),
                "steps": ("INT", {"default": 12, "min": 1, "max": 100, "tooltip": "采样步数"}),
                "batch_size": ("INT", {"default": 12, "min": 1, "max": 100, "tooltip": "批处理大小"}),
                "use_seconds_for_length": ("BOOLEAN", {"default": False, "tooltip": "如果启用，总帧数将根据 '秒数' * '帧率' + 1 计算得出"}), # 控制是否使用秒数计算length
                "seconds": ("INT", {"default": 2, "min": 1, "max": 1000, "tooltip": "视频时长（秒）"}), # 秒数
                "fps": ("INT", {"default": 16, "min": 1, "max": 100, "tooltip": "每秒帧数"}), # 帧率
            }
        }

    RETURN_TYPES = ("INT", "INT", "INT", "INT", "INT", "INT", "FLOAT")
    RETURN_NAMES = ("width", "height", "length", "steps", "batch_size", "seconds", "fps")
    FUNCTION = "adjust_i2v_config"
    CATEGORY = "A_my_nodes/math"

    def adjust_i2v_config(self, images, enable_ratio_adjustment, output_width, output_height, length, steps, batch_size, use_seconds_for_length, seconds, fps):
        # 如果启用秒数控制, 则根据秒数和帧率计算总帧数
        if use_seconds_for_length:
            length = seconds * fps + 1

        # 从图像张量中获取尺寸信息 (n,h,w,c)
        _, input_height, input_width, _ = images.shape
        
        if not enable_ratio_adjustment:
            return (output_width, output_height, length, steps, batch_size, seconds, float(fps))
        
        # 计算输入和输出的宽高比
        input_ratio = input_width / input_height
        output_ratio = output_width / output_height
        
        # 如果输入是宽图(ratio > 1)，输出也应该是宽图
        # 如果输入是长图(ratio < 1)，输出也应该是长图
        if (input_ratio > 1 and output_ratio < 1) or (input_ratio < 1 and output_ratio > 1):
            # 交换输出的宽和高
            output_width, output_height = output_height, output_width
            
        return (output_width, output_height, length, steps, batch_size, seconds, float(fps))

