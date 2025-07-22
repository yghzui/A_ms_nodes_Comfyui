import torch

def hex_to_rgb(hex_color):
    """
    将HEX颜色字符串转换为RGB元组。
    支持 #RRGGBB 和 #RGB 格式。
    """
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 3:
        # e.g., #F0C -> #FF00CC
        hex_color = hex_color[0]*2 + hex_color[1]*2 + hex_color[2]*2
    if len(hex_color) != 6:
        raise ValueError("无效的HEX颜色代码，必须是3位或6位。")
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def parse_color(color_str):
    """
    解析颜色字符串，可以是HEX或RGB格式。
    返回一个归一化的 (r, g, b) 元组，值范围为 [0.0, 1.0]。
    """
    color_str = color_str.strip()
    if color_str.startswith('#'):
        rgb = hex_to_rgb(color_str)
    else:
        try:
            # 尝试解析 "r,g,b" 格式
            parts = color_str.replace('(', '').replace(')', '').split(',')
            if len(parts) != 3:
                raise ValueError("RGB颜色值必须包含三个逗号分隔的数值。")
            rgb = tuple(int(p.strip()) for p in parts)
            # 检查RGB值是否在0-255范围内
            for val in rgb:
                if not 0 <= val <= 255:
                    raise ValueError("RGB值必须在0到255之间。")
        except ValueError as e:
            # 如果解析失败，抛出更明确的异常
            raise ValueError(f"无法解析颜色字符串 '{color_str}'. 请使用 '#RRGGBB' 或 'R,G,B' (0-255) 格式。") from e

    # 归一化到 [0.0, 1.0]
    return (rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0)


class ImageMaskedColorFill:
    """
    一个ComfyUI节点，它接收一个图像和一个颜色值。
    如果图像有4个通道（RGBA），它会用指定的颜色填充Alpha通道=0的区域（完全透明区域）。
    如果图像只有3个通道（RGB），它会直接返回原始图像。
    输出始终是3通道RGB图像。
    """
    @classmethod
    def INPUT_TYPES(s):
        """
        定义节点的输入参数。
        """
        return {
            "required": {
                "image": ("IMAGE",),
                "color": ("STRING", {
                    "default": "#FF0000",
                    "tooltip": "要填充的颜色，可以是HEX格式（例如'#FF0000'）或RGB格式（例如'255,0,0'）。"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "fill_color"

    CATEGORY = "A_my_nodes/Image"

    def fill_color(self, image: torch.Tensor, color: str):
        """
        执行节点功能的核心方法。
        
        Args:
            image (torch.Tensor): 输入图像张量，形状为 (n, h, w, c)。
            color (str): 用户输入的颜色字符串。
        
        Returns:
            tuple[torch.Tensor]: 返回一个包含处理后的3通道RGB图像的元组。
        """
        # 如果是3通道图像，直接返回
        if image.shape[-1] == 3:
            return (image,)
            
        # 如果不是4通道图像，打印警告并返回前3个通道
        if image.shape[-1] != 4:
            print(f"警告：输入图像通道数为 {image.shape[-1]}，将只返回前3个通道。")
            return (image[..., :3],)

        try:
            # 解析并归一化颜色值
            r, g, b = parse_color(color)
        except ValueError as e:
            # 在ComfyUI中，直接抛出异常会导致执行中断。
            # 更友好的方式是记录错误并返回未修改的图像的RGB部分。
            print(f"错误：无效的颜色格式 - {e}。将返回原始图像的RGB部分。")
            return (image[..., :3],)

        # 复制输入图像的RGB通道以避免修改原始张量
        image_out = image[..., :3].clone()
        
        # 提取Alpha通道。在RGBA中，Alpha通道索引为3。
        # Alpha = 0 的区域被认为是完全透明的区域（非遮罩区域）。
        alpha_channel = image[..., 3]
        non_mask = alpha_channel == 0
        
        # 将完全透明区域的RGB通道设置为指定的颜色
        image_out[..., 0][non_mask] = r
        image_out[..., 1][non_mask] = g
        image_out[..., 2][non_mask] = b

        # 返回处理后的3通道RGB图像张量
        return (image_out,)


class ImageBlackColorFill:
    """
    一个ComfyUI节点，用于将图像中的黑色像素替换为指定颜色。
    黑色像素的判定可以通过阈值调整，默认为RGB值都小于0.1的像素判定为黑色。
    """
    @classmethod
    def INPUT_TYPES(s):
        """
        定义节点的输入参数。
        """
        return {
            "required": {
                "image": ("IMAGE",),
                "color": ("STRING", {
                    "default": "#FF0000",
                    "tooltip": "要填充的颜色，可以是HEX格式（例如'#FF0000'）或RGB格式（例如'255,0,0'）。"
                }),
                "threshold": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "判定为黑色的阈值，RGB三个通道值都小于此阈值时判定为黑色。"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "fill_black"

    CATEGORY = "A_my_nodes/Image"

    def fill_black(self, image: torch.Tensor, color: str, threshold: float = 0.1):
        """
        执行节点功能的核心方法。
        
        Args:
            image (torch.Tensor): 输入图像张量，形状为 (n, h, w, c)。
            color (str): 用户输入的颜色字符串。
            threshold (float): 判定为黑色的阈值。
        
        Returns:
            tuple[torch.Tensor]: 返回一个包含处理后的RGB图像的元组。
        """
        # 确保输出是3通道
        if image.shape[-1] > 3:
            image = image[..., :3]

        try:
            # 解析并归一化颜色值
            r, g, b = parse_color(color)
        except ValueError as e:
            print(f"错误：无效的颜色格式 - {e}。将返回原始图像。")
            return (image,)

        # 复制输入图像以避免修改原始张量
        image_out = image.clone()
        
        # 找出所有RGB值都小于阈值的像素（黑色像素）
        black_pixels = (image[..., 0] < threshold) & (image[..., 1] < threshold) & (image[..., 2] < threshold)
        
        # 将黑色像素替换为指定颜色
        image_out[..., 0][black_pixels] = r
        image_out[..., 1][black_pixels] = g
        image_out[..., 2][black_pixels] = b

        return (image_out,)

