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


class ImageLayerMix:
    """
    一个ComfyUI节点，用于根据遮罩将图层图像覆盖到背景图像上。
    接收背景图像、图层图像和遮罩，根据遮罩将图层图像覆盖到背景图像上。
    """
    @classmethod
    def INPUT_TYPES(s):
        """
        定义节点的输入参数。
        """
        return {
            "required": {
                "background": ("IMAGE",),
                "layer": ("IMAGE",),
                "mask": ("MASK",),
                "invert_mask": ("BOOLEAN", {"default": False, "tooltip": "是否反转遮罩，True表示反转。"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "mix_images"
    CATEGORY = "A_my_nodes/Image"

    def mix_images(self, background: torch.Tensor, layer: torch.Tensor, mask: torch.Tensor, invert_mask: bool = False):
        """
        执行节点功能的核心方法。
        
        Args:
            background (torch.Tensor): 背景图像张量，形状为 (n, h, w, c)。
            layer (torch.Tensor): 图层图像张量，形状为 (n, h, w, 3)。
            mask (torch.Tensor): 遮罩张量，形状为 (n, h, w)。
            invert_mask (bool): 是否反转遮罩。
        
        Returns:
            tuple[torch.Tensor]: 返回一个包含处理后的图像的元组。
        """
        # 确保背景和图层的批次数、高度和宽度相同
        if background.shape[0] != layer.shape[0] or background.shape[1] != layer.shape[1] or background.shape[2] != layer.shape[2]:
            print(f"错误：背景图像形状 {background.shape} 与图层图像形状 {layer.shape} 不兼容。")
            return (background,)
        
        # 确保遮罩的形状与图像匹配
        if mask.shape[0] != background.shape[0] or mask.shape[1] != background.shape[1] or mask.shape[2] != background.shape[2]:
            print(f"错误：遮罩形状 {mask.shape} 与图像形状 {background.shape[:3]} 不匹配。")
            return (background,)
        
        # 复制背景图像以避免修改原始张量
        result = background.clone()
        
        # 处理图层图像，确保是3通道RGB
        if layer.shape[-1] > 3:
            layer = layer[..., :3]
        
        # 如果需要反转遮罩
        if invert_mask:
            mask = 1.0 - mask
        
        # 扩展遮罩维度以匹配图像通道数
        expanded_mask = mask.unsqueeze(-1)
        
        # 根据通道数处理
        if background.shape[-1] == 3:
            # 对于RGB图像，直接使用扩展的遮罩
            expanded_mask = expanded_mask.expand(-1, -1, -1, 3)
        elif background.shape[-1] == 4:
            # 对于RGBA图像，扩展遮罩到4通道
            expanded_mask = expanded_mask.expand(-1, -1, -1, 4)
            # 如果图层是3通道，需要扩展为4通道
            if layer.shape[-1] == 3:
                # 创建一个新的4通道图层，Alpha通道设为1
                layer_rgba = torch.ones_like(background)
                layer_rgba[..., :3] = layer
                layer = layer_rgba
        
        # 混合图像：result = background * (1 - mask) + layer * mask
        result = background * (1.0 - expanded_mask) + layer * expanded_mask
        
        return (result,)


class ImageDualMaskColorFill:
    """
    一个ComfyUI节点，用于在两个遮罩不重叠的非零区域填充指定颜色。
    接收一个图像和两个遮罩，找出两个遮罩中不为零且互相不重叠的区域，
    然后在图像中对该区域填充指定颜色。
    同时输出不重叠区域的遮罩。
    """
    @classmethod
    def INPUT_TYPES(s):
        """
        定义节点的输入参数。
        """
        return {
            "required": {
                "image": ("IMAGE",),
                "mask_1": ("MASK",),
                "mask_2": ("MASK",),
                "color": ("STRING", {
                    "default": "#FF0000",
                    "tooltip": "要填充的颜色，可以是HEX格式（例如'#FF0000'）或RGB格式（例如'255,0,0'）。"
                }),
                "threshold": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "遮罩阈值，大于此值的像素被认为是遮罩区域。"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "non_overlapping_mask")
    FUNCTION = "fill_non_overlapping"
    CATEGORY = "A_my_nodes/Image"

    def fill_non_overlapping(self, image: torch.Tensor, mask_1: torch.Tensor, mask_2: torch.Tensor, color: str, threshold: float = 0.5):
        """
        执行节点功能的核心方法。
        
        Args:
            image (torch.Tensor): 输入图像张量，形状为 (n, h, w, c)。
            mask_1 (torch.Tensor): 第一个遮罩张量，形状为 (n, h, w)。
            mask_2 (torch.Tensor): 第二个遮罩张量，形状为 (n, h, w)。
            color (str): 用户输入的颜色字符串。
            threshold (float): 遮罩阈值，大于此值的像素被认为是遮罩区域。
        
        Returns:
            tuple: 返回处理后的图像和不重叠区域的遮罩。
        """
        # 确保遮罩的形状与图像匹配
        if (mask_1.shape[0] != image.shape[0] or mask_1.shape[1] != image.shape[1] or mask_1.shape[2] != image.shape[2] or
            mask_2.shape[0] != image.shape[0] or mask_2.shape[1] != image.shape[1] or mask_2.shape[2] != image.shape[2]):
            print(f"错误：遮罩形状与图像形状不匹配。")
            print(f"图像形状: {image.shape[:3]}, 遮罩1形状: {mask_1.shape}, 遮罩2形状: {mask_2.shape}")
            return (image, torch.zeros_like(mask_1))
        
        try:
            # 解析并归一化颜色值
            r, g, b = parse_color(color)
        except ValueError as e:
            print(f"错误：无效的颜色格式 - {e}。将返回原始图像。")
            return (image, torch.zeros_like(mask_1))
        
        # 复制输入图像以避免修改原始张量
        result = image.clone()
        
        # 二值化遮罩
        mask_1_binary = mask_1 > threshold
        mask_2_binary = mask_2 > threshold
        
        # 找出两个遮罩中不为零且互相不重叠的区域
        # mask_1有效区域 = mask_1为1且mask_2为0的区域
        mask_1_only = mask_1_binary & (~mask_2_binary)
        # mask_2有效区域 = mask_2为1且mask_1为0的区域
        mask_2_only = mask_2_binary & (~mask_1_binary)
        
        # 合并两个不重叠的区域
        non_overlapping_mask = mask_1_only | mask_2_only
        
        # 处理不同通道数的图像
        if image.shape[-1] == 3:
            # 对于RGB图像
            # 将不重叠区域的RGB通道设置为指定的颜色
            result[..., 0][non_overlapping_mask] = r
            result[..., 1][non_overlapping_mask] = g
            result[..., 2][non_overlapping_mask] = b
        elif image.shape[-1] == 4:
            # 对于RGBA图像
            # 将不重叠区域的RGB通道设置为指定的颜色，保留Alpha通道
            result[..., 0][non_overlapping_mask] = r
            result[..., 1][non_overlapping_mask] = g
            result[..., 2][non_overlapping_mask] = b
        else:
            # 对于其他通道数的图像，只处理前3个通道
            print(f"警告：输入图像通道数为 {image.shape[-1]}，将只处理前3个通道。")
            if image.shape[-1] >= 3:
                result[..., 0][non_overlapping_mask] = r
                result[..., 1][non_overlapping_mask] = g
                result[..., 2][non_overlapping_mask] = b
        
        # 将布尔遮罩转换为浮点遮罩
        non_overlapping_mask_float = non_overlapping_mask.float()
        
        # 返回处理后的图像和不重叠区域的遮罩
        return (result, non_overlapping_mask_float)

