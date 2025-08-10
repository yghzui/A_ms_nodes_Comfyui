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

class FramesSplitCalculator:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # 视频总帧数，例如 155
                "total_frames": ("INT", {"default": 155, "min": 0, "max": 100000000, "tooltip": "视频总帧数"}),
                # 每段包含的帧数（切分长度），例如 33
                "split_value": ("INT", {"default": 33, "min": 1, "max": 100000000, "tooltip": "每段包含的帧数（切分长度）"}),
                # 相邻两段之间的重叠帧数，必须小于切分长度
                "overlap": ("INT", {"default": 0, "min": 0, "max": 100000000, "tooltip": "相邻两段之间的重叠帧数，必须小于切分长度"}),
            }
        }

    RETURN_TYPES = ("INT", "INT", "INT")
    RETURN_NAMES = ("split_value", "num_segments", "overlap")
    FUNCTION = "calc_frames_split"
    CATEGORY = "A_my_nodes/math"

    def calc_frames_split(self, total_frames, split_value, overlap):
        # 规范化输入
        total_frames = int(total_frames)
        split_value = int(split_value)
        overlap = int(overlap)

        # 基本健壮性处理
        if split_value <= 0:
            # 切分长度无效时，直接返回0
            return (0, 0)

        # 限制 overlap 小于 split_value，若不满足则自动修正
        if overlap >= split_value:
            overlap = split_value - 1 if split_value > 1 else 0

        # 总帧数<=0，无需切分
        if total_frames <= 0:
            return (split_value, 0)

        # 总帧数不超过一个切分长度，只需一次
        if total_frames <= split_value:
            return (split_value, 1)

        # 步长 = 切分长度 - 重叠
        step = max(1, split_value - overlap)

        # 段数计算：1 + ceil((total_frames - split_value) / step)
        # 使用整数运算避免浮点误差
        num_segments = 1 + ((total_frames - split_value + step - 1) // step)
        return (split_value, int(num_segments), overlap)

class FramesSegmentSlicer:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # 视频总帧数，例如 155
                "total_frames": ("INT", {"default": 155, "min": 0, "max": 100000000, "tooltip": "视频总帧数"}),
                # 每段包含的帧数（切分长度），例如 33
                "split_value": ("INT", {"default": 33, "min": 1, "max": 100000000, "tooltip": "每段包含的帧数（切分长度）"}),
                # 相邻两段之间的重叠帧数，必须小于切分长度
                "overlap": ("INT", {"default": 0, "min": 0, "max": 100000000, "tooltip": "相邻两段之间的重叠帧数，必须小于切分长度"}),
                # 需要截取的段索引，从0开始，例如总段数为6时，索引范围0-5
                "index": ("INT", {"default": 0, "min": 0, "max": 100000000, "tooltip": "需要截取的段索引，从0开始"}),
            },
            "optional": {
                # 图像为 (n,h,w,c)，可为空
                "images": ("IMAGE",),
                # 遮罩为 (n,h,w)，可为空
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT")
    RETURN_NAMES = ("image_segment", "mask_segment", "start_index", "length")
    FUNCTION = "slice_segment"
    CATEGORY = "A_my_nodes/math"

    def slice_segment(self, total_frames, split_value, overlap, index, images=None, mask=None):
        # 规范化与健壮性
        total_frames = int(total_frames)
        split_value = int(split_value)
        overlap = int(overlap)
        index = int(index)

        if split_value <= 0:
            # 切分长度无效，直接返回None与0长度
            return (None, None, 0, 0)

        # 确保 overlap < split_value
        if overlap >= split_value:
            overlap = split_value - 1 if split_value > 1 else 0

        # 计算步长与当前段起点
        step = max(1, split_value - overlap)
        start = max(0, index * step)

        # 计算理论长度（不会超过总帧数）
        if total_frames <= 0 or start >= total_frames:
            # 没有有效帧
            segment_length = 0
        else:
            segment_length = min(split_value, total_frames - start)

        image_segment = None
        mask_segment = None

        # 截取 images (n,h,w,c)
        if images is not None and hasattr(images, 'shape'):
            try:
                n = int(images.shape[0])
                # 对齐到实际可用的帧范围
                end_idx = min(start + segment_length, n)
                if end_idx > start:
                    image_segment = images[start:end_idx]
            except Exception:
                # 若输入不是预期张量形态，保持为 None
                image_segment = None

        # 截取 mask (n,h,w)
        if mask is not None and hasattr(mask, 'shape'):
            try:
                n_m = int(mask.shape[0])
                end_idx_m = min(start + segment_length, n_m)
                if end_idx_m > start:
                    mask_segment = mask[start:end_idx_m]
            except Exception:
                mask_segment = None

        return (image_segment, mask_segment, int(start), int(segment_length))

class ImagesConcatWithOverlap:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # 重叠帧数，表示 images_a 的最后 overlap 帧将被 images_b 的前 overlap 帧覆盖
                "overlap": ("INT", {"default": 0, "min": 0, "max": 100000000, "tooltip": "重叠帧数(后者覆盖前者)，输出为 n+m-overlap 帧"}),
            },
            "optional": {
                # 第一段图像，形状为 (n,h,w,c)，可为空
                "images_a": ("IMAGE", {}),
                # 第二段图像，形状为 (m,h,w,c)，可为空
                "images_b": ("IMAGE", {}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "concat_with_overlap"
    CATEGORY = "A_my_nodes/math"

    def concat_with_overlap(self, overlap, images_a=None, images_b=None):
        # 任意一端不存在时，直接返回存在的一端；两端都不存在则返回 None
        if images_a is None and images_b is None:
            return (None,)
        if images_a is None:
            return (images_b,)
        if images_b is None:
            return (images_a,)

        # 为避免外部依赖问题，仅在需要拼接时导入 torch
        import torch

        # 基本形状与参数处理
        n1 = int(images_a.shape[0]) if hasattr(images_a, 'shape') else 0
        n2 = int(images_b.shape[0]) if hasattr(images_b, 'shape') else 0
        overlap = int(overlap)

        # 将 overlap 限制到 [0, min(n1, n2)] 范围内
        overlap_used = 0
        if n1 > 0 and n2 > 0 and overlap > 0:
            overlap_used = min(overlap, n1, n2)
        
        # 取出第一段有效部分（去掉需要被覆盖的末尾 overlap_used 帧）
        if n1 - overlap_used > 0:
            head_a = images_a[: n1 - overlap_used]
        else:
            # 保持空维度以便拼接
            head_a = images_a[:0]

        # 拼接得到目标输出，大小为 n1 + n2 - overlap_used
        if head_a.shape[0] == 0:
            result = images_b
        elif n2 == 0:
            result = head_a
        else:
            # 确保 dtype 和 device 一致
            if head_a.dtype != images_b.dtype:
                images_b = images_b.to(head_a.dtype)
            if head_a.device != images_b.device:
                images_b = images_b.to(head_a.device)
            result = torch.cat([head_a, images_b], dim=0)

        return (result,)

