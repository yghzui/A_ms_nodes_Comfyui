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
                # 每段最大允许的帧数
                "max_frames": ("INT", {"default": 50, "min": 1, "max": 100000000, "tooltip": "每段最大允许的帧数"}),
                # 每段最小需要的帧数
                "min_frames": ("INT", {"default": 16, "min": 1, "max": 100000000, "tooltip": "每段最小需要的帧数"}),
                # 处理模式：0=合并到最后一段，1=均分到所有段
                "merge_mode": ("INT", {"default": 0, "min": 0, "max": 1, "tooltip": "处理模式：0=合并到最后一段，1=均分到所有段"}),
            }
        }

    RETURN_TYPES = ("INT", "INT", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("split_value", "num_segments", "overlap", "max_frames", "min_frames", "merge_mode")
    FUNCTION = "calc_frames_split"
    CATEGORY = "A_my_nodes/math"

    def calc_frames_split(self, total_frames, split_value, overlap, max_frames, min_frames, merge_mode):
        # 规范化输入
        total_frames = int(total_frames)
        split_value = int(split_value)
        overlap = int(overlap)
        max_frames = int(max_frames)
        min_frames = int(min_frames)
        merge_mode = int(merge_mode)

        # 基本健壮性处理
        if split_value <= 0:
            # 切分长度无效时，直接返回0
            return (0, 0, 0, max_frames, min_frames, merge_mode)

        # 确保最小帧数不大于最大帧数
        min_frames = min(min_frames, max_frames)
        # 确保基础切分长度不大于最大帧数
        split_value = min(split_value, max_frames)
        # 限制 overlap 小于 split_value，若不满足则自动修正
        if overlap >= split_value:
            overlap = split_value - 1 if split_value > 1 else 0

        # 总帧数<=0，无需切分
        if total_frames <= 0:
            return (split_value, 0, overlap, max_frames, min_frames, merge_mode)

        # 总帧数不超过一个切分长度，只需一次
        if total_frames <= split_value:
            return (split_value, 1, overlap, max_frames, min_frames, merge_mode)

        # 步长 = 切分长度 - 重叠
        step = max(1, split_value - overlap)

        # 初步计算段数
        num_segments = 1 + ((total_frames - split_value + step - 1) // step)
        
        # 计算最后一段的帧数
        last_segment_frames = total_frames - (num_segments - 1) * step
        
        # 如果最后一段小于最小帧数，需要特殊处理
        if last_segment_frames < min_frames:
            if merge_mode == 0:  # 合并到最后一段模式
                if num_segments > 1:  # 确保有多于一段可以合并
                    # 计算合并后最后一段的长度
                    merged_last_length = last_segment_frames + step
                    # 如果合并后超过最大帧数，则维持原状
                    if merged_last_length <= max_frames:
                        num_segments -= 1
                        split_value = max(split_value, merged_last_length)
            else:  # 均分模式
                if num_segments > 1:
                    # 计算需要分配的总帧数
                    extra_frames = last_segment_frames
                    # 计算每段可以额外分配的帧数
                    frames_per_segment = extra_frames // num_segments
                    # 如果分配后不会超过最大帧数，则进行分配
                    if split_value + frames_per_segment <= max_frames:
                        split_value += frames_per_segment
                        # 重新计算段数
                        step = max(1, split_value - overlap)
                        num_segments = 1 + ((total_frames - split_value + step - 1) // step)

        return (split_value, int(num_segments), overlap, max_frames, min_frames, merge_mode)

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
                # 每段最大允许的帧数
                "max_frames": ("INT", {"default": 50, "min": 1, "max": 100000000, "tooltip": "每段最大允许的帧数"}),
                # 每段最小需要的帧数
                "min_frames": ("INT", {"default": 16, "min": 1, "max": 100000000, "tooltip": "每段最小需要的帧数"}),
                # 处理模式：0=合并到最后一段，1=均分到所有段
                "merge_mode": ("INT", {"default": 0, "min": 0, "max": 1, "tooltip": "处理模式：0=合并到最后一段，1=均分到所有段"}),
                # 需要截取的段索引，从0开始，例如总段数为6时，索引范围0-5
                "index": ("INT", {"default": 0, "min": 0, "max": 100000000, "tooltip": "需要截取的段索引，从0开始"}),
                # 是否启用后处理覆盖
                "enable_post_cover": ("BOOLEAN", {"default": False, "tooltip": "启用后，用上一段图像的尾部有效重叠帧覆盖当前片段的开头"}),
                # 有效重叠数，必须小于等于 overlap
                "effective_overlap": ("INT", {"default": 0, "min": 0, "max": 100000000, "tooltip": "有效重叠帧数，必须小于等于重叠数"}),
            },
            "optional": {
                # 图像为 (n,h,w,c)，可为空
                "images": ("IMAGE",),
                # 遮罩为 (n,h,w)，可为空
                "mask": ("MASK",),
                # 上一段的图像 (n,h,w,c)，可为空
                "prev_images": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("image_segment", "mask_segment", "start_index", "length", "max_frames", "min_frames", "merge_mode", "overlap")
    FUNCTION = "slice_segment"
    CATEGORY = "A_my_nodes/math"

    def slice_segment(self, total_frames, split_value, overlap, max_frames, min_frames, merge_mode, index, enable_post_cover, effective_overlap, images=None, mask=None, prev_images=None):
        # 规范化与健壮性
        total_frames = int(total_frames)
        split_value = int(split_value)
        overlap = int(overlap)
        index = int(index)
        enable_post_cover = bool(enable_post_cover)
        effective_overlap = int(effective_overlap)

        if split_value <= 0:
            # 切分长度无效，直接返回None与0长度
            return (None, None, 0, 0, max_frames, min_frames, merge_mode, overlap)

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

        # 后处理覆盖逻辑：仅在开启、prev_images 存在、当前片段存在、有效重叠>0时生效
        if enable_post_cover and (prev_images is not None and hasattr(prev_images, 'shape')) and (image_segment is not None and hasattr(image_segment, 'shape')):
            # 约束有效重叠数：不得超过 overlap 与当前片段长度
            eff = max(0, min(effective_overlap, overlap, int(image_segment.shape[0]) if hasattr(image_segment, 'shape') else 0))
            if eff > 0:
                import torch
                # 计算从上一段取帧的切片：prev_images[-overlap : -overlap + eff]
                # 若上一段长度不足以从 -overlap 开始，则降级为取末尾 eff 帧
                try:
                    n_prev = int(prev_images.shape[0])
                except Exception:
                    n_prev = 0
                prev_slice = None
                if n_prev > 0:
                    if n_prev >= overlap and overlap > 0:
                        # 常规切片，等价于 Python 负索引 prev_images[-overlap : -overlap + eff]
                        start_idx = n_prev - overlap
                        end_idx = start_idx + eff
                        # 边界保护
                        start_idx = max(0, min(start_idx, n_prev))
                        end_idx = max(0, min(end_idx, n_prev))
                        prev_slice = prev_images[start_idx:end_idx]
                        # 如果切片帧数不足 eff，则退化为末尾 eff 帧
                        if hasattr(prev_slice, 'shape') and int(prev_slice.shape[0]) < eff:
                            prev_slice = prev_images[max(0, n_prev - eff): n_prev]
                    else:
                        # 无法从 -overlap 起取，退化为末尾 eff 帧
                        prev_slice = prev_images[max(0, n_prev - eff): n_prev]

                # 若 prev_slice 可用且长度与 eff 一致，执行覆盖
                if prev_slice is not None and hasattr(prev_slice, 'shape') and int(prev_slice.shape[0]) > 0:
                    # 对齐 dtype/device
                    if prev_slice.dtype != image_segment.dtype:
                        prev_slice = prev_slice.to(image_segment.dtype)
                    if prev_slice.device != image_segment.device:
                        prev_slice = prev_slice.to(image_segment.device)

                    # 替换规则：prev_slice + image_segment[eff:]
                    tail_image = image_segment[eff:] if int(image_segment.shape[0]) > eff else image_segment[:0]
                    new_image = torch.cat([prev_slice, tail_image], dim=0)
                    image_segment = new_image

                    # mask 的对应处理：前 eff 帧为纯黑，后接原 mask 去掉前 eff 帧
                    if mask_segment is not None and hasattr(mask_segment, 'shape'):
                        m_len = int(mask_segment.shape[0])
                        if m_len > 0:
                            eff_mask = min(eff, m_len)
                            h = int(mask_segment.shape[1])
                            w = int(mask_segment.shape[2])
                            zeros_mask = torch.zeros((eff_mask, h, w), dtype=mask_segment.dtype, device=mask_segment.device)
                            tail_mask = mask_segment[eff_mask:] if m_len > eff_mask else mask_segment[:0]
                            mask_segment = torch.cat([zeros_mask, tail_mask], dim=0)

        return (image_segment, mask_segment, int(start), int(segment_length), max_frames, min_frames, merge_mode, overlap)

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

class ImageToSequenceWithMask:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # 是否使用“秒数*帧率+1”来计算长度；关闭则直接使用 frames
                "use_seconds_for_length": ("BOOLEAN", {"default": True, "tooltip": "启用后，帧数 = 秒数 * 帧率 + 1；否则直接使用帧数"}),
                # 秒数与帧率，仅在 use_seconds_for_length 为 True 时生效
                "seconds": ("INT", {"default": 2, "min": 1, "max": 100000000, "tooltip": "视频时长（秒）"}),
                "fps": ("INT", {"default": 16, "min": 1, "max": 100000000, "tooltip": "每秒帧数"}),
                # 直接帧数，仅在 use_seconds_for_length 为 False 时生效
                "frames": ("INT", {"default": 33, "min": 1, "max": 100000000, "tooltip": "直接使用的帧数（关闭秒数控制时生效）"}),
                # 当有输入图像时，索引 [1:] 的填充策略
                "fill_mode_when_image_present": (["repeat_input", "white", "black"], {"default": "repeat_input", "tooltip": "当存在输入图像时，后续帧的填充策略"}),
                # 当没有输入图像时，全段颜色
                "fallback_color_when_image_missing": (["white", "black"], {"default": "white", "tooltip": "当输入图像不存在时，整段的填充颜色"}),
                # 当没有输入图像时用于生成分辨率
                "output_width": ("INT", {"default": 512, "min": 1, "max": 16384, "tooltip": "当输入图像不存在时，生成图像的宽度"}),
                "output_height": ("INT", {"default": 512, "min": 1, "max": 16384, "tooltip": "当输入图像不存在时，生成图像的高度"}),
            },
            "optional": {
                # 输入图像，(n,h,w,c)，可以为空；若存在仅使用首帧作为基准
                "images": ("IMAGE", {}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT")
    RETURN_NAMES = ("images", "mask", "frames", "fps")
    FUNCTION = "build_sequence"
    CATEGORY = "A_my_nodes/math"

    def _ensure_three_channels(self, tensor):
        """确保输入张量最后一维为3通道：
        - 若为1通道则重复到3通道
        - 若大于3通道则裁剪为前3通道
        - 若介于(1,3)之间则重复最后一通道直至3
        期望形状：(n,h,w,c)
        """
        try:
            import torch
            if tensor is None or not hasattr(tensor, 'shape'):
                return tensor
            if len(tensor.shape) != 4:
                return tensor
            c = int(tensor.shape[-1])
            if c == 3:
                return tensor
            if c == 1:
                return tensor.repeat(1, 1, 1, 3)
            if c > 3:
                return tensor[..., :3]
            # c 为 2 的情况：复制最后一通道到第3通道
            last = tensor[..., -1:]
            return torch.cat([tensor, last], dim=-1)
        except Exception:
            return tensor

    def _make_constant_frames(self, num_frames, height, width, value, dtype=None, device=None):
        """构造常量帧：(num_frames, h, w, 3)，value 为 0 或 255（按需求）。"""
        import torch
        if dtype is None:
            dtype = torch.float32
        if device is None:
            device = "cpu"
        # 生成 [num_frames, h, w, 3] 的常量张量
        const = torch.full((int(num_frames), int(height), int(width), 3), float(value), dtype=dtype, device=device)
        return const

    def _make_mask(self, num_frames, height, width, first_zero=True, device=None, dtype=None):
        """生成遮罩：(num_frames, h, w)。first_zero=True 时，第0帧为0，其余为1；否则全1。"""
        import torch
        if dtype is None:
            dtype = torch.float32
        if device is None:
            device = "cpu"
        m = int(max(1, num_frames))
        if first_zero and m >= 1:
            mask = torch.ones((m, int(height), int(width)), dtype=dtype, device=device)
            mask[0].zero_()
        else:
            mask = torch.ones((m, int(height), int(width)), dtype=dtype, device=device)
        return mask

    def build_sequence(self, use_seconds_for_length, seconds, fps, frames,
                       fill_mode_when_image_present, fallback_color_when_image_missing,
                       output_width, output_height,
                       images=None):
        # 计算目标帧数 m
        if bool(use_seconds_for_length):
            m = int(seconds) * int(fps) + 1
        else:
            m = int(frames)
        m = max(1, int(m))

        # 判断输入图像是否有效
        has_image = (images is not None and hasattr(images, 'shape') and int(images.shape[0]) > 0)

        if has_image:
            # 从输入中获取首帧，并确保三通道
            base = images[0:1]
            base = self._ensure_three_channels(base)

            # 解析尺寸与设备、dtype
            h = int(base.shape[1])
            w = int(base.shape[2])
            try:
                dtype = base.dtype
                device = base.device
            except Exception:
                dtype = None
                device = None

            # 根据填充策略构造序列
            if fill_mode_when_image_present == "repeat_input":
                # 直接重复首帧 m 次
                import torch
                seq = base.repeat(int(m), 1, 1, 1)
            else:
                # 先放入首帧，再拼接 (m-1) 常量帧（白=255，黑=0）
                color_value = 255 if fill_mode_when_image_present == "white" else 0
                tail = self._make_constant_frames(max(0, m - 1), h, w, color_value, dtype=dtype, device=device)
                import torch
                seq = torch.cat([base, tail], dim=0) if tail.shape[0] > 0 else base

            # 遮罩：首帧为0，其余为1
            mask = self._make_mask(m, h, w, first_zero=True, device=device, dtype=None)
            return (seq, mask, int(m), int(fps))
        else:
            # 无输入图像：使用给定高宽直接构造三通道常量帧
            h = int(output_height)
            w = int(output_width)
            color_value = 255 if fallback_color_when_image_missing == "white" else 0
            seq = self._make_constant_frames(m, h, w, color_value, dtype=None, device=None)
            # 遮罩：全部为1（全部重绘）
            mask = self._make_mask(m, h, w, first_zero=False, device=seq.device if hasattr(seq, 'device') else None, dtype=None)
            return (seq, mask, int(m), int(fps))

