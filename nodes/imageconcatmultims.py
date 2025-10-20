import torch
import comfy.utils

class ImageConcanateMs:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image1": ("IMAGE",),
            "image2": ("IMAGE",),
            "direction": (
            [   'right',
                'down',
                'left',
                'up',
            ],
            {
            "default": 'right'
             }),
            "match_image_size": ("BOOLEAN", {"default": True}),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "concatenate"
    CATEGORY = "A_my_nodes/image"
    DESCRIPTION = """
Concatenates the image2 to image1 in the specified direction.
"""

    def concatenate(self, image1, image2, direction, match_image_size, first_image_shape=None):
        # Check if the batch sizes are different
        batch_size1 = image1.shape[0]
        batch_size2 = image2.shape[0]

        if batch_size1 != batch_size2:
            # Calculate the number of repetitions needed
            max_batch_size = max(batch_size1, batch_size2)
            repeats1 = max_batch_size - batch_size1
            repeats2 = max_batch_size - batch_size2
            
            # Repeat the last image to match the largest batch size
            if repeats1 > 0:
                last_image1 = image1[-1].unsqueeze(0).repeat(repeats1, 1, 1, 1)
                image1 = torch.cat([image1.clone(), last_image1], dim=0)
            if repeats2 > 0:
                last_image2 = image2[-1].unsqueeze(0).repeat(repeats2, 1, 1, 1)
                image2 = torch.cat([image2.clone(), last_image2], dim=0)

        if match_image_size:
            # Use first_image_shape if provided; otherwise, default to image1's shape
            target_shape = first_image_shape if first_image_shape is not None else image1.shape

            original_height = image2.shape[1]
            original_width = image2.shape[2]
            original_aspect_ratio = original_width / original_height

            if direction in ['left', 'right']:
                # Match the height and adjust the width to preserve aspect ratio
                target_height = target_shape[1]  # B, H, W, C format
                target_width = int(target_height * original_aspect_ratio)
            elif direction in ['up', 'down']:
                # Match the width and adjust the height to preserve aspect ratio
                target_width = target_shape[2]  # B, H, W, C format
                target_height = int(target_width / original_aspect_ratio)
            
            # Adjust image2 to the expected format for common_upscale
            image2_for_upscale = image2.movedim(-1, 1)  # Move C to the second position (B, C, H, W)
            
            # Resize image2 to match the target size while preserving aspect ratio
            image2_resized = comfy.utils.common_upscale(image2_for_upscale, target_width, target_height, "lanczos", "disabled")
            
            # Adjust image2 back to the original format (B, H, W, C) after resizing
            image2_resized = image2_resized.movedim(1, -1)
        else:
            image2_resized = image2

        # Ensure both images have the same number of channels
        channels_image1 = image1.shape[-1]
        channels_image2 = image2_resized.shape[-1]

        if channels_image1 != channels_image2:
            if channels_image1 < channels_image2:
                # Add alpha channel to image1 if image2 has it
                alpha_channel = torch.ones((*image1.shape[:-1], channels_image2 - channels_image1), device=image1.device)
                image1 = torch.cat((image1, alpha_channel), dim=-1)
            else:
                # Add alpha channel to image2 if image1 has it
                alpha_channel = torch.ones((*image2_resized.shape[:-1], channels_image1 - channels_image2), device=image2_resized.device)
                image2_resized = torch.cat((image2_resized, alpha_channel), dim=-1)


        # Concatenate based on the specified direction
        if direction == 'right':
            concatenated_image = torch.cat((image1, image2_resized), dim=2)  # Concatenate along width
        elif direction == 'down':
            concatenated_image = torch.cat((image1, image2_resized), dim=1)  # Concatenate along height
        elif direction == 'left':
            concatenated_image = torch.cat((image2_resized, image1), dim=2)  # Concatenate along width
        elif direction == 'up':
            concatenated_image = torch.cat((image2_resized, image1), dim=1)  # Concatenate along height
        return concatenated_image,
class ImageConcatMultiMs:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_1": ("IMAGE", ),
                "direction": (
                [   'right',
                    'down',
                    'left',
                    'up',
                ],
                {
                "default": 'right',
                "tooltip": "图像拼接方向：right-向右拼接，down-向下拼接，left-向左拼接，up-向上拼接。"
                }),
                "match_image_size": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "是否匹配图像尺寸。启用时会将所有图像调整为相同尺寸后再拼接。"
                }),
                "batch_align_mode": (
                [   'pad_black',
                    'repeat_tensor', 
                    'repeat_last_frame',
                    'truncate_to_shortest',
                ],
                {
                "default": 'pad_black',
                "tooltip": "批次对齐模式：pad_black-用黑色填充，repeat_tensor-重复整个张量，repeat_last_frame-重复最后一帧，truncate_to_shortest-截断到最短批次。"
                }),
                "device_align_mode": (
                [   'first_image',
                    'gpu',
                    'cpu',
                ],
                {
                "default": 'first_image',
                "tooltip": "设备对齐模式：first_image-使用第一张图像的设备，gpu-强制使用GPU，cpu-强制使用CPU。"
                }),
            },
            "optional": {
                "image_2": ("IMAGE", ),
            },
    }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "combine"
    CATEGORY = "A_my_nodes/image"
    DESCRIPTION = """
Creates an image from multiple images with dynamic inputs.
Supports different batch alignment modes and device alignment options.
Connect images to automatically add more input slots.
"""

    def align_batch_sizes(self, images, mode):
        """对齐不同张数的图像批次"""
        if not images:
            return images
            
        batch_sizes = [img.shape[0] for img in images]
        max_batch_size = max(batch_sizes)
        min_batch_size = min(batch_sizes)
        
        # 如果所有批次大小相同，直接返回
        if max_batch_size == min_batch_size:
            return images
            
        aligned_images = []
        
        # 根据模式确定目标批次大小
        if mode == 'truncate_to_shortest':
            target_batch_size = min_batch_size
        else:
            target_batch_size = max_batch_size
        
        for img in images:
            current_batch_size = img.shape[0]
            
            if mode == 'truncate_to_shortest':
                # 截取到最短长度
                aligned_img = img[:target_batch_size]
                aligned_images.append(aligned_img)
            elif current_batch_size == target_batch_size:
                aligned_images.append(img)
            elif mode == 'pad_black':
                # 添加纯黑图像
                padding_needed = target_batch_size - current_batch_size
                black_images = torch.zeros((padding_needed,) + img.shape[1:], 
                                         dtype=img.dtype, device=img.device)
                aligned_img = torch.cat([img, black_images], dim=0)
                aligned_images.append(aligned_img)
            elif mode == 'repeat_tensor':
                # 重复整个张量
                repeat_times = target_batch_size // current_batch_size
                remainder = target_batch_size % current_batch_size
                repeated_img = img.repeat(repeat_times, 1, 1, 1)
                if remainder > 0:
                    repeated_img = torch.cat([repeated_img, img[:remainder]], dim=0)
                aligned_images.append(repeated_img)
            elif mode == 'repeat_last_frame':
                # 重复最后一帧
                padding_needed = target_batch_size - current_batch_size
                last_frame = img[-1].unsqueeze(0).repeat(padding_needed, 1, 1, 1)
                aligned_img = torch.cat([img, last_frame], dim=0)
                aligned_images.append(aligned_img)
                
        return aligned_images

    def align_devices(self, images, mode, reference_device=None):
        """对齐张量设备"""
        if not images:
            return images
            
        if mode == 'first_image' and reference_device is not None:
            target_device = reference_device
        elif mode == 'gpu':
            target_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif mode == 'cpu':
            target_device = torch.device('cpu')
        else:
            target_device = images[0].device
            
        aligned_images = []
        for img in images:
            aligned_images.append(img.to(target_device))
            
        return aligned_images

    def combine(self, direction, match_image_size, batch_align_mode, device_align_mode, **kwargs):
        # 收集所有输入的图像
        images = []
        image_keys = sorted([k for k in kwargs.keys() if k.startswith("image_")])
        
        for key in image_keys:
            if kwargs[key] is not None:
                images.append(kwargs[key])
        
        if len(images) == 0:
            raise ValueError("至少需要一个输入图像")
        
        if len(images) == 1:
            return (images[0],)
        
        # 设备对齐
        reference_device = images[0].device
        images = self.align_devices(images, device_align_mode, reference_device)
        
        # 批次对齐
        images = self.align_batch_sizes(images, batch_align_mode)
        
        # 开始拼接
        result_image = images[0]
        first_image_shape = result_image.shape
        
        for i in range(1, len(images)):
            result_image, = ImageConcanateMs.concatenate(
                self, result_image, images[i], direction, match_image_size, 
                first_image_shape=first_image_shape
            )
        
        return (result_image,)