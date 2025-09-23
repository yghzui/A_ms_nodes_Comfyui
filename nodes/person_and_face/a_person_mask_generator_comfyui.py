import math
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from functools import reduce
import cv2

import torch
import numpy as np
from PIL import Image
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
ImageSegmenter = mp.tasks.vision.ImageSegmenter
ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions
VisionRunningMode = mp.tasks.vision.RunningMode

import folder_paths


def get_a_person_mask_generator_model_path() -> str:
    model_folder_name = "mediapipe"
    model_name = "selfie_multiclass_256x256.tflite"

    model_folder_path = os.path.join(folder_paths.models_dir, model_folder_name)
    model_file_path = os.path.join(model_folder_path, model_name)

    if not os.path.exists(model_file_path):
        import urllib.request

        model_url = f"https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/{model_name}"
        print(f"Downloading '{model_name}' model")
        os.makedirs(model_folder_path, exist_ok=True)
        urllib.request.urlretrieve(model_url, model_file_path)

    return model_file_path


class APersonMaskGenerator:

    def __init__(self):
        # download the model if we need it
        get_a_person_mask_generator_model_path()

    @classmethod
    def INPUT_TYPES(self):
        false_widget = (
            "BOOLEAN",
            {"default": False, "label_on": "enabled", "label_off": "disabled"},
        )
        true_widget = (
            "BOOLEAN",
            {"default": True, "label_on": "enabled", "label_off": "disabled"},
        )

        return {
            "required": {
                "images": ("IMAGE",),
            },
            "optional": {
                "face_mask": true_widget,
                "background_mask": false_widget,
                "hair_mask": false_widget,
                "body_mask": false_widget,
                "clothes_mask": false_widget,
                "confidence": (
                    "FLOAT",
                    {"default": 0.40, "min": 0.01, "max": 1.0, "step": 0.01},
                ),
                "refine_mask": true_widget,
            },
        }

    CATEGORY = "A Person Mask Generator - David Bielejeski"
    # 预先定义所有可能的输出：合并遮罩 + 各个区域的单独遮罩
    RETURN_TYPES = ("MASK", "MASK", "MASK", "MASK", "MASK", "MASK")
    RETURN_NAMES = ("merged_mask", "face_mask", "background_mask", "hair_mask", "body_mask", "clothes_mask")

    FUNCTION = "generate_mask"

    def get_mediapipe_image(self, image: Image) -> mp.Image:
        # Convert image to NumPy array
        numpy_image = np.asarray(image)

        image_format = mp.ImageFormat.SRGB

        # Convert BGR to RGB (if necessary)
        if numpy_image.shape[-1] == 4:
            image_format = mp.ImageFormat.SRGBA
        elif numpy_image.shape[-1] == 3:
            image_format = mp.ImageFormat.SRGB
            numpy_image = cv2.cvtColor(numpy_image, cv2.COLOR_BGR2RGB)

        return mp.Image(image_format=image_format, data=numpy_image)

    def get_bbox_for_mask(self, mask_image: Image):
        # Convert the image to grayscale
        grayscale = mask_image.convert("L")

        # Create a binary mask where non-black pixels are white (255)
        mask_for_bbox = grayscale.point(lambda p: 255 if p > 0 else 0)

        # Get the bounding box of the non-black areas
        bbox = mask_for_bbox.getbbox()

        if bbox != None:
            left = bbox[0]
            upper = bbox[1]
            right = bbox[2]
            lower = bbox[3]

            bbox_width = right - left
            bbox_height = lower - upper

            # expand the box by 20% in each direction if possible
            bbox_padding_x = round(bbox_width * 0.2)
            bbox_padding_y = round(bbox_height * 0.2)

            # left, upper, right, lower
            bbox = (
                # left
                left - bbox_padding_x if left > bbox_padding_x else 0,
                # upper
                upper - bbox_padding_y if upper > bbox_padding_y else 0,
                # right
                right + bbox_padding_x if right < grayscale.width - bbox_padding_x else grayscale.width,
                # lower
                lower + bbox_padding_y if lower < grayscale.height - bbox_padding_y else grayscale.height,
            )

        return bbox

    def __get_mask(
            self,
            image: Image,
            segmenter,
            face_mask: bool,
            background_mask: bool,
            hair_mask: bool,
            body_mask: bool,
            clothes_mask: bool,
            confidence: float,
            refine_mask: bool,
    ) -> tuple[Image, dict]:
        # Retrieve the masks for the segmented image
        media_pipe_image = self.get_mediapipe_image(image=image)
        if any(
                [face_mask, background_mask, hair_mask, body_mask, clothes_mask]
        ):
            segmented_masks = segmenter.segment(media_pipe_image)

        # https://developers.google.com/mediapipe/solutions/vision/image_segmenter#multiclass-model
        # 0 - background
        # 1 - hair
        # 2 - body - skin
        # 3 - face - skin
        # 4 - clothes
        # 5 - others(accessories)
        
        # 存储各个区域的遮罩
        individual_masks = {
            'face': None,
            'background': None,
            'hair': None,
            'body': None,
            'clothes': None
        }
        
        # 用于合并的遮罩列表
        masks_for_merge = []

        image_data = media_pipe_image.numpy_view()
        image_shape = image_data.shape

        # convert the image shape from "rgb" to "rgba" aka add the alpha channel
        if image_shape[-1] == 3:
            image_shape = (image_shape[0], image_shape[1], 4)

        mask_background_array = np.zeros(image_shape, dtype=np.uint8)
        mask_background_array[:] = (0, 0, 0, 255)

        mask_foreground_array = np.zeros(image_shape, dtype=np.uint8)
        mask_foreground_array[:] = (255, 255, 255, 255)

        # 生成各个区域的单独遮罩
        if background_mask:
            condition = (
                np.stack((segmented_masks.confidence_masks[0].numpy_view(),) * image_shape[-1], axis=-1)
                > confidence
            )
            mask_array = np.where(condition, mask_foreground_array, mask_background_array)
            individual_masks['background'] = Image.fromarray(mask_array)
            masks_for_merge.append(mask_array)
            
        if hair_mask:
            condition = (
                np.stack((segmented_masks.confidence_masks[1].numpy_view(),) * image_shape[-1], axis=-1)
                > confidence
            )
            mask_array = np.where(condition, mask_foreground_array, mask_background_array)
            individual_masks['hair'] = Image.fromarray(mask_array)
            masks_for_merge.append(mask_array)
            
        if body_mask:
            condition = (
                np.stack((segmented_masks.confidence_masks[2].numpy_view(),) * image_shape[-1], axis=-1)
                > confidence
            )
            mask_array = np.where(condition, mask_foreground_array, mask_background_array)
            individual_masks['body'] = Image.fromarray(mask_array)
            masks_for_merge.append(mask_array)
            
        if face_mask:
            condition = (
                np.stack((segmented_masks.confidence_masks[3].numpy_view(),) * image_shape[-1], axis=-1)
                > confidence
            )
            mask_array = np.where(condition, mask_foreground_array, mask_background_array)
            individual_masks['face'] = Image.fromarray(mask_array)
            masks_for_merge.append(mask_array)
            
        if clothes_mask:
            condition = (
                np.stack((segmented_masks.confidence_masks[4].numpy_view(),) * image_shape[-1], axis=-1)
                > confidence
            )
            mask_array = np.where(condition, mask_foreground_array, mask_background_array)
            individual_masks['clothes'] = Image.fromarray(mask_array)
            masks_for_merge.append(mask_array)

        # 合并选中的遮罩
        if len(masks_for_merge) == 0:
            merged_mask_arrays = mask_background_array
        else:
            merged_mask_arrays = reduce(np.maximum, masks_for_merge)

        # Create the image
        mask_image = Image.fromarray(merged_mask_arrays)

        # refine the mask by zooming in on the area where we detected our segments
        if refine_mask:
            bbox = self.get_bbox_for_mask(mask_image=mask_image)
            if bbox != None:
                cropped_image_pil = image.crop(bbox)

                cropped_mask_image, cropped_individual_masks = self.__get_mask(image=cropped_image_pil,
                                                   segmenter=segmenter,
                                                   face_mask=face_mask,
                                                   background_mask=background_mask,
                                                   hair_mask=hair_mask,
                                                   body_mask=body_mask,
                                                   clothes_mask=clothes_mask,
                                                   confidence=confidence,
                                                   refine_mask=False,
                                                   )

                # 更新合并遮罩
                updated_mask_image = Image.new('RGBA', image.size, (0, 0, 0))
                updated_mask_image.paste(cropped_mask_image, bbox)
                mask_image = updated_mask_image
                
                # 更新各个区域的遮罩
                for key, cropped_mask in cropped_individual_masks.items():
                    if cropped_mask is not None:
                        updated_individual_mask = Image.new('RGBA', image.size, (0, 0, 0))
                        updated_individual_mask.paste(cropped_mask, bbox)
                        individual_masks[key] = updated_individual_mask

        return mask_image, individual_masks

    def get_mask_images(
            self,
            images, # tensors
            face_mask: bool,
            background_mask: bool,
            hair_mask: bool,
            body_mask: bool,
            clothes_mask: bool,
            confidence: float,
            refine_mask: bool,
    ) -> tuple[list[Image], list[dict]]:
        a_person_mask_generator_model_path = get_a_person_mask_generator_model_path()
        a_person_mask_generator_model_buffer = None

        with open(a_person_mask_generator_model_path, "rb") as f:
            a_person_mask_generator_model_buffer = f.read()

        image_segmenter_base_options = BaseOptions(
            model_asset_buffer=a_person_mask_generator_model_buffer
        )
        options = mp.tasks.vision.ImageSegmenterOptions(
            base_options=image_segmenter_base_options,
            running_mode=VisionRunningMode.IMAGE,
            output_category_mask=True,
        )

        mask_images: list[Image] = []
        individual_masks_list: list[dict] = []

        # Create the image segmenter
        with ImageSegmenter.create_from_options(options) as segmenter:
            for tensor_image in images:
                # Convert the Tensor to a PIL image
                i = 255.0 * tensor_image.cpu().numpy()

                # The media pipe library does a much better job with images with an alpha channel for some reason.
                if i.shape[-1] == 3:  # If the image is RGB
                    # Add a fully transparent alpha channel (255)
                    i = np.dstack((i, np.full((i.shape[0], i.shape[1]), 255)))  # Create an RGBA image

                image_pil = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
                mask_image, individual_masks = self.__get_mask(
                    image=image_pil,
                    segmenter=segmenter,
                    face_mask=face_mask,
                    background_mask=background_mask,
                    hair_mask=hair_mask,
                    body_mask=body_mask,
                    clothes_mask=clothes_mask,
                    confidence=confidence,
                    refine_mask=refine_mask,
                )
                mask_images.append(mask_image)
                individual_masks_list.append(individual_masks)

        return mask_images, individual_masks_list

    def generate_mask(
            self,
            images,
            face_mask: bool,
            background_mask: bool,
            hair_mask: bool,
            body_mask: bool,
            clothes_mask: bool,
            confidence: float,
            refine_mask: bool,
    ):
        """Create a segmentation mask from an image

        Args:
            image (torch.Tensor): The image to create the mask for.
            face_mask (bool): create a mask for the background.
            background_mask (bool): create a mask for the hair.
            hair_mask (bool): create a mask for the body .
            body_mask (bool): create a mask for the face.
            clothes_mask (bool): create a mask for the clothes.
            confidence (float): how confident the model is that the detected item is there.
            break_image_into_tiles ("none" or "auto"): break large images into tiles to improve detection.

        Returns:
            torch.Tensor: The segmentation masks.
        """

        mask_images, individual_masks_list = self.get_mask_images(
            images=images,
            face_mask=face_mask,
            background_mask=background_mask,
            hair_mask=hair_mask,
            body_mask=body_mask,
            clothes_mask=clothes_mask,
            confidence=confidence,
            refine_mask=refine_mask,
        )

        # 转换合并遮罩为tensor
        merged_tensor_masks = []
        for mask_image in mask_images:
            tensor_mask = mask_image.convert("RGB")
            tensor_mask = np.array(tensor_mask).astype(np.float32) / 255.0
            tensor_mask = torch.from_numpy(tensor_mask)[None,]
            tensor_mask = tensor_mask.squeeze(3)[..., 0]
            merged_tensor_masks.append(tensor_mask)

        # 获取图像尺寸用于创建纯黑遮罩
        batch_size = len(images)
        if batch_size > 0:
            image_height, image_width = images[0].shape[:2]
        else:
            image_height, image_width = 256, 256  # 默认尺寸

        # 创建纯黑遮罩的函数
        def create_black_mask():
            """创建纯黑遮罩tensor"""
            black_mask = torch.zeros((batch_size, image_height, image_width), dtype=torch.float32)
            return black_mask

        # 转换各个区域遮罩为tensor的函数
        def convert_mask_to_tensor(mask_image):
            if mask_image is None:
                return None
            tensor_mask = mask_image.convert("RGB")
            tensor_mask = np.array(tensor_mask).astype(np.float32) / 255.0
            tensor_mask = torch.from_numpy(tensor_mask)[None,]
            tensor_mask = tensor_mask.squeeze(3)[..., 0]
            return tensor_mask

        # 收集各个区域的tensor遮罩
        face_tensor_masks = []
        background_tensor_masks = []
        hair_tensor_masks = []
        body_tensor_masks = []
        clothes_tensor_masks = []

        for individual_masks in individual_masks_list:
            face_mask_tensor = convert_mask_to_tensor(individual_masks['face'])
            background_mask_tensor = convert_mask_to_tensor(individual_masks['background'])
            hair_mask_tensor = convert_mask_to_tensor(individual_masks['hair'])
            body_mask_tensor = convert_mask_to_tensor(individual_masks['body'])
            clothes_mask_tensor = convert_mask_to_tensor(individual_masks['clothes'])
            
            face_tensor_masks.append(face_mask_tensor)
            background_tensor_masks.append(background_mask_tensor)
            hair_tensor_masks.append(hair_mask_tensor)
            body_tensor_masks.append(body_mask_tensor)
            clothes_tensor_masks.append(clothes_mask_tensor)

        # 合并遮罩
        merged_masks = torch.cat(merged_tensor_masks, dim=0) if merged_tensor_masks else create_black_mask()

        # 处理各个区域的遮罩：如果启用则使用实际遮罩，否则使用纯黑遮罩
        # 按照RETURN_NAMES的固定顺序：face_mask, background_mask, hair_mask, body_mask, clothes_mask
        
        # Face遮罩
        if face_mask and any(m is not None for m in face_tensor_masks):
            face_masks = torch.cat([m for m in face_tensor_masks if m is not None], dim=0)
        else:
            face_masks = create_black_mask()
        
        # Background遮罩
        if background_mask and any(m is not None for m in background_tensor_masks):
            background_masks = torch.cat([m for m in background_tensor_masks if m is not None], dim=0)
        else:
            background_masks = create_black_mask()
        
        # Hair遮罩
        if hair_mask and any(m is not None for m in hair_tensor_masks):
            hair_masks = torch.cat([m for m in hair_tensor_masks if m is not None], dim=0)
        else:
            hair_masks = create_black_mask()
        
        # Body遮罩
        if body_mask and any(m is not None for m in body_tensor_masks):
            body_masks = torch.cat([m for m in body_tensor_masks if m is not None], dim=0)
        else:
            body_masks = create_black_mask()
        
        # Clothes遮罩
        if clothes_mask and any(m is not None for m in clothes_tensor_masks):
            clothes_masks = torch.cat([m for m in clothes_tensor_masks if m is not None], dim=0)
        else:
            clothes_masks = create_black_mask()

        # 始终返回6个固定的遮罩，按照RETURN_NAMES的顺序
        # ("merged_mask", "face_mask", "background_mask", "hair_mask", "body_mask", "clothes_mask")
        return (merged_masks, face_masks, background_masks, hair_masks, body_masks, clothes_masks)
      
