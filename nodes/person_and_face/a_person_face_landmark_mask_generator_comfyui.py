import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import cv2 as cv
import torch
import numpy as np
from PIL import Image
import mediapipe as mp
from .a_person_mask_generator_comfyui import APersonMaskGenerator


class APersonFaceLandmarkMaskGenerator:
    # https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/python/solutions/face_mesh_connections.py
    # order matters for these
    FACEMESH_FACE_OVAL = [
        10,
        338,
        297,
        332,
        284,
        251,
        389,
        356,
        454,
        323,
        361,
        288,
        397,
        365,
        379,
        378,
        400,
        377,
        152,
        148,
        176,
        149,
        150,
        136,
        172,
        58,
        132,
        93,
        234,
        127,
        162,
        21,
        54,
        103,
        67,
        109,
    ]

    FACEMESH_LEFT_EYEBROW = [336, 296, 334, 293, 300, 276, 283, 282, 295, 285]
    FACEMESH_RIGHT_EYEBROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]

    FACEMESH_LEFT_EYE = [
        362,
        382,
        381,
        380,
        374,
        373,
        390,
        249,
        263,
        466,
        388,
        387,
        386,
        385,
        384,
        398,
    ]
    FACEMESH_RIGHT_EYE = [
        33,
        7,
        163,
        144,
        145,
        153,
        154,
        155,
        133,
        173,
        157,
        158,
        159,
        160,
        161,
        246,
    ]

    FACEMESH_LEFT_PUPIL = [474, 475, 476, 477]
    FACEMESH_RIGHT_PUPIL = [469, 470, 471, 472]

    FACEMESH_LIPS = [
        61,
        146,
        91,
        181,
        84,
        17,
        314,
        405,
        321,
        375,
        291,
        308,
        324,
        318,
        402,
        317,
        14,
        87,
        178,
        88,
        95,
        185,
        40,
        39,
        37,
        0,
        267,
        269,
        270,
        409,
        415,
        310,
        311,
        312,
        13,
        82,
        81,
        42,
        183,
        78,
    ]
    FACEMESH_NOSE = [
        168,
        417,
        465,
        343,
        437,
        420,
        429,
        279,
        294,
        305,
        459,
        370,
        141,
        238,
        240,
        64,
        49,
        209,
        198,
        217,
        114,
        188,
        245,
        193,

    ]
    def __init__(self):
        pass

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
                "face": false_widget,
                "left_eyebrow": false_widget,
                "right_eyebrow": false_widget,
                "left_eye": true_widget,
                "right_eye": true_widget,
                "left_pupil": false_widget,
                "right_pupil": false_widget,
                "lips": true_widget,
                "fill_lips": (
                    "BOOLEAN",
                    {"default": True, "label_on": "fill", "label_off": "outline"},
                ),
                "lips_expand_up": (
                    "INT",
                    {"default": 0, "min": 0, "max": 1000, "step": 1},
                ),
                "lips_expand_down": (
                    "INT",
                    {"default": 0, "min": 0, "max": 1000, "step": 1},
                ),
                "lips_expand_left": (
                    "INT",
                    {"default": 0, "min": 0, "max": 1000, "step": 1},
                ),
                "lips_expand_right": (
                    "INT",
                    {"default": 0, "min": 0, "max": 1000, "step": 1},
                ),
                "nose": true_widget,
                "number_of_faces": (
                    "INT",
                    {"default": 1, "min": 1, "max": 10, "step": 1},
                ),
                "confidence": (
                    "FLOAT",
                    {"default": 0.40, "min": 0.01, "max": 1.0, "step": 0.01},
                ),
                "refine_mask": true_widget,
            },
        }

    CATEGORY = "My_node/face_mask"
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("masks",)

    FUNCTION = "generate_mask"

    def generate_mask(
        self,
        images,
        face: bool,
        left_eyebrow: bool,
        right_eyebrow: bool,
        left_eye: bool,
        right_eye: bool,
        left_pupil: bool,
        right_pupil: bool,
        lips: bool,
        fill_lips: bool,
        lips_expand_up: int,
        lips_expand_down: int,
        lips_expand_left: int,
        lips_expand_right: int,
        nose: bool,
        number_of_faces: int,
        confidence: float,
        refine_mask: bool,
    ):
        """Create a segmentation mask from an image

        Args:
            image (torch.Tensor): The image to create the mask for.
            face (bool): create a mask for the face.
            left_eyebrow (bool): create a mask for the left eyebrow.
            right_eyebrow (bool): create a mask for the right eyebrow.
            left_eye (bool): create a mask for the left eye.
            right_eye (bool): create a mask for the right eye.
            left_pupil (bool): create a mask for the left eye pupil.
            right_pupil (bool): create a mask for the right eye pupil.
            lips (bool): create a mask for the lips.
            fill_lips (bool): fill the lips mask.
            lips_expand_up (int): expand the lips mask up.
            lips_expand_down (int): expand the lips mask down.
            lips_expand_left (int): expand the lips mask left.
            lips_expand_right (int): expand the lips mask right.
            nose (bool): create a mask for the nose

        Returns:
            torch.Tensor: The segmentation masks.
        """

        # use our APersonMaskGenerator with the face specified to get the image we should focus on

        a_person_mask_generator = None
        face_masks = None
        if refine_mask:
            a_person_mask_generator = APersonMaskGenerator()
            face_masks = a_person_mask_generator.get_mask_images(
                images=images,
                face_mask=True,
                background_mask=False,
                hair_mask=False,
                body_mask=False,
                clothes_mask=False,
                confidence=0.4,
                refine_mask=True,
            )

        res_masks = []
        with mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            refine_landmarks=True,
            max_num_faces=number_of_faces,
            min_detection_confidence=confidence,
        ) as face_mesh:

            for index, image in enumerate(images):
                # Convert the Tensor to a PIL image
                i = 255.0 * image.cpu().numpy()
                image_pil = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

                do_uncrop = False
                cropped_image_pil = image_pil

                # use the face mask to refine the image we are processing
                if refine_mask:
                    face_mask = face_masks[index]

                    # create the bounding box around the mask
                    bbox = a_person_mask_generator.get_bbox_for_mask(mask_image=face_mask)

                    if bbox != None:
                        cropped_image_pil = image_pil.crop(bbox)
                        do_uncrop = True

                # Process the image
                results = (
                    face_mesh.process(np.asarray(cropped_image_pil))
                    if any(
                        [
                            face,
                            left_eyebrow,
                            right_eyebrow,
                            left_eye,
                            right_eye,
                            left_pupil,
                            right_pupil,
                            lips,
                            nose,
                        ]
                    )
                    else None
                )

                img_width, img_height = cropped_image_pil.size
                mask = np.zeros((img_height, img_width), dtype=np.uint8)

                if results and results.multi_face_landmarks:
                    mesh_coords = [
                        (int(point.x * img_width), int(point.y * img_height))
                        for point in results.multi_face_landmarks[0].landmark
                    ]

                    if face:
                        face_coords = [mesh_coords[p] for p in self.FACEMESH_FACE_OVAL]
                        cv.fillPoly(mask, [np.array(face_coords, dtype=np.int32)], 255)
                    else:
                        if left_eyebrow:
                            left_eyebrow_coords = [
                                mesh_coords[p] for p in self.FACEMESH_LEFT_EYEBROW
                            ]
                            cv.fillPoly(
                                mask,
                                [np.array(left_eyebrow_coords, dtype=np.int32)],
                                255,
                            )

                        if right_eyebrow:
                            right_eyebrow_coords = [
                                mesh_coords[p] for p in self.FACEMESH_RIGHT_EYEBROW
                            ]
                            cv.fillPoly(
                                mask,
                                [np.array(right_eyebrow_coords, dtype=np.int32)],
                                255,
                            )

                        if left_eye:
                            left_eye_coords = [
                                mesh_coords[p] for p in self.FACEMESH_LEFT_EYE
                            ]
                            cv.fillPoly(
                                mask, [np.array(left_eye_coords, dtype=np.int32)], 255
                            )

                        if right_eye:
                            right_eye_coords = [
                                mesh_coords[p] for p in self.FACEMESH_RIGHT_EYE
                            ]
                            cv.fillPoly(
                                mask, [np.array(right_eye_coords, dtype=np.int32)], 255
                            )

                        if left_pupil:
                            left_pupil_coords = [
                                mesh_coords[p] for p in self.FACEMESH_LEFT_PUPIL
                            ]
                            cv.fillPoly(
                                mask, [np.array(left_pupil_coords, dtype=np.int32)], 255
                            )

                        if right_pupil:
                            right_pupil_coords = [
                                mesh_coords[p] for p in self.FACEMESH_RIGHT_PUPIL
                            ]
                            cv.fillPoly(
                                mask,
                                [np.array(right_pupil_coords, dtype=np.int32)],
                                255,
                            )

                        if lips:
                            # 将嘴唇轮廓点分为上下唇
                            # 上唇关键点
                            upper_lip_indices = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 308, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78]
                            # 下唇关键点
                            lower_lip_indices = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78]
                            
                            # 获取上下唇的坐标点
                            upper_lip_coords = [mesh_coords[i] for i in upper_lip_indices]
                            lower_lip_coords = [mesh_coords[i] for i in lower_lip_indices]
                            
                            # 使用固定的61和308点作为左右方向
                            left_point = mesh_coords[61]  # 左边点
                            right_point = mesh_coords[308]  # 右边点
                            
                            # 创建临时mask用于绘制原始嘴唇形状
                            temp_mask = np.zeros((img_height, img_width), dtype=np.uint8)
                            
                            if fill_lips:
                                # 填充完整的嘴唇区域（包括上下唇之间的区域）
                                full_lip_coords = np.array(upper_lip_coords + lower_lip_coords[::-1], dtype=np.int32)
                                cv.fillPoly(temp_mask, [full_lip_coords], 255)
                            else:
                                # 只绘制轮廓
                                cv.polylines(temp_mask, [np.array(upper_lip_coords + lower_lip_coords[::-1], dtype=np.int32)], True, 255, thickness=2)
                            
                            # 如果需要扩展
                            if any([lips_expand_up, lips_expand_down, lips_expand_left, lips_expand_right]):
                                # 创建结构元素用于扩展
                                if lips_expand_left > 0:
                                    kernel = np.zeros((3, lips_expand_left * 2 + 1), dtype=np.uint8)
                                    kernel[1, :lips_expand_left] = 1
                                    temp_mask = cv.dilate(temp_mask, kernel, iterations=1)
                                
                                if lips_expand_right > 0:
                                    kernel = np.zeros((3, lips_expand_right * 2 + 1), dtype=np.uint8)
                                    kernel[1, lips_expand_right:] = 1
                                    temp_mask = cv.dilate(temp_mask, kernel, iterations=1)
                                
                                if lips_expand_up > 0:
                                    kernel = np.zeros((lips_expand_up * 2 + 1, 3), dtype=np.uint8)
                                    kernel[:lips_expand_up, 1] = 1
                                    temp_mask = cv.dilate(temp_mask, kernel, iterations=1)
                                
                                if lips_expand_down > 0:
                                    kernel = np.zeros((lips_expand_down * 2 + 1, 3), dtype=np.uint8)
                                    kernel[lips_expand_down:, 1] = 1
                                    temp_mask = cv.dilate(temp_mask, kernel, iterations=1)
                            
                            # 将临时mask复制到最终mask
                            mask[temp_mask > 0] = 255
                            
                        if nose:
                            nose_coords = [mesh_coords[p] for p in self.FACEMESH_NOSE]
                            cv.fillPoly(
                                mask, [np.array(nose_coords, dtype=np.int32)], 255
                            )

                # Create the image
                mask_image = Image.fromarray(mask)

                if do_uncrop:
                    uncropped_mask_image = Image.new('RGBA', image_pil.size, (0, 0, 0))
                    uncropped_mask_image.paste(mask_image, bbox)
                    mask_image = uncropped_mask_image

                # convert PIL image to tensor image
                tensor_mask = mask_image.convert("RGB")
                tensor_mask = np.array(tensor_mask).astype(np.float32) / 255.0
                tensor_mask = torch.from_numpy(tensor_mask)[None,]
                tensor_mask = tensor_mask.squeeze(3)[..., 0]

                res_masks.append(tensor_mask)

        return (torch.cat(res_masks, dim=0),)
