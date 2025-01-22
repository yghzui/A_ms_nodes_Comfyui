import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import cv2 as cv
import torch
import numpy as np
from PIL import Image
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import folder_paths

print("Mediapipe version:", mp.__version__)


def format_landmarks(landmarks):
    formatted_landmarks = []
    for landmark in landmarks:
        formatted_landmarks.append(landmark)
    return formatted_landmarks


def get_a_person_mask_generator_model_path() -> str:
    model_folder_name = "mediapipe"
    model_name = "face_landmarker.task"

    model_folder_path = os.path.join(folder_paths.models_dir, model_folder_name)
    model_file_path = os.path.join(model_folder_path, model_name)

    if not os.path.exists(model_file_path):
        import wget
        # https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
        model_url = f"https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/{model_name}"
        print(f"Downloading '{model_name}' model")
        os.makedirs(model_folder_path, exist_ok=True)
        wget.download(model_url, model_file_path)

    return model_file_path


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
                "nose": true_widget,
                "number_of_faces": (
                    "INT",
                    {"default": 1, "min": 1, "max": 10, "step": 1},
                ),
                "confidence": (
                    "FLOAT",
                    {"default": 0.40, "min": 0.01, "max": 1.0, "step": 0.01},
                ),
            },
        }

    CATEGORY = "A Person Mask Generator - David Bielejeski"
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
            nose: bool,
            number_of_faces: int,
            confidence: float,
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

        Returns:
            torch.Tensor: The segmentation masks.
        """

        res_masks = []
        model_path = get_a_person_mask_generator_model_path()
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(base_options=base_options,
                                               output_face_blendshapes=True,
                                               output_facial_transformation_matrixes=True,
                                               num_faces=number_of_faces)
        # detector = vision.FaceLandmarker.create_from_options(options)
        # with mp.solutions.face_mesh.FaceMesh(
        #     static_image_mode=True,
        #     refine_landmarks=True,
        #     max_num_faces=number_of_faces,
        #     min_detection_confidence=confidence,
        # ) as face_mesh:
        with vision.FaceLandmarker.create_from_options(options) as face_mesh:

            for image in images:
                # Convert the Tensor to a PIL image
                i = 255.0 * image.cpu().numpy()  # [H, W, 3]
                i = i.transpose((1, 2, 0))
                # print(i.shape)
                image_pil = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
                # # Convert PIL image to mediapipe Image
                # mp_image = np.asarray(image_pil)
                # 将数组中的值限制在 0 到 255 之间，并转换为 uint8 类型
                clipped_i = np.clip(i, 0, 255).astype(np.uint8)

                # 创建 mediapipe.Image 对象
                # 假设图像格式为 SRGB，你可以根据实际情况调整
                image_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=clipped_i)
                # Process the image
                results = (
                    face_mesh.detect(image_mp)
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
                        ]
                    )
                    else None
                )

                img_width, img_height = image_pil.size
                mask = np.zeros((img_height, img_width), dtype=np.uint8)

                if results and results.face_landmarks:
                    mesh_coords = [
                        (int(point.x * img_width), int(point.y * img_height))
                        for point in format_landmarks(results.face_landmarks[0])
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
                            lips_coords = [mesh_coords[p] for p in self.FACEMESH_LIPS]
                            cv.fillPoly(
                                mask, [np.array(lips_coords, dtype=np.int32)], 255
                            )
                        if nose:
                            nose_coords = [mesh_coords[p] for p in self.FACEMESH_NOSE]
                            cv.fillPoly(
                                mask, [np.array(nose_coords, dtype=np.int32)], 255
                            )

                # Create the image
                mask_image = Image.fromarray(mask)

                # convert PIL image to tensor image
                tensor_mask = mask_image.convert("RGB")
                tensor_mask = np.array(tensor_mask).astype(np.float32) / 255.0
                tensor_mask = torch.from_numpy(tensor_mask)[None,]
                tensor_mask = tensor_mask.squeeze(3)[..., 0]

                res_masks.append(tensor_mask)

        return (torch.cat(res_masks, dim=0),)
