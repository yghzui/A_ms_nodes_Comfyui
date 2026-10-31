"""ViTPose standalone extractor + POSEDATA converter to common POSE_KEYPOINT.

This node reuses the ONNX models already loaded by WanAnimatePreprocess'
OnnxDetectionModelLoader, but runs ViTPose only: no face crops, no face bboxes,
no retarget image. The raw body/hand/face keypoints are converted to the
SDPose/OpenPose POSE_KEYPOINT JSON shape so they can be saved by
SDPoseSaveJson, rendered by SDPoseDrawKeypoints, and mapped by BodyRatioMapper.
"""
import importlib
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from comfy.utils import ProgressBar


def _wan_nodes():
    """Import WanAnimatePreprocess.nodes lazily, including when A_my_nodes loads first."""
    for name in list(sys.modules):
        if name.endswith("ComfyUI-WanAnimatePreprocess"):
            return importlib.import_module(name + ".nodes")
    return importlib.import_module("custom_nodes.ComfyUI-WanAnimatePreprocess.nodes")


def _flat(points, count, width, height):
    """Normalized (N,3) ViTPose points -> absolute OpenPose (x,y,confidence) triples."""
    arr = np.asarray(points, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 3)
    result = []
    for index in range(count):
        if index < len(arr):
            row = arr[index]
            x = float(row[0]) * width
            y = float(row[1]) * height
            confidence = float(row[2]) if len(row) > 2 else 1.0
            if not np.isfinite(confidence) or confidence < 0:
                confidence = 0.0
            result.extend([x, y, confidence])
        else:
            result.extend([0.0, 0.0, 0.0])
    return result


def _metas_to_common(metas, include_face=True):
    """Convert ViTPose pose metas to the common SDPose/OpenPose POSE_KEYPOINT list."""
    frames = []
    for meta in metas:
        width, height = int(meta["width"]), int(meta["height"])
        body = np.asarray(meta["keypoints_body"], dtype=float).reshape(-1, 3)
        feet = [0.0] * 18
        # VitPose body 18/19 are LToe/RToe; OpenPose foot 1/4 are the big toes.
        for source, target in ((18, 1), (19, 4)):
            if source < len(body):
                feet[target * 3:target * 3 + 3] = [
                    float(body[source, 0]) * width,
                    float(body[source, 1]) * height,
                    float(body[source, 2]) if np.isfinite(body[source, 2]) else 0.0,
                ]
        person = {
            # OpenPose 18 body points: ViTPose's first 18 are already in this order.
            "pose_keypoints_2d": _flat(meta["keypoints_body"], 18, width, height),
            "face_keypoints_2d": (_flat(meta["keypoints_face"], 70, width, height)
                                  if include_face else [0.0] * 210),
            "hand_left_keypoints_2d": _flat(meta["keypoints_left_hand"], 21, width, height),
            "hand_right_keypoints_2d": _flat(meta["keypoints_right_hand"], 21, width, height),
            "foot_keypoints_2d": feet,
        }
        frames.append({
            "people": [person],
            "canvas_width": width,
            "canvas_height": height,
        })
    return frames


class ViTPoseKeypointExtractor:
    """Standalone ViTPose detection: body/face/hands -> common POSE_KEYPOINT, no face crops."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("POSEMODEL",),
                "images": ("IMAGE",),
            },
            "optional": {
                "include_face": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Keep ViTPose face keypoints in the common data. No face images/bboxes are produced.",
                }),
            },
        }

    RETURN_TYPES = ("POSE_KEYPOINT",)
    RETURN_NAMES = ("pose_keypoint",)
    FUNCTION = "extract"
    CATEGORY = "A_my_nodes/ViTPose"
    DESCRIPTION = ("Runs only ViTPose from an existing POSEMODEL and converts body/face/hand keypoints "
                   "to common POSE_KEYPOINT. Face crops/bboxes are intentionally not produced; use a "
                   "separate face task if needed.")

    def extract(self, model, images, include_face=True):
        wan = _wan_nodes()
        detector = model["yolo"]
        pose_model = model["vitpose"]
        batch, height, width, channels = images.shape
        shape = np.array([height, width])[None]
        images_np = images.detach().cpu().numpy() if hasattr(images, "detach") else np.asarray(images)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        input_resolution = (256, 192)
        rescale = 1.25
        detector.reinit()
        pose_model.reinit()
        bar = ProgressBar(batch * 2)
        progress = 0
        try:
            bboxes = []
            for image in tqdm(images_np, total=len(images_np), desc="Detecting ViTPose bboxes"):
                bbox = detector(cv2.resize(image, (640, 640)).transpose(2, 0, 1)[None], shape)[0][0]["bbox"]
                bboxes.append(bbox)
                progress += 1
                if progress % 10 == 0:
                    bar.update_absolute(progress)
            detector.cleanup()

            keypoints = []
            for image, bbox in tqdm(zip(images_np, bboxes), total=len(images_np), desc="Extracting ViTPose keypoints"):
                if bbox is None or bbox[-1] <= 0 or (bbox[2] - bbox[0]) < 10 or (bbox[3] - bbox[1]) < 10:
                    bbox = np.array([0, 0, image.shape[1], image.shape[0]])
                center, scale = wan.bbox_from_detector(bbox, input_resolution, rescale=rescale)
                cropped = wan.crop(image, center, scale, (input_resolution[0], input_resolution[1]))[0]
                normalized = (cropped - mean) / std
                normalized = normalized.transpose(2, 0, 1).astype(np.float32)
                keypoints.append(pose_model(normalized[None], np.array(center)[None], np.array(scale)[None]))
                progress += 1
                if progress % 10 == 0:
                    bar.update_absolute(progress)
            pose_model.cleanup()
            metas = wan.load_pose_metas_from_kp2ds_seq(np.concatenate(keypoints, 0), width=width, height=height)
            return (_metas_to_common(metas, bool(include_face)),)
        finally:
            try:
                detector.cleanup()
            except Exception:
                pass
            try:
                pose_model.cleanup()
            except Exception:
                pass


class ViTPoseDataToKeypoint:
    """Convert an existing PoseAndFaceDetection POSEDATA output to common POSE_KEYPOINT."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"pose_data": ("POSEDATA",)},
            "optional": {
                "include_face": ("BOOLEAN", {"default": True}),
                "use_original": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Use pose_metas_original instead of retargeted pose_metas.",
                }),
            },
        }

    RETURN_TYPES = ("POSE_KEYPOINT",)
    RETURN_NAMES = ("pose_keypoint",)
    FUNCTION = "convert"
    CATEGORY = "A_my_nodes/ViTPose"
    DESCRIPTION = "Converts ViTPose POSEDATA (including PoseAndFaceDetection) to common POSE_KEYPOINT JSON shape."

    def convert(self, pose_data, include_face=True, use_original=True):
        metas = None
        if use_original:
            metas = pose_data.get("pose_metas_original")
        if not metas:
            metas = pose_data.get("pose_metas")
        if not metas:
            raise ValueError("POSEDATA 中没有 pose_metas / pose_metas_original")
        return (_metas_to_common(metas, bool(include_face)),)


class PoseJSONSave:
    """Saves common POSE_KEYPOINT to an explicit JSON path; used by the director for managed data."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_keypoint": ("POSE_KEYPOINT",),
                "path": ("STRING", {"default": "pose_data.json"}),
            },
            "optional": {
                "fps": ("FLOAT", {"default": 24.0, "min": 0.0, "max": 240.0, "step": 0.01}),
                "indent": ("INT", {"default": 2, "min": 0, "max": 8, "step": 1}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("path",)
    FUNCTION = "save"
    CATEGORY = "A_my_nodes/ViTPose"
    OUTPUT_NODE = True
    DESCRIPTION = "Writes common POSE_KEYPOINT JSON (SDPose-compatible {fps,frames}) to a specified path."

    def save(self, pose_keypoint, path, fps=24.0, indent=2):
        target = Path(path).expanduser()
        if not target.is_absolute():
            import folder_paths
            target = Path(folder_paths.get_output_directory())/target
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = pose_keypoint
        if isinstance(payload, list):
            payload = {"fps": float(fps), "frames": payload}
        elif isinstance(payload, dict) and "frames" not in payload:
            payload = {"fps": float(fps), "frames": [payload]}
        target.write_text(json.dumps(payload, ensure_ascii=False,
                                     indent=int(indent) if indent else None), encoding="utf-8")
        return (str(target),)
