# -*- coding: utf-8 -*-
import torch
import numpy as np
import cv2
import onnxruntime
import os
import sys
import folder_paths

# --------------------------------------------------------------------------------
# SCRFD Logic (Ported and Adapted)
# --------------------------------------------------------------------------------

def distance2bbox(points, distance, max_shape=None):
    """Decode distance prediction to bounding box."""
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = np.clip(x1, 0, max_shape[1])
        y1 = np.clip(y1, 0, max_shape[0])
        x2 = np.clip(x2, 0, max_shape[1])
        y2 = np.clip(y2, 0, max_shape[0])
    return np.stack([x1, y1, x2, y2], axis=-1)

def _scaled_box_from_face(cx, cy, w_box, h_box, scale_factor, w_orig, h_orig):
    w_box = max(1.0, w_box * scale_factor)
    h_box = max(1.0, h_box * scale_factor)
    nx1 = cx - w_box / 2
    ny1 = cy - h_box / 2
    nx2 = cx + w_box / 2
    ny2 = cy + h_box / 2
    nx1 = max(0.0, nx1)
    ny1 = max(0.0, ny1)
    nx2 = min(float(w_orig), nx2)
    ny2 = min(float(h_orig), ny2)
    return nx1, ny1, nx2, ny2

def _center_max_crop_by_ratio(nx1, ny1, nx2, ny2, ratio):
    w = nx2 - nx1
    h = ny2 - ny1
    if w <= 0 or h <= 0:
        return int(nx1), int(ny1), int(nx2), int(ny2)
    if ratio <= 0:
        ratio = 1.0
    if w / h >= ratio:
        new_h = h
        new_w = h * ratio
    else:
        new_w = w
        new_h = w / ratio
    new_w = max(1.0, new_w)
    new_h = max(1.0, new_h)
    cx = (nx1 + nx2) / 2
    cy = (ny1 + ny2) / 2
    x1 = cx - new_w / 2
    y1 = cy - new_h / 2
    x2 = x1 + new_w
    y2 = y1 + new_h
    if x1 < nx1:
        x2 += (nx1 - x1)
        x1 = nx1
    if y1 < ny1:
        y2 += (ny1 - y1)
        y1 = ny1
    if x2 > nx2:
        x1 -= (x2 - nx2)
        x2 = nx2
    if y2 > ny2:
        y1 -= (y2 - ny2)
        y2 = ny2
    ix1 = int(round(x1))
    iy1 = int(round(y1))
    ix2 = int(round(x2))
    iy2 = int(round(y2))
    ix1 = max(0, min(ix1, int(nx2)))
    iy1 = max(0, min(iy1, int(ny2)))
    ix2 = max(ix1 + 1, min(ix2, int(nx2)))
    iy2 = max(iy1 + 1, min(iy2, int(ny2)))
    return ix1, iy1, ix2, iy2

class SCRFD:
    def __init__(self, model_file):
        self.model_file = model_file
        # Force CPU execution as requested
        try:
            self.session = onnxruntime.InferenceSession(self.model_file, providers=['CPUExecutionProvider'])
        except Exception as e:
            print(f"Error loading model {model_file}: {e}")
            raise e
        
        self.center_cache = {}
        self.nms_thresh = 0.4
        # self.det_thresh will be passed in detect
        self.fmc = 3
        self._feat_stride_fpn = [8, 16, 32]
        self._num_anchors = 2
        self._init_vars()

    def _init_vars(self):
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        self.input_name = inputs[0].name
        
        # Simple heuristic to determine output structure
        if len(outputs) == 6:
            self.fmc = 3
            self.use_kps = False
        elif len(outputs) == 9:
            self.fmc = 3
            self.use_kps = True
        else:
            self.use_kps = False

    def detect(self, img, input_size, thresh=0.5):
        # Resize and Pad logic
        im_ratio = float(img.shape[0]) / img.shape[1]
        model_ratio = float(input_size[1]) / input_size[0]
        
        if im_ratio > model_ratio:
            new_height = input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = input_size[0]
            new_height = int(new_width * im_ratio)
            
        det_scale = float(new_height) / img.shape[0]
        resized_img = cv2.resize(img, (new_width, new_height))
        
        # Create input tensor
        det_img = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
        det_img[:new_height, :new_width, :] = resized_img

        # SCRFD expects RGB image if model trained on RGB. 
        # OpenCV blobFromImage with swapRB=True converts BGR to RGB.
        # Since our input 'img' is BGR (converted before calling detect), swapRB=True makes it RGB for the model.
        blob = cv2.dnn.blobFromImage(det_img, 1.0/128.0, input_size, (127.5, 127.5, 127.5), swapRB=True)
        net_outs = self.session.run(None, {self.input_name: blob})

        input_height = blob.shape[2]
        input_width = blob.shape[3]
        
        scores_list = []
        bboxes_list = []
        
        for idx, stride in enumerate(self._feat_stride_fpn):
            scores = net_outs[idx]
            bbox_preds = net_outs[idx + self.fmc]
            
            height = input_height // stride
            width = input_width // stride
            key = (height, width, stride)
            
            if key in self.center_cache:
                anchor_centers = self.center_cache[key]
            else:
                anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
                anchor_centers = (anchor_centers * stride).reshape( (-1, 2) )
                if self._num_anchors > 1:
                    anchor_centers = np.stack([anchor_centers]*self._num_anchors, axis=1).reshape( (-1,2) )
                if len(self.center_cache) < 100:
                    self.center_cache[key] = anchor_centers
            
            # Filter by threshold
            idx_anchor = np.where(scores[:, 0] >= thresh)[0]
            
            if len(idx_anchor) > 0:
                scores = scores[idx_anchor]
                bbox_preds = bbox_preds[idx_anchor] * stride
                anchor_centers = anchor_centers[idx_anchor]
                
                bboxes = distance2bbox(anchor_centers, bbox_preds)
                scores_list.append(scores)
                bboxes_list.append(bboxes)

        if len(scores_list) == 0:
            return np.array([])
            
        scores = np.vstack(scores_list)
        bboxes = np.vstack(bboxes_list)
        bboxes = bboxes / det_scale
        
        # Combine scores and bboxes
        # scores is [N, 1] (class score), bboxes is [N, 4]
        # output needs to be [N, 5] (x1, y1, x2, y2, score)
        return np.hstack((bboxes, scores))

    def nms(self, dets):
        if dets.shape[0] == 0:
            return dets
            
        thresh = self.nms_thresh
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]

        return dets[keep]

# --------------------------------------------------------------------------------
# Node Implementation
# --------------------------------------------------------------------------------

class CropFaceFast:
    @classmethod
    def INPUT_TYPES(s):
        # Locate onnx files
        # We assume models/detection/face exists as per user request
        bbox_path = os.path.join(folder_paths.models_dir, "detection", "face")
        if not os.path.exists(bbox_path):
             os.makedirs(bbox_path, exist_ok=True)
        
        # List all .onnx files
        model_files = [f for f in os.listdir(bbox_path) if f.endswith('.onnx')]
        if not model_files:
            model_files = ["None"]

        return {
            "required": {
                "image": ("IMAGE",),
                "model_name": (model_files,),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "scale_factor": ("FLOAT", {"default": 1.5, "min": 0.5, "max": 5.0, "step": 0.01}),
                "square_crop": ("BOOLEAN", {"default": False, "tooltip": "是否强制正方形裁剪（以宽为基准）"}),
                "use_output_size": ("BOOLEAN", {"default": True, "tooltip": "是否按输出尺寸进行缩放"}),
                "output_width": ("INT", {"default": 512, "min": 0, "max": 4096, "step": 8}),
                "output_height": ("INT", {"default": 512, "min": 0, "max": 4096, "step": 8}),
                "face_indices": ("STRING", {"default": "-1", "tooltip": "选择要处理的人脸索引，-1表示所有人脸，多个索引用逗号分隔，如'0,1'"}),
                "only_max_score": ("BOOLEAN", {"default": False, "tooltip": "是否只输出检测分数最高的人脸"}),
            },
            "optional": {
                "input_mask": ("MASK", {"default": None, "tooltip": "可选的输入遮罩，将与图像使用相同的裁剪逻辑"}),
                "empty_mask_mode": (["black", "white", "original"], {"default": "original", "tooltip": "当裁剪区域内的mask全黑时: black=保持黑色, white=使用全白mask, original=使用原始图像区域作为mask"})
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "MASK", "STRING")
    RETURN_NAMES = ("cropped_face", "bbox_mask", "cropped_input_mask", "crop_info")
    FUNCTION = "crop_face"
    CATEGORY = "My_node/image"

    def __init__(self):
        self.scrfd_model = None
        self.current_model_path = None

    def crop_face(self, image, model_name, threshold, scale_factor, square_crop, use_output_size, output_width, output_height, face_indices, only_max_score, input_mask=None, empty_mask_mode="original"):
        
        # 1. Load Model
        model_path = os.path.join(folder_paths.models_dir, "detection", "face", model_name)
        if model_name == "None":
            orig_h = image.shape[1]
            orig_w = image.shape[2]
            if use_output_size:
                if output_width == 0 and output_height == 0:
                    dummy_w, dummy_h = orig_w, orig_h
                elif output_width == 0:
                    dummy_h = max(1, output_height)
                    dummy_w = max(1, int(round(orig_w * (dummy_h / orig_h)))) if orig_h > 0 else 1
                elif output_height == 0:
                    dummy_w = max(1, output_width)
                    dummy_h = max(1, int(round(orig_h * (dummy_w / orig_w)))) if orig_w > 0 else 1
                else:
                    dummy_w, dummy_h = output_width, output_height
            else:
                dummy_w, dummy_h = orig_w, orig_h
            return self._create_dummy_output(image.shape[0], dummy_w, dummy_h, orig_w, orig_h)
        if self.scrfd_model is None or self.current_model_path != model_path:
            if not os.path.exists(model_path):
                print(f"Model not found: {model_path}")
                raise ValueError(f"Model not found: {model_path}")
            self.scrfd_model = SCRFD(model_path)
            self.current_model_path = model_path
        
        if self.scrfd_model is None:
             raise ValueError(f"Could not load model: {model_name}")

        # Prepare inputs
        # ComfyUI image is [B, H, W, C] in RGB, float 0-1
        images_np = (image.cpu().numpy() * 255).astype(np.uint8)
        
        # Prepare mask if exists
        # ComfyUI mask is [B, H, W], float 0-1
        if input_mask is not None:
            masks_np = input_mask.cpu().numpy()
            if len(masks_np.shape) == 2:
                 masks_np = masks_np[np.newaxis, ...]
        else:
            masks_np = None

        out_images = []
        out_bbox_masks = []
        out_cropped_masks = []
        out_crop_infos = []

        batch_size = images_np.shape[0]

        for i in range(batch_size):
            img = images_np[i] # [H, W, 3] RGB
            
            # Create full-size bbox mask base
            h_orig, w_orig = img.shape[:2]
            bbox_mask_base = np.zeros((h_orig, w_orig), dtype=np.float32)
            
            # Handle input mask for this image
            if masks_np is not None:
                # Handle case where mask batch < image batch
                mask_idx = min(i, masks_np.shape[0] - 1)
                curr_input_mask = masks_np[mask_idx] # [H, W]
            else:
                curr_input_mask = None

            # Detect
            # Convert RGB to BGR for detection (because we use swapRB=True in blobFromImage which expects BGR input to convert to RGB)
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            # Run detection
            dets = self.scrfd_model.detect(img_bgr, (640, 640), thresh=threshold)
            
            if dets.size > 0:
                dets = self.scrfd_model.nms(dets)

            # Filter faces
            faces_to_process = []
            if dets.size > 0:
                # Filter by only_max_score
                if only_max_score:
                    faces_to_process = [dets[0]]
                else:
                     # Parse indices
                    if face_indices == "-1":
                        faces_to_process = [d for d in dets]
                    else:
                        try:
                            indices = [int(idx.strip()) for idx in face_indices.split(',')]
                            for idx in indices:
                                if 0 <= idx < len(dets):
                                    faces_to_process.append(dets[idx])
                        except:
                            # If parsing fails, default to all? Or none? 
                            # User said: "选择要处理的人脸索引... -1表示所有人脸"
                            # If invalid, safe to fallback to all or log warning.
                            print(f"Warning: Invalid face_indices '{face_indices}', using all detected faces.")
                            faces_to_process = [d for d in dets]
            
            if not faces_to_process:
                # No faces found or selected
                print(f"No faces to process in image {i}")
                # Output black dummy for this image to keep batch alignment?
                # But if we return multiple faces per image, batch size changes.
                # ComfyUI nodes that return variable batch size are fine.
                # But if we return empty list, it might break downstream.
                # Let's just continue, effectively filtering out this image if no face.
                # BUT if ALL images have no faces, we must return something.
                continue

            output_ratio = None
            if use_output_size and output_width > 0 and output_height > 0:
                output_ratio = output_width / output_height
            face_params = []
            ratios = []
            for face in faces_to_process:
                x1, y1, x2, y2 = face[:4]
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                w_box = max(1.0, x2 - x1)
                h_box = max(1.0, y2 - y1)
                w_scaled = w_box * scale_factor
                h_scaled = h_box * scale_factor
                ratio = w_scaled / h_scaled if h_scaled != 0 else 1.0
                ratios.append(ratio)
                face_params.append((cx, cy, w_box, h_box))
            if square_crop:
                reference_ratio = 1.0
            else:
                reference_ratio = min(ratios, key=lambda r: abs(r - 1.0)) if ratios else 1.0

            for face_idx, face in enumerate(faces_to_process):
                x1, y1, x2, y2 = face[:4]
                cx, cy, w_box, h_box = face_params[face_idx]
                sx1, sy1, sx2, sy2 = _scaled_box_from_face(cx, cy, w_box, h_box, scale_factor, w_orig, h_orig)
                bx1, by1, bx2, by2 = _center_max_crop_by_ratio(sx1, sy1, sx2, sy2, reference_ratio)
                if output_ratio is not None:
                    fx1, fy1, fx2, fy2 = _center_max_crop_by_ratio(bx1, by1, bx2, by2, output_ratio)
                else:
                    fx1, fy1, fx2, fy2 = bx1, by1, bx2, by2
                crop_info_str = f"[{fx1},{fy1},{fx2},{fy2}]"
                out_crop_infos.append(crop_info_str)
                this_bbox_mask = bbox_mask_base.copy()
                if fx2 > fx1 and fy2 > fy1:
                    this_bbox_mask[fy1:fy2, fx1:fx2] = 1.0
                out_bbox_masks.append(torch.from_numpy(this_bbox_mask))
 
                if fx2 > fx1 and fy2 > fy1:
                    crop_img = img[fy1:fy2, fx1:fx2, :]
                    crop_h, crop_w = crop_img.shape[:2]
                    if use_output_size:
                        if output_width == 0 and output_height == 0:
                            target_w, target_h = crop_w, crop_h
                        elif output_width == 0:
                            target_h = max(1, output_height)
                            target_w = max(1, int(round(crop_w * (target_h / crop_h)))) if crop_h > 0 else 1
                        elif output_height == 0:
                            target_w = max(1, output_width)
                            target_h = max(1, int(round(crop_h * (target_w / crop_w)))) if crop_w > 0 else 1
                        else:
                            target_w, target_h = output_width, output_height
                    else:
                        target_w, target_h = crop_w, crop_h
                    if target_w == crop_w and target_h == crop_h:
                        resized_face = crop_img
                    else:
                        resized_face = cv2.resize(crop_img, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
                    out_images.append(resized_face)
                    
                    # Crop Input Mask
                    # "裁剪后的输入mask"
                    if curr_input_mask is not None:
                        crop_mask = curr_input_mask[fy1:fy2, fx1:fx2]
                        if target_w == crop_w and target_h == crop_h:
                            resized_mask = crop_mask
                        else:
                            resized_mask = cv2.resize(crop_mask, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                        
                        # Handle empty mask logic
                        if np.max(resized_mask) <= 0.001:
                             if empty_mask_mode == "white":
                                 resized_mask = np.ones((target_h, target_w), dtype=np.float32)
                             elif empty_mask_mode == "original":
                                 # Use image brightness
                                 gray = cv2.cvtColor(resized_face, cv2.COLOR_RGB2GRAY)
                                 resized_mask = gray.astype(np.float32) / 255.0
                        
                        out_cropped_masks.append(resized_mask.astype(np.float32))
                    else:
                        # No input mask provided, return zeros? 
                        # Reference implementation returns zeros if invalid/no mask.
                        out_cropped_masks.append(np.zeros((target_h, target_w), dtype=np.float32))

                else:
                    # Invalid crop
                    if use_output_size:
                        if output_width == 0 and output_height == 0:
                            target_w, target_h = w_orig, h_orig
                        elif output_width == 0:
                            target_h = max(1, output_height)
                            target_w = max(1, int(round(w_orig * (target_h / h_orig)))) if h_orig > 0 else 1
                        elif output_height == 0:
                            target_w = max(1, output_width)
                            target_h = max(1, int(round(h_orig * (target_w / w_orig)))) if w_orig > 0 else 1
                        else:
                            target_w, target_h = output_width, output_height
                    else:
                        target_w, target_h = w_orig, h_orig
                    empty_img = np.zeros((target_h, target_w, 3), dtype=np.uint8)
                    out_images.append(empty_img)
                    out_cropped_masks.append(np.zeros((target_h, target_w), dtype=np.float32))

        # Pack outputs
        if not out_images:
            orig_h = images_np.shape[1]
            orig_w = images_np.shape[2]
            if use_output_size:
                if output_width == 0 and output_height == 0:
                    dummy_w, dummy_h = orig_w, orig_h
                elif output_width == 0:
                    dummy_h = max(1, output_height)
                    dummy_w = max(1, int(round(orig_w * (dummy_h / orig_h)))) if orig_h > 0 else 1
                elif output_height == 0:
                    dummy_w = max(1, output_width)
                    dummy_h = max(1, int(round(orig_h * (dummy_w / orig_w)))) if orig_w > 0 else 1
                else:
                    dummy_w, dummy_h = output_width, output_height
            else:
                dummy_w, dummy_h = orig_w, orig_h
            return self._create_dummy_output(1, dummy_w, dummy_h, orig_w, orig_h)

        # Convert images to tensor [B, H, W, C]
        max_h = max(img.shape[0] for img in out_images)
        max_w = max(img.shape[1] for img in out_images)
        padded_images = []
        padded_masks = []
        for img, mask in zip(out_images, out_cropped_masks):
            h, w = img.shape[:2]
            if h != max_h or w != max_w:
                padded_img = np.zeros((max_h, max_w, 3), dtype=img.dtype)
                padded_img[:h, :w, :] = img
            else:
                padded_img = img
            padded_images.append(padded_img)
            mh, mw = mask.shape[:2]
            if mh != max_h or mw != max_w:
                padded_mask = np.zeros((max_h, max_w), dtype=np.float32)
                padded_mask[:mh, :mw] = mask
            else:
                padded_mask = mask
            padded_masks.append(padded_mask.astype(np.float32))
        processed_images = np.array(padded_images).astype(np.float32) / 255.0
        output_image_tensor = torch.from_numpy(processed_images)
        
        # BBox masks [B, H_orig, W_orig]
        output_bbox_mask_tensor = torch.stack(out_bbox_masks)
        
        # Cropped masks [B, H_out, W_out]
        output_cropped_mask_tensor = torch.from_numpy(np.stack(padded_masks))
        
        # Crop info string
        output_crop_info = "; ".join(out_crop_infos)

        return (output_image_tensor, output_bbox_mask_tensor, output_cropped_mask_tensor, output_crop_info)

    def _create_dummy_output(self, batch_size, out_w, out_h, orig_w, orig_h):
        dummy_img = torch.zeros((batch_size, out_h, out_w, 3), dtype=torch.float32)
        dummy_mask = torch.zeros((batch_size, out_h, out_w), dtype=torch.float32)
        dummy_bbox_mask = torch.zeros((batch_size, orig_h, orig_w), dtype=torch.float32)
        return (dummy_img, dummy_bbox_mask, dummy_mask, "")

def _run_crop_strategy_tests():
    cases = [
        {"faces": [(10, 20, 110, 60)], "img": (200, 300), "scale": 1.0, "output": (0, 0)},
        {"faces": [(0, 0, 50, 100), (120, 40, 220, 120)], "img": (160, 260), "scale": 1.2, "output": (0, 0)},
        {"faces": [(250, 50, 299, 90), (10, 10, 80, 60)], "img": (100, 300), "scale": 2.0, "output": (512, 288)},
        {"faces": [(5, 90, 25, 110)], "img": (128, 128), "scale": 1.0, "output": (256, 256)},
        {"faces": [(30, 40, 160, 220), (180, 10, 250, 90)], "img": (256, 256), "scale": 0.8, "output": (320, 240)},
    ]
    for case in cases:
        h_orig, w_orig = case["img"]
        scale = case["scale"]
        output_w, output_h = case["output"]
        output_ratio = None
        if output_w > 0 and output_h > 0:
            output_ratio = output_w / output_h
        ratios = []
        face_params = []
        for face in case["faces"]:
            x1, y1, x2, y2 = face
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            w_box = max(1.0, x2 - x1)
            h_box = max(1.0, y2 - y1)
            w_scaled = w_box * scale
            h_scaled = h_box * scale
            ratio = w_scaled / h_scaled if h_scaled != 0 else 1.0
            ratios.append(ratio)
            face_params.append((cx, cy, w_box, h_box))
        reference_ratio = min(ratios, key=lambda r: abs(r - 1.0)) if ratios else 1.0
        for cx, cy, w_box, h_box in face_params:
            sx1, sy1, sx2, sy2 = _scaled_box_from_face(cx, cy, w_box, h_box, scale, w_orig, h_orig)
            bx1, by1, bx2, by2 = _center_max_crop_by_ratio(sx1, sy1, sx2, sy2, reference_ratio)
            if output_ratio is not None:
                fx1, fy1, fx2, fy2 = _center_max_crop_by_ratio(bx1, by1, bx2, by2, output_ratio)
                ratio = (fx2 - fx1) / (fy2 - fy1)
                assert abs(ratio - output_ratio) <= 0.05
            else:
                fx1, fy1, fx2, fy2 = bx1, by1, bx2, by2
            assert fx2 > fx1 and fy2 > fy1
            assert 0 <= fx1 <= fx2 <= w_orig
            assert 0 <= fy1 <= fy2 <= h_orig

if __name__ == "__main__":
    _run_crop_strategy_tests()
    print("crop strategy tests passed")
