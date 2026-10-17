# -*- coding: utf-8 -*-
# 加载 Live 图（动态照片）节点：输出显示图、内嵌视频帧、内嵌音频
# 参考 live_photo_support.py 的动态照片视频流提取逻辑
import io
import os
import re
import hashlib

import av
import numpy as np
import torch
from PIL import Image, ImageOps
import folder_paths

# 注册 HEIF/HEIC 解码支持（失败不影响其它格式）
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass


# ===================== 数据预处理层：动态照片视频流提取 =====================
_MOTION_PHOTO_LENGTH_PATTERN = re.compile(
    rb'Item:Semantic="MotionPhoto".*?Item:Length="(\d+)"',
    re.DOTALL | re.IGNORECASE,
)
_MICRO_VIDEO_OFFSET_PATTERN = re.compile(
    rb'GCamera:MicroVideoOffset="(\d+)"',
    re.IGNORECASE,
)

JPEG_MAGIC = b"\xff\xd8\xff"
HEIF_BRANDS = {b"heic", b"heix", b"hevc", b"hevx", b"mif1", b"msf1", b"heim", b"heis"}

# 可选作输入的图片扩展名（显式列出以兼容 HEIC/HEIF）
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif", ".bmp", ".tif", ".tiff"}


def _is_valid_mp4_start(data, start):
    if start < 0 or start + 12 > len(data):
        return False
    box_size = int.from_bytes(data[start:start + 4], "big")
    return data[start + 4:start + 8] == b"ftyp" and 8 <= box_size <= len(data) - start


def _video_start_from_length(data, pattern):
    match = pattern.search(data)
    if match is None:
        return None
    video_length = int(match.group(1))
    start = len(data) - video_length
    return start if video_length >= 1024 and _is_valid_mp4_start(data, start) else None


def _video_start_from_header(data):
    search_from = 0
    while True:
        ftyp_position = data.find(b"ftyp", search_from)
        if ftyp_position < 0:
            return None
        start = ftyp_position - 4
        if start > 2 and _is_valid_mp4_start(data, start):
            return start
        search_from = ftyp_position + 4


def extract_live_photo_video(data):
    """从动态照片字节流中提取内嵌 MP4，未找到返回 None。"""
    if len(data) < 1024 or not data.startswith(JPEG_MAGIC):
        return None

    start = _video_start_from_length(data, _MOTION_PHOTO_LENGTH_PATTERN)
    if start is None:
        start = _video_start_from_length(data, _MICRO_VIDEO_OFFSET_PATTERN)
    if start is None:
        start = _video_start_from_header(data)
    if start is None:
        return None

    return data[start:]


def heic_to_jpeg_bytes(data):
    """将 HEIC/HEIF 字节流转换为 JPEG 字节流，便于复用动态照片提取逻辑。"""
    with Image.open(io.BytesIO(data)) as image:
        if image.mode != "RGB":
            image = image.convert("RGB")
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=95)
        return output.getvalue()


# ===================== 数据预处理层：媒体解码（PyAV） =====================
def _normalize_audio_tensor(buffer):
    if buffer.dtype.is_floating_point:
        return buffer.float()
    # 整型 PCM 归一化到 [-1, 1]
    info = torch.iinfo(buffer.dtype)
    scale = float(max(abs(info.min), info.max))
    return buffer.float() / scale


def decode_video_frames(video_bytes, max_frames=0):
    """从视频字节流解码图像帧，返回 [N,H,W,3] float32 张量；无帧返回 None。"""
    frames = []
    with av.open(io.BytesIO(video_bytes)) as container:
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format="rgb24"))
            if max_frames and len(frames) >= max_frames:
                break

    if not frames:
        return None

    array = np.stack(frames).astype(np.float32) / 255.0
    return torch.from_numpy(array)


def decode_audio(video_bytes):
    """从视频字节流解码音频，返回 ComfyUI AUDIO 字典；无音频返回 None。"""
    try:
        with av.open(io.BytesIO(video_bytes)) as container:
            if not container.streams.audio:
                return None

            stream = container.streams.audio[0]
            sample_rate = stream.codec_context.sample_rate
            channels = stream.channels

            chunks = []
            for frame in container.decode(streams=stream.index):
                buffer = torch.from_numpy(frame.to_ndarray())
                if buffer.shape[0] != channels:
                    buffer = buffer.view(-1, channels).t()
                chunks.append(_normalize_audio_tensor(buffer))

            if not chunks:
                return None

            waveform = torch.cat(chunks, dim=1)
            return {"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate}
    except Exception as error:
        print(f"⚠️ [LoadLivePhoto] 音频解码失败: {error}")
        return None


# ===================== 节点定义 =====================
class LoadLivePhoto:
    """加载 Live 图（动态照片）节点。

    - image：显示图/封面（始终输出）
    - frames：内嵌视频的全部帧，batch 形状 [N,H,W,3]
    - audio：内嵌视频音频

    当所选文件为普通静态图片时，frames 与 audio 输出 None，退化为普通图片加载器。
    """

    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = [
            name for name in os.listdir(input_dir)
            if os.path.isfile(os.path.join(input_dir, name))
            and os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS
        ]
        return {
            "required": {
                "image": (sorted(files), {"image_upload": True}),
                "max_frames": ("INT", {"default": 0, "min": 0, "max": 1000000, "label": "视频帧上限(0=全部)"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "AUDIO")
    RETURN_NAMES = ("image", "frames", "audio")
    FUNCTION = "load_live_photo"
    CATEGORY = "A_my_nodes/Image"

    def load_live_photo(self, image, max_frames=0):
        image_path = folder_paths.get_annotated_filepath(image)

        display_image = self._load_display_image(image_path)

        with open(image_path, "rb") as source_file:
            raw_data = source_file.read()

        video_bytes = self._extract_video_bytes(raw_data)
        if video_bytes is None:
            return (display_image, None, None)

        frames = decode_video_frames(video_bytes, max_frames)
        audio = decode_audio(video_bytes)
        return (display_image, frames, audio)

    @staticmethod
    def _load_display_image(image_path):
        """按 ComfyUI LoadImage 的方式加载显示图：EXIF 方向校正、转 RGB、归一化到 [0,1]。"""
        image = Image.open(image_path)
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")
        array = np.array(image).astype(np.float32) / 255.0
        return torch.from_numpy(array)[None,]

    @staticmethod
    def _extract_video_bytes(raw_data):
        """从原始文件字节流中提取内嵌视频，纯静态图片返回 None。"""
        # JPG 动态照片：直接在原始字节上提取
        video_bytes = extract_live_photo_video(raw_data)
        if video_bytes is not None:
            return video_bytes

        # HEIC/HEIF 动态照片：先转 JPEG 再提取
        if raw_data[4:8] == b"ftyp" and raw_data[8:12] in HEIF_BRANDS:
            try:
                jpeg_data = heic_to_jpeg_bytes(raw_data)
            except Exception as error:
                print(f"⚠️ [LoadLivePhoto] HEIC 转 JPEG 失败: {error}")
                return None
            return extract_live_photo_video(jpeg_data)

        return None

    @classmethod
    def IS_CHANGED(cls, image, max_frames=0):
        image_path = folder_paths.get_annotated_filepath(image)
        digest = hashlib.sha256()
        with open(image_path, "rb") as source_file:
            digest.update(source_file.read())
        return digest.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(cls, image, max_frames=0):
        if not folder_paths.exists_annotated_filepath(image):
            return f"Invalid image file: {image}"
        return True
