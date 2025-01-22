# -*- coding: utf-8 -*-
# Created time : 2024/09/12 22:34 
# Auther : ygh
# File   : __init__.py
# Description :
from .nodes.image_nodes import *

from .nodes.mask_nodes import *
from .nodes.text_nodes import *
from .nodes.a_person_face_landmark_mask_generator_comfyui_add_nose import APersonFaceLandmarkMaskGeneratorAddNose
from .nodes.a_person_face_landmark_mask_generator_comfyui_by_my import APersonFaceLandmarkMaskGenerator

NODE_CLASS_MAPPINGS = {
    "LoadAndResizeImageMy": LoadAndResizeImageMy,
    "CropFaceMy": CropFaceMy,
    "CreateFaceBboxMask": CreateBboxMask,
    "CreateTextMask": CreateTextMask,
    "CoordinateTessPosNeg": CoordinateTessPosNeg,
    "TextMaskMy": TextMaskMy,
    "GroundingDinoGetBbox": GroundingDinoGetBbox,
    "MaskAdd": MaskAdd,
    "MaskSubtract": MaskSubtract,
    "MaskOverlap": MaskOverlap,
    "FilterClothingWords": FilterClothingWords,
    "PasteFacesMy": PasteFacesMy,
    "PasteMasksMy": PasteMasksMy,
    "GenerateBlackTensor": GenerateWhiteTensor,
    "MyLoadImageListPlus": MyLoadImageListPlus,
    "RemoveGlassesFaceMask": RemoveGlassesFaceMask,
    "AdjustMaskValues": AdjustMaskValues,
    "APersonFaceLandmarkMaskGeneratorAddNose": APersonFaceLandmarkMaskGeneratorAddNose,
    "APersonFaceLandmarkMaskGeneratorByMy": APersonFaceLandmarkMaskGenerator,

}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadAndResizeImageMy": "Load & Resize Image by My",
    "CropFaceMy": "Crop Face by My",
    "CreateFaceBboxMask": "Create Face Bbox Mask by My",
    "CreateTextMask": "Text Mask path by My",
    "example_class": "example_class by My",
    "TextMaskMy": "TextMask by My",
    "GroundingDinoGetBbox": "GroundingDinoGetBbox by My",
    "CoordinateTessPosNeg": "CoordinateTessPosNeg by My",
    "MaskAdd": "MaskAdd + by My",
    "MaskSubtract": "MaskSubtract - by My",
    "MaskOverlap": "MaskOverlap 重叠度 by My",
    "FilterClothingWords": "FilterClothingWords 过滤服装关键词 by My",
    "PasteFacesMy": "PasteFacesMy 粘贴面部 by My",
    "PasteMasksMy": "PasteMasksMy 粘贴面部遮罩 by My",
    "GenerateBlackTensor": "GenerateBlackTensor 生成纯黑张量 by My",
    "MyLoadImageListPlus": "MyLoadImageListPlus 加载图片列表 by My",
    "RemoveGlassesFaceMask": "RemoveGlassesFaceMask 去除眼镜 by My",
    "AdjustMaskValues": "AdjustMaskValues 调整遮罩值 by My",
    "APersonFaceLandmarkMaskGeneratorAddNose": "APersonFaceLandmarkMaskGeneratorAddNoseOld 生成面部遮罩 by My",
    "APersonFaceLandmarkMaskGeneratorByMy": "APersonFaceLandmarkMaskGeneratorByMyNew 生成面部遮罩 by My",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
