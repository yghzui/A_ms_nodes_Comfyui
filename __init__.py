# -*- coding: utf-8 -*-
# Created time : 2024/09/12 22:34 
# Auther : ygh
# File   : __init__.py
# Description :
from .nodes.image_nodes import *

from .nodes.mask_nodes import *
from .nodes.text_nodes import *
from .nodes.notice import NoticeSound
from .nodes.resize_image_by_person import ResizeImageByPerson,CropInfoToNumbers
# from .nodes.a_person_face_landmark_mask_generator_comfyui_add_nose import APersonFaceLandmarkMaskGeneratorAddNose
# from .nodes.a_person_face_landmark_mask_generator_comfyui_by_my import APersonFaceLandmarkMaskGenerator
from .nodes.person_and_face.a_person_face_landmark_mask_generator_comfyui import  APersonFaceLandmarkMaskGenerator
from .nodes.math import AspectRatioAdjuster,I2VConfigureNode
from .nodes.face_flip import FaceFlip
from .nodes.create_color_image_mask import CreateColorImageAndMask
# 导入新的批量加载节点
from .nodes.load_image_batch import LoadImageBatchAdvanced
from .nodes.image_mix import ImageMaskedColorFill,ImageBlackColorFill,ImageLayerMix, ImageDualMaskColorFill

NODE_CLASS_MAPPINGS = {
    "LoadAndResizeImageMy": LoadAndResizeImageMy,
    "ResizeImagesAndMasks": ResizeImagesAndMasks,
    "ResizeImageByPerson": ResizeImageByPerson,
    "CropInfoToNumbers": CropInfoToNumbers,
    "CropFaceMy": CropFaceMy,
    "CropFaceMyDetailed": CropFaceMyDetailed,
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
    "PasteFacesAdvanced": PasteFacesAdvanced,
    "PasteMasksMy": PasteMasksMy,
    "GenerateBlackTensor": GenerateWhiteTensor,
    "MyLoadImageListPlus": MyLoadImageListPlus,
    "RemoveGlassesFaceMask": RemoveGlassesFaceMask,
    "AdjustMaskValues": AdjustMaskValues,
    "APersonFaceLandmarkMaskGeneratorByMy":APersonFaceLandmarkMaskGenerator,
    "NoticeSound": NoticeSound,
    "AspectRatioAdjuster": AspectRatioAdjuster,
    "I2VConfigureNode": I2VConfigureNode,
    "ImageFlipNode": FaceFlip,
    "CreateColorImageAndMask": CreateColorImageAndMask,
    "NormalizeMask": NormalizeMask,
    "AnalyzeMask": AnalyzeMask,
    # 注册新的节点
    "LoadImageBatchAdvanced": LoadImageBatchAdvanced,
    "ImageMaskedColorFill": ImageMaskedColorFill,
    "ImageBlackColorFill": ImageBlackColorFill,
    "ImageLayerMix": ImageLayerMix,
    "ImageDualMaskColorFill": ImageDualMaskColorFill,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadAndResizeImageMy": "Load & Resize Image by My",
    "ResizeImagesAndMasks": "Resize Images and Masks by My",
    "ResizeImageByPerson": "Resize Image by Person by My",
    "CropInfoToNumbers": "Crop Info to Numbers by My",
    "CropFaceMy": "Crop Face by My",
    "CropFaceMyDetailed": "Crop Face Detailed by My",
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
    "PasteFacesAdvanced": "PasteFacesAdvanced 粘贴面部 by My",
    "PasteMasksMy": "PasteMasksMy 粘贴面部遮罩 by My",
    "GenerateBlackTensor": "GenerateBlackTensor 生成纯黑张量 by My",
    "MyLoadImageListPlus": "MyLoadImageListPlus 加载图片列表 by My",
    "RemoveGlassesFaceMask": "RemoveGlassesFaceMask 去除眼镜 by My",
    "AdjustMaskValues": "AdjustMaskValues 调整遮罩值 by My",
    "APersonFaceLandmarkMaskGeneratorByMy":"APersonFaceLandmarkMaskGeneratorByMy 生成面部遮罩 by My",
    "NoticeSound": "铃声提醒节点 by My",
    "AspectRatioAdjuster": "宽高比调整节点 by My",
    "I2VConfigureNode": "I2V配置节点 by My",
    "ImageFlipNode": "图像翻转节点 by My",
    "CreateColorImageAndMask": "创建颜色图像和遮罩节点 by My",
    "NormalizeMask": "NormalizeMask 归一化遮罩节点 by My",
    "AnalyzeMask": "AnalyzeMask 分析遮罩节点 by My",
    # 为新节点添加显示名称
    "LoadImageBatchAdvanced": "Load Image Batch (Advanced) 批量加载 by my",
    "ImageMaskedColorFill": "ImageMaskedColorFill 图像颜色填充 by My",
    "ImageBlackColorFill": "ImageBlackColorFill 图像黑色填充 by My",
    "ImageLayerMix": "ImageLayerMix 图层混合 by My",
    "ImageDualMaskColorFill": "ImageDualMaskColorFill 双遮罩不重叠区域颜色填充 by My",
}

WEB_DIRECTORY = "./web/js"
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
"""
 "APersonFaceLandmarkMaskGeneratorAddNose": APersonFaceLandmarkMaskGeneratorAddNose,
    "APersonFaceLandmarkMaskGeneratorByMy": APersonFaceLandmarkMaskGenerator,
 "APersonFaceLandmarkMaskGeneratorAddNose": "APersonFaceLandmarkMaskGeneratorAddNoseOld 生成面部遮罩 by My",
    "APersonFaceLandmarkMaskGeneratorByMy": "APersonFaceLandmarkMaskGeneratorByMyNew 生成面部遮罩 by My",
"""