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
from .nodes.math import AspectRatioAdjuster,I2VConfigureNode, FramesSplitCalculator, FramesSegmentSlicer, ImagesConcatWithOverlap, ImageTakeLast
from .nodes.face_flip import FaceFlip
from .nodes.create_color_image_mask import CreateColorImageAndMask
# 导入新的批量加载节点
from .nodes.load_image_batch import LoadImageBatchAdvanced, LoadImageByIndex
from .nodes.image_mix import ImageMaskedColorFill,ImageBlackColorFill,ImageLayerMix, ImageDualMaskColorFill
from .nodes.load_lora_batch import LoadLoraBatch
from .nodes.wan_video_lora_batch import WanVideoLoraBatch
from .nodes.load_lora_merge import LoadLoraMerge
from .nodes.index_selector import IndexSelector
from .nodes.my_batch_manager import MyBatchManager
from .nodes.image_batch_accumulator import ImageBatchAccumulator
from .nodes.show_result_last import ShowResultLast
from .nodes.manual_video_input import ManualVideoInput
from .nodes.load_video import LoadVideoFromFolder
from .nodes.load_latent_upload import LoadLatentUpload

# 新增导入：图像序列与遮罩生成节点
from .nodes.math import ImageToSequenceWithMask
# 新增导入：文本批量输入
from .nodes.text_input_batch import TextInputBatch, TextDictSplitter
# 新增导入：文本字典检查节点
from .nodes.text_dict_checker import TextDictChecker
# 新增导入：图像扩展节点
from .nodes.image_expand import ImageExpand
# 新增导入：增强版图像混合节点
from .nodes.ImageBlendAdvance_my import ImageBlendAdvanceMy
# 新增导入：图像拼接多输入节点
from .nodes.imageconcatmultims import ImageConcatMultiMs
from .nodes.resolutionpreset import ResolutionPresetNode
from .nodes.load_ui_node_value import GetNodeInputValue
from .nodes.wan_video_double_stream import WanVideoDoubleStream



# 延迟注册路由 - 确保在ComfyUI完全初始化后注册
def register_custom_routes():
    """延迟注册自定义路由"""
    try:
        # 延迟导入路由模块，避免启动时的依赖问题
        from . import routes
        
        # 检查是否已经注册过
        if hasattr(routes, '_routes_registered') and routes._routes_registered:
            print("⚠️ 路由已经注册过了，跳过重复注册")
            return
            
        routes.register_routes()
    except Exception as e:
        print(f"❌ 自定义路由注册失败: {e}")

# 延迟注册路由，避免在模块导入时就执行
def delayed_register_routes():
    """延迟注册路由的包装函数"""
    try:
        from . import routes
        routes.delayed_register_routes()
    except Exception as e:
        print(f"❌ 延迟路由注册失败: {e}")

# 在模块加载完成后尝试注册路由
try:
    delayed_register_routes()
except:
    # 如果路由注册失败，不影响节点的正常加载
    pass

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
    "NoticeSound": NoticeSound,
    "AspectRatioAdjuster": AspectRatioAdjuster,
    "I2VConfigureNode": I2VConfigureNode,
    "ResolutionPresetNode": ResolutionPresetNode,
    "FramesSplitCalculator": FramesSplitCalculator,
    "FramesSegmentSlicer": FramesSegmentSlicer,
    "ImagesConcatWithOverlap": ImagesConcatWithOverlap,
    "ImageFlipNode": FaceFlip,
    "CreateColorImageAndMask": CreateColorImageAndMask,
    "NormalizeMask": NormalizeMask,
    "AnalyzeMask": AnalyzeMask,
    # 注册新的节点
    "LoadImageBatchAdvanced": LoadImageBatchAdvanced,
    "LoadImageByIndex": LoadImageByIndex,
    "ImageMaskedColorFill": ImageMaskedColorFill,
    "ImageBlackColorFill": ImageBlackColorFill,
    "ImageLayerMix": ImageLayerMix,
    "ImageDualMaskColorFill": ImageDualMaskColorFill,
    "LoadLoraBatch": LoadLoraBatch,
    "WanVideoLoraBatch": WanVideoLoraBatch,
    "ShowResultLast": ShowResultLast,
    "ManualVideoInput": ManualVideoInput,
    "LoadVideoFromFolder": LoadVideoFromFolder,
    "LoadLatentUpload": LoadLatentUpload,
    # 新增注册：图像序列与遮罩生成节点
    "ImageToSequenceWithMask": ImageToSequenceWithMask,
    # 新增注册：文本批量输入
    "TextInputBatch": TextInputBatch,
    "TextDictSplitter": TextDictSplitter,
    # 新增注册：文本字典检查节点
    "TextDictChecker": TextDictChecker,
    # 新增注册：图像扩展节点
    "ImageExpand": ImageExpand,
    # 新增注册：增强版图像混合节点
    "ImageBlendAdvanceMy": ImageBlendAdvanceMy,
    # 新增注册：图像截取节点
    "ImageTakeLast": ImageTakeLast,
    "LoadLoraMerge": LoadLoraMerge,
    "IndexSelector": IndexSelector,
    "MyBatchManager": MyBatchManager,
    # 新增注册：图像拼接多输入节点
    "ImageConcatMultiMs": ImageConcatMultiMs,
    "ImageBatchAccumulator": ImageBatchAccumulator,
    "GetNodeInputValue": GetNodeInputValue,
    "WanVideoDoubleStream": WanVideoDoubleStream,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadAndResizeImageMy": "Load & Resize Image by ms",
    "ResizeImagesAndMasks": "Resize Images and Masks by ms",
    "ResizeImageByPerson": "Resize Image by Person by ms",
    "CropInfoToNumbers": "Crop Info to Numbers by ms",
    "CropFaceMy": "Crop Face by ms",
    "CropFaceMyDetailed": "Crop Face Detailed by ms",
    "CreateFaceBboxMask": "Create Face Bbox Mask by ms",
    "CreateTextMask": "Text Mask path by ms",
    "example_class": "example_class by ms",
    "GroundingDinoGetBbox": "GroundingDinoGetBbox by ms",
    "CoordinateTessPosNeg": "CoordinateTessPosNeg by ms",
    "MaskAdd": "MaskAdd + 遮罩相加 by ms",
    "MaskSubtract": "MaskSubtract - 遮罩相减 by ms",
    "MaskOverlap": "MaskOverlap 重叠度 by ms",
    "FilterClothingWords": "FilterClothingWords 过滤服装关键词 by ms",
    "PasteFacesMy": "PasteFacesMy 粘贴面部 by ms",
    "PasteFacesAdvanced": "PasteFacesAdvanced 粘贴面部 by ms",
    "PasteMasksMy": "PasteMasksMy 粘贴面部遮罩 by ms",
    "GenerateBlackTensor": "GenerateBlackTensor 生成纯黑张量 by ms",
    "MyLoadImageListPlus": "MyLoadImageListPlus 加载图片列表 by ms",
    "RemoveGlassesFaceMask": "RemoveGlassesFaceMask 去除眼镜 by ms",
    "AdjustMaskValues": "AdjustMaskValues 调整遮罩值 by ms",
    "NoticeSound": "铃声提醒节点 by ms",
    "AspectRatioAdjuster": "宽高比调整节点 by ms",
    "I2VConfigureNode": "I2V配置节点 by ms",
    "ResolutionPresetNode": "宽高预设节点 by ms",
    "FramesSplitCalculator": "循环 按帧数切分计算(含重叠) by ms",
    "FramesSegmentSlicer": "循环 按索引截取图像与遮罩段 by ms",
    "ImagesConcatWithOverlap": "循环 按重叠覆盖拼接图像 by ms",
    "ImageFlipNode": "图像翻转节点 by ms",
    "CreateColorImageAndMask": "创建颜色图像和遮罩节点 by ms",
    "NormalizeMask": "NormalizeMask 归一化遮罩节点 by ms",
    "AnalyzeMask": "AnalyzeMask 分析遮罩节点 by ms",
    # 为新节点添加显示名称
    "LoadImageBatchAdvanced": "Load Image Batch (Advanced) 批量加载 by ms",
    "LoadImageByIndex": "Load Image By Index 按索引加载图像 by ms",
    "ImageMaskedColorFill": "ImageMaskedColorFill 图像颜色填充 by ms",
    "ImageBlackColorFill": "ImageBlackColorFill 图像黑色填充 by ms",
    "ImageLayerMix": "ImageLayerMix 图层混合 by ms",
    "ImageDualMaskColorFill": "ImageDualMaskColorFill 双遮罩不重叠区域颜色填充 by ms",
    "LoadLoraBatch": "LoadLoraBatch 批量加载LoRA by ms",
    "WanVideoLoraBatch": "WanVideoLoraBatch 批量收集WanVideo LoRA by ms",
    "ShowResultLast": "Show VHS_FILENAMES by path  显示视频结果 通过路径by ms",
    "ManualVideoInput": "手动输入视频文件名 by ms",
    "LoadVideoFromFolder": "批量加载视频文件 by ms",
    "LoadLatentUpload": "Load Latent (Upload) 加载Latent文件(支持上传) by ms",
    # 新增显示名称
    "ImageToSequenceWithMask": "ImageToSequenceWithMask 图像序列与遮罩生成 by ms",
    "TextInputBatch": "PromptInputBatch 提示词 by ms",
    "TextDictSplitter": "TextDictSplitter 字典拆分 by ms",
    "TextDictChecker": "TextDictChecker 文本字典提示词检查 by ms",
    "ImageExpand": "ImageExpand 图像扩展节点 by ms",
    "ImageBlendAdvanceMy": "ImageBlendAdvanceMy 增强版图像混合节点 by ms",
    # 新增显示名称：图像截取节点
    "ImageTakeLast": "ImageTakeLast 图像截取(后N张) by ms",
    "LoadLoraMerge": "LoadLoraMerge 合并加载LoRA by ms",
    "IndexSelector": "IndexSelector 索引选择器 by ms",
    "MyBatchManager": "MyBatchManager 循环管理器 by ms",
    # 新增显示名称：图像拼接多输入节点
    "ImageConcatMultiMs": "ImageConcatMultiMs 图像拼接(多输入动态) by ms",
    "ImageBatchAccumulator": "ImageBatchAccumulator 图像批量累加器 by ms",
    "GetNodeInputValue": "获取节点输入值 by ms",
    "WanVideoDoubleStream": "WanVideoDoubleStream 双流视频LoRA加载 by ms",
}

WEB_DIRECTORY = "./web/js"
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
