# -*- coding: utf-8 -*-
# Created time : 2024/09/12 22:34 
# Auther : ygh
# File   : __init__.py
# Description :
from .nodes.image_nodes import *
from .nodes.mask_nodes import *
from .nodes.text_nodes import *
from .nodes.notice import NoticeSound
from .nodes.crop_person import CropPerson,CropInfoToNumbers
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
from .nodes.any_batch_accumulator import AnyBatchAccumulator,AnyBatchListConverter,AnyStopOnNone#,AnyDataAnalyzer
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
from .nodes.wan_video_double_stream_asset import WanVideoDoubleStreamAsset
from .nodes.crop_face_fast import CropFaceFast
from .nodes.save_image_toggle import SaveImageWithToggle
from .nodes.auto_vae_encode import AutoVAEEncode


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
    "CropPerson": CropPerson,
    "CropInfoToNumbers": CropInfoToNumbers,
    "CropFaceMy": CropFaceMy,
    "CropFaceMyDetailed": CropFaceMyDetailed,
    "CropFaceFast": CropFaceFast,
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
    "AnyBatchAccumulator": AnyBatchAccumulator,
    "AnyBatchListConverter": AnyBatchListConverter,
    "AnyStopOnNone": AnyStopOnNone,
    # "AnyDataAnalyzer": AnyDataAnalyzer,
    "GetNodeInputValue": GetNodeInputValue,
    "WanVideoDoubleStream": WanVideoDoubleStream,
    "WanVideoDoubleStreamAsset": WanVideoDoubleStreamAsset,
    "SaveImageWithToggle": SaveImageWithToggle,
    "AutoVAEEncode": AutoVAEEncode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadAndResizeImageMy": "Load & Resize Image by ms",
    "ResizeImagesAndMasks": "Resize Images and Masks by ms",
    "CropPerson": "Crop Person by ms",
    "CropInfoToNumbers": "Crop Info to Numbers by ms",
    "CropFaceMy": "Crop Face by ms",
    "CropFaceMyDetailed": "Crop Face Detailed by ms",
    "CropFaceFast": "Crop Face Fast (SCRFD) by ms",
    "CreateFaceBboxMask": "Create Face Bbox Mask by ms",
    "CreateTextMask": "Create Text Mask Path by ms",
    "example_class": "Example Class by ms",
    "GroundingDinoGetBbox": "Grounding Dino Get Bbox by ms",
    "CoordinateTessPosNeg": "Coordinate Tess Pos Neg by ms",
    "MaskAdd": "Mask Add by ms",
    "MaskSubtract": "Mask Subtract by ms",
    "MaskOverlap": "Mask Overlap by ms",
    "FilterClothingWords": "Filter Clothing Words by ms",
    "PasteFacesMy": "Paste Faces My by ms",
    "PasteFacesAdvanced": "Paste Faces Advanced by ms",
    "PasteMasksMy": "Paste Masks My by ms",
    "GenerateBlackTensor": "Generate Black Tensor by ms",
    "MyLoadImageListPlus": "Load Image List Plus by ms",
    "RemoveGlassesFaceMask": "Remove Glasses Face Mask by ms",
    "AdjustMaskValues": "Adjust Mask Values by ms",
    "NoticeSound": "Notice Sound Node by ms",
    "AspectRatioAdjuster": "Aspect Ratio Adjuster by ms",
    "I2VConfigureNode": "I2V Configure Node by ms",
    "ResolutionPresetNode": "Resolution Preset Node by ms",
    "FramesSplitCalculator": "Frames Split Calculator by ms",
    "FramesSegmentSlicer": "Frames Segment Slicer by ms",
    "ImagesConcatWithOverlap": "Images Concat With Overlap by ms",
    "ImageFlipNode": "Image Flip Node by ms",
    "CreateColorImageAndMask": "Create Color Image And Mask by ms",
    "NormalizeMask": "Normalize Mask by ms",
    "AnalyzeMask": "Analyze Mask by ms",
    "LoadImageBatchAdvanced": "Load Image Batch by ms",
    "LoadImageByIndex": "Load Image By Index by ms",
    "ImageMaskedColorFill": "Image Masked Color Fill by ms",
    "ImageBlackColorFill": "Image Black Color Fill by ms",
    "ImageLayerMix": "Image Layer Mix by ms",
    "ImageDualMaskColorFill": "Image Dual Mask Color Fill by ms",
    "LoadLoraBatch": "Load Lora Batch by ms",
    "WanVideoLoraBatch": "Wan Video Lora Batch by ms",
    "ShowResultLast": "Show Result Last (VHS Filenames) by ms",
    "ManualVideoInput": "Manual Video Input by ms",
    "LoadVideoFromFolder": "Load Video From Folder by ms",
    "LoadLatentUpload": "Load Latent Upload by ms",
    "ImageToSequenceWithMask": "Image To Sequence With Mask by ms",
    "TextInputBatch": "Text Input Batch by ms",
    "TextDictSplitter": "Text Dict Splitter by ms",
    "TextDictChecker": "Text Dict Checker by ms",
    "ImageExpand": "Image Expand by ms",
    "ImageBlendAdvanceMy": "Image Blend Advance My by ms",
    "ImageTakeLast": "Image Take Last by ms",
    "LoadLoraMerge": "Load Lora Merge by ms",
    "IndexSelector": "Index Selector by ms",
    "MyBatchManager": "My Batch Manager by ms",
    "ImageConcatMultiMs": "Image Concat Multi by ms",
    "ImageBatchAccumulator": "Image Batch Accumulator by ms",
    "AnyBatchAccumulator": "Any Batch Accumulator by ms",
    "AnyBatchListConverter": "Any Batch List Converter by ms",
    "AnyStopOnNone": "Any Stop On None by ms",
    "GetNodeInputValue": "Get Node Input Value by ms",
    "WanVideoDoubleStream": "Wan Video Double Stream by ms",
    "WanVideoDoubleStreamAsset": "Wan Video Double Stream Asset by ms",
    "SaveImageWithToggle": "Save Image With Toggle by ms",
    "AutoVAEEncode": "Auto VAE Encode (Switch) by ms",
}

WEB_DIRECTORY = "./web/js"
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
