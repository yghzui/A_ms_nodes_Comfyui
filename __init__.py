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
from .nodes.any_batch_accumulator import AnyBatchAccumulator,AnyBatchListConverter,AnyStopOnNone,AnyValidityChecker
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
from .nodes.auto_vae_encode import AutoVAEEncode, FluxLatentMaskBinder
from .nodes.auto_latent_source import AutoLatentSource
from .nodes.workflow_force_rerun import WorkflowForceRerunPassthrough
from .nodes.group_switch_any import GroupSwitchAny
from .nodes.multi_input_state_mapper import MultiInputStateMapper
from .nodes.multi_image_condition_reference import MultiImageConditionReference
from .nodes.workflow_group_preset_manager import WorkflowGroupPresetManager
from .nodes.bg_removal_colorize import BackgroundRemovalColorize


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
    "LoadAndResizeImageMy": LoadAndResizeImageMy, # 加载并调整图片大小
    "ResizeImagesAndMasks": ResizeImagesAndMasks, # 调整图片和遮罩大小
    "CropPerson": CropPerson, # 裁剪人物区域
    "CropInfoToNumbers": CropInfoToNumbers, # 裁剪信息转数值
    "CropFaceMy": CropFaceMy, # 裁剪面部
    "CropFaceMyDetailed": CropFaceMyDetailed, # 裁剪面部（详细）
    "CropFaceFast": CropFaceFast, # 快速裁剪面部（SCRFD）
    "CreateFaceBboxMask": CreateBboxMask, # 创建面部边界框遮罩
    "CreateTextMask": CreateTextMask, # 创建文本遮罩路径
    "CoordinateTessPosNeg": CoordinateTessPosNeg, # 坐标 Tess 正负
    "GroundingDinoGetBbox": GroundingDinoGetBbox, # Grounding Dino 获取边界框
    "MaskAdd": MaskAdd, # 遮罩相加
    "MaskSubtract": MaskSubtract, # 遮罩相减
    "MaskOverlap": MaskOverlap, # 遮罩重叠
    "FilterClothingWords": FilterClothingWords, # 过滤服装词汇
    "PasteFacesMy": PasteFacesMy, # 粘贴面部
    "PasteFacesAdvanced": PasteFacesAdvanced, # 粘贴面部（高级）
    "PasteMasksMy": PasteMasksMy, # 粘贴遮罩
    "GenerateBlackTensor": GenerateWhiteTensor, # 生成黑色张量
    "GenerateBlackMaskByMode": GenerateBlackMaskByMode, # 按模式生成纯黑遮罩
    "MyLoadImageListPlus": MyLoadImageListPlus, # 加载图片列表增强版
    "RemoveGlassesFaceMask": RemoveGlassesFaceMask, # 移除眼镜面部遮罩
    "AdjustMaskValues": AdjustMaskValues, # 调整遮罩值
    "NoticeSound": NoticeSound, # 提示音节点
    "AspectRatioAdjuster": AspectRatioAdjuster, # 宽高比调整
    "I2VConfigureNode": I2VConfigureNode, # 图生视频配置节点
    "ResolutionPresetNode": ResolutionPresetNode, # 分辨率预设节点
    "FramesSplitCalculator": FramesSplitCalculator, # 帧分割计算器
    "FramesSegmentSlicer": FramesSegmentSlicer, # 帧分段切片器
    "ImagesConcatWithOverlap": ImagesConcatWithOverlap, # 图片拼接（带重叠）
    "ImageFlipNode": FaceFlip, # 图像翻转
    "CreateColorImageAndMask": CreateColorImageAndMask, # 创建彩色图片和遮罩
    "NormalizeMask": NormalizeMask, # 标准化遮罩
    "AnalyzeMask": AnalyzeMask, # 分析遮罩
    # 批量加载相关
    "LoadImageBatchAdvanced": LoadImageBatchAdvanced, # 批量加载图片
    "LoadImageByIndex": LoadImageByIndex, # 按索引加载图片
    "ImageMaskedColorFill": ImageMaskedColorFill, # 遮罩区域颜色填充
    "ImageBlackColorFill": ImageBlackColorFill, # 黑色填充
    "ImageLayerMix": ImageLayerMix, # 图像图层混合
    "ImageDualMaskColorFill": ImageDualMaskColorFill, # 双遮罩颜色填充
    "LoadLoraBatch": LoadLoraBatch, # 批量加载 LoRA
    "WanVideoLoraBatch": WanVideoLoraBatch, # Wan 视频 LoRA 批量
    "ShowResultLast": ShowResultLast, # 显示最后结果（VHS 文件名）
    "ManualVideoInput": ManualVideoInput, # 手动视频输入
    "LoadVideoFromFolder": LoadVideoFromFolder, # 从文件夹加载视频
    "LoadLatentUpload": LoadLatentUpload, # 加载上传的潜空间数据
    # 图像序列与遮罩生成
    "ImageToSequenceWithMask": ImageToSequenceWithMask, # 图像转序列带遮罩
    # 文本批量输入
    "TextInputBatch": TextInputBatch, # 文本批量输入
    "TextDictSplitter": TextDictSplitter, # 文本字典分割
    # 文本字典检查
    "TextDictChecker": TextDictChecker, # 文本字典检查
    # 图像扩展
    "ImageExpand": ImageExpand, # 图像扩展
    # 增强版图像混合
    "ImageBlendAdvanceMy": ImageBlendAdvanceMy, # 增强版图像混合
    # 图像截取
    "ImageTakeLast": ImageTakeLast, # 截取最后几张图片
    "LoadLoraMerge": LoadLoraMerge, # 加载并合并 LoRA
    "IndexSelector": IndexSelector, # 索引选择器
    "MyBatchManager": MyBatchManager, # 批量管理器
    # 图像拼接多输入
    "ImageConcatMultiMs": ImageConcatMultiMs, # 图像拼接多输入
    "ImageBatchAccumulator": ImageBatchAccumulator, # 图像批量累加器
    "AnyBatchAccumulator": AnyBatchAccumulator, # 任意类型批量累加器
    "AnyBatchListConverter": AnyBatchListConverter, # 任意类型批量列表转换器
    "AnyStopOnNone": AnyStopOnNone, # 遇到空值停止
    "AnyValidityChecker": AnyValidityChecker, # 任意类型数据校验器
    # "AnyDataAnalyzer": AnyDataAnalyzer,
    "GetNodeInputValue": GetNodeInputValue, # 获取节点输入值
    "WanVideoDoubleStream": WanVideoDoubleStream, # Wan 视频双流
    "WanVideoDoubleStreamAsset": WanVideoDoubleStreamAsset, # Wan 视频双流资产
    "SaveImageWithToggle": SaveImageWithToggle, # 保存带开关的图片
    "AutoVAEEncode": AutoVAEEncode, # 自动 VAE 编码（切换）
    "AutoLatentSource": AutoLatentSource, # 自动潜空间来源选择
    "FluxLatentMaskBinder": FluxLatentMaskBinder, # Flux 潜空间遮罩绑定
    "WorkflowForceRerunPassthrough": WorkflowForceRerunPassthrough, # 强制让工作流每次更新
    "GroupSwitchAny": GroupSwitchAny, # 任意类型分组切换
    "MultiInputStateMapper": MultiInputStateMapper, # 多输入存在状态映射
    "MultiImageConditionReference": MultiImageConditionReference, # 多图条件参考
    "WorkflowGroupPresetManager": WorkflowGroupPresetManager, # 工作流切换组管理器
    "BackgroundRemovalColorize": BackgroundRemovalColorize, # 背景移除换色增强
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadAndResizeImageMy": "Load & Resize Image by ms", # 加载并调整图片大小
    "ResizeImagesAndMasks": "Resize Images and Masks by ms", # 调整图片和遮罩大小
    "CropPerson": "Crop Person by ms", # 裁剪人物区域
    "CropInfoToNumbers": "Crop Info to Numbers by ms", # 裁剪信息转数值
    "CropFaceMy": "Crop Face by ms", # 裁剪面部
    "CropFaceMyDetailed": "Crop Face Detailed by ms", # 裁剪面部（详细）
    "CropFaceFast": "Crop Face Fast (SCRFD) by ms", # 快速裁剪面部（SCRFD）
    "CreateFaceBboxMask": "Create Face Bbox Mask by ms", # 创建面部边界框遮罩
    "CreateTextMask": "Create Text Mask Path by ms", # 创建文本遮罩路径
    "example_class": "Example Class by ms", # 示例类
    "GroundingDinoGetBbox": "Grounding Dino Get Bbox by ms", # Grounding Dino 获取边界框
    "CoordinateTessPosNeg": "Coordinate Tess Pos Neg by ms", # 坐标 Tess 正负
    "MaskAdd": "Mask Add by ms", # 遮罩相加
    "MaskSubtract": "Mask Subtract by ms", # 遮罩相减
    "MaskOverlap": "Mask Overlap by ms", # 遮罩重叠
    "FilterClothingWords": "Filter Clothing Words by ms", # 过滤服装词汇
    "PasteFacesMy": "Paste Faces My by ms", # 粘贴面部
    "PasteFacesAdvanced": "Paste Faces Advanced by ms", # 粘贴面部（高级）
    "PasteMasksMy": "Paste Masks My by ms", # 粘贴遮罩
    "GenerateBlackTensor": "Generate Black Tensor by ms", # 生成黑色张量
    "GenerateBlackMaskByMode": "Generate Black Mask By Mode by ms", # 按模式生成纯黑遮罩
    "MyLoadImageListPlus": "Load Image List Plus by ms", # 加载图片列表增强版
    "RemoveGlassesFaceMask": "Remove Glasses Face Mask by ms", # 移除眼镜面部遮罩
    "AdjustMaskValues": "Adjust Mask Values by ms", # 调整遮罩值
    "NoticeSound": "Notice Sound Node by ms", # 提示音节点
    "AspectRatioAdjuster": "Aspect Ratio Adjuster by ms", # 宽高比调整
    "I2VConfigureNode": "I2V Configure Node by ms", # 图生视频配置节点
    "ResolutionPresetNode": "Resolution Preset Node by ms", # 分辨率预设节点
    "FramesSplitCalculator": "Frames Split Calculator by ms", # 帧分割计算器
    "FramesSegmentSlicer": "Frames Segment Slicer by ms", # 帧分段切片器
    "ImagesConcatWithOverlap": "Images Concat With Overlap by ms", # 图片拼接（带重叠）
    "ImageFlipNode": "Image Flip Node by ms", # 图像翻转
    "CreateColorImageAndMask": "Create Color Image And Mask by ms", # 创建彩色图片和遮罩
    "NormalizeMask": "Normalize Mask by ms", # 标准化遮罩
    "AnalyzeMask": "Analyze Mask by ms", # 分析遮罩
    "LoadImageBatchAdvanced": "Load Image Batch by ms", # 批量加载图片
    "LoadImageByIndex": "Load Image By Index by ms", # 按索引加载图片
    "ImageMaskedColorFill": "Image Masked Color Fill by ms", # 遮罩区域颜色填充
    "ImageBlackColorFill": "Image Black Color Fill by ms", # 黑色填充
    "ImageLayerMix": "Image Layer Mix by ms", # 图像图层混合
    "ImageDualMaskColorFill": "Image Dual Mask Color Fill by ms", # 双遮罩颜色填充
    "LoadLoraBatch": "Load Lora Batch by ms", # 批量加载 LoRA
    "WanVideoLoraBatch": "Wan Video Lora Batch by ms", # Wan 视频 LoRA 批量
    "ShowResultLast": "Show Result Last (VHS Filenames) by ms", # 显示最后结果（VHS 文件名）
    "ManualVideoInput": "Manual Video Input by ms", # 手动视频输入
    "LoadVideoFromFolder": "Load Video From Folder by ms", # 从文件夹加载视频
    "LoadLatentUpload": "Load Latent Upload by ms", # 加载上传的潜空间数据
    "ImageToSequenceWithMask": "Image To Sequence With Mask by ms", # 图像转序列带遮罩
    "TextInputBatch": "Text Input Batch by ms", # 文本批量输入
    "TextDictSplitter": "Text Dict Splitter by ms", # 文本字典分割
    "TextDictChecker": "Text Dict Checker by ms", # 文本字典检查
    "ImageExpand": "Image Expand by ms", # 图像扩展
    "ImageBlendAdvanceMy": "Image Blend Advance My by ms", # 增强版图像混合
    "ImageTakeLast": "Image Take Last by ms", # 截取最后几张图片
    "LoadLoraMerge": "Load Lora Merge by ms", # 加载并合并 LoRA
    "IndexSelector": "Index Selector by ms", # 索引选择器
    "MyBatchManager": "My Batch Manager by ms", # 批量管理器
    "ImageConcatMultiMs": "Image Concat Multi by ms", # 图像拼接多输入
    "ImageBatchAccumulator": "Image Batch Accumulator by ms", # 图像批量累加器
    "AnyBatchAccumulator": "Any Batch Accumulator by ms", # 任意类型批量累加器
    "AnyBatchListConverter": "Any Batch List Converter by ms", # 任意类型批量列表转换器
    "AnyStopOnNone": "Any Stop On None by ms", # 遇到空值停止
    "AnyValidityChecker": "Any Validity Checker by ms", # 任意类型数据校验器
    "GetNodeInputValue": "Get Node Input Value by ms", # 获取节点输入值
    "WanVideoDoubleStream": "Wan Video Double Stream by ms", # Wan 视频双流
    "WanVideoDoubleStreamAsset": "Wan Video Double Stream Asset by ms", # Wan 视频双流资产
    "SaveImageWithToggle": "Save Image With Toggle by ms", # 保存带开关的图片
    "AutoVAEEncode": "Auto VAE Encode (Switch) by ms", # 自动 VAE 编码（切换）
    "AutoLatentSource": "Auto Latent Source by ms", # 自动潜空间来源选择
    "FluxLatentMaskBinder": "Flux Latent Mask Binder by ms", # Flux 潜空间遮罩绑定
    "WorkflowForceRerunPassthrough": "Workflow Force Rerun Passthrough by ms", # 强制让工作流每次更新
    "GroupSwitchAny": "Group Switch Any by ms", # 任意类型分组切换
    "MultiInputStateMapper": "Multi Input State Mapper by ms", # 多输入存在状态映射
    "MultiImageConditionReference": "Multi Image Conditioning Reference by ms", # 多图条件参考
    "WorkflowGroupPresetManager": "Workflow Group Preset Manager by ms", # 工作流切换组管理器
    "BackgroundRemovalColorize": "Background Removal Colorize by ms", # 背景移除换色增强
}

WEB_DIRECTORY = "./web/js"
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
