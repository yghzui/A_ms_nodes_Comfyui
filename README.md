### A_ms_nodes_Comfyui 自定义节点集合

> 面向 ComfyUI 的一组实用自定义节点，涵盖图像/遮罩处理、人脸与人物相关处理、批量与视频流程辅助、LoRA 批处理、以及一些工作流实用工具。部分节点的思路或实现细节借鉴了社区优秀开源项目（见文末致谢）。

---

### 安装与使用

1. 将本文件夹放入 ComfyUI 的 `custom_nodes/` 目录下（例如 `ComfyUI/custom_nodes/A_my_nodes`）。
2. 安装依赖：
   - 标准环境（已创建 venv）：
     ```bat
     venv\Scripts\activate.bat
     cd custom_nodes\A_my_nodes
     pip install -r requirements.txt
     ```
   - Windows 便携版（portable）：
     ```bat
     python_embeded\python.exe -m pip install -r ComfyUI\custom_nodes\A_my_nodes\requirements.txt
     ```
3. 重启 ComfyUI，在节点搜索中通过中文名称或英文类名检索相应节点。

> 注意：部分节点依赖第三方推理/检测模型或外部库，请根据报错提示补齐依赖与模型文件。

---

### 节点清单（依据 `__init__.py` 已注册映射整理）

#### 图像与基础处理
- `LoadAndResizeImageMy`（Load & Resize Image by My）：加载并按指定尺寸/比例缩放图像。
- `ResizeImagesAndMasks`（Resize Images and Masks by My）：同时缩放图像与对应遮罩，保持对齐。
- `ImageFlipNode`（图像翻转节点 by My）：水平/垂直翻转图像。
- `CreateColorImageAndMask`（创建颜色图像和遮罩节点 by My）：生成纯色图像及对应遮罩。
- `ImageMaskedColorFill`（图像颜色填充 by My）：根据遮罩对图像区域进行颜色填充。
- `ImageBlackColorFill`（图像黑色填充 by My）：对指定区域进行黑色填充以抠底或遮挡。
- `ImageLayerMix`（图层混合 by My）：对多层图像进行混合叠加。
- `ImageDualMaskColorFill`（双遮罩不重叠区域颜色填充 by My）：对两遮罩的非重叠区进行差异化填充。
- `GenerateBlackTensor`（生成纯黑张量 by My）：生成全黑图像张量用于占位或管线占位。

#### 遮罩处理
- `CreateFaceBboxMask`（Create Face Bbox Mask by My）：根据人脸框生成遮罩。
- `CreateTextMask`（Text Mask path by My）：基于文字或路径生成遮罩。
- `MaskAdd`（遮罩相加 by My）：对两个遮罩做并集。
- `MaskSubtract`（遮罩相减 by My）：对两个遮罩做差集。
- `MaskOverlap`（重叠度 by My）：计算两个遮罩的重叠比例/区域。
- `AdjustMaskValues`（调整遮罩值 by My）：阈值、强度等数值级别调节。
- `NormalizeMask`（归一化遮罩节点 by My）：对遮罩进行归一化处理。
- `AnalyzeMask`（分析遮罩节点 by My）：统计或可视化遮罩特征用于调参。
- `PasteMasksMy`（粘贴面部遮罩 by My）：将遮罩以坐标/缩放方式粘贴到目标图像。

#### 人脸 / 人物相关
- `ResizeImageByPerson`（Resize Image by Person by My）：基于人物占比/检测结果进行自适应缩放。
- `CropInfoToNumbers`（Crop Info to Numbers by My）：将裁剪信息解析为数值输出（供后续节点复用）。
- `CropFaceMy`（Crop Face by My）：快速人脸裁剪。
- `CropFaceMyDetailed`（Crop Face Detailed by My）：更细粒度的人脸裁剪选项。
- `PasteFacesMy`（粘贴面部 by My）：将人脸区域以指定方式粘贴到目标图像。
- `PasteFacesAdvanced`（粘贴面部（高级） by My）：提供更细节的粘贴/对齐控制。
- `RemoveGlassesFaceMask`（去除眼镜 by My）：在人脸遮罩层面做眼镜区域的去除/平滑处理。
- `APersonFaceLandmarkMaskGeneratorByMy`（生成面部遮罩 by My）：基于人脸关键点/分割生成面部相关遮罩。
- `GroundingDinoGetBbox`（GroundingDinoGetBbox by My）：使用 GroundingDINO 获取检测框（需相应权重和依赖）。

#### 工作流数学/工具
- `CoordinateTessPosNeg`（CoordinateTessPosNeg by My）：坐标变换/正负样本辅助工具。
- `AspectRatioAdjuster`（宽高比调整节点 by My）：根据目标比例对分辨率进行对齐与适配。
- `FilterClothingWords`（过滤服装关键词 by My）：对服装相关词汇进行筛选/清洗以辅助下游流程。
- `NoticeSound`（铃声提醒节点 by My）：在流程完成或关键节点时播放提示音，便于批量流程监控。

#### 批量 / 资源加载
- `LoadImageBatchAdvanced`（批量加载 by My）：按规则批量加载图像集。
- `MyLoadImageListPlus`（加载图片列表 by My）：读取列表并输出图片序列。
- `LoadLoraBatch`（批量加载LoRA by My）：根据目录/规则批量收集 LoRA。

#### 视频 / I2V 流程辅助
- `I2VConfigureNode`（I2V 配置节点 by My）：为图生视频/视频流程提供统一配置输出。
- `FramesSplitCalculator`（按帧数切分（含重叠） by My）：将长序列按窗口与重叠量切分，返回索引区间。
- `FramesSegmentSlicer`（按索引截取图像与遮罩段 by My）：基于索引切片输出对应图像/遮罩子序列。
- `ImagesConcatWithOverlap`（按重叠覆盖拼接图像 by My）：将分段结果按重叠策略回拼。
- `WanVideoLoraBatch`（批量收集 WanVideo LoRA by My）：为 WanVideo 工作流收集/组织相关 LoRA。
- `ManualVideoInput`（手动输入视频文件名 by My）：将手动输入的路径/文件名接入到流程。
- `ShowResultLast`（显示视频结果（路径） by My）：读取 VHS 结果路径，便于最终结果查看。

---

### 路由与前端
- `routes.py`：注册自定义 HTTP 路由，延迟注册以兼容 ComfyUI 启动顺序。
- `web/js`：前端静态资源目录（如有网页交互或前端面板，会从此处提供静态文件）。

---

### 学习参考与致谢
本节点集的部分设计/实现参考或借鉴了以下优秀开源项目（排名不分先后）：
- a-person-mask-generator（人物/面部分割与关键点遮罩思路）
  - 链接：[`https://github.com/djbielejeski/a-person-mask-generator`](https://github.com/djbielejeski/a-person-mask-generator)
- ComfyUI-WanVideoWrapper（WanVideo 流程封装与长序列窗口化思路）
  - 链接：[`https://github.com/kijai/ComfyUI-WanVideoWrapper`](https://github.com/kijai/ComfyUI-WanVideoWrapper)
- rgthree-comfy（实用 UI/工作流增强思路）
  - 链接：[`https://github.com/rgthree/rgthree-comfy`](https://github.com/rgthree/rgthree-comfy)

上述项目均遵循其各自的开源许可，请在遵守相关许可的前提下使用与分发。本节点集仅作为学习与工作流效率提升之用。

---

### 反馈
如在使用中遇到问题或有功能建议，欢迎在本目录提交 Issue/PR（若仓库未开启 Issue，可通过注释或私有渠道反馈）。
