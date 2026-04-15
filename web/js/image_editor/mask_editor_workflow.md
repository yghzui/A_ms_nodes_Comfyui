# ComfyUI MaskEditor 批量图片加载工作流分析

本文档记录了 `LoadImageBatchAdvanced` 节点如何通过复用 ComfyUI 官方内置的 `MaskEditor`（遮罩编辑器），实现在批量图片节点中对单张图片进行独立遮罩绘制和预览的完整生命周期。

## 1. 前端唤起与 `[input]` 标识符的意义
当在一张图片（例如 `comment_4a7e8e53...edit.jpg`）上右键选择“编辑图片”时：
- **剥离标识符**：前端 JS 代码会先检查并切掉 ` [input]` 这个后缀。
- **为何需要 `[input]`？**：在 ComfyUI 的文件系统里，图片通常分为 `input`、`output`、`temp` 三类。`[input]` 是一种显式的状态标记，告诉整个系统：“这是一张需要从 `input` 文件夹去寻找的用户输入图”，而不是一张由模型生成的输出图。
- **发送请求**：去除标识符后，拿到干净的路径，通过构造一个“伪造的节点”（Mock Node），触发官方的 `Comfy.MaskEditor.OpenMaskEditor` 命令，从而将官方编辑器“骗”出来为该图片服务。

## 2. 编辑并保存：为何会产生 4 个时间戳文件？
当你在 `MaskEditor` 中涂抹并点击“Save to node”时，ComfyUI 前端会向后端发送一个 `/upload/mask` 请求。为了实现“非破坏性编辑”和“历史记录追踪”，后端会在 `input/clipspace` 文件夹下生成带有**唯一时间戳**（如 `1776257523204`）的 4 个衍生文件：

1. **`clipspace-mask-<ts>.png`**：
   - **内容**：纯粹的黑白遮罩图（涂抹区域为白色，其余黑色）。
   - **作用**：记录纯净的 Mask 区域形状，主要用于 UI 重新打开编辑器时的形状还原。
2. **`clipspace-paint-<ts>.png`**：
   - **内容**：彩色笔触本身（背景全透明，只保留涂抹的色彩）。
   - **作用**：如果使用了带有颜色的画笔（而不是单纯的透明橡皮擦），这层会保存颜色信息。
3. **`clipspace-painted-<ts>.png`**：
   - **内容**：原图与上述 `paint` 彩色笔触叠加后的合成图像。
   - **作用**：用于展示带有彩色涂鸦的底图。
4. **`clipspace-painted-masked-<ts>.png`** (🌟 核心交付物)：
   - **内容**：**原图 + 彩色笔触 + Alpha透明通道** 的终极合成体。
   - **作用**：这就是前端 UI 用于渲染预览，以及后端 Python 用于提取数据的最终目标文件。

**时间戳机制的意义**：每次保存，即使是清除遮罩，也会生成一组全新的时间戳文件。这避免了浏览器缓存导致图片不更新的问题（如果文件名不变，浏览器可能直接读旧图），也相当于一种简单的“版本控制”。如果没有保存操作，就不会产生新文件。

## 3. 前端状态更替
保存完成后，MaskEditor 返回核心文件路径 `clipspace/clipspace-painted-masked-<ts>.png`：
- 代码拦截到这个新路径后，如果原图带有 `[input]`，会再把 ` [input]` 给**缝补回去**，变成：`clipspace/clipspace-painted-masked-<ts>.png [input]`。
- 接着，替换掉隐藏的 `image_paths` 输入框里原本的旧路径，并刷新 UI 显示这张带有透明通道的 PNG。前端此时发生了真正的“移花接木”。

## 4. 后端加载与“解剖” (Python 端)
当点击“Queue Prompt”运行工作流时，任务交接给了后端 `load_image_batch.py`：
- **脱下马甲**：后端第一件事就是把传进来的 ` [input]` 标识符砍掉，还原出真实的相对路径 `clipspace/clipspace-painted-masked-<ts>.png`。
- **寻找实体**：拼接上 `input_dir`（ComfyUI 的 input 目录），验证这个带时间戳的文件是否在硬盘上真的存在。
- **解剖提取**：
  - **分离遮罩**：把它的 Alpha 通道拿出来，反转后作为 `MASK` 张量输出（因为涂抹区域透明 Alpha=0，变成 Mask=1）。
  - **分离原图**：调用 `.convert("RGB")`，强制丢弃 Alpha 通道，完美“揭开”透明层，暴露出被透明层盖住的原图色彩，作为 `IMAGE` 张量输出。
- **可选的透明度应用**：如果开启了 `apply_alpha_to_image`，代码会手动将刚才分离出的 Alpha 通道乘以 RGB 像素，从而真正在图像数据上抹除涂抹区域。

## 5. 完整流程架构图总结

```text
[用户行为] 
  1. 上传/选择图片 comment_...edit.jpg
  2. 右键点击 -> 编辑图片 (MaskEditor)
       ↓
[前端劫持] 
  3. 剥离 [input] 后缀，将真实路径喂给伪造的节点。
  4. 唤起官方 MaskEditor 界面。
       ↓
[MaskEditor & 临时文件] 
  5. 用户涂抹并点击 Save。
  6. 在 input/clipspace/ 下生成 4 个带新时间戳的文件：
     - mask / paint / painted / painted-masked
       ↓
[前端状态更新]
  7. 拿到 clipspace/clipspace-painted-masked-<ts>.png。
  8. 补回 [input] 后缀。
  9. 将节点内部的图片路径从 comment... 替换为 clipspace...
 10. 刷新节点预览图（由于自带Alpha，直接透出遮罩效果）。
       ↓
[后端执行]
 11. 点击运行，后端收到 clipspace... [input]。
 12. 再次剥离 [input]，找到硬盘上的 clipspace-painted-masked-<ts>.png。
 13. 读取图片 -> Alpha通道转 MASK -> 丢弃Alpha保留RGB转 IMAGE。
```

这种设计极为巧妙：**它不动你的原始图片分毫**（`comment_...edit.jpg` 永远躺在原文件夹里），而是通过不断产生时间戳“快照文件”，并在前端动态替换引用路径，完美实现了非破坏性的编辑和数据分离。