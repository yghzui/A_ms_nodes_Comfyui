# ComfyUI `LoadImageBatchAdvanced` 图片编辑工作流说明

本文档面向两类读者：

- 需要从外部 Web 项目接入该节点的人。
- 需要继续维护前后端实现的人。

文档目标是先讲清“怎么接”，再解释“内部为什么这样工作”。核心内容包括两条编辑链路、`clipspace` 文件生成规则、`[input]` 标记的作用，以及后端如何从最终 PNG 拆分 `IMAGE` 与 `MASK`。

核心结论如下：

- 这个节点同时支持两条编辑链路：`官方 MaskEditor 复用链路` 与 `自定义全屏编辑器 / 外部 API 链路`。
- 两条链路的入口不同，但最终都会落到 `input/clipspace` 下的一组带时间戳的派生文件。
- 工作流运行时，后端读取的核心文件是 `clipspace-painted-masked-<ts>.png`，并从其 `Alpha` 通道拆出 `MASK`。
- 原始图片不会被覆盖；节点只是把内部引用路径替换到新的 `clipspace` 文件，实现非破坏性编辑。

## 0. 外部接入速览

如果你只是想从外部 Web 项目接入这套机制，可以先记住下面 5 步：

1. 准备一张已经位于 ComfyUI `input` 目录中的原图路径。
2. 前端分别导出 `mask_data` 与 `paint_data` 两张透明 PNG 的 Base64。
3. 调用 `POST /a_my_nodes/upload_custom_edited_image`。
4. 取回返回值里的 `filepath`，并在末尾补上 ` [input]`。
5. 把这个新路径写回 `LoadImageBatchAdvanced` 节点，再提交工作流到 `/prompt`。

如果你只关心外部调用，优先阅读 `第 3 节`、`第 5 节`、`第 8 节` 即可。

## 1. 核心概念

### 1.1 `[input]` 标记

`[input]` 不是文件名本体的一部分，而是一个路径解析标记，表示这张图应当从 ComfyUI 的 `input` 目录中查找。

它的作用不是单纯标识“这是原图”，而是明确告诉前后端：

- 这张图片属于 `input` 资源，而不是 `output` 或 `temp`。
- 后续构造预览 URL、提交工作流、后端加载文件时，都应以 `input_dir` 为根进行解析。

### 1.2 `clipspace` 目录

`clipspace` 位于 ComfyUI 的 `input` 目录下，用于保存编辑后的派生文件。每次保存都会生成一组新的时间戳文件，避免浏览器缓存命中旧图，也相当于保留了一份轻量级历史快照。

### 1.3 `paint` 与 `mask`

- `paint`：彩色绘制层，保存用户的彩色笔触。
- `mask`：遮罩层，本质上对应最终图像的透明度控制信息。

这两层在前端编辑时是分开的，在后端合成时也分别参与输出文件生成。

## 2. 链路 A：官方 MaskEditor 复用流程

这条链路对应节点右键菜单中的“编辑图片 (官方MaskEditor)”。

### 2.1 前端如何唤起官方编辑器

当用户在某张图片上右键选择官方编辑器时，前端会执行以下步骤：

1. 检查路径是否带有 ` [input]` 后缀。
2. 若存在该后缀，先临时剥离，得到可直接解析的相对路径。
3. 通过一个 `Mock Node` 把当前图片伪装成官方 `MaskEditor` 可处理的节点输入。
4. 调用 `Comfy.MaskEditor.OpenMaskEditor` 命令，唤起 ComfyUI 官方编辑器。

这一过程的重点不是直接向自定义后端接口发请求，而是复用 ComfyUI 官方前端扩展能力。

### 2.2 保存后会发生什么

用户在官方 `MaskEditor` 中点击 `Save to node` 后，ComfyUI 会生成新的 `clipspace` 文件，并返回新的文件路径。

前端接到保存结果后会：

1. 取出新的 `clipspace/clipspace-painted-masked-<ts>.png`。
2. 如果原图属于 `input` 资源，则把 ` [input]` 后缀补回去。
3. 用新路径替换节点中原本的图片路径。
4. 刷新节点缩略图与预览。

这一步真正变化的不是原文件本身，而是节点内部指向的路径。

## 3. 链路 B：自定义全屏编辑器与外部 API 流程

这条链路对应节点中的“编辑图片 (全屏)”，以及外部 Web 项目直接调用 `/a_my_nodes/upload_custom_edited_image` 的接入方式。

需要注意的是，这条链路和官方 `MaskEditor` 不是同一套前端实现，但最终生成的落盘结果是兼容的。

### 3.1 前端需要准备什么

无论你是在节点内使用自定义全屏编辑器，还是在外部 Web 项目中自己做 UI，前端都不需要手动生成那 4 个 `clipspace` 文件。

前端只需要提供两份透明 PNG 数据：

- `mask_data`：遮罩层图像。
  涂抹区域的 `Alpha` 应为 `255`，未涂抹区域应为 `0`。
- `paint_data`：彩色笔触图像。
  背景应保持完全透明，只保留彩色笔触本身。

推荐的前端图层结构如下：

1. 底图层：原始图片。
2. 遮罩层：用户绘制的遮罩。
3. 绘画层：用户绘制的彩色笔触。

### 3.2 自定义上传接口

后端提供如下接口：

`POST /a_my_nodes/upload_custom_edited_image`

请求体示例：

```json
{
  "image_path": "your_original_image_name.jpg [input]",
  "mask_data": "data:image/png;base64,iVBORw0KGgo...",
  "paint_data": "data:image/png;base64,iVBORw0KGgo..."
}
```

参数说明：

- `image_path`：建议携带 ` [input]` 后缀。后端会先剥离该标记，再到 `input_dir` 下拼接实际物理路径。
- `mask_data`：可为空；为空时表示没有新增遮罩。
- `paint_data`：可为空；为空时表示没有新增彩色笔触。

额外注意：

- `image_path` 必须能映射到 ComfyUI 的 `input` 目录内部。
- 后端会做路径越界校验；不能传一个跳出 `input_dir` 的任意磁盘路径。

### 3.3 返回结果与工作流替换

上传成功后，接口会返回：

```json
{
  "success": true,
  "filepath": "clipspace/clipspace-painted-masked-1776257523204.png"
}
```

拿到这个结果后，外部调用方还需要执行一步关键处理：

1. 将返回的 `filepath` 补成 `clipspace/clipspace-painted-masked-1776257523204.png [input]`。
2. 把它写回 `LoadImageBatchAdvanced` 节点的图片路径参数。
3. 再将更新后的工作流提交给 `/prompt`。

如果省略 ` [input]`，后续路径解析就可能与节点约定不一致。

## 4. 四类 `clipspace` 文件的准确含义

每次保存后，后端都会在 `input/clipspace` 下生成一组新的时间戳文件：

### 4.1 `clipspace-mask-<ts>.png`

- 实际内容不是“纯黑白遮罩位图”。
- 它保存的是：`原图 RGB + 反转后的 Alpha`。
- 主要用途是帮助前端在重新打开编辑器时恢复遮罩状态。

也就是说，前端在恢复遮罩时，主要消费的是它的 `Alpha` 信息，而不是直接把它当作一张独立的纯 mask 图片来展示。

### 4.2 `clipspace-paint-<ts>.png`

- 保存彩色笔触本身。
- 背景为完全透明。
- 用于恢复绘画层内容。

### 4.3 `clipspace-painted-<ts>.png`

- 保存原图与 `paint` 层叠加后的结果。
- 主要作用是作为可视化合成中间态。

### 4.4 `clipspace-painted-masked-<ts>.png`

- 这是最终交付文件。
- 它包含：`原图/paint 叠加结果 + Alpha 透明信息`。
- 节点前端预览与后端运行时都以它为核心输入。

## 5. 后端加载与 `IMAGE` / `MASK` 拆分逻辑

当工作流执行到 `load_image_batch.py` 时，处理流程如下：

1. 读取传入路径。
2. 如果路径末尾存在 ` [input]`，先剥离该后缀。
3. 以 `input_dir` 为根拼接实际文件路径。
4. 读取目标图片。
5. 从 `Alpha` 通道生成 `MASK`。
6. 从图像的 `RGB` 通道生成 `IMAGE`。

### 5.1 `MASK` 如何生成

后端会取目标图的 `Alpha` 通道，并执行反转：

- 透明区域 `Alpha = 0`
- 反转后变成 `Mask = 1`

因此，编辑时被“挖空”或标记为遮罩的区域，会在 `MASK` 输出中成为有效遮罩区域。

### 5.2 `IMAGE` 实际包含什么

这里最容易误解。

`IMAGE` 并不是“自动还原出来的纯原图”，而是目标 PNG 的 `RGB` 内容：

- 如果最终图里包含彩色 `paint` 笔触，那么这些笔触也会进入 `IMAGE`。
- `convert("RGB")` 的作用只是丢弃 `Alpha` 通道，不会自动回退到未经编辑的原始像素。

因此，`IMAGE` 是否看起来像“纯原图”，取决于用户有没有在 `paint` 层绘制内容。

### 5.3 `apply_alpha_to_image` 的作用

如果启用了 `apply_alpha_to_image`，后端会将恢复出的 `alpha` 再乘回 `RGB` 图像：

- `MASK` 的计算方式不变。
- `IMAGE` 中被遮罩的区域会真正被透明度影响，从而在图像数据层面被抹除或压暗。

如果未启用该选项，`IMAGE` 仍然只是“丢弃 Alpha 后的 RGB 图像”。

## 6. 为什么这是非破坏性编辑

这个设计的关键点在于：

- 原始图片不会被覆盖。
- 每次保存都会得到新的时间戳文件。
- 节点只是把引用从旧路径切换到新的 `clipspace` 路径。

因此：

- 原图始终保留在原位置。
- 不同保存版本天然具备隔离性。
- 浏览器不会因为文件名不变而继续使用旧缓存。

## 7. 两条链路的关系总结

可以把整个机制理解成两层：

- 上层是两种不同的编辑入口：
  `官方 MaskEditor` 或 `自定义全屏编辑器 / 外部 API`。
- 下层是统一的落盘与加载逻辑：
  `clipspace` 时间戳文件 + 运行时从最终 PNG 拆 `MASK` 与 `IMAGE`。

两条链路的区别主要在“前端如何编辑、如何提交保存”；它们的共同点在于“最终都把节点路径替换到新的 `clipspace-painted-masked-<ts>.png [input]`”。

## 8. 外部 Web 接入的最小实现清单

如果你要在外部 Web 项目中适配这个节点，可以按以下最小步骤实现：

1. 准备原图路径，并确保它能映射到 ComfyUI 的 `input` 目录。
2. 用前端画布分别导出 `mask_data` 与 `paint_data`。
3. 调用 `/a_my_nodes/upload_custom_edited_image`。
4. 获取返回的 `filepath`，并补上 ` [input]` 后缀。
5. 将新路径写回 `LoadImageBatchAdvanced` 节点，再提交到 `/prompt`。

### 8.1 推荐的请求顺序

推荐将外部接入过程理解成两次提交：

1. 第一次提交编辑数据到 `/a_my_nodes/upload_custom_edited_image`，生成新的 `clipspace` 文件。
2. 第二次提交更新后的工作流到 `/prompt`，让节点读取新的最终图。

这样做的好处是职责清晰：

- 编辑数据生成由专用接口负责。
- 工作流执行仍然保持 ComfyUI 原生的 `/prompt` 调度方式。

### 8.2 最容易漏掉的两个点

- `image_path` 必须指向 `input` 目录内可访问的图片，不能是任意绝对路径。
- 返回的 `filepath` 只是相对路径；外部调用时要主动补上 ` [input]`，再写回节点。

## 9. 常见误区

### 9.1 误区：`clipspace-mask-<ts>.png` 就是一张纯黑白遮罩图

不准确。它更接近“用于恢复编辑状态的中间文件”，实际包含原图 RGB 与反转后的 Alpha。

### 9.2 误区：`IMAGE` 输出一定是未编辑的原图

不准确。`IMAGE` 读取的是最终图像文件的 RGB 内容，因此可能包含彩色 `paint` 笔触。

### 9.3 误区：整个节点完全不走 ComfyUI 常规上传接口

不准确。初始导入图片时，前端仍然可以通过 ComfyUI 常规的 `/upload/image` 上传；自定义接口主要负责“编辑后重新生成 clipspace 文件”。

### 9.4 误区：拿到 `filepath` 后可以直接提交工作流，不需要补 ` [input]`

不建议这样做。这个节点的路径约定依赖 ` [input]` 标记，外部调用时应主动补回，保证前后端逻辑一致。

## 10. 源码对应关系附录

下面给出文档各结论对应的源码职责，便于后续排查和维护。

### 10.1 前端入口与节点交互

- `web/js/load_image_batch.js`
  负责把 `LoadImageBatchAdvanced` 节点接入 ComfyUI 前端，包括：
  - 图片选择、拖拽、粘贴上传。
  - 右键菜单中的“编辑图片 (官方MaskEditor)”与“编辑图片 (全屏)”入口。
  - 保存后把新路径回写到节点的 `image_paths`。

### 10.2 官方 MaskEditor 复用封装

- `web/js/load_image/image_manager.js`
  负责封装 `openMaskEditorForImage()`：
  - 构造 `Mock Node`。
  - 查找 `Comfy.MaskEditor` 扩展与 `Comfy.MaskEditor.OpenMaskEditor` 命令。
  - 调起官方编辑器，并接收保存回调。

### 10.3 自定义全屏编辑器

- `web/js/image_editor/image_editor.js`
  负责自定义全屏编辑器的主要逻辑，包括：
  - 画布 UI、遮罩层与绘画层管理。
  - 从已有 `clipspace` 文件恢复 `mask` 与 `paint` 图层。
  - 导出 `mask_data` 与 `paint_data`。
  - 调用 `/a_my_nodes/upload_custom_edited_image` 并在保存后补回 ` [input]`。

### 10.4 自定义后端上传接口

- `routes.py`
  负责注册并实现 `/a_my_nodes/upload_custom_edited_image`：
  - 接收 `image_path`、`mask_data`、`paint_data`。
  - 校验路径是否位于 `input_dir` 内。
  - 生成 `clipspace-mask-*`、`clipspace-paint-*`、`clipspace-painted-*`、`clipspace-painted-masked-*`。
  - 返回最终的 `filepath`。

### 10.5 工作流运行时加载逻辑

- `nodes/load_image_batch.py`
  负责节点运行时的文件读取与张量输出：
  - 剥离 ` [input]`。
  - 拼接 `input_dir` 定位真实文件。
  - 从 `Alpha` 通道反转生成 `MASK`。
  - 从最终图像的 `RGB` 内容生成 `IMAGE`。
  - 在需要时执行 `apply_alpha_to_image`。

### 10.6 问题排查建议

如果后续出现问题，可按职责逆推：

1. `保存后路径没更新`：先查 `load_image_batch.js` 与 `image_editor.js`。
2. `官方 MaskEditor 打不开`：先查 `image_manager.js` 中的扩展命令查找逻辑。
3. `接口保存成功但文件异常`：先查 `routes.py` 的合成与落盘逻辑。
4. `工作流运行后 MASK/IMAGE 不符合预期`：先查 `nodes/load_image_batch.py` 的 Alpha 与 RGB 拆分逻辑。
