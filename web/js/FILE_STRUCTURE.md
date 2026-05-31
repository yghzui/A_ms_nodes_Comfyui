# A_my_nodes `web/js` 文件结构说明

> 目标：
> 1. 快速定位某个文件负责什么。
> 2. 开发新节点时先查可复用模块，避免重复造轮子。
> 3. 区分“通用基础模块”和“某个节点专属实现”。

---

## 先看这里

开发新功能前，优先按这个顺序找：

1. `utils/`
   - 通用函数、通知、弹窗、DOM 创建、Canvas 绘制、Widget 基类。
2. `load_image/`
   - 如果需求和图片预览、布局、批量图片管理有关，先复用这里。
3. `lora/`
   - 如果需求和 LoRA 选择、LoRA Widget、LoRA 信息弹窗有关，先看这里。
4. `text_input_batch/`
   - 如果需求和“列表型输入 + 批量编辑 + 菜单动作”有关，先看这里的拆分方式。
5. `asset_manager/`
   - 如果需求涉及资产管理、分组、预览图、拖拽、多选、导入导出，先查这里。
6. 根目录节点脚本
   - 某个具体节点的业务逻辑通常都在这里落地。

---

## 目录总览

```text
web/js/
├── core/                  # 基础节点类、rgthree 相关 API
├── utils/                 # 最通用的复用层，优先看
├── asset_manager/         # 资产管理器整套 UI / 数据 / 交互
├── load_image/            # 批量图片节点的绘制、布局、图片管理
├── text_input_batch/      # 文本批量输入节点拆分模块
├── lora/                  # LoRA 选择器、LoRA 信息、LoRA Widget
├── image_editor/          # 图片编辑器
├── *.js                   # 各具体节点入口文件
└── FILE_STRUCTURE.md      # 本说明文档
```

---

## 一、最值得复用的文件

这几个文件基本属于“下次先查，不要自己再写一份”。

### `utils/shared_utils.js`

用途：通用工具总库。最常复用。

推荐优先使用：

| 导出项 | 作用 | 典型场景 |
|---|---|---|
| `showTopNotification(message, type)` | 顶部通知，支持 success/error/warning/info，已支持队列式堆叠显示 | 成功/失败/警告提示，替代 `alert()` |
| `$el(tag, propsOrChildren, children)` | 创建 DOM 元素，替代旧式 `$el` | 资产管理器、对话框、自定义小面板 |
| `wait(ms)` | 延时等待 | 动画后处理、异步 UI 刷新 |
| `debounce(fn, ms)` | 防抖 | 输入框联想、重绘控制、搜索 |
| `moveArrayItem(arr, from, to)` | 调整数组顺序 | Widget 重排、分组排序 |
| `removeArrayItem(arr, itemOrIndex)` | 删除数组项 | 删除 widget / 数据项 |
| `getObjectValue()` / `setObjectValue()` | 读写嵌套对象 | 配置树、复杂设置对象 |
| `convertToBase64()` / `convertToArrayBuffer()` | 多种来源互转 | 图片、Canvas、Blob、剪贴板处理 |
| `getCanvasImageData()` | 读取图片像素 | 图片编辑、比较、分析 |
| `areArrayBuffersEqual()` / `areDataViewsEqual()` | 二进制比较 | 缓存判重、图像数据比对 |
| `Broadcaster` / `broadcastOnChannel()` | 广播通信 | 跨窗口、跨面板同步 |

备注：

- 现在全项目的普通提示应优先走 `showTopNotification()`。
- 如果只是创建小型 DOM 结构，优先用 `$el()`，不要重复手写一堆 `createElement()`。

### `utils/common.js`

用途：轻量公共函数。

推荐优先使用：

| 导出项 | 作用 | 典型场景 |
|---|---|---|
| `chainCallback(object, property, callback)` | 安全追加原型方法，不覆盖原有逻辑 | `onNodeCreated`、`onConfigure`、`getExtraMenuOptions` 扩展 |
| `getAdjustedFontSize(ctx, text, maxWidth, minFontSize, maxFontSize)` | 自动压缩字体到合适宽度 | Canvas 上绘制文件名、标签、标题 |

备注：

- 写节点扩展时，优先 `chainCallback(...)`，不要直接粗暴覆盖已有生命周期。

### `utils/modal.js`

用途：通用模态框单例。

推荐优先使用：

| 导出项 | 作用 | 典型场景 |
|---|---|---|
| `modal.show({ title, content, buttons, width, onClose })` | 打开模态框 | 确认删除、导入内容、选择覆盖/追加 |
| `modal.close()` | 关闭模态框 | 按钮回调结束后关闭 |

什么时候用：

- 需要用户明确点击按钮确认的流程。
- 需要输入框、选择项、表单式交互。

什么时候不要用：

- 只是“提示成功/失败/警告”，这类统一用 `showTopNotification()`。

### `utils/lightbox_preview.js`

用途：全屏图片/视频预览。

推荐优先使用：

| 导出项 | 作用 | 典型场景 |
|---|---|---|
| `showLightbox(urls, currentIndex, type)` | 通用全屏预览入口 | 图片/视频预览 |
| `showImageLightbox(imagePaths, currentIndex)` | 图片预览快捷入口 | 节点图片全屏查看 |
| `showVideoLightbox(videoPaths, currentIndex)` | 视频预览快捷入口 | 视频节点预览 |

### `utils/utils_canvas.js`

用途：Canvas 绘图工具。

推荐优先使用：

| 导出项 | 作用 | 典型场景 |
|---|---|---|
| `fitString()` | 超长文本截断 | 节点标题、文件名 |
| `drawRoundedRectangle()` | 画圆角矩形 | 按钮、卡片、标签底板 |
| `drawNodeWidget()` | 绘制 widget 背景 | 自定义节点部件 |
| `drawNumberWidgetPart()` / `drawTogglePart()` | 画数字/开关部件 | 自定义 widget |
| `drawInfoIcon()` / `drawPlusIcon()` / `drawWidgetButton()` | 常用小图标/按钮 | 信息按钮、加号按钮 |
| `isLowQuality()` | 判断低质量模式 | 大量绘制时降级 |

### `utils/utils_widgets.js`

用途：自定义 Widget 的基础抽象层。

推荐优先使用：

| 导出项 | 作用 | 典型场景 |
|---|---|---|
| `RgthreeBaseWidget` | 自定义 widget 基类 | 所有复杂 widget |
| `RgthreeBetterButtonWidget` | 按钮 widget | 节点内按钮 |
| `RgthreeBetterTextWidget` | 文本 widget | 可编辑文本 |
| `RgthreeDividerWidget` | 分割线 widget | 分块 UI |
| `RgthreeLabelWidget` | 标签 widget | 分组标题 |
| `RgthreeInvisibleWidget` | 隐形占位 widget | 对齐、兼容 |
| `drawLabelAndValue()` | 画 label + value | 自绘 widget |

---

## 二、按模块定位

### `core/`

#### `core/base_node.js`

用途：

- 提供基础节点类。
- 如果后续要抽“通用节点父类”，先看这里。

主要导出：

- `RgthreeBaseNode`
- `RgthreeBaseVirtualNode`
- `RgthreeBaseServerNode`

#### `core/rgthree.js`

用途：

- rgthree 运行时对象、日志级别、全局状态。

主要导出：

- `LogLevel`
- `rgthree`

#### `core/rgthree_api.js`

用途：

- 封装 rgthree 风格 API 调用。

推荐复用：

| 导出项 | 作用 | 典型场景 |
|---|---|---|
| `rgthreeApi.fetchApi()` / `fetchJson()` | 请求 rgthree API | rgthree 子系统接口调用 |
| `rgthreeApi.getLoras()` | 获取 LoRA 列表 | LoRA 选择菜单 |
| `rgthreeApi.getLorasInfo()` | 取 LoRA 详细信息 | LoRA 信息面板 |
| `rgthreeApi.refreshLorasInfo()` | 刷新 LoRA 元数据 | 重新扫描 |

备注：

- 如果目标接口是你自己的 `/a_my_nodes/...`，优先还是用 ComfyUI 的 `api.fetchApi(...)`。

---

## 三、图片加载链路

### `load_image/`

这是 `load_image_batch.js` 的核心配套模块，图片类节点优先复用这里。

#### `load_image/image_manager.js`

用途：

- 图片路径与选中状态管理。
- 图片显示、清除、编辑器打开、菜单动作、覆盖/追加选择。

推荐复用：

| 导出项 | 作用 | 典型场景 |
|---|---|---|
| `getCustomButtons(node)` | 生成多图模式底部按钮配置 | 需要批量操作图片时 |
| `updateWidgetValue(node)` | 把勾选状态同步回 widget | 多选图片节点 |
| `showImages(node, paths)` | 根据路径加载并展示图片 | 图片预览主入口 |
| `updateImagePreviews(node, paths)` | 更新预览 | 配置变化后刷新 |
| `populate(imagePaths)` | 处理新图片列表并刷新 | 节点接收新图片 |
| `ensureCustomDrawMethod(node)` | 安装自定义绘制函数 | 需要接管节点前景绘制 |
| `clearImageWithConfirmation()` | 带确认的清理入口 | 删除当前图片 |
| `executeClear()` | 真正执行清理 | 已确认后调用 |
| `openMaskEditorForImage()` | 打开 MaskEditor | 对单张图做遮罩编辑 |
| `askAppendOrReplaceIfNeeded()` | 弹出“追加/替换”选择 | 导入/粘贴时复用 |

#### `load_image/layout.js`

用途：

- 图片区域和按钮布局计算。

主要导出：

- `computeButtonLayout(node, buttons)`
- `calculateImageLayout(node, imageCount)`

#### `load_image/draw.js`

用途：

- 纯绘制层，把计算结果画到节点上。

主要导出：

- `drawNodeImages(node, ctx)`

推荐做法：

- 布局逻辑不要写进绘制函数。
- 图片节点的新功能优先分到：
  - 状态管理：`image_manager.js`
  - 几何布局：`layout.js`
  - Canvas 绘制：`draw.js`

---

## 四、文本批量输入链路

### `text_input_batch/`

这是拆分最清晰的一组，适合作为“复杂节点拆模块”的参考模板。

主要导出：

| 文件 | 导出项 | 作用 |
|---|---|---|
| `text_input_batch_actions.js` | `createTextInputBatchActionsApi()` | 导入导出、批量删除、资产管理联动 |
| `text_input_batch_lifecycle.js` | `createTextInputBatchLifecycleApi()` | 节点生命周期行为 |
| `text_input_batch_menu.js` | `createTextInputBatchMenuApi()` | 菜单行为 |
| `text_input_batch_render_core.js` | `createTextInputBatchRenderCoreApi()` | 渲染与布局 |

适合参考的场景：

- 一个节点业务很复杂，需要拆成“渲染 / 生命周期 / 动作 / 菜单”。
- 想降低单文件行数、提高可维护性。

---

## 五、LoRA 复用层

### `lora/lora_widgets.js`

用途：

- LoRA 选择器、LoRA 右键菜单、LoRA 基础 Widget。

推荐复用：

| 导出项 | 作用 | 典型场景 |
|---|---|---|
| `showLoraChooser(event, callback, filter, opts, node, widget)` | 打开 LoRA 选择菜单 | 添加/替换 LoRA |
| `getLoraSlotMenuOptions(slot, event, node)` | 生成 LoRA 槽位菜单项 | LoRA 右键菜单 |
| `BaseLoraWidget` | LoRA widget 基类 | 新的 LoRA 节点 |

### `lora/lora_info_service.js`

用途：

- LoRA 信息读写、缓存、服务层。

主要导出：

- `LoraInfoService`
- `LORA_INFO_SERVICE`

### `lora/lora_info_dialog.js`

用途：

- LoRA 信息弹窗 UI。

主要导出：

- `LoraInfoDialog`

---

## 六、资产管理器

### `asset_manager/`

这是最完整的一套“前端子系统”。

建议定位方式：

| 文件 | 作用 |
|---|---|
| `asset_manager_ui.js` | 主 UI 入口，调度各子模块 |
| `asset_manager_data_handler.js` | 导入导出、数据转换、数据落盘前处理 |
| `asset_manager_drag_select.js` | 多选、复制粘贴、拖拽、删除 |
| `asset_manager_preview_handler.js` | 预览图路径与预览源处理 |
| `asset_manager_quick_menu.js` | 快捷菜单 |
| `asset_manager_tooltip.js` | tooltip 管理 |
| `asset_manager_window.js` | 窗口/面板相关控制 |
| `asset_manager_style.js` | 样式 |
| `am_dialog.js` | 资产管理器自己的对话框实现 |

推荐复用：

- 如果只是做普通提示，不要再直接调 `AMDialog.alert()`，优先用 `showTopNotification()`。
- 如果要做资产库分组、批量选择、预览图拖拽，直接参考 `asset_manager_drag_select.js` 和 `asset_manager_data_handler.js`。

---

## 七、各节点入口文件快速索引

下面这些文件大多是“单节点入口文件”，一般负责：

- `app.registerExtension(...)`
- 节点生命周期扩展
- 业务逻辑拼装
- 调用 `utils/` 和子模块

### 图片/媒体相关

| 文件 | 主要作用 |
|---|---|
| `load_image_batch.js` | 批量图片节点，拖拽/粘贴/复制/右键菜单/单图与多图切换 |
| `en_load_latent.js` | latent 文件上传与选择 |
| `en_load_video.js` | 视频上传与选择 |
| `image_concat_multi_dynamic.js` | 多图拼接类节点 |
| `show_result_last.js` | 结果展示类节点 |
| `analyze_mask.js` | 遮罩分析 |
| `mask_add_dynamic.js` / `mask_subtract_dynamic.js` | 遮罩相关动态节点 |
| `bg_removal_colorize.js` | 去背与换色 |

### LoRA / 模型相关

| 文件 | 主要作用 |
|---|---|
| `load_lora_batch.js` | 批量 LoRA 节点 |
| `load_lora_merge.js` | LoRA 合并、导入资产、保存资产 |
| `wan_video_lora_batch.js` | Wan 视频相关 LoRA 批量节点 |
| `wan_video_double_stream.js` | 双流模型/LoRA 节点 |
| `wan_video_double_stream_asset.js` | Wan 双流资产节点 |

### 文本 / 配置 / 工作流相关

| 文件 | 主要作用 |
|---|---|
| `text_input_batch.js` | 文本批量输入节点入口 |
| `text_dict_checker.js` | 文本字典校验 |
| `resolutionpreset.js` | 分辨率预设管理 |
| `workflow_group_preset_manager.js` | 工作流分组预设 |
| `group_switch_any.js` | 分组切换 |
| `load_ui_node_value.js` | 读取/回写 UI 节点值 |
| `index_selector.js` | 索引选择器 |
| `i2v_configure.js` | 图生视频配置 |
| `multi_input_state_mapper_dynamic.js` | 多输入状态映射 |
| `multi_image_condition_reference_dynamic.js` | 多图条件引用 |
| `any_batch_accumulator_dynamic.js` | 批量累积 |

---

## 八、常用开发决策表

在真正开始写代码前，如果你只知道“要做什么”，不知道“该去哪个文件”，可以先查下面这张表。

| 关键词 / 需求 | 优先查看文件 | 说明 |
|---|---|---|
| 通知提示 | `utils/shared_utils.js` | 统一用 `showTopNotification()`，不要再写 `alert()` |
| 确认弹窗 / 选择弹窗 | `utils/modal.js` | 统一模态框入口 |
| 创建 DOM | `utils/shared_utils.js` | 优先用 `$el()` |
| 节点生命周期扩展 | `utils/common.js` | 优先用 `chainCallback()` |
| 图片预览 | `utils/lightbox_preview.js`、`load_image/image_manager.js` | 全屏预览和节点内图片显示 |
| 图片节点绘制 | `load_image/draw.js`、`load_image/layout.js` | 绘制和布局拆分 |
| 图片清除 / 追加替换 / Mask 编辑 | `load_image/image_manager.js` | 图片节点核心管理逻辑 |
| 图片复制 / 粘贴 / 拖拽上传 | `load_image_batch.js` | 图片批量节点入口逻辑 |
| LoRA 选择 | `lora/lora_widgets.js` | 选择器和槽位菜单 |
| LoRA 信息查看 / 缓存 | `lora/lora_info_service.js`、`lora/lora_info_dialog.js` | 信息服务和弹窗 |
| 自定义 Widget | `utils/utils_widgets.js` | 不要从零写鼠标命中和绘制 |
| Canvas 按钮 / 图标 / 开关 | `utils/utils_canvas.js` | 已有现成绘图工具 |
| 文本批量导入导出 | `text_input_batch/text_input_batch_actions.js` | 文本节点动作集合 |
| 文本批量菜单 | `text_input_batch/text_input_batch_menu.js` | 菜单项生成 |
| 复杂节点拆模块参考 | `text_input_batch/` | 最适合参考的拆分模板 |
| 资产管理器主入口 | `asset_manager/asset_manager_ui.js` | 先从这里看整体调度 |
| 资产导入导出 / 数据处理 | `asset_manager/asset_manager_data_handler.js` | JSON、剪贴板、落盘前处理 |
| 资产拖拽 / 多选 / 粘贴 | `asset_manager/asset_manager_drag_select.js` | 多选与拖拽交互 |
| 资产预览图处理 | `asset_manager/asset_manager_preview_handler.js` | 预览图地址和预览逻辑 |
| 资产快捷菜单 | `asset_manager/asset_manager_quick_menu.js` | 右键或快捷菜单 |
| LoRA 合并 + 资产联动 | `load_lora_merge.js` | 具体业务节点实现 |
| Wan 双流资产联动 | `wan_video_double_stream.js`、`wan_video_double_stream_asset.js` | 模型/资产双流相关 |
| 工作流预设 / 分组配置 | `workflow_group_preset_manager.js`、`group_switch_any.js` | 工作流分组和预设 |

---

## 九、常用开发决策表

### 1. 只是要提示用户

- 成功/失败/警告：
  - 用 `showTopNotification()`
- 需要用户点按钮确认：
  - 用 `modal.show()`

### 2. 要扩展节点生命周期

- 优先用 `chainCallback(nodeType.prototype, "onNodeCreated", fn)`
- 不要直接无脑覆盖原型方法

### 3. 要创建 DOM

- 优先用 `$el()`
- 不要重复堆 `document.createElement()`

### 4. 要做自定义 Widget

- 先看 `utils/utils_widgets.js`
- 再看已有节点里是不是已经有同类 widget 可复用

### 5. 要做图片节点

- 先看 `load_image/`
- 尤其是：
  - `showImages()`
  - `calculateImageLayout()`
  - `drawNodeImages()`
  - `askAppendOrReplaceIfNeeded()`

### 6. 要做 LoRA 选择

- 先看 `lora/lora_widgets.js`
- 优先复用 `showLoraChooser()`

### 7. 要做导入导出弹窗

- 先看：
  - `text_input_batch/text_input_batch_actions.js`
  - `asset_manager/asset_manager_data_handler.js`
  - `wan_video_double_stream.js`
  - `load_lora_merge.js`

---

## 十、避免重复造轮子的建议

下次开发前，先回答这几个问题：

1. 这是不是“纯提示”？
   - 是：直接 `showTopNotification()`
2. 这是不是“确认/选择/输入”？
   - 是：直接 `modal.show()`
3. 这是不是“节点原型扩展”？
   - 是：先看 `chainCallback()`
4. 这是不是“自定义 widget”？
   - 是：先看 `utils_widgets.js`
5. 这是不是“图片预览 / 布局 / 清理 / 追加替换”？
   - 是：先看 `load_image/`
6. 这是不是“LoRA 菜单 / LoRA Widget / LoRA 信息”？
   - 是：先看 `lora/`
7. 这是不是“多项列表批量编辑”？
   - 是：先看 `text_input_batch/` 和 `asset_manager/`

---

## 十一、维护建议

- 新增复用函数时，优先放到 `utils/` 或对应子模块目录，不要先堆进某个节点入口文件。
- 某个节点文件如果越来越大，优先按这几类拆：
  - `actions`
  - `menu`
  - `render`
  - `layout`
  - `manager`
- 新增了通用函数、通用组件、通用交互模式后，记得同步更新本文档。

---

## 十二、建议优先查看的文件

如果只想最快熟悉这个目录，先按这个顺序读：

1. `utils/shared_utils.js`
2. `utils/common.js`
3. `utils/modal.js`
4. `utils/utils_widgets.js`
5. `load_image/image_manager.js`
6. `lora/lora_widgets.js`
7. `text_input_batch/text_input_batch_actions.js`
8. `asset_manager/asset_manager_ui.js`
9. `load_image_batch.js`
10. `load_lora_merge.js`

这样基本就能掌握这个目录里 80% 的复用方式。
