# ComfyUI 前端 Subgraph 与 GroupNode 机制分析

本文档深入分析 ComfyUI 新前端（ComfyUI_frontend）中节点的构造原理，特别是 GroupNode 和 Subgraph 的实现机制，以及它们对节点开发（特别是 Widget 交互）的影响。同时涵盖 Nodes 2.0 (Schema V2) 的变化。

## 1. 核心概念区分

在 ComfyUI 中，存在两种容易混淆的“组合”机制：**Group Node** 和 **Subgraph Node**。它们虽然目的相似（封装复杂性），但实现原理截然不同。

### 1.1 Group Node (`extensions/core/groupNode.ts`)
*   **本质**：通过序列化选中的节点，动态注册一个新的自定义节点类型（`ComfyNodeDef`）。
*   **表现**：在画布上是一个单一的“黑盒”节点，拥有预定义的输入和输出插槽。
*   **内部结构**：用户无法直接双击进入编辑。其内部逻辑由后端加载自定义节点定义来执行，或者前端通过特殊的逻辑（`groupNode.ts`）在后台处理。
*   **坐标系**：普通的 `LGraphNode`，位于全局坐标系中。

### 1.2 Subgraph Node (`lib/litegraph/src/subgraph/SubgraphNode.ts`)
*   **本质**：LiteGraph 原生的子图容器，是一个特殊的 `LGraphNode`。
*   **表现**：一个包含标题栏按钮（Enter Subgraph）的节点。
*   **内部结构**：内部持有一个完整的 `Subgraph` 对象（继承自 `LGraph`）。可以双击进入子图视图进行编辑。
*   **坐标系**：
    *   **外部**：`SubgraphNode` 本身在父图中拥有全局坐标。
    *   **内部**：子图内部的节点拥有相对于子图原点的**局部坐标**。
*   **Widget 代理**：使用 `ProxyWidget` 机制将子图内部节点的 Widget 暴露到外部节点上。

## 2. 转化为 Subgraph 的原理

当执行 "Convert to Subgraph" 操作时（`graph.convertToSubgraph`），发生以下过程：

1.  **创建容器**：创建一个新的 `Subgraph` 对象。
2.  **节点迁移**：选中的节点从当前 Graph 移除，添加到新的 `Subgraph` 中。
3.  **连接重定向**：
    *   创建 `SubgraphInput` 和 `SubgraphOutput` 节点，用于桥接数据流。
    *   原有的外部连接被重定向到新的 `SubgraphNode` 的输入/输出插槽。
4.  **Widget 晋升 (Promotion)**：
    *   子图内部节点的 Widget 可以被“晋升”到外部 `SubgraphNode` 上显示。
    *   这是通过 `ProxyWidget` 实现的。

## 3. Widget 代理与交互问题

这是导致“点击模型无法打开列表”等问题的核心原因。

### 3.1 ProxyWidget (`core/graph/subgraph/proxyWidget.ts`)
为了让用户在不进入子图的情况下调整参数，ComfyUI 使用 `Proxy` 对象包装了内部节点的 Widget。

*   **机制**：外部 `SubgraphNode` 上的 Widget 实际上是一个 Proxy。
*   **属性拦截**：
    *   `value` 等数据属性会直接读写内部真实 Widget。
    *   `_overlay` 属性存储了元数据（`nodeId`, `graph` 等）。
*   **位置问题**：
    *   当外部代码尝试获取 Widget 的位置（如用于弹出菜单）时，如果直接读取内部 Widget 的坐标，得到的是**相对于子图的局部坐标**。
    *   而弹出菜单（Context Menu）通常需要**屏幕绝对坐标**或**画布全局坐标**。
    *   **后果**：菜单会在错误的位置弹出，或者因为坐标超出视口而被系统隐藏，导致用户感觉“无响应”。

### 3.2 解决方案
在开发自定义节点的 Widget 时，**必须优先使用鼠标事件（Event）的坐标**，而不是依赖节点的坐标属性。

**错误做法（依赖节点坐标）：**
```javascript
// 如果 node 是 Subgraph 内部的节点，node.pos 是局部坐标
const nodeX = node.pos[0];
const nodeY = node.pos[1];
// 计算出的 menuX, menuY 会出错
```

**正确做法（使用事件坐标）：**
```javascript
if (event && event.clientX !== undefined) {
    // clientX/Y 是相对于浏览器视口的绝对坐标，不受子图影响
    targetX = event.clientX;
    targetY = event.clientY;
}
```

## 4. Nodes 2.0 (Schema V2)

`ComfyUI_frontend/src/schemas/nodeDef/nodeDefSchemaV2.ts` 引入了更严格的节点定义规范。

*   **类型安全**：使用 Zod 定义了输入/输出的严格类型（`INT`, `FLOAT`, `STRING`, `COMBO`, `IMAGE` 等）。
*   **结构化选项**：`options` 字段被规范化，例如 `INT` 类型的 `min`, `max`, `step`，`COMBO` 类型的选项列表。
*   **适配建议**：
    *   在编写自定义节点时，应遵循 V2 Schema 定义 `INPUT_TYPES`。
    *   前端代码应能够解析 V2 格式的节点定义，处理更丰富的类型约束。

## 5. 改造与适配建议

针对后续的节点改造：

1.  **事件处理标准化**：
    *   全面检查所有自定义 Widget 的交互逻辑（点击、拖拽）。
    *   将坐标获取逻辑统一封装，优先使用 `event.clientX/Y`。
    *   如果必须使用画布坐标，需检查 `node.graph` 是否为 `Subgraph`，如果是，则需要进行坐标变换（加上父级 SubgraphNode 的偏移）。

2.  **Widget 兼容性**：
    *   确保自定义 Widget 能够被 `ProxyWidget` 正确代理。
    *   避免在 Widget 内部存储非序列化的状态，因为 Proxy 可能无法完美同步所有内部状态。

3.  **V2 Schema 支持**：
    *   逐步迁移节点定义到更严格的类型。
    *   利用新的 Schema 提供的验证能力，减少运行时的类型错误。

## 6. Load Image 节点交互机制分析

本节深入分析 `Load Image` 节点的交互逻辑，包括拖拽上传、粘贴上传以及与 MaskEditor 的集成。

### 6.1 核心组件与文件

*   **Widget 实现**: `renderer/extensions/vueNodes/widgets/composables/useImageUploadWidget.ts`
*   **上传逻辑**: `composables/node/useNodeImageUpload.ts`
*   **预览逻辑**: `composables/node/useNodeImage.ts`
*   **MaskEditor 集成**: `extensions/core/maskeditor.ts` & `maskEditorOld.ts`

### 6.2 拖拽与粘贴 (Drag & Drop / Paste)

`Load Image` 节点通过 `useNodeImageUpload` Composable 实现了统一的文件交互逻辑。

1.  **拖拽 (Drag & Drop)**:
    *   利用 `useNodeDragAndDrop` 监听节点的 `drop` 事件。
    *   当文件被拖入节点时，触发 `handleUploadBatch`。
    *   **流程**: 前端获取 File 对象 -> 调用 `/upload/image` 接口上传 -> 获取服务器返回的文件路径 -> 更新节点的 Combo Widget 值。

2.  **粘贴 (Paste)**:
    *   利用 `useNodePaste` 监听全局或节点的 `paste` 事件。
    *   **逻辑**: 检查粘贴板中的 `files`。如果是图片（通常命名为 `image.png` 且为最近生成），则视为粘贴操作。
    *   **处理**: 同样调用 `uploadFile` 上传到 `/upload/image`，并在 FormData 中标记 `subfolder=pasted`。

3.  **数据流**:
    *   文件上传成功后，后端返回 `{ name, subfolder, type }`。
    *   前端将路径格式化为 `subfolder/filename` (如果存在子文件夹)。
    *   **关键点**: 更新 Widget 的 value 会触发 `nodeOutputStore.setNodeOutputs`，进而刷新图片预览。

### 6.3 Open in MaskEditor / Image Canvas

右键菜单中的 "Open in MaskEditor" 并不是节点自带的功能，而是通过全局扩展动态添加的。

1.  **菜单注册**:
    *   `composables/graph/useImageMenuOptions.ts` 定义了 `getImageMenuOptions`，其中包含 `Open in Mask Editor` 选项。
    *   该选项触发 `Comfy.MaskEditor.OpenMaskEditor` 命令。

2.  **编辑器启动 (`maskeditor.ts`)**:
    *   命令执行 `openMaskEditor(node)`。
    *   **新版**: 调用 `useMaskEditor().openMaskEditor(node)`。
    *   **旧版**:
        *   调用 `ComfyApp.copyToClipspace(node)` 将当前节点信息（包括图片路径）复制到全局 `clipspace` 对象。
        *   设置 `ComfyApp.clipspace_return_node = node`，标记编辑完成后数据要回传给该节点。
        *   打开 `MaskEditorDialogOld`。

### 6.4 编辑后的数据回传

当用户在 MaskEditor 中完成编辑并点击 "Save to node" 时：

1.  **保存逻辑 (`maskEditorOld.ts`)**:
    *   编辑器将 Canvas 内容转换为 Blob。
    *   调用 `uploadMask` (POST `/upload/mask`)，将图片/蒙版上传到服务器（通常作为 `input` 类型，subfolder 为 `clipspace`）。
2.  **更新节点 (`app.ts` - `pasteFromClipspace`)**:
    *   上传成功后，调用 `ComfyApp.onClipspaceEditorSave()`。
    *   该函数检查 `ComfyApp.clipspace_return_node` 是否存在。
    *   如果存在，调用 `ComfyApp.pasteFromClipspace(node)`。
    *   **关键步骤**:
        *   `pasteFromClipspace` 会读取 `ComfyApp.clipspace` 中的数据（包含刚上传的文件信息）。
        *   它会遍历节点的 Widget，寻找名为 `image` 的 Widget。
        *   找到后，将 Widget 的值更新为新上传的文件名（格式：`clipspace/filename [input]`）。
        *   最后触发 `app.graph.setDirtyCanvas(true)` 重绘画布，并调用 `useNodeOutputStore().updateNodeImages(node)` 更新预览。

### 6.5 对自定义节点的启示

如果您开发了自定义的图像加载或处理节点，并希望支持类似功能：

1.  **支持拖拽/粘贴**: 复用 `useNodeImageUpload` 或参考其实现，确保处理 `drop` 和 `paste` 事件并正确上传文件。
2.  **支持 MaskEditor**:
    *   确保节点有 `imgs` 属性（预览图片元素），这样 `useImageMenuOptions` 才会显示编辑选项。
    *   或者，手动在 `getExtraMenuOptions` 中添加调用 `Comfy.MaskEditor.OpenMaskEditor` 的选项。
    *   **核心兼容性要求**:
        *   节点必须有一个名为 `image` 的 Widget (通常是 Combo 或 String 类型) 用于接收文件名。
        *   或者，MaskEditor 能够识别并更新您的特定 Widget（目前逻辑主要硬编码为查找名为 `image` 的 Widget）。

## 7. MaskEditor 图像加载与重编辑机制

ComfyUI 的 MaskEditor 具有一套特殊的机制，用于支持对已编辑图片的“重编辑”（Re-editing）。这主要依赖于特定的文件命名规范和加载逻辑。

### 7.1 文件保存机制
当在 MaskEditor 中编辑并保存（Save to node）时，后端实际上会生成一组相关联的文件（通常为 4 张），文件名中包含相同的**时间戳**。

例如，时间戳为 `1765573661746` 时：
1.  **`clipspace-mask-1765573661746.png`**:
    *   **用途**：作为重编辑时的**基础层（Base Layer）**和**遮罩层（Mask Layer）**来源。
    *   **结构**：通常是一个 RGBA 图像，RGB 通道存储原始图片（或底图），Alpha 通道存储遮罩信息。
2.  **`clipspace-paint-1765573661746.png`**:
    *   **用途**：存储**涂鸦层（Paint Layer）**。
    *   **结构**：通常是透明背景的涂鸦内容。
3.  **`clipspace-painted-1765573661746.png`**:
    *   **用途**：原始图片 + 涂鸦的合成图（无遮罩）。
4.  **`clipspace-painted-masked-1765573661746.png`**:
    *   **用途**：最终输出结果（原始图片 + 涂鸦 + 遮罩应用后的效果）。
    *   **重要性**：这是通常回传给节点 Widget 的文件名。

### 7.2 加载与解析逻辑 (`useMaskEditorLoader.ts`)

当 MaskEditor 尝试加载图片时，会检查文件名是否符合特定模式，以决定是否加载关联的图层数据。

核心函数 `imageLayerFilenamesIfApplicable` 的逻辑如下：

1.  **触发条件**：检查输入文件名是否以 `clipspace-painted-masked-` 开头。
2.  **提取时间戳**：截取前缀后的部分，解析出时间戳（如 `1765573661746`）。
3.  **重构路径**：根据时间戳推断出其他关联文件的名称。

### 7.3 图层恢复流程

如果触发了上述机制，加载器将按以下方式初始化编辑器：

1.  **Base Layer (底图)**：
    *   加载 `clipspace-mask-{timestamp}.png` 的 **RGB 通道**。
2.  **Mask Layer (遮罩)**：
    *   加载 `clipspace-mask-{timestamp}.png` 的 **Alpha 通道**。
3.  **Paint Layer (涂鸦)**：
    *   加载 `clipspace-paint-{timestamp}.png`。

### 7.4 对自定义开发的启示

如果你希望自定义的图片编辑流程能够支持 ComfyUI 原生的“重编辑”体验，你需要：
1.  **保持命名一致性**：生成的图片文件名必须遵循 `clipspace-painted-masked-{timestamp}.png` 的格式。
2.  **保留关联文件**：必须同时保存对应的 `mask` 和 `paint` 文件到 `clipspace` 目录（或其他可访问位置，虽然默认逻辑硬编码了 `clipspace` 前缀，但实际路径由 `subfolder` 决定，通常是 `clipspace`）。
3.  **合成逻辑一致**：`clipspace-mask-*.png` 必须包含正确的 RGB 底图和 Alpha 遮罩。

### 7.5 Widget 值解析
加载器还会尝试解析 Widget 中的值（如 `clipspace/filename.png [input]`）：
*   **Filename**: 提取文件名。
*   **Subfolder**: 提取子文件夹（如 `clipspace`）。
*   **Type**: 提取类型（如 `input`）。
这确保了即使文件位于子目录中，也能被正确找到并触发上述的时间戳解析逻辑。

### 7.6 路径处理与 [input] 标记的特殊注意事项

在实际开发中（如自定义 Load Image 节点），要确保 MaskEditor 能正确加载和保存图片，必须处理好路径解析和自定义标记：

#### 7.6.1 子文件夹路径解析问题
ComfyUI 的后端 `/view` 接口对 `filename` 参数的处理比较简单，通常只提取文件名（`basename`）。如果图片位于子文件夹中（例如 `clipspace/image.png`），直接传递 `filename=clipspace/image.png` 会导致后端在根目录查找文件，从而返回 404 错误。

**正确做法**：
在构造 MaskEditor 的图片 URL 时，必须将路径拆分为 `filename` 和 `subfolder`：

```javascript
// 错误示例
const url = `/view?filename=${encodeURIComponent("clipspace/image.png")}&type=input`;

// 正确示例
const url = `/view?filename=${encodeURIComponent("image.png")}&subfolder=${encodeURIComponent("clipspace")}&type=input`;
```

#### 7.6.2 [input] 后缀与重编辑状态保持
为了标识文件已被编辑或属于输入类型，某些节点实现会在文件名后追加 ` [input]` 后缀（例如 `clipspace/xxx.png [input]`）。

*   **加载前 (Pre-Edit)**：必须移除 ` [input]` 后缀，否则后端无法找到对应文件。
*   **保存后 (Post-Save)**：当 MaskEditor 返回新的文件路径后，如果原文件带有 ` [input]` 标记，必须将其**恢复**。

这是实现“无限次重编辑”的关键：只有文件名保持一致（或标记一致），节点才能在下次打开时正确识别并加载关联的 `mask` 和 `paint` 图层。

## 8. INPUT_TYPES 与前端 Widget 映射关系

本节详细阐述后端 Python 节点定义中的 `INPUT_TYPES` 如何映射到前端的 Widget，涵盖旧版（LiteGraph）和 Nodes 2.0 (Schema V2) 的差异。

### 8.1 映射概览

| 后端类型 (Python) | Schema V2 类型 | 默认 Widget (前端) | 备注 |
| :--- | :--- | :--- | :--- |
| `INT` | `INT` | `number` / `slider` | 可配置 min/max/step |
| `FLOAT` | `FLOAT` | `number` / `slider` | 可配置 min/max/step/round |
| `STRING` | `STRING` | `text` / `customtext` | `multiline=True` 时使用 `customtext` (textarea) |
| `BOOLEAN` | `BOOLEAN` | `toggle` | 显示为开关 |
| `COMBO` (List) | `COMBO` | `combo` | 下拉选择框 |
| `IMAGE` | `IMAGE` | `image` (自定义) | 实际上是上传按钮 + 预览 |
| `MASK` | `MASK` | - | 通常作为输入插槽，非 Widget |

### 8.2 详细映射逻辑

#### 8.2.1 INT (整数)
*   **后端定义**: `("INT", {"default": 1, "min": 0, "max": 10, "step": 1})`
*   **前端处理 (`useIntWidget.ts`)**:
    *   **Display**: 默认为 `number` 输入框。如果 `display="slider"` 且未禁用 Slider，则显示滑块。
    *   **Step**: V2 中直接使用 `step`。旧版中曾使用 `step * 10` 的逻辑，现在已废弃但保留兼容。
    *   **Seed**: 如果输入名称为 `seed` 或 `noise_seed`，会自动添加 `randomize` 和 `reuse` 的控制按钮（`control_after_generate`）。

#### 8.2.2 FLOAT (浮点数)
*   **后端定义**: `("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01})`
*   **前端处理 (`useFloatWidget.ts`)**:
    *   **Precision**: 根据 `step` 自动计算小数位数，或者通过 `round` 参数指定。
    *   **Rounding**: 前端会根据配置（`Comfy.FloatRoundingPrecision`）进行数值舍入，避免浮点数精度问题。

#### 8.2.3 STRING (字符串)
*   **后端定义**: `("STRING", {"default": "", "multiline": True})`
*   **前端处理 (`useStringWidget.ts`)**:
    *   **Single Line**: 使用标准的 LiteGraph `text` widget。
    *   **Multi Line**: 使用 `customtext` widget，底层是一个 HTML `textarea` 元素，支持多行编辑、滚动和动态缩放。
    *   **Dynamic Prompts**: 支持动态提示词语法的处理。

#### 8.2.4 COMBO (下拉框)
*   **后端定义**: `(["Option A", "Option B"],)`
*   **前端处理 (`useComboWidget.ts`)**:
    *   **Standard**: 使用 LiteGraph `combo` widget。
    *   **Image/Video Loaders**: 对于 `LoadImage` 等特定节点，会使用特殊的 `createInputMappingWidget` 或 `createAssetBrowserWidget`，支持预览图和文件上传。
    *   **Remote**: V2 支持 `remote` 属性，允许通过 API 动态获取选项列表。

#### 8.2.5 IMAGE / IMAGEUPLOAD (图片上传)
*   **后端定义**: 通常不直接作为类型，而是通过 `INPUT_TYPES` 返回特定结构，或者在前端通过 `imageInputName` 增强。
*   **前端处理 (`useImageUploadWidget.ts`)**:
    *   **核心**: 实际上是一个 `combo` widget（存储文件名） + 一个 `button` widget（触发上传）。
    *   **交互**: 点击按钮 -> 选择文件 -> 上传 -> 更新 Combo 值 -> 触发预览更新。
    *   **Preview**: 使用 `useImagePreviewWidget` 在节点上绘制图片预览。

### 8.3 Schema V2 的改进

Nodes 2.0 (V2) 在 `src/schemas/nodeDef/nodeDefSchemaV2.ts` 中定义了更严格的规范：

1.  **明确的类型**: 不再依赖隐式的列表结构，而是使用 Zod 定义明确的 `InputSpec` 对象。
2.  **Options 结构化**: `min`, `max`, `step` 等属性被标准化，不再混杂在字典中。
3.  **UI 提示 (Hints)**: 引入 `display` (slider/knob), `placeholder`, `tooltip` 等 UI 专用属性。
4.  **验证**: 前端在创建 Widget 时会校验 InputSpec 是否符合 Schema，提供更好的错误提示。

### 8.4 对开发者的建议

*   **优先使用 V2**: 新开发的节点应尽量遵循 V2 Schema 定义输入，以获得更好的类型支持和 UI 控制。
*   **特殊 Widget**: 如果需要 Image Upload 或 Mask Editor 功能，参考 `LoadImage` 的实现，组合使用 Combo 和 Button，并利用 Composable (`useNodeImageUpload`) 复用逻辑。
*   **类型转换**: 前端会自动处理 V2 到 V1 的兼容性转换（`transformInputSpecV2ToV1`），因此旧版节点通常能无缝运行，但新特性需要新 Schema 支持。

## 9. 模型列表获取与管理

本节介绍如何在前端代码中获取模型列表（如 Checkpoints, LoRAs），包括初始化加载和动态刷新。

### 9.1 通过 API 获取模型列表

`ComfyUI_frontend/src/scripts/api.ts` 提供了直接与后端交互的方法。

#### 9.1.1 获取指定类型的模型 (`getModels`)

如果需要在运行时动态获取某种类型的模型列表（例如点击刷新按钮时）：

```typescript
import { api } from '@/scripts/api'

// 获取所有 Checkpoint 模型列表
const checkpoints = await api.getModels('checkpoints')
console.log(checkpoints) 
// 输出示例: [{name: "v1-5-pruned.ckpt", ...}, {name: "SDXL.safetensors", ...}]

// 获取 LoRA 列表
const loras = await api.getModels('loras')
```

#### 9.1.2 获取所有可用的模型文件夹 (`getModelFolders`)

如果不确定有哪些模型类型可用：

```typescript
const folders = await api.getModelFolders()
console.log(folders)
// 输出示例: [{name: "checkpoints"}, {name: "loras"}, {name: "embeddings"}, ...]
```

### 9.2 从节点定义中获取 (静态)

标准的 `Load Checkpoint` 等节点，其模型列表是在后端启动时生成的，并作为 `INPUT_TYPES` 的一部分发送给前端。

1.  调用 `api.getNodeDefs()` 获取所有节点定义。
2.  查找目标节点（如 `CheckpointLoaderSimple`）。
3.  读取其输入参数中的列表。

```typescript
const defs = await api.getNodeDefs()
const loaderDef = defs['CheckpointLoaderSimple']
const modelList = loaderDef.input.required.ckpt_name[0] // 这是一个字符串数组
```

### 9.3 动态刷新 Widget 选项

如果您开发了一个自定义 Widget，并希望它能刷新模型列表：

1.  **定义 Widget**: 使用 `combo` 类型。
2.  **添加刷新逻辑**:
    *   在 Widget 上添加一个按钮或右键菜单项。
    *   点击时调用 `api.getModels('your_model_type')`。
    *   更新 Widget 的 `options.values`。

```typescript
// 示例：自定义刷新逻辑
async function refreshModelWidget(node, widgetName, folderName) {
    const widget = node.widgets.find(w => w.name === widgetName)
    if (!widget) return

    // 1. 获取最新列表
    const files = await api.getModels(folderName)
    const fileNames = files.map(f => f.name)

    // 2. 更新 Widget 选项
    widget.options.values = fileNames
    
    // 3. 保持当前选中值（如果仍在列表中），否则重置
    if (!fileNames.includes(widget.value)) {
        widget.value = fileNames[0] || ''
    }
    
    // 4. 触发重绘
    node.setDirtyCanvas(true)
}
```

## 10. 遮罩编辑器 (MaskEditor) 深度分析与改造

ComfyUI 的新版 MaskEditor (`ComfyUI_frontend/src/components/maskeditor`) 实际上是一个功能完整的图像编辑器，不仅支持遮罩绘制，还支持图层合成（绘画层）。

### 10.1 现状：与 Node 强耦合

目前的实现中，编辑器与 `LGraphNode` 高度绑定：

1.  **入口**: `useMaskEditor().openMaskEditor(node)` 强制要求传入 `LGraphNode`。
2.  **数据加载**: `MaskEditorContent.vue` 调用 `loader.loadFromNode(node)`。
3.  **数据解析**: `useMaskEditorLoader.ts` 内部解析节点的 `imgs` 属性或 Widgets 来确定图片 URL。

这种设计导致无法直接打开编辑器来编辑一张不在节点上的图片（例如从剪贴板粘贴的，或者生成的中间结果）。

### 10.2 改造方案：支持任意图像输入

为了支持传入任意图像数据，需要对前端代码进行解耦改造。

#### 10.2.1 定义通用输入接口

在 `src/stores/maskEditorDataStore.ts` 中，`EditorInputData` 结构已经比较通用，但 `nodeId` 和 `sourceRef` 依然暗示了文件系统的依赖。

建议扩展 `EditorInputData` 或引入新的 `CustomInputData`：

```typescript
// 建议的接口扩展
interface CustomEditorInput {
  baseImage: string | Blob | HTMLImageElement; // 底图
  maskImage?: string | Blob | HTMLImageElement; // 初始遮罩 (可选)
  paintImage?: string | Blob | HTMLImageElement; // 初始绘画层 (可选，用于 Image Editor 模式)
  nodeId?: number; // 可选，如果需要回传给特定节点
  callback?: (result: EditorOutputData) => void; // 保存时的回调，返回合成结果
}
```

#### 10.2.2 改造 Loader (`useMaskEditorLoader.ts`)

需要在 `useMaskEditorLoader.ts` 中添加 `loadFromData` 方法：

```typescript
// 伪代码示例
async function loadFromData(input: CustomEditorInput) {
  // 1. 将 Blob/String 转换为 HTMLImageElement (复用 loadImageLayer 逻辑)
  const baseLayer = await createLayerFromInput(input.baseImage);
  const maskLayer = input.maskImage 
    ? await createLayerFromInput(input.maskImage) 
    : createBlankMask(baseLayer.width, baseLayer.height);
  
  const paintLayer = input.paintImage
    ? await createLayerFromInput(input.paintImage)
    : undefined;

  // 2. 填充 Store
  dataStore.inputData = {
    baseLayer,
    maskLayer,
    paintLayer,
    nodeId: input.nodeId || -1,
    // ...
  };
  
  // 3. 标记非节点来源，改变保存行为
  dataStore.isCustomSource = true; 
  dataStore.saveCallback = input.callback;
}
```

#### 10.2.3 改造 UI 组件 (`MaskEditorContent.vue`)

修改 Props 定义，允许不传 `node`，而是传 `inputData`：

```vue
<script setup lang="ts">
const props = defineProps<{
  node?: LGraphNode;
  customData?: CustomEditorInput;
}>();

const initUI = async () => {
  // ...
  if (props.node) {
    await loader.loadFromNode(props.node);
  } else if (props.customData) {
    await loader.loadFromData(props.customData);
  }
  // ...
};
</script>
```

#### 10.2.4 改造入口 (`useMaskEditor.ts`)

暴露更通用的打开方法：

```typescript
export function useMaskEditor() {
  // 原有方法保持兼容
  const openMaskEditor = (node: LGraphNode) => { ... };

  // 新增通用方法
  const openEditorWithImage = (data: CustomEditorInput) => {
    useDialogStore().showDialog({
      key: 'global-mask-editor',
      component: MaskEditorContent,
      props: {
        customData: data
      },
      // ...
    });
  };

  return { openMaskEditor, openEditorWithImage };
}
```

### 10.3 扩展应用场景

一旦完成上述改造，可以实现以下功能：

1.  **全局图片编辑**: 在 Gallery 或历史记录中，右键任意图片点击 "Edit Mask"，直接调用编辑器。
2.  **剪贴板编辑**: 监听全局粘贴事件，检测到图片后弹出 "Edit?" 提示，直接进入编辑模式。
3.  **自定义节点交互**: 开发更复杂的节点（如 `Inpaint Anything`），在节点 UI 上提供 "Draw Mask" 按钮，点击后传入节点当前的 Tensor（转为 Base64）进行编辑，编辑完成后回调上传 Mask。
4.  **独立 Image Editor**: 通过传入 `paintImage`，可以实现对图片的涂鸦、标记等功能，而不仅仅是遮罩。

### 10.4 进阶：深度借用 UI 与保存逻辑 (Reuse Strategy)

针对 "新建函数、自定义传参、只借用 UI 和保存/上传逻辑" 的需求，我们需要对 `useMaskEditorSaver.ts` 进行更细粒度的拆分，将 **"数据准备"**、**"上传"** 和 **"节点更新"** 三个步骤解耦。

#### 10.4.1 核心逻辑拆解

目前 `save()` 函数将所有逻辑耦合在一起。为了复用，建议将上传逻辑提取为独立服务：

```typescript
// src/composables/maskeditor/useMaskEditorSaver.ts (重构建议)

// 1. 导出纯粹的上传逻辑，不依赖 dataStore/node
export async function uploadEditorLayers(outputData: EditorOutputData, originalRef: ImageRef): Promise<EditorOutputData> {
    // 复用原有的 uploadAllLayers 逻辑
    const actualMaskedRef = await uploadMask(outputData.maskedImage, originalRef)
    const actualPaintRef = await uploadImage(outputData.paintLayer, originalRef)
    // ...
    // 返回更新了 Ref 的数据对象
    return {
        ...outputData,
        maskedImage: { ...outputData.maskedImage, ref: actualMaskedRef },
        // ...
    }
}
```

#### 10.4.2 自定义入口函数实现

实现一个完全独立的入口函数，不依赖 `LGraphNode`，但完整复用 UI 和上传机制：

```typescript
// 你的自定义业务逻辑文件

import { useDialogStore } from '@/stores/dialogStore'
import MaskEditorContent from '@/components/maskeditor/MaskEditorContent.vue'
import { uploadEditorLayers } from '@/composables/maskeditor/useMaskEditorSaver'

interface CustomEditOptions {
    imageUrl: string; // 初始图片 URL
    onSave: (uploadedRefs: EditorOutputData) => void; // 保存后的回调
}

export function openCustomImageEditor(options: CustomEditOptions) {
    // 1. 构造符合 EditorInputData 接口的数据
    // 注意：这里需要自行实现 loadFromUrl 逻辑，或者复用 useMaskEditorLoader 中的 helper
    const customInputData = {
        baseLayer: await loadImageLayer(options.imageUrl),
        maskLayer: createBlankMask(), 
        // 关键：传入自定义的保存处理器
        saveHandler: async (outputData) => {
            // A. 复用 ComfyUI 的上传逻辑
            // 构造一个临时的 originalRef 用于上传参数
            const tempRef = { filename: 'custom_edit.png', type: 'temp' };
            const uploadedData = await uploadEditorLayers(outputData, tempRef);
            
            // B. 执行用户回调，将上传后的路径传出去
            options.onSave(uploadedData);
            
            // C. 关闭弹窗
            useDialogStore().closeDialog('global-mask-editor');
        }
    };

    // 2. 打开弹窗 (复用 UI)
    useDialogStore().showDialog({
        key: 'global-mask-editor',
        component: MaskEditorContent,
        props: {
            customData: customInputData // 需要 MaskEditorContent 支持此 Prop
        },
        // ... 保持原有样式配置
    });
}
```

#### 10.4.3 效果与优势

通过这种方式，你实际上是**"借用"**了整个编辑器：

1.  **UI 借用**: 直接使用 `MaskEditorContent`，获得完整的画布、笔刷、图层 UI。
2.  **逻辑借用**: 
    *   **编辑逻辑**: 笔刷绘制、撤销重做等完全由组件内部管理。
    *   **合成逻辑**: `prepareOutputData` 依然在内部工作，生成最终的 Blob。
    *   **上传逻辑**: 通过提取的 `uploadEditorLayers`，你不需要自己写 `FormData` 和 API 调用。
3.  **完全解耦**: 
    *   不需要 `LGraphNode` 存在。
    *   `onSave` 回调给你的是**已经上传到服务器的文件路径** (filename, subfolder, type)。
    *   你可以拿这个路径去更新任意变量、数据库，或者通过 API 发送给其他服务，而仅限于更新节点 Widget。

### 10.5 终极方案：基于 Mock Node 的无侵入式集成 (Hack Strategy)

鉴于 **"不能修改原生代码"** 的严格约束，上述修改源码的方案虽然优雅但不可行。我们必须采用一种 **"欺骗" (Mocking)** 策略，利用 JS 的动态特性，让 MaskEditor 以为它在编辑一个节点，但实际上是在编辑我们提供的任意数据。

#### 10.5.1 原理分析

1.  **利用已注册的命令**: `Comfy.MaskEditor` 扩展注册了一个命令 `Comfy.MaskEditor.OpenMaskEditor`。这个命令的逻辑是：获取当前选中的节点 (`app.canvas.selected_nodes`)，然后打开编辑器。
2.  **利用 Duck Typing**: 编辑器并不检查节点是否真的在图表中，只要它长得像 `LGraphNode`（有 `imgs`, `widgets` 等属性），编辑器就能工作。
3.  **拦截保存结果**: 编辑器保存时会更新节点的 `widgets` 并调用 `callback`。我们可以通过在伪造节点上挂载 callback 来截获上传后的文件路径。

#### 10.5.2 实现代码 (可直接在你的扩展中使用)

```javascript
import { app } from "../../scripts/app.js"; // 假设在 web/js 环境下

/**
 * 打开 MaskEditor 编辑任意图片
 * @param {string} imageUrl - 图片的 URL (必须是浏览器可访问的，如 /view?filename=...)
 * @param {Function} onSave - 保存回调，接收 (filename, subfolder, type)
 */
export function openMaskEditorForImage(imageUrl, onSave) {
    // 1. 构造 Mock Node
    // 编辑器需要读取 node.imgs[0].src 作为底图
    // 编辑器保存时会查找名为 'image' 的 widget 并更新它
    const mockNode = {
        id: -1, // 虚拟 ID
        type: "MockNode",
        title: "Mock Image Editor",
        imgs: [{
            src: imageUrl,
            width: 512, // 尺寸不影响加载，编辑器会重新读取图片
            height: 512
        }],
        widgets: [{
            name: "image",
            value: "", // 初始值
            // 关键：拦截保存结果
            callback: (newValue) => {
                console.log("[MockEditor] Saved:", newValue);
                // newValue 格式通常为: "clipspace/filename.png [input]"
                // 解析出文件名
                // 简单正则提取: (subfolder/)?filename.ext [type]
                // 但通常编辑器返回的是标准格式，我们可以简单解析
                
                // 注意：useMaskEditorSaver 会更新 widget value
                if (onSave) {
                    onSave(newValue);
                }
            }
        }],
        // 模拟必要的方法防止报错
        setDirtyCanvas: () => {},
        setSize: () => {},
        getBounding: () => [0,0,100,100],
        isResizeable: () => false,
        properties: {}
    };

    // 2. 找到 MaskEditor 的打开命令
    const ext = app.extensions.find(e => e.name === "Comfy.MaskEditor");
    if (!ext) {
        console.error("Comfy.MaskEditor extension not found");
        return;
    }

    const cmd = ext.commands.find(c => c.id === "Comfy.MaskEditor.OpenMaskEditor");
    if (!cmd) {
        console.error("OpenMaskEditor command not found");
        return;
    }

    // 3. 实施欺骗：伪造选中状态并执行命令
    const originalSelection = app.canvas.selected_nodes;
    
    // 临时替换选中节点为我们的 Mock Node
    app.canvas.selected_nodes = { [mockNode.id]: mockNode };

    try {
        // 执行命令，这会调用 useMaskEditor().openMaskEditor(mockNode)
        cmd.function();
    } catch (e) {
        console.error("Failed to open MaskEditor:", e);
    } finally {
        // 4. 恢复现场
        app.canvas.selected_nodes = originalSelection;
    }
}
```

#### 10.5.3 使用示例

假设你在开发一个自定义节点的右键菜单，或者在图库中添加按钮：

```javascript
// 在你的扩展代码中
import { openMaskEditorForImage } from "./utils.js";

// 按钮点击事件
handleEditBtnClick(imageSrc) {
    openMaskEditorForImage(imageSrc, (resultString) => {
        // resultString 例如: "clipspace/clipspace-mask-123456.png [input]"
        alert("编辑完成，保存路径: " + resultString);
        
        // 你可以将这个字符串发送给后端，或者更新当前节点的某个 widget
        this.updateMyWidget(resultString);
    });
}
```

#### 10.5.4 局限性

1.  **依赖内部实现**: 此方法依赖 `Comfy.MaskEditor` 扩展的内部结构（commands 数组），如果未来 ComfyUI 更改了扩展注册方式或命令 ID，此代码可能会失效.
2.  **Clipspace 限制**: 新版 MaskEditor 似乎不支持通过 Clipspace 按钮打开（源码中逻辑缺失），因此必须使用上述 Command Hack 方法。
3.  **图片 URL**: 传入的 `imageUrl` 必须是有效的。如果是上传的图片，通常格式为 `./view?filename=xxx&subfolder=yyy&type=input`。

通过这种方式，你可以在**完全不修改 ComfyUI 源码**的情况下，复用其强大的 MaskEditor 功能。