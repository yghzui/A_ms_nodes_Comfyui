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

## 11. 节点输出 Preview Image 机制（静态图 / 动图 / 视频）

本节说明前端是如何从后端输出构建节点上的图片预览（Preview Image），包括数据流、图片加载和 Canvas 渲染逻辑，便于在自定义节点或 Subgraph 场景下正确复用。

### 11.1 数据来源：节点输出到预览 URL

相关文件：

*   `ComfyUI_frontend/src/stores/imagePreviewStore.ts`
*   `ComfyUI_frontend/src/scripts/api.ts`

核心流程：

1.  **后端执行完成 → 输出结构**  
    *   后端执行某个节点后，通过 WebSocket/HTTP 返回 `ExecutedWsMessage['output']`，其中常见结构为：  
        *   `output.images: [{ filename, subfolder, type }, ...]`  
        *   `output.animated: [true/false, ...]`（标记对应输出是否为动图）
2.  **前端存储输出：`useNodeOutputStore`**  
    *   `useNodeOutputStore()` 在 `imagePreviewStore.ts:39` 中定义，内部维护：  
        *   `app.nodeOutputs[nodeLocatorId]`：节点输出原始结构。  
        *   `app.nodePreviewImages[nodeLocatorId]`：已生成的预览 URL 队列。  
    *   入口方法：  
        *   `setNodeOutputsByExecutionId(executionId, outputs)`：根据执行 ID 写入 `app.nodeOutputs`（`imagePreviewStore.ts:196-205`）。  
        *   `setNodePreviewsByExecutionId(executionId, previewImages)`：直接写入预览 URL（`imagePreviewStore.ts:214-227`）。  
3.  **获取输出与预览：`getNodeOutputs` / `getNodePreviews`**  
    *   `getNodeOutputs(node)`：根据当前 Graph + 节点 ID 计算 NodeLocatorId，从 `app.nodeOutputs` 取出对应输出（`imagePreviewStore.ts:63-67`）。  
    *   `getNodePreviews(node)`：从 `app.nodePreviewImages` 读取已经缓存的预览 URL（`imagePreviewStore.ts:69-71`）。  
4.  **构造预览 URL：`getNodeImageUrls(node)`**  
    *   优先使用 `getNodePreviews(node)`（如果有手动注入的预览 URL）（`imagePreviewStore.ts:108-110`）。  
    *   否则从 `getNodeOutputs(node)` 取 `outputs.images` 并生成 `/view` URL（`imagePreviewStore.ts:112-121`）：  
        1.  根据 `isImageOutputs(node, outputs)` 判断是否是「普通静态图输出」：  
            *   若节点本身为视频节点或 `node.animatedImages` 为真，则返回 `false`（`imagePreviewStore.ts:77-83`）。  
            *   若没有 `images` 或包含 `svg`，也返回 `false`（`imagePreviewStore.ts:84-89`）。  
        2.  若是静态图输出，则通过 `app.getPreviewFormatParam()` 生成如 `&preview=1&type=...` 的参数（`imagePreviewStore.ts:95-106`）。  
        3.  通过 `parseFilePath(image)` + `new URLSearchParams(image)` 拼出查询串，再加上 `app.getRandParam()`（防缓存），最后调用 `api.apiURL('/view?...')` 得到完整 URL（`imagePreviewStore.ts:115-121`）。

**要点：**

*   Preview Image 的 URL 与 `/view?filename=...&subfolder=...&type=...` 一致，和 MaskEditor 部分的路径规则保持统一。
*   是否走「静态图预览」取决于 `isImageOutputs` 判断，视频节点会走单独的视频预览逻辑（见后文）。

### 11.2 节点侧图片 / 视频加载：`useNodeImage` / `useNodeVideo`

相关文件：

*   `ComfyUI_frontend/src/composables/node/useNodeImage.ts`

公共加载器 `useNodePreview(node, options)`（`useNodeImage.ts:38-96`）：

*   参数 `options`：
    *   `loadElement(url)`：把单个 URL 转为 `HTMLImageElement` 或 `HTMLVideoElement`。  
    *   `onLoaded(elements)`：所有媒体加载成功后的回调。  
    *   `onFailedLoading()`：全部失败时的处理。
*   内部逻辑：
    1.  使用 `loadElementWithTimeout(url, retryCount)` 包裹 `loadElement`，带超时（`MEDIA_LOAD_TIMEOUT = 8192` ms）和最多一次重试（`MAX_RETRIES = 1`）（`useNodeImage.ts:45-59`）。  
    2.  `loadElements(urls)` 并发加载所有 URL（`useNodeImage.ts:61-62`）。  
    3.  对外暴露 `showPreview({ block?: boolean })`（`useNodeImage.ts:67-91`）：  
        *   若 `node.isLoading` 为 true，则直接返回，防止重复加载。  
        *   调用 `nodeOutputStore.getNodeImageUrls(node)` 获取 URL 列表。  
        *   若 `options.block` 为真，加载期间将 `node.isLoading = true`。  
        *   加载完成后过滤掉 `null`，将有效元素塞给 `onLoaded(validElements)`，并调用 `node.graph?.setDirtyCanvas(true)` 触发重绘。  
        *   出错则调用 `onFailedLoading`，最后 `finally` 保证 `node.isLoading = false`。

#### 11.2.1 静态图预览：`useNodeImage`

*   `useNodeImage(node, callback?)`（`useNodeImage.ts:98-125`）：
    *   将 `node.previewMediaType` 标记为 `'image'`。  
    *   `loadElement(url)` 通过 `new Image()` 加载 URL，成功回调 `resolve(img)`，失败则 `resolve(null)`（`useNodeImage.ts:104-110`）。  
    *   `onLoaded(elements)`：  
        1.  将 `node.imageIndex = null`，表示当前处于「缩略图模式」。  
        2.  将 `node.imgs = elements`。后续所有预览绘制逻辑都围绕 `node.imgs` 实现（`useNodeImage.ts:113-115`）。  
        3.  调用可选 `callback()`。
    *   `onFailedLoading` 时重置 `node.imgs = undefined`（`useNodeImage.ts:121-123`）。  
    *   返回通用的 `showPreview()` 接口供外部调用。

#### 11.2.2 视频预览：`useNodeVideo`

*   `useNodeVideo(node, callback?)`（`useNodeImage.ts:127-202`）：  
    *   将 `node.previewMediaType` 标记为 `'video'`。  
    *   使用 `<video>` DOM 元素加载 URL，默认开启 `playsInline / controls / loop`（`useNodeImage.ts:149-165`）。  
    *   计算视频在当前节点宽度下的合适宽高（`fitDimensionsToNodeWidth`，`useNodeImage.ts:137-147`）。  
    *   通过 `node.addDOMWidget('video-preview', 'video', container, { canvasOnly: true, hideOnZoom: false })` 在节点上挂一个 DOM Widget（`useNodeImage.ts:167-179`）。  
    *   `onLoaded(videoElements)` 中将 `<video>` 放入 `node.videoContainer`，并调用 `callback()`（`useNodeImage.ts:182-193`）。  
    *   `onFailedLoading` 时重置 `node.videoContainer`。

**要点：**

*   对节点来说，Preview Image 实质就是：  
    *   静态图：`node.imgs: HTMLImageElement[]`。  
    *   视频：`node.videoContainer` 中的 `<video>` DOM。
*   后续无论是 Canvas 绘制还是自定义 Lightbox，只要复用这两个字段，即可得到当前节点的预览媒体。

### 11.3 Canvas 预览 Widget：`useImagePreviewWidget` 与节点绘制

相关文件：

*   `ComfyUI_frontend/src/renderer/extensions/vueNodes/widgets/composables/useImagePreviewWidget.ts`
*   `ComfyUI_frontend/src/composables/node/useNodeCanvasImagePreview.ts`
*   `ComfyUI_frontend/src/scripts/ui/imagePreview.ts`

#### 11.3.1 Widget 构造与挂载：`useImagePreviewWidget` / `useNodeCanvasImagePreview`

*   `ImagePreviewWidget` 继承自 `BaseWidget`（`useImagePreviewWidget.ts:239-299`）：  
    *   类型为 `type: 'custom'`，`value` 只是占位。  
    *   `serialize = false`，不会出现在工作流 JSON 中。  
    *   `drawWidget(ctx)` 调用 `renderPreview(ctx, this.node, this.y, this.computedHeight)`，在节点内部 Canvas 区域绘制图片（`useImagePreviewWidget.ts:259-260`）。  
    *   `onPointerDown` 中接入拖拽节点逻辑（委托给 `app.canvas`，`useImagePreviewWidget.ts:263-288`）。  
    *   `computeLayoutSize` 返回最小高度 220（`useImagePreviewWidget.ts:293-297`）。
*   工厂函数 `useImagePreviewWidget()`：  
    *   返回一个 `widgetConstructor(node, inputSpec)`，内部通过 `node.addCustomWidget(new ImagePreviewWidget(...))` 将预览 Widget 挂到节点上（`useImagePreviewWidget.ts:301-311`）。
*   `useNodeCanvasImagePreview()`（`useNodeCanvasImagePreview.ts`）：  
    *   `showCanvasImagePreview(node)`：  
        *   若 `node.imgs` 为空，直接返回。  
        *   若 `node.widgets` 中不存在名为 `'$$canvas-image-preview'` 的 Widget，则调用 `imagePreviewWidget(node, { type: 'IMAGE_PREVIEW', name: '$$canvas-image-preview' })` 新建一个 `ImagePreviewWidget`（`useNodeCanvasImagePreview.ts:13-24`）。  
    *   `removeCanvasImagePreview(node)`：  
        *   查找名为 `'$$canvas-image-preview'` 的 Widget，调用 `onRemove()` 并从 `node.widgets` 列表中删除（`useNodeCanvasImagePreview.ts:26-41`）。

#### 11.3.2 多图缩略图与大图模式：`renderPreview`

核心绘制逻辑在 `renderPreview` 中（`useImagePreviewWidget.ts:15-237`）：

1.  **点击识别**：  
    *   通过 `canvas.graph_mouse` 获取鼠标在画布中的坐标。  
    *   利用 `node.pointerDown` 记录按下的图片 index 和坐标，鼠标释放时若坐标未变，则认为是点击而不是拖拽，从而更新 `node.imageIndex`（`useImagePreviewWidget.ts:24-31`）。
2.  **数据准备**：  
    *   `const imgs = node.imgs ?? []`；`numImages = imgs.length`（`useImagePreviewWidget.ts:34-36`）。  
    *   若只有一张图片且 `imageIndex` 为空，自动进入单图模式（`useImagePreviewWidget.ts:37-40`）。  
    *   从设置中读取 `Comfy.Node.AllowImageSizeDraw`，决定是否在图下方绘制「宽 × 高」文字（`useImagePreviewWidget.ts:42-44`）。
3.  **缩略图矩阵模式（imageIndex == null）**：  
    *   判断是否所有图片的宽高比相同（`is_all_same_aspect_ratio(imgs)`）。  
        *   若不相同，则构造一个方形占位图数组，用 `calculateImageGrid(fakeImgs, dw, dh)` 计算网格，避免严重拉伸（`useImagePreviewWidget.ts:56-76`）。  
        *   若相同，则直接 `calculateImageGrid(imgs, dw, dh)`（`useImagePreviewWidget.ts:78-83`）。  
    *   遍历每张图片计算所在行列 `(row, col)`，转换为绘制坐标 `(x, y)`（`useImagePreviewWidget.ts:88-93`）。  
    *   使用 `LiteGraph.isInsideRectangle` 判断鼠标是否悬停在某个格子上：  
        *   若是，设置 `node.overIndex = i`，调整 `ctx.filter` 和光标样式，模拟 hover / click 效果（`useImagePreviewWidget.ts:95-114`）。  
        *   将每个 cell 的矩形信息保存在 `node.imageRects` 中，供其他逻辑（例如右键菜单、Lightbox）使用（`useImagePreviewWidget.ts:86-88, 116-117`）。  
    *   按比例缩放图片填满 cell，必要时画矩形边框（`useImagePreviewWidget.ts:118-143`）。  
    *   若没有任何 cell 被 hover，清空 `node.pointerDown` 与 `node.overIndex`（`useImagePreviewWidget.ts:149-152`）。
4.  **单图大图模式（imageIndex != null）**：  
    *   取当前 `img = imgs[imageIndex]`，根据节点宽高 `dw / dh` 计算缩放（`useImagePreviewWidget.ts:156-167`）。  
    *   居中绘制在节点内部 Canvas 区域（`useImagePreviewWidget.ts:168-170`）。  
    *   若 `Comfy.Node.AllowImageSizeDraw` 为真，在图下方绘制 `宽 × 高` 文本（`useImagePreviewWidget.ts:172-179`）。
5.  **翻页与关闭按钮**：  
    *   内部 `drawButton(x, y, size, text)` 负责绘制按钮矩形并检测点击（`useImagePreviewWidget.ts:182-220`）。  
    *   当 `numImages > 1` 时：  
        *   在右下角绘制翻页按钮，文本为 `当前索引/总数`（`useImagePreviewWidget.ts:222-225`）。  
        *   点击时将下一张 index 写入 `node.pointerDown.index`，在下一轮绘制时切换图片（`useImagePreviewWidget.ts:226-229`）。  
        *   在右上角绘制关闭按钮 `"x"`，点击时将 `node.pointerDown.index` 设为 `null`，回到缩略图矩阵模式（`useImagePreviewWidget.ts:232-235`）。

#### 11.3.3 键盘控制：左右翻页与退出

`litegraphService.ts` 中通过 `addNodeKeyHandler` 为节点挂载了统一的 `onKeyDown`（`litegraphService.ts:777-820`）：

*   前提条件：节点未折叠、`this.imgs` 非空且 `this.imageIndex !== null`（处于单图模式）（`litegraphService.ts:786-787`）。  
*   `ArrowLeft` / `ArrowRight`：  
    *   左键减一、右键加一，并对 `this.imgs.length` 取模实现循环翻页（`litegraphService.ts:792-803`）。  
*   `Escape`：  
    *   将 `this.imageIndex = null`，回到缩略图模式（`litegraphService.ts:809-812`）。  
*   处理成功后阻止默认事件和后续传播（`litegraphService.ts:814-818`）。

### 11.4 总调度入口：`updatePreviews` 与节点生命周期

相关文件：

*   `ComfyUI_frontend/src/services/litegraphService.ts`

#### 11.4.1 `updatePreviews(node)`：判断何时加载 / 更新预览

*   `addDrawBackgroundHandler(nodeClass)` 中将所有节点类的 `onDrawBackground` 指向 `updatePreviews(this)`（`litegraphService.ts:763-775`）。  
    *   即：每次节点需要绘制背景时，都会尝试更新预览。
*   `updatePreviews(node, callback?)` 包装了 `unsafeUpdatePreviews.call(node, callback)` 并捕获异常（`litegraphService.ts:702-707`）。
*   `unsafeUpdatePreviews(this, callback?)` 逻辑（`litegraphService.ts:709-756`）：  
    1.  若节点已折叠（`this.flags.collapsed`），直接返回。  
    2.  通过 `useNodeOutputStore()` 获取 `output` 与 `preview`：  
        *   `output = nodeOutputStore.getNodeOutputs(this)`。  
        *   `preview = nodeOutputStore.getNodePreviews(this)`（`litegraphService.ts:718-719`）。  
    3.  判断是否有新的输出 / 预览：  
        *   `isNewOutput = output && this.images !== output.images`。  
        *   `isNewPreview = preview && this.preview !== preview`（`litegraphService.ts:721-723`）。  
        *   有新数据时更新 `this.images` / `this.preview`（`litegraphService.ts:724-725`）。  
    4.  若有新输出/预览：  
        *   将 `this.animatedImages` 设置为 `output?.animated?.find(Boolean)`（`litegraphService.ts:728-729`）。  
        *   根据文件名和节点类型判断是否是视频：  
            *   若 `this.animatedImages` 为真且任一文件名包含 `webp` 或 `png`，视为动图。  
            *   否则若 `this.animatedImages` 为真但文件名不含上述后缀，或 `isVideoNode(this)` 为真，则视为视频（`litegraphService.ts:730-738`）。  
        *   若是视频：调用 `useNodeVideo(this, callback).showPreview()`。  
        *   否则：调用 `useNodeImage(this, callback).showPreview()`（`litegraphService.ts:739-743`）。  
    5.  若最终 `this.imgs` 仍为空，说明没有可用图片，直接返回（`litegraphService.ts:746-747`）。  
    6.  根据 `this.animatedImages` 决定采用哪种展示方式：  
        *   若为真：  
            *   `removeCanvasImagePreview(this)`，移除静态图 Canvas Widget。  
            *   `showAnimatedPreview(this)`，使用 `useNodeAnimatedImage()` 以 DOM Widget 动态显示动图（`litegraphService.ts:749-751`）。  
        *   否则：  
            *   `removeAnimatedPreview(this)`，移除动画 DOM Widget。  
            *   `showCanvasImagePreview(this)`，使用 Canvas Widget 绘制静态图（`litegraphService.ts:752-755`）。

#### 11.4.2 对 Subgraph / ProxyWidget 的影响

*   Subgraph 外层节点本质上也是 `LGraphNode`，只要其 `nodeId` 对应的输出被写入 `useNodeOutputStore`，就会走与普通节点一致的 Preview 流程。  
*   内部节点若将某个 Widget 晋升为 `ProxyWidget`，只要该节点最终有图片输出，同样通过 `updatePreviews` 驱动预览，差异只在于：  
    *   拖拽 / 右键菜单中如果要打开 Lightbox 或 MaskEditor，需要注意坐标体系（详见前文 ProxyWidget 小节：必须使用 `event.clientX/Y` 而不是 `node.pos`）。  
*   对于自定义 Subgraph 或 GroupNode 的开发者：  
    *   如需在外层节点上展示内部节点的输出预览，可以：  
        1.  让内部节点产生实际 `IMAGE` 输出（正常执行即可）。  
        2.  或者手动调用 `useNodeOutputStore().setNodePreviewsByNodeId(nodeId, previewUrls)`，给外层节点直接注入预览 URL。  
    *   一旦 `app.nodeOutputs` / `app.nodePreviewImages` 中的数据准备好，`updatePreviews` 会自动处理静态图 / 动图 / 视频的展示，无需手动操作 `node.imgs`。

### 11.5 对自定义节点 / 扩展的实践建议

1.  **不要手动绕过 `useNodeOutputStore`**：  
    *   自己拼 `node.imgs = [...]` 虽然也能画，但丢失了与后端输出的关联，可能影响重新执行时的预览刷新与内存回收。  
2.  **复用 `/view` 路径规则**：  
    *   无论是 Preview Image 还是 MaskEditor，都应将 `filename` 与 `subfolder` 拆开传给 `/view`，避免子目录 404 问题。  
3.  **为自定义输出显式设置 ResultItem**：  
    *   若你的节点在前端主动生成图片文件名，可以构造 `ResultItem` 并调用：  
        *   `setNodeOutputs(node, filenames, { folder: 'input' | 'output', isAnimated })`。  
    *   这样可以保证 Preview / Gallery / MaskEditor 等所有后续功能都能复用统一数据结构。  
4.  **Subgraph 中的 Lightbox / 右键预览**：  
    *   可以通过 `node.imgs` + `node.imageIndex` / `node.overIndex` 决定当前高亮或被选中的图片，再利用前文的 Mock Node 技巧打开 MaskEditor，或者自定义 Lightbox 对话框。  
5.  **注意动画与视频节点的区分**：  
    *   若你的自定义节点输出动图或视频，务必正确设置 `animated` 标记或让 `isVideoNode(node)` 能识别你的节点类型，以便 `updatePreviews` 自动走 `useNodeVideo`/`useNodeAnimatedImage` 路径。

通过理解这一整套 Preview Image 机制，可以在不修改 ComfyUI 核心代码的前提下，为自定义节点、Subgraph、GroupNode 以及外部扩展可靠地接入图片预览、动图以及视频预览功能，并与 MaskEditor 的路径解析规则保持完全一致。
