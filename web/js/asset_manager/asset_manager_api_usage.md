# ComfyUI `A_my_nodes` 资产管理 API 调用指南

本文档只说明当前项目源码中已经实现并注册的真实行为，不写泛化能力，也不假设后端存在额外的字段校验或自动适配逻辑。

适用范围：

- `custom_nodes/A_my_nodes/routes.py`
- `web/js/asset_manager/asset_manager_ui.js`
- `web/js/asset_manager/asset_manager_preview_handler.js`

## 1. 实际已注册接口

当前项目在 `routes.py` 中注册了以下资产管理接口：

| 接口 | 方法 | 实际作用 |
| :--- | :--- | :--- |
| `/a_my_nodes/assets/prompts` | `GET` | 读取 `prompts_db.json` |
| `/a_my_nodes/assets/prompts` | `POST` | 原样写回 `prompts_db.json` |
| `/a_my_nodes/assets/models` | `GET` | 读取 `models_db.json` |
| `/a_my_nodes/assets/models` | `POST` | 原样写回 `models_db.json` |
| `/a_my_nodes/assets/view_preview` | `GET` | 按 `path` 或 `fallback_lora` 返回预览图文件 |
| `/a_my_nodes/assets/upload_preview` | `POST` | 把上传图片保存到插件 `previews` 目录 |
| `/a_my_nodes/assets/register_local_preview` | `POST` | 把本地路径转换为 `models://...` 或 `previews://...` |
| `/a_my_nodes/assets/check_model_exists` | `GET` | 检查单个模型是否存在，仅查 `loras` / `checkpoints` |
| `/a_my_nodes/assets/check_models_exist` | `POST` | 批量检查模型是否存在，仅查 `loras` / `checkpoints` |
| `/a_my_nodes/assets/search_pinyin` | `POST` | 原文包含匹配 + 拼音 / 首字母匹配 |

所有接口都挂载在 ComfyUI 服务地址下。例如：

```text
http://127.0.0.1:8188/a_my_nodes/assets/prompts
```

## 2. 数据文件与前端约束

### 2.1 后端真实行为

`/prompts` 和 `/models` 这两组接口在后端只是“读写 JSON 文件”：

- `GET`：直接读取 JSON 文件并返回。
- `POST`：直接把收到的 JSON 原样写回文件。

后端当前**没有**做以下事情：

- 不校验字段 schema。
- 不补默认字段。
- 不检查 `items` 内部结构。
- 不自动对齐提示词分组和模型组分组。

### 2.2 前端真实依赖

虽然服务端不校验结构，但 `asset_manager_ui.js` 有明确前端约束：

- 顶层必须是对象。
- 前端读取后要求存在 `groups` 字段；如果没有，会补成空数组。

因此，如果你要与当前内置 UI 兼容，建议始终按下面的顶层结构保存：

```json
{
  "groups": []
}
```

### 2.3 分组索引对齐是当前项目的重要约定

这个项目里，提示词分组和模型组分组不是完全独立的两套概念。前端在重命名分组时，会按同一个索引同步修改两边的组名。

也就是说：

- `promptsData.groups[0]`
- `modelsData.groups[0]`

在 UI 设计上被视为“同一个逻辑分组”的两侧数据。

另外，删除分组按钮当前不是“删除整个分组对象”，而是只把该分组的 `items` 清空，保留分组位置，目的也是尽量不破坏两边的索引对应关系。

如果外部程序只修改其中一边的组顺序、组数量或组名，当前内置资产管理界面可能出现错位或同步异常。

### 2.4 `prompts_db.json` 实际结构样例

下面的结构样例按当前项目真实文件组织方式整理，可直接作为外部读取时的参考：

```json
{
  "groups": [
    {
      "name": "flux-klein",
      "items": [
        {
          "id": "17753937599887009",
          "title": "侧面口交",
          "content": "Make the woman cheeking a gigantic penis, the woman hand close to the camera grasping the penis.",
          "preview_image": "previews://demo.png"
        },
        {
          "id": "1775402793899",
          "title": "保持人物id",
          "content": "",
          "preview_image": ""
        }
      ]
    },
    {
      "name": "通用提示词",
      "items": [
        {
          "id": "1775402798299",
          "title": "替换背景",
          "content": "",
          "preview_image": ""
        }
      ]
    }
  ]
}
```

字段含义：

- `groups[].name`：提示词分组标题
- `groups[].items[].title`：提示词模板标题
- `groups[].items[].content`：提示词正文
- `groups[].items[].preview_image`：预览图路径

补充说明：

- 当前你的真实文件里存在个别 `undefined` 字段，这不是标准结构，属于历史遗留脏字段，不应作为外部接入依赖。

### 2.5 `models_db.json` 实际结构样例

下面的结构样例按当前项目真实文件组织方式整理：

```json
{
  "groups": [
    {
      "name": "flux-klein",
      "items": [
        {
          "id": "17753965532519854",
          "keyword": "精液;cum",
          "check_mode": "contains",
          "high_loras": [
            {
              "on": true,
              "lora": "flux_klein\\精液\\PornMaster_cum_flux-2-klein-9b_V1.safetensors",
              "strength": 0.8
            }
          ],
          "low_loras": [],
          "preview_image": "H:\\models\\loras\\flux_klein\\精液\\c4msh0t_.png"
        },
        {
          "id": "17754005614061290",
          "keyword": "拳交",
          "check_mode": "contains",
          "high_loras": [
            {
              "on": true,
              "lora": "flux_klein\\拳交\\SelfFisting_Vaginal_v1.safetensors",
              "strength": 0.75
            }
          ],
          "low_loras": [],
          "preview_image": ""
        }
      ]
    },
    {
      "name": "qwen-edit",
      "items": [
        {
          "id": "17754043086225632",
          "keyword": "木乃伊式捆绑",
          "check_mode": "contains",
          "high_loras": [
            {
              "on": true,
              "lora": "Qwen\\口球sm\\木乃伊式捆绑\\qwen_encased mummification_V1E3.safetensors",
              "strength": 1
            }
          ],
          "low_loras": [],
          "preview_image": ""
        }
      ]
    }
  ]
}
```

字段含义：

- `groups[].name`：模型组标题
- `groups[].items[].keyword`：模型条目标题，也是当前项目里用于显示和插入输入框的字段
- `groups[].items[].check_mode`：当前实际常见值为 `contains`
- `groups[].items[].high_loras` / `low_loras`：模型流配置
- `high_loras[].lora`：模型路径
- `high_loras[].strength`：模型权重
- `high_loras[].on`：是否启用

### 2.6 外部 Web 最小字段映射

如果你的外部 Web 只是想读取“标题文本”并插入某个输入框，当前项目应按下面映射读取：

| 目的 | 实际字段 |
| :--- | :--- |
| 提示词分组标题 | `prompts.groups[].name` |
| 提示词模板标题 | `prompts.groups[].items[].title` |
| 模型组标题 | `models.groups[].name` |
| 模型条目标题 | `models.groups[].items[].keyword` |

这里最容易混淆的一点是：

- 提示词条目用的是 `title`
- 模型条目用的是 `keyword`

当前项目的快捷插入菜单点击写入输入框时，实际也是按这个规则取值，而不是统一取 `title`。

## 3. `/a_my_nodes/assets/prompts`

### 3.1 GET：读取提示词库

请求：

```http
GET http://127.0.0.1:8188/a_my_nodes/assets/prompts
```

真实行为：

- 如果 `prompts_db.json` 存在且能正常解析，直接返回文件内容。
- 如果文件不存在或读取失败，返回：

```json
{
  "groups": []
}
```

示例：

```json
{
  "groups": [
    {
      "name": "人物动作",
      "items": [
        {
          "title": "奔跑",
          "content": "running, dynamic pose, 1girl, high quality;",
          "preview_image": "previews://demo.png"
        }
      ]
    }
  ]
}
```

### 3.2 POST：保存提示词库

请求：

```http
POST http://127.0.0.1:8188/a_my_nodes/assets/prompts
Content-Type: application/json
```

真实行为：

- 后端把请求体 JSON 原样写入 `prompts_db.json`。
- 成功时返回：

```json
{
  "success": true
}
```

- 失败时返回：

```json
{
  "success": false,
  "error": "错误信息"
}
```

注意：

- 这是整体覆盖保存，不是局部更新。
- 如果你只传某个片段，原文件其余内容会被覆盖掉。

## 4. `/a_my_nodes/assets/models`

### 4.1 GET：读取模型组

请求：

```http
GET http://127.0.0.1:8188/a_my_nodes/assets/models
```

真实行为：

- 如果 `models_db.json` 存在且能正常解析，直接返回文件内容。
- 如果文件不存在或读取失败，返回：

```json
{
  "groups": []
}
```

示例：

```json
{
  "groups": [
    {
      "name": "动漫风格",
      "items": [
        {
          "keyword": "anime_v1",
          "check_mode": "contains",
          "high_loras": [
            { "lora": "anime_lora_v1.safetensors", "strength": 1.0, "on": true }
          ],
          "low_loras": [
            { "lora": "detail_enhancer.safetensors", "strength": 0.5, "on": true }
          ]
        }
      ]
    }
  ]
}
```

### 4.2 POST：保存模型组

请求：

```http
POST http://127.0.0.1:8188/a_my_nodes/assets/models
Content-Type: application/json
```

真实行为：

- 后端把请求体 JSON 原样写入 `models_db.json`。
- 成功时返回：

```json
{
  "success": true
}
```

- 失败时返回：

```json
{
  "success": false,
  "error": "错误信息"
}
```

注意：

- 这同样是整体覆盖保存。
- 如果你希望继续兼容内置 UI，建议保持 `models.groups` 与 `prompts.groups` 的索引对应关系。

## 5. 预览图相关接口

当前项目里，预览图不是只靠一个上传接口完成，而是分成三类职责：

1. `view_preview`：读取预览图。
2. `register_local_preview`：处理本地路径，优先转成 `models://...` 或复制成 `previews://...`。
3. `upload_preview`：无法得到本地绝对路径时的兜底上传。

### 5.1 `/a_my_nodes/assets/view_preview`

请求方式一：

```http
GET /a_my_nodes/assets/view_preview?path=<path_uri>
```

请求方式二：

```http
GET /a_my_nodes/assets/view_preview?fallback_lora=<lora_name>
```

#### `path` 的真实处理逻辑

当传入 `path` 时，后端按下面顺序处理：

1. 如果以 `models://` 开头，就映射到 `folder_paths.models_dir`。
2. 如果以 `previews://` 开头，就映射到插件内部的 `previews` 目录。
3. 否则把它当作普通路径直接使用。

如果该路径对应的文件存在，就直接返回文件。

#### 针对错误绝对路径的容错

如果传入的是一个绝对路径，但这个路径在当前机器上不存在，后端还会尝试容错修复：

- 查找路径中是否包含 `\loras\` 或 `/loras/`
- 如果找到，就截取后面的相对部分
- 再拼回当前 ComfyUI 的 `models/loras` 目录下检查

因此，像下面这类路径即使原始盘符不一致，也有机会被修正：

```text
H:\models\loras\demo\cover.png
```

#### `fallback_lora` 的真实处理逻辑

当 `path` 没命中，且传了 `fallback_lora` 时，后端会：

1. 用 `folder_paths.get_full_path("loras", fallback_lora)` 找到对应 LoRA 文件。
2. 进入该 LoRA 所在目录。
3. 查找“文件名以该 LoRA 基础名开头，且扩展名为 `.png`”的图片文件。
4. 找到后直接返回该 PNG。

这里不是模糊搜索，也不是通用关联规则，只是“同目录 + 同名前缀 PNG”。

#### 返回结果

- 成功时：直接返回图片文件内容。
- 失败时：返回 `404 Preview not found`。

### 5.2 `/a_my_nodes/assets/register_local_preview`

请求：

```http
POST /a_my_nodes/assets/register_local_preview
Content-Type: application/json
```

请求体：

```json
{
  "path": "H:/models/loras/demo/cover.png"
}
```

这个接口的真实职责是“注册本地路径”，不是简单上传。

#### 实际处理流程

后端会先判断这个路径是否能映射到当前 ComfyUI 的 `models_dir`：

1. 直接检查真实路径或原始路径是否位于 `models_dir` 内。
2. 如果不在，则尝试从路径中识别已知模型子目录，再拼接回当前 `models_dir`。

当前硬编码支持识别的模型子目录只有：

- `loras`
- `checkpoints`
- `embeddings`
- `controlnet`
- `unet`
- `vae`

#### 返回结果 1：引用 models 内文件

如果路径最终能映射到 `models_dir` 内，返回：

```json
{
  "success": true,
  "uri": "models://loras/demo/cover.png",
  "action": "referenced"
}
```

#### 返回结果 2：复制到 previews 目录

如果路径存在，但不属于可映射的 `models_dir`，后端会把文件复制到插件 `previews` 目录，返回：

```json
{
  "success": true,
  "uri": "previews://1d23ab45_cover.png",
  "action": "copied"
}
```

#### 返回结果 3：路径无效

```json
{
  "success": false,
  "error": "文件不存在或无法访问: ..."
}
```

### 5.3 `/a_my_nodes/assets/upload_preview`

请求：

```http
POST /a_my_nodes/assets/upload_preview
Content-Type: multipart/form-data
```

#### 这个接口的真实限制

当前后端实现只读取 `multipart` 的第一个字段，并且要求：

- 第一个字段名必须是 `image`

否则就会走失败分支。

前端虽然会额外附带 `original_name`，但当前后端实现**没有使用它**。

#### 成功返回

```json
{
  "success": true,
  "uri": "previews://8b56f4d2_demo.png"
}
```

#### 失败返回

```json
{
  "success": false,
  "error": "Upload failed"
}
```

#### 在当前项目里的实际定位

根据 `asset_manager_preview_handler.js`，这个接口是兜底方案：

1. 能拿到本地绝对路径时，优先走 `register_local_preview`
2. 只有拿不到路径、只拿到纯 `File/Blob` 时，才退回 `upload_preview`

## 6. 模型存在性检查接口

### 6.1 `/a_my_nodes/assets/check_model_exists`

请求：

```http
GET /a_my_nodes/assets/check_model_exists?path=anime_lora_v1.safetensors
```

真实行为：

- 若 `path` 为空或为 `"None"`，直接返回 `{"exists": false}`
- 否则：
  1. 先按 `loras` 类型查
  2. 查不到再按 `checkpoints` 类型查

成功返回示例：

```json
{
  "exists": true
}
```

注意：

- 当前实现**只查 `loras` 和 `checkpoints`**
- 不会查 `vae`、`controlnet`、`unet` 等其他模型目录

### 6.2 `/a_my_nodes/assets/check_models_exist`

请求：

```http
POST /a_my_nodes/assets/check_models_exist
Content-Type: application/json
```

请求体：

```json
{
  "paths": [
    "anime_lora_v1.safetensors",
    "detail_enhancer.safetensors",
    "missing_model.safetensors"
  ]
}
```

真实行为：

- 遍历 `paths`
- 对每个路径依次按 `loras`、`checkpoints` 查找
- 返回映射表

返回示例：

```json
{
  "results": {
    "anime_lora_v1.safetensors": true,
    "detail_enhancer.safetensors": true,
    "missing_model.safetensors": false
  }
}
```

## 7. `/a_my_nodes/assets/search_pinyin`

请求：

```http
POST /a_my_nodes/assets/search_pinyin
Content-Type: application/json
```

请求体：

```json
{
  "texts": ["女孩", "奔跑", "夜晚星空", "Night Sky"],
  "keyword": "yw"
}
```

### 真实匹配顺序

这个接口不是“纯拼音搜索”，而是按下面顺序判断：

1. 如果 `keyword` 为空，直接返回全 `true`
2. 先做原文小写包含匹配
3. 再做拼音全拼匹配
4. 再做首字母匹配

所以它实际是：

- 原文包含搜索
- 拼音包含搜索
- 首字母包含搜索

三者的组合。

返回示例：

```json
{
  "matches": [false, false, true, false]
}
```

说明：

- 返回数组长度与 `texts` 一致
- 每个位置的布尔值对应原数组同索引项是否命中

## 8. 前端当前真实调用方式

这一节只总结当前项目中前端是怎么调用这些接口的，便于外部项目对齐行为。

### 8.1 资产主界面

`asset_manager_ui.js` 中的真实调用方式：

- 打开界面时，同时请求：
  - `/a_my_nodes/assets/prompts`
  - `/a_my_nodes/assets/models`
- 保存当前页签时：
  - `prompts` 页签保存到 `/a_my_nodes/assets/prompts`
  - `models` 页签保存到 `/a_my_nodes/assets/models`

### 8.2 组名同步

当前前端在重命名组名时，会同时更新：

- `promptsData.groups[index].name`
- `modelsData.groups[index].name`

并分别保存当前页和另一侧数据。

这说明当前 UI 把两边同索引的组视为一对。

### 8.3 预览图处理优先级

`asset_manager_preview_handler.js` 中的真实处理优先级是：

1. 文本里如果像绝对路径、`models://`、`previews://`，先走 `register_local_preview`
2. `File` 对象里如果能探测到绝对路径，也先走 `register_local_preview`
3. 只有拿不到可注册路径时，才走 `upload_preview`

## 9. 建议的外部接入方式

如果你希望外部程序尽量和当前项目行为保持一致，建议按下面方式接入：

1. 读取与保存 `prompts` / `models` 时，始终保持顶层为 `{ "groups": [...] }`
2. 如果还要兼容内置 UI，尽量保持 `prompts.groups` 与 `models.groups` 的索引对齐
3. 处理预览图时：
   - 能拿到本地绝对路径，优先调 `register_local_preview`
   - 拿不到路径，只拿到图片文件，再调 `upload_preview`
4. 前端展示预览图时，不要直接把 `models://...` 或 `previews://...` 当浏览器地址用，而是交给 `view_preview`
5. 检查模型存在性时，要知道当前接口只覆盖 `loras` / `checkpoints`

## 10. 源码对应关系

### 10.1 后端

- `routes.py`
  - 实现所有资产管理 HTTP 接口
  - 定义 `previews` 目录与智能 URI 解析逻辑
  - 实现模型检查与拼音搜索

### 10.2 前端

- `web/js/asset_manager/asset_manager_ui.js`
  - 负责主界面的读取、保存、组名同步、模型检查与预览展示
- `web/js/asset_manager/asset_manager_preview_handler.js`
  - 负责本地路径注册、剪贴板路径解析、兜底图片上传

## 11. 总结

当前项目里的资产管理 API 本质上是一套“轻量级 JSON 存储 + 预览图解析 / 注册 + 模型存在性检查 + 拼音搜索”接口。

最关键的项目级事实有三点：

1. `prompts` 和 `models` 后端都是原样读写，没有 schema 校验。
2. 内置前端实际上依赖 `groups` 顶层结构，并把两侧分组按索引配对使用。
3. 预览图处理优先走 `register_local_preview`，`upload_preview` 只是拿不到路径时的兜底方案。
