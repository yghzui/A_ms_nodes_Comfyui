import os
import json
import io
from aiohttp import web
from PIL import Image
from server import PromptServer

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass
from folder_paths import get_output_directory
from .nodes.resolutionpreset import (
    RESOLUTION_PRESET_LIMITS,
    find_default_custom_preset_name,
    normalize_preset_name,
    normalize_resolution_value,
    save_resolution_presets_to_file,
    serialize_builtin_resolution_presets,
    sanitize_resolution_presets,
    load_resolution_presets_from_file,
)
from .nodes.workflow_group_runtime import apply_workflow_groups_to_prompt_payload
from .nodes.workflow_group_store import (
    get_workflow_groups_schema as build_workflow_groups_schema,
    load_workflow_groups_db,
    save_workflow_groups_db,
)

# 全局日志打印控制开关
ENABLE_DEBUG_PRINT = False

def printf(*args, force=False, **kwargs):
    """统一管理的打印函数，受 ENABLE_DEBUG_PRINT 控制"""
    if ENABLE_DEBUG_PRINT or force:
        print(*args, **kwargs)

# 全局标志，用于防止重复注册
_routes_registered = False
_workflow_group_prompt_handler_registered = False


async def get_resolution_presets(request):
    custom_presets = load_resolution_presets_from_file()
    return web.json_response(_build_resolution_presets_payload(custom_presets))


async def save_resolution_presets(request):
    try:
        body = await request.json()
    except Exception as e:
        print(f"❌ [ResolutionPreset] 解析请求JSON失败: {e}")
        return web.Response(
            status=400,
            text=json.dumps({"error": "invalid json"}),
            content_type="application/json",
        )
    action = str(body.get("action", "") or "").strip().lower()
    existing = load_resolution_presets_from_file()
    try:
        if not action:
            if "presets" in body:
                action = "replace"
            elif "name" in body and "w" in body and "h" in body:
                action = "upsert"
            else:
                action = "replace"

        if action == "replace":
            presets = body.get("custom_presets", body.get("presets"))
            if not isinstance(presets, dict):
                raise ValueError("custom_presets must be an object")
            result = sanitize_resolution_presets(presets)
        elif action == "upsert":
            name = normalize_preset_name(body.get("name"))
            if not name:
                raise ValueError("name is required")
            if name in serialize_builtin_resolution_presets():
                raise ValueError("builtin preset name cannot be overwritten")
            step = body.get("step", RESOLUTION_PRESET_LIMITS["default_step"])
            width = normalize_resolution_value(body.get("w"), step=step, field_name="width")
            height = normalize_resolution_value(body.get("h"), step=step, field_name="height")
            choose = bool(body.get("choose", False))
            result = dict(existing)
            result[name] = {
                "w": width,
                "h": height,
                "choose": choose,
            }
            if choose:
                for preset_name, item in result.items():
                    item["choose"] = preset_name == name
            result = sanitize_resolution_presets(result)
        elif action == "delete":
            name = normalize_preset_name(body.get("name"))
            if not name:
                raise ValueError("name is required")
            if name in serialize_builtin_resolution_presets():
                raise ValueError("builtin preset cannot be deleted")
            if name not in existing:
                raise ValueError("preset does not exist")
            result = dict(existing)
            del result[name]
        elif action == "set_default":
            name = normalize_preset_name(body.get("name"))
            if not name:
                raise ValueError("name is required")
            if name not in existing:
                raise ValueError("preset does not exist")
            result = dict(existing)
            for preset_name, item in result.items():
                item["choose"] = preset_name == name
            result = sanitize_resolution_presets(result)
        else:
            raise ValueError(f"unsupported action: {action}")
    except ValueError as e:
        print(f"❌ [ResolutionPreset] 请求参数错误: {e}")
        return web.Response(
            status=400,
            text=json.dumps({"error": str(e)}),
            content_type="application/json",
        )

    try:
        _save_resolution_presets_to_file(result)
        printf(f"✅ [ResolutionPreset] 预设保存成功")
    except Exception as e:
        print(f"❌ [ResolutionPreset] 保存文件失败: {e}")
        return web.Response(
            status=500,
            text=json.dumps({"error": str(e)}),
            content_type="application/json",
        )
    payload = _build_resolution_presets_payload(result)
    payload.update({
        "success": True,
        "action": action,
    })
    return web.json_response(payload)


def _build_resolution_presets_payload(custom_presets):
    sanitized = sanitize_resolution_presets(custom_presets)
    return {
        "builtin_presets": serialize_builtin_resolution_presets(),
        "custom_presets": sanitized,
        "constraints": dict(RESOLUTION_PRESET_LIMITS),
        "default_custom_preset": find_default_custom_preset_name(sanitized),
    }


def _save_resolution_presets_to_file(presets):
    printf("💾 [ResolutionPreset] 正在保存预设文件")
    save_resolution_presets_to_file(presets)


async def get_workflow_groups(request):
    return web.json_response(load_workflow_groups_db())


async def save_workflow_groups(request):
    try:
        body = await request.json()
    except Exception as e:
        return web.Response(
            status=400,
            text=json.dumps({"error": f"invalid json: {e}"}),
            content_type="application/json",
        )

    try:
        saved = save_workflow_groups_db(body)
        return web.json_response({"success": True, "data": saved})
    except Exception as e:
        return web.Response(
            status=500,
            text=json.dumps({"error": str(e)}),
            content_type="application/json",
        )


async def get_workflow_groups_schema(request):
    return web.json_response(build_workflow_groups_schema())


def register_workflow_group_prompt_handler():
    global _workflow_group_prompt_handler_registered

    if _workflow_group_prompt_handler_registered:
        return

    if hasattr(PromptServer, "instance") and PromptServer.instance is not None:
        PromptServer.instance.add_on_prompt_handler(apply_workflow_groups_to_prompt_payload)
        _workflow_group_prompt_handler_registered = True
        print("✅ 工作流切换组 on_prompt 处理器注册成功！")

async def serve_output_file(request):
    """处理静态输出文件请求 - 提供实际的文件服务功能"""
    path = request.match_info["path"]
    output_dir = get_output_directory()
    full_path = os.path.normpath(os.path.join(output_dir, path))

    # 安全性检查：防止目录穿越
    if not full_path.startswith(output_dir):
        return web.Response(status=403, text="Forbidden")

    if not os.path.isfile(full_path):
        return web.Response(status=404, text="File not found")

    return web.FileResponse(full_path)

async def view_input_file(request):
    """安全地查看输入文件，处理 HEIC 转换以供浏览器显示"""
    filename = request.query.get("filename")
    subfolder = request.query.get("subfolder", "")
    printf(f"🔍 [view_input_file] 请求预览: {filename}, 子文件夹: {subfolder}")

    if not filename:
        return web.Response(status=400, text="Missing filename")
    
    input_dir = folder_paths.get_input_directory()
    
    # 构建完整路径
    if subfolder:
        full_path = os.path.join(input_dir, subfolder, filename)
    else:
        full_path = os.path.join(input_dir, filename)
    
    full_path = os.path.normpath(full_path)
    printf(f"🔍 [view_input_file] 完整物理路径: {full_path}")
    
    # 安全检查
    if not full_path.startswith(os.path.normpath(input_dir)):
        print(f"❌ [view_input_file] 安全检查失败，禁止访问: {full_path}")
        return web.Response(status=403, text="Forbidden")
        
    if not os.path.isfile(full_path):
        print(f"❌ [view_input_file] 文件不存在: {full_path}")
        return web.Response(status=404, text="File not found")
    
    # 检测是否是 HEIC (即使后缀是 .png)
    is_heic = False
    try:
        with open(full_path, 'rb') as f:
            header = f.read(16)
            if b'ftypheic' in header or b'ftypmif1' in header:
                is_heic = True
                printf(f"ℹ️ [view_input_file] 检测到 HEIC 格式头: {filename}")
    except Exception as e:
        printf(f"⚠️ [view_input_file] 读取文件头失败: {e}")

    if is_heic:
        try:
            # 浏览器通常不支持 HEIC，所以我们在服务端转成 PNG 发送
            printf(f"🔄 [view_input_file] 正在将 HEIC 转换为 PNG 以供浏览器预览...")
            img = Image.open(full_path)
            output = io.BytesIO()
            img.save(output, format="PNG")
            printf(f"✅ [view_input_file] 转换成功: {filename}")
            return web.Response(body=output.getvalue(), content_type="image/png")
        except Exception as e:
            print(f"❌ [view_input_file] HEIC 转换失败: {e}")
            return web.FileResponse(full_path)
    else:
        return web.FileResponse(full_path)

# 删除文件API处理函数
async def delete_output_file(request):
    """删除output目录内的文件"""
    try:
        # 获取请求体数据
        data = await request.json()
        relative_path = data.get('path')
        
        if not relative_path:
            return web.Response(
                status=400, 
                text=json.dumps({"error": "缺少path参数"}), 
                content_type='application/json'
            )
        
        output_dir = get_output_directory()
        full_path = os.path.normpath(os.path.join(output_dir, relative_path))
        
        # 安全性检查：防止目录穿越
        if not full_path.startswith(output_dir):
            return web.Response(
                status=403, 
                text=json.dumps({"error": "禁止访问该路径"}), 
                content_type='application/json'
            )
        
        # 检查文件是否存在
        if not os.path.isfile(full_path):
            return web.Response(
                status=404, 
                text=json.dumps({"error": "文件不存在"}), 
                content_type='application/json'
            )
        
        # 删除文件
        os.remove(full_path)
        
        printf(f"✅ 成功删除文件: {relative_path}")
        
        return web.Response(
            status=200,
            text=json.dumps({
                "success": True,
                "message": f"文件 {relative_path} 删除成功",
                "deleted_path": relative_path
            }),
            content_type='application/json'
        )
        
    except PermissionError:
        return web.Response(
            status=403,
            text=json.dumps({"error": "没有权限删除该文件"}),
            content_type='application/json'
        )
    except OSError as e:
        return web.Response(
            status=500,
            text=json.dumps({"error": f"删除文件失败: {str(e)}"}),
            content_type='application/json'
        )
    except Exception as e:
        return web.Response(
            status=500,
            text=json.dumps({"error": f"服务器内部错误: {str(e)}"}),
            content_type='application/json'
        )

import uuid
import shutil
import folder_paths

# ================= 资产管理系统 (Asset Manager) API =================
ASSETS_DIR = os.path.dirname(os.path.abspath(__file__))
PREVIEWS_DIR = os.path.join(ASSETS_DIR, "previews")
os.makedirs(PREVIEWS_DIR, exist_ok=True)

def get_asset_db_path(db_name):
    return os.path.join(ASSETS_DIR, f"{db_name}.json")

async def get_asset_prompts(request):
    db_path = get_asset_db_path("prompts_db")
    if os.path.exists(db_path):
        try:
            with open(db_path, "r", encoding="utf-8") as f:
                return web.json_response(json.load(f))
        except Exception as e:
            print(f"❌ [AssetManager] 读取提示词库失败: {e}")
    return web.json_response({"groups": []})

async def save_asset_prompts(request):
    db_path = get_asset_db_path("prompts_db")
    try:
        data = await request.json()
        with open(db_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return web.json_response({"success": True})
    except Exception as e:
        print(f"❌ [AssetManager] 保存提示词库失败: {e}")
        return web.json_response({"success": False, "error": str(e)})

async def get_asset_models(request):
    db_path = get_asset_db_path("models_db")
    if os.path.exists(db_path):
        try:
            with open(db_path, "r", encoding="utf-8") as f:
                return web.json_response(json.load(f))
        except Exception as e:
            print(f"❌ [AssetManager] 读取模型库失败: {e}")
    return web.json_response({"groups": []})

async def save_asset_models(request):
    db_path = get_asset_db_path("models_db")
    try:
        data = await request.json()
        with open(db_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return web.json_response({"success": True})
    except Exception as e:
        print(f"❌ [AssetManager] 保存模型库失败: {e}")
        return web.json_response({"success": False, "error": str(e)})

async def view_asset_preview(request):
    """根据智能路径协议返回预览图"""
    path_uri = request.query.get("path", "")
    fallback_lora = request.query.get("fallback_lora", "")
    
    real_path = ""
    try:
        # 处理 path_uri
        if path_uri:
            if path_uri.startswith("models://"):
                rel_path = path_uri[len("models://"):]
                real_path = os.path.join(folder_paths.models_dir, os.path.normpath(rel_path))
            elif path_uri.startswith("previews://"):
                rel_path = path_uri[len("previews://"):]
                real_path = os.path.join(PREVIEWS_DIR, os.path.normpath(rel_path))
            else:
                real_path = path_uri
                
            if os.path.exists(real_path) and os.path.isfile(real_path):
                return web.FileResponse(real_path)
            
            # 如果绝对路径不存在，尝试提取 models 相对路径
            # 兼容用户输入类似 H:\models\loras\... 或 H:\model\loras\... 的错误路径
            lower_path = real_path.lower()
            idx = lower_path.find("\\loras\\")
            if idx == -1:
                idx = lower_path.find("/loras/")
            if idx != -1:
                rel_path = real_path[idx + 7:] # skip \loras\
                alt_path = os.path.join(folder_paths.models_dir, "loras", rel_path)
                if os.path.exists(alt_path) and os.path.isfile(alt_path):
                    return web.FileResponse(alt_path)
                        
        # 如果 path_uri 没找到，或者为空，检查 fallback_lora
        if fallback_lora:
            lora_path = folder_paths.get_full_path("loras", fallback_lora)
            if lora_path and os.path.exists(lora_path):
                lora_dir = os.path.dirname(lora_path)
                lora_base = os.path.splitext(os.path.basename(lora_path))[0]
                
                # 查找同级目录下以模型名开头的 png 文件
                for file in os.listdir(lora_dir):
                    if file.lower().endswith(".png") and file.startswith(lora_base):
                        fallback_path = os.path.join(lora_dir, file)
                        if os.path.exists(fallback_path) and os.path.isfile(fallback_path):
                            return web.FileResponse(fallback_path)

    except Exception as e:
        print(f"❌ [AssetManager] 预览图读取异常: {e}")
        
    return web.Response(status=404, text="Preview not found")

async def upload_asset_preview(request):
    """处理纯外部图片的上传，保存到 previews 目录"""
    try:
        reader = await request.multipart()
        field = await reader.next()
        
        # 为了兼容可能在后续 chunk 里出现的 original_name 字段，我们先用默认名
        original_name = ""
        
        # 很多时候 multipart 的顺序是不确定的，这里为了简化，我们先直接检查是不是 image
        if field.name == 'image':
            filename = str(uuid.uuid4()) + "_" + (field.filename or "uploaded.png")
            file_path = os.path.join(PREVIEWS_DIR, filename)
            with open(file_path, 'wb') as f:
                while True:
                    chunk = await field.read_chunk()
                    if not chunk: break
                    f.write(chunk)
            return web.json_response({"success": True, "uri": f"previews://{filename}"})
    except Exception as e:
        print(f"❌ [AssetManager] 图片上传失败: {e}")
        return web.json_response({"success": False, "error": str(e)})
    return web.json_response({"success": False, "error": "Upload failed"})

async def register_local_preview(request):
    """智能路径分析：判断本地绝对路径是否在 models 目录下，支持软连接(symlink)及异盘映射，决定是零拷贝引用还是复制"""
    try:
        data = await request.json()
        local_path = data.get("path", "")
        
        models_root = os.path.abspath(folder_paths.models_dir)
        real_models_root = os.path.realpath(models_root)
        
        # 辅助函数：判断目标绝对路径是否能映射到 models_dir
        def get_models_uri_if_valid(test_path):
            if not os.path.exists(test_path):
                return None
            abs_p = os.path.abspath(test_path)
            real_p = os.path.realpath(abs_p)
            
            # 1. 真实路径在 models 真实路径内 (软连接解析后匹配)
            if real_p.startswith(real_models_root):
                rel = os.path.relpath(real_p, real_models_root)
                return f"models://{rel.replace(os.sep, '/')}"
            # 2. 原始路径在 models 目录内 (防某些特殊挂载)
            if abs_p.startswith(models_root):
                rel = os.path.relpath(abs_p, models_root)
                return f"models://{rel.replace(os.sep, '/')}"
            return None

        # 尝试 1: 直接使用拖拽传过来的绝对路径进行智能校验
        uri = get_models_uri_if_valid(local_path)
        if uri:
            return web.json_response({"success": True, "uri": uri, "action": "referenced"})
            
        # 尝试 2: 容错处理。用户可能拖拽的是 H:\models\loras\...
        # 即使它和 ComfyUI 不在一个盘符且不是标准软连接，只要路径包含已知模型子目录，我们就尝试把它拼接到当前 ComfyUI 的 models 目录下检查
        lower_path = local_path.lower()
        # 常见模型目录名
        for sub in ["loras", "checkpoints", "embeddings", "controlnet", "unet", "vae"]:
            for sep in ["\\", "/"]:
                keyword = f"{sep}{sub}{sep}"
                idx = lower_path.find(keyword)
                if idx != -1:
                    # 提取相对部分，比如 loras\flux_klein\口交\FK_giantbbcoral_123864001.png
                    rel_path = local_path[idx + 1:] 
                    alt_path = os.path.join(models_root, rel_path)
                    
                    uri = get_models_uri_if_valid(alt_path)
                    if uri:
                        return web.json_response({"success": True, "uri": uri, "action": "referenced"})

        # 如果前两步都没匹配上，说明这真的只是一个外部文件（比如桌面的一张截图）
        if os.path.exists(local_path):
            filename = str(uuid.uuid4()) + "_" + os.path.basename(local_path)
            dest_path = os.path.join(PREVIEWS_DIR, filename)
            shutil.copy2(local_path, dest_path)
            return web.json_response({"success": True, "uri": f"previews://{filename}", "action": "copied"})
        else:
            return web.json_response({"success": False, "error": f"文件不存在或无法访问: {local_path}"})
            
    except Exception as e:
        print(f"❌ [AssetManager] 智能路径注册失败: {e}")
        return web.json_response({"success": False, "error": str(e)})

async def check_model_exists(request):
    """接收模型相对路径（如 loras/flux/xxx.safetensors），检查是否存在"""
    try:
        model_path = request.rel_url.query.get("path", "")
        if not model_path or model_path == "None":
            return web.json_response({"exists": False})
            
        # ComfyUI 的 folder_paths.get_full_path_or_raise 可能会抛出异常
        # 这里我们用 get_full_path 安全地获取
        import folder_paths
        
        # 尝试猜测类型，默认从 loras 找，如果找不到也可以尝试 checkpoints 等
        # 因为资产管理器主要存的是 lora
        full_path = folder_paths.get_full_path("loras", model_path)
        
        if full_path and os.path.exists(full_path):
            return web.json_response({"exists": True})
            
        # 如果作为 lora 找不到，再作为 checkpoints 找
        full_path_ckpt = folder_paths.get_full_path("checkpoints", model_path)
        if full_path_ckpt and os.path.exists(full_path_ckpt):
            return web.json_response({"exists": True})
            
        return web.json_response({"exists": False})
        
    except Exception as e:
        print(f"❌ [AssetManager] 检查模型存在失败: {e}")
        return web.json_response({"exists": False, "error": str(e)})

async def check_models_exist(request):
    """批量接收模型路径，检查是否存在
    请求体: {"paths": ["loras/a.safetensors", "loras/b.safetensors"]}
    返回: {"results": {"loras/a.safetensors": True, "loras/b.safetensors": False}}
    """
    try:
        data = await request.json()
        paths = data.get("paths", [])
        results = {}
        
        import folder_paths
        import os
        
        for path in paths:
            if not path or path == "None":
                results[path] = False
                continue
                
            # 1. 查 loras
            full_path = folder_paths.get_full_path("loras", path)
            if full_path and os.path.exists(full_path):
                results[path] = True
                continue
                
            # 2. 查 checkpoints
            full_path_ckpt = folder_paths.get_full_path("checkpoints", path)
            if full_path_ckpt and os.path.exists(full_path_ckpt):
                results[path] = True
                continue
                
            results[path] = False
            
        return web.json_response({"results": results})
    except Exception as e:
        print(f"❌ [AssetManager] 批量检查模型存在失败: {e}")
        return web.json_response({"results": {}, "error": str(e)})

import functools
from pypinyin import lazy_pinyin, Style

@functools.lru_cache(maxsize=10000)
def get_pinyin_info(text):
    """
    获取字符串的全拼和首字母（带LRU缓存机制）
    返回: (全拼字符串, 首字母字符串)
    """
    if not text:
        return "", ""
    try:
        full_pinyin = "".join(lazy_pinyin(text))
        initials = "".join(lazy_pinyin(text, style=Style.FIRST_LETTER))
        return full_pinyin.lower(), initials.lower()
    except Exception as e:
        print(f"获取拼音信息失败: {e}")
        return "", ""

async def search_pinyin(request):
    """
    接收要搜索的文本（text）和关键词（keyword），返回是否匹配。
    """
    try:
        data = await request.json()
        texts = data.get("texts", [])
        search_text = data.get("keyword", "")
        
        if not search_text:
            return web.json_response({"matches": [True] * len(texts)})
            
        search_text = str(search_text).lower()
        results = []
        
        for text in texts:
            if not text:
                results.append(False)
                continue
                
            text_lower = str(text).lower()
            
            # 1. 优先匹配原文本
            if search_text in text_lower:
                results.append(True)
                continue
                
            # 2. 匹配拼音
            full_pinyin, initials = get_pinyin_info(str(text))
            if search_text in full_pinyin or search_text in initials:
                results.append(True)
            else:
                results.append(False)
                
        return web.json_response({"matches": results})
    except Exception as e:
        print(f"❌ [AssetManager] 拼音搜索失败: {e}")
        return web.json_response({"error": str(e), "matches": []})

# ====================================================================

# 路由注册函数 - 这个函数会在ComfyUI初始化时被调用
def register_routes():
    """注册自定义路由到PromptServer"""
    global _routes_registered
    
    # 防止重复注册
    if _routes_registered:
        print("⚠️ 路由已经注册过了，跳过重复注册")
        return
    
    try:
        # 确保PromptServer已经初始化
        if hasattr(PromptServer, 'instance') and PromptServer.instance is not None:
            # 检查路由是否已经存在
            existing_routes = [route.path for route in PromptServer.instance.routes]

            if "/static_output/{path:.*}" not in existing_routes:
                PromptServer.instance.routes.get("/static_output/{path:.*}")(
                    serve_output_file
                )
                print("✅ 静态文件服务路由 /static_output/{path:.*} 注册成功！")
            else:
                print("⚠️ 路由 /static_output/{path:.*} 已经存在，跳过注册")

            if "/delete_output_file" not in existing_routes:
                PromptServer.instance.routes.post("/delete_output_file")(
                    delete_output_file
                )
                print("✅ 删除文件API路由 /delete_output_file 注册成功！")
            else:
                print("⚠️ 路由 /delete_output_file 已经存在，跳过注册")

            if "/a_my_nodes/view_input" not in existing_routes:
                PromptServer.instance.routes.get("/a_my_nodes/view_input")(
                    view_input_file
                )
                print("✅ 视图服务路由 /a_my_nodes/view_input 注册成功！")
            else:
                print("⚠️ 路由 /a_my_nodes/view_input 已经存在，跳过注册")

            if "/a_my_nodes/resolution_presets" not in existing_routes:
                PromptServer.instance.routes.get("/a_my_nodes/resolution_presets")(
                    get_resolution_presets
                )
                PromptServer.instance.routes.post("/a_my_nodes/resolution_presets")(
                    save_resolution_presets
                )
                print("✅ 分辨率预设路由 /a_my_nodes/resolution_presets 注册成功！")
            else:
                print("⚠️ 路由 /a_my_nodes/resolution_presets 已经存在，跳过注册")

            if "/a_my_nodes/workflow_groups" not in existing_routes:
                PromptServer.instance.routes.get("/a_my_nodes/workflow_groups")(get_workflow_groups)
                PromptServer.instance.routes.post("/a_my_nodes/workflow_groups")(save_workflow_groups)
                PromptServer.instance.routes.get("/a_my_nodes/workflow_groups/schema")(get_workflow_groups_schema)
                print("✅ 工作流切换组路由 /a_my_nodes/workflow_groups 注册成功！")
            else:
                print("⚠️ 路由 /a_my_nodes/workflow_groups 已经存在，跳过注册")

            register_workflow_group_prompt_handler()

            if "/a_my_nodes/upload_custom_edited_image" not in existing_routes:
                PromptServer.instance.routes.post("/a_my_nodes/upload_custom_edited_image")(upload_custom_edited_image)
                print("✅ 自定义大图编辑上传路由 /a_my_nodes/upload_custom_edited_image 注册成功！")
            else:
                print("⚠️ 路由 /a_my_nodes/upload_custom_edited_image 已经存在，跳过注册")

            # 注册资产管理系统 (Asset Manager) API
            if "/a_my_nodes/assets/prompts" not in existing_routes:
                PromptServer.instance.routes.get("/a_my_nodes/assets/prompts")(get_asset_prompts)
                PromptServer.instance.routes.post("/a_my_nodes/assets/prompts")(save_asset_prompts)
                PromptServer.instance.routes.get("/a_my_nodes/assets/models")(get_asset_models)
                PromptServer.instance.routes.post("/a_my_nodes/assets/models")(save_asset_models)
                PromptServer.instance.routes.get("/a_my_nodes/assets/view_preview")(view_asset_preview)
                PromptServer.instance.routes.post("/a_my_nodes/assets/upload_preview")(upload_asset_preview)
                PromptServer.instance.routes.post("/a_my_nodes/assets/register_local_preview")(register_local_preview)
                PromptServer.instance.routes.get("/a_my_nodes/assets/check_model_exists")(check_model_exists)
                PromptServer.instance.routes.post("/a_my_nodes/assets/check_models_exist")(check_models_exist)
                PromptServer.instance.routes.post("/a_my_nodes/assets/search_pinyin")(search_pinyin)
                print("✅ 资产管理系统 (Asset Manager) API 注册成功！")

            _routes_registered = True  # 设置注册标志
        else:
            print("⚠️ PromptServer.instance 未初始化，路由注册延迟")
    except Exception as e:
        print(f"❌ 路由注册失败: {e}")

# 延迟注册函数 - 用于在PromptServer初始化后注册
def delayed_register_routes():
    """延迟注册路由，确保PromptServer已经初始化"""
    import threading
    import time
    
    def _register():
        max_attempts = 30  # 增加尝试次数
        attempt = 0
        
        while attempt < max_attempts:
            try:
                if hasattr(PromptServer, 'instance') and PromptServer.instance is not None:
                    register_routes()
                    break
                else:
                    if attempt % 5 == 0: # 减少日志频率
                        print(f"⏳ [A_my_nodes] 等待PromptServer初始化... (尝试 {attempt + 1}/{max_attempts})")
                    time.sleep(1)
                    attempt += 1
            except Exception as e:
                print(f"❌ [A_my_nodes] 延迟注册失败: {e}")
                attempt += 1
        
        if attempt >= max_attempts:
            print("❌ [A_my_nodes] 路由注册超时，PromptServer可能未正确初始化")
    
    # 在后台线程中执行延迟注册
    thread = threading.Thread(target=_register, daemon=True)
    thread.start()

# 导出注册函数，供其他模块调用
# __all__ = ['register_routes', 'static_output_file']

async def upload_custom_edited_image(request):
    try:
        import time
        import base64
        from io import BytesIO
        from PIL import Image, ImageOps, ImageChops
        import folder_paths
        import os
        
        data = await request.json()
        image_path_str = data.get("image_path")
        mask_b64 = data.get("mask_data")
        paint_b64 = data.get("paint_data")
        
        if not image_path_str:
            return web.Response(status=400, text="Missing image_path")
            
        if image_path_str.endswith(" [input]"):
            image_path_str = image_path_str[:-8]
            
        input_dir = folder_paths.get_input_directory()
        full_image_path = os.path.normpath(os.path.join(input_dir, image_path_str))
        
        if not full_image_path.startswith(os.path.normpath(input_dir)):
            return web.Response(status=403, text="Forbidden")
            
        if not os.path.isfile(full_image_path):
            return web.Response(status=404, text="Original image not found")
            
        orig_img = Image.open(full_image_path)
        orig_img = ImageOps.exif_transpose(orig_img)
        orig_img = orig_img.convert("RGBA")
        orig_w, orig_h = orig_img.size
        
        mask_alpha = None
        if mask_b64 and mask_b64.startswith("data:image"):
            mask_bytes = base64.b64decode(mask_b64.split(",")[1])
            mask_rgba = Image.open(BytesIO(mask_bytes)).convert("RGBA")
            mask_rgba = mask_rgba.resize((orig_w, orig_h), Image.Resampling.LANCZOS)
            mask_alpha = mask_rgba.split()[3] # Alpha channel: 255 where drawn, 0 where empty
            
        paint_img = None
        if paint_b64 and paint_b64.startswith("data:image"):
            paint_bytes = base64.b64decode(paint_b64.split(",")[1])
            paint_img = Image.open(BytesIO(paint_bytes)).convert("RGBA")
            paint_img = paint_img.resize((orig_w, orig_h), Image.Resampling.LANCZOS)
            
        ts = int(time.time() * 1000)
        clipspace_dir = os.path.join(input_dir, "clipspace")
        os.makedirs(clipspace_dir, exist_ok=True)
        
        # 1. clipspace-mask: 原图底 + 反转的 Alpha 遮罩
        # 这样未涂抹区域Alpha为255(显示原图)，涂抹区域Alpha透明(在白底查看器中显示为白色遮罩)
        inv_mask_alpha = ImageOps.invert(mask_alpha) if mask_alpha else Image.new("L", (orig_w, orig_h), 255)
        mask_out = orig_img.copy()
        mask_out.putalpha(inv_mask_alpha)
        mask_out.save(os.path.join(clipspace_dir, f"clipspace-mask-{ts}.png"))
            
        # 2. clipspace-paint: 完全透明底的彩色笔触，保留100%精确透明度
        paint_path = os.path.join(clipspace_dir, f"clipspace-paint-{ts}.png")
        if paint_img:
            paint_img.save(paint_path)
        else:
            Image.new("RGBA", (orig_w, orig_h), (0,0,0,0)).save(paint_path)
            
        # 3. clipspace-painted: 原图底叠加精确的透明笔触
        painted_img = orig_img.copy()
        if paint_img:
            painted_img.alpha_composite(paint_img)
        painted_path = os.path.join(clipspace_dir, f"clipspace-painted-{ts}.png")
        painted_img.save(painted_path)
        
        # 4. clipspace-painted-masked: 带有笔触的原图 + 反转的 Alpha 遮罩
        final_img = painted_img.copy()
        final_img.putalpha(inv_mask_alpha)
        final_path = os.path.join(clipspace_dir, f"clipspace-painted-masked-{ts}.png")
        final_img.save(final_path)
        
        return web.json_response({
            "success": True,
            "filepath": f"clipspace/clipspace-painted-masked-{ts}.png"
        })
    except Exception as e:
        print(f"❌ [ImageEditor] Error saving edited image: {e}")
        import traceback
        traceback.print_exc()
        return web.Response(status=500, text=str(e))

