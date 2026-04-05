import os
import json
from aiohttp import web
from server import PromptServer
from folder_paths import get_output_directory
from .nodes.resolutionpreset import get_resolution_presets_file_path

# 全局标志，用于防止重复注册
_routes_registered = False


async def get_resolution_presets(request):
    file_path = get_resolution_presets_file_path()
    data = {}
    if os.path.isfile(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                data = raw
                print(f"📖 [ResolutionPreset] 成功读取预设文件: {file_path}, 包含 {len(data)} 个预设")
            else:
                print(f"⚠️ [ResolutionPreset] 预设文件格式错误 (非字典): {type(raw)}")
        except Exception as e:
            print(f"❌ [ResolutionPreset] 读取预设文件失败: {e}")
            data = {}
    else:
        print(f"ℹ️ [ResolutionPreset] 预设文件不存在: {file_path}")
    return web.json_response({"presets": data})


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
    presets = body.get("presets")
    if not isinstance(presets, dict):
        print(f"❌ [ResolutionPreset] presets格式错误: {type(presets)}")
        return web.Response(
            status=400,
            text=json.dumps({"error": "presets must be an object"}),
            content_type="application/json",
        )
    file_path = get_resolution_presets_file_path()
    try:
        print(f"💾 [ResolutionPreset] 正在保存预设到: {file_path}")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(presets, f, ensure_ascii=False, indent=2)
        print(f"✅ [ResolutionPreset] 预设保存成功")
    except Exception as e:
        print(f"❌ [ResolutionPreset] 保存文件失败: {e}")
        return web.Response(
            status=500,
            text=json.dumps({"error": str(e)}),
            content_type="application/json",
        )
    return web.json_response({"success": True})

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
        
        print(f"✅ 成功删除文件: {relative_path}")
        
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
        if field.name == 'image':
            filename = str(uuid.uuid4()) + "_" + field.filename
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
    """智能路径分析：判断本地绝对路径是否在 models 目录下，决定是零拷贝引用还是复制到 previews"""
    try:
        data = await request.json()
        local_path = data.get("path", "")
        if not os.path.exists(local_path):
            return web.json_response({"success": False, "error": "File does not exist"})
            
        models_root = os.path.abspath(folder_paths.models_dir)
        abs_local = os.path.abspath(local_path)
        
        # 智能路径分析：如果在 models 目录下，则生成 models:// 协议路径（零拷贝）
        if abs_local.startswith(models_root):
            rel_path = os.path.relpath(abs_local, models_root)
            uri = f"models://{rel_path.replace(os.sep, '/')}"
            return web.json_response({"success": True, "uri": uri, "action": "referenced"})
        else:
            # 不在 models 目录下，拷贝到 previews 目录
            filename = str(uuid.uuid4()) + "_" + os.path.basename(local_path)
            dest_path = os.path.join(PREVIEWS_DIR, filename)
            shutil.copy2(abs_local, dest_path)
            return web.json_response({"success": True, "uri": f"previews://{filename}", "action": "copied"})
    except Exception as e:
        print(f"❌ [AssetManager] 智能路径注册失败: {e}")
        return web.json_response({"success": False, "error": str(e)})

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

            # 注册资产管理系统 (Asset Manager) API
            if "/a_my_nodes/assets/prompts" not in existing_routes:
                PromptServer.instance.routes.get("/a_my_nodes/assets/prompts")(get_asset_prompts)
                PromptServer.instance.routes.post("/a_my_nodes/assets/prompts")(save_asset_prompts)
                PromptServer.instance.routes.get("/a_my_nodes/assets/models")(get_asset_models)
                PromptServer.instance.routes.post("/a_my_nodes/assets/models")(save_asset_models)
                PromptServer.instance.routes.get("/a_my_nodes/assets/view_preview")(view_asset_preview)
                PromptServer.instance.routes.post("/a_my_nodes/assets/upload_preview")(upload_asset_preview)
                PromptServer.instance.routes.post("/a_my_nodes/assets/register_local_preview")(register_local_preview)
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
