import os
from aiohttp import web
from server import PromptServer
from folder_paths import get_output_directory

# 全局标志，用于防止重复注册
_routes_registered = False

# 定义路由处理函数 - 提供实际的文件服务功能
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
            if "/static_output/{path:.*}" in existing_routes:
                print("⚠️ 路由 /static_output/{path:.*} 已经存在，跳过注册")
                _routes_registered = True
                return
                
            # 注册路由到PromptServer的路由表
            PromptServer.instance.routes.get("/static_output/{path:.*}")(serve_output_file)
            _routes_registered = True  # 设置注册标志
            print("✅ 我自己的路由 /static_output/{path:.*} 注册成功！")
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
        max_attempts = 10
        attempt = 0
        
        while attempt < max_attempts:
            try:
                if hasattr(PromptServer, 'instance') and PromptServer.instance is not None:
                    register_routes()
                    break
                else:
                    print(f"⏳ 等待PromptServer初始化... (尝试 {attempt + 1}/{max_attempts})")
                    time.sleep(1)
                    attempt += 1
            except Exception as e:
                print(f"❌ 延迟注册失败: {e}")
                attempt += 1
        
        if attempt >= max_attempts:
            print("❌ 路由注册超时，PromptServer可能未正确初始化")
    
    # 在后台线程中执行延迟注册
    thread = threading.Thread(target=_register, daemon=True)
    thread.start()

# 导出注册函数，供其他模块调用
# __all__ = ['register_routes', 'static_output_file']
