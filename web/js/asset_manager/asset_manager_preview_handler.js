import { api } from "../../../scripts/api.js";

export class PreviewHandler {
    /**
     * 核心处理函数：优先判断本地绝对路径（智能重定向或零拷贝），兜底执行Blob上传
     * @param {string|null} sourceText - 可能包含的路径文本
     * @param {File|null} fileObj - 文件对象
     * @returns {Promise<string|null>} - 返回处理好的 URI (models:// 或 previews://)
     */
    static async processPreviewSource(sourceText, fileObj) {
        try {
            // 1. 如果有明确的文本路径，且看起来像绝对路径或内部协议
            if (sourceText && typeof sourceText === "string") {
                const text = sourceText.trim();
                if (text.includes(":\\") || text.startsWith("/") || text.startsWith("models://") || text.startsWith("previews://")) {
                    const res = await api.fetchApi("/a_my_nodes/assets/register_local_preview", {
                        method: "POST",
                        body: JSON.stringify({ path: text }),
                        headers: { "Content-Type": "application/json" }
                    });
                    const data = await res.json();
                    if (data.success) return data.uri;
                }
            }
            
            // 2. 如果是文件对象
            if (fileObj instanceof File) {
                let absolutePath = null;
                
                // 探测各种可能藏有绝对路径的私有属性（Electron / Chromium 扩展）
                if (fileObj.path && (fileObj.path.includes(":\\") || fileObj.path.startsWith("/"))) {
                    absolutePath = fileObj.path;
                } else if (fileObj.name && (fileObj.name.includes(":\\") || fileObj.name.startsWith("/"))) {
                    absolutePath = fileObj.name;
                }
                
                if (absolutePath) {
                    const res = await api.fetchApi("/a_my_nodes/assets/register_local_preview", {
                        method: "POST",
                        body: JSON.stringify({ path: absolutePath }),
                        headers: { "Content-Type": "application/json" }
                    });
                    const data = await res.json();
                    if (data.success) return data.uri;
                }

                // 3. 兜底方案：无法获取绝对路径的纯文件/Blob，直接走二进制上传流程
                const formData = new FormData();
                formData.append("image", fileObj);
                // 顺便把文件名传过去，后端在兜底时可以用作保存的参考
                formData.append("original_name", fileObj.name || "unknown.png");
                const res = await api.fetchApi("/a_my_nodes/assets/upload_preview", {
                    method: "POST",
                    body: formData
                });
                const data = await res.json();
                if (data.success) {
                    return data.uri;
                } else {
                    throw new Error(data.error || "上传图片失败");
                }
            }
        } catch (e) {
            console.error("[AssetManager] 处理预览图失败:", e);
            throw e;
        }
        return null;
    }

    /**
     * 基于浏览器 paste 事件提取图片或路径
     * @param {ClipboardEvent} e 
     * @returns {Promise<string|null>}
     */
    static async handlePasteEvent(e) {
        if (!e.clipboardData) return null;

        // 1. 尝试读取可能附带的路径文本 (如果是文件管理器里复制的)
        let textPath = e.clipboardData.getData("text/plain") || e.clipboardData.getData("text/uri-list");
        if (textPath) {
            textPath = textPath.trim().replace(/^"|"$/g, '');
            if (textPath.startsWith("file:///")) {
                textPath = decodeURI(textPath.replace("file:///", ""));
                if (textPath.match(/^[a-zA-Z]:\//)) {
                    textPath = textPath.replace(/\//g, "\\");
                } else {
                    textPath = "/" + textPath;
                }
            }
            if (textPath.includes(":\\") || textPath.startsWith("/") || textPath.startsWith("models://")) {
                return await this.processPreviewSource(textPath, null);
            }
        }

        // 2. 如果没能提取到路径，尝试从 items 中读取文件对象
        const items = e.clipboardData.items;
        for (let i = 0; i < items.length; i++) {
            const item = items[i];
            if (item.kind === "file" && item.type.startsWith("image/")) {
                const file = item.getAsFile();
                if (file) {
                    return await this.processPreviewSource(null, file);
                }
            }
        }

        // 如果是外部浏览器直接拖拽过来的 File 对象集合
        if (e.clipboardData.files && e.clipboardData.files.length > 0) {
            const file = e.clipboardData.files[0];
            if (file.type.startsWith("image/")) {
                return await this.processPreviewSource(null, file);
            }
        }
        
        throw new Error("剪贴板中没有发现图片文件！");
    }
    static async handlePasteFromClipboard() {
        // 尝试读取文本（如果是复制的文件路径）
        try {
            let text = await navigator.clipboard.readText();
            if (text) {
                text = text.trim().replace(/^"|"$/g, ''); // 兼容 Windows "复制文件地址" 自带的双引号
                if (text.includes(":\\") || text.startsWith("/") || text.startsWith("models://") || text.startsWith("previews://")) {
                    return await this.processPreviewSource(text, null);
                }
            }
        } catch (e) {
            // 忽略读取文本错误，可能是图片
        }

        // 尝试读取图片 Blob (部分环境下剪贴板图片无法转文本)
        try {
            const items = await navigator.clipboard.read();
            for (const item of items) {
                // 1. 先尝试读取作为文本的文件路径
                const uriType = item.types.find(t => t === "text/uri-list");
                const plainType = item.types.find(t => t === "text/plain");
                
                let text = "";
                if (plainType) {
                    const textBlob = await item.getType(plainType);
                    text = await textBlob.text();
                } else if (uriType) {
                    const uriBlob = await item.getType(uriType);
                    text = await uriBlob.text();
                }
                
                if (text) {
                    const lines = text.split('\n');
                    let firstPath = lines[0].trim().replace(/^"|"$/g, '');
                    
                    if (firstPath.startsWith("file:///")) {
                        firstPath = decodeURI(firstPath.replace("file:///", ""));
                        if (firstPath.match(/^[a-zA-Z]:\//)) {
                            firstPath = firstPath.replace(/\//g, "\\");
                        } else {
                            firstPath = "/" + firstPath;
                        }
                    }
                    
                    if (firstPath.includes(":\\") || firstPath.startsWith("/") || firstPath.startsWith("models://") || firstPath.startsWith("previews://")) {
                        return await this.processPreviewSource(firstPath, null);
                    }
                }

                // 2. 如果不是路径，再尝试读取纯图片流
                const imageType = item.types.find(t => t.startsWith("image/"));
                if (imageType) {
                    const blob = await item.getType(imageType);
                    const file = new File([blob], "pasted_image.png", { type: blob.type });
                    return await this.processPreviewSource(null, file);
                }
            }
        } catch (e) {
            console.error("无法读取剪贴板图片", e);
            throw new Error("无法读取剪贴板，请确保网页有权限访问剪贴板，或尝试复制图片文件路径后重试。");
        }
        
        throw new Error("剪贴板中没有图片或有效的路径文本！");
    }
}
