// 在脚本顶部添加日志，以便在浏览器控制台中确认脚本是否被加载
console.log("Loading custom node: A_my_nodes/web/js/load_image_batch.js");

import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";
import { showImageEditor } from "./image_editor/image_editor.js";
import { modal } from "./utils/modal.js";

// 导入提取的模块
import { chainCallback } from "./utils/common.js";
import { calculateImageLayout } from "./load_image/layout.js";
import { 
    updateWidgetValue, 
    showImages, 
    populate, 
    clearImageWithConfirmation, 
    executeClear, 
    openMaskEditorForImage, 
    askAppendOrReplaceIfNeeded 
} from "./load_image/image_manager.js";

// --- ComfyUI 节点扩展 ---
app.registerExtension({
    name: "A_my_nodes.LoadImageBatchAdvanced.JS",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        // 只对我们的目标节点进行操作
        if (nodeData.name === "LoadImageBatchAdvanced") {
            
            console.log(`Patching node: ${nodeData.name}`);

            // 使用 chainCallback 为 onNodeCreated 添加功能
            chainCallback(nodeType.prototype, "onNodeCreated", function() {
                const node = this; // `this` 指向当前的节点实例

                const pathWidget = node.widgets.find((w) => w.name === "image_paths");
                const pathUseWidget = node.widgets.find((w) => w.name === "image_path_use");
                if (pathWidget) pathWidget.hidden = true;
                if (pathUseWidget) pathUseWidget.hidden = false;

                let isForceAppend = false; // 用于标记是否由"追加图片"按钮触发

                const fileInput = document.createElement("input");
                Object.assign(fileInput, {
                    type: "file",
                    accept: "image/jpeg,image/png,image/webp",
                    multiple: true,
                    style: "display: none",
                    onchange: async (event) => {
                        if (!event.target.files.length) return;
                        try {
                            const files = Array.from(event.target.files);
                            await handleIncomingFiles(files, isForceAppend);
                        } catch (error) {
                            console.error("处理选择的图片时出错:", error);
                        } finally {
                            event.target.value = "";
                            isForceAppend = false; // 重置状态
                        }
                    },
                });

                document.body.appendChild(fileInput);
                this.onRemoved = () => fileInput.remove();
                
                const uploadWidget = node.addWidget("button", "选择图片", "select_files", () => {
                    isForceAppend = false;
                    fileInput.click();
                });
                uploadWidget.options.serialize = false;

                node._customTriggerAppend = () => {
                    isForceAppend = true;
                    fileInput.click();
                };

                // ---------------- 新增：通用工具与拖拽/粘贴支持 ----------------
                // 判断 DataTransfer 是否包含文件
                function hasFilesFromDataTransfer(dt) {
                    try {
                        if (!dt) return false;
                        if (dt.items && dt.items.length) {
                            for (const item of dt.items) {
                                if (item.kind === 'file') return true;
                            }
                        }
                        if (dt.files && dt.files.length) return true;
                    } catch (e) {
                        console.warn('检测拖拽文件失败:', e);
                    }
                    return false;
                }

                // 通过 DataTransfer 获取 File[]
                function getFilesFromDataTransfer(dt) {
                    const files = [];
                    if (!dt) return files;
                    if (dt.items && dt.items.length) {
                        for (const item of dt.items) {
                            if (item.kind === 'file') {
                                const f = item.getAsFile();
                                if (f) files.push(f);
                            }
                        }
                    } else if (dt.files && dt.files.length) {
                        for (const f of dt.files) files.push(f);
                    }
                    return files;
                }

                // 通过 ClipboardData 获取图片 File[]
                function getImageFilesFromClipboard(clipboardData) {
                    const files = [];
                    if (!clipboardData || !clipboardData.items) return files;
                    for (const item of clipboardData.items) {
                        if (item.kind === 'file' && item.type.startsWith('image/')) {
                            const f = item.getAsFile();
                            if (f) files.push(f);
                        }
                    }
                    return files;
                }

                // 上传一组文件，返回路径数组
                async function uploadFiles(files) {
                    const uploadPromises = files.map(file => {
                        const formData = new FormData();
                        formData.append("image", file, file.name);
                        return api.fetchApi("/upload/image", { method: "POST", body: formData });
                    });
                    const responses = await Promise.all(uploadPromises);
                    const paths = [];
                    for (const response of responses) {
                        if (response.status === 200 || response.status === 201) {
                            const data = await response.json();
                            const path = data.subfolder ? `${data.subfolder}/${data.name}` : data.name;
                            paths.push(path);
                        } else {
                            console.error("图片上传失败:", await response.text());
                        }
                    }
                    return paths;
                }

                // 将新得到的路径合并/替换进节点
                function applyPathsToNode(newPaths, mode) {
                    const oldStr = (pathWidget?.value || '').trim();
                    const oldList = oldStr ? oldStr.split(',').filter(s => s.trim()) : [];
                    let finalList = [];
                    if (mode === 'append') {
                        finalList = [...oldList, ...newPaths];
                    } else { // replace
                        finalList = newPaths;
                    }
                    pathWidget.value = finalList.join(',');
                    const useWidget = node.widgets.find(w => w.name === "image_path_use");
                    if (useWidget) {
                        if (mode === 'append') {
                            const oldSelStr = (useWidget.value || '').trim();
                            const oldSelList = oldSelStr ? oldSelStr.split(',').filter(s => s.trim()) : [];
                            const selectedUnion = Array.from(new Set([...oldSelList, ...newPaths]));
                            useWidget.value = selectedUnion.join(',');
                        } else {
                            useWidget.value = finalList.join(',');
                        }
                    }
                    populate.call(node, finalList);
                }

                // 处理拖拽/粘贴得到的文件，含"追加/替换"选择
                async function handleIncomingFiles(files, forceAppend = false) {
                    if (!files || files.length === 0) return;
                    try {
                        // 先上传
                        const newPaths = await uploadFiles(files);
                        if (newPaths.length === 0) return;
                        // 根据现有列表与用户选择应用（默认有旧图则询问）
                        const oldStr = (pathWidget?.value || '').trim();
                        const oldList = oldStr ? oldStr.split(',').filter(s => s.trim()) : [];
                        let mode = 'replace';
                        if (oldList.length > 0) {
                            if (forceAppend) {
                                mode = 'append';
                            } else {
                                const choice = await askAppendOrReplaceIfNeeded(node, oldList, newPaths.length);
                                if (choice === 'cancel') return;
                                mode = choice === 'append' ? 'append' : 'replace';
                            }
                        }
                        applyPathsToNode(newPaths, mode);
                    } catch (err) {
                        console.error('处理文件时出错:', err);
                    }
                }

                // 拖拽命中（告知系统本节点可接收文件，触发官方高亮）
                node.onDragOver = function (e) {
                    try {
                        return hasFilesFromDataTransfer(e?.dataTransfer);
                    } catch (err) {
                        console.warn('onDragOver 异常:', err);
                        return false;
                    }
                };

                // 释放到节点：拦截并上传，多图支持
                node.onDragDrop = async function (e) {
                    try {
                        const files = getFilesFromDataTransfer(e?.dataTransfer);
                        if (!files || files.length === 0) return false;
                        e.preventDefault();
                        e.stopPropagation();
                        await handleIncomingFiles(files);
                        return true; // 返回 true 表示本节点已处理，阻止默认创建节点/工作流
                    } catch (err) {
                        console.warn('onDragDrop 异常:', err);
                        return false;
                    }
                };

                // 文档级粘贴：若节点被选中或悬浮，并且剪贴板有图片，则拦截为本节点上传
                if (!window.__A_MY_NODES_LOAD_IMAGE_BATCH_PASTE_INSTALLED__) {
                    window.__A_MY_NODES_LOAD_IMAGE_BATCH_PASTE_INSTALLED__ = true;
                    document.addEventListener('paste', async (evt) => {
                        try {
                            const target = evt.target;
                            if (!target) return;
                            const clsList = target.classList || { contains: () => false };
                            const isCanvasZone = clsList.contains('litegraph') || clsList.contains('graph-canvas-container');
                            if (!isCanvasZone) return;

                            const canvas = app?.canvas;
                            if (!canvas) return;

                            let targets = [];

                            const sel = canvas.selected_nodes;
                            if (sel && typeof sel === 'object') {
                                for (const k in sel) {
                                    const n = sel[k];
                                    if (n && n.type === 'LoadImageBatchAdvanced') {
                                        targets.push(n);
                                    }
                                }
                            }

                            if (!targets.length) {
                                const over = canvas.over_node;
                                if (over && over.type === 'LoadImageBatchAdvanced') {
                                    targets = [over];
                                }
                            }
                            
                            if (!targets.length) return;

                            let files = getImageFilesFromClipboard(evt.clipboardData);

                            if (!files || files.length === 0) {
                                if (navigator.clipboard && navigator.clipboard.read) {
                                    try {
                                        const items = await navigator.clipboard.read();
                                        const out = [];
                                        for (const item of items) {
                                            for (const type of item.types) {
                                                if (type && type.startsWith('image/')) {
                                                    const blob = await item.getType(type);
                                                    const ext = (type.split('/')[1] || 'png').toLowerCase();
                                                    const file = new File([blob], `pasted-${Date.now()}.${ext}`, { type });
                                                    out.push(file);
                                                }
                                            }
                                        }
                                        files = out;
                                    } catch (clipErr) {
                                        // 权限不足或不支持，忽略
                                    }
                                }
                            }

                            if (!files || files.length === 0) return;

                            evt.preventDefault();
                            evt.stopPropagation();

                            await Promise.all(targets.map(t => {
                                if (t._customHandleIncomingFiles) {
                                    return t._customHandleIncomingFiles(files);
                                }
                                return Promise.resolve();
                            }));
                        } catch (err) {
                            console.warn('paste 处理异常:', err);
                        }
                    }, true);
                }

                // 将内部处理方法暴露到实例，供右键菜单调用
                this._customHandleIncomingFiles = handleIncomingFiles; // 处理文件入口
                // 从异步 Clipboard API 读取图片文件
                this._customReadClipboardImages = async function() {
                    try {
                        if (!navigator.clipboard || !navigator.clipboard.read) {
                            console.warn('浏览器不支持 navigator.clipboard.read');
                            return [];
                        }
                        const items = await navigator.clipboard.read();
                        const out = [];
                        for (const item of items) {
                            for (const type of item.types) {
                                if (type && type.startsWith('image/')) {
                                    const blob = await item.getType(type);
                                    const ext = (type.split('/')[1] || 'png').toLowerCase();
                                    const file = new File([blob], `pasted-${Date.now()}.${ext}`, { type });
                                    out.push(file);
                                }
                            }
                        }
                        return out;
                    } catch (err) {
                        console.warn('读取剪贴板图片失败:', err);
                        return [];
                    }
                };
                // ---------------- 新增结束 ----------------
            });

            // 限制节点最小尺寸
            chainCallback(nodeType.prototype, "onResize", function(size) {
                const TOP_MARGIN = 210;
                const minHeight = TOP_MARGIN + 200;
                const buttonSpacing = 10;
                const selectW = 60;
                const deselectW = 70;
                const invertW = 60;
                const clearW = 90;
                const showSelectedW = 90;
                const reuseMaskW = 100;
                const leftMargin = 10;
                const totalButtonWidth = leftMargin + 
                                       selectW + buttonSpacing + 
                                       deselectW + buttonSpacing + 
                                       invertW + buttonSpacing + 
                                       clearW + buttonSpacing + 
                                       clearW + buttonSpacing + 
                                       showSelectedW + buttonSpacing + 
                                       reuseMaskW;
                const minWidth = totalButtonWidth + 10;

                if (size[0] < minWidth) size[0] = minWidth;
                if (size[1] < minHeight) size[1] = minHeight;
            });

            // 新增：为节点追加右键菜单"粘贴"项（与官方 Load Image 一致的入口）
            chainCallback(nodeType.prototype, "getExtraMenuOptions", function(_, options) {
                const self = this;
                
                // 确保只处理目标节点
                if (self.type !== "LoadImageBatchAdvanced") {
                    return; // 重要：对于非目标节点，不应有任何返回值，以便 chainCallback 返回 originalReturn
                }

                // --- 新增：检查是否有图片被点击，如果有则添加编辑选项 ---
                if (self._customImgs && self._customImageRects && self._customImagePaths) {
                    const nodePos = self.pos;
                    // app.canvas.graph_mouse 是全局坐标 [x, y]
                    const canvasX = app.canvas.graph_mouse[0];
                    const canvasY = app.canvas.graph_mouse[1];
                    const relX = canvasX - nodePos[0];
                    const relY = canvasY - nodePos[1];

                    // 检查点击了哪个图片
                    let clickedImageIndex = -1;
                    for (let i = 0; i < self._customImageRects.length; i++) {
                        const rect = self._customImageRects[i];
                        if (rect && rect.visible !== false &&
                            relX >= rect.x && relX <= rect.x + rect.width &&
                            relY >= rect.y && relY <= rect.y + rect.height) {
                            clickedImageIndex = i;
                            break;
                        }
                    }

                    if (clickedImageIndex !== -1) {
                         options.unshift({
                            content: "编辑图片 (全屏)",
                            callback: () => {
                                if (self._customImagePaths && self._customImagePaths.length > 0) {
                                    showImageEditor(self._customImagePaths, clickedImageIndex, self, (newPaths) => {
                                        self._customImagePaths = [...newPaths];
                                        
                                        // 更新文件名
                                        for(let j = 0; j < self._customImagePaths.length; j++) {
                                            const pathParts = self._customImagePaths[j].split(/[\\\/]/);
                                            if (self._customImageFileNames) {
                                                self._customImageFileNames[j] = pathParts[pathParts.length - 1];
                                            }
                                        }
                                        
                                        // 更新 widget
                                        const imagePathsWidget = self.widgets.find(w => w.name === "image_paths");
                                        if (imagePathsWidget) {
                                            imagePathsWidget.value = self._customImagePaths.join(',');
                                        }
                                        updateWidgetValue(self);
                                        
                                        // 刷新显示
                                        showImages(self, self._customImagePaths);
                                        app.graph.setDirtyCanvas(true, false);
                                    });
                                }
                            }
                         });
                         options.unshift({
                            content: "编辑图片 (官方MaskEditor)",
                            callback: () => {
                                // 保存原始路径，防止官方编辑器覆盖后丢失原图引用
                                if (!self.properties) self.properties = {};
                                if (!self.properties.original_image_paths || self.properties.original_image_paths.length !== self._customImagePaths.length) {
                                    self.properties.original_image_paths = [...self._customImagePaths];
                                }

                                let imagePath = self._customImagePaths[clickedImageIndex];
                                let isInput = false;
                                // 处理 [input] 后缀：如果有该后缀，先去除以便正确加载图片，但在保存时需要恢复
                                if (imagePath && imagePath.endsWith(" [input]")) {
                                    imagePath = imagePath.substring(0, imagePath.length - 8);
                                    isInput = true;
                                }

                                // 解析路径，分离 filename 和 subfolder
                                let filename = imagePath;
                                let subfolder = "";
                                const lastSlashIndex = imagePath.lastIndexOf('/');
                                const lastBackslashIndex = imagePath.lastIndexOf('\\');
                                const slashIndex = Math.max(lastSlashIndex, lastBackslashIndex);
                                
                                if (slashIndex !== -1) {
                                    subfolder = imagePath.substring(0, slashIndex);
                                    filename = imagePath.substring(slashIndex + 1);
                                }

                                // 构造完整 URL
                                let urlParams = `?filename=${encodeURIComponent(filename)}&type=input`;
                                if (subfolder) {
                                    urlParams += `&subfolder=${encodeURIComponent(subfolder)}`;
                                }
                                const imageUrl = api.apiURL(`/view${urlParams}`);
                                
                                openMaskEditorForImage(imageUrl, (result) => {
                                    console.log("Editor saved result:", result);
                                    
                                    // 解析结果
                                    let newPath = "";
                                    if (typeof result === 'string') {
                                        newPath = result;
                                    } else if (result && typeof result === 'object') {
                                         if (result.filename) {
                                             newPath = result.subfolder ? `${result.subfolder}/${result.filename}` : result.filename;
                                         }
                                    }

                                    // 恢复 [input] 后缀
                                    if (newPath && isInput && !newPath.endsWith(" [input]")) {
                                        newPath += " [input]";
                                    }
                                    
                                    if (newPath) {
                                        // 更新路径
                                        self._customImagePaths[clickedImageIndex] = newPath;
                                        
                                        // 更新文件名
                                        const pathParts = newPath.split(/[\\\/]/);
                                        const fileName = pathParts[pathParts.length - 1];
                                        if (self._customImageFileNames) {
                                            self._customImageFileNames[clickedImageIndex] = fileName;
                                        }

                                        // 更新 widget
                                        const imagePathsWidget = self.widgets.find(w => w.name === "image_paths");
                                        if (imagePathsWidget) {
                                            imagePathsWidget.value = self._customImagePaths.join(',');
                                        }
                                        updateWidgetValue(self);
                                        
                                        // 刷新显示
                                        showImages(self, self._customImagePaths);
                                        app.graph.setDirtyCanvas(true, false);
                                    }
                                });
                            }
                        });
                    }
                }
                // --- 结束新增 ---

                options.push({
                    content: "粘贴图像",
                    callback: async () => {
                        try {
                            // 优先使用异步 Clipboard API 读取图片
                            const files = (await self._customReadClipboardImages?.()) || [];
                            if (!files.length) {
                                modal.show({ title: '提示', content: '剪贴板中没有图片或浏览器不支持从右键菜单读取图片，请使用 Ctrl+V 粘贴。' });
                                return;
                            }
                            // 复用与拖拽/全局粘贴一致的处理逻辑（含 追加/替换 选择）
                            await self._customHandleIncomingFiles?.(files);
                        } catch (err) {
                            console.error('右键粘贴处理失败:', err);
                        }
                    }
                });
            });

            // 当节点大小改变时，重新计算图片布局
            chainCallback(nodeType.prototype, "onResize", function(size) {
                if (this._customImgs && this._customImageRects) {
                    calculateImageLayout(this, this._customImgs.length);
                    app.graph.setDirtyCanvas(true, true);
                }
            });
            
            // 当工作流加载时，恢复预览
            chainCallback(nodeType.prototype, "onConfigure", function() {
                const imagePathsWidget = this.widgets.find(w => w.name === "image_paths");
                if (imagePathsWidget && imagePathsWidget.value) {
                    const paths = imagePathsWidget.value.split(',').filter(path => path.trim());
                    if (paths.length > 0) {
                        populate.call(this, paths);
                    }
                }
            });
            
            // 添加鼠标事件处理（只在有图片数据时处理）
            const originalOnMouseMove = nodeType.prototype.onMouseMove;
            nodeType.prototype.onMouseMove = function(e) {
                if (originalOnMouseMove) {
                    originalOnMouseMove.call(this, e);
                }
                
                if (this.type !== "LoadImageBatchAdvanced") {
                    return;
                }
                
                // 计算新的鼠标位置
                const newMouseX = e.canvasX - this.pos[0];
                const newMouseY = e.canvasY - this.pos[1];
                
                // 检查鼠标位置是否真的改变了
                const mousePositionChanged = this._customMouseX !== newMouseX || this._customMouseY !== newMouseY;
                
                // 保存鼠标位置用于悬浮检测
                this._customMouseX = newMouseX;
                this._customMouseY = newMouseY;

                // 检查按键状态并设置提示
                this._customHoverKeyStatus = null;
                if (this._customImageRects) {
                    for (let i = 0; i < this._customImageRects.length; i++) {
                        const rect = this._customImageRects[i];
                        if (!rect || rect.visible === false) continue;
                        if (newMouseX >= rect.x && newMouseX <= rect.x + rect.width &&
                            newMouseY >= rect.y && newMouseY <= rect.y + rect.height) {
                            
                            const isSelected = this._customSelectedImages ? this._customSelectedImages[i] : true;
                            const statusText = isSelected ? "已选择" : "未选择";
                            
                            if (e.shiftKey) {
                                this._customHoverKeyStatus = `Shift+Click: 连续选择 (当前: ${statusText})`;
                            } else if (e.ctrlKey) {
                                this._customHoverKeyStatus = `Ctrl+Click: 反转选择 (当前: ${statusText})`;
                            }
                            break;
                        }
                    }
                }
                
                let tooltipShown = false;
                if (this._customFileNameRects && this._customFileNameRects.length > 0) {
                    for (let i = 0; i < this._customFileNameRects.length; i++) {
                        const fileNameRect = this._customFileNameRects[i];
                        if (!fileNameRect) continue;
                        const nodePos = this.pos;
                        const ax = nodePos[0] + fileNameRect.x;
                        const ay = nodePos[1] + fileNameRect.y;
                        const aw = fileNameRect.width;
                        const ah = fileNameRect.height;
                        const mouseIn = e.canvasX >= ax && e.canvasX <= ax + aw && e.canvasY >= ay && e.canvasY <= ay + ah;
                        if (mouseIn && this._customImagePaths && this._customImagePaths[i]) {
                            this.showTooltip(e, i);
                            tooltipShown = true;
                            break;
                        }
                    }
                }
                if (!tooltipShown) {
                    const nodePos = this.pos;
                    if (this._customSingleImageMode) {
                        const controls = [
                            { r: this._customPrevButtonRect, t: '上一张' },
                            { r: this._customNextButtonRect, t: '下一张' },
                            { r: this._customRestoreButtonRect, t: '还原到网格' },
                            { r: this._customClearButtonRect, t: '清除当前图片' },
                            { r: this._customFullscreenButtonRect, t: '全屏预览' }
                        ];
                        for (const c of controls) {
                            if (!c.r) continue;
                            const ax = nodePos[0] + c.r.x, ay = nodePos[1] + c.r.y, aw = c.r.width, ah = c.r.height;
                            if (e.canvasX >= ax && e.canvasX <= ax + aw && e.canvasY >= ay && e.canvasY <= ay + ah) {
                                this.showControlTooltip(e, c.t);
                                tooltipShown = true;
                                break;
                            }
                        }
                    } else {
                        if (this._customButtons && this._customButtons.length > 0) {
                            for (const btn of this._customButtons) {
                                const ax = nodePos[0] + btn.rect.x;
                                const ay = nodePos[1] + btn.rect.y;
                                const aw = btn.rect.w;
                                const ah = btn.rect.h;
                                
                                if (e.canvasX >= ax && e.canvasX <= ax + aw && 
                                    e.canvasY >= ay && e.canvasY <= ay + ah) {
                                    
                                    if (btn.tooltip) {
                                        this.showControlTooltip(e, btn.tooltip);
                                        tooltipShown = true;
                                    }
                                    break;
                                }
                            }
                        }
                    }
                }
                if (!tooltipShown) this.hideTooltip();
                if (mousePositionChanged) app.graph.setDirtyCanvas(true, false);
            };
            
            // 鼠标离开时清除位置
            const originalOnMouseLeave = nodeType.prototype.onMouseLeave;
            nodeType.prototype.onMouseLeave = function(e) {
                if (originalOnMouseLeave) {
                    originalOnMouseLeave.call(this, e);
                }
                
                if (this.type !== "LoadImageBatchAdvanced") {
                    return;
                }
                
                this._customMouseX = undefined;
                this._customMouseY = undefined;
                this.hideTooltip();
                app.graph.setDirtyCanvas(true, false);
            };
            
            // 处理鼠标点击事件
            const originalOnMouseDown = nodeType.prototype.onMouseDown;
            nodeType.prototype.onMouseDown = function(e, localPos) {
                if (this.type !== "LoadImageBatchAdvanced") {
                    if (originalOnMouseDown) {
                        return originalOnMouseDown.call(this, e, localPos);
                    }
                    return false;
                }
                
                const relX = e.canvasX - this.pos[0];
                const relY = e.canvasY - this.pos[1];
                const nodePos = this.pos;

                // 检查是否点击复选框
                if (this._customCheckboxRects && this._customCheckboxRects.length > 0) {
                    for (let i = 0; i < this._customCheckboxRects.length; i++) {
                        const checkboxRect = this._customCheckboxRects[i];
                        if (!checkboxRect) continue;
                        if (this._customImageRects && this._customImageRects[i] && this._customImageRects[i].visible === false) continue;
                        
                        if (relX >= checkboxRect.x && relX <= checkboxRect.x + checkboxRect.width &&
                            relY >= checkboxRect.y && relY <= checkboxRect.y + checkboxRect.height) {
                            
                            e.preventDefault();
                            e.stopPropagation();
                            
                            if (this._customSelectedImages && this._customSelectedImages[i] !== undefined) {
                                this._customSelectedImages[i] = !this._customSelectedImages[i];
                                this._customLastSelectedImageIndex = i;
                                updateWidgetValue(this);
                                app.graph.setDirtyCanvas(true, false);
                            }
                            return true;
                        }
                    }
                }

                // 检查点击图片区域 (包括普通点击、Ctrl、Shift)
                if (this._customImageRects && this._customImageRects.length > 0) {
                    for (let i = 0; i < this._customImageRects.length; i++) {
                        const imageRect = this._customImageRects[i];
                        if (!imageRect || imageRect.visible === false) continue;

                        if (relX >= imageRect.x && relX <= imageRect.x + imageRect.width &&
                            relY >= imageRect.y && relY <= imageRect.y + imageRect.height) {
                            
                            e.preventDefault();
                            e.stopPropagation();

                            if (!this._customSelectedImages) {
                                 this._customSelectedImages = new Array(this._customImageRects.length).fill(true);
                            }

                            if (e.shiftKey) {
                                if (this._customLastSelectedImageIndex === undefined) {
                                    this._customLastSelectedImageIndex = i;
                                    if (this._customSelectedImages) {
                                        this._customSelectedImages[i] = true;
                                    }
                                    updateWidgetValue(this);
                                    app.graph.setDirtyCanvas(true, false);
                                    return true;
                                }
                                
                                const lastIndex = this._customLastSelectedImageIndex;
                                const start = Math.min(lastIndex, i);
                                const end = Math.max(lastIndex, i);
                                
                                if (this._customSelectedImages) {
                                    for (let j = 0; j < this._customSelectedImages.length; j++) {
                                        if (j >= start && j <= end) {
                                            this._customSelectedImages[j] = true;
                                        } else {
                                            this._customSelectedImages[j] = false;
                                        }
                                    }
                                }
                                updateWidgetValue(this);
                                app.graph.setDirtyCanvas(true, false);
                                return true;
                            } else {
                                this._customLastSelectedImageIndex = undefined;
                            }
                        }
                    }
                }
                
                if (!this._customSingleImageMode) {
                    if (this._customButtons && this._customButtons.length > 0) {
                        for (const btn of this._customButtons) {
                            const ax = nodePos[0] + btn.rect.x;
                            const ay = nodePos[1] + btn.rect.y;
                            const aw = btn.rect.w;
                            const ah = btn.rect.h;
                            
                            if (e.canvasX >= ax && e.canvasX <= ax + aw && 
                                e.canvasY >= ay && e.canvasY <= ay + ah) {
                                
                                e.preventDefault();
                                e.stopPropagation();
                                
                                if (btn.callback) {
                                    btn.callback();
                                }
                                return true;
                            }
                        }
                    }
                }
                
                if (this._customSingleImageMode) {
                    if (this._customPrevButtonRect) {
                        const absPrevButtonX = nodePos[0] + this._customPrevButtonRect.x;
                        const absPrevButtonY = nodePos[1] + this._customPrevButtonRect.y;
                        const absPrevButtonWidth = this._customPrevButtonRect.width;
                        const absPrevButtonHeight = this._customPrevButtonRect.height;
                        
                        if (e.canvasX >= absPrevButtonX && e.canvasX <= absPrevButtonX + absPrevButtonWidth &&
                            e.canvasY >= absPrevButtonY && e.canvasY <= absPrevButtonY + absPrevButtonHeight) {
                            
                            e.preventDefault();
                            e.stopPropagation();
                            
                            if (this._customImagePaths && this._customImagePaths.length > 0) {
                                this._customFocusedImageIndex = (this._customFocusedImageIndex - 1 + this._customImagePaths.length) % this._customImagePaths.length;
                                calculateImageLayout(this, this._customImagePaths.length);
                                app.graph.setDirtyCanvas(true, false);
                            }
                            return true;
                        }
                    }
                    
                    if (this._customNextButtonRect) {
                        const absNextButtonX = nodePos[0] + this._customNextButtonRect.x;
                        const absNextButtonY = nodePos[1] + this._customNextButtonRect.y;
                        const absNextButtonWidth = this._customNextButtonRect.width;
                        const absNextButtonHeight = this._customNextButtonRect.height;
                        
                        if (e.canvasX >= absNextButtonX && e.canvasX <= absNextButtonX + absNextButtonWidth &&
                            e.canvasY >= absNextButtonY && e.canvasY <= absNextButtonY + absNextButtonHeight) {
                            
                            e.preventDefault();
                            e.stopPropagation();
                            
                            if (this._customImagePaths && this._customImagePaths.length > 0) {
                                this._customFocusedImageIndex = (this._customFocusedImageIndex + 1) % this._customImagePaths.length;
                                calculateImageLayout(this, this._customImagePaths.length);
                                app.graph.setDirtyCanvas(true, false);
                            }
                            return true;
                        }
                    }
                    
                    {
                        let hitX, hitY, hitW, hitH;
                        if (this._customRestoreButtonRect) {
                            hitX = nodePos[0] + this._customRestoreButtonRect.x;
                            hitY = nodePos[1] + this._customRestoreButtonRect.y;
                            hitW = this._customRestoreButtonRect.width;
                            hitH = this._customRestoreButtonRect.height;
                        } else {
                            const currentImageRect = this._customImageRects ? this._customImageRects[this._customFocusedImageIndex] : null;
                            if (currentImageRect) {
                                hitX = nodePos[0] + currentImageRect.x + (currentImageRect.width * 3) / 4;
                                hitY = nodePos[1] + currentImageRect.y;
                                hitW = currentImageRect.width / 4;
                                hitH = currentImageRect.height / 4;
                            } else {
                                const buttonSize = 20;
                                const restoreButtonX = this.size[0] - buttonSize - 10;
                                const restoreButtonY = 10;
                                hitW = Math.round(buttonSize * 1.8);
                                hitH = Math.round(buttonSize * 1.8);
                                hitX = nodePos[0] + restoreButtonX - Math.floor((hitW - buttonSize) / 2);
                                hitY = nodePos[1] + restoreButtonY - Math.floor((hitH - buttonSize) / 2);
                            }
                        }
                        if (e.canvasX >= hitX && e.canvasX <= hitX + hitW &&
                            e.canvasY >= hitY && e.canvasY <= hitY + hitH) {
                            
                            e.preventDefault();
                            e.stopPropagation();
                            
                            this._customSingleImageMode = false;
                            this._customFocusedImageIndex = -1;
                            
                            if (this._customImagePaths && this._customImagePaths.length > 0) {
                                calculateImageLayout(this, this._customImagePaths.length);
                            }
                            app.graph.setDirtyCanvas(true, false);
                            return true;
                        }
                    }
                    
                    if (this._customClearButtonRect) {
                        const absClearButtonX = nodePos[0] + this._customClearButtonRect.x;
                        const absClearButtonY = nodePos[1] + this._customClearButtonRect.y;
                        const absClearButtonWidth = this._customClearButtonRect.width;
                        const absClearButtonHeight = this._customClearButtonRect.height;
                        
                        if (e.canvasX >= absClearButtonX && e.canvasX <= absClearButtonX + absClearButtonWidth &&
                            e.canvasY >= absClearButtonY && e.canvasY <= absClearButtonY + absClearButtonHeight) {
                            
                            e.preventDefault();
                            e.stopPropagation();
                            
                            this.clearImageWithConfirmation(this._customFocusedImageIndex);
                            return true;
                        }
                    }
                    
                    if (this._customFullscreenButtonRect) {
                        const absFullscreenButtonX = nodePos[0] + this._customFullscreenButtonRect.x;
                        const absFullscreenButtonY = nodePos[1] + this._customFullscreenButtonRect.y;
                        const absFullscreenButtonWidth = this._customFullscreenButtonRect.width;
                        const absFullscreenButtonHeight = this._customFullscreenButtonRect.height;
                        
                        if (e.canvasX >= absFullscreenButtonX && e.canvasX <= absFullscreenButtonX + absFullscreenButtonWidth &&
                            e.canvasY >= absFullscreenButtonY && e.canvasY <= absFullscreenButtonY + absFullscreenButtonHeight) {
                            
                            e.preventDefault();
                            e.stopPropagation();
                            
                            if (this._customImagePaths && this._customImagePaths.length > 0) {
                                showImageEditor(this._customImagePaths, this._customFocusedImageIndex, this, (newPaths) => {
                                    this._customImagePaths = [...newPaths];
                                    for(let i = 0; i < this._customImagePaths.length; i++) {
                                        const pathParts = this._customImagePaths[i].split(/[\\\/]/);
                                        if (this._customImageFileNames) {
                                            this._customImageFileNames[i] = pathParts[pathParts.length - 1];
                                        }
                                    }
                                    const imagePathsWidget = this.widgets.find(w => w.name === "image_paths");
                                    if (imagePathsWidget) {
                                        imagePathsWidget.value = this._customImagePaths.join(',');
                                    }
                                    updateWidgetValue(this);
                                    showImages(this, this._customImagePaths);
                                    app.graph.setDirtyCanvas(true, false);
                                });
                            }
                            return true;
                        }
                    }
                }
                        
                if (this._customClearButtonRects && this._customClearButtonRects.length > 0) {
                    for (let i = 0; i < this._customClearButtonRects.length; i++) {
                        const clearRect = this._customClearButtonRects[i];
                        if (!clearRect) continue;
                        if (this._customImageRects && this._customImageRects[i] && this._customImageRects[i].visible === false) continue;
                                
                        const absClearButtonX = nodePos[0] + clearRect.x;
                        const absClearButtonY = nodePos[1] + clearRect.y;
                        const absClearButtonWidth = clearRect.width;
                        const absClearButtonHeight = clearRect.height;
                        
                        if (e.canvasX >= absClearButtonX && e.canvasX <= absClearButtonX + absClearButtonWidth &&
                            e.canvasY >= absClearButtonY && e.canvasY <= absClearButtonY + absClearButtonHeight) {
                            
                            e.preventDefault();
                            e.stopPropagation();
                            
                            this.clearImageWithConfirmation(i);
                            return true;
                        }
                    }
                }
                
                if (this._customEditButtonRects && this._customEditButtonRects.length > 0) {
                    for (let i = 0; i < this._customEditButtonRects.length; i++) {
                        const editRect = this._customEditButtonRects[i];
                        if (!editRect) continue;
                        if (this._customImageRects && this._customImageRects[i] && this._customImageRects[i].visible === false) continue;
                                
                        const absEditButtonX = nodePos[0] + editRect.x;
                        const absEditButtonY = nodePos[1] + editRect.y;
                        const absEditButtonWidth = editRect.width;
                        const absEditButtonHeight = editRect.height;
                        
                        if (e.canvasX >= absEditButtonX && e.canvasX <= absEditButtonX + absEditButtonWidth &&
                            e.canvasY >= absEditButtonY && e.canvasY <= absEditButtonY + absEditButtonHeight) {
                            
                            e.preventDefault();
                            e.stopPropagation();
                            
                            if (this._customImagePaths && this._customImagePaths.length > 0) {
                                showImageEditor(this._customImagePaths, i, this, (newPaths) => {
                                    this._customImagePaths = [...newPaths];
                                    for(let j = 0; j < this._customImagePaths.length; j++) {
                                        const pathParts = this._customImagePaths[j].split(/[\\\/]/);
                                        if (this._customImageFileNames) {
                                            this._customImageFileNames[j] = pathParts[pathParts.length - 1];
                                        }
                                    }
                                    const imagePathsWidget = this.widgets.find(w => w.name === "image_paths");
                                    if (imagePathsWidget) {
                                        imagePathsWidget.value = this._customImagePaths.join(',');
                                    }
                                    updateWidgetValue(this);
                                    showImages(this, this._customImagePaths);
                                    app.graph.setDirtyCanvas(true, false);
                                });
                            }
                            return true;
                        }
                    }
                }
                        
                if (this._customImageRects && this._customImageRects.length > 0) {
                    for (let i = 0; i < this._customImageRects.length; i++) {
                        const rect = this._customImageRects[i];
                        if (rect.visible === false) continue;
                        
                        const absRectX = nodePos[0] + rect.x;
                        const absRectY = nodePos[1] + rect.y;
                        const absRectWidth = rect.width;
                        const absRectHeight = rect.height;
                        
                        if (e.canvasX >= absRectX && e.canvasX <= absRectX + absRectWidth &&
                            e.canvasY >= absRectY && e.canvasY <= absRectY + absRectHeight) {
                            
                            e.preventDefault();
                            e.stopPropagation();
                            
                            if (!this._customSingleImageMode) {
                                this._customSingleImageMode = true;
                                this._customFocusedImageIndex = i;
                                
                                if (this._customImagePaths && this._customImagePaths.length > 0) {
                                    calculateImageLayout(this, this._customImagePaths.length);
                                }
                                app.graph.setDirtyCanvas(true, false);
                            }
                            return true;
                        }
                    }
                }
                
                if (!e.shiftKey) {
                    this._customLastSelectedImageIndex = undefined;
                }

                if (originalOnMouseDown) {
                    return originalOnMouseDown.call(this, e);
                }
                return false;
            };
            
            // 添加双击事件处理
            const originalOnDblClick = nodeType.prototype.onDblClick;
            nodeType.prototype.onDblClick = function(e) {
                if (this.type !== "LoadImageBatchAdvanced") {
                    if (originalOnDblClick) {
                        return originalOnDblClick.call(this, e);
                    }
                    return false;
                }
                
                const nodePos = this.pos;
                if (this._customSingleImageMode && this._customImageRects && this._customImageRects.length > 0) {
                    const currentImageRect = this._customImageRects[this._customFocusedImageIndex];
                    if (currentImageRect && currentImageRect.visible !== false) {
                        const absRectX = nodePos[0] + currentImageRect.x;
                        const absRectY = nodePos[1] + currentImageRect.y;
                        const absRectWidth = currentImageRect.width;
                        const absRectHeight = currentImageRect.height;
                        
                        if (e.canvasX >= absRectX && e.canvasX <= absRectX + absRectWidth &&
                            e.canvasY >= absRectY && e.canvasY <= absRectY + absRectHeight) {
                            
                            e.preventDefault();
                            e.stopPropagation();
                            
                            if (this._customImagePaths && this._customImagePaths.length > 0) {
                                showImageEditor(this._customImagePaths, this._customFocusedImageIndex, this, (newPaths) => {
                                    this._customImagePaths = [...newPaths];
                                    for(let i = 0; i < this._customImagePaths.length; i++) {
                                        const pathParts = this._customImagePaths[i].split(/[\\\/]/);
                                        if (this._customImageFileNames) {
                                            this._customImageFileNames[i] = pathParts[pathParts.length - 1];
                                        }
                                    }
                                    const imagePathsWidget = this.widgets.find(w => w.name === "image_paths");
                                    if (imagePathsWidget) {
                                        imagePathsWidget.value = this._customImagePaths.join(',');
                                    }
                                    updateWidgetValue(this);
                                    showImages(this, this._customImagePaths);
                                    app.graph.setDirtyCanvas(true, false);
                                });
                            }
                            return true;
                        }
                    }
                }
                
                if (originalOnDblClick) {
                    return originalOnDblClick.call(this, e);
                }
                return false;
            };
            
            // 添加tooltip管理方法
            nodeType.prototype.showTooltip = function(e, imageIndex) {
                this.hideTooltip();
                
                if (this._customImagePaths && this._customImagePaths[imageIndex]) {
                    const tooltip = document.createElement('div');
                    tooltip.id = 'image-tooltip-' + this.id;
                    tooltip.style.cssText = `
                        position: fixed;
                        background: rgba(0, 0, 0, 0.9);
                        color: white;
                        padding: 8px 12px;
                        border-radius: 4px;
                        font-size: 12px;
                        max-width: 400px;
                        word-wrap: break-word;
                        z-index: 10000;
                        pointer-events: none;
                        white-space: nowrap;
                    `;
                    
                    const img = this._customImgs[imageIndex];
                    let sizeInfo = '';
                    if (img && img.naturalWidth && img.naturalHeight) {
                        sizeInfo = ` (${img.naturalWidth}x${img.naturalHeight})`;
                    }
                    
                    let indexInfo = '';
                    if (this._customImagePaths && this._customImagePaths.length > 1) {
                        const currentIndex = imageIndex + 1;
                        const totalCount = this._customImagePaths.length;
                        indexInfo = ` [${currentIndex}/${totalCount}]`;
                    }
                    
                    tooltip.textContent = `相对路径: ${this._customImagePaths[imageIndex]}${sizeInfo}${indexInfo}`;
                    document.body.appendChild(tooltip);
                    
                    const tooltipRect = tooltip.getBoundingClientRect();
                    let left = e.clientX + 10;
                    let top = e.clientY - 30;
                    
                    if (left + tooltipRect.width > window.innerWidth) left = e.clientX - tooltipRect.width - 10;
                    if (top + tooltipRect.height > window.innerHeight) top = e.clientY - tooltipRect.height - 10;
                    
                    tooltip.style.left = left + 'px';
                    tooltip.style.top = top + 'px';
                }
            };
            
            nodeType.prototype.hideTooltip = function() {
                const t1 = document.getElementById('image-tooltip-' + this.id);
                if (t1) t1.remove();
                const t2 = document.getElementById('control-tooltip-' + this.id);
                if (t2) t2.remove();
            };
            
            nodeType.prototype.showControlTooltip = function(e, content) {
                this.hideTooltip();
                const tooltip = document.createElement('div');
                tooltip.id = 'control-tooltip-' + this.id;
                tooltip.style.cssText = `
                    position: fixed;
                    background: rgba(245,245,250,0.95);
                    color: rgba(30,30,35,1);
                    padding: 8px 10px;
                    border-radius: 8px;
                    border: 1px solid rgba(180,180,190,0.6);
                    font-size: 12px;
                    max-width: 300px;
                    z-index: 10000;
                    pointer-events: none;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.15);
                    white-space: nowrap;
                `;
                tooltip.textContent = content;
                document.body.appendChild(tooltip);
                const rect = tooltip.getBoundingClientRect();
                let left = e.clientX + 10;
                let top = e.clientY - 30;
                if (left + rect.width > window.innerWidth) left = e.clientX - rect.width - 10;
                if (top + rect.height > window.innerHeight) top = e.clientY - rect.height - 10;
                tooltip.style.left = left + 'px';
                tooltip.style.top = top + 'px';
            };
            
            // 添加清除图片的方法到节点原型
            nodeType.prototype.clearImageWithConfirmation = clearImageWithConfirmation;
            nodeType.prototype.executeClear = executeClear;
            
            // 添加节点销毁时的清理逻辑
            chainCallback(nodeType.prototype, "onRemoved", function() {
                this._customDrawMethodSet = false;
                this._customImgs = null;
                this._customImageRects = null;
                this._customClearButtonRects = null;
                this._customClearButtonRect = null;
                this._customFullscreenButtonRect = null;
                this._customImageFileNames = null;
                this._customImagePaths = null;
                this._customFileNameRects = null;
                this._customCheckboxRects = null;
                this._customSelectedImages = null;
                this._customSelectAllButtonRect = null;
                this._customInvertSelectionButtonRect = null;
                this._customMouseX = null;
                this._customMouseY = null;
                this._customIsHovered = null;
                
                console.log("节点清理完成");
            });
        }
    },
});
