import { app } from "../../../../scripts/app.js";
import { api } from "../../../../scripts/api.js";
import { showImageEditor } from "../image_editor/image_editor.js";
import { modal } from "../utils/modal.js";
import { showTopNotification } from "../utils/shared_utils.js";
import { calculateImageLayout } from "./layout.js";
import { drawNodeImages } from "./draw.js";

/**
 * 获取多图片模式下的底部按钮配置
 */
export function getCustomButtons(node) {
    return [
        {
            text: '全选',
            width: 60,
            tooltip: '全选所有图片',
            callback: () => {
                if (node._customSelectedImages && node._customSelectedImages.length > 0) {
                    node._customSelectedImages.fill(true);
                    updateWidgetValue(node);
                    app.graph.setDirtyCanvas(true, false);
                }
            }
        },
        {
            text: '全不选',
            width: 70,
            tooltip: '取消所有选择',
            callback: () => {
                if (node._customSelectedImages && node._customSelectedImages.length > 0) {
                    node._customSelectedImages.fill(false);
                    updateWidgetValue(node);
                    app.graph.setDirtyCanvas(true, false);
                }
            }
        },
        {
            text: '反选',
            width: 60,
            tooltip: '反转当前选择',
            callback: () => {
                if (node._customSelectedImages && node._customSelectedImages.length > 0) {
                    for (let i = 0; i < node._customSelectedImages.length; i++) {
                        node._customSelectedImages[i] = !node._customSelectedImages[i];
                    }
                    updateWidgetValue(node);
                    app.graph.setDirtyCanvas(true, false);
                }
            }
        },
        {
            text: '清除选中',
            width: 90,
            tooltip: '清除选中的图片',
            callback: () => {
                if (node._customImagePaths && node._customSelectedImages) {
                    const newPaths = [];
                    const newSelected = [];
                    const newNames = [];
                    for (let i = 0; i < node._customImagePaths.length; i++) {
                        if (!node._customSelectedImages[i]) {
                            newPaths.push(node._customImagePaths[i]);
                            newSelected.push(node._customSelectedImages[i]);
                            if (node._customImageFileNames && node._customImageFileNames[i]) newNames.push(node._customImageFileNames[i]);
                        }
                    }
                    node._customImagePaths = newPaths;
                    node._customSelectedImages = newSelected.length ? newSelected : new Array(newPaths.length).fill(true);
                    node._customImageFileNames = newNames;
                    const imagePathsWidget = node.widgets.find(w => w.name === "image_paths");
                    if (imagePathsWidget) imagePathsWidget.value = (node._customImagePaths || []).join(',');
                    updateWidgetValue(node);
                    showImages(node, node._customImagePaths);
                    app.graph.setDirtyCanvas(true, false);
                }
            }
        },
        {
            text: '清除未选',
            width: 90,
            tooltip: '清除未选中的图片',
            callback: () => {
                if (node._customImagePaths && node._customSelectedImages) {
                    const newPaths = [];
                    const newSelected = [];
                    const newNames = [];
                    for (let i = 0; i < node._customImagePaths.length; i++) {
                        if (node._customSelectedImages[i]) {
                            newPaths.push(node._customImagePaths[i]);
                            newSelected.push(node._customSelectedImages[i]);
                            if (node._customImageFileNames && node._customImageFileNames[i]) newNames.push(node._customImageFileNames[i]);
                        }
                    }
                    node._customImagePaths = newPaths;
                    node._customSelectedImages = newSelected.length ? newSelected : new Array(newPaths.length).fill(true);
                    node._customImageFileNames = newNames;
                    const imagePathsWidget = node.widgets.find(w => w.name === "image_paths");
                    if (imagePathsWidget) imagePathsWidget.value = (node._customImagePaths || []).join(',');
                    updateWidgetValue(node);
                    showImages(node, node._customImagePaths);
                    app.graph.setDirtyCanvas(true, false);
                }
            }
        },
        {
            text: node._customShowOnlySelected ? '显示全部' : '仅显示勾选',
            width: 90,
            tooltip: node._customShowOnlySelected ? '恢复显示全部图片' : '仅显示勾选的图片',
            callback: () => {
                if (!node._customShowOnlySelected) {
                    if (!node._customSelectedImages || !node._customSelectedImages.some(v => v)) {
                        showTopNotification("当前没有勾选的图片，无法仅显示勾选。", "warning");
                        return;
                    }
                    node._customShowOnlySelected = true;
                } else {
                    node._customShowOnlySelected = false;
                }
                if (node._customImagePaths && node._customImagePaths.length > 0) {
                    calculateImageLayout(node, node._customImagePaths.length);
                    app.graph.setDirtyCanvas(true, false);
                }
            }
        },
        {
            text: '复制选中',
            width: 90,
            tooltip: '复制选中图片，供同类节点粘贴',
            callback: () => {
                if (node._customCopySelectedPaths) {
                    node._customCopySelectedPaths();
                    return;
                }

                showTopNotification("节点尚未完全初始化或不支持该操作", "warning");
            }
        },
        {
            text: node._customMaskReuseEnabled ? '遮罩复用✓' : '遮罩复用',
            width: 100,
            tooltip: '相同尺寸的图片复用第一个已编辑遮罩',
            callback: () => {
                node._customMaskReuseEnabled = !node._customMaskReuseEnabled;
                const widget = node.widgets.find(w => w.name === "reuse_mask");
                if (widget) {
                    widget.value = !!node._customMaskReuseEnabled;
                }
                app.graph.setDirtyCanvas(true, false);
            }
        }, 
        {
            text: '追加图片',
            width: 90,
            tooltip: '选择并直接追加图片',
            callback: () => {
                if (node._customTriggerAppend) {
                    node._customTriggerAppend();
                } else {
                    showTopNotification("节点尚未完全初始化或不支持该操作", "warning");
                }
            }
        }
    ];
}

/**
 * 根据选择状态更新widget的值
 * @param {object} node - LiteGraph节点实例
 */
export function updateWidgetValue(node) {
    if (!node._customImagePaths || !node._customSelectedImages) {
        return;
    }
    
    const selectedPaths = [];
    for (let i = 0; i < node._customImagePaths.length; i++) {
        if (node._customSelectedImages[i]) {
            selectedPaths.push(node._customImagePaths[i]);
        }
    }
    
    const imagePathUseWidget = node.widgets.find(w => w.name === "image_path_use");
    if (imagePathUseWidget) {
        imagePathUseWidget.value = selectedPaths.join(',');
        console.log("更新选中图片数量:", selectedPaths.length);
    }
}

/**
 * 显示图片的核心实现
 * @param {object} node - LiteGraph节点实例
 * @param {string[]} paths - 图片路径数组
 */
export function showImages(node, paths) {
    console.log("显示图片，路径:", paths);
    
    if (!paths || paths.length === 0) {
        node._customImgs = [];
        node._customImageRects = [];
        node._customClearButtonRects = [];
        node._customImageFileNames = [];
        node._customImagePaths = [];
        node._customFileNameRects = [];
        node._customSingleImageMode = false;
        node._customFocusedImageIndex = -1;
        node._customPrevButtonRect = null;
        node._customNextButtonRect = null;
        node._customRestoreButtonRect = null;
        node._customCheckboxRects = [];
        node._customSelectedImages = [];
        node._customSelectAllButtonRect = null;
        node._customInvertSelectionButtonRect = null;
        return [];
    }
    
    const validPaths = paths.filter(path => path.trim());
    console.log("有效路径数量:", validPaths.length);
    
    node._customImgs = [];
    node._customImageFileNames = [];
    node._customImagePaths = validPaths;
    node._customFileNameRects = [];
    node._customClearButtonRects = [];
    node._customCheckboxRects = [];
    
    const imagePathUseWidget = node.widgets.find(w => w.name === "image_path_use");
    const selectedList = (imagePathUseWidget && imagePathUseWidget.value) ? imagePathUseWidget.value.split(',').filter(s => s.trim()) : [];
    if (selectedList.length) {
        node._customSelectedImages = validPaths.map(p => selectedList.includes(p));
    } else {
        node._customSelectedImages = new Array(validPaths.length).fill(false);
    }
    
    node._customSingleImageMode = false;
    node._customFocusedImageIndex = -1;
    
    validPaths.forEach((path, index) => {
        const img = new Image();
        node._customImgs.push(img);
        
        let filename = path;
        let type = 'input';
        let subfolder = '';

        const typeMatch = path.match(/^(.*)\s+\[(input|output|temp)\]$/);
        if (typeMatch) {
            filename = typeMatch[1];
            type = typeMatch[2];
        }

        const lastSlash = filename.lastIndexOf('/');
        const lastBackslash = filename.lastIndexOf('\\');
        const splitIndex = Math.max(lastSlash, lastBackslash);
        
        if (splitIndex !== -1) {
            subfolder = filename.substring(0, splitIndex);
            filename = filename.substring(splitIndex + 1);
        }

        node._customImageFileNames.push(filename);
        
        img.onload = () => { 
            console.log(`图片 ${index} 加载完成:`, path);
            app.graph.setDirtyCanvas(true, true); 
        };
        img.onerror = () => {
            console.error(`图片 ${index} 加载失败:`, path);
        };
        
        const params = new URLSearchParams({
            filename: filename,
            type: type,
            subfolder: subfolder
        });
        
        if (type === 'input') {
            img.src = api.apiURL(`/a_my_nodes/view_input?${params.toString()}`);
        } else {
            img.src = api.apiURL(`/view?${params.toString()}`);
        }
    });
    
    calculateImageLayout(node, validPaths.length);
    updateWidgetValue(node);
    
    console.log("图片显示设置完成，图片数量:", node._customImgs.length);
    return node._customImgs;
}

/**
 * 更新节点上的图片预览区域。
 * @param {object} node - LiteGraph节点实例。
 * @param {string[]} paths - 图片的相对路径数组。
 */
export function updateImagePreviews(node, paths) {
    console.log("更新图片预览，路径:", paths);
    
    if (node._customImgs) {
        node._customImgs = [];
    }
    if (node._customImageRects) {
        node._customImageRects = [];
    }
    if (node._customClearButtonRects) {
        node._customClearButtonRects = [];
    }
    
    if (!paths || paths.length === 0 || (paths.length === 1 && !paths[0])) {
        console.log("没有有效路径，清除预览");
        app.graph.setDirtyCanvas(true, true);
        return;
    }
    
    showImages(node, paths);
    app.graph.setDirtyCanvas(true, true);
    
    console.log("图片预览更新完成");
}

/**
 * 处理图片数据更新的核心函数
 * @param {string[]} imagePaths - 图片路径数组
 */
export function populate(imagePaths) {
    console.log("收到新的图片数据，开始更新显示...");
    console.log("新图片路径:", imagePaths);
    console.log("节点当前尺寸:", this.size);
    
    const oldPaths = this._customImagePaths || [];
    const newPaths = imagePaths || [];
    
    const hasChanged = oldPaths.length !== newPaths.length || 
                      oldPaths.some((oldPath, index) => oldPath !== newPaths[index]);
    
    if (!hasChanged) {
        console.log("图片数据没有变化，跳过更新");
        return;
    }
    
    console.log("检测到图片数据变化，开始清除旧数据并加载新数据");
    
    this._customImagePaths = imagePaths;
    showImages(this, imagePaths);
    ensureCustomDrawMethod(this);
    
    // 原有的鼠标事件绑定逻辑已经移到 load_image_batch.js 中以避免循环依赖或过度耦合
    // 在这里触发一次重绘
    setTimeout(() => {
        console.log("延迟后的节点尺寸:", this.size);
        app.graph.setDirtyCanvas(true, false);
    }, 100);
}

export function ensureCustomDrawMethod(node) {
    if (node._customDrawMethodSet) {
        return;
    }

    console.log("设置自定义绘制方法");
    const originalOnDrawForeground = node.onDrawForeground;

    const customDrawForeground = function(ctx) {
        if (originalOnDrawForeground) {
            originalOnDrawForeground.call(this, ctx);
        }
        if (this.type === "LoadImageBatchAdvanced" && this._customImgs && this._customImageRects) {
            drawNodeImages(this, ctx);
        }
    };

    node.onDrawForeground = customDrawForeground;
    node._customDrawMethodSet = true;
    console.log("自定义绘制方法设置完成");
}

/**
 * 清除图片的确认对话框 (重构为使用通用的 modal.js)
 * @param {number} imageIndex - 要清除的图片索引
 */
export function clearImageWithConfirmation(imageIndex) {
    if (!this._customImagePaths || imageIndex < 0 || imageIndex >= this._customImagePaths.length) {
        console.error("无效的图片索引:", imageIndex);
        return;
    }
    
    const content = `
        <h3 style="margin: 0 0 15px 0; color: #ff6b6b;">⚠️ 确认清除图片</h3>
        <p style="margin: 0 0 20px 0;">确定要清除这张图片的预览和路径吗？</p>
        <p style="margin: 0 0 20px 0; color: #ff6b6b;"><strong>此操作不可撤销！</strong></p>
    `;

    modal.show({
        title: '确认清除',
        content: content,
        buttons: [
            { text: '取消' },
            { 
                text: '确认清除', 
                type: 'danger',
                onClick: (e, modalInstance) => {
                    modalInstance.close();
                    this.executeClear(imageIndex);
                }
            }
        ]
    });
}

/**
 * 执行清除操作
 * @param {number} imageIndex - 要清除的图片索引
 */
export function executeClear(imageIndex) {
    console.log(`开始清除图片 ${imageIndex}`);
    
    if (this._customImagePaths && imageIndex < this._customImagePaths.length) {
        this._customImagePaths.splice(imageIndex, 1);
        
        if (this._customSelectedImages && imageIndex < this._customSelectedImages.length) {
            this._customSelectedImages.splice(imageIndex, 1);
        }
        
        if (this._customImageFileNames && imageIndex < this._customImageFileNames.length) {
            this._customImageFileNames.splice(imageIndex, 1);
        }
        
        const imagePathsWidget = this.widgets.find(w => w.name === "image_paths");
        if (imagePathsWidget) imagePathsWidget.value = (this._customImagePaths || []).join(',');
        updateWidgetValue(this);
        showImages(this, this._customImagePaths);
        
        console.log(`✅ 成功清除图片 ${imageIndex}`);
    } else {
        console.error("图片索引超出范围或没有图片数据");
    }
}

/**
 * 打开 MaskEditor 编辑任意图片
 * @param {string} imageUrl - 图片的 URL
 * @param {Function} onSave - 保存回调，接收 (filename, subfolder, type) 或 路径字符串
 */
export function openMaskEditorForImage(imageUrl, onSave) {
    const mockNode = {
        id: -1,
        type: "MockNode",
        title: "Mock Image Editor",
        imgs: [{
            src: imageUrl,
            width: 512,
            height: 512
        }],
        widgets: [{
            name: "image",
            value: "",
            callback: (newValue) => {
                console.log("[MockEditor] Saved:", newValue);
                if (onSave) {
                    onSave(newValue);
                }
            }
        }],
        setDirtyCanvas: () => {},
        setSize: () => {},
        getBounding: () => [0,0,100,100],
        isResizeable: () => false,
        properties: {}
    };

    const ext = app.extensions.find(e => e.name === "Comfy.MaskEditor");
    if (!ext) {
        console.error("Comfy.MaskEditor extension not found");
        showTopNotification("未找到 MaskEditor 插件，请先安装。", "error");
        return;
    }

    const cmd = ext.commands.find(c => c.id === "Comfy.MaskEditor.OpenMaskEditor");
    if (!cmd) {
        console.error("OpenMaskEditor command not found");
        showTopNotification("MaskEditor 插件未注册打开命令。", "error");
        return;
    }

    const originalSelection = app.canvas.selected_nodes;
    app.canvas.selected_nodes = { [mockNode.id]: mockNode };

    try {
        cmd.function();
    } catch (e) {
        console.error("Failed to open MaskEditor:", e);
        showTopNotification("打开编辑器失败: " + e.message, "error");
    } finally {
        app.canvas.selected_nodes = originalSelection;
    }
}

/**
 * 弹出"追加/替换"选择对话框（重构为使用通用的 modal.js）
 */
export function askAppendOrReplaceIfNeeded(node, existingList, incomingCount) {
    return new Promise((resolve) => {
        if (!existingList || existingList.length === 0) {
            return resolve('replace');
        }
        
        const content = `
            <p style="margin:0 0 12px 0;">当前已有 <strong>${existingList.length}</strong> 张图片，将要添加 <strong>${incomingCount}</strong> 张图片。</p>
            <p style="margin:0 0 12px 0;">请选择如何处理：</p>
        `;
        
        modal.show({
            title: '检测到已有图片',
            content: content,
            buttons: [
                { text: '取消', onClick: (e, modalInstance) => { modalInstance.close(); resolve('cancel'); } },
                { text: '替换', type: 'danger', onClick: (e, modalInstance) => { modalInstance.close(); resolve('replace'); } },
                { text: '追加', type: 'primary', onClick: (e, modalInstance) => { modalInstance.close(); resolve('append'); } }
            ]
        });
    });
}
