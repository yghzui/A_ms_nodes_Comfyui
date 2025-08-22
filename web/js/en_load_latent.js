// 在脚本顶部添加日志，以便在浏览器控制台中确认脚本是否被加载
console.log("Loading custom node: A_my_nodes/web/js/en_load_latent.js");

import { app } from '../../../scripts/app.js'
import { api } from '../../../scripts/api.js'

/**
 * 从 VideoHelperSuite 示例中借鉴的健壮的回调链函数。
 * 它可以安全地将我们的新功能附加到现有函数（如 onNodeCreated）上，
 * 而不会破坏原始函数的行为或返回值。
 * @param {object} object 要修改的对象 (通常是 nodeType.prototype)
 * @param {string} property 要修改的函数名 (例如 "onNodeCreated")
 * @param {function} callback 我们要附加的新函数
 */
function chainCallback(object, property, callback) {
    if (object == undefined) {
        console.error("chainCallback: 尝试修改一个不存在的对象！");
        return;
    }
    if (property in object && object[property]) {
        const originalCallback = object[property];
        object[property] = function () {
            // 首先调用原始函数，并保存其返回值
            const originalReturn = originalCallback.apply(this, arguments);
            // 然后调用我们的新函数
            // 如果我们的函数有返回值，则使用它，否则沿用原始的返回值
            return callback.apply(this, arguments) ?? originalReturn;
        };
    } else {
        // 如果原始函数不存在，则直接设置我们的函数
        object[property] = callback;
    }
}

/**
 * 上传单个latent文件
 * @param {File} file - 要上传的文件
 * @returns {Promise<Object>} 上传结果
 */
async function uploadLatentFile(file) {
    try {
        console.log('开始上传latent文件:', file.name, 'size=', file.size);
        
        const formData = new FormData();
        formData.append("image", file, file.name); // 使用image端点上传latent文件
        
        const response = await api.fetchApi("/upload/image", { 
            method: "POST", 
            body: formData 
        });
        
        if (response.status === 200 || response.status === 201) {
            const data = await response.json();
            console.log('latent文件上传成功:', data);
            return data;
        } else {
            const errorText = await response.text();
            console.error("latent文件上传失败:", errorText);
            throw new Error(`上传失败: ${response.status} - ${errorText}`);
        }
    } catch (error) {
        console.error('上传latent文件时出错:', error);
        throw error;
    }
}

// 允许从widget拖拽
function allowDragFromWidget(widget) {
    widget.onPointerDown = function(pointer, node) {
        pointer.onDragStart = () => {
            app.canvas.emitBeforeChange()
            app.canvas.graph?.beforeChange()
            // 确保拖拽正确清理
            pointer.finally = () => {
                app.canvas.isDragging = false
                app.canvas.graph?.afterChange()
                app.canvas.emitAfterChange()
            }
            app.canvas.processSelect(node, pointer.eDown, true)
            app.canvas.isDragging = true
        }
        pointer.onDragEnd = (e) => {
            if (e.shiftKey || LiteGraph.alwaysSnapToGrid)
                app.graph?.snapToGrid(app.canvas.selectedItems)
            app.canvas.dirty_canvas = true
            app.canvas.dirty_bgcanvas = true
            app.canvas.onNodeMoved?.(app.canvas.selectedItems.find(item => item.type === 'node'))
        }
        app.canvas.dirty_canvas = true
        return true
    }
}

/**
 * 检查DataTransfer是否包含文件
 * @param {DataTransfer} dataTransfer - 拖拽数据传输对象
 * @returns {boolean} 是否包含文件
 */
function hasFiles(dataTransfer) {
    return dataTransfer.types.includes("Files");
}

/**
 * 从DataTransfer获取文件数组
 * @param {DataTransfer} dataTransfer - 拖拽数据传输对象
 * @returns {File[]} 文件数组
 */
function getFilesFromDataTransfer(dataTransfer) {
    const files = [];
    if (dataTransfer.items) {
        for (let i = 0; i < dataTransfer.items.length; i++) {
            if (dataTransfer.items[i].kind === "file") {
                files.push(dataTransfer.items[i].getAsFile());
            }
        }
    } else {
        for (let i = 0; i < dataTransfer.files.length; i++) {
            files.push(dataTransfer.files[i]);
        }
    }
    return files;
}

/**
 * 上传多个latent文件
 * @param {File[]} files - 要上传的文件数组
 * @returns {Promise<string[]>} 上传成功的文件名数组
 */
async function uploadLatentFiles(files) {
    const uploadedFiles = [];
    
    for (const file of files) {
        try {
            // 检查文件扩展名
            if (!file.name.toLowerCase().endsWith('.latent')) {
                console.warn(`跳过非latent文件: ${file.name}`);
                continue;
            }
            
            const result = await uploadLatentFile(file);
            if (result && result.name) {
                uploadedFiles.push(result.name);
            }
        } catch (error) {
            console.error(`上传文件 ${file.name} 失败:`, error);
            alert(`上传文件 ${file.name} 失败: ${error.message}`);
        }
    }
    
    return uploadedFiles;
}

/**
 * 处理传入的文件（拖拽或粘贴）
 * @param {File[]} files - 文件数组
 * @param {Object} node - ComfyUI节点对象
 */
async function handleIncomingFiles(files, node) {
    const latentFiles = files.filter(file => file.name.toLowerCase().endsWith('.latent'));
    
    if (latentFiles.length === 0) {
        alert("请选择.latent文件");
        return;
    }
    
    // 找到上传按钮widget并修改文字为"上传中"
    const uploadButton = node.widgets.find(w => w.type === "button" && w.name.includes("选择Latent文件"));
    const originalButtonText = uploadButton ? uploadButton.name : null;
    if (uploadButton) {
        uploadButton.name = "⏳ 上传中...";
        node.setDirtyCanvas(true); // 触发重绘
    }
    
    try {
        const uploadedFiles = await uploadLatentFiles(latentFiles);
        
        if (uploadedFiles.length > 0) {
             // 上传成功后直接更新选中名和刷新列表选项
             const fileName = uploadedFiles[0];
             const latentWidget = node.widgets.find(w => w.name === "latent");
             
             if (latentWidget) {
                 latentWidget.value = fileName;
                 node.setDirtyCanvas(true);
             }
             
             // 刷新文件列表选项
             await refreshLatentFileList(node, fileName);
             // 拖拽上传成功不需要额外提醒
         }
    } catch (error) {
        console.error('处理文件时出错:', error);
        alert(`处理文件时出错: ${error.message}`);
    } finally {
        // 恢复按钮原始文字
        if (uploadButton && originalButtonText) {
            uploadButton.name = originalButtonText;
            node.setDirtyCanvas(true); // 触发重绘
        }
    }
}

/**
 * 刷新latent文件列表
 * @param {Object} node - ComfyUI节点对象
 * @param {string} selectFile - 要选择的文件名（可选）
 */
async function refreshLatentFileList(node, selectFile = null) {
    try {
        // 获取最新的文件列表
        const response = await api.fetchApi("/object_info");
        if (response.status === 200) {
            const data = await response.json();
            
            // 查找LoadLatentUpload节点的信息
            const nodeInfo = data.LoadLatentUpload || data.LoadLatent;
            if (nodeInfo && nodeInfo.input && nodeInfo.input.required && nodeInfo.input.required.latent) {
                const latentFiles = nodeInfo.input.required.latent[0];
                
                // 更新节点的文件列表
                if (node.widgets) {
                    const latentWidget = node.widgets.find(w => w.name === "latent");
                    if (latentWidget) {
                        latentWidget.options.values = latentFiles;
                        
                        // 如果指定了要选择的文件，则选择它
                        if (selectFile && latentFiles.includes(selectFile)) {
                            latentWidget.value = selectFile;
                        }
                        
                        // 触发重绘
                        node.setDirtyCanvas(true);
                    }
                }
            }
        }
    } catch (error) {
        console.error("刷新latent文件列表失败:", error);
    }
}

// 注册LoadLatentUpload节点扩展
app.registerExtension({
    name: "A_my_nodes.LoadLatentUpload",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "LoadLatentUpload") {
            console.log("注册LoadLatentUpload节点扩展");
            
            // 使用chainCallback安全地扩展onNodeCreated方法
            chainCallback(nodeType.prototype, "onNodeCreated", function() {
                const node = this;
                console.log("LoadLatentUpload节点已创建", node);
                
                // 创建隐藏的文件输入元素
                const fileInput = document.createElement("input");
                fileInput.type = "file";
                fileInput.accept = ".latent";
                fileInput.multiple = true;
                fileInput.style.display = "none";
                document.body.appendChild(fileInput);
                
                // 处理文件选择事件
                fileInput.onchange = async function() {
                    const files = Array.from(this.files);
                    if (files.length > 0) {
                        await handleIncomingFiles(files, node);
                    }
                    // 清空文件输入，以便可以重复选择相同文件
                    this.value = '';
                };
                
                // 添加上传按钮widget
                const uploadButton = node.addWidget("button", "📁 选择Latent文件", null, () => {
                    fileInput.click();
                });
                
                // 允许从上传按钮拖拽（可选功能）
                 // allowDragFromWidget(uploadButton);
                
                // 存储文件输入引用，以便在节点销毁时清理
                node._latentFileInput = fileInput;
            });
            
            // 添加拖拽支持
            chainCallback(nodeType.prototype, "onDragOver", function(event) {
                // 只处理包含文件的拖拽事件
                if (hasFiles(event.dataTransfer)) {
                    event.preventDefault();
                    event.stopPropagation();
                    event.dataTransfer.dropEffect = "copy";
                    return true;
                }
            });
            
            chainCallback(nodeType.prototype, "onDragDrop", function(event) {
                // 只处理包含文件的拖拽事件
                if (hasFiles(event.dataTransfer)) {
                    event.preventDefault();
                    event.stopPropagation();
                    
                    const files = getFilesFromDataTransfer(event.dataTransfer);
                    if (files.length > 0) {
                        handleIncomingFiles(files, this);
                    }
                    return true;
                }
            });
            
            // 添加节点销毁时的清理逻辑
            chainCallback(nodeType.prototype, "onRemoved", function() {
                // 清理文件输入元素
                if (this._latentFileInput && this._latentFileInput.parentNode) {
                    this._latentFileInput.parentNode.removeChild(this._latentFileInput);
                }
            });
        }
    }
});

console.log('LoadLatentUpload扩展已加载');