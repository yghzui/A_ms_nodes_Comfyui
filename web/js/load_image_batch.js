// 在脚本顶部添加日志，以便在浏览器控制台中确认脚本是否被加载
console.log("Loading custom node: A_my_nodes/web/js/load_image_batch.js");

import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

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
 * 创建并显示一个灯箱用于图片预览。
 * @param {string} url - 要显示的图片URL。
 */
function showLightbox(url) {
    const lightbox = document.createElement("div");
    lightbox.id = "my-nodes-lightbox";
    Object.assign(lightbox.style, {
        position: "fixed", top: "0", left: "0", width: "100%", height: "100%",
        backgroundColor: "rgba(0, 0, 0, 0.85)", display: "flex",
        justifyContent: "center", alignItems: "center", zIndex: "1001",
    });
    const img = document.createElement("img");
    img.src = url;
    Object.assign(img.style, {
        maxWidth: "90%", maxHeight: "90%", objectFit: "contain",
    });
    lightbox.appendChild(img);
    document.body.appendChild(lightbox);
    lightbox.addEventListener("click", () => document.body.removeChild(lightbox));
}

/**
 * 更新节点上的图片预览区域。
 * @param {object} node - LiteGraph节点实例。
 * @param {string[]} paths - 图片的相对路径数组。
 */
function updateImagePreviews(node, paths) {
    const PREVIEW_WIDGET_NAME = "image_previews";

    // 每次更新前，先尝试移除旧的小部件
    const existingPreview = node.widgets.find(w => w.name === PREVIEW_WIDGET_NAME);
    if (existingPreview) {
        // 直接从widgets数组中移除
        node.widgets.splice(node.widgets.indexOf(existingPreview), 1);
    }
    
    // 如果路径为空或无效，则不创建新的小部件，并确保图形刷新
    if (!paths || paths.length === 0 || (paths.length === 1 && !paths[0])) {
        node.computeSize();
        app.graph.setDirtyCanvas(true, true);
        return;
    }

    const previewContainer = document.createElement("div");
    Object.assign(previewContainer.style, {
        display: "flex", flexWrap: "wrap", gap: "5px",
        padding: "5px", maxHeight: "250px", overflowY: "auto",
    });

    paths.forEach(path => {
        if (!path.trim()) return;
        const imageUrl = api.apiURL(`/view?filename=${encodeURIComponent(path)}&type=input`);
        
        const thumb = document.createElement("img");
        thumb.src = imageUrl;
        Object.assign(thumb.style, {
            width: "70px", height: "70px", objectFit: "cover",
            cursor: "pointer", border: "1px solid #444", borderRadius: "4px",
        });
        
        thumb.addEventListener("click", (e) => {
            e.stopPropagation();
            showLightbox(imageUrl);
        });
        
        previewContainer.appendChild(thumb);
    });

    const widget = node.addDOMWidget(PREVIEW_WIDGET_NAME, "div", previewContainer);
    widget.options.serialize = false; 

    node.computeSize();
    app.graph.setDirtyCanvas(true, true);
}


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
                const triggerWidget = node.widgets.find((w) => w.name === "trigger");

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
                            
                            // 使用 Promise.all 并发上传所有文件
                            const uploadPromises = files.map(file => {
                                const formData = new FormData();
                                formData.append("image", file, file.name);
                                // 为每个文件创建一个独立的上传请求
                                return api.fetchApi("/upload/image", { method: "POST", body: formData });
                            });

                            const responses = await Promise.all(uploadPromises);

                            const allPaths = [];
                            let hasError = false;

                            for (const response of responses) {
                                if (response.status === 200 || response.status === 201) {
                                    const data = await response.json();
                                    const path = data.subfolder ? `${data.subfolder}/${data.name}` : data.name;
                                    allPaths.push(path);
                                } else {
                                    console.error("图片上传失败:", await response.text());
                                    hasError = true;
                                }
                            }

                            if (hasError) {
                                alert("部分或全部图片上传失败，请查看浏览器控制台获取详细信息。");
                            }

                            if (allPaths.length > 0) {
                                // 将所有成功上传的路径合并
                                pathWidget.value = allPaths.join(',');
                                triggerWidget.value = (triggerWidget.value || 0) + 1;
                                updateImagePreviews(node, allPaths);
                            }

                        } catch (error) {
                            alert(`上传出错: ${error}`);
                            console.error(error);
                        }
                    },
                });

                document.body.appendChild(fileInput);
                this.onRemoved = () => fileInput.remove();
                
                const uploadWidget = node.addWidget("button", "选择图片", "select_files", () => fileInput.click());
                uploadWidget.options.serialize = false;
            });
            
            // 当工作流加载时，恢复预览
            chainCallback(nodeType.prototype, "onConfigure", function() {
                const imagePathsWidget = this.widgets.find(w => w.name === "image_paths");
                if (imagePathsWidget && imagePathsWidget.value) {
                    updateImagePreviews(this, imagePathsWidget.value.split(','));
                }
            });
        }
    },
}); 