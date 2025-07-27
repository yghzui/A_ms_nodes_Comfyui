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
 * 创建并显示一个功能丰富的灯箱用于图片预览。
 * 支持缩放、平移、重置和图片切换。
 * @param {string[]} urls - 要显示的图片URL数组。
 * @param {number} currentIndex - 当前要显示的图片在数组中的索引。
 */
function showLightbox(urls, currentIndex) {
    // 移除已存在的灯箱
    const existingLightbox = document.getElementById("my-nodes-lightbox");
    if (existingLightbox) {
        existingLightbox.remove();
    }

    const lightbox = document.createElement("div");
    lightbox.id = "my-nodes-lightbox";
    Object.assign(lightbox.style, {
        position: "fixed", top: "0", left: "0", width: "100%", height: "100%",
        backgroundColor: "rgba(0, 0, 0, 0.85)", display: "flex",
        flexDirection: "column", justifyContent: "center", alignItems: "center", 
        zIndex: "1001", userSelect: "none"
    });

    const container = document.createElement("div");
    container.className = "lightbox-image-container"; // 用于事件委托
    Object.assign(container.style, {
        display: "flex", flexDirection: "column",
        alignItems: "center", gap: "10px"
    });

    const img = document.createElement("img");
    const sizeInfo = document.createElement("div");

    // --- 状态变量 ---
    let scale = 1;
    let panX = 0;
    let panY = 0;
    let isPanning = false;
    let panStart = { x: 0, y: 0 };

    // --- 核心功能函数 ---
    const updateTransform = () => {
        img.style.transform = `translate(${panX}px, ${panY}px) scale(${scale})`;
    };

    const updateSizeInfo = () => {
        if (img.naturalWidth) {
            sizeInfo.textContent = `${img.naturalWidth} × ${img.naturalHeight} | ${Math.round(scale * 100)}%`;
        }
    };
    
    const resetView = () => {
        scale = 1; panX = 0; panY = 0;
        img.style.transition = 'transform 0.2s ease-out';
        updateTransform();
        updateSizeInfo();
        setTimeout(() => img.style.transition = 'none', 200);
    };

    const loadImage = (index) => {
        if (index < 0 || index >= urls.length) return;
        currentIndex = index;
        img.src = urls[currentIndex];
        sizeInfo.textContent = "加载中...";
        resetView();
    };

    const closeLightbox = () => {
        window.removeEventListener("mousemove", handleMouseMove);
        window.removeEventListener("mouseup", handleMouseUp);
        window.removeEventListener("keydown", handleKeyDown);
        lightbox.remove();
    };

    // --- 事件处理器 ---
    const handleMouseMove = (e) => {
        if (!isPanning) return;
        e.preventDefault();
        panX = e.clientX - panStart.x;
        panY = e.clientY - panStart.y;
        updateTransform();
    };
    const handleMouseUp = (e) => {
        if (isPanning) {
            isPanning = false;
            img.style.cursor = "grab";
        }
    };
    const handleKeyDown = (e) => {
        if (e.key === "Escape") closeLightbox();
        if (e.key === "ArrowLeft" && urls.length > 1) loadImage((currentIndex - 1 + urls.length) % urls.length);
        if (e.key === "ArrowRight" && urls.length > 1) loadImage((currentIndex + 1) % urls.length);
    };

    // --- 元素设置和事件绑定 ---
    Object.assign(img.style, {
        maxWidth: "95vw", maxHeight: "90vh", objectFit: "contain",
        cursor: "grab", transition: "none",
    });
    img.onload = () => {
        resetView(); // 使用 resetView 来设置初始文字和变换
    };
    img.addEventListener("mousedown", (e) => {
        if (e.button !== 0) return;
        e.preventDefault();
        isPanning = true;
        panStart.x = e.clientX - panX;
        panStart.y = e.clientY - panY;
        img.style.cursor = "grabbing";
    });
    img.addEventListener("wheel", (e) => {
        e.preventDefault();
        const rect = img.getBoundingClientRect();
        const zoomSpeed = 0.1;
        const oldScale = scale;

        // Calculate new scale
        scale *= (1 - Math.sign(e.deltaY) * zoomSpeed);
        scale = Math.max(0.1, Math.min(scale, 15));

        const scaleRatio = scale / oldScale;

        // Get scaling origin (center of image)
        const originX = rect.left + rect.width / 2;
        const originY = rect.top + rect.height / 2;
        
        // Get mouse position
        const mouseX = e.clientX;
        const mouseY = e.clientY;

        // Calculate the pan adjustment needed to keep mouse position constant
        const panXDelta = (mouseX - originX) * (1 - scaleRatio);
        const panYDelta = (mouseY - originY) * (1 - scaleRatio);

        // Apply the adjustment to the current pan values
        panX += panXDelta;
        panY += panYDelta;
        
        updateTransform();
        updateSizeInfo();
    });

    lightbox.addEventListener("dblclick", (e) => {
        // 只有双击背景时才关闭
        if (e.target === lightbox || e.target === container) {
            closeLightbox();
        }
    });
    // 防止拖动时意外选中文本
    lightbox.addEventListener("mousedown", (e) => { if (e.detail > 1) e.preventDefault(); });
    
    Object.assign(sizeInfo.style, {
        position: "absolute",
        bottom: "20px",
        left: "50%",
        transform: "translateX(-50%)",
        zIndex: 1003,
        color: "white", fontSize: "16px", fontFamily: "monospace",
        padding: "5px 10px", backgroundColor: "rgba(0, 0, 0, 0.7)",
        borderRadius: "4px"
    });

    container.appendChild(img);
    lightbox.appendChild(container);
    lightbox.appendChild(sizeInfo);


    // --- 控制按钮 ---
    const createButton = (text, styles) => {
        const button = document.createElement("button");
        button.textContent = text;
        Object.assign(button.style, {
            position: "fixed", zIndex: "1002",
            backgroundColor: "rgba(0, 0, 0, 0.5)", color: "white", 
            border: "1px solid #555", cursor: "pointer",
            display: "flex", justifyContent: "center", alignItems: "center",
            transition: "background-color 0.2s ease",
            ...styles
        });
        button.addEventListener("dblclick", (e) => e.stopPropagation());
        button.addEventListener("mouseenter", () => button.style.backgroundColor = "rgba(255, 255, 255, 0.2)");
        button.addEventListener("mouseleave", () => button.style.backgroundColor = "rgba(0, 0, 0, 0.5)");
        return button;
    };
    
    if (urls.length > 1) {
        const prevButton = createButton("‹", { left: "20px", top: "50%", transform: "translateY(-50%)", borderRadius: "50%", width: "50px", height: "50px", fontSize: "24px" });
        prevButton.addEventListener("click", () => loadImage((currentIndex - 1 + urls.length) % urls.length));
        lightbox.appendChild(prevButton);
        
        const nextButton = createButton("›", { right: "20px", top: "50%", transform: "translateY(-50%)", borderRadius: "50%", width: "50px", height: "50px", fontSize: "24px" });
        nextButton.addEventListener("click", () => loadImage((currentIndex + 1) % urls.length));
        lightbox.appendChild(nextButton);
    }

    const resetButton = createButton("⭯", { top: "20px", right: "70px", borderRadius: "8px", width: "40px", height: "40px", fontSize: "20px" });
    resetButton.addEventListener("click", resetView);
    lightbox.appendChild(resetButton);

    const closeButton = createButton("✕", { top: "20px", right: "20px", borderRadius: "8px", width: "40px", height: "40px", fontSize: "20px" });
    closeButton.addEventListener("click", closeLightbox);
    lightbox.appendChild(closeButton);

    // --- 启动 ---
    document.body.appendChild(lightbox);
    window.addEventListener("mousemove", handleMouseMove);
    window.addEventListener("mouseup", handleMouseUp);
    window.addEventListener("keydown", handleKeyDown);
    loadImage(currentIndex);
}

/**
 * 计算给定容器尺寸和图片数量下的最佳行列数
 * @param {number} containerWidth - 容器宽度
 * @param {number} containerHeight - 容器高度
 * @param {number} imageCount - 图片数量
 * @returns {{rows: number, cols: number, size: number}} 最佳的行列数和单个图片的大小
 */
function calculateOptimalGrid(containerWidth, containerHeight, imageCount) {
    if (imageCount === 0) return { rows: 0, cols: 0, size: 0 };
    if (imageCount === 1) return { rows: 1, cols: 1, size: Math.min(containerWidth, containerHeight) };

    const GAP = 2; // 图片间的间距
    const PADDING = 5; // 容器的内边距
    
    // 可用空间
    const availableWidth = containerWidth - (PADDING * 2);
    const availableHeight = containerHeight - (PADDING * 2);
    
    // 初始化最佳值
    let bestRows = 1;
    let bestCols = 1;
    let bestSize = 0;
    
    // 尝试不同的行数
    for (let rows = 1; rows <= imageCount; rows++) {
        // 计算对应的列数（向上取整以确保能容纳所有图片）
        const cols = Math.ceil(imageCount / rows);
        
        // 计算基于宽度的单个图片大小
        const sizeFromWidth = (availableWidth - (GAP * (cols - 1))) / cols;
        // 计算基于高度的单个图片大小
        const sizeFromHeight = (availableHeight - (GAP * (rows - 1))) / rows;
        
        // 取较小的尺寸确保不会超出容器
        const size = Math.min(sizeFromWidth, sizeFromHeight);
        
        // 检查这个尺寸是否合适
        const totalWidth = (size * cols) + (GAP * (cols - 1));
        const totalHeight = (size * rows) + (GAP * (rows - 1));
        
        if (totalWidth <= availableWidth && totalHeight <= availableHeight) {
            // 如果这个尺寸比之前找到的更大，就更新最佳值
            if (size > bestSize) {
                bestSize = size;
                bestRows = rows;
                bestCols = cols;
            }
        }
    }
    
    return { 
        rows: bestRows, 
        cols: bestCols, 
        size: bestSize 
    };
}

/**
 * 更新节点上的图片预览区域。
 * @param {object} node - LiteGraph节点实例。
 * @param {string[]} paths - 图片的相对路径数组。
 */
function updateImagePreviews(node, paths) {
    const PREVIEW_WIDGET_NAME = "image_previews";
    const GAP = 2; // 图片间的间距
    const PADDING = 5; // 容器的内边距

    // 清理所有旧的预览相关元素
    function cleanupOldPreviews() {
        // 1. 查找并移除所有相关的DOM元素
        const oldElements = document.querySelectorAll(`[data-node-id="${node.id}"].image-preview-container`);
        oldElements.forEach(el => {
            if (el && el.parentNode) {
                el.parentNode.removeChild(el);
            }
        });

        // 2. 查找并清理所有相关的widget
        const widgetsToRemove = node.widgets.filter(w => w.name === PREVIEW_WIDGET_NAME);
        widgetsToRemove.forEach(widget => {
            // 调用清理函数
            if (widget.onRemoved) {
                widget.onRemoved();
            }
            // 移除DOM元素
            if (widget.inputEl && widget.inputEl.parentNode) {
                widget.inputEl.parentNode.removeChild(widget.inputEl);
            }
            // 从widgets数组中移除
            const index = node.widgets.indexOf(widget);
            if (index !== -1) {
                node.widgets.splice(index, 1);
            }
        });
    }

    // 先清理所有旧的预览
    cleanupOldPreviews();
    
    if (!paths || paths.length === 0 || (paths.length === 1 && !paths[0])) {
        node.computeSize();
        app.graph.setDirtyCanvas(true, true);
        return;
    }

    const previewContainer = document.createElement("div");
    previewContainer.className = "image-preview-container";
    previewContainer.dataset.nodeId = node.id;
    Object.assign(previewContainer.style, {
        display: "flex",
        flexDirection: "column",
        padding: `${PADDING}px`,
        width: "calc(100% - ${PADDING * 2}px)",
        height: "calc(100% - ${PADDING * 2}px)",
        boxSizing: "border-box",
        overflow: "hidden"
    });

    // 创建图片网格容器
    const gridContainer = document.createElement("div");
    Object.assign(gridContainer.style, {
        display: "flex",
        flexDirection: "column",
        gap: `${GAP}px`,
        width: "fit-content", // 让容器宽度适应内容
        height: "100%",
        margin: "0 auto" // 使用 margin auto 实现水平居中
    });

    // 清除旧的图片元素
    gridContainer.innerHTML = '';
    
    // 初始化图片加载
    const validPaths = paths.filter(path => path.trim());
    const imageUrls = validPaths.map(path => api.apiURL(`/view?filename=${encodeURIComponent(path)}&type=input`));
    const imageElements = imageUrls.map((imageUrl, index) => {
        const thumb = document.createElement("img");
        thumb.src = imageUrl;
        thumb.style.objectFit = "contain";
        thumb.style.cursor = "pointer";
        thumb.style.border = "1px solid #444";
        thumb.style.borderRadius = "4px";
        thumb.style.backgroundColor = "#1a1a1a";
        
        thumb.addEventListener("click", (e) => {
            e.stopPropagation();
            showLightbox(imageUrls, index);
        });
        
        return thumb;
    });

    // 更新布局的函数
    const updateLayout = () => {
        // 清空现有内容
        gridContainer.innerHTML = "";
        
        // 获取容器尺寸
        const containerWidth = previewContainer.clientWidth;
        const containerHeight = previewContainer.clientHeight;
        
        // 计算最佳行列数和图片大小
        const { rows, cols, size } = calculateOptimalGrid(
            containerWidth, 
            containerHeight, 
            imageElements.length
        );

        // 创建行
        for (let r = 0; r < rows; r++) {
            const row = document.createElement("div");
            Object.assign(row.style, {
                display: "flex",
                gap: `${GAP}px`,
                justifyContent: "flex-start", // 改回左对齐
                width: "fit-content", // 宽度自适应内容
                height: `${size}px`, // 使用计算出的大小
                minHeight: `${size}px` // 确保最小高度
            });

            // 填充每一行的图片
            for (let c = 0; c < cols; c++) {
                const index = r * cols + c;
                if (index < imageElements.length) {
                    const imgContainer = document.createElement("div");
                    Object.assign(imgContainer.style, {
                        width: `${size}px`,
                        height: `${size}px`,
                        minWidth: `${size}px`,
                        minHeight: `${size}px`,
                        display: "flex",
                        justifyContent: "center",
                        alignItems: "center"
                    });
                    
                    const img = imageElements[index];
                    Object.assign(img.style, {
                        width: "100%",
                        height: "100%"
                    });
                    
                    imgContainer.appendChild(img);
                    row.appendChild(imgContainer);
                }
            }
            
            gridContainer.appendChild(row);
        }
    };

    previewContainer.appendChild(gridContainer);

    // 创建并添加widget
    const widget = node.addDOMWidget(PREVIEW_WIDGET_NAME, "div", previewContainer);
    widget.options.serialize = false;

    // 初始化布局
    setTimeout(updateLayout, 0);

    // 添加resize观察器
    const resizeObserver = new ResizeObserver(entries => {
        for (const entry of entries) {
            if (entry.contentBoxSize) {
                updateLayout();
            }
        }
    });
    resizeObserver.observe(previewContainer);

    // 清理函数
    widget.onRemoved = () => {
        resizeObserver.disconnect();
    };

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
                                // 先清除旧的预览
                                updateImagePreviews(node, []);
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

            // 当节点大小改变时，动态调整预览区域的高度
            chainCallback(nodeType.prototype, "onResize", function(size) {
                const previewWidget = this.widgets.find(w => w.name === "image_previews");
                if (previewWidget && previewWidget.inputEl) {
                    // 使用 offsetTop 动态计算预览区域上方所有元素占用的空间。
                    // size[1] 是节点的总高度。
                    const headerHeight = previewWidget.inputEl.offsetTop;
                    
                    // 从总高度中减去头部高度，并留出一些底部间距
                    const newHeight = size[1] - headerHeight - 15; // 15px for bottom margin
                    
                    // 应用新高度，并设置一个最小高度防止其过小
                    previewWidget.inputEl.style.maxHeight = `${Math.max(50, newHeight)}px`;
                }
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