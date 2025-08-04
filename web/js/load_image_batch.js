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
 * 计算图片网格布局，支持单图片模式和多图片模式
 * @param {object} node - LiteGraph节点实例
 * @param {number} imageCount - 图片数量
 */
function calculateImageLayout(node, imageCount) {
    console.log("计算图片布局，图片数量:", imageCount);
    
    if (imageCount === 0) {
        node._customImageRects = [];
        return;
    }
    
    const containerWidth = node.size[0];
    const containerHeight = node.size[1];
    const GAP = 3;
    const PADDING = 8;
    
    // 为顶部输入控件和图片标题预留空间
    const TOP_MARGIN = 50; // 顶部控件的高度
    const TITLE_HEIGHT = 25; // 图片标题的高度
    
    const availableWidth = containerWidth - (PADDING * 2);
    const availableHeight = containerHeight - (PADDING * 2) - TOP_MARGIN - TITLE_HEIGHT;
    
    // 检查是否处于单图片模式
    if (node._customSingleImageMode && node._customFocusedImageIndex >= 0 && node._customFocusedImageIndex < imageCount) {
        // 单图片模式：只显示一个图片，最大化显示
        const imageSize = Math.min(availableWidth, availableHeight);
        const x = PADDING + (availableWidth - imageSize) / 2;
        const y = PADDING + TOP_MARGIN + (availableHeight - imageSize) / 2;
        
        node._customImageRects = [];
        for (let i = 0; i < imageCount; i++) {
            if (i === node._customFocusedImageIndex) {
                // 显示聚焦的图片
                node._customImageRects.push({
                    x: x,
                    y: y,
                    width: imageSize,
                    height: imageSize,
                    visible: true
                });
            } else {
                // 隐藏其他图片
                node._customImageRects.push({
                    x: 0,
                    y: 0,
                    width: 0,
                    height: 0,
                    visible: false
                });
            }
        }
        
        // 单图片模式不改变节点大小，保持当前大小
        console.log("单图片模式，保持节点大小:", node.size);
    } else {
        // 多图片模式：计算最佳网格
    let bestRows = 1;
    let bestCols = 1;
    let bestSize = 0;
    
    for (let rows = 1; rows <= imageCount; rows++) {
        const cols = Math.ceil(imageCount / rows);
        const sizeFromWidth = (availableWidth - (GAP * (cols - 1))) / cols;
        const sizeFromHeight = (availableHeight - (GAP * (rows - 1))) / rows;
        const size = Math.min(sizeFromWidth, sizeFromHeight);
        
            if (size > bestSize) {
                bestSize = size;
                bestRows = rows;
                bestCols = cols;
            }
        }
        
        // 计算每个图片的位置
        node._customImageRects = [];
        for (let i = 0; i < imageCount; i++) {
            const row = Math.floor(i / bestCols);
            const col = i % bestCols;
            const x = PADDING + col * (bestSize + GAP);
            const y = PADDING + TOP_MARGIN + row * (bestSize + GAP);
            
            node._customImageRects.push({
                x: x,
                y: y,
                width: bestSize,
                height: bestSize,
                visible: true
            });
        }
        
        // 只在初始化时调整节点大小，模式切换时不改变大小
        if (!node._customSizeInitialized) {
            const totalWidth = (bestSize * bestCols) + (GAP * (bestCols - 1)) + (PADDING * 2);
            const totalHeight = (bestSize * bestRows) + (GAP * (bestRows - 1)) + (PADDING * 2) + TOP_MARGIN;
            
            const newSize = [Math.max(totalWidth, 200), Math.max(totalHeight, 100)];
            console.log("初始化多图片模式，设置节点大小:", newSize);
            
            node.size[0] = newSize[0];
            node.size[1] = newSize[1];
            node._customSizeInitialized = true;
            node.setDirtyCanvas(true, false);
            app.graph.setDirtyCanvas(true, false);
        } else {
            console.log("多图片模式，保持节点大小:", node.size);
        }
    }
    
    console.log("图片布局计算完成，区域数量:", node._customImageRects.length);
}

/**
 * 自动调整字体大小以适应宽度
 */
function getAdjustedFontSize(ctx, text, maxWidth, minFontSize = 8, maxFontSize = 12) {
    let fontSize = maxFontSize;
    ctx.font = `bold ${fontSize}px Arial`;
    
    while (ctx.measureText(text).width > maxWidth && fontSize > minFontSize) {
        fontSize--;
        ctx.font = `bold ${fontSize}px Arial`;
    }
    
    return fontSize;
}

/**
 * 显示图片的核心实现
 * @param {object} node - LiteGraph节点实例
 * @param {string[]} paths - 图片路径数组
 */
function showImages(node, paths) {
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
        node._customSizeInitialized = false;
        node._customPrevButtonRect = null;
        node._customNextButtonRect = null;
        node._customRestoreButtonRect = null;
        return [];
    }
    
    const validPaths = paths.filter(path => path.trim());
    console.log("有效路径数量:", validPaths.length);
    
    // 重新初始化数组
    node._customImgs = [];
    node._customImageFileNames = [];
    node._customImagePaths = validPaths; // 保存当前图片路径
    node._customFileNameRects = []; // 初始化文件名区域数组
    node._customClearButtonRects = []; // 初始化清除按钮区域数组
    
    // 初始化单图片显示状态
    node._customSingleImageMode = false;
    node._customFocusedImageIndex = -1;
    node._customSizeInitialized = false; // 标记节点大小未初始化
    
    validPaths.forEach((path, index) => {
        const img = new Image();
        node._customImgs.push(img);
        
        // 从相对路径中提取文件名
        const pathParts = path.split(/[\\\/]/);
        const fileName = pathParts[pathParts.length - 1];
        node._customImageFileNames.push(fileName);
        
        img.onload = () => { 
            console.log(`图片 ${index} 加载完成:`, path);
            app.graph.setDirtyCanvas(true, true); 
        };
        img.onerror = () => {
            console.error(`图片 ${index} 加载失败:`, path);
        };
        // 通过API获取图片URL
        img.src = api.apiURL(`/view?filename=${encodeURIComponent(path)}&type=input`);
    });
    
    // 计算图片布局
    calculateImageLayout(node, validPaths.length);
    
    console.log("图片显示设置完成，图片数量:", node._customImgs.length);
    return node._customImgs;
}

/**
 * 在Canvas上绘制图片
 * @param {object} node - LiteGraph节点实例
 * @param {CanvasRenderingContext2D} ctx - Canvas上下文
 */
function drawNodeImages(node, ctx) {
    if (!node._customImgs || !node._customImageRects) return;
    
    console.log("开始绘制图片，图片数量:", node._customImgs.length);
    
    ctx.save();
    
    for (let i = 0; i < node._customImgs.length && i < node._customImageRects.length; i++) {
        const img = node._customImgs[i];
        const rect = node._customImageRects[i];
        
        // 检查图片是否可见（单图片模式）
        if (rect.visible === false) {
            continue;
        }
        
        // 绘制图片背景
        ctx.fillStyle = '#2a2a2a';
        ctx.fillRect(rect.x, rect.y, rect.width, rect.height);
        
        // 绘制图片边框
        ctx.strokeStyle = '#666';
        ctx.lineWidth = 1;
        ctx.strokeRect(rect.x, rect.y, rect.width, rect.height);
        
        // 绘制图片到Canvas - 保持原始比例，向下偏移避免被文件名遮挡
        if (img.complete && img.naturalWidth > 0) {
            try {
                // 为文件名预留空间
                const titleHeight = 20;
                const imageRect = {
                    x: rect.x,
                    y: rect.y + titleHeight, // 向下偏移
                    width: rect.width,
                    height: rect.height - titleHeight // 减去文件名高度
                };
                
                // 计算图片的原始比例
                const imageAspectRatio = img.naturalWidth / img.naturalHeight;
                const rectAspectRatio = imageRect.width / imageRect.height;
                
                let drawWidth, drawHeight, drawX, drawY;
                
                if (imageAspectRatio > rectAspectRatio) {
                    // 图片更宽，以宽度为准
                    drawWidth = imageRect.width;
                    drawHeight = imageRect.width / imageAspectRatio;
                    drawX = imageRect.x;
                    drawY = imageRect.y + (imageRect.height - drawHeight) / 2;
                } else {
                    // 图片更高，以高度为准
                    drawHeight = imageRect.height;
                    drawWidth = imageRect.height * imageAspectRatio;
                    drawX = imageRect.x + (imageRect.width - drawWidth) / 2;
                    drawY = imageRect.y;
                }
                
                // 绘制图片，保持原始比例
                ctx.drawImage(img, drawX, drawY, drawWidth, drawHeight);
                
                // 在图片周围绘制边框，显示实际显示区域
                ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
        ctx.lineWidth = 1;
                ctx.strokeRect(drawX, drawY, drawWidth, drawHeight);
            } catch (e) {
                console.warn(`绘制图片失败: ${e.message}`);
            }
        }
        
        // 绘制图片标题 - 在顶部显示文件名
        ctx.textAlign = 'center';
        
        // 使用保存的文件名
        const fileName = node._customImageFileNames && node._customImageFileNames[i] ? node._customImageFileNames[i] : 'Unknown';
        
        // 在顶部绘制文件名背景
        ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
        ctx.fillRect(rect.x, rect.y, rect.width, 20);
        
        // 自动调整字体大小
        const maxTextWidth = rect.width - 10; // 留出边距
        const fontSize = getAdjustedFontSize(ctx, fileName, maxTextWidth);
        ctx.font = `bold ${fontSize}px Arial`;
        
        // 绘制文件名
        ctx.fillStyle = '#fff';
        ctx.fillText(fileName, rect.x + rect.width / 2, rect.y + 15);
        
        // 保存文件名区域信息，用于tooltip检测
        if (!node._customFileNameRects) {
            node._customFileNameRects = [];
        }
        node._customFileNameRects[i] = {
            x: rect.x,
            y: rect.y,
            width: rect.width,
            height: 20 // 文件名区域高度
        };
        
        // 绘制右上角清除按钮
        const buttonSize = 16;
        const buttonMargin = 5;
        const clearButtonX = rect.x + rect.width - buttonMargin - buttonSize;
        const clearButtonY = rect.y + buttonMargin;
        
        // 检查鼠标是否悬浮在清除按钮上
        const mouseInClearButton = node._customMouseX !== undefined && node._customMouseY !== undefined &&
            node._customMouseX >= clearButtonX && node._customMouseX <= clearButtonX + buttonSize &&
            node._customMouseY >= clearButtonY && node._customMouseY <= clearButtonY + buttonSize;
        
        // 绘制清除按钮背景（悬浮效果）
        ctx.fillStyle = mouseInClearButton ? 'rgba(255, 0, 0, 0.9)' : 'rgba(255, 0, 0, 0.7)';
        ctx.beginPath();
        ctx.arc(clearButtonX + buttonSize/2, clearButtonY + buttonSize/2, buttonSize/2, 0, 2 * Math.PI);
        ctx.fill();
        
        // 绘制清除按钮边框
        ctx.strokeStyle = mouseInClearButton ? 'rgba(255, 255, 255, 1)' : 'rgba(255, 255, 255, 0.8)';
        ctx.lineWidth = mouseInClearButton ? 2 : 1;
        ctx.stroke();
        
        // 绘制清除图标 (×)
        ctx.fillStyle = 'rgba(255, 255, 255, 1)';
        ctx.font = `${buttonSize - 4}px Arial`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText('×', clearButtonX + buttonSize/2, clearButtonY + buttonSize/2);
        
        // 保存清除按钮区域信息
        if (!node._customClearButtonRects) {
            node._customClearButtonRects = [];
        }
        node._customClearButtonRects[i] = {
            x: clearButtonX,
            y: clearButtonY,
            width: buttonSize,
            height: buttonSize
        };
        
        // 绘制点击指示器 - 只在鼠标悬浮时显示
        const centerX = rect.x + rect.width / 2;
        const centerY = rect.y + rect.height / 2;
        
        // 检查鼠标是否在图片区域内
        if (node._customMouseX !== undefined && node._customMouseY !== undefined) {
            const mouseInImage = node._customMouseX >= rect.x && node._customMouseX <= rect.x + rect.width &&
                               node._customMouseY >= rect.y && node._customMouseY <= rect.y + rect.height;
            
            if (mouseInImage) {
                // 绘制半透明背景圆形
                ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
                ctx.beginPath();
                ctx.arc(centerX, centerY, 18, 0, 2 * Math.PI);
                ctx.fill();
                
                // 绘制边框
                ctx.strokeStyle = 'rgba(255, 255, 255, 0.9)';
                ctx.lineWidth = 1.5;
                ctx.stroke();
                
                // 绘制放大图标
                ctx.fillStyle = 'rgba(255, 255, 255, 1)';
                ctx.font = '16px Arial';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText('🔍', centerX, centerY);
            }
        }
    }
    
    // 绘制控制按钮（只在单图片模式下显示）
    if (node._customSingleImageMode) {
        const buttonSize = 20;
        const buttonSpacing = 5;
        
        // 检查鼠标是否悬浮在按钮上
        const mouseInRestoreButton = node._customMouseX !== undefined && node._customMouseY !== undefined &&
            node._customMouseX >= node.size[0] - buttonSize - 10 && node._customMouseX <= node.size[0] - 10 &&
            node._customMouseY >= node.size[1] - buttonSize - 10 && node._customMouseY <= node.size[1] - 10;
        
        const mouseInPrevButton = node._customMouseX !== undefined && node._customMouseY !== undefined &&
            node._customMouseX >= node.size[0] - buttonSize * 2 - buttonSpacing - 10 && node._customMouseX <= node.size[0] - buttonSize - buttonSpacing - 10 &&
            node._customMouseY >= node.size[1] - buttonSize - 10 && node._customMouseY <= node.size[1] - 10;
        
        const mouseInNextButton = node._customMouseX !== undefined && node._customMouseY !== undefined &&
            node._customMouseX >= node.size[0] - buttonSize * 3 - buttonSpacing * 2 - 10 && node._customMouseX <= node.size[0] - buttonSize * 2 - buttonSpacing * 2 - 10 &&
            node._customMouseY >= node.size[1] - buttonSize - 10 && node._customMouseY <= node.size[1] - 10;
        
        // 绘制索引信息 (n/m) - 在上一个按钮的左边
        if (node._customImagePaths && node._customImagePaths.length > 1 && 
            node._customFocusedImageIndex >= 0 && node._customFocusedImageIndex < node._customImagePaths.length) {
            const currentIndex = node._customFocusedImageIndex + 1;
            const totalCount = node._customImagePaths.length;
            const indexText = `(${currentIndex}/${totalCount})`;
            
            // 设置文本样式
            ctx.fillStyle = 'rgba(255, 255, 255, 0.9)';
            ctx.font = '12px Arial';
            ctx.textAlign = 'right';
            ctx.textBaseline = 'middle';
            
            // 计算索引文本位置（在上一个按钮的左边）
            const indexX = node.size[0] - buttonSize * 3 - buttonSpacing * 2 - 15;
            const indexY = node.size[1] - buttonSize - 10 + buttonSize / 2;
            
            // 绘制索引文本
            ctx.fillText(indexText, indexX, indexY);
        }
        
        // 绘制上一个按钮 (‹)
        const prevButtonX = node.size[0] - buttonSize * 3 - buttonSpacing * 2 - 10;
        const prevButtonY = node.size[1] - buttonSize - 10;
        
        // 按钮背景（悬浮效果）
        ctx.fillStyle = mouseInPrevButton ? 'rgba(0, 0, 0, 0.9)' : 'rgba(0, 0, 0, 0.7)';
        ctx.fillRect(prevButtonX, prevButtonY, buttonSize, buttonSize);
        
        // 按钮边框
        ctx.strokeStyle = mouseInPrevButton ? 'rgba(255, 255, 255, 1)' : 'rgba(255, 255, 255, 0.8)';
        ctx.lineWidth = mouseInPrevButton ? 2 : 1;
        ctx.strokeRect(prevButtonX, prevButtonY, buttonSize, buttonSize);
        
        // 绘制‹符号
        ctx.fillStyle = 'rgba(255, 255, 255, 1)';
        ctx.font = `${buttonSize - 4}px Arial`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText('‹', prevButtonX + buttonSize / 2, prevButtonY + buttonSize / 2);
        
        // 绘制下一个按钮 (›)
        const nextButtonX = node.size[0] - buttonSize * 2 - buttonSpacing - 10;
        const nextButtonY = node.size[1] - buttonSize - 10;
        
        // 按钮背景（悬浮效果）
        ctx.fillStyle = mouseInNextButton ? 'rgba(0, 0, 0, 0.9)' : 'rgba(0, 0, 0, 0.7)';
        ctx.fillRect(nextButtonX, nextButtonY, buttonSize, buttonSize);
        
        // 按钮边框
        ctx.strokeStyle = mouseInNextButton ? 'rgba(255, 255, 255, 1)' : 'rgba(255, 255, 255, 0.8)';
        ctx.lineWidth = mouseInNextButton ? 2 : 1;
        ctx.strokeRect(nextButtonX, nextButtonY, buttonSize, buttonSize);
        
        // 绘制›符号
        ctx.fillStyle = 'rgba(255, 255, 255, 1)';
        ctx.fillText('›', nextButtonX + buttonSize / 2, nextButtonY + buttonSize / 2);
        
        // 绘制恢复按钮 (⭯)
        const restoreButtonX = node.size[0] - buttonSize - 10;
        const restoreButtonY = node.size[1] - buttonSize - 10;
        
        // 按钮背景（悬浮效果）
        ctx.fillStyle = mouseInRestoreButton ? 'rgba(0, 0, 0, 0.9)' : 'rgba(0, 0, 0, 0.7)';
        ctx.fillRect(restoreButtonX, restoreButtonY, buttonSize, buttonSize);
        
        // 按钮边框
        ctx.strokeStyle = mouseInRestoreButton ? 'rgba(255, 255, 255, 1)' : 'rgba(255, 255, 255, 0.8)';
        ctx.lineWidth = mouseInRestoreButton ? 2 : 1;
        ctx.strokeRect(restoreButtonX, restoreButtonY, buttonSize, buttonSize);
        
        // 绘制⭯符号
        ctx.fillStyle = 'rgba(255, 255, 255, 1)';
        ctx.fillText('⭯', restoreButtonX + buttonSize / 2, restoreButtonY + buttonSize / 2);
        
        // 保存按钮区域信息
        node._customPrevButtonRect = {
            x: prevButtonX,
            y: prevButtonY,
            width: buttonSize,
            height: buttonSize
        };
        node._customNextButtonRect = {
            x: nextButtonX,
            y: nextButtonY,
            width: buttonSize,
            height: buttonSize
        };
        node._customRestoreButtonRect = {
            x: restoreButtonX,
            y: restoreButtonY,
            width: buttonSize,
            height: buttonSize
        };
    } else {
        // 清除按钮区域信息
        node._customPrevButtonRect = null;
        node._customNextButtonRect = null;
        node._customRestoreButtonRect = null;
    }
    
    ctx.restore();
}

/**
 * 更新节点上的图片预览区域。
 * @param {object} node - LiteGraph节点实例。
 * @param {string[]} paths - 图片的相对路径数组。
 */
function updateImagePreviews(node, paths) {
    console.log("更新图片预览，路径:", paths);
    
    // 清理旧的图片数据
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
        node.computeSize();
        app.graph.setDirtyCanvas(true, true);
        return;
    }
    
    // 加载图片
    showImages(node, paths);
    
    // 更新节点大小
    node.computeSize();
    app.graph.setDirtyCanvas(true, true);
    
    console.log("图片预览更新完成");
}

/**
 * 处理图片数据更新的核心函数
 * @param {string[]} imagePaths - 图片路径数组
 */
function populate(imagePaths) {
    console.log("收到新的图片数据，开始更新显示...");
    console.log("新图片路径:", imagePaths);
    console.log("节点当前尺寸:", this.size);
    
    // 检查是否有数据变化
    const oldPaths = this._customImagePaths || [];
    const newPaths = imagePaths || [];
    
    // 比较新旧数据是否相同
    const hasChanged = oldPaths.length !== newPaths.length || 
                      oldPaths.some((oldPath, index) => oldPath !== newPaths[index]);
    
    if (!hasChanged) {
        console.log("图片数据没有变化，跳过更新");
        return;
    }
    
    console.log("检测到图片数据变化，开始清除旧数据并加载新数据");
    
    // 保存新的图片路径
    this._customImagePaths = imagePaths;
    
    // 显示图片
    showImages(this, imagePaths);
    
    // 使用Symbol来确保绘制方法不会被覆盖
    const CUSTOM_DRAW_SYMBOL = Symbol('customDrawForeground');
    
    // 确保只设置一次绘制方法，避免重复设置
    if (!this[CUSTOM_DRAW_SYMBOL]) {
        console.log("设置自定义绘制方法");
        
        // 重写节点的绘制方法
        const originalOnDrawForeground = this.onDrawForeground;
        
        // 创建一个包装函数，确保我们的绘制逻辑始终被执行
        const customDrawForeground = function(ctx) {
            // 首先调用原始绘制方法
            if (originalOnDrawForeground) {
                originalOnDrawForeground.call(this, ctx);
            }
            
            // 然后绘制我们的自定义内容
            if (this._customImgs && this._customImageRects) {
                console.log("执行自定义绘制，图片数量:", this._customImgs.length);
                drawNodeImages(this, ctx);
            }
        };
        
        // 设置绘制方法
        this.onDrawForeground = customDrawForeground;
        
        // 标记已设置
        this[CUSTOM_DRAW_SYMBOL] = true;
        
        // 添加一个检查机制，定期验证绘制方法是否被覆盖
        const checkDrawMethod = () => {
            if (this.onDrawForeground !== customDrawForeground) {
                console.warn("检测到绘制方法被覆盖，重新设置");
                this.onDrawForeground = customDrawForeground;
            }
        };
        
        // 每5秒检查一次
        this._customDrawCheckInterval = setInterval(checkDrawMethod, 5000);
        
        console.log("自定义绘制方法设置完成");
    } else {
        console.log("自定义绘制方法已存在，跳过设置");
    }
    
    // 添加鼠标事件处理
    const originalOnMouseDown = this.onMouseDown;
    const originalOnMouseMove = this.onMouseMove;
    
    console.log("设置鼠标事件处理器");
    
    // 跟踪鼠标位置
    this.onMouseMove = function(e) {
        if (originalOnMouseMove) {
            originalOnMouseMove.call(this, e);
        }
        
        // 保存鼠标位置用于悬浮检测
        this._customMouseX = e.canvasX - this.pos[0];
        this._customMouseY = e.canvasY - this.pos[1];
        
        // 处理悬浮tooltip - 只在悬浮在文件名区域时显示
        let tooltipShown = false;
        if (this._customFileNameRects && this._customFileNameRects.length > 0) {
            for (let i = 0; i < this._customFileNameRects.length; i++) {
                const fileNameRect = this._customFileNameRects[i];
                
                // 计算文件名区域在Canvas中的绝对坐标
                const nodePos = this.pos;
                const absFileNameX = nodePos[0] + fileNameRect.x;
                const absFileNameY = nodePos[1] + fileNameRect.y;
                const absFileNameWidth = fileNameRect.width;
                const absFileNameHeight = fileNameRect.height;
                
                // 检查鼠标是否在文件名区域内
                const mouseInFileName = e.canvasX >= absFileNameX && e.canvasX <= absFileNameX + absFileNameWidth &&
                                      e.canvasY >= absFileNameY && e.canvasY <= absFileNameY + absFileNameHeight;
                
                if (mouseInFileName && this._customImagePaths && this._customImagePaths[i]) {
                    console.log(`鼠标悬浮在文件名区域 ${i}，显示tooltip`);
                    // 显示tooltip
                    this.showTooltip(e, i);
                    tooltipShown = true;
                    break;
                }
            }
        }
        
        // 如果没有悬浮在文件名区域，隐藏tooltip
        if (!tooltipShown) {
            this.hideTooltip();
        }
                
        // 触发重绘以更新悬浮状态
        app.graph.setDirtyCanvas(true, false);
    };
            
    // 鼠标离开时清除位置
    const originalOnMouseLeave = this.onMouseLeave;
    this.onMouseLeave = function(e) {
        if (originalOnMouseLeave) {
            originalOnMouseLeave.call(this, e);
        }
        
        // 清除鼠标位置
        this._customMouseX = undefined;
        this._customMouseY = undefined;

        // 隐藏tooltip
        this.hideTooltip();
                
        // 触发重绘以隐藏指示器
        app.graph.setDirtyCanvas(true, false);
    };
    
    this.onMouseDown = function(e) {
        console.log("onMouseDown 被调用", e);
        console.log("节点信息:", this.id, this.type, this.size);
        console.log("图片区域:", this._customImageRects);
        
        // 获取节点的Canvas坐标
        const nodePos = this.pos;

        // 检查是否点击控制按钮（单图片模式下）
        if (this._customSingleImageMode) {
            // 检查点击上一个按钮 (‹)
            if (this._customPrevButtonRect) {
                const absPrevButtonX = nodePos[0] + this._customPrevButtonRect.x;
                const absPrevButtonY = nodePos[1] + this._customPrevButtonRect.y;
                const absPrevButtonWidth = this._customPrevButtonRect.width;
                const absPrevButtonHeight = this._customPrevButtonRect.height;
                
                if (e.canvasX >= absPrevButtonX && e.canvasX <= absPrevButtonX + absPrevButtonWidth &&
                    e.canvasY >= absPrevButtonY && e.canvasY <= absPrevButtonY + absPrevButtonHeight) {
                    
                    console.log("点击上一个按钮");
                    
                    // 阻止事件冒泡
                    e.preventDefault();
                    e.stopPropagation();
                    
                    // 切换到上一个图片
                    if (this._customImagePaths && this._customImagePaths.length > 0) {
                        this._customFocusedImageIndex = (this._customFocusedImageIndex - 1 + this._customImagePaths.length) % this._customImagePaths.length;
                        console.log(`切换到上一个图片，当前索引: ${this._customFocusedImageIndex}`);
                        
                        // 重新计算布局
                        calculateImageLayout(this, this._customImagePaths.length);
                        
                        // 触发重绘
                        app.graph.setDirtyCanvas(true, false);
                    }
                    
                    return true;
                }
            }
            
            // 检查点击下一个按钮 (›)
            if (this._customNextButtonRect) {
                const absNextButtonX = nodePos[0] + this._customNextButtonRect.x;
                const absNextButtonY = nodePos[1] + this._customNextButtonRect.y;
                const absNextButtonWidth = this._customNextButtonRect.width;
                const absNextButtonHeight = this._customNextButtonRect.height;
                
                if (e.canvasX >= absNextButtonX && e.canvasX <= absNextButtonX + absNextButtonWidth &&
                    e.canvasY >= absNextButtonY && e.canvasY <= absNextButtonY + absNextButtonHeight) {
                    
                    console.log("点击下一个按钮");
                    
                    // 阻止事件冒泡
                    e.preventDefault();
                    e.stopPropagation();
                    
                    // 切换到下一个图片
                    if (this._customImagePaths && this._customImagePaths.length > 0) {
                        this._customFocusedImageIndex = (this._customFocusedImageIndex + 1) % this._customImagePaths.length;
                        console.log(`切换到下一个图片，当前索引: ${this._customFocusedImageIndex}`);
                        
                        // 重新计算布局
                        calculateImageLayout(this, this._customImagePaths.length);
                        
                        // 触发重绘
                        app.graph.setDirtyCanvas(true, false);
                    }
                    
                    return true;
                }
            }
            
            // 检查点击恢复按钮 (⭯)
            if (this._customRestoreButtonRect) {
                const absRestoreButtonX = nodePos[0] + this._customRestoreButtonRect.x;
                const absRestoreButtonY = nodePos[1] + this._customRestoreButtonRect.y;
                const absRestoreButtonWidth = this._customRestoreButtonRect.width;
                const absRestoreButtonHeight = this._customRestoreButtonRect.height;
                
                if (e.canvasX >= absRestoreButtonX && e.canvasX <= absRestoreButtonX + absRestoreButtonWidth &&
                    e.canvasY >= absRestoreButtonY && e.canvasY <= absRestoreButtonY + absRestoreButtonHeight) {
                    
                    console.log("点击恢复按钮，退出单图片模式");
                    
                    // 阻止事件冒泡
                    e.preventDefault();
                    e.stopPropagation();
                    
                    // 退出单图片模式
                    this._customSingleImageMode = false;
                    this._customFocusedImageIndex = -1;
                    
                    // 重新计算布局
                    if (this._customImagePaths && this._customImagePaths.length > 0) {
                        calculateImageLayout(this, this._customImagePaths.length);
                    }
                    
                    // 触发重绘
                    app.graph.setDirtyCanvas(true, false);
                    
                    return true;
                }
            }
        }
                
        // 检查是否点击清除按钮
        if (this._customClearButtonRects && this._customClearButtonRects.length > 0) {
            for (let i = 0; i < this._customClearButtonRects.length; i++) {
                const clearRect = this._customClearButtonRects[i];

                // 检查图片是否可见
                if (this._customImageRects && this._customImageRects[i] && this._customImageRects[i].visible === false) {
                    continue;
                }
                        
                // 计算清除按钮在Canvas中的绝对坐标
                const absClearButtonX = nodePos[0] + clearRect.x;
                const absClearButtonY = nodePos[1] + clearRect.y;
                const absClearButtonWidth = clearRect.width;
                const absClearButtonHeight = clearRect.height;
                
                if (e.canvasX >= absClearButtonX && e.canvasX <= absClearButtonX + absClearButtonWidth &&
                    e.canvasY >= absClearButtonY && e.canvasY <= absClearButtonY + absClearButtonHeight) {
                    
                    console.log(`点击清除按钮，图片索引: ${i}`);
                    
                    // 阻止事件冒泡
                    e.preventDefault();
                    e.stopPropagation();
                    
                    // 执行清除操作
                    this.clearImageWithConfirmation(i);
                    
                    return true;
                }
            }
        }
                
        // 检查鼠标是否在图片框内
        if (this._customImageRects && this._customImageRects.length > 0) {
            console.log("检查图片区域点击", this._customImageRects.length, "个图片区域");
            
            for (let i = 0; i < this._customImageRects.length; i++) {
                const rect = this._customImageRects[i];
                
                // 检查图片是否可见
                if (rect.visible === false) {
                    continue;
                }
                
                // 计算图片区域在Canvas中的绝对坐标
                const absRectX = nodePos[0] + rect.x;
                const absRectY = nodePos[1] + rect.y;
                const absRectWidth = rect.width;
                const absRectHeight = rect.height;
                
                console.log(`检查图片 ${i}:`, {
                    rect: rect,
                    绝对坐标: {x: absRectX, y: absRectY, width: absRectWidth, height: absRectHeight},
                    鼠标位置: {x: e.canvasX, y: e.canvasY}
                });
                
                // 检查鼠标是否在图片区域内
                if (e.canvasX >= absRectX && e.canvasX <= absRectX + absRectWidth &&
                    e.canvasY >= absRectY && e.canvasY <= absRectY + absRectHeight) {
                    
                    console.log(`鼠标在图片 ${i} 区域内`);
                    
                    // 检查是否点击在放大指示器区域（中心区域）
                    const centerX = absRectX + absRectWidth / 2;
                    const centerY = absRectY + absRectHeight / 2;
                    const indicatorSize = 25; // 指示器的大小
                    
                    const inCenter = e.canvasX >= centerX - indicatorSize/2 && 
                                   e.canvasX <= centerX + indicatorSize/2 &&
                                   e.canvasY >= centerY - indicatorSize/2 && 
                                   e.canvasY <= centerY + indicatorSize/2;
                    
                    if (inCenter) {
                        console.log(`点击在放大指示器上，图片 ${i}`);
                        
                        // 阻止事件冒泡，避免触发节点选择
                        e.preventDefault();
                        e.stopPropagation();
                        
                        // 点击在放大指示器上，进入单图片模式
                        if (!this._customSingleImageMode) {
                            console.log(`进入单图片模式，聚焦图片 ${i}`);
                            this._customSingleImageMode = true;
                            this._customFocusedImageIndex = i;
                            
                            // 重新计算布局
                            if (this._customImagePaths && this._customImagePaths.length > 0) {
                                calculateImageLayout(this, this._customImagePaths.length);
                            }
                            
                            // 触发重绘
                            app.graph.setDirtyCanvas(true, false);
                        }
                        
                        // 返回true表示事件已处理
                        return true;
                    } else {
                        console.log(`点击在图片 ${i} 其他区域`);
                        
                        // 阻止事件冒泡，避免触发节点选择
                        e.preventDefault();
                        e.stopPropagation();
                        
                        // 如果不在单图片模式，进入单图片模式
                        if (!this._customSingleImageMode) {
                            console.log(`进入单图片模式，聚焦图片 ${i}`);
                            this._customSingleImageMode = true;
                            this._customFocusedImageIndex = i;
                            
                            // 重新计算布局
                            if (this._customImagePaths && this._customImagePaths.length > 0) {
                                calculateImageLayout(this, this._customImagePaths.length);
                            }
                            
                            // 触发重绘
                            app.graph.setDirtyCanvas(true, false);
                        }
                        
                        // 返回true表示事件已处理
                        return true;
                    }
                }
            }
        }
        
        // 如果没有处理图片区域点击，调用原始事件处理
        if (originalOnMouseDown) {
            return originalOnMouseDown.call(this, e);
        }
        
        return false;
    };
    
    // 重写节点的resize方法，当大小改变时重新计算布局
    const originalOnResize = this.onResize;
    this.onResize = function(size) {
        if (originalOnResize) {
            originalOnResize.call(this, size);
        }
        console.log("节点大小改变，重新计算布局:", size);
        
        // 重新计算图片布局，但不调整节点大小
        if (this._customImagePaths && this._customImagePaths.length > 0) {
            // 临时保存当前大小
            const currentSize = [this.size[0], this.size[1]];
            
            // 计算布局但不调整大小
            calculateImageLayout(this, this._customImagePaths.length);
            
            // 恢复原始大小，避免递归
            this.size[0] = currentSize[0];
            this.size[1] = currentSize[1];
        }
    };
    
    // 添加tooltip管理方法
    this.showTooltip = function(e, imageIndex) {
        // 如果已经有tooltip，先移除
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
            
            // 获取图片的原始尺寸信息
            const img = this._customImgs[imageIndex];
            let sizeInfo = '';
            if (img && img.naturalWidth && img.naturalHeight) {
                sizeInfo = ` (${img.naturalWidth}x${img.naturalHeight})`;
            }
            
            // 添加索引信息到tooltip
            let indexInfo = '';
            if (this._customImagePaths && this._customImagePaths.length > 1) {
                const currentIndex = imageIndex + 1;
                const totalCount = this._customImagePaths.length;
                indexInfo = ` [${currentIndex}/${totalCount}]`;
            }
            
            tooltip.textContent = `相对路径: ${this._customImagePaths[imageIndex]}${sizeInfo}${indexInfo}`;
            document.body.appendChild(tooltip);
            
            // 设置tooltip位置，确保不超出屏幕边界
            const tooltipRect = tooltip.getBoundingClientRect();
            let left = e.clientX + 10;
            let top = e.clientY - 30;
            
            // 检查右边界
            if (left + tooltipRect.width > window.innerWidth) {
                left = e.clientX - tooltipRect.width - 10;
            }
            
            // 检查下边界
            if (top + tooltipRect.height > window.innerHeight) {
                top = e.clientY - tooltipRect.height - 10;
            }
            
            tooltip.style.left = left + 'px';
            tooltip.style.top = top + 'px';
        }
    };
    
    this.hideTooltip = function() {
        const existingTooltip = document.getElementById('image-tooltip-' + this.id);
        if (existingTooltip) {
            existingTooltip.remove();
        }
    };
    
    // 延迟触发重绘，确保布局计算完成
    setTimeout(() => {
        console.log("延迟后的节点尺寸:", this.size);
        console.log("图片区域信息:", this._customImageRects);
        app.graph.setDirtyCanvas(true, false);
    }, 100);
}

/**
 * 清除图片的确认对话框
 * @param {number} imageIndex - 要清除的图片索引
 */
function clearImageWithConfirmation(imageIndex) {
    if (!this._customImagePaths || imageIndex < 0 || imageIndex >= this._customImagePaths.length) {
                    console.error("无效的图片索引:", imageIndex);
                    return;
                }
                
                // 创建确认对话框
                const confirmDialog = document.createElement('div');
                confirmDialog.id = 'clear-confirm-dialog-' + this.id;
                confirmDialog.style.cssText = `
                    position: fixed;
                    top: 50%;
                    left: 50%;
                    transform: translate(-50%, -50%);
                    background: #2a2a2a;
                    border: 2px solid #666;
                    border-radius: 8px;
                    padding: 20px;
                    z-index: 10001;
                    max-width: 400px;
                    color: white;
                    font-family: Arial, sans-serif;
                `;
                
                // 构建确认消息
                let confirmMessage = `<h3 style="margin: 0 0 15px 0; color: #ff6b6b;">⚠️ 确认清除图片</h3>`;
                confirmMessage += `<p style="margin: 0 0 20px 0;">确定要清除这张图片的预览和路径吗？</p>`;
                confirmMessage += `<p style="margin: 0 0 20px 0; color: #ff6b6b;"><strong>此操作不可撤销！</strong></p>`;
                
                // 添加按钮
                confirmMessage += `
                    <div style="display: flex; gap: 10px; justify-content: flex-end;">
                        <button id="cancel-clear-${this.id}" style="
                            padding: 8px 16px;
                            background: #666;
                            color: white;
                            border: none;
                            border-radius: 4px;
                            cursor: pointer;
                        ">取消</button>
                        <button id="confirm-clear-${this.id}" style="
                            padding: 8px 16px;
                            background: #ff6b6b;
                            color: white;
                            border: none;
                            border-radius: 4px;
                            cursor: pointer;
                        ">确认清除</button>
                    </div>
                `;
                
                confirmDialog.innerHTML = confirmMessage;
                document.body.appendChild(confirmDialog);
                
                // 添加背景遮罩
                const overlay = document.createElement('div');
                overlay.id = 'clear-overlay-' + this.id;
                overlay.style.cssText = `
                    position: fixed;
                    top: 0;
                    left: 0;
                    width: 100%;
                    height: 100%;
                    background: rgba(0, 0, 0, 0.5);
                    z-index: 10000;
                `;
                document.body.appendChild(overlay);
                
                // 绑定按钮事件
                document.getElementById(`cancel-clear-${this.id}`).onclick = () => {
                    this.removeClearDialog();
                };
                
                document.getElementById(`confirm-clear-${this.id}`).onclick = () => {
                    this.removeClearDialog();
                    this.executeClear(imageIndex);
                };
                
                // 点击遮罩关闭对话框
                overlay.onclick = () => {
                    this.removeClearDialog();
                };
}

/**
 * 移除清除确认对话框
 */
function removeClearDialog() {
                const dialog = document.getElementById('clear-confirm-dialog-' + this.id);
                const overlay = document.getElementById('clear-overlay-' + this.id);
                if (dialog) dialog.remove();
                if (overlay) overlay.remove();
}

/**
 * 执行清除操作
 * @param {number} imageIndex - 要清除的图片索引
 */
function executeClear(imageIndex) {
                console.log(`开始清除图片 ${imageIndex}`);
                
                // 获取当前的图片路径
                const imagePathsWidget = this.widgets.find(w => w.name === "image_paths");
                if (!imagePathsWidget || !imagePathsWidget.value) {
                    console.log("没有图片路径数据");
                    return;
                }
                
                const currentPaths = imagePathsWidget.value.split(',').filter(path => path.trim());
                if (imageIndex >= currentPaths.length) {
                    console.error("图片索引超出范围");
                    return;
                }
                
                // 从路径数组中移除指定索引的路径
                currentPaths.splice(imageIndex, 1);
                
                // 更新widget的值
                imagePathsWidget.value = currentPaths.join(',');
                
                // 更新预览
    populate.call(this, currentPaths);
                
                console.log(`✅ 成功清除图片 ${imageIndex}`);
                
                // 显示清除成功提示
                this.showClearResult(true);
}

/**
 * 显示清除结果
 * @param {boolean} success - 是否成功
 */
function showClearResult(success) {
                const resultDialog = document.createElement('div');
                resultDialog.id = 'clear-result-dialog-' + this.id;
                resultDialog.style.cssText = `
                    position: fixed;
                    top: 50%;
                    left: 50%;
                    transform: translate(-50%, -50%);
                    background: #2a2a2a;
                    border: 2px solid #666;
                    border-radius: 8px;
                    padding: 20px;
                    z-index: 10001;
                    max-width: 300px;
                    color: white;
                    font-family: Arial, sans-serif;
                `;
                
                const resultMessage = success ? 
                    `<h3 style="margin: 0 0 15px 0; color: #4CAF50;">✅ 清除成功</h3>
                     <p style="margin: 0 0 20px 0;">图片已从预览和路径中移除</p>` :
                    `<h3 style="margin: 0 0 15px 0; color: #ff6b6b;">❌ 清除失败</h3>
                     <p style="margin: 0 0 20px 0;">清除操作失败，请重试</p>`;
                
                resultDialog.innerHTML = resultMessage + `
                    <div style="display: flex; gap: 10px; justify-content: flex-end;">
                        <button id="close-clear-result-${this.id}" style="
                            padding: 8px 16px;
                            background: #666;
                            color: white;
                            border: none;
                            border-radius: 4px;
                            cursor: pointer;
                        ">关闭</button>
                    </div>
                `;
                
                document.body.appendChild(resultDialog);
                
                // 绑定关闭按钮事件
                document.getElementById(`close-clear-result-${this.id}`).onclick = () => {
                    this.removeClearResultDialog();
                };
                
                // 2秒后自动关闭
                setTimeout(() => {
                    this.removeClearResultDialog();
                }, 2000);
}

/**
 * 移除清除结果对话框
 */
function removeClearResultDialog() {
                const dialog = document.getElementById('clear-result-dialog-' + this.id);
                if (dialog) dialog.remove();
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
                                populate.call(node, []);
                                // 将所有成功上传的路径合并
                                pathWidget.value = allPaths.join(',');
                                triggerWidget.value = (triggerWidget.value || 0) + 1;
                                populate.call(node, allPaths);
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
                    populate.call(this, imagePathsWidget.value.split(','));
                }
            });
            
            // 添加鼠标事件处理
            chainCallback(nodeType.prototype, "onMouseMove", function(e) {
                // 保存鼠标位置用于悬浮检测
                this._customMouseX = e.canvasX - this.pos[0];
                this._customMouseY = e.canvasY - this.pos[1];
                
                // 触发重绘以更新悬浮状态
                app.graph.setDirtyCanvas(true, false);
            });
            
            // 鼠标离开时清除位置
            chainCallback(nodeType.prototype, "onMouseLeave", function(e) {
                // 清除鼠标位置
                this._customMouseX = undefined;
                this._customMouseY = undefined;
                
                // 触发重绘以隐藏指示器
                app.graph.setDirtyCanvas(true, false);
            });
            
            // 处理鼠标点击事件
            chainCallback(nodeType.prototype, "onMouseDown", function(e) {
                // 获取节点的Canvas坐标
                const nodePos = this.pos;
                
                // 检查是否点击清除按钮
                if (this._customClearButtonRects && this._customClearButtonRects.length > 0) {
                    for (let i = 0; i < this._customClearButtonRects.length; i++) {
                        const clearRect = this._customClearButtonRects[i];
                        
                        // 计算清除按钮在Canvas中的绝对坐标
                        const absClearButtonX = nodePos[0] + clearRect.x;
                        const absClearButtonY = nodePos[1] + clearRect.y;
                        const absClearButtonWidth = clearRect.width;
                        const absClearButtonHeight = clearRect.height;
                        
                        if (e.canvasX >= absClearButtonX && e.canvasX <= absClearButtonX + absClearButtonWidth &&
                            e.canvasY >= absClearButtonY && e.canvasY <= absClearButtonY + absClearButtonHeight) {
                            
                            console.log(`点击清除按钮，图片索引: ${i}`);
                            
                            // 阻止事件冒泡
                            e.preventDefault();
                            e.stopPropagation();
                            
                            // 执行清除操作
                            clearImageWithConfirmation.call(this, i);
                            
                            return true;
                        }
                    }
                }
                
                return false;
            });
            
            // 添加清除图片的方法到节点原型
            nodeType.prototype.clearImageWithConfirmation = clearImageWithConfirmation;
            nodeType.prototype.removeClearDialog = removeClearDialog;
            nodeType.prototype.executeClear = executeClear;
            nodeType.prototype.showClearResult = showClearResult;
            nodeType.prototype.removeClearResultDialog = removeClearResultDialog;
            
            // 添加节点销毁时的清理逻辑
            chainCallback(nodeType.prototype, "onRemoved", function() {
                // 清理清除相关的对话框
                if (this.removeClearDialog) {
                    this.removeClearDialog();
                }
                if (this.removeClearResultDialog) {
                    this.removeClearResultDialog();
                }
                
                // 清理定时器
                if (this._customDrawCheckInterval) {
                    clearInterval(this._customDrawCheckInterval);
                    this._customDrawCheckInterval = null;
                }
                
                // 清理自定义属性
                this._customImgs = null;
                this._customImageRects = null;
                this._customClearButtonRects = null;
                this._customImageFileNames = null;
                this._customImagePaths = null;
                this._customFileNameRects = null;
                this._customMouseX = null;
                this._customMouseY = null;
                
                console.log("节点清理完成");
            });
        }
    },
}); 