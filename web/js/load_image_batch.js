// 在脚本顶部添加日志，以便在浏览器控制台中确认脚本是否被加载
console.log("Loading custom node: A_my_nodes/web/js/load_image_batch.js");

import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";
import { showImageLightbox } from "./lightbox_preview.js";

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
        node._customVisibleIndices = [];
        return;
    }
    
    const containerWidth = node.size[0];
    const containerHeight = node.size[1];
    const GAP = 3;
    const PADDING = 8;
    
    // 为顶部输入控件和图片标题预留更多空间
    const TOP_MARGIN = 185; // 再向下腾挪空间，容纳新增控件（如"应用透明到图像"开关）
    const TITLE_HEIGHT = 25; // 图片标题的高度
    const BOTTOM_CONTROLS_HEIGHT = 25; // 底部控制按钮的高度
    
    const availableWidth = containerWidth - (PADDING * 2);
    const availableHeight = containerHeight - (PADDING * 2) - TOP_MARGIN - TITLE_HEIGHT - BOTTOM_CONTROLS_HEIGHT;
    
    const visibleIndices = [];
    for (let i = 0; i < imageCount; i++) {
        let inRange = true;
        const totalCount = imageCount;
        const hasRange = Number.isInteger(node._customViewStartIndex) || Number.isInteger(node._customViewEndIndex);
        if (hasRange && totalCount > 0) {
            let startIndex = Number.isInteger(node._customViewStartIndex) ? node._customViewStartIndex : 1;
            let endIndex = Number.isInteger(node._customViewEndIndex) ? node._customViewEndIndex : totalCount;
            if (startIndex < 1) startIndex = 1;
            if (endIndex > totalCount) endIndex = totalCount;
            if (startIndex > endIndex) {
                const t = startIndex;
                startIndex = endIndex;
                endIndex = t;
            }
            const oneBased = i + 1;
            if (oneBased < startIndex || oneBased > endIndex) inRange = false;
        }
        let selectedOk = true;
        if (node._customShowOnlySelected) {
            selectedOk = node._customSelectedImages && node._customSelectedImages[i];
        }
        if (inRange && selectedOk) {
            visibleIndices.push(i);
        }
    }
    node._customVisibleIndices = visibleIndices;
    const effectiveCount = node._customSingleImageMode ? imageCount : visibleIndices.length;
    
    if (!node._customSingleImageMode && effectiveCount === 0) {
        node._customImageRects = [];
        for (let i = 0; i < imageCount; i++) {
            node._customImageRects.push({
                x: 0,
                y: 0,
                width: 0,
                height: 0,
                visible: false
            });
        }
        console.log("没有可见图片，布局为空");
        return;
    }
    
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
        
        console.log("单图片模式，保持节点大小:", node.size);
    } else {
        // 多图片模式：计算最佳网格
        let bestRows = 1;
        let bestCols = 1;
        let bestSize = 0;
        
        for (let rows = 1; rows <= effectiveCount; rows++) {
            const cols = Math.ceil(effectiveCount / rows);
            const sizeFromWidth = (availableWidth - (GAP * (cols - 1))) / cols;
            const sizeFromHeight = (availableHeight - (GAP * (rows - 1))) / rows;
            const size = Math.min(sizeFromWidth, sizeFromHeight);
            
            if (size > bestSize) {
                bestSize = size;
                bestRows = rows;
                bestCols = cols;
            }
        }
        
        node._customImageRects = [];
        for (let i = 0; i < imageCount; i++) {
            node._customImageRects.push({
                x: 0,
                y: 0,
                width: 0,
                height: 0,
                visible: false
            });
        }
        const totalGridWidth = bestCols * bestSize + GAP * Math.max(0, bestCols - 1);
        const leftX = PADDING + Math.max(0, (availableWidth - totalGridWidth) / 2);
        for (let visibleIndex = 0; visibleIndex < effectiveCount; visibleIndex++) {
            const imageIndex = visibleIndices[visibleIndex];
            const row = Math.floor(visibleIndex / bestCols);
            const col = visibleIndex % bestCols;
            const x = leftX + col * (bestSize + GAP);
            const y = PADDING + TOP_MARGIN + row * (bestSize + GAP);
            
            node._customImageRects[imageIndex] = {
                x: x,
                y: y,
                width: bestSize,
                height: bestSize,
                visible: true
            };
        }
        
        console.log("多图片模式，保持节点大小:", node.size);
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
 * 根据选择状态更新widget的值
 * @param {object} node - LiteGraph节点实例
 */
function updateWidgetValue(node) {
    if (!node._customImagePaths || !node._customSelectedImages) {
        return;
    }
    
    // 获取选中的图片路径
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
    
    // 重新初始化数组
    node._customImgs = [];
    node._customImageFileNames = [];
    node._customImagePaths = validPaths; // 保存当前图片路径
    node._customFileNameRects = []; // 初始化文件名区域数组
    node._customClearButtonRects = []; // 初始化清除按钮区域数组
    node._customCheckboxRects = []; // 初始化复选框区域数组
    
    const imagePathUseWidget = node.widgets.find(w => w.name === "image_path_use");
    const selectedList = (imagePathUseWidget && imagePathUseWidget.value) ? imagePathUseWidget.value.split(',').filter(s => s.trim()) : [];
    if (!selectedList.length) {
        node._customSelectedImages = new Array(validPaths.length).fill(true);
    } else {
        node._customSelectedImages = validPaths.map(p => selectedList.includes(p));
    }
    
    // 初始化单图片显示状态
    node._customSingleImageMode = false;
    node._customFocusedImageIndex = -1;
    
    validPaths.forEach((path, index) => {
        const img = new Image();
        node._customImgs.push(img);
        
        // 解析路径，处理 [input] 等后缀
        // Parse path to handle [input], [output], [temp] suffixes
        // Example: "clipspace/image.png [input]" -> filename: "image.png", subfolder: "clipspace", type: "input"
        let filename = path;
        let type = 'input';
        let subfolder = '';

        // 匹配 "filename [type]" 格式
        const typeMatch = path.match(/^(.*)\s+\[(input|output|temp)\]$/);
        if (typeMatch) {
            filename = typeMatch[1];
            type = typeMatch[2];
        }

        // 解析子文件夹
        const lastSlash = filename.lastIndexOf('/');
        const lastBackslash = filename.lastIndexOf('\\');
        const splitIndex = Math.max(lastSlash, lastBackslash);
        
        if (splitIndex !== -1) {
            subfolder = filename.substring(0, splitIndex);
            filename = filename.substring(splitIndex + 1);
        }

        // 保存文件名用于显示 (去除路径和后缀)
        node._customImageFileNames.push(filename);
        
        img.onload = () => { 
            console.log(`图片 ${index} 加载完成:`, path);
            app.graph.setDirtyCanvas(true, true); 
        };
        img.onerror = () => {
            console.error(`图片 ${index} 加载失败:`, path);
        };
        
        // 构建 API URL
        // 使用 URLSearchParams 确保参数正确编码
        const params = new URLSearchParams({
            filename: filename,
            type: type,
            subfolder: subfolder
        });
        
        // 通过API获取图片URL
        img.src = api.apiURL(`/view?${params.toString()}`);
    });
    
    // 计算图片布局
    calculateImageLayout(node, validPaths.length);
    
    // 更新widget的值以反映选择状态
    updateWidgetValue(node);
    
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
    
    // 绘制图片（已优化，避免频繁调用）
    // console.log("开始绘制图片，图片数量:", node._customImgs.length);
    
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
        
        // 绘制图片到Canvas - 保持原始比例
        if (img.complete && img.naturalWidth > 0) {
            try {
                // 计算图片的原始比例
                const imageAspectRatio = img.naturalWidth / img.naturalHeight;
                const rectAspectRatio = rect.width / rect.height;
                
                let drawWidth, drawHeight, drawX, drawY;
                
                if (imageAspectRatio > rectAspectRatio) {
                    // 图片更宽，以宽度为准
                    drawWidth = rect.width;
                    drawHeight = rect.width / imageAspectRatio;
                    drawX = rect.x;
                    drawY = rect.y + (rect.height - drawHeight) / 2;
                } else {
                    // 图片更高，以高度为准
                    drawHeight = rect.height;
                    drawWidth = rect.height * imageAspectRatio;
                    drawX = rect.x + (rect.width - drawWidth) / 2;
                    drawY = rect.y;
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
        
        const clickW = rect.width / 3;
        const clickH = rect.height / 3;
        const clickX = rect.x;
        const clickY = rect.y + rect.height - clickH;
        const mouseInCheckbox = node._customMouseX !== undefined && node._customMouseY !== undefined &&
            node._customMouseX >= clickX && node._customMouseX <= clickX + clickW &&
            node._customMouseY >= clickY && node._customMouseY <= clickY + clickH;
        const iconSize = Math.max(16, Math.min(20, rect.width * 0.08));
        const iconMargin = 6;
        const iconX = rect.x + iconMargin;
        const iconY = rect.y + rect.height - iconMargin - iconSize;
        ctx.fillStyle = mouseInCheckbox ? 'rgba(255, 255, 255, 0.18)' : 'rgba(255, 255, 255, 0.12)';
        ctx.fillRect(iconX, iconY, iconSize, iconSize);
        ctx.strokeStyle = mouseInCheckbox ? 'rgba(255, 255, 255, 0.95)' : 'rgba(255, 255, 255, 0.75)';
        ctx.lineWidth = mouseInCheckbox ? 2 : 1;
        ctx.strokeRect(iconX, iconY, iconSize, iconSize);
        if (node._customSelectedImages && node._customSelectedImages[i]) {
            ctx.strokeStyle = 'rgba(0, 200, 0, 0.95)';
            ctx.lineWidth = 3;
            ctx.beginPath();
            ctx.moveTo(iconX + iconSize * 0.2, iconY + iconSize * 0.55);
            ctx.lineTo(iconX + iconSize * 0.40, iconY + iconSize * 0.75);
            ctx.lineTo(iconX + iconSize * 0.85, iconY + iconSize * 0.25);
            ctx.stroke();
        }
        if (!node._customCheckboxRects) {
            node._customCheckboxRects = [];
        }
        node._customCheckboxRects[i] = {
            x: clickX,
            y: clickY,
            width: clickW,
            height: clickH
        };
        
        // 在多图片模式下，只在悬浮时显示文件名和清除按钮
        if (!node._customSingleImageMode) {
            const mouseInImage = node._customMouseX !== undefined && node._customMouseY !== undefined &&
                node._customMouseX >= rect.x && node._customMouseX <= rect.x + rect.width &&
                node._customMouseY >= rect.y && node._customMouseY <= rect.y + rect.height;
            
            if (mouseInImage) {
                // 绘制图片标题 - 在顶部显示文件名，与图片重叠
                ctx.textAlign = 'center';
                
                // 使用保存的文件名
                const fileName = node._customImageFileNames && node._customImageFileNames[i] ? node._customImageFileNames[i] : 'Unknown';
                
                // 在顶部绘制文件名背景（半透明，与图片重叠）
                ctx.fillStyle = 'rgba(0, 0, 0, 0.6)';
                ctx.fillRect(rect.x, rect.y, rect.width, 30);
                
                // 自动调整字体大小
                const maxTextWidth = rect.width - 10; // 留出边距
                const fontSize = getAdjustedFontSize(ctx, fileName, maxTextWidth);
                ctx.font = `bold ${fontSize}px Arial`;
                
                // 绘制文件名
                ctx.fillStyle = '#fff';
                ctx.fillText(fileName, rect.x + rect.width / 2, rect.y + 20);
                
                // 绘制右上角清除按钮
                const buttonSize = 16;
                const buttonMargin = 5;
                const clearButtonX = rect.x + rect.width - buttonMargin - buttonSize;
                const clearButtonY = rect.y + buttonMargin;
                
                // 检查鼠标是否悬浮在清除按钮上
                const mouseInClearButton = node._customMouseX >= clearButtonX && node._customMouseX <= clearButtonX + buttonSize &&
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
                
                // 保存文件名区域信息，用于tooltip检测
                if (!node._customFileNameRects) {
                    node._customFileNameRects = [];
                }
                node._customFileNameRects[i] = {
                    x: rect.x,
                    y: rect.y,
                    width: rect.width,
                    height: 30
                };
            } else {
                // 鼠标不在图片上时，清除按钮区域为空
                if (!node._customClearButtonRects) {
                    node._customClearButtonRects = [];
                }
                node._customClearButtonRects[i] = null;
                
                // 保存文件名区域信息
                if (!node._customFileNameRects) {
                    node._customFileNameRects = [];
                }
                node._customFileNameRects[i] = null;
            }
        } else {
            // 单图片模式下，清除按钮区域为空（将在控制按钮区域绘制）
            if (!node._customClearButtonRects) {
                node._customClearButtonRects = [];
            }
            node._customClearButtonRects[i] = null;
            
            // 单图片模式下，文件名区域为空（将在底部绘制）
            if (!node._customFileNameRects) {
                node._customFileNameRects = [];
            }
            node._customFileNameRects[i] = null;
        }
    }
    
    // 绘制控制按钮（只在单图片模式下显示）
    if (node._customSingleImageMode) {
        const buttonSize = 20;
        const buttonSpacing = 12;
        const cornerSafe = 22;
        
        // 获取当前显示的图片位置，用于计算恢复按钮位置
        const currentImageRect = node._customImageRects[node._customFocusedImageIndex];
        const restoreButtonX = currentImageRect ? currentImageRect.x + currentImageRect.width - buttonSize - 5 : node.size[0] - buttonSize - 10;
        const restoreButtonY = currentImageRect ? currentImageRect.y + 5 : 10;
        
        let restoreHitX, restoreHitY, restoreHitW, restoreHitH;
        if (currentImageRect) {
            restoreHitX = currentImageRect.x + (currentImageRect.width * 3) / 4;
            restoreHitY = currentImageRect.y;
            restoreHitW = currentImageRect.width / 4;
            restoreHitH = currentImageRect.height / 4;
        } else {
            restoreHitW = Math.round(buttonSize * 1.8);
            restoreHitH = Math.round(buttonSize * 1.8);
            restoreHitX = restoreButtonX - Math.floor((restoreHitW - buttonSize) / 2);
            restoreHitY = restoreButtonY - Math.floor((restoreHitH - buttonSize) / 2);
        }
        const mouseInRestoreButton = node._customMouseX !== undefined && node._customMouseY !== undefined &&
            node._customMouseX >= restoreHitX && node._customMouseX <= restoreHitX + restoreHitW &&
            node._customMouseY >= restoreHitY && node._customMouseY <= restoreHitY + restoreHitH;
        
        const mouseInPrevButton = node._customMouseX !== undefined && node._customMouseY !== undefined &&
            node._customMouseX >= node.size[0] - buttonSize * 3 - buttonSpacing * 2 - 10 - cornerSafe && node._customMouseX <= node.size[0] - buttonSize * 2 - buttonSpacing * 2 - 10 - cornerSafe &&
            node._customMouseY >= node.size[1] - buttonSize - 10 && node._customMouseY <= node.size[1] - 10;
        
        const mouseInNextButton = node._customMouseX !== undefined && node._customMouseY !== undefined &&
            node._customMouseX >= node.size[0] - buttonSize * 2 - buttonSpacing - 10 - cornerSafe && node._customMouseX <= node.size[0] - buttonSize - buttonSpacing - 10 - cornerSafe &&
            node._customMouseY >= node.size[1] - buttonSize - 10 && node._customMouseY <= node.size[1] - 10;
        
        // 检查鼠标是否悬浮在清除按钮上（左下角）
        const mouseInClearButton = node._customMouseX !== undefined && node._customMouseY !== undefined &&
            node._customMouseX >= 10 && node._customMouseX <= 10 + buttonSize &&
            node._customMouseY >= node.size[1] - buttonSize - 10 && node._customMouseY <= node.size[1] - 10;
        
        // 检查鼠标是否悬浮在全屏预览按钮上（右下角）
        const mouseInFullscreenButton = node._customMouseX !== undefined && node._customMouseY !== undefined &&
            node._customMouseX >= node.size[0] - buttonSize - 10 && node._customMouseX <= node.size[0] - 10 &&
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
            const indexX = node.size[0] - buttonSize * 3 - buttonSpacing * 2 - 15 - cornerSafe;
            const indexY = node.size[1] - buttonSize - 10 + buttonSize / 2;
            
            // 绘制索引文本
            ctx.fillText(indexText, indexX, indexY);
        }
        
        // 绘制上一个按钮 (‹) - 左边
        const prevButtonX = node.size[0] - buttonSize * 3 - buttonSpacing * 2 - 10 - cornerSafe;
        const prevButtonY = node.size[1] - buttonSize - 10;
        
        // 按钮背景（悬浮效果）
        const rPrev = 6;
        ctx.fillStyle = mouseInPrevButton ? 'rgba(235,235,240,0.95)' : 'rgba(235,235,240,0.85)';
        ctx.strokeStyle = mouseInPrevButton ? 'rgba(80,80,90,0.9)' : 'rgba(120,120,130,0.8)';
        ctx.lineWidth = mouseInPrevButton ? 2 : 1;
        ctx.beginPath();
        ctx.moveTo(prevButtonX + rPrev, prevButtonY);
        ctx.lineTo(prevButtonX + buttonSize - rPrev, prevButtonY);
        ctx.quadraticCurveTo(prevButtonX + buttonSize, prevButtonY, prevButtonX + buttonSize, prevButtonY + rPrev);
        ctx.lineTo(prevButtonX + buttonSize, prevButtonY + buttonSize - rPrev);
        ctx.quadraticCurveTo(prevButtonX + buttonSize, prevButtonY + buttonSize, prevButtonX + buttonSize - rPrev, prevButtonY + buttonSize);
        ctx.lineTo(prevButtonX + rPrev, prevButtonY + buttonSize);
        ctx.quadraticCurveTo(prevButtonX, prevButtonY + buttonSize, prevButtonX, prevButtonY + buttonSize - rPrev);
        ctx.lineTo(prevButtonX, prevButtonY + rPrev);
        ctx.quadraticCurveTo(prevButtonX, prevButtonY, prevButtonX + rPrev, prevButtonY);
        ctx.closePath();
        ctx.fill();
        ctx.stroke();
        ctx.fillStyle = 'rgba(30, 30, 35, 1)';
        ctx.font = `${buttonSize - 4}px Arial`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText('‹', prevButtonX + buttonSize / 2, prevButtonY + buttonSize / 2);
        
        // 绘制下一个按钮 (›) - 右边
        const nextButtonX = node.size[0] - buttonSize * 2 - buttonSpacing - 10 - cornerSafe;
        const nextButtonY = node.size[1] - buttonSize - 10;
        
        // 按钮背景（悬浮效果）
        const rNext = 6;
        ctx.fillStyle = mouseInNextButton ? 'rgba(235,235,240,0.95)' : 'rgba(235,235,240,0.85)';
        ctx.strokeStyle = mouseInNextButton ? 'rgba(80,80,90,0.9)' : 'rgba(120,120,130,0.8)';
        ctx.lineWidth = mouseInNextButton ? 2 : 1;
        ctx.beginPath();
        ctx.moveTo(nextButtonX + rNext, nextButtonY);
        ctx.lineTo(nextButtonX + buttonSize - rNext, nextButtonY);
        ctx.quadraticCurveTo(nextButtonX + buttonSize, nextButtonY, nextButtonX + buttonSize, nextButtonY + rNext);
        ctx.lineTo(nextButtonX + buttonSize, nextButtonY + buttonSize - rNext);
        ctx.quadraticCurveTo(nextButtonX + buttonSize, nextButtonY + buttonSize, nextButtonX + buttonSize - rNext, nextButtonY + buttonSize);
        ctx.lineTo(nextButtonX + rNext, nextButtonY + buttonSize);
        ctx.quadraticCurveTo(nextButtonX, nextButtonY + buttonSize, nextButtonX, nextButtonY + buttonSize - rNext);
        ctx.lineTo(nextButtonX, nextButtonY + rNext);
        ctx.quadraticCurveTo(nextButtonX, nextButtonY, nextButtonX + rNext, nextButtonY);
        ctx.closePath();
        ctx.fill();
        ctx.stroke();
        ctx.fillStyle = 'rgba(30, 30, 35, 1)';
        ctx.font = `${buttonSize - 4}px Arial`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText('›', nextButtonX + buttonSize / 2, nextButtonY + buttonSize / 2);
        
        // 绘制恢复按钮 (⭯) - 放在图片区域的右上角
        
        ctx.fillStyle = 'rgba(0, 0, 0, 0)';
        ctx.fillRect(restoreButtonX, restoreButtonY, buttonSize, buttonSize);
        ctx.strokeStyle = mouseInRestoreButton ? 'rgba(255, 255, 255, 1)' : 'rgba(255, 255, 255, 0.8)';
        ctx.lineWidth = mouseInRestoreButton ? 2 : 1;
        ctx.strokeRect(restoreButtonX, restoreButtonY, buttonSize, buttonSize);
        ctx.fillStyle = 'rgba(255, 255, 255, 1)';
        ctx.fillText('⭯', restoreButtonX + buttonSize / 2, restoreButtonY + buttonSize / 2);
        
        // 绘制左下角清除按钮
        const clearButtonX = 10;
        const clearButtonY = node.size[1] - buttonSize - 10;
        
        const rClear = 6;
        ctx.fillStyle = 'rgba(235,235,240,0.85)';
        ctx.strokeStyle = 'rgba(120,120,130,0.8)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(clearButtonX + rClear, clearButtonY);
        ctx.lineTo(clearButtonX + buttonSize - rClear, clearButtonY);
        ctx.quadraticCurveTo(clearButtonX + buttonSize, clearButtonY, clearButtonX + buttonSize, clearButtonY + rClear);
        ctx.lineTo(clearButtonX + buttonSize, clearButtonY + buttonSize - rClear);
        ctx.quadraticCurveTo(clearButtonX + buttonSize, clearButtonY + buttonSize, clearButtonX + buttonSize - rClear, clearButtonY + buttonSize);
        ctx.lineTo(clearButtonX + rClear, clearButtonY + buttonSize);
        ctx.quadraticCurveTo(clearButtonX, clearButtonY + buttonSize, clearButtonX, clearButtonY + buttonSize - rClear);
        ctx.lineTo(clearButtonX, clearButtonY + rClear);
        ctx.quadraticCurveTo(clearButtonX, clearButtonY, clearButtonX + rClear, clearButtonY);
        ctx.closePath();
        ctx.fill();
        ctx.stroke();
        ctx.fillStyle = 'rgba(30, 30, 35, 1)';
        ctx.font = `${buttonSize - 4}px Arial`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText('×', clearButtonX + buttonSize / 2, clearButtonY + buttonSize / 2);
        
        // 绘制右下角全屏预览按钮
        const fullscreenButtonX = node.size[0] - buttonSize - 10 - cornerSafe;
        const fullscreenButtonY = node.size[1] - buttonSize - 10;
        
        const rFull = 6;
        ctx.fillStyle = mouseInFullscreenButton ? 'rgba(235,235,240,0.95)' : 'rgba(235,235,240,0.85)';
        ctx.strokeStyle = mouseInFullscreenButton ? 'rgba(80,80,90,0.9)' : 'rgba(120,120,130,0.8)';
        ctx.lineWidth = mouseInFullscreenButton ? 2 : 1;
        ctx.beginPath();
        ctx.moveTo(fullscreenButtonX + rFull, fullscreenButtonY);
        ctx.lineTo(fullscreenButtonX + buttonSize - rFull, fullscreenButtonY);
        ctx.quadraticCurveTo(fullscreenButtonX + buttonSize, fullscreenButtonY, fullscreenButtonX + buttonSize, fullscreenButtonY + rFull);
        ctx.lineTo(fullscreenButtonX + buttonSize, fullscreenButtonY + buttonSize - rFull);
        ctx.quadraticCurveTo(fullscreenButtonX + buttonSize, fullscreenButtonY + buttonSize, fullscreenButtonX + buttonSize - rFull, fullscreenButtonY + buttonSize);
        ctx.lineTo(fullscreenButtonX + rFull, fullscreenButtonY + buttonSize);
        ctx.quadraticCurveTo(fullscreenButtonX, fullscreenButtonY + buttonSize, fullscreenButtonX, fullscreenButtonY + buttonSize - rFull);
        ctx.lineTo(fullscreenButtonX, fullscreenButtonY + rFull);
        ctx.quadraticCurveTo(fullscreenButtonX, fullscreenButtonY, fullscreenButtonX + rFull, fullscreenButtonY);
        ctx.closePath();
        ctx.fill();
        ctx.stroke();
        ctx.fillStyle = 'rgba(30, 30, 35, 1)';
        ctx.font = `${buttonSize - 4}px Arial`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText('⛶', fullscreenButtonX + buttonSize / 2, fullscreenButtonY + buttonSize / 2);
        
        node._customPrevButtonRect = { x: prevButtonX, y: prevButtonY, width: buttonSize, height: buttonSize };
        node._customNextButtonRect = { x: nextButtonX, y: nextButtonY, width: buttonSize, height: buttonSize };
        node._customRestoreButtonRect = { x: restoreHitX, y: restoreHitY, width: restoreHitW, height: restoreHitH };
        node._customClearButtonRect = { x: clearButtonX, y: clearButtonY, width: buttonSize, height: buttonSize };
        node._customFullscreenButtonRect = { x: fullscreenButtonX, y: fullscreenButtonY, width: buttonSize, height: buttonSize };
        
        // 绘制底部文件名
        if (node._customImageFileNames && node._customImageFileNames[node._customFocusedImageIndex]) {
            const fileName = node._customImageFileNames[node._customFocusedImageIndex];
            
            // 设置文本样式
            ctx.fillStyle = 'rgba(255, 255, 255, 0.9)';
            ctx.font = '12px Arial';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            
            // 在底部中间绘制文件名
            const fileNameY = node.size[1] - 15;
            ctx.fillText(fileName, node.size[0] / 2, fileNameY);
        }
    } else {
        const buttonHeight = 25;
        const buttonSpacing = 10;
        const buttonY = node.size[1] - buttonHeight - 5;
        const selectW = 60;
        const deselectW = 70;
        const invertW = 60;
        const clearW = 90;
        const showSelectedW = 90;
        const reuseMaskW = 100;
        const selectAllButtonX = 10;
        const deselectAllButtonX = selectAllButtonX + selectW + buttonSpacing;
        const invertSelectionButtonX = deselectAllButtonX + deselectW + buttonSpacing;
        const clearSelectedButtonX = invertSelectionButtonX + invertW + buttonSpacing;
        const clearUnselectedButtonX = clearSelectedButtonX + clearW + buttonSpacing;
        const showSelectedButtonX = clearUnselectedButtonX + clearW + buttonSpacing;
        const reuseMaskButtonX = showSelectedButtonX + showSelectedW + buttonSpacing;
        const mouseInSelectAllButton = node._customMouseX !== undefined && node._customMouseY !== undefined &&
            node._customMouseX >= selectAllButtonX && node._customMouseX <= selectAllButtonX + selectW &&
            node._customMouseY >= buttonY && node._customMouseY <= buttonY + buttonHeight;
        const mouseInDeselectAllButton = node._customMouseX !== undefined && node._customMouseY !== undefined &&
            node._customMouseX >= deselectAllButtonX && node._customMouseX <= deselectAllButtonX + deselectW &&
            node._customMouseY >= buttonY && node._customMouseY <= buttonY + buttonHeight;
        const mouseInInvertSelectionButton = node._customMouseX !== undefined && node._customMouseY !== undefined &&
            node._customMouseX >= invertSelectionButtonX && node._customMouseX <= invertSelectionButtonX + invertW &&
            node._customMouseY >= buttonY && node._customMouseY <= buttonY + buttonHeight;
        const mouseInClearSelectedButton = node._customMouseX !== undefined && node._customMouseY !== undefined &&
            node._customMouseX >= clearSelectedButtonX && node._customMouseX <= clearSelectedButtonX + clearW &&
            node._customMouseY >= buttonY && node._customMouseY <= buttonY + buttonHeight;
        const mouseInClearUnselectedButton = node._customMouseX !== undefined && node._customMouseY !== undefined &&
            node._customMouseX >= clearUnselectedButtonX && node._customMouseX <= clearUnselectedButtonX + clearW &&
            node._customMouseY >= buttonY && node._customMouseY <= buttonY + buttonHeight;
        const mouseInShowSelectedButton = node._customMouseX !== undefined && node._customMouseY !== undefined &&
            node._customMouseX >= showSelectedButtonX && node._customMouseX <= showSelectedButtonX + showSelectedW &&
            node._customMouseY >= buttonY && node._customMouseY <= buttonY + buttonHeight;
        const mouseInReuseMaskButton = node._customMouseX !== undefined && node._customMouseY !== undefined &&
            node._customMouseX >= reuseMaskButtonX && node._customMouseX <= reuseMaskButtonX + reuseMaskW &&
            node._customMouseY >= buttonY && node._customMouseY <= buttonY + buttonHeight;
        const r = 6;
        function drawButton(x, w, text, hover) {
            const y = buttonY, h = buttonHeight;
            ctx.fillStyle = hover ? 'rgba(235,235,240,0.95)' : 'rgba(235,235,240,0.85)';
            ctx.strokeStyle = hover ? 'rgba(80,80,90,0.9)' : 'rgba(120,120,130,0.8)';
            ctx.lineWidth = hover ? 2 : 1;
            ctx.beginPath();
            ctx.moveTo(x + r, y);
            ctx.lineTo(x + w - r, y);
            ctx.quadraticCurveTo(x + w, y, x + w, y + r);
            ctx.lineTo(x + w, y + h - r);
            ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
            ctx.lineTo(x + r, y + h);
            ctx.quadraticCurveTo(x, y + h, x, y + h - r);
            ctx.lineTo(x, y + r);
            ctx.quadraticCurveTo(x, y, x + r, y);
            ctx.closePath();
            ctx.fill();
            ctx.stroke();
            ctx.fillStyle = 'rgba(30,30,35,1)';
            ctx.font = 'bold 13px "Microsoft YaHei", Arial';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(text, x + w / 2, y + h / 2);
        }
        drawButton(selectAllButtonX, selectW, '全选', mouseInSelectAllButton);
        drawButton(deselectAllButtonX, deselectW, '全不选', mouseInDeselectAllButton);
        drawButton(invertSelectionButtonX, invertW, '反选', mouseInInvertSelectionButton);
        drawButton(clearSelectedButtonX, clearW, '清除选中', mouseInClearSelectedButton);
        drawButton(clearUnselectedButtonX, clearW, '清除未选', mouseInClearUnselectedButton);
        drawButton(showSelectedButtonX, showSelectedW, node._customShowOnlySelected ? '显示全部' : '仅显示勾选', mouseInShowSelectedButton);
        drawButton(reuseMaskButtonX, reuseMaskW, node._customMaskReuseEnabled ? '遮罩复用✓' : '遮罩复用', mouseInReuseMaskButton);
        node._customSelectAllButtonRect = {
            x: selectAllButtonX,
            y: buttonY,
            width: selectW,
            height: buttonHeight
        };
        node._customDeselectAllButtonRect = {
            x: deselectAllButtonX,
            y: buttonY,
            width: deselectW,
            height: buttonHeight
        };
        node._customInvertSelectionButtonRect = {
            x: invertSelectionButtonX,
            y: buttonY,
            width: invertW,
            height: buttonHeight
        };
        node._customClearSelectedButtonRect = {
            x: clearSelectedButtonX,
            y: buttonY,
            width: clearW,
            height: buttonHeight
        };
        node._customClearUnselectedButtonRect = {
            x: clearUnselectedButtonX,
            y: buttonY,
            width: clearW,
            height: buttonHeight
        };
        node._customShowSelectedButtonRect = {
            x: showSelectedButtonX,
            y: buttonY,
            width: showSelectedW,
            height: buttonHeight
        };
        node._customReuseMaskButtonRect = {
            x: reuseMaskButtonX,
            y: buttonY,
            width: reuseMaskW,
            height: buttonHeight
        };
    }
    
    if (node._customSingleImageMode) { ctx.restore(); return;
        const buttonSize = 20;
        const buttonSpacing = 5;
        
        // 获取当前显示的图片位置，用于计算恢复按钮位置
        const currentImageRect = node._customImageRects[node._customFocusedImageIndex];
        const restoreButtonX = currentImageRect ? currentImageRect.x + currentImageRect.width - buttonSize - 5 : node.size[0] - buttonSize - 10;
        const restoreButtonY = currentImageRect ? currentImageRect.y + 5 : 10;
        
        // 检查鼠标是否悬浮在按钮上
        let restoreHitX, restoreHitY, restoreHitW, restoreHitH;
        if (currentImageRect) {
            restoreHitX = currentImageRect.x + (currentImageRect.width * 3) / 4;
            restoreHitY = currentImageRect.y;
            restoreHitW = currentImageRect.width / 4;
            restoreHitH = currentImageRect.height / 4;
        } else {
            restoreHitW = Math.round(buttonSize * 1.8);
            restoreHitH = Math.round(buttonSize * 1.8);
            restoreHitX = restoreButtonX - Math.floor((restoreHitW - buttonSize) / 2);
            restoreHitY = restoreButtonY - Math.floor((restoreHitH - buttonSize) / 2);
        }
        const mouseInRestoreButton = node._customMouseX !== undefined && node._customMouseY !== undefined &&
            node._customMouseX >= restoreHitX && node._customMouseX <= restoreHitX + restoreHitW &&
            node._customMouseY >= restoreHitY && node._customMouseY <= restoreHitY + restoreHitH;
        
        const mouseInPrevButton = node._customMouseX !== undefined && node._customMouseY !== undefined &&
            node._customMouseX >= node.size[0] - buttonSize * 3 - buttonSpacing - 10 && node._customMouseX <= node.size[0] - buttonSize * 2 - buttonSpacing - 10 &&
            node._customMouseY >= node.size[1] - buttonSize - 10 && node._customMouseY <= node.size[1] - 10;
        
        const mouseInNextButton = node._customMouseX !== undefined && node._customMouseY !== undefined &&
            node._customMouseX >= node.size[0] - buttonSize * 2 - 10 && node._customMouseX <= node.size[0] - buttonSize - 10 &&
            node._customMouseY >= node.size[1] - buttonSize - 10 && node._customMouseY <= node.size[1] - 10;
        
        // 检查鼠标是否悬浮在清除按钮上（左下角）
        const mouseInClearButton = node._customMouseX !== undefined && node._customMouseY !== undefined &&
            node._customMouseX >= 10 && node._customMouseX <= 10 + buttonSize &&
            node._customMouseY >= node.size[1] - buttonSize - 10 && node._customMouseY <= node.size[1] - 10;
        
        // 检查鼠标是否悬浮在全屏预览按钮上（右下角）
        const mouseInFullscreenButton = node._customMouseX !== undefined && node._customMouseY !== undefined &&
            node._customMouseX >= node.size[0] - buttonSize - 10 && node._customMouseX <= node.size[0] - 10 &&
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
            const indexX = node.size[0] - buttonSize * 3 - buttonSpacing - 15;
            const indexY = node.size[1] - buttonSize - 10 + buttonSize / 2;
            
            // 绘制索引文本
            ctx.fillText(indexText, indexX, indexY);
        }
        
        // 绘制上一个按钮 (‹) - 左边
        const prevButtonX = node.size[0] - buttonSize * 3 - buttonSpacing - 10;
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
        
        // 绘制下一个按钮 (›) - 右边
        const nextButtonX = node.size[0] - buttonSize * 2 - 10;
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
        
        // 绘制恢复按钮 (⭯) - 放在图片区域的右上角
        
        ctx.fillStyle = 'rgba(0, 0, 0, 0)';
        ctx.fillRect(restoreButtonX, restoreButtonY, buttonSize, buttonSize);
        ctx.strokeStyle = mouseInRestoreButton ? 'rgba(255, 255, 255, 1)' : 'rgba(255, 255, 255, 0.8)';
        ctx.lineWidth = mouseInRestoreButton ? 2 : 1;
        ctx.strokeRect(restoreButtonX, restoreButtonY, buttonSize, buttonSize);
        ctx.fillStyle = 'rgba(255, 255, 255, 1)';
        ctx.fillText('⭯', restoreButtonX + buttonSize / 2, restoreButtonY + buttonSize / 2);
        
        // 绘制左下角清除按钮
        const clearButtonX = 10;
        const clearButtonY = node.size[1] - buttonSize - 10;
        
        // 按钮背景（固定样式，无悬浮效果）
        ctx.fillStyle = 'rgba(255, 0, 0, 0.8)';
        ctx.fillRect(clearButtonX, clearButtonY, buttonSize, buttonSize);
        
        // 按钮边框
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.8)';
        ctx.lineWidth = 1;
        ctx.strokeRect(clearButtonX, clearButtonY, buttonSize, buttonSize);
        
        // 绘制清除图标 (×)
        ctx.fillStyle = 'rgba(255, 255, 255, 1)';
        ctx.font = `${buttonSize - 4}px Arial`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText('×', clearButtonX + buttonSize / 2, clearButtonY + buttonSize / 2);
        
        // 绘制右下角全屏预览按钮
        const fullscreenButtonX = node.size[0] - buttonSize - 10;
        const fullscreenButtonY = node.size[1] - buttonSize - 10;
        
        // 按钮背景（悬浮效果）
        ctx.fillStyle = mouseInFullscreenButton ? 'rgba(0, 0, 0, 0.9)' : 'rgba(0, 0, 0, 0.7)';
        ctx.fillRect(fullscreenButtonX, fullscreenButtonY, buttonSize, buttonSize);
        
        // 按钮边框
        ctx.strokeStyle = mouseInFullscreenButton ? 'rgba(255, 255, 255, 1)' : 'rgba(255, 255, 255, 0.8)';
        ctx.lineWidth = mouseInFullscreenButton ? 2 : 1;
        ctx.strokeRect(fullscreenButtonX, fullscreenButtonY, buttonSize, buttonSize);
        
        // 绘制全屏图标 (⛶)
        ctx.fillStyle = 'rgba(255, 255, 255, 1)';
        ctx.font = `${buttonSize - 4}px Arial`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText('⛶', fullscreenButtonX + buttonSize / 2, fullscreenButtonY + buttonSize / 2);
        
        // 绘制底部文件名
        if (node._customImageFileNames && node._customImageFileNames[node._customFocusedImageIndex]) {
            const fileName = node._customImageFileNames[node._customFocusedImageIndex];
            
            // 设置文本样式
            ctx.fillStyle = 'rgba(255, 255, 255, 0.9)';
            ctx.font = '12px Arial';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            
            // 在底部中间绘制文件名
            const fileNameY = node.size[1] - 15;
            ctx.fillText(fileName, node.size[0] / 2, fileNameY);
        }
        
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
            x: restoreHitX,
            y: restoreHitY,
            width: restoreHitW,
            height: restoreHitH
        };
        node._customClearButtonRect = {
            x: clearButtonX,
            y: clearButtonY,
            width: buttonSize,
            height: buttonSize
        };
        node._customFullscreenButtonRect = {
            x: fullscreenButtonX,
            y: fullscreenButtonY,
            width: buttonSize,
            height: buttonSize
        };
    } else {
        // 清除按钮区域信息
        node._customPrevButtonRect = null;
        node._customNextButtonRect = null;
        node._customRestoreButtonRect = null;
        node._customClearButtonRect = null;
        node._customFullscreenButtonRect = null;
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
        app.graph.setDirtyCanvas(true, true);
        return;
    }
    
    // 加载图片
    showImages(node, paths);
    
    // 触发重绘
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
    
    // 重写节点的绘制方法（只在第一次调用时设置）
    if (!this._customDrawMethodSet) {
        console.log("设置自定义绘制方法");
        
        const originalOnDrawForeground = this.onDrawForeground;
        
        // 创建一个包装函数，确保我们的绘制逻辑始终被执行
        const customDrawForeground = function(ctx) {
            // 首先调用原始绘制方法
            if (originalOnDrawForeground) {
                originalOnDrawForeground.call(this, ctx);
            }
            
            // 只有LoadImageBatchAdvanced节点才执行自定义绘制
            if (this.type === "LoadImageBatchAdvanced" && this._customImgs && this._customImageRects) {
                drawNodeImages(this, ctx);
            }
        };
        
        // 设置绘制方法
        this.onDrawForeground = customDrawForeground;
        
        // 标记已设置
        this._customDrawMethodSet = true;
        
        console.log("自定义绘制方法设置完成");
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
        
        // 只有LoadImageBatchAdvanced节点才处理自定义鼠标事件
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
                const controls = [
                    { r: this._customSelectAllButtonRect, t: '全选所有图片' },
                    { r: this._customInvertSelectionButtonRect, t: '反选当前选择' },
                    { r: this._customClearSelectedButtonRect, t: '清除选中的图片' },
                    { r: this._customClearUnselectedButtonRect, t: '清除未选的图片' },
                    { r: this._customShowSelectedButtonRect, t: this._customShowOnlySelected ? '恢复显示全部图片' : '仅显示勾选的图片' },
                    { r: this._customReuseMaskButtonRect, t: '相同尺寸的图片复用第一个已编辑遮罩' }
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
            }
        }
        if (!tooltipShown) this.hideTooltip();
        if (mousePositionChanged) app.graph.setDirtyCanvas(true, false);
    };
            
    // 鼠标离开时清除位置
    const originalOnMouseLeave = this.onMouseLeave;
    this.onMouseLeave = function(e) {
        if (originalOnMouseLeave) {
            originalOnMouseLeave.call(this, e);
        }
        
        // 只有LoadImageBatchAdvanced节点才处理自定义鼠标事件
        if (this.type !== "LoadImageBatchAdvanced") {
            return;
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
        // 只有LoadImageBatchAdvanced节点才处理自定义鼠标事件
        if (this.type !== "LoadImageBatchAdvanced") {
            if (originalOnMouseDown) {
                return originalOnMouseDown.call(this, e);
            }
            return false;
        }
        
        console.log("onMouseDown 被调用", e);
        console.log("节点信息:", this.id, this.type, this.size);
        console.log("图片区域:", this._customImageRects);
        
        // 获取节点的Canvas坐标
        const nodePos = this.pos;

        // 检查是否点击复选框
        if (this._customCheckboxRects && this._customCheckboxRects.length > 0) {
            for (let i = 0; i < this._customCheckboxRects.length; i++) {
                const checkboxRect = this._customCheckboxRects[i];
                
                // 检查复选框是否存在
                if (!checkboxRect) {
                    continue;
                }
                
                // 检查图片是否可见
                if (this._customImageRects && this._customImageRects[i] && this._customImageRects[i].visible === false) {
                    continue;
                }
                
                // 计算复选框在Canvas中的绝对坐标
                const absCheckboxX = nodePos[0] + checkboxRect.x;
                const absCheckboxY = nodePos[1] + checkboxRect.y;
                const absCheckboxWidth = checkboxRect.width;
                const absCheckboxHeight = checkboxRect.height;
                
                if (e.canvasX >= absCheckboxX && e.canvasX <= absCheckboxX + absCheckboxWidth &&
                    e.canvasY >= absCheckboxY && e.canvasY <= absCheckboxY + absCheckboxHeight) {
                    
                    console.log(`点击复选框，图片索引: ${i}`);
                    
                    // 阻止事件冒泡
                    e.preventDefault();
                    e.stopPropagation();
                    
                    // 切换选择状态
                    if (this._customSelectedImages && this._customSelectedImages[i] !== undefined) {
                        this._customSelectedImages[i] = !this._customSelectedImages[i];
                        console.log(`图片 ${i} 选择状态切换为: ${this._customSelectedImages[i]}`);
                        
                        // 更新widget的值
                        updateWidgetValue(this);
                        
                        // 触发重绘
                        app.graph.setDirtyCanvas(true, false);
                    }
                    
                    return true;
                }
            }
        }
        
        if (!this._customSingleImageMode) {
            if (this._customSelectAllButtonRect) {
                const ax = nodePos[0] + this._customSelectAllButtonRect.x;
                const ay = nodePos[1] + this._customSelectAllButtonRect.y;
                const aw = this._customSelectAllButtonRect.width;
                const ah = this._customSelectAllButtonRect.height;
                if (e.canvasX >= ax && e.canvasX <= ax + aw && e.canvasY >= ay && e.canvasY <= ay + ah) {
                    e.preventDefault();
                    e.stopPropagation();
                    if (this._customSelectedImages && this._customSelectedImages.length > 0) {
                        this._customSelectedImages.fill(true);
                        updateWidgetValue(this);
                        app.graph.setDirtyCanvas(true, false);
                    }
                    return true;
                }
            }
            if (this._customDeselectAllButtonRect) {
                const ax = nodePos[0] + this._customDeselectAllButtonRect.x;
                const ay = nodePos[1] + this._customDeselectAllButtonRect.y;
                const aw = this._customDeselectAllButtonRect.width;
                const ah = this._customDeselectAllButtonRect.height;
                if (e.canvasX >= ax && e.canvasX <= ax + aw && e.canvasY >= ay && e.canvasY <= ay + ah) {
                    e.preventDefault();
                    e.stopPropagation();
                    if (this._customSelectedImages && this._customSelectedImages.length > 0) {
                        this._customSelectedImages.fill(false);
                        updateWidgetValue(this);
                        app.graph.setDirtyCanvas(true, false);
                    }
                    return true;
                }
            }
            if (this._customInvertSelectionButtonRect) {
                const ax = nodePos[0] + this._customInvertSelectionButtonRect.x;
                const ay = nodePos[1] + this._customInvertSelectionButtonRect.y;
                const aw = this._customInvertSelectionButtonRect.width;
                const ah = this._customInvertSelectionButtonRect.height;
                if (e.canvasX >= ax && e.canvasX <= ax + aw && e.canvasY >= ay && e.canvasY <= ay + ah) {
                    e.preventDefault();
                    e.stopPropagation();
                    if (this._customSelectedImages && this._customSelectedImages.length > 0) {
                        for (let i = 0; i < this._customSelectedImages.length; i++) {
                            this._customSelectedImages[i] = !this._customSelectedImages[i];
                        }
                        updateWidgetValue(this);
                        app.graph.setDirtyCanvas(true, false);
                    }
                    return true;
                }
            }
            if (this._customClearSelectedButtonRect) {
                const ax = nodePos[0] + this._customClearSelectedButtonRect.x;
                const ay = nodePos[1] + this._customClearSelectedButtonRect.y;
                const aw = this._customClearSelectedButtonRect.width;
                const ah = this._customClearSelectedButtonRect.height;
                if (e.canvasX >= ax && e.canvasX <= ax + aw && e.canvasY >= ay && e.canvasY <= ay + ah) {
                    e.preventDefault();
                    e.stopPropagation();
                    if (this._customImagePaths && this._customSelectedImages) {
                        const newPaths = [];
                        const newSelected = [];
                        const newNames = [];
                        for (let i = 0; i < this._customImagePaths.length; i++) {
                            if (!this._customSelectedImages[i]) {
                                newPaths.push(this._customImagePaths[i]);
                                newSelected.push(this._customSelectedImages[i]);
                                if (this._customImageFileNames && this._customImageFileNames[i]) newNames.push(this._customImageFileNames[i]);
                            }
                        }
                        this._customImagePaths = newPaths;
                        this._customSelectedImages = newSelected.length ? newSelected : new Array(newPaths.length).fill(true);
                        this._customImageFileNames = newNames;
                        const imagePathsWidget = this.widgets.find(w => w.name === "image_paths");
                        if (imagePathsWidget) imagePathsWidget.value = (this._customImagePaths || []).join(',');
                        updateWidgetValue(this);
                        showImages(this, this._customImagePaths);
                        app.graph.setDirtyCanvas(true, false);
                    }
                    return true;
                }
            }
            if (this._customClearUnselectedButtonRect) {
                const ax = nodePos[0] + this._customClearUnselectedButtonRect.x;
                const ay = nodePos[1] + this._customClearUnselectedButtonRect.y;
                const aw = this._customClearUnselectedButtonRect.width;
                const ah = this._customClearUnselectedButtonRect.height;
                if (e.canvasX >= ax && e.canvasX <= ax + aw && e.canvasY >= ay && e.canvasY <= ay + ah) {
                    e.preventDefault();
                    e.stopPropagation();
                    if (this._customImagePaths && this._customSelectedImages) {
                        const newPaths = [];
                        const newSelected = [];
                        const newNames = [];
                        for (let i = 0; i < this._customImagePaths.length; i++) {
                            if (this._customSelectedImages[i]) {
                                newPaths.push(this._customImagePaths[i]);
                                newSelected.push(this._customSelectedImages[i]);
                                if (this._customImageFileNames && this._customImageFileNames[i]) newNames.push(this._customImageFileNames[i]);
                            }
                        }
                        this._customImagePaths = newPaths;
                        this._customSelectedImages = newSelected.length ? newSelected : new Array(newPaths.length).fill(true);
                        this._customImageFileNames = newNames;
                        const imagePathsWidget = this.widgets.find(w => w.name === "image_paths");
                        if (imagePathsWidget) imagePathsWidget.value = (this._customImagePaths || []).join(',');
                        updateWidgetValue(this);
                        showImages(this, this._customImagePaths);
                        app.graph.setDirtyCanvas(true, false);
                    }
                    return true;
                }
            }
            if (this._customShowSelectedButtonRect) {
                const ax = nodePos[0] + this._customShowSelectedButtonRect.x;
                const ay = nodePos[1] + this._customShowSelectedButtonRect.y;
                const aw = this._customShowSelectedButtonRect.width;
                const ah = this._customShowSelectedButtonRect.height;
                if (e.canvasX >= ax && e.canvasX <= ax + aw && e.canvasY >= ay && e.canvasY <= ay + ah) {
                    e.preventDefault();
                    e.stopPropagation();
                    if (!this._customShowOnlySelected) {
                        if (!this._customSelectedImages || !this._customSelectedImages.some(v => v)) {
                            alert('当前没有勾选的图片，无法仅显示勾选。');
                            return true;
                        }
                        this._customShowOnlySelected = true;
                    } else {
                        this._customShowOnlySelected = false;
                    }
                    if (this._customImagePaths && this._customImagePaths.length > 0) {
                        calculateImageLayout(this, this._customImagePaths.length);
                        app.graph.setDirtyCanvas(true, false);
                    }
                    return true;
                }
            }
            if (this._customReuseMaskButtonRect) {
                const ax = nodePos[0] + this._customReuseMaskButtonRect.x;
                const ay = nodePos[1] + this._customReuseMaskButtonRect.y;
                const aw = this._customReuseMaskButtonRect.width;
                const ah = this._customReuseMaskButtonRect.height;
                if (e.canvasX >= ax && e.canvasX <= ax + aw && e.canvasY >= ay && e.canvasY <= ay + ah) {
                    e.preventDefault();
                    e.stopPropagation();
                    this._customMaskReuseEnabled = !this._customMaskReuseEnabled;
                    const widget = this.widgets.find(w => w.name === "reuse_mask");
                    if (widget) {
                        widget.value = !!this._customMaskReuseEnabled;
                    }
                    return true;
                }
            }
        }
        
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
                    
                    console.log("点击恢复按钮，退出单图片模式");
                    
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
            
            // 检查点击左下角清除按钮（单图片模式）
            if (this._customClearButtonRect) {
                const absClearButtonX = nodePos[0] + this._customClearButtonRect.x;
                const absClearButtonY = nodePos[1] + this._customClearButtonRect.y;
                const absClearButtonWidth = this._customClearButtonRect.width;
                const absClearButtonHeight = this._customClearButtonRect.height;
                
                if (e.canvasX >= absClearButtonX && e.canvasX <= absClearButtonX + absClearButtonWidth &&
                    e.canvasY >= absClearButtonY && e.canvasY <= absClearButtonY + absClearButtonHeight) {
                    
                    console.log(`点击左下角清除按钮，图片索引: ${this._customFocusedImageIndex}`);
                    
                    // 阻止事件冒泡
                    e.preventDefault();
                    e.stopPropagation();
                    
                    // 执行清除操作
                    this.executeClear(this._customFocusedImageIndex);
                    
                    return true;
                }
            }
            
            // 检查点击右下角全屏预览按钮（单图片模式）
            if (this._customFullscreenButtonRect) {
                const absFullscreenButtonX = nodePos[0] + this._customFullscreenButtonRect.x;
                const absFullscreenButtonY = nodePos[1] + this._customFullscreenButtonRect.y;
                const absFullscreenButtonWidth = this._customFullscreenButtonRect.width;
                const absFullscreenButtonHeight = this._customFullscreenButtonRect.height;
                
                if (e.canvasX >= absFullscreenButtonX && e.canvasX <= absFullscreenButtonX + absFullscreenButtonWidth &&
                    e.canvasY >= absFullscreenButtonY && e.canvasY <= absFullscreenButtonY + absFullscreenButtonHeight) {
                    
                    console.log(`点击全屏预览按钮，图片索引: ${this._customFocusedImageIndex}`);
                    
                    // 阻止事件冒泡
                    e.preventDefault();
                    e.stopPropagation();
                    
                    // 执行全屏预览
                    if (this._customImagePaths && this._customImagePaths.length > 0) {
                        showImageLightbox(this._customImagePaths, this._customFocusedImageIndex);
                    }
                    
                    return true;
                }
            }
        }
                
        // 检查是否点击清除按钮（多图片模式）
        if (this._customClearButtonRects && this._customClearButtonRects.length > 0) {
            for (let i = 0; i < this._customClearButtonRects.length; i++) {
                const clearRect = this._customClearButtonRects[i];

                // 检查清除按钮是否存在（只在悬浮时存在）
                if (!clearRect) {
                    continue;
                }

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
                    this.executeClear(i);
                    
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
                    
                    // 阻止事件冒泡，避免触发节点选择
                    e.preventDefault();
                    e.stopPropagation();
                    
                    // 点击图片进入单图片模式
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
        
        // 如果没有处理图片区域点击，调用原始事件处理
        if (originalOnMouseDown) {
            return originalOnMouseDown.call(this, e);
        }
        
        return false;
    };
    
    // 添加双击事件处理
    const originalOnDblClick = this.onDblClick;
    this.onDblClick = function(e) {
        // 只有LoadImageBatchAdvanced节点才处理自定义双击事件
        if (this.type !== "LoadImageBatchAdvanced") {
            if (originalOnDblClick) {
                return originalOnDblClick.call(this, e);
            }
            return false;
        }
        
        console.log("onDblClick 被调用", e);
        
        // 获取节点的Canvas坐标
        const nodePos = this.pos;
        
        // 检查是否双击图片区域（单图片模式）
        if (this._customSingleImageMode && this._customImageRects && this._customImageRects.length > 0) {
            const currentImageRect = this._customImageRects[this._customFocusedImageIndex];
            
            if (currentImageRect && currentImageRect.visible !== false) {
                // 计算图片区域在Canvas中的绝对坐标
                const absRectX = nodePos[0] + currentImageRect.x;
                const absRectY = nodePos[1] + currentImageRect.y;
                const absRectWidth = currentImageRect.width;
                const absRectHeight = currentImageRect.height;
                
                // 检查鼠标是否在图片区域内
                if (e.canvasX >= absRectX && e.canvasX <= absRectX + absRectWidth &&
                    e.canvasY >= absRectY && e.canvasY <= absRectY + absRectHeight) {
                    
                    console.log(`双击图片，进入全屏预览，图片索引: ${this._customFocusedImageIndex}`);
                    
                    // 阻止事件冒泡
                    e.preventDefault();
                    e.stopPropagation();
                    
                    // 执行全屏预览
                    if (this._customImagePaths && this._customImagePaths.length > 0) {
                        showImageLightbox(this._customImagePaths, this._customFocusedImageIndex);
                    }
                    
                    return true;
                }
            }
        }
        
        // 如果没有处理双击事件，调用原始事件处理
        if (originalOnDblClick) {
            return originalOnDblClick.call(this, e);
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
        
        // 重新计算图片布局，适应新的节点大小
        if (this._customImagePaths && this._customImagePaths.length > 0) {
            calculateImageLayout(this, this._customImagePaths.length);
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
        const t1 = document.getElementById('image-tooltip-' + this.id);
        if (t1) t1.remove();
        const t2 = document.getElementById('control-tooltip-' + this.id);
        if (t2) t2.remove();
    };
    this.showControlTooltip = function(e, content) {
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
    
    // 从原始图片路径数组中移除指定索引的路径
    if (this._customImagePaths && imageIndex < this._customImagePaths.length) {
        this._customImagePaths.splice(imageIndex, 1);
        
        // 同时移除对应的选择状态
        if (this._customSelectedImages && imageIndex < this._customSelectedImages.length) {
            this._customSelectedImages.splice(imageIndex, 1);
        }
        
        // 同时移除对应的文件名
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

/**
 * 打开 MaskEditor 编辑任意图片
 * @param {string} imageUrl - 图片的 URL
 * @param {Function} onSave - 保存回调，接收 (filename, subfolder, type) 或 路径字符串
 */
function openMaskEditorForImage(imageUrl, onSave) {
    // 1. 构造 Mock Node
    const mockNode = {
        id: -1, // 虚拟 ID
        type: "MockNode",
        title: "Mock Image Editor",
        imgs: [{
            src: imageUrl,
            width: 512, // 临时宽高，MaskEditor 会重新加载图片
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

    // 2. 找到 MaskEditor 的打开命令
    const ext = app.extensions.find(e => e.name === "Comfy.MaskEditor");
    if (!ext) {
        console.error("Comfy.MaskEditor extension not found");
        alert("未找到 MaskEditor 插件，请先安装。");
        return;
    }

    const cmd = ext.commands.find(c => c.id === "Comfy.MaskEditor.OpenMaskEditor");
    if (!cmd) {
        console.error("OpenMaskEditor command not found");
        alert("MaskEditor 插件未注册打开命令。");
        return;
    }

    // 3. 实施欺骗：伪造选中状态并执行命令
    const originalSelection = app.canvas.selected_nodes;
    app.canvas.selected_nodes = { [mockNode.id]: mockNode };

    try {
        cmd.function();
    } catch (e) {
        console.error("Failed to open MaskEditor:", e);
        alert("打开编辑器失败: " + e.message);
    } finally {
        app.canvas.selected_nodes = originalSelection;
    }
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
                const pathUseWidget = node.widgets.find((w) => w.name === "image_path_use");
                if (pathWidget) pathWidget.hidden = true;
                if (pathUseWidget) pathUseWidget.hidden = false;

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
                                const joined = allPaths.join(',');
                                pathWidget.value = joined;
                                if (pathUseWidget) pathUseWidget.value = joined;
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

                // 弹出"追加/替换"选择对话框（有现有图片时）
                function askAppendOrReplaceIfNeeded(existingList, incomingCount) {
                    return new Promise((resolve) => {
                        // 没有现有列表或为空，直接替换为新列表
                        if (!existingList || existingList.length === 0) {
                            return resolve('replace');
                        }
                        // 构建对话框
                        const dialog = document.createElement('div');
                        dialog.id = 'append-or-replace-' + node.id;
                        dialog.style.cssText = `
                            position: fixed;
                            top: 50%; left: 50%; transform: translate(-50%, -50%);
                            background: #2a2a2a; border: 2px solid #666; border-radius: 8px;
                            padding: 20px; z-index: 10001; color: white; max-width: 420px;
                        `;
                        dialog.innerHTML = `
                            <h3 style="margin:0 0 12px 0;">检测到已有图片</h3>
                            <p style="margin:0 0 12px 0;">当前已有 <strong>${existingList.length}</strong> 张图片，将要添加 <strong>${incomingCount}</strong> 张图片。</p>
                            <p style="margin:0 0 12px 0;">请选择如何处理：</p>
                            <div style="display:flex; gap:10px; justify-content:flex-end; margin-top:10px;">
                                <button id="append-btn-${node.id}" style="padding:8px 14px; background:#4CAF50; color:#fff; border:none; border-radius:4px; cursor:pointer;">追加</button>
                                <button id="replace-btn-${node.id}" style="padding:8px 14px; background:#ff6b6b; color:#fff; border:none; border-radius:4px; cursor:pointer;">替换</button>
                                <button id="cancel-btn-${node.id}" style="padding:8px 14px; background:#666; color:#fff; border:none; border-radius:4px; cursor:pointer;">取消</button>
                            </div>
                        `;
                        const overlay = document.createElement('div');
                        overlay.style.cssText = `position:fixed; inset:0; background:rgba(0,0,0,.5); z-index:10000;`;
                        document.body.appendChild(overlay);
                        document.body.appendChild(dialog);
                        const cleanup = (val) => {
                            dialog.remove(); overlay.remove(); resolve(val);
                        };
                        document.getElementById(`append-btn-${node.id}`).onclick = () => cleanup('append');
                        document.getElementById(`replace-btn-${node.id}`).onclick = () => cleanup('replace');
                        document.getElementById(`cancel-btn-${node.id}`).onclick = () => cleanup('cancel');
                        overlay.onclick = () => cleanup('cancel');
                    });
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
                async function handleIncomingFiles(files) {
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
                            const choice = await askAppendOrReplaceIfNeeded(oldList, newPaths.length);
                            if (choice === 'cancel') return;
                            mode = choice === 'append' ? 'append' : 'replace';
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

                // 跟踪悬浮（用于粘贴时判断目标）
                chainCallback(nodeType.prototype, "onMouseMove", function(e) {
                    this._customIsHovered = true;
                });
                chainCallback(nodeType.prototype, "onMouseLeave", function(e) {
                    this._customIsHovered = false;
                });

                // 文档级粘贴：若节点被选中或悬浮，并且剪贴板有图片，则拦截为本节点上传
                if (!window.__A_MY_NODES_LOAD_IMAGE_BATCH_PASTE_INSTALLED__) {
                    window.__A_MY_NODES_LOAD_IMAGE_BATCH_PASTE_INSTALLED__ = true;
                    document.addEventListener('paste', async (evt) => {
                        try {
                            // 当前活跃的 LiteGraph 节点集合里，优先找到"选中或悬浮且类型匹配"的节点
                            const graph = app?.graph;
                            if (!graph || !graph._nodes) return; // 兜底
                            const candidates = graph._nodes.filter(n => n && n.type === 'LoadImageBatchAdvanced');
                            if (!candidates.length) return;

                            // 取"被选中优先，否则悬浮"的目标节点
                            const target = candidates.find(n => n.selected) || candidates.find(n => n._customIsHovered);
                            if (!target) return;

                            const files = getImageFilesFromClipboard(evt.clipboardData);
                            if (!files || files.length === 0) return;

                            // 拦截默认粘贴（否则会走官方创建 LoadImage 节点的逻辑）
                            evt.preventDefault();
                            evt.stopPropagation();

                            // 在目标节点环境执行
                            await handleIncomingFiles.call(target, files);
                        } catch (err) {
                            console.warn('paste 处理异常:', err);
                        }
                    }, true); // 捕获阶段，优先于全局处理
                }

                // 将内部处理方法暴露到实例，供右键菜单调用
                this._customHandleIncomingFiles = handleIncomingFiles; // 处理文件入口
                this._customAskAppendOrReplaceIfNeeded = askAppendOrReplaceIfNeeded; // 选择对话框
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

            // 新增：为节点追加右键菜单"粘贴"项（与官方 Load Image 一致的入口）
            chainCallback(nodeType.prototype, "getExtraMenuOptions", function(_, options) {
                const self = this;

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
                            content: "编辑图片 (MaskEditor)",
                            callback: () => {
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
                                alert('剪贴板中没有图片或浏览器不支持从右键菜单读取图片，请使用 Ctrl+V 粘贴。');
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
                    const useWidget = this.widgets.find(w => w.name === "image_path_use");
                    if (useWidget && (!useWidget.value || !String(useWidget.value).trim())) {
                        useWidget.value = imagePathsWidget.value;
                    }
                    const paths = imagePathsWidget.value.split(',').filter(path => path.trim());
                    if (paths.length > 0) {
                        populate.call(this, paths);
                    }
                }
            });
            
            // 添加鼠标事件处理（只在有图片数据时处理）
            chainCallback(nodeType.prototype, "onMouseMove", function(e) {
                // 只有LoadImageBatchAdvanced节点且有图片数据时才处理
                if (this.type === "LoadImageBatchAdvanced" && this._customImgs && this._customImgs.length > 0) {
                    // 计算新的鼠标位置
                    const newMouseX = e.canvasX - this.pos[0];
                    const newMouseY = e.canvasY - this.pos[1];
                    
                    // 检查鼠标位置是否真的改变了
                    const mousePositionChanged = this._customMouseX !== newMouseX || this._customMouseY !== newMouseY;
                    
                    // 保存鼠标位置用于悬浮检测
                    this._customMouseX = newMouseX;
                    this._customMouseY = newMouseY;
                    
                    // 只在鼠标位置真正改变时才触发重绘
                    if (mousePositionChanged) {
                        app.graph.setDirtyCanvas(true, false);
                    }
                }
            });
            
            // 鼠标离开时清除位置
            chainCallback(nodeType.prototype, "onMouseLeave", function(e) {
                // 只有LoadImageBatchAdvanced节点且有图片数据时才处理
                if (this.type === "LoadImageBatchAdvanced" && this._customImgs && this._customImgs.length > 0) {
                    // 清除鼠标位置
                    this._customMouseX = undefined;
                    this._customMouseY = undefined;
                    
                    // 触发重绘以隐藏指示器
                    app.graph.setDirtyCanvas(true, false);
                }
            });
            
            // 处理鼠标点击事件
            chainCallback(nodeType.prototype, "onMouseDown", function(e) {
                // 获取节点的Canvas坐标
                const nodePos = this.pos;
                
                // 检查是否点击清除按钮
                if (this._customClearButtonRects && this._customClearButtonRects.length > 0) {
                    for (let i = 0; i < this._customClearButtonRects.length; i++) {
                        const clearRect = this._customClearButtonRects[i];
                        
                        // 检查清除按钮是否存在（只在悬浮时存在）
                        if (!clearRect) {
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
                            this.executeClear(i);
                            
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
                
                // 清理自定义绘制方法标记
                this._customDrawMethodSet = false;
                
                // 清理自定义属性
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
