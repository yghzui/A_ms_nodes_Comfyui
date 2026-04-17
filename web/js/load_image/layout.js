import { getCustomButtons } from "./image_manager.js";

/**
 * 计算按钮布局
 */
export function computeButtonLayout(node, buttons) {
    const buttonHeight = 25;
    const buttonSpacing = 10;
    const rowSpacing = 5;
    const startX = 10;
    const nodeWidth = node.size[0];

    const rows = [];
    let currentRow = [];
    let currentRowWidth = startX;

    buttons.forEach(btn => {
        if (currentRowWidth + btn.width + buttonSpacing > nodeWidth - 10) { // -10 padding right
            if (currentRow.length > 0) {
                rows.push(currentRow);
                currentRow = [];
                currentRowWidth = startX;
            }
        }
        currentRow.push(btn);
        currentRowWidth += btn.width + buttonSpacing;
    });
    if (currentRow.length > 0) rows.push(currentRow);

    const totalHeight = rows.length * buttonHeight + Math.max(0, rows.length - 1) * rowSpacing + 10; // +10 padding bottom
    
    return { rows, totalHeight, buttonHeight, buttonSpacing, rowSpacing, startX };
}

/**
 * 计算图片网格布局，支持单图片模式和多图片模式
 * @param {object} node - LiteGraph节点实例
 * @param {number} imageCount - 图片数量
 */
export function calculateImageLayout(node, imageCount) {
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
    const TOP_MARGIN = 210; // 再向下腾挪空间，容纳新增控件（如"应用透明到图像"开关）
    const TITLE_HEIGHT = 25; // 图片标题的高度
    
    // 计算底部控制按钮的高度
    let BOTTOM_CONTROLS_HEIGHT = 0;
    if (node._customSingleImageMode) {
        BOTTOM_CONTROLS_HEIGHT = 0; // 单图模式下，底部已经由 PADDING(8) + TITLE_HEIGHT(25) 预留了 33px 的空间，足够显示按钮和文件名
    } else {
        const buttons = getCustomButtons(node);
        const layout = computeButtonLayout(node, buttons);
        BOTTOM_CONTROLS_HEIGHT = layout.totalHeight; // 底部控制按钮的高度
    }
    
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
        // 单图片模式：只显示一个图片，最大化显示，利用全部可用宽和高
        const x = PADDING;
        const y = PADDING + TOP_MARGIN;
        
        node._customImageRects = [];
        for (let i = 0; i < imageCount; i++) {
            if (i === node._customFocusedImageIndex) {
                // 显示聚焦的图片
                node._customImageRects.push({
                    x: x,
                    y: y,
                    width: availableWidth,
                    height: availableHeight,
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
