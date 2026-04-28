import { getAdjustedFontSize } from "../utils/common.js";
import { getCustomButtons } from "./image_manager.js";
import { computeButtonLayout } from "./layout.js";

/**
 * 在Canvas上绘制图片
 * @param {object} node - LiteGraph节点实例
 * @param {CanvasRenderingContext2D} ctx - Canvas上下文
 */
export function drawNodeImages(node, ctx) {
    if (!node._customImgs || !node._customImageRects) return;
    
    ctx.save();
    
    for (let i = 0; i < node._customImgs.length && i < node._customImageRects.length; i++) {
        const img = node._customImgs[i];
        const rect = node._customImageRects[i];
        
        if (rect.visible === false) {
            continue;
        }
        
        ctx.fillStyle = '#2a2a2a';
        ctx.fillRect(rect.x, rect.y, rect.width, rect.height);
        
        ctx.strokeStyle = '#666';
        ctx.lineWidth = 1;
        ctx.strokeRect(rect.x, rect.y, rect.width, rect.height);
        
        if (img.complete && img.naturalWidth > 0) {
            try {
                const imageAspectRatio = img.naturalWidth / img.naturalHeight;
                const rectAspectRatio = rect.width / rect.height;
                
                let drawWidth, drawHeight, drawX, drawY;
                
                if (imageAspectRatio > rectAspectRatio) {
                    drawWidth = rect.width;
                    drawHeight = rect.width / imageAspectRatio;
                    drawX = rect.x;
                    drawY = rect.y + (rect.height - drawHeight) / 2;
                } else {
                    drawHeight = rect.height;
                    drawWidth = rect.height * imageAspectRatio;
                    drawX = rect.x + (rect.width - drawWidth) / 2;
                    drawY = rect.y;
                }
                
                ctx.drawImage(img, drawX, drawY, drawWidth, drawHeight);
                
                ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
                ctx.lineWidth = 1;
                ctx.strokeRect(drawX, drawY, drawWidth, drawHeight);
            } catch (e) {
                console.warn(`绘制图片失败: ${e.message}`);
            }
        } else if (img.complete) {
            ctx.fillStyle = '#3a1a1a';
            ctx.fillRect(rect.x, rect.y, rect.width, rect.height);
            
            ctx.fillStyle = '#ff6666';
            ctx.font = 'bold 12px Arial';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText('Load Error', rect.x + rect.width / 2, rect.y + rect.height / 2);
            
            ctx.font = '24px Arial';
            ctx.fillText('⚠️', rect.x + rect.width / 2, rect.y + rect.height / 2 - 20);
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

        if (img.complete && img.naturalWidth > 0) {
            const dimText = `${img.naturalWidth}x${img.naturalHeight}`;
            ctx.font = '12px Arial';
            const dimTextWidth = ctx.measureText(dimText).width;
            const dimPadding = 4;
            const dimBgWidth = dimTextWidth + dimPadding * 2;
            const dimBgHeight = 20;
            
            const dimX = rect.x + (rect.width - dimBgWidth) / 2;
            const dimY = rect.y + rect.height - dimBgHeight - 5;
            
            ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
            ctx.beginPath();
            const r = 4;
            ctx.moveTo(dimX + r, dimY);
            ctx.lineTo(dimX + dimBgWidth - r, dimY);
            ctx.quadraticCurveTo(dimX + dimBgWidth, dimY, dimX + dimBgWidth, dimY + r);
            ctx.lineTo(dimX + dimBgWidth, dimY + dimBgHeight - r);
            ctx.quadraticCurveTo(dimX + dimBgWidth, dimY + dimBgHeight, dimX + dimBgWidth - r, dimY + dimBgHeight);
            ctx.lineTo(dimX + r, dimY + dimBgHeight);
            ctx.quadraticCurveTo(dimX, dimY + dimBgHeight, dimX, dimY + dimBgHeight - r);
            ctx.lineTo(dimX, dimY + r);
            ctx.quadraticCurveTo(dimX, dimY, dimX + r, dimY);
            ctx.closePath();
            ctx.fill();
            
            ctx.fillStyle = '#fff';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(dimText, dimX + dimBgWidth / 2, dimY + dimBgHeight / 2);
        }

        if (!node._customSingleImageMode) {
            const editClickW = rect.width / 3;
            const editClickH = rect.height / 3;
            const editClickX = rect.x + rect.width - editClickW;
            const editClickY = rect.y + rect.height - editClickH;
            
            const mouseInEditArea = node._customMouseX !== undefined && node._customMouseY !== undefined &&
                node._customMouseX >= editClickX && node._customMouseX <= editClickX + editClickW &&
                node._customMouseY >= editClickY && node._customMouseY <= editClickY + editClickH;
                
            const buttonSize = 16;
            const buttonMargin = 5;
            const editButtonX = rect.x + rect.width - buttonMargin - buttonSize;
            const editButtonY = rect.y + rect.height - buttonMargin - buttonSize;
            
            ctx.fillStyle = mouseInEditArea ? 'rgba(0, 0, 0, 0.8)' : 'rgba(0, 0, 0, 0.5)';
            ctx.beginPath();
            ctx.arc(editButtonX + buttonSize/2, editButtonY + buttonSize/2, buttonSize/2, 0, 2 * Math.PI);
            ctx.fill();
            
            ctx.strokeStyle = mouseInEditArea ? 'rgba(255, 255, 255, 1)' : 'rgba(255, 255, 255, 0.8)';
            ctx.lineWidth = mouseInEditArea ? 2 : 1;
            ctx.stroke();
            
            ctx.fillStyle = 'rgba(255, 255, 255, 1)';
            ctx.font = `${buttonSize - 6}px Arial`;
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText('⛶', editButtonX + buttonSize/2, editButtonY + buttonSize/2);
            
            if (!node._customEditButtonRects) {
                node._customEditButtonRects = [];
            }
            node._customEditButtonRects[i] = {
                x: editClickX,
                y: editClickY,
                width: editClickW,
                height: editClickH
            };

            const mouseInImage = node._customMouseX !== undefined && node._customMouseY !== undefined &&
                node._customMouseX >= rect.x && node._customMouseX <= rect.x + rect.width &&
                node._customMouseY >= rect.y && node._customMouseY <= rect.y + rect.height;
            
            if (mouseInImage) {
                ctx.textAlign = 'center';
                const fileName = node._customImageFileNames && node._customImageFileNames[i] ? node._customImageFileNames[i] : 'Unknown';
                
                ctx.fillStyle = 'rgba(0, 0, 0, 0.6)';
                ctx.fillRect(rect.x, rect.y, rect.width, 30);
                
                const maxTextWidth = rect.width - 10;
                const fontSize = getAdjustedFontSize(ctx, fileName, maxTextWidth);
                ctx.font = `bold ${fontSize}px Arial`;
                
                ctx.fillStyle = '#fff';
                ctx.fillText(fileName, rect.x + rect.width / 2, rect.y + 20);
                
                const clearButtonX = rect.x + rect.width - buttonMargin - buttonSize;
                const clearButtonY = rect.y + buttonMargin;
                
                const mouseInClearButton = node._customMouseX >= clearButtonX && node._customMouseX <= clearButtonX + buttonSize &&
                    node._customMouseY >= clearButtonY && node._customMouseY <= clearButtonY + buttonSize;
                
                ctx.fillStyle = mouseInClearButton ? 'rgba(255, 0, 0, 0.9)' : 'rgba(255, 0, 0, 0.7)';
                ctx.beginPath();
                ctx.arc(clearButtonX + buttonSize/2, clearButtonY + buttonSize/2, buttonSize/2, 0, 2 * Math.PI);
                ctx.fill();
                
                ctx.strokeStyle = mouseInClearButton ? 'rgba(255, 255, 255, 1)' : 'rgba(255, 255, 255, 0.8)';
                ctx.lineWidth = mouseInClearButton ? 2 : 1;
                ctx.stroke();
                
                ctx.fillStyle = 'rgba(255, 255, 255, 1)';
                ctx.font = `${buttonSize - 4}px Arial`;
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText('×', clearButtonX + buttonSize/2, clearButtonY + buttonSize/2);

                if (!node._customClearButtonRects) {
                    node._customClearButtonRects = [];
                }
                node._customClearButtonRects[i] = {
                    x: clearButtonX,
                    y: clearButtonY,
                    width: buttonSize,
                    height: buttonSize
                };
                
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
                if (!node._customClearButtonRects) {
                    node._customClearButtonRects = [];
                }
                node._customClearButtonRects[i] = null;
                
                if (!node._customFileNameRects) {
                    node._customFileNameRects = [];
                }
                node._customFileNameRects[i] = null;
            }
        } else {
            if (!node._customClearButtonRects) {
                node._customClearButtonRects = [];
            }
            node._customClearButtonRects[i] = null;
            
            if (!node._customEditButtonRects) {
                node._customEditButtonRects = [];
            }
            node._customEditButtonRects[i] = null;
            
            if (!node._customFileNameRects) {
                node._customFileNameRects = [];
            }
            node._customFileNameRects[i] = null;
        }
    }
    
    if (node._customSingleImageMode) {
        const buttonSize = 20;
        const buttonSpacing = 12;
        const cornerSafe = 22;
        
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
        
        const mouseInClearButton = node._customMouseX !== undefined && node._customMouseY !== undefined &&
            node._customMouseX >= 10 && node._customMouseX <= 10 + buttonSize &&
            node._customMouseY >= node.size[1] - buttonSize - 10 && node._customMouseY <= node.size[1] - 10;
        
        const mouseInFullscreenButton = node._customMouseX !== undefined && node._customMouseY !== undefined &&
            node._customMouseX >= node.size[0] - buttonSize - 10 && node._customMouseX <= node.size[0] - 10 &&
            node._customMouseY >= node.size[1] - buttonSize - 10 && node._customMouseY <= node.size[1] - 10;
        
        if (node._customImagePaths && node._customImagePaths.length > 1 && 
            node._customFocusedImageIndex >= 0 && node._customFocusedImageIndex < node._customImagePaths.length) {
            const currentIndex = node._customFocusedImageIndex + 1;
            const totalCount = node._customImagePaths.length;
            const indexText = `(${currentIndex}/${totalCount})`;
            
            ctx.fillStyle = 'rgba(255, 255, 255, 0.9)';
            ctx.font = '12px Arial';
            ctx.textAlign = 'right';
            ctx.textBaseline = 'middle';
            
            const indexX = node.size[0] - buttonSize * 3 - buttonSpacing * 2 - 15 - cornerSafe;
            const indexY = node.size[1] - buttonSize - 10 + buttonSize / 2;
            
            ctx.fillText(indexText, indexX, indexY);
        }
        
        const prevButtonX = node.size[0] - buttonSize * 3 - buttonSpacing * 2 - 10 - cornerSafe;
        const prevButtonY = node.size[1] - buttonSize - 10;
        
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
        
        const nextButtonX = node.size[0] - buttonSize * 2 - buttonSpacing - 10 - cornerSafe;
        const nextButtonY = node.size[1] - buttonSize - 10;
        
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
        
        ctx.fillStyle = 'rgba(0, 0, 0, 0)';
        ctx.fillRect(restoreButtonX, restoreButtonY, buttonSize, buttonSize);
        ctx.strokeStyle = mouseInRestoreButton ? 'rgba(255, 255, 255, 1)' : 'rgba(255, 255, 255, 0.8)';
        ctx.lineWidth = mouseInRestoreButton ? 2 : 1;
        ctx.strokeRect(restoreButtonX, restoreButtonY, buttonSize, buttonSize);
        ctx.fillStyle = 'rgba(255, 255, 255, 1)';
        ctx.fillText('⭯', restoreButtonX + buttonSize / 2, restoreButtonY + buttonSize / 2);
        
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
        
        if (node._customImageFileNames && node._customImageFileNames[node._customFocusedImageIndex]) {
            const fileName = node._customImageFileNames[node._customFocusedImageIndex];
            ctx.fillStyle = 'rgba(255, 255, 255, 0.9)';
            ctx.font = '12px Arial';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            const fileNameY = node.size[1] - 15;
            ctx.fillText(fileName, node.size[0] / 2, fileNameY);
        }
    } else {
        const buttons = getCustomButtons(node);
        const layout = computeButtonLayout(node, buttons);
        const startY = node.size[1] - layout.totalHeight + 5; 

        node._customButtons = [];
        node._customSelectAllButtonRect = null;
        node._customDeselectAllButtonRect = null;
        node._customInvertSelectionButtonRect = null;
        node._customClearSelectedButtonRect = null;
        node._customClearUnselectedButtonRect = null;
        node._customShowSelectedButtonRect = null;
        node._customCopyButtonRect = null;
        node._customReuseMaskButtonRect = null;

        const r = 6;
        
        layout.rows.forEach((row, rowIndex) => {
            let x = layout.startX;
            let y = startY + rowIndex * (layout.buttonHeight + layout.rowSpacing);

            row.forEach(btn => {
                const hover = node._customMouseX >= x && node._customMouseX <= x + btn.width &&
                              node._customMouseY >= y && node._customMouseY <= y + layout.buttonHeight;
                
                const h = layout.buttonHeight;
                ctx.fillStyle = hover ? 'rgba(235,235,240,0.95)' : 'rgba(235,235,240,0.85)';
                ctx.strokeStyle = hover ? 'rgba(80,80,90,0.9)' : 'rgba(120,120,130,0.8)';
                ctx.lineWidth = hover ? 2 : 1;
                ctx.beginPath();
                ctx.moveTo(x + r, y);
                ctx.lineTo(x + btn.width - r, y);
                ctx.quadraticCurveTo(x + btn.width, y, x + btn.width, y + r);
                ctx.lineTo(x + btn.width, y + h - r);
                ctx.quadraticCurveTo(x + btn.width, y + h, x + btn.width - r, y + h);
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
                ctx.fillText(btn.text, x + btn.width / 2, y + h / 2);

                node._customButtons.push({
                    rect: { x, y, w: btn.width, h: layout.buttonHeight },
                    callback: btn.callback,
                    tooltip: btn.tooltip
                });
                
                x += btn.width + layout.buttonSpacing;
            });
        });
        
        node._customPrevButtonRect = null;
        node._customNextButtonRect = null;
        node._customRestoreButtonRect = null;
        node._customClearButtonRect = null;
        node._customFullscreenButtonRect = null;
    }
    
    if (node._customHoverKeyStatus && node._customMouseX !== undefined && node._customMouseY !== undefined) {
        const text = node._customHoverKeyStatus;
        ctx.font = "12px Arial";
        const textWidth = ctx.measureText(text).width;
        const padding = 6;
        const boxHeight = 24;
        const boxWidth = textWidth + padding * 2;
        
        const x = node._customMouseX + 16;
        const y = node._customMouseY;
        
        ctx.fillStyle = "rgba(0, 0, 0, 0.85)";
        ctx.strokeStyle = "rgba(255, 255, 255, 0.6)";
        ctx.lineWidth = 1;
        
        ctx.beginPath();
        const r = 4;
        ctx.moveTo(x + r, y);
        ctx.lineTo(x + boxWidth - r, y);
        ctx.quadraticCurveTo(x + boxWidth, y, x + boxWidth, y + r);
        ctx.lineTo(x + boxWidth, y + boxHeight - r);
        ctx.quadraticCurveTo(x + boxWidth, y + boxHeight, x + boxWidth - r, y + boxHeight);
        ctx.lineTo(x + r, y + boxHeight);
        ctx.quadraticCurveTo(x, y + boxHeight, x, y + boxHeight - r);
        ctx.lineTo(x, y + r);
        ctx.quadraticCurveTo(x, y, x + r, y);
        ctx.closePath();
        
        ctx.fill();
        ctx.stroke();
        
        ctx.fillStyle = "#ffffff";
        ctx.textAlign = "left";
        ctx.textBaseline = "middle";
        ctx.fillText(text, x + padding, y + boxHeight / 2);
    }

    ctx.restore();
}
