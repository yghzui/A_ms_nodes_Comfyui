// 本文件由主项目 tools/sync_image_editor_core.py 自动生成，禁止手改。
// 源文件: web_apps/static/image_editor/crop_utils.js
export const clamp = (value, min, max) => Math.min(Math.max(value, min), max);

export const cloneRect = (rect) => rect ? { x: rect.x, y: rect.y, width: rect.width, height: rect.height } : null;

export const roundRect = (rect) => rect ? {
    x: Math.round(rect.x),
    y: Math.round(rect.y),
    width: Math.round(rect.width),
    height: Math.round(rect.height)
} : null;

export const getHandleSize = (scale, hitBoost = 0) => Math.max(10 + hitBoost, (12 + hitBoost) / Math.max(scale, 0.25));

const getCurrentCropAspectRatio = (cropPreset, canvasW, canvasH) => {
    if (cropPreset === 'free') return null;
    if (cropPreset === 'original') return canvasH ? canvasW / canvasH : null;
    const parts = String(cropPreset || '').split(':').map(Number);
    if (parts.length === 2 && parts[0] > 0 && parts[1] > 0) return parts[0] / parts[1];
    return null;
};

const normalizeRect = (rect) => {
    let x = rect.x;
    let y = rect.y;
    let width = rect.width;
    let height = rect.height;
    if (width < 0) {
        x += width;
        width = Math.abs(width);
    }
    if (height < 0) {
        y += height;
        height = Math.abs(height);
    }
    return { x, y, width, height };
};

export const clampRectToCanvas = (rect, canvasW, canvasH, minSize = 1) => {
    const normalized = normalizeRect(rect);
    return {
        x: normalized.x,
        y: normalized.y,
        width: Math.max(minSize, normalized.width),
        height: Math.max(minSize, normalized.height)
    };
};

export const createDefaultCropRect = (cropPreset, canvasW, canvasH) => {
    if (!canvasW || !canvasH) return null;
    const aspectRatio = getCurrentCropAspectRatio(cropPreset, canvasW, canvasH);
    const padding = 0.08;
    const maxWidth = Math.max(1, canvasW * (1 - padding * 2));
    const maxHeight = Math.max(1, canvasH * (1 - padding * 2));
    if (!aspectRatio) {
        return { x: canvasW * padding, y: canvasH * padding, width: maxWidth, height: maxHeight };
    }
    let width = maxWidth;
    let height = width / aspectRatio;
    if (height > maxHeight) {
        height = maxHeight;
        width = height * aspectRatio;
    }
    return { x: (canvasW - width) / 2, y: (canvasH - height) / 2, width, height };
};

export const getCropHandleAt = (cropRect, scale, pos, hitBoost = 0) => {
    if (!cropRect) return 'new';
    const handleSize = getHandleSize(scale, hitBoost);
    const half = handleSize / 2;
    const corners = [
        { id: 'nw', x: cropRect.x, y: cropRect.y },
        { id: 'ne', x: cropRect.x + cropRect.width, y: cropRect.y },
        { id: 'sw', x: cropRect.x, y: cropRect.y + cropRect.height },
        { id: 'se', x: cropRect.x + cropRect.width, y: cropRect.y + cropRect.height }
    ];
    for (const corner of corners) {
        if (pos.x >= corner.x - half && pos.x <= corner.x + half && pos.y >= corner.y - half && pos.y <= corner.y + half) {
            return corner.id;
        }
    }
    const edgePadding = Math.max(6 + hitBoost / 2, (8 + hitBoost) / Math.max(scale, 0.25));
    const withinX = pos.x >= cropRect.x && pos.x <= cropRect.x + cropRect.width;
    const withinY = pos.y >= cropRect.y && pos.y <= cropRect.y + cropRect.height;
    if (withinX && Math.abs(pos.y - cropRect.y) <= edgePadding) return 'n';
    if (withinX && Math.abs(pos.y - (cropRect.y + cropRect.height)) <= edgePadding) return 's';
    if (withinY && Math.abs(pos.x - cropRect.x) <= edgePadding) return 'w';
    if (withinY && Math.abs(pos.x - (cropRect.x + cropRect.width)) <= edgePadding) return 'e';
    if (withinX && withinY) return 'move';
    return 'new';
};

export const getCropCursor = (handle) => ({
    move: 'move',
    nw: 'nwse-resize',
    se: 'nwse-resize',
    ne: 'nesw-resize',
    sw: 'nesw-resize',
    n: 'ns-resize',
    s: 'ns-resize',
    e: 'ew-resize',
    w: 'ew-resize',
    new: 'crosshair'
}[handle] || 'crosshair');

export const buildAspectRectFromAnchor = (anchor, point, cropPreset, canvasW, canvasH) => {
    const aspectRatio = getCurrentCropAspectRatio(cropPreset, canvasW, canvasH);
    if (!aspectRatio) {
        return clampRectToCanvas({ x: anchor.x, y: anchor.y, width: point.x - anchor.x, height: point.y - anchor.y }, canvasW, canvasH, 10);
    }
    const dx = point.x - anchor.x;
    const dy = point.y - anchor.y;
    let width = Math.abs(dx);
    let height = Math.abs(dy);
    if (width === 0 && height === 0) {
        width = 1;
        height = 1 / aspectRatio;
    } else if (height === 0) {
        height = width / aspectRatio;
    } else if (width / height > aspectRatio) {
        height = width / aspectRatio;
    } else {
        width = height * aspectRatio;
    }
    let x = anchor.x;
    let y = anchor.y;
    if (dx < 0) x -= width;
    if (dy < 0) y -= height;
    return clampRectToCanvas({ x, y, width, height }, canvasW, canvasH, 10);
};

export const resizeCropRect = (startRect, handle, pos, cropPreset, canvasW, canvasH) => {
    const aspectRatio = getCurrentCropAspectRatio(cropPreset, canvasW, canvasH);
    const minSize = 10;
    if (!aspectRatio) {
        let nextRect = { ...startRect };
        if (handle.includes('n')) {
            nextRect.y = Math.min(pos.y, startRect.y + startRect.height - minSize);
            nextRect.height = startRect.y + startRect.height - nextRect.y;
        }
        if (handle.includes('s')) nextRect.height = Math.max(pos.y - startRect.y, minSize);
        if (handle.includes('w')) {
            nextRect.x = Math.min(pos.x, startRect.x + startRect.width - minSize);
            nextRect.width = startRect.x + startRect.width - nextRect.x;
        }
        if (handle.includes('e')) nextRect.width = Math.max(pos.x - startRect.x, minSize);
        return clampRectToCanvas(nextRect, canvasW, canvasH, minSize);
    }
    if (['nw', 'ne', 'sw', 'se'].includes(handle)) {
        const anchors = {
            nw: { x: startRect.x + startRect.width, y: startRect.y + startRect.height },
            ne: { x: startRect.x, y: startRect.y + startRect.height },
            sw: { x: startRect.x + startRect.width, y: startRect.y },
            se: { x: startRect.x, y: startRect.y }
        };
        return buildAspectRectFromAnchor(anchors[handle], pos, cropPreset, canvasW, canvasH);
    }
    const centerX = startRect.x + startRect.width / 2;
    const centerY = startRect.y + startRect.height / 2;
    if (handle === 'e' || handle === 'w') {
        const fixedX = handle === 'e' ? startRect.x : startRect.x + startRect.width;
        const rawWidth = handle === 'e' ? pos.x - fixedX : fixedX - pos.x;
        const width = Math.max(rawWidth, minSize);
        const height = width / aspectRatio;
        const x = handle === 'e' ? fixedX : fixedX - width;
        return clampRectToCanvas({ x, y: centerY - height / 2, width, height }, canvasW, canvasH, minSize);
    }
    const fixedY = handle === 's' ? startRect.y : startRect.y + startRect.height;
    const rawHeight = handle === 's' ? pos.y - fixedY : fixedY - pos.y;
    const height = Math.max(rawHeight, minSize);
    const width = height * aspectRatio;
    const y = handle === 's' ? fixedY : fixedY - height;
    return clampRectToCanvas({ x: centerX - width / 2, y, width, height }, canvasW, canvasH, minSize);
};

export const getCropSnapTolerance = (scale) => Math.max(6, 10 / Math.max(scale, 0.25));

export const hasCropSnapEdge = (edges) => !!(edges && (edges.left || edges.right || edges.top || edges.bottom));

export const getCropSnapLabels = (edges) => {
    if (!hasCropSnapEdge(edges)) return [];
    return [
        edges.left ? '左边' : null,
        edges.right ? '右边' : null,
        edges.top ? '上边' : null,
        edges.bottom ? '下边' : null
    ].filter(Boolean);
};

export const getCropSnapEdges = (rect, canvasW, canvasH, tolerance) => {
    if (!rect) return null;
    const normalized = normalizeRect(rect);
    return {
        left: Math.abs(normalized.x) <= tolerance,
        right: Math.abs((normalized.x + normalized.width) - canvasW) <= tolerance,
        top: Math.abs(normalized.y) <= tolerance,
        bottom: Math.abs((normalized.y + normalized.height) - canvasH) <= tolerance
    };
};

export const applyCropSnap = (rect, action, edges, canvasW, canvasH, minSize = 10) => {
    if (!rect || !hasCropSnapEdge(edges)) return rect;
    const normalized = normalizeRect(rect);
    const right = normalized.x + normalized.width;
    const bottom = normalized.y + normalized.height;
    let { x, y, width, height } = normalized;
    const isMove = action && action.type === 'move';

    if (edges.left && edges.right) {
        x = 0;
        width = canvasW;
    } else {
        if (edges.left) {
            if (isMove) x = 0;
            else {
                x = 0;
                width = right;
            }
        }
        if (edges.right) {
            if (isMove) x = canvasW - width;
            else width = canvasW - x;
        }
    }

    if (edges.top && edges.bottom) {
        y = 0;
        height = canvasH;
    } else {
        if (edges.top) {
            if (isMove) y = 0;
            else {
                y = 0;
                height = bottom;
            }
        }
        if (edges.bottom) {
            if (isMove) y = canvasH - height;
            else height = canvasH - y;
        }
    }

    return clampRectToCanvas({ x, y, width, height }, canvasW, canvasH, minSize);
};

export const fitCropRectToAspect = (rect, cropPreset, canvasW, canvasH) => {
    const normalized = normalizeRect(rect);
    const aspectRatio = getCurrentCropAspectRatio(cropPreset, canvasW, canvasH);
    if (!aspectRatio) return clampRectToCanvas(normalized, canvasW, canvasH, 10);
    const centerX = normalized.x + normalized.width / 2;
    const centerY = normalized.y + normalized.height / 2;
    let width = normalized.width;
    let height = width / aspectRatio;
    if (height < normalized.height) {
        height = normalized.height;
        width = height * aspectRatio;
    }
    return clampRectToCanvas({
        x: centerX - width / 2,
        y: centerY - height / 2,
        width,
        height
    }, canvasW, canvasH, 10);
};

export const scaleCropRect = (rect, scaleFactor, pivot, canvasW, canvasH, minSize = 10) => {
    const normalized = normalizeRect(rect);
    const width = Math.max(minSize, normalized.width * scaleFactor);
    const height = Math.max(minSize, normalized.height * scaleFactor);
    const pivotX = pivot ? pivot.x : normalized.x + normalized.width / 2;
    const pivotY = pivot ? pivot.y : normalized.y + normalized.height / 2;
    const ratioX = normalized.width ? (pivotX - normalized.x) / normalized.width : 0.5;
    const ratioY = normalized.height ? (pivotY - normalized.y) / normalized.height : 0.5;
    return clampRectToCanvas({
        x: pivotX - width * ratioX,
        y: pivotY - height * ratioY,
        width,
        height
    }, canvasW, canvasH, minSize);
};
