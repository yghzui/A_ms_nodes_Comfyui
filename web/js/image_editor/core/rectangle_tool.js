// 本文件由主项目 tools/sync_image_editor_core.py 自动生成，禁止手改。
// 源文件: apps/web_apps/static/image_editor/rectangle_tool.js
const clamp = (value, max) => Math.max(0, Math.min(value, max));

export function rectangleFromPoints(anchor, point, width, height, tolerance, square = false) {
    const snap = (value, max) => value <= tolerance ? 0 : max - value <= tolerance ? max : value;
    const x = snap(clamp(point.x, width), width);
    const y = snap(clamp(point.y, height), height);
    let dx = x - anchor.x;
    let dy = y - anchor.y;
    if (square) {
        const directionX = dx < 0 || (dx === 0 && anchor.x === width) ? -1 : 1;
        const directionY = dy < 0 || (dy === 0 && anchor.y === height) ? -1 : 1;
        const size = Math.min(Math.max(Math.abs(dx), Math.abs(dy)),
            directionX < 0 ? anchor.x : width - anchor.x,
            directionY < 0 ? anchor.y : height - anchor.y);
        dx = directionX * size;
        dy = directionY * size;
    }
    return { x: Math.min(anchor.x, anchor.x + dx), y: Math.min(anchor.y, anchor.y + dy),
        width: Math.abs(dx), height: Math.abs(dy) };
}

export function moveRectangle(rect, dx, dy, width, height, tolerance) {
    let x = clamp(rect.x + dx, width - rect.width);
    let y = clamp(rect.y + dy, height - rect.height);
    if (x <= tolerance) x = 0;
    else if (width - x - rect.width <= tolerance) x = width - rect.width;
    if (y <= tolerance) y = 0;
    else if (height - y - rect.height <= tolerance) y = height - rect.height;
    return { ...rect, x, y };
}

const corners = rect => [
    { x: rect.x, y: rect.y }, { x: rect.x + rect.width, y: rect.y },
    { x: rect.x, y: rect.y + rect.height }, { x: rect.x + rect.width, y: rect.y + rect.height },
];

export function createRectangleTool({ workArea, state, getSize, getPos, getContext, saveState, createBtn, compositeLayer }) {
    const layerOpacity = () => state.layer === 'mask' ? state.maskOpacity : state.paintOpacity;
    let rect = null;
    let maskRectangles = [];
    let rectangles = state.layer === 'mask' ? maskRectangles : [];
    let selectedIndex = -1;
    let action = null;
    let dragStart = null;
    let activeLayer = state.layer;
    let maskBase = null;
    const maskStrokeBase = document.createElement('canvas');
    let color = '#808080';
    const drawingColor = () => state.layer === 'mask' ? (state.maskColor || '#000000') : color;
    let previewSource = null;
    let sourceVisibility = '';
    let maskPreviewFrame = null;
    let maskRaster = null;
    const preview = document.createElement('canvas');
    const releasePreview = () => {
        if (maskPreviewFrame !== null) cancelAnimationFrame(maskPreviewFrame);
        maskPreviewFrame = null;
        if (previewSource) previewSource.style.visibility = sourceVisibility;
        previewSource = null;
        preview.remove();
    };
    const overlay = document.createElement('div');
    overlay.dataset.rectangleOverlay = '';
    Object.assign(overlay.style, { position: 'absolute', inset: '0', pointerEvents: 'none', display: 'none' });
    const box = document.createElement('div');
    Object.assign(box.style, { position: 'absolute', boxSizing: 'border-box', border: '2px solid #ff684b' });
    const otherBoxes = [];
    const makeHandle = (parent, color) => {
        const handle = document.createElement('div');
        Object.assign(handle.style, { position: 'absolute', width: '12px', height: '12px',
            background: color, border: '1px solid white', boxSizing: 'border-box', transform: 'translate(-50%, -50%)' });
        parent.appendChild(handle);
        return handle;
    };
    const imageHandles = Array.from({ length: 4 }, () => makeHandle(overlay, '#00d2ff'));
    const rectHandles = Array.from({ length: 4 }, () => makeHandle(box, '#ff684b'));
    const guides = Array.from({ length: 4 }, () => {
        const guide = document.createElement('div');
        Object.assign(guide.style, { position: 'absolute', background: '#ffb300', display: 'none' });
        overlay.appendChild(guide);
        return guide;
    });
    overlay.appendChild(box);
    workArea.appendChild(overlay);

    const panel = document.createElement('div');
    Object.assign(panel.style, { display: 'flex', flexDirection: 'column', gap: '10px' });
    const hint = document.createElement('div');
    Object.assign(hint.style, { fontSize: '12px', color: '#aaa', lineHeight: '1.5' });
    const status = document.createElement('div');
    Object.assign(status.style, { fontSize: '12px', color: '#ffca5a' });
    const colorLabel = document.createElement('label');
    colorLabel.textContent = '矩形颜色';
    Object.assign(colorLabel.style, { display: 'flex', alignItems: 'center', gap: '8px', fontSize: '12px' });
    const colorInput = document.createElement('input');
    colorInput.type = 'color';
    colorInput.value = color;
    Object.assign(colorInput.style, { flex: '1', minWidth: '0', height: '30px', cursor: 'pointer' });
    colorInput.oninput = event => { color = event.target.value; render(); };
    colorLabel.appendChild(colorInput);
    const buttons = document.createElement('div');
    Object.assign(buttons.style, { display: 'flex', gap: '8px' });
    const applyBtn = createBtn('应用矩形', '#4CAF50');
    const cancelBtn = createBtn('取消矩形', '#555');
    const deleteBtn = createBtn('删除当前', '#555');
    buttons.style.flexWrap = 'wrap';
    buttons.append(applyBtn, cancelBtn, deleteBtn);
    panel.append(hint, colorLabel, status, buttons);

    const isMask = () => state.layer === 'mask';
    const cancelMaskDrag = () => {
        if (activeLayer !== 'mask' || !action) return;
        maskRectangles.splice(0, maskRectangles.length, ...JSON.parse(dragStart));
        action = null;
        dragStart = null;
    };
    const syncLayer = () => {
        if (activeLayer === state.layer) return;
        cancelMaskDrag();
        activeLayer = state.layer;
        rectangles = isMask() ? maskRectangles : [];
        selectedIndex = rectangles.length - 1;
        rect = rectangles[selectedIndex] || null;
        action = null;
    };
    const renderMaskPixels = (ctx, base) => {
        const { width, height } = base;
        if (maskRaster?.width !== width || maskRaster?.height !== height) {
            maskRaster = ctx.createImageData(width, height);
        }
        const image = maskRaster;
        image.data.set(base.data);
        for (const item of maskRectangles) {
            const left = Math.round(item.x);
            const top = Math.round(item.y);
            const right = Math.round(item.x + item.width);
            const bottom = Math.round(item.y + item.height);
            for (let y = top; y < bottom; y++) {
                for (let x = left; x < right; x++) {
                    const pixel = (y * width + x) * 4;
                    const previous = image.data[pixel + 3];
                    if (!item.overlap && previous >= item.alpha) continue;
                    image.data[pixel] = image.data[pixel + 1] = image.data[pixel + 2] = 0;
                    image.data[pixel + 3] = item.overlap
                        ? Math.round(previous + item.alpha * (1 - previous / 255)) : item.alpha;
                }
            }
        }
        ctx.putImageData(image, 0, 0);
    };
    const commitMask = () => {
        const ctx = getContext('mask');
        const { width, height } = getSize();
        maskBase ??= ctx.getImageData(0, 0, width, height);
        renderMaskPixels(ctx, maskBase);
        if (!maskRectangles.length) maskBase = null;
    };
    const scheduleMaskPreview = () => {
        if (maskPreviewFrame !== null) return;
        maskPreviewFrame = requestAnimationFrame(() => {
            maskPreviewFrame = null;
            if (!action || !isMask() || state.tool !== 'rectangle') return;
            const ctx = getContext('mask');
            const source = ctx.canvas;
            if (previewSource !== source) {
                releasePreview();
                preview.width = source.width;
                preview.height = source.height;
                preview.style.cssText = source.style.cssText;
                source.parentElement.insertBefore(preview, source);
                previewSource = source;
                sourceVisibility = source.style.visibility;
                source.style.visibility = 'hidden';
            }
            const base = maskBase ?? (action.previewBase ??= ctx.getImageData(0, 0, source.width, source.height));
            renderMaskPixels(preview.getContext('2d'), base);
        });
    };
    const previewPaint = () => {
        const source = getContext('paint').canvas;
        preview.width = source.width;
        preview.height = source.height;
        preview.style.cssText = source.style.cssText;
        const previewCtx = preview.getContext('2d');
        previewCtx.drawImage(source, 0, 0);
        if (state.allowOverlap) {
            previewCtx.globalAlpha = layerOpacity();
            previewCtx.fillStyle = drawingColor();
            rectangles.forEach(item => {
                const x = Math.round(item.x), y = Math.round(item.y);
                previewCtx.fillRect(x, y, Math.round(item.x + item.width) - x, Math.round(item.y + item.height) - y);
            });
        } else {
            const shapes = document.createElement('canvas');
            shapes.width = source.width;
            shapes.height = source.height;
            const shapesCtx = shapes.getContext('2d');
            shapesCtx.fillStyle = drawingColor();
            rectangles.forEach(item => {
                const x = Math.round(item.x), y = Math.round(item.y);
                shapesCtx.fillRect(x, y, Math.round(item.x + item.width) - x, Math.round(item.y + item.height) - y);
            });
            compositeLayer(previewCtx, shapes);
        }
        source.parentElement.insertBefore(preview, source);
        sourceVisibility = source.style.visibility;
        previewSource = source;
        source.style.visibility = 'hidden';
    };
    const render = () => {
        syncLayer();
        if (selectedIndex >= 0 && rect) rectangles[selectedIndex] = rect;
        const active = state.tool === 'rectangle';
        if (active && isMask() && action) scheduleMaskPreview();
        else releasePreview();
        panel.style.display = active ? 'flex' : 'none';
        colorLabel.style.display = state.layer === 'paint' ? 'flex' : 'none';
        overlay.style.display = active ? 'block' : 'none';
        applyBtn.style.display = cancelBtn.style.display = isMask() ? 'none' : 'flex';
        hint.textContent = isMask()
            ? '松手即保存矩形。拖动框内可移动，拖角可调整，Ctrl 可画重叠框，Shift 可画正方形，Delete 删除当前框。'
            : '连续拖出多个矩形，点击已有框可移动或拉角调整；Ctrl 拖出重叠框。边界自动吸合，Shift 正方形，Enter 全部应用，Esc 全部取消，Delete 删除当前框。';
        applyBtn.disabled = !active || !rectangles.some(item => item.width >= 1 && item.height >= 1);
        cancelBtn.disabled = rectangles.length === 0;
        deleteBtn.disabled = selectedIndex < 0;
        status.textContent = rect ? `共 ${rectangles.length} 个框 · 当前 ${selectedIndex + 1}：${Math.round(rect.width)} × ${Math.round(rect.height)} @ (${Math.round(rect.x)}, ${Math.round(rect.y)})` : '尚未绘制矩形';
        otherBoxes.forEach(item => item.remove());
        otherBoxes.length = 0;
        if (!active) return;
        const { width, height } = getSize();
        const imagePoints = corners({ x: 0, y: 0, width, height });
        imageHandles.forEach((handle, index) => {
            handle.style.left = `${state.panX + imagePoints[index].x * state.scale}px`;
            handle.style.top = `${state.panY + imagePoints[index].y * state.scale}px`;
        });
        box.style.display = rect ? 'block' : 'none';
        guides.forEach(guide => { guide.style.display = 'none'; });
        if (!rect) return;
        if (!isMask()) previewPaint();
        rectangles.forEach((item, index) => {
            if (index === selectedIndex) return;
            const outline = document.createElement('div');
            Object.assign(outline.style, { position: 'absolute', boxSizing: 'border-box', border: '1px solid #ff684b',
                left: `${state.panX + item.x * state.scale}px`, top: `${state.panY + item.y * state.scale}px`,
                width: `${item.width * state.scale}px`, height: `${item.height * state.scale}px` });
            overlay.appendChild(outline);
            otherBoxes.push(outline);
        });
        Object.assign(box.style, { left: `${state.panX + rect.x * state.scale}px`,
            top: `${state.panY + rect.y * state.scale}px`, width: `${rect.width * state.scale}px`, height: `${rect.height * state.scale}px` });
        [[0, 0], [100, 0], [0, 100], [100, 100]].forEach(([x, y], index) => {
            rectHandles[index].style.left = `${x}%`;
            rectHandles[index].style.top = `${y}%`;
        });
        const snapped = [rect.x === 0, rect.x + rect.width === width, rect.y === 0, rect.y + rect.height === height];
        snapped.forEach((hit, index) => {
            if (!hit) return;
            const vertical = index < 2;
            Object.assign(guides[index].style, { display: 'block',
                left: `${state.panX + (index === 1 ? width * state.scale : 0)}px`,
                top: `${state.panY + (index === 3 ? height * state.scale : 0)}px`,
                width: vertical ? '2px' : `${width * state.scale}px`,
                height: vertical ? `${height * state.scale}px` : '2px' });
        });
        if (snapped.some(Boolean)) status.textContent += ' · 已吸合图片边界';
    };
    const cancel = () => {
        syncLayer();
        cancelMaskDrag();
        action = null;
        if (!isMask()) rectangles = [];
        selectedIndex = rectangles.length - 1;
        rect = rectangles[selectedIndex] || null;
        render();
    };
    const flattenMask = () => {
        if (activeLayer === 'mask' && action) cancel();
        if (!maskRectangles.length) return;
        maskRectangles.length = 0;
        maskBase = null;
        if (activeLayer === 'mask') cancel();
    };
    const addMaskStroke = stroke => {
        if (!maskBase) return;
        const { width, height } = getSize();
        maskStrokeBase.width = width;
        maskStrokeBase.height = height;
        const ctx = maskStrokeBase.getContext('2d', { willReadFrequently: true });
        ctx.putImageData(maskBase, 0, 0);
        compositeLayer(ctx, stroke);
        maskBase = ctx.getImageData(0, 0, width, height);
        commitMask();
    };
    const cancelEdit = () => {
        if (!rectangles.length) return;
        cancel();
        saveState();
    };
    const apply = () => {
        if (isMask()) return;
        const valid = rectangles.filter(item => item.width >= 1 && item.height >= 1);
        if (!valid.length) return;
        const ctx = getContext(state.layer);
        if (state.allowOverlap) {
            ctx.save();
            ctx.globalCompositeOperation = 'source-over';
            ctx.globalAlpha = layerOpacity();
            ctx.fillStyle = drawingColor();
            valid.forEach(item => {
                const x = Math.round(item.x), y = Math.round(item.y);
                ctx.fillRect(x, y, Math.round(item.x + item.width) - x, Math.round(item.y + item.height) - y);
            });
            ctx.restore();
        } else {
            const shapes = document.createElement('canvas');
            shapes.width = ctx.canvas.width;
            shapes.height = ctx.canvas.height;
            const shapesCtx = shapes.getContext('2d');
            shapesCtx.fillStyle = drawingColor();
            valid.forEach(item => {
                const x = Math.round(item.x), y = Math.round(item.y);
                shapesCtx.fillRect(x, y, Math.round(item.x + item.width) - x, Math.round(item.y + item.height) - y);
            });
            compositeLayer(ctx, shapes);
        }
        cancel();
        saveState();
    };
    applyBtn.onclick = apply;
    cancelBtn.onclick = cancelEdit;
    const deleteSelected = () => {
        if (selectedIndex < 0) return;
        rectangles.splice(selectedIndex, 1);
        selectedIndex = rectangles.length - 1;
        rect = rectangles[selectedIndex] || null;
        action = null;
        if (isMask()) commitMask();
        render();
        saveState();
    };
    deleteBtn.onclick = deleteSelected;

    const hitCorner = (points, pos) => points.findIndex(point =>
        Math.hypot(point.x - pos.x, point.y - pos.y) <= 6 / state.scale);
    const getCursor = event => {
        if (action?.type === 'move') return 'grabbing';
        const pos = getPos(event);
        if (action?.type === 'resize') {
            const dx = pos.x - action.anchor.x, dy = pos.y - action.anchor.y;
            return dx * dy >= 0 ? 'nwse-resize' : 'nesw-resize';
        }
        if (action?.type === 'new' || event.ctrlKey) return 'crosshair';
        for (let index = rectangles.length - 1; index >= 0; index--) {
            const item = rectangles[index];
            const corner = hitCorner(corners(item), pos);
            if (corner >= 0) return corner === 0 || corner === 3 ? 'nwse-resize' : 'nesw-resize';
            if (pos.x >= item.x && pos.x <= item.x + item.width && pos.y >= item.y && pos.y <= item.y + item.height) return 'grab';
        }
        const { width, height } = getSize();
        if (hitCorner(corners({ x: 0, y: 0, width, height }), pos) >= 0) return 'crosshair';
        return pos.x >= 0 && pos.y >= 0 && pos.x <= width && pos.y <= height ? 'crosshair' : 'default';
    };
    const down = event => {
        if (state.tool !== 'rectangle' || event.button !== 0) return;
        const { width, height } = getSize();
        if (!width || !height) return;
        const pos = getPos(event);
        dragStart = JSON.stringify(rectangles);
        if (!event.ctrlKey) {
            for (let index = rectangles.length - 1; index >= 0; index--) {
                const item = rectangles[index];
                if (hitCorner(corners(item), pos) >= 0 ||
                    (pos.x >= item.x && pos.x <= item.x + item.width && pos.y >= item.y && pos.y <= item.y + item.height)) {
                    selectedIndex = index;
                    rect = item;
                    break;
                }
            }
        }
        const imagePoints = corners({ x: 0, y: 0, width, height });
        const imageCorner = hitCorner(imagePoints, pos);
        const selectionCorner = rect && !event.ctrlKey ? hitCorner(corners(rect), pos) : -1;
        if (selectionCorner >= 0) {
            action = { type: 'resize', anchor: corners(rect)[3 - selectionCorner], rect: { ...rect } };
        } else if (imageCorner >= 0) {
            action = { type: 'new', anchor: imagePoints[imageCorner] };
        } else if (!event.ctrlKey && rect && pos.x >= rect.x && pos.x <= rect.x + rect.width && pos.y >= rect.y && pos.y <= rect.y + rect.height) {
            action = { type: 'move', start: pos, rect: { ...rect } };
        } else {
            if (pos.x < 0 || pos.y < 0 || pos.x > width || pos.y > height) return;
            const tolerance = 3 / state.scale;
            const anchor = { x: pos.x <= tolerance ? 0 : width - pos.x <= tolerance ? width : pos.x,
                y: pos.y <= tolerance ? 0 : height - pos.y <= tolerance ? height : pos.y };
            action = { type: 'new', anchor };
        }
        if (action.type === 'new') {
            rect = { ...action.anchor, width: 0, height: 0 };
            if (isMask()) {
                rect.alpha = Math.round(layerOpacity() * 255);
                rect.overlap = state.allowOverlap;
            }
            rectangles.push(rect);
            selectedIndex = rectangles.length - 1;
        }
        event.preventDefault();
        render();
        workArea.style.cursor = getCursor(event);
    };
    const move = event => {
        if (state.tool !== 'rectangle') return;
        workArea.style.cursor = getCursor(event);
        if (!action) return;
        const pos = getPos(event);
        const { width, height } = getSize();
        const tolerance = 3 / state.scale;
        const geometry = action.type === 'move'
            ? moveRectangle(action.rect, pos.x - action.start.x, pos.y - action.start.y, width, height, tolerance)
            : rectangleFromPoints(action.anchor, pos, width, height, tolerance, event.shiftKey);
        rect = { ...rect, ...geometry };
        render();
    };
    const up = event => {
        if (!action) return;
        move(event);
        const newRectangle = action.type === 'new';
        if (isMask() && action.type === 'resize' && (rect.width < 1 || rect.height < 1)) {
            rect = rectangles[selectedIndex] = action.rect;
        }
        action = null;
        if (isMask() && newRectangle) {
            if (rect.width < 1 || rect.height < 1) {
                rectangles.pop();
                selectedIndex = rectangles.length - 1;
                rect = rectangles[selectedIndex] || null;
            }
        }
        if (JSON.stringify(rectangles) !== dragStart) {
            if (isMask()) commitMask();
            saveState();
        }
        render();
        workArea.style.cursor = getCursor(event);
        dragStart = null;
    };
    const key = event => {
        if (state.tool !== 'rectangle' || !rect || event.target.matches?.('input, textarea, select, [contenteditable="true"]')) return;
        if (!isMask() && (event.key === 'Enter' || event.key === 'Escape')) {
            event.preventDefault();
            if (event.key === 'Enter') apply();
            else cancelEdit();
        } else if (event.key === 'Delete' || event.key === 'Backspace') {
            event.preventDefault();
            deleteSelected();
        }
    };
    workArea.addEventListener('mousedown', down);
    window.addEventListener('mousemove', move);
    window.addEventListener('mouseup', up);
    window.addEventListener('keydown', key);
    return { panel, render, cancel, apply, getCursor, flattenMask, addMaskStroke,
        snapshot() {
            return { rectangles: rectangles.map(item => ({ ...item })), selectedIndex, color,
                layer: state.layer, opacity: layerOpacity(), maskBase,
                maskRectangles: maskRectangles.map(item => ({ ...item })) };
        },
        restore(snapshot) {
            cancel();
            maskBase = snapshot.maskBase;
            maskRectangles = snapshot.maskRectangles.map(item => ({ ...item }));
            rectangles = isMask() ? maskRectangles : (snapshot.layer === 'paint'
                ? snapshot.rectangles.map(item => ({ ...item })) : []);
            selectedIndex = snapshot.layer === state.layer ? snapshot.selectedIndex : rectangles.length - 1;
            rect = rectangles[selectedIndex] || null;
            color = snapshot.color;
            colorInput.value = color;
            render();
        }, setLayerVisibility(layer, visible) {
        const source = getContext(layer).canvas;
        const visibility = visible ? 'visible' : 'hidden';
        if (previewSource === source) {
            sourceVisibility = visibility;
            preview.style.visibility = visibility;
        } else {
            source.style.visibility = visibility;
        }
        render();
    }, dispose() {
        releasePreview();
        workArea.removeEventListener('mousedown', down);
        window.removeEventListener('mousemove', move);
        window.removeEventListener('mouseup', up);
        window.removeEventListener('keydown', key);
        overlay.remove();
    } };
}
