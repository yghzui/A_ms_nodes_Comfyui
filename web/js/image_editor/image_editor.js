import { api } from "../../../scripts/api.js";
import { showTopNotification } from "../utils/shared_utils.js";

const MAX_DIM = 1536;

export function showImageEditor(imagePaths, currentIndex, node, onSaveCallback) {
    if (!imagePaths || imagePaths.length === 0) return;

    let index = typeof currentIndex === 'number' ? currentIndex : 0;
    if (index < 0) index = 0;
    if (index >= imagePaths.length) index = imagePaths.length - 1;

    const existing = document.getElementById("my-nodes-image-editor");
    if (existing) existing.remove();

    let savedConfig = {};
    try {
        const stored = localStorage.getItem("myNodesImageEditorConfig");
        if (stored) savedConfig = JSON.parse(stored);
    } catch(e) { console.warn("Failed to load editor config", e); }

    let state = {
        tool: 'brush', // Default to brush
        layer: 'mask', // Default to mask
        shape: savedConfig.shape || 'round', // round, square
        size: savedConfig.size || 60,
        opacity: savedConfig.opacity !== undefined ? savedConfig.opacity : 1.0,
        maskPreviewOpacity: savedConfig.maskPreviewOpacity !== undefined ? savedConfig.maskPreviewOpacity : 0.65, // 遮罩层整体预览透明度
        color: savedConfig.color || '#ff0000', // 画笔默认红色
        maskColor: 'rgba(0, 0, 0, 1)', // 遮罩固定使用黑色绘制
        scale: 1,
        panX: 0,
        panY: 0,
        cropPreset: savedConfig.cropPreset || 'free',
        cropRect: null,
        cropHover: null,
        baseImagePath: '',
        baseImageDirty: false,
        history: [],
        historyIndex: -1
    };

    const saveConfigToStorage = () => {
        try {
            localStorage.setItem("myNodesImageEditorConfig", JSON.stringify({
                shape: state.shape,
                size: state.size,
                opacity: state.opacity,
                maskPreviewOpacity: state.maskPreviewOpacity,
                color: state.color,
                cropPreset: state.cropPreset
            }));
        } catch(e) { console.warn("Failed to save editor config", e); }
    };

    let origW = 0, origH = 0, canvasW = 0, canvasH = 0;
    let currentLoadId = 0;
    let isPanning = false, isDrawing = false, cropScaledByWheel = false, cropWheelSaveTimer = null;
    let cropAction = null;
    let suppressBaseImageOnload = false;
    let panStart = { x: 0, y: 0 };
    let lastPos = { x: 0, y: 0 };

    // --- UI Creation ---
    const editor = document.createElement("div");
    editor.id = "my-nodes-image-editor";
    Object.assign(editor.style, {
        position: "fixed", top: "0", left: "0", width: "100%", height: "100%",
        backgroundColor: "#1e1e1e", display: "flex", flexDirection: "column",
        zIndex: "10001", userSelect: "none", fontFamily: "sans-serif", color: "#eee"
    });

    // Cursor
    const cursor = document.createElement('div');
    Object.assign(cursor.style, {
        position: 'fixed', pointerEvents: 'none', zIndex: '10002',
        border: '1.5px dashed black',
        transform: 'translate(-50%, -50%)', display: 'none',
        boxSizing: 'border-box'
    });
    const cursorInner = document.createElement('div');
    Object.assign(cursorInner.style, {
        position: 'absolute', top: '-1.5px', left: '-1.5px', right: '-1.5px', bottom: '-1.5px',
        border: '1.5px solid white',
        borderRadius: 'inherit',
        boxSizing: 'border-box',
        zIndex: '-1'
    });
    cursor.appendChild(cursorInner);
    document.body.appendChild(cursor);

    // Top Bar
    const topBar = document.createElement("div");
    Object.assign(topBar.style, {
        height: "50px", background: "#2a2a2a", display: "flex", alignItems: "center",
        padding: "0 20px", gap: "10px", borderBottom: "1px solid #444"
    });
    
    const title = document.createElement("div");
    title.textContent = "图片编辑器";
    title.style.fontWeight = "bold";
    title.style.marginRight = "20px";

    const createBtn = (text, bg = "#333", color = "#fff") => {
        const btn = document.createElement("button");
        btn.textContent = text;
        Object.assign(btn.style, {
            padding: "6px 12px", background: bg, color: color, border: "none",
            borderRadius: "4px", cursor: "pointer", fontSize: "14px",
            display: "flex", alignItems: "center", gap: "5px"
        });
        btn.onmouseenter = () => btn.style.opacity = "0.8";
        btn.onmouseleave = () => btn.style.opacity = "1";
        return btn;
    };

    const undoBtn = createBtn("↺ 撤销");
    const redoBtn = createBtn("↻ 重做");
    const invertBtn = createBtn("◐ 反转遮罩");
    const resetViewBtn = createBtn("□ 缩放归位");
    const clearBtn = createBtn("∅ 一键清除");
    const saveBtn = createBtn("✓ 保存", "#4CAF50");
    const cancelBtn = createBtn("⨯ 取消", "#f44336");

    const spacer = document.createElement("div");
    spacer.style.flex = "1";

    const closeBtn = createBtn("✖");
    closeBtn.style.background = "transparent";
    closeBtn.style.padding = "4px 8px";
    closeBtn.style.fontSize = "18px";
    closeBtn.style.marginLeft = "10px";
    closeBtn.onmouseenter = () => closeBtn.style.color = "#f44336";
    closeBtn.onmouseleave = () => closeBtn.style.color = "#fff";

    topBar.append(title, undoBtn, redoBtn, invertBtn, resetViewBtn, clearBtn, saveBtn, cancelBtn, spacer, closeBtn);
    editor.appendChild(topBar);

    // Main Area
    const mainArea = document.createElement("div");
    Object.assign(mainArea.style, { display: "flex", flex: "1", overflow: "hidden" });
    editor.appendChild(mainArea);

    // Left Sidebar
    const leftBar = document.createElement("div");
    Object.assign(leftBar.style, {
        width: "60px", background: "#2a2a2a", borderRight: "1px solid #444",
        display: "flex", flexDirection: "column", alignItems: "center", paddingTop: "15px", gap: "15px",
        position: "relative"
    });
    mainArea.appendChild(leftBar);

    const infoLabel = document.createElement("div");
    Object.assign(infoLabel.style, {
        position: "absolute", bottom: "10px", width: "100%",
        textAlign: "center", fontSize: "11px", color: "#aaa",
        pointerEvents: "none", lineHeight: "1.4"
    });
    leftBar.appendChild(infoLabel);

    const tools = [
        { id: 'pan', icon: '🖐️', title: '平移/缩放' },
        { id: 'crop', icon: '✂️', title: '裁剪 (支持自由/常见比例/原比例)' },
        { id: 'brushMask', icon: '⚫', title: '遮罩画笔 (绘制遮罩层)' },
        { id: 'brushPaint', icon: '🖌️', title: '绘画画笔 (绘制绘画层)' },
        { id: 'eraser', icon: '🧽', title: '橡皮擦 (擦除当前层)' },
        { id: 'eyedropper', icon: '💧', title: '吸管 (取色)' }
    ];

    const toolBtns = {};
    tools.forEach(t => {
        const btn = document.createElement("button");
        btn.textContent = t.icon;
        btn.title = t.title;
        Object.assign(btn.style, {
            width: "40px", height: "40px", fontSize: "20px", background: "transparent",
            border: "1px solid transparent", borderRadius: "8px", cursor: "pointer", color: "#fff",
            transition: "all 0.2s"
        });
        btn.onclick = () => setTool(t.id);
        toolBtns[t.id] = btn;
        leftBar.appendChild(btn);
    });

    // Work Area
    const workArea = document.createElement("div");
    Object.assign(workArea.style, { flex: "1", position: "relative", overflow: "hidden", background: "#111" });
    mainArea.appendChild(workArea);

    const canvasWrapper = document.createElement("div");
    Object.assign(canvasWrapper.style, { position: "absolute", transformOrigin: "0 0" });
    workArea.appendChild(canvasWrapper);

    const baseImg = document.createElement("img");
    Object.assign(baseImg.style, { position: "absolute", top: "0", left: "0", width: "100%", height: "100%", pointerEvents: "none", userSelect: "none" });
    
    const paintCanvas = document.createElement("canvas");
    const maskCanvas = document.createElement("canvas");
    const tempCanvas = document.createElement("canvas");
    [paintCanvas, maskCanvas, tempCanvas].forEach(c => {
        Object.assign(c.style, { position: "absolute", top: "0", left: "0", width: "100%", height: "100%", pointerEvents: "none" });
    });

    canvasWrapper.append(baseImg, paintCanvas, maskCanvas, tempCanvas);

    const cropOverlay = document.createElement("div");
    const cropBox = document.createElement("div");
    const cropSnapGuides = {};
    const cropHandles = {};
    Object.assign(cropOverlay.style, { position: "absolute", inset: "0", pointerEvents: "none", overflow: "hidden", display: "none" });
    Object.assign(cropBox.style, {
        position: "absolute", boxSizing: "border-box", border: "2px solid #00d2ff",
        boxShadow: "0 0 0 99999px rgba(0, 0, 0, 0.45)",
        backgroundImage: "linear-gradient(to right, transparent 33.1%, rgba(255,255,255,0.5) 33.1%, rgba(255,255,255,0.5) 33.9%, transparent 33.9%, transparent 66.1%, rgba(255,255,255,0.5) 66.1%, rgba(255,255,255,0.5) 66.9%, transparent 66.9%), linear-gradient(to bottom, transparent 33.1%, rgba(255,255,255,0.5) 33.1%, rgba(255,255,255,0.5) 33.9%, transparent 33.9%, transparent 66.1%, rgba(255,255,255,0.5) 66.1%, rgba(255,255,255,0.5) 66.9%, transparent 66.9%)"
    });
    ['left', 'right', 'top', 'bottom'].forEach((edge) => {
        const guide = document.createElement("div");
        Object.assign(guide.style, {
            position: "absolute",
            display: "none",
            background: "#ffb300",
            boxShadow: "0 0 0 1px rgba(255, 179, 0, 0.35)"
        });
        cropSnapGuides[edge] = guide;
        cropOverlay.appendChild(guide);
    });
    [['nw', '0%', '0%'], ['ne', '100%', '0%'], ['sw', '0%', '100%'], ['se', '100%', '100%']].forEach(([id, left, top]) => {
        const handle = document.createElement("div");
        Object.assign(handle.style, {
            position: "absolute", left, top, width: "12px", height: "12px", transform: "translate(-50%, -50%)",
            boxSizing: "border-box", background: "#00d2ff", border: "1px solid #fff"
        });
        cropHandles[id] = handle;
        cropBox.appendChild(handle);
    });
    cropOverlay.appendChild(cropBox);
    workArea.appendChild(cropOverlay);

    const pCtx = paintCanvas.getContext("2d", { willReadFrequently: true });
    const mCtx = maskCanvas.getContext("2d", { willReadFrequently: true });
    const tempCtx = tempCanvas.getContext("2d", { willReadFrequently: true });

    // Right Sidebar
    const rightBar = document.createElement("div");
    Object.assign(rightBar.style, {
        width: "260px", background: "#2a2a2a", borderLeft: "1px solid #444",
        padding: "20px", display: "flex", flexDirection: "column", gap: "25px", boxSizing: "border-box",
        overflowY: "auto"
    });
    mainArea.appendChild(rightBar);

    const createSection = (titleText, contentEl) => {
        const sec = document.createElement("div");
        const title = document.createElement("div");
        title.textContent = titleText;
        title.style.marginBottom = "10px";
        title.style.fontSize = "14px";
        title.style.color = "#ccc";
        sec.appendChild(title);
        sec.appendChild(contentEl);
        return sec;
    };

    // Layer Toggle
    const layerPanel = document.createElement("div");
    layerPanel.style.cssText = `margin-top:20px; border-top: 1px solid #444; padding-top:10px; display:flex; flex-direction:column; gap:10px;`;
    
    layerPanel.innerHTML = `
        <div style="font-weight:bold; margin-bottom:5px; color:#ddd; font-size:14px;">图层面板</div>
        
        <div id="A_layer_mask" style="display:flex; align-items:center; justify-content:space-between; padding:8px; background:#333; border-radius:5px; cursor:pointer; border:1px solid transparent;">
            <div style="display:flex; align-items:center; gap:8px;">
                <input type="checkbox" id="A_chk_mask" checked title="显示/隐藏" style="cursor:pointer;" />
                <span style="font-size:14px;">⚫ 遮罩层</span>
            </div>
            <span class="active-badge" style="font-size:12px; background:#4CAF50; padding:2px 6px; border-radius:3px; display:none;">活跃层</span>
        </div>
        
        <div id="A_layer_paint" style="display:flex; align-items:center; justify-content:space-between; padding:8px; background:#333; border-radius:5px; cursor:pointer; border:1px solid transparent;">
            <div style="display:flex; align-items:center; gap:8px;">
                <input type="checkbox" id="A_chk_paint" checked title="显示/隐藏" style="cursor:pointer;" />
                <span style="font-size:14px;">🖌️ 绘画层</span>
            </div>
            <span class="active-badge" style="font-size:12px; background:#4CAF50; padding:2px 6px; border-radius:3px; display:none;">活跃层</span>
        </div>
        
        <div id="A_layer_base" style="display:flex; align-items:center; justify-content:space-between; padding:8px; background:#333; border-radius:5px; border:1px solid transparent;">
            <div style="display:flex; align-items:center; gap:8px;">
                <input type="checkbox" id="A_chk_base" checked title="显示/隐藏" style="cursor:pointer;" />
                <div id="A_base_thumb" style="width:20px; height:20px; border-radius:2px; background-size:cover; background-position:center; border:1px solid #555;"></div>
                <span style="font-size:14px;">基础图像层</span>
            </div>
        </div>
    `;
    
    rightBar.appendChild(layerPanel);

    const cropPanel = document.createElement("div");
    cropPanel.style.display = "flex";
    cropPanel.style.flexDirection = "column";
    cropPanel.style.gap = "10px";

    const cropHint = document.createElement("div");
    cropHint.textContent = "拖拽创建裁剪框，支持框出图片外区域作为扩图区域，框内拖动可移动，拖角点可缩放。";
    cropHint.style.fontSize = "12px";
    cropHint.style.color = "#9aa0a6";
    cropHint.style.lineHeight = "1.5";

    const cropPresetWrap = document.createElement("div");
    Object.assign(cropPresetWrap.style, {
        display: "grid",
        gridTemplateColumns: "repeat(2, minmax(0, 1fr))",
        gap: "8px"
    });

    const cropPresets = [
        { id: 'free', label: '自由' },
        { id: 'original', label: '原比例' },
        { id: '1:1', label: '1:1' },
        { id: '4:3', label: '4:3' },
        { id: '3:4', label: '3:4' },
        { id: '16:9', label: '16:9' },
        { id: '9:16', label: '9:16' }
    ];
    const cropPresetBtns = {};
    cropPresets.forEach((preset) => {
        const btn = createBtn(preset.label, "transparent");
        btn.style.padding = "6px 8px";
        btn.style.justifyContent = "center";
        btn.onclick = () => {
            setCropPreset(preset.id);
            saveConfigToStorage();
        };
        cropPresetBtns[preset.id] = btn;
        cropPresetWrap.appendChild(btn);
    });

    const cropActionWrap = document.createElement("div");
    cropActionWrap.style.display = "flex";
    cropActionWrap.style.gap = "10px";

    const applyCropBtn = createBtn("应用裁剪", "#4CAF50");
    applyCropBtn.style.flex = "1";
    const resetCropBtn = createBtn("重置裁剪", "#555");
    resetCropBtn.style.flex = "1";
    cropActionWrap.append(applyCropBtn, resetCropBtn);
    const fitImageCropBtn = createBtn("与原图重合", "#555");
    fitImageCropBtn.style.width = "100%";

    const cropMeta = document.createElement("div");
    cropMeta.style.fontSize = "12px";
    cropMeta.style.color = "#bdbdbd";
    cropMeta.style.lineHeight = "1.5";
    cropMeta.textContent = "裁剪工具未激活";
    const cropSnapStatus = document.createElement("div");
    cropSnapStatus.style.fontSize = "12px";
    cropSnapStatus.style.color = "#ffca5a";
    cropSnapStatus.style.lineHeight = "1.5";
    cropSnapStatus.textContent = "";

    cropPanel.append(cropHint, cropPresetWrap, cropActionWrap, fitImageCropBtn, cropMeta, cropSnapStatus);
    rightBar.appendChild(createSection("图像裁剪", cropPanel));

    // Shape Toggle
    const shapeDiv = document.createElement("div");
    shapeDiv.style.display = "flex"; shapeDiv.style.gap = "10px";
    const shapeBtns = {
        round: createBtn("⚫ 圆形", "transparent"),
        square: createBtn("⬛ 方形", "transparent")
    };
    shapeBtns.round.style.flex = "1"; shapeBtns.square.style.flex = "1";
    shapeBtns.round.onclick = () => { setShape('round'); saveConfigToStorage(); };
    shapeBtns.square.onclick = () => { setShape('square'); saveConfigToStorage(); };
    shapeDiv.append(shapeBtns.round, shapeBtns.square);
    rightBar.appendChild(createSection("笔刷形状", shapeDiv));

    // Color
    const colorInput = document.createElement("input");
    colorInput.type = "color"; colorInput.value = state.color;
    colorInput.style.width = "100%"; colorInput.style.height = "40px";
    colorInput.style.cursor = "pointer"; colorInput.style.border = "none";
    colorInput.style.padding = "0"; colorInput.style.background = "transparent";
    colorInput.oninput = (e) => { state.color = e.target.value; saveConfigToStorage(); };
    rightBar.appendChild(createSection("色彩获取 (仅绘制层有效)", colorInput));

    // Size
    const sizeDiv = document.createElement("div");
    const sizeVal = document.createElement("span"); sizeVal.textContent = state.size;
    const sizeSlider = document.createElement("input");
    sizeSlider.type = "range"; sizeSlider.min = "1"; sizeSlider.max = "500"; sizeSlider.value = state.size;
    sizeSlider.style.width = "100%";
    sizeSlider.oninput = (e) => { state.size = parseInt(e.target.value); sizeVal.textContent = state.size; updateCursorPos(); saveConfigToStorage(); };
    sizeDiv.append(sizeSlider);
    const sizeSec = createSection("笔刷大小: ", sizeDiv);
    sizeSec.children[0].appendChild(sizeVal);
    rightBar.appendChild(sizeSec);

    // Opacity
    const opDiv = document.createElement("div");
    const opVal = document.createElement("span"); opVal.textContent = state.opacity.toFixed(2);
    const opSlider = document.createElement("input");
    opSlider.type = "range"; opSlider.min = "0.01"; opSlider.max = "1"; opSlider.step = "0.01"; opSlider.value = state.opacity;
    opSlider.style.width = "100%";
    opSlider.oninput = (e) => { state.opacity = parseFloat(e.target.value); opVal.textContent = state.opacity.toFixed(2); saveConfigToStorage(); };
    opDiv.append(opSlider);
    const opSec = createSection("绘制不透明度: ", opDiv);
    opSec.children[0].appendChild(opVal);
    rightBar.appendChild(opSec);

    // Mask Preview Opacity
    const maskPreviewOpDiv = document.createElement("div");
    const maskPreviewOpVal = document.createElement("span"); maskPreviewOpVal.textContent = state.maskPreviewOpacity.toFixed(2);
    const maskPreviewOpSlider = document.createElement("input");
    maskPreviewOpSlider.type = "range"; maskPreviewOpSlider.min = "0.0"; maskPreviewOpSlider.max = "1"; maskPreviewOpSlider.step = "0.01"; maskPreviewOpSlider.value = state.maskPreviewOpacity;
    maskPreviewOpSlider.style.width = "100%";
    maskPreviewOpSlider.oninput = (e) => { 
        state.maskPreviewOpacity = parseFloat(e.target.value); 
        maskPreviewOpVal.textContent = state.maskPreviewOpacity.toFixed(2);
        // Only update real-time if we are currently looking at the mask layer or if it's visible
        if (state.layer === 'mask') {
            maskCanvas.style.opacity = state.maskPreviewOpacity;
        } else {
            maskCanvas.style.opacity = state.maskPreviewOpacity * 0.5; // Dimmer when not active
        }
        saveConfigToStorage();
    };
    maskPreviewOpDiv.append(maskPreviewOpSlider);
    const maskPreviewOpSec = createSection("遮罩预览透明度: ", maskPreviewOpDiv);
    maskPreviewOpSec.children[0].appendChild(maskPreviewOpVal);
    rightBar.appendChild(maskPreviewOpSec);


    let lastMouseE = null;
    let cropSnapEdges = null;
    const clamp = (value, min, max) => Math.min(Math.max(value, min), max);
    const cloneRect = (rect) => rect ? { x: rect.x, y: rect.y, width: rect.width, height: rect.height } : null;
    const roundRect = (rect) => rect ? {
        x: Math.round(rect.x),
        y: Math.round(rect.y),
        width: Math.round(rect.width),
        height: Math.round(rect.height)
    } : null;
    const getCurrentCropAspectRatio = () => {
        if (state.cropPreset === 'free') return null;
        if (state.cropPreset === 'original') {
            return canvasH ? canvasW / canvasH : null;
        }
        const parts = state.cropPreset.split(':').map(Number);
        if (parts.length === 2 && parts[0] > 0 && parts[1] > 0) {
            return parts[0] / parts[1];
        }
        return null;
    };
    const fitAspectRect = (aspectRatio) => {
        const padding = 0.08;
        const maxWidth = Math.max(1, canvasW * (1 - padding * 2));
        const maxHeight = Math.max(1, canvasH * (1 - padding * 2));
        if (!aspectRatio) {
            return {
                x: canvasW * padding,
                y: canvasH * padding,
                width: maxWidth,
                height: maxHeight
            };
        }

        let width = maxWidth;
        let height = width / aspectRatio;
        if (height > maxHeight) {
            height = maxHeight;
            width = height * aspectRatio;
        }

        return {
            x: (canvasW - width) / 2,
            y: (canvasH - height) / 2,
            width,
            height
        };
    };
    const createDefaultCropRect = () => {
        if (!canvasW || !canvasH) return null;
        return fitAspectRect(getCurrentCropAspectRatio());
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
    const clampRectToCanvas = (rect, minSize = 1) => {
        const normalized = normalizeRect(rect);
        return {
            x: normalized.x,
            y: normalized.y,
            width: Math.max(minSize, normalized.width),
            height: Math.max(minSize, normalized.height)
        };
    };
    const getHandleSize = () => Math.max(10, 12 / Math.max(state.scale, 0.25));
    const getCropHandleAt = (pos) => {
        if (!state.cropRect) return 'new';
        const rect = normalizeRect(state.cropRect);
        const handle = getHandleSize();
        const half = handle / 2;
        const corners = [
            { id: 'nw', x: rect.x, y: rect.y },
            { id: 'ne', x: rect.x + rect.width, y: rect.y },
            { id: 'sw', x: rect.x, y: rect.y + rect.height },
            { id: 'se', x: rect.x + rect.width, y: rect.y + rect.height }
        ];
        for (const corner of corners) {
            if (
                pos.x >= corner.x - half && pos.x <= corner.x + half &&
                pos.y >= corner.y - half && pos.y <= corner.y + half
            ) {
                return corner.id;
            }
        }

        const edgePadding = Math.max(6, 8 / Math.max(state.scale, 0.25));
        const withinX = pos.x >= rect.x && pos.x <= rect.x + rect.width;
        const withinY = pos.y >= rect.y && pos.y <= rect.y + rect.height;
        if (withinX && Math.abs(pos.y - rect.y) <= edgePadding) return 'n';
        if (withinX && Math.abs(pos.y - (rect.y + rect.height)) <= edgePadding) return 's';
        if (withinY && Math.abs(pos.x - rect.x) <= edgePadding) return 'w';
        if (withinY && Math.abs(pos.x - (rect.x + rect.width)) <= edgePadding) return 'e';
        if (withinX && withinY) return 'move';
        return 'new';
    };
    const getCropCursor = (handle) => {
        const cursorMap = {
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
        };
        return cursorMap[handle] || 'crosshair';
    };
    const getCropSnapTolerance = () => Math.max(6, 10 / Math.max(state.scale, 0.25));
    const hasCropSnapEdge = (edges) => !!(edges && (edges.left || edges.right || edges.top || edges.bottom));
    const getCropSnapLabels = (edges) => {
        if (!hasCropSnapEdge(edges)) return [];
        return [
            edges.left ? '左边' : null,
            edges.right ? '右边' : null,
            edges.top ? '上边' : null,
            edges.bottom ? '下边' : null
        ].filter(Boolean);
    };
    const getCropSnapEdges = (rect, tolerance = getCropSnapTolerance()) => {
        if (!rect) return null;
        const normalized = normalizeRect(rect);
        return {
            left: Math.abs(normalized.x) <= tolerance,
            right: Math.abs((normalized.x + normalized.width) - canvasW) <= tolerance,
            top: Math.abs(normalized.y) <= tolerance,
            bottom: Math.abs((normalized.y + normalized.height) - canvasH) <= tolerance
        };
    };
    const applyCropSnap = (rect, action, edges = getCropSnapEdges(rect)) => {
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

        return clampRectToCanvas({ x, y, width, height }, 10);
    };
    const fitCropRectToAspect = (rect, aspectRatio) => {
        const normalized = normalizeRect(rect);
        if (!aspectRatio) return clampRectToCanvas(normalized, 10);
        const centerX = normalized.x + normalized.width / 2;
        const centerY = normalized.y + normalized.height / 2;
        let width = normalized.width;
        let height = width / aspectRatio;
        if (height < normalized.height) {
            height = normalized.height;
            width = height * aspectRatio;
        }
        let nextRect = {
            x: centerX - width / 2,
            y: centerY - height / 2,
            width,
            height
        };
        return clampRectToCanvas(nextRect, 10);
    };
    const scaleCropRect = (rect, scaleFactor, pivot, minSize = 10) => {
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
        }, minSize);
    };
    const buildAspectRectFromAnchor = (anchor, point, aspectRatio) => {
        if (!aspectRatio) {
            return clampRectToCanvas({
                x: anchor.x,
                y: anchor.y,
                width: point.x - anchor.x,
                height: point.y - anchor.y
            }, 10);
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
        return clampRectToCanvas({ x, y, width, height }, 10);
    };
    const resizeCropRect = (startRect, handle, pos) => {
        const aspectRatio = getCurrentCropAspectRatio();
        const minSize = 10;
        if (!aspectRatio) {
            let nextRect = { ...startRect };
            if (handle.includes('n')) {
                nextRect.y = Math.min(pos.y, startRect.y + startRect.height - minSize);
                nextRect.height = startRect.y + startRect.height - nextRect.y;
            }
            if (handle.includes('s')) {
                nextRect.height = Math.max(pos.y - startRect.y, minSize);
            }
            if (handle.includes('w')) {
                nextRect.x = Math.min(pos.x, startRect.x + startRect.width - minSize);
                nextRect.width = startRect.x + startRect.width - nextRect.x;
            }
            if (handle.includes('e')) {
                nextRect.width = Math.max(pos.x - startRect.x, minSize);
            }
            return clampRectToCanvas(nextRect, minSize);
        }

        if (['nw', 'ne', 'sw', 'se'].includes(handle)) {
            const anchors = {
                nw: { x: startRect.x + startRect.width, y: startRect.y + startRect.height },
                ne: { x: startRect.x, y: startRect.y + startRect.height },
                sw: { x: startRect.x + startRect.width, y: startRect.y },
                se: { x: startRect.x, y: startRect.y }
            };
            return buildAspectRectFromAnchor(anchors[handle], pos, aspectRatio);
        }

        const centerX = startRect.x + startRect.width / 2;
        const centerY = startRect.y + startRect.height / 2;
        if (handle === 'e' || handle === 'w') {
            const fixedX = handle === 'e' ? startRect.x : startRect.x + startRect.width;
            const rawWidth = handle === 'e' ? pos.x - fixedX : fixedX - pos.x;
            const width = Math.max(rawWidth, minSize);
            const height = width / aspectRatio;
            const x = handle === 'e' ? fixedX : fixedX - width;
            return clampRectToCanvas({ x, y: centerY - height / 2, width, height }, minSize);
        }
        const fixedY = handle === 's' ? startRect.y : startRect.y + startRect.height;
        const rawHeight = handle === 's' ? pos.y - fixedY : fixedY - pos.y;
        const height = Math.max(rawHeight, minSize);
        const width = height * aspectRatio;
        const y = handle === 's' ? fixedY : fixedY - height;
        return clampRectToCanvas({ x: centerX - width / 2, y, width, height }, minSize);
    };
    const updateCropMeta = () => {
        if (state.tool !== 'crop') {
            cropMeta.textContent = "裁剪工具未激活";
            return;
        }
        if (!state.cropRect) {
            cropMeta.textContent = `当前比例: ${state.cropPreset === 'original' ? '原比例' : state.cropPreset === 'free' ? '自由' : state.cropPreset}`;
            cropSnapStatus.textContent = "";
            return;
        }
        const rect = roundRect(normalizeRect(state.cropRect));
        cropMeta.textContent = `当前比例: ${state.cropPreset === 'original' ? '原比例' : state.cropPreset === 'free' ? '自由' : state.cropPreset} | 区域 ${rect.width} x ${rect.height} @ (${rect.x}, ${rect.y})`;
        const snapLabels = getCropSnapLabels(cropSnapEdges);
        cropSnapStatus.textContent = snapLabels.length ? `贴边提示: ${snapLabels.join(' / ')}` : "";
    };
    const drawCropOverlay = () => {
        tempCtx.clearRect(0, 0, canvasW, canvasH);
        if (state.tool !== 'crop' || !state.cropRect) {
            cropOverlay.style.display = "none";
            cropSnapEdges = null;
            Object.values(cropSnapGuides).forEach((guide) => { guide.style.display = "none"; });
            updateCropMeta();
            return;
        }
        const rect = normalizeRect(state.cropRect);
        cropSnapEdges = getCropSnapEdges(rect);
        cropOverlay.style.display = "block";
        cropBox.style.left = `${state.panX + rect.x * state.scale}px`;
        cropBox.style.top = `${state.panY + rect.y * state.scale}px`;
        cropBox.style.width = `${Math.max(1, rect.width * state.scale)}px`;
        cropBox.style.height = `${Math.max(1, rect.height * state.scale)}px`;
        const imageLeft = state.panX;
        const imageTop = state.panY;
        const imageWidth = canvasW * state.scale;
        const imageHeight = canvasH * state.scale;
        Object.assign(cropSnapGuides.left.style, {
            display: cropSnapEdges.left ? "block" : "none",
            left: `${imageLeft}px`,
            top: `${imageTop}px`,
            width: "2px",
            height: `${Math.max(1, imageHeight)}px`
        });
        Object.assign(cropSnapGuides.right.style, {
            display: cropSnapEdges.right ? "block" : "none",
            left: `${imageLeft + imageWidth - 2}px`,
            top: `${imageTop}px`,
            width: "2px",
            height: `${Math.max(1, imageHeight)}px`
        });
        Object.assign(cropSnapGuides.top.style, {
            display: cropSnapEdges.top ? "block" : "none",
            left: `${imageLeft}px`,
            top: `${imageTop}px`,
            width: `${Math.max(1, imageWidth)}px`,
            height: "2px"
        });
        Object.assign(cropSnapGuides.bottom.style, {
            display: cropSnapEdges.bottom ? "block" : "none",
            left: `${imageLeft}px`,
            top: `${imageTop + imageHeight - 2}px`,
            width: `${Math.max(1, imageWidth)}px`,
            height: "2px"
        });
        updateCropMeta();
    };
    const setCanvasSize = (width, height) => {
        canvasW = Math.max(1, Math.round(width));
        canvasH = Math.max(1, Math.round(height));
        canvasWrapper.style.width = `${canvasW}px`;
        canvasWrapper.style.height = `${canvasH}px`;
        paintCanvas.width = canvasW; paintCanvas.height = canvasH;
        maskCanvas.width = canvasW; maskCanvas.height = canvasH;
        tempCanvas.width = canvasW; tempCanvas.height = canvasH;
    };
    const updateInfoLabel = () => {
        if (!origW || !origH || !canvasW || !canvasH) {
            infoLabel.innerHTML = "";
            return;
        }
        infoLabel.innerHTML = `${Math.round((canvasW / origW) * 100)}%<br/>${canvasW}x${canvasH}`;
    };
    const updateBaseThumbnail = (src = baseImg.src) => {
        const thumbDiv = document.getElementById("A_base_thumb");
        if (thumbDiv) {
            thumbDiv.style.backgroundImage = src ? `url("${src}")` : "";
        }
    };
    const setBaseImagePreview = (src) => {
        suppressBaseImageOnload = true;
        baseImg.src = src;
        updateBaseThumbnail(src);
    };
    const getRectIntersection = (rect, bounds) => {
        const x = Math.max(rect.x, bounds.x);
        const y = Math.max(rect.y, bounds.y);
        const right = Math.min(rect.x + rect.width, bounds.x + bounds.width);
        const bottom = Math.min(rect.y + rect.height, bounds.y + bounds.height);
        if (right <= x || bottom <= y) return null;
        return { x, y, width: right - x, height: bottom - y };
    };
    const rasterizeBaseImage = () => {
        const raster = document.createElement("canvas");
        raster.width = canvasW;
        raster.height = canvasH;
        raster.getContext("2d").drawImage(baseImg, 0, 0, canvasW, canvasH);
        return raster;
    };
    const setCropPreset = (preset) => {
        state.cropPreset = preset;
        Object.entries(cropPresetBtns).forEach(([id, btn]) => {
            btn.style.background = id === preset ? "#555" : "transparent";
            btn.style.border = id === preset ? "1px solid #888" : "1px solid transparent";
        });
        if (canvasW && canvasH) {
            if (state.cropRect) {
                state.cropRect = fitCropRectToAspect(state.cropRect, getCurrentCropAspectRatio());
            } else {
                state.cropRect = createDefaultCropRect();
            }
        }
        drawCropOverlay();
        updateCursorPos(lastMouseE);
    };
    const resetCropSelection = () => {
        if (!canvasW || !canvasH) return;
        state.cropRect = createDefaultCropRect();
        cropAction = null;
        drawCropOverlay();
        updateCursorPos(lastMouseE);
    };
    fitImageCropBtn.onclick = () => {
        if (!canvasW || !canvasH) return;
        state.cropRect = { x: 0, y: 0, width: canvasW, height: canvasH };
        cropAction = null;
        drawCropOverlay();
        updateCursorPos(lastMouseE);
    };
    const cropCurrentView = (rect) => {
        const rounded = roundRect(clampRectToCanvas(rect, 10));
        const sourceBounds = { x: 0, y: 0, width: canvasW, height: canvasH };
        const cropCanvas = (source, maskOutside = false) => {
            const out = document.createElement("canvas");
            out.width = rounded.width;
            out.height = rounded.height;
            const ctx = out.getContext("2d");
            if (maskOutside) {
                ctx.fillStyle = state.maskColor;
                ctx.fillRect(0, 0, rounded.width, rounded.height);
                const overlap = getRectIntersection(rounded, sourceBounds);
                if (overlap) ctx.clearRect(overlap.x - rounded.x, overlap.y - rounded.y, overlap.width, overlap.height);
            }
            ctx.drawImage(source, -rounded.x, -rounded.y);
            return out;
        };

        const baseCropped = cropCanvas(rasterizeBaseImage());
        const paintCropped = cropCanvas(paintCanvas);
        const maskCropped = cropCanvas(maskCanvas, true);

        setCanvasSize(rounded.width, rounded.height);
        pCtx.clearRect(0, 0, canvasW, canvasH);
        mCtx.clearRect(0, 0, canvasW, canvasH);
        tempCtx.clearRect(0, 0, canvasW, canvasH);
        pCtx.drawImage(paintCropped, 0, 0);
        mCtx.drawImage(maskCropped, 0, 0);
        setBaseImagePreview(baseCropped.toDataURL("image/png"));
        state.baseImageDirty = true;
        state.cropRect = null;
        cropAction = null;
        updateInfoLabel();
        resetView();
        drawCropOverlay();
        saveState();
    };
    applyCropBtn.onclick = () => {
        if (!state.cropRect) {
            resetCropSelection();
            return;
        }
        const rounded = roundRect(clampRectToCanvas(state.cropRect, 10));
        if (rounded.width === canvasW && rounded.height === canvasH && rounded.x === 0 && rounded.y === 0) {
            state.cropRect = null;
            drawCropOverlay();
            return;
        }
        cropCurrentView(rounded);
    };
    resetCropBtn.onclick = resetCropSelection;
    const updateCursorPos = (e = lastMouseE) => {
        if (!e) return;
        lastMouseE = e;
        const { x, y } = getPos(e);
        const isWithinImage = (x >= 0 && x <= canvasW && y >= 0 && y <= canvasH);

        if (state.tool === 'crop') {
            cursor.style.display = 'none';
            state.cropHover = getCropHandleAt({ x, y });
            workArea.style.cursor = cropAction ? getCropCursor(cropAction.handle || cropAction.type) : getCropCursor(state.cropHover);
            return;
        }

        if (state.tool === 'pan' || !isWithinImage) {
            cursor.style.display = 'none';
            if (state.tool === 'eyedropper' && isWithinImage) {
                workArea.style.cursor = 'crosshair';
            } else if (state.tool === 'pan') {
                workArea.style.cursor = isPanning ? 'grabbing' : 'grab';
            } else {
                workArea.style.cursor = 'default';
            }
            return;
        }

        if (state.tool === 'eyedropper') {
            cursor.style.display = 'none';
            workArea.style.cursor = 'crosshair';
            return;
        }

        workArea.style.cursor = 'none';
        cursor.style.display = 'block';
        const displaySize = state.size * state.scale;
        
        cursor.style.width = displaySize + 'px';
        cursor.style.height = displaySize + 'px';
        cursor.style.left = e.clientX + 'px';
        cursor.style.top = e.clientY + 'px';
        cursor.style.borderRadius = state.shape === 'round' ? '50%' : '0';
        
        cursorInner.style.borderRadius = state.shape === 'round' ? '50%' : '0';
    };

    // --- State Management ---
    const setTool = (tool) => {
        state.tool = tool;
        Object.values(toolBtns).forEach(b => {
            b.style.background = "transparent";
            b.style.border = "1px solid transparent";
        });
        cropAction = null;
        
        // Handle tool selection
        if (tool === 'brushMask') {
            toolBtns.brushMask.style.background = '#4CAF50';
            toolBtns.brushMask.style.border = '1px solid #66bb6a';
            setLayer('mask'); // Automatically switch to mask layer
            tempCanvas.style.opacity = 0;
        } else if (tool === 'brushPaint') {
            toolBtns.brushPaint.style.background = '#4CAF50';
            toolBtns.brushPaint.style.border = '1px solid #66bb6a';
            setLayer('paint'); // Automatically switch to paint layer
            tempCanvas.style.opacity = 0;
        } else if (tool === 'crop') {
            toolBtns.crop.style.background = '#4CAF50';
            toolBtns.crop.style.border = '1px solid #66bb6a';
            tempCanvas.style.opacity = 0;
            if (!state.cropRect && canvasW && canvasH) {
                state.cropRect = createDefaultCropRect();
            }
        } else if (toolBtns[tool]) {
            toolBtns[tool].style.background = '#4CAF50';
            toolBtns[tool].style.border = '1px solid #66bb6a';
            tempCanvas.style.opacity = 0;
        }
        if (tool !== 'crop') tempCtx.clearRect(0, 0, canvasW, canvasH);
        drawCropOverlay();
        updateCropMeta();
        updateCursorPos(lastMouseE);
    };

    const setLayer = (layer) => {
        state.layer = layer;
        
        // Update layer panel active state
        const maskDiv = document.getElementById("A_layer_mask");
        const paintDiv = document.getElementById("A_layer_paint");
        
        if (maskDiv && paintDiv) {
            maskDiv.style.border = layer === 'mask' ? "1px solid #4CAF50" : "1px solid transparent";
            maskDiv.querySelector('.active-badge').style.display = layer === 'mask' ? "block" : "none";
            
            paintDiv.style.border = layer === 'paint' ? "1px solid #4CAF50" : "1px solid transparent";
            paintDiv.querySelector('.active-badge').style.display = layer === 'paint' ? "block" : "none";
        }
        
        // Visual feedback for layer
        if (layer === 'mask') {
            maskCanvas.style.opacity = state.maskPreviewOpacity;
            paintCanvas.style.opacity = "0.3";
            colorInput.disabled = true;
            colorInput.style.opacity = '0.3';
        } else {
            maskCanvas.style.opacity = state.maskPreviewOpacity * 0.5; // Dimmer when not active
            paintCanvas.style.opacity = "1";
            colorInput.disabled = false;
            colorInput.style.opacity = '1';
        }
    };

    const setShape = (shape) => {
        state.shape = shape;
        Object.values(shapeBtns).forEach(b => {
            b.style.background = "transparent";
            b.style.border = "1px solid transparent";
        });
        shapeBtns[shape].style.background = "#555";
        shapeBtns[shape].style.border = "1px solid #888";
        updateCursorPos();
    };

    // Initialize UI state completely before doing anything else
    setTool('brushMask');
    setShape(state.shape);

    // Update Layer Visibility Toggles
    setTimeout(() => {
        const chkMask = document.getElementById('A_chk_mask');
        const chkPaint = document.getElementById('A_chk_paint');
        const chkBase = document.getElementById('A_chk_base');
        
        if (chkMask) chkMask.onchange = (e) => {
            maskCanvas.style.visibility = e.target.checked ? "visible" : "hidden";
        };
        if (chkPaint) chkPaint.onchange = (e) => {
            paintCanvas.style.visibility = e.target.checked ? "visible" : "hidden";
        };
        if (chkBase) chkBase.onchange = (e) => {
            baseImg.style.visibility = e.target.checked ? "visible" : "hidden";
        };
        
        // Layer Panel Click to set active layer
        const maskDiv = document.getElementById('A_layer_mask');
        const paintDiv = document.getElementById('A_layer_paint');
        if (maskDiv) maskDiv.onclick = (e) => {
            if(e.target !== chkMask) {
                setTool('brushMask');
            }
        };
        if (paintDiv) paintDiv.onclick = (e) => {
            if(e.target !== chkPaint) {
                setTool('brushPaint');
            }
        };
    }, 100);

    const updateTransform = () => {
        canvasWrapper.style.transform = `translate(${state.panX}px, ${state.panY}px) scale(${state.scale})`;
        drawCropOverlay();
        updateCursorPos();
    };

    const resetView = () => {
        if (!canvasW || !canvasH) return;
        const availableW = workArea.clientWidth;
        const availableH = workArea.clientHeight;
        state.scale = Math.min(availableW / canvasW, availableH / canvasH, 1) * 0.9;
        state.panX = (availableW - canvasW * state.scale) / 2;
        state.panY = (availableH - canvasH * state.scale) / 2;
        updateTransform();
    };
    resetViewBtn.onclick = resetView;

    // --- History ---
    const saveState = () => {
        state.history = state.history.slice(0, state.historyIndex + 1);
        state.history.push({
            base: baseImg.src,
            width: canvasW,
            height: canvasH,
            baseDirty: state.baseImageDirty,
            paint: paintCanvas.toDataURL(),
            mask: maskCanvas.toDataURL()
        });
        if (state.history.length > 30) state.history.shift();
        else state.historyIndex++;
        
        undoBtn.style.opacity = state.historyIndex > 0 ? "1" : "0.5";
        redoBtn.style.opacity = state.historyIndex < state.history.length - 1 ? "1" : "0.5";
    };

    const restoreState = (hState) => {
        setCanvasSize(hState.width || canvasW, hState.height || canvasH);
        state.baseImageDirty = !!hState.baseDirty;
        state.cropRect = null;
        cropAction = null;
        setBaseImagePreview(hState.base);

        const pImg = new Image();
        pImg.onload = () => { pCtx.clearRect(0, 0, canvasW, canvasH); pCtx.drawImage(pImg, 0, 0); };
        pImg.src = hState.paint;
        
        const mImg = new Image();
        mImg.onload = () => { mCtx.clearRect(0, 0, canvasW, canvasH); mCtx.drawImage(mImg, 0, 0); };
        mImg.src = hState.mask;
        updateInfoLabel();
        drawCropOverlay();
        updateTransform();
    };

    undoBtn.onclick = () => {
        if (state.historyIndex > 0) {
            state.historyIndex--;
            restoreState(state.history[state.historyIndex]);
            undoBtn.style.opacity = state.historyIndex > 0 ? "1" : "0.5";
            redoBtn.style.opacity = "1";
        }
    };

    redoBtn.onclick = () => {
        if (state.historyIndex < state.history.length - 1) {
            state.historyIndex++;
            restoreState(state.history[state.historyIndex]);
            undoBtn.style.opacity = "1";
            redoBtn.style.opacity = state.historyIndex < state.history.length - 1 ? "1" : "0.5";
        }
    };

    clearBtn.onclick = () => {
        pCtx.clearRect(0, 0, canvasW, canvasH);
        mCtx.clearRect(0, 0, canvasW, canvasH);
        saveState();
    };

    invertBtn.onclick = () => {
        const mData = mCtx.getImageData(0, 0, canvasW, canvasH);
        for (let i = 0; i < mData.data.length; i += 4) {
            // Invert the alpha channel (0 becomes 255, >0 becomes 0)
            // We use 255 for solid mask, 0 for transparent
            mData.data[i+3] = mData.data[i+3] > 0 ? 0 : 255;
            // Set color to black for the newly masked areas
            if (mData.data[i+3] > 0) {
                mData.data[i] = 0;
                mData.data[i+1] = 0;
                mData.data[i+2] = 0;
            }
        }
        mCtx.putImageData(mData, 0, 0);
        saveState();
    };

    // --- Drawing ---
    window.addEventListener('mousemove', updateCursorPos);

    const getPos = (e) => {
        const rect = canvasWrapper.getBoundingClientRect();
        return {
            x: (e.clientX - rect.left) / state.scale,
            y: (e.clientY - rect.top) / state.scale
        };
    };

    const mergeTempCanvas = () => {
        if (state.tool !== 'eraser') {
            // Because we already drew the tempCanvas directly to targetCtx during the real-time preview loop in drawLine,
            // we just need to clear the backup state and temp canvas.
            tempCtx.isFirstPoint = true;
            tempCtx.backupData = null;
            tempCtx.clearRect(0, 0, canvasW, canvasH);
            return;
        }
        
        const isMask = state.layer === 'mask';
        const targetCtx = isMask ? mCtx : pCtx;
        
        targetCtx.globalCompositeOperation = 'destination-out';
        targetCtx.globalAlpha = state.opacity;
        
        targetCtx.drawImage(tempCanvas, 0, 0);
        
        // Reset
        targetCtx.globalCompositeOperation = 'source-over';
        targetCtx.globalAlpha = 1.0;
        
        tempCtx.clearRect(0, 0, canvasW, canvasH);
    };

    const drawLine = (ctx, from, to) => {
        ctx.globalCompositeOperation = 'source-over';
        ctx.globalAlpha = 1.0;
        
        if (state.tool === 'eraser') {
            ctx.fillStyle = 'rgba(255, 255, 255, 1)'; // 橡皮擦在tempCanvas上绘制白色
        } else {
            ctx.fillStyle = state.layer === 'paint' ? state.color : state.maskColor;
        }

        const dx = to.x - from.x;
        const dy = to.y - from.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        
        // 根据画笔大小动态计算步长，保证连续印章没有缝隙
        const step = Math.max(1, state.size / 15);
        const steps = Math.ceil(dist / step);
        
        const halfSize = state.size / 2;

        for (let i = 0; i <= steps; i++) {
            const t = steps === 0 ? 1 : i / steps;
            const cx = from.x + dx * t;
            const cy = from.y + dy * t;

            ctx.beginPath();
            if (state.shape === 'round') {
                ctx.arc(cx, cy, halfSize, 0, Math.PI * 2);
                ctx.fill();
            } else {
                ctx.fillRect(cx - halfSize, cy - halfSize, state.size, state.size);
            }
        }
        
        // 实时渲染：将当前绘制的临时轨迹以目标不透明度实时预览到底层，并覆盖原有的 tempCanvas 显示
        if (state.tool !== 'eraser') {
            const isMask = state.layer === 'mask';
            const targetCtx = isMask ? mCtx : pCtx;
            
            // 先清理掉之前已经预览过的历史内容，防止叠加越来越深
            if (!ctx.isFirstPoint && ctx.backupData) {
                targetCtx.putImageData(ctx.backupData, 0, 0);
            }
            ctx.isFirstPoint = false;
            
            // 把现在的 tempCanvas 合并到底层作为预览
            targetCtx.globalCompositeOperation = 'source-over';
            targetCtx.globalAlpha = state.opacity;
            targetCtx.drawImage(tempCanvas, 0, 0);
            targetCtx.globalAlpha = 1.0;
        }
    };

    workArea.addEventListener("mousedown", (e) => {
        if (e.button === 1 || (e.button === 0 && (state.tool === 'pan' || e.shiftKey || e.code === "Space"))) {
            isPanning = true;
            panStart.x = e.clientX - state.panX;
            panStart.y = e.clientY - state.panY;
            workArea.style.cursor = "grabbing";
        } else if (e.button === 0) {
            const pos = getPos(e);

            if (state.tool === 'crop') {
                const handle = getCropHandleAt(pos);
                if (handle === 'move' && state.cropRect) {
                    cropAction = {
                        type: 'move',
                        handle,
                        startPos: pos,
                        startRect: cloneRect(state.cropRect)
                    };
                } else if (handle !== 'new' && state.cropRect) {
                    cropAction = {
                        type: 'resize',
                        handle,
                        startPos: pos,
                        startRect: cloneRect(state.cropRect)
                    };
                } else {
                    cropAction = {
                        type: 'new',
                        handle: 'new',
                        startPos: pos
                    };
                    state.cropRect = { x: pos.x, y: pos.y, width: 1, height: 1 };
                }
                drawCropOverlay();
                updateCursorPos(e);
                return;
            }
            
            if (state.tool === 'eyedropper') {
                const x = Math.floor(pos.x);
                const y = Math.floor(pos.y);
                
                if (x < 0 || y < 0 || x >= canvasW || y >= canvasH) return;
                
                const tmp = document.createElement('canvas');
                tmp.width = canvasW; tmp.height = canvasH;
                const ctx = tmp.getContext('2d', { willReadFrequently: true });
                ctx.drawImage(baseImg, 0, 0, canvasW, canvasH);
                const bgData = ctx.getImageData(0, 0, canvasW, canvasH);
                
                const targetIdx = (y * canvasW + x) * 4;
                const targetR = bgData.data[targetIdx];
                const targetG = bgData.data[targetIdx+1];
                const targetB = bgData.data[targetIdx+2];
                const targetA = bgData.data[targetIdx+3];
                
                if (targetA === 0) return; // Ignore transparent background clicks

                // If in mask layer, perform "Magic Wand" style color selection and fill
                if (state.layer === 'mask') {
                    const tolerance = 30; // Color distance tolerance
                    const mData = mCtx.getImageData(0, 0, canvasW, canvasH);
                    
                    for (let i = 0; i < bgData.data.length; i += 4) {
                        const r = bgData.data[i];
                        const g = bgData.data[i+1];
                        const b = bgData.data[i+2];
                        const a = bgData.data[i+3];
                        
                        if (a > 0) {
                            const dist = Math.abs(r - targetR) + Math.abs(g - targetG) + Math.abs(b - targetB);
                            if (dist <= tolerance) {
                                mData.data[i] = 0;   // R
                                mData.data[i+1] = 0; // G
                                mData.data[i+2] = 0; // B
                                mData.data[i+3] = 255; // A (Fully opaque mask)
                            }
                        }
                    }
                    mCtx.putImageData(mData, 0, 0);
                    saveState();
                    return;
                }

                // If in paint layer, just pick the color
                const hex = "#" + (1 << 24 | targetR << 16 | targetG << 8 | targetB).toString(16).padStart(6, '0').slice(1);
                colorInput.value = hex;
                state.color = hex;
                return;
            }

            isDrawing = true;
            lastPos = pos;
            
            // Draw a single point (ensure square shape draws correctly on single click)
            tempCtx.clearRect(0, 0, canvasW, canvasH);
            
            // Set up real-time preview state for new stroke
            tempCtx.isFirstPoint = true;
            const isMask = state.layer === 'mask';
            tempCtx.backupData = (isMask ? mCtx : pCtx).getImageData(0, 0, canvasW, canvasH);
            
            if (state.tool === 'eraser') {
                tempCanvas.style.opacity = 0.5;
            } else {
                tempCanvas.style.opacity = 0; // Hide the tempCanvas overlay itself, we do it in targetCtx now
            }
            drawLine(tempCtx, pos, pos);
        }
    });

    window.addEventListener("mousemove", (e) => {
        if (isPanning) {
            state.panX = e.clientX - panStart.x;
            state.panY = e.clientY - panStart.y;
            updateTransform();
        } else if (cropAction && state.tool === 'crop') {
            const pos = getPos(e);
            if (cropAction.type === 'move' && state.cropRect) {
                const dx = pos.x - cropAction.startPos.x;
                const dy = pos.y - cropAction.startPos.y;
                const startRect = cropAction.startRect;
                state.cropRect = {
                    x: startRect.x + dx,
                    y: startRect.y + dy,
                    width: startRect.width,
                    height: startRect.height
                };
            } else if (cropAction.type === 'resize' && state.cropRect) {
                state.cropRect = resizeCropRect(cropAction.startRect, cropAction.handle, pos);
            } else if (cropAction.type === 'new') {
                state.cropRect = buildAspectRectFromAnchor(cropAction.startPos, pos, getCurrentCropAspectRatio());
            }
            drawCropOverlay();
            updateCursorPos(e);
        } else if (isDrawing) {
            const pos = getPos(e);
            drawLine(tempCtx, lastPos, pos);
            lastPos = pos;
        }
    });

    window.addEventListener("mouseup", (e) => {
        if (isPanning) {
            isPanning = false;
            workArea.style.cursor = state.tool === 'pan' ? "grab" : "none";
        }
        if (cropAction && state.tool === 'crop') {
            state.cropRect = applyCropSnap(state.cropRect, cropAction, cropSnapEdges);
            cropAction = null;
            drawCropOverlay();
            updateCursorPos(e);
        }
        if (isDrawing) {
            isDrawing = false;
            mergeTempCanvas();
            saveState(); // push to history
        }
    });

    workArea.addEventListener("wheel", (e) => {
        e.preventDefault();
        if (state.tool === 'crop' && state.cropRect && e.altKey) {
            const wheelStep = Math.min(Math.abs(e.deltaY) / 600, 0.25);
            const scaleFactor = e.deltaY < 0 ? 1 + wheelStep : Math.max(0.1, 1 - wheelStep);
            state.cropRect = scaleCropRect(state.cropRect, scaleFactor, getPos(e), 10);
            cropScaledByWheel = true;
            if (cropWheelSaveTimer) clearTimeout(cropWheelSaveTimer);
            cropWheelSaveTimer = setTimeout(() => {
                if (cropScaledByWheel) {
                    cropScaledByWheel = false;
                    saveState();
                }
                cropWheelSaveTimer = null;
            }, 180);
            drawCropOverlay();
            updateCursorPos(e);
            return;
        }
        const zoomSpeed = 0.1;
        const oldScale = state.scale;
        state.scale *= (1 - Math.sign(e.deltaY) * zoomSpeed);
        state.scale = Math.max(0.05, Math.min(state.scale, 20));
        const scaleRatio = state.scale / oldScale;
        
        const rect = workArea.getBoundingClientRect();
        const mouseX = e.clientX - rect.left;
        const mouseY = e.clientY - rect.top;

        state.panX = mouseX - (mouseX - state.panX) * scaleRatio;
        state.panY = mouseY - (mouseY - state.panY) * scaleRatio;
        
        updateTransform();
    });

    // --- Loading & Saving ---
    const loadMedia = (idx) => {
        index = idx;
        currentLoadId++;
        const myLoadId = currentLoadId;
        const currentPath = imagePaths[index];
        if (!currentPath) return;

        if (node) {
            if (!node.properties) node.properties = {};
            if (!node.properties.original_image_paths) node.properties.original_image_paths = [];
            for (let i = 0; i < imagePaths.length; i++) {
                if (imagePaths[i] && !imagePaths[i].includes("clipspace-painted-masked-")) {
                    node.properties.original_image_paths[i] = imagePaths[i];
                }
            }
        }
        const origPath = (node && node.properties && node.properties.original_image_paths && node.properties.original_image_paths[index]) 
                            ? node.properties.original_image_paths[index] 
                            : currentPath;

        let filename = origPath, type = 'input', subfolder = '';
        const typeMatch = origPath.match(/^(.*)\s+\[(input|output|temp)\]$/);
        if (typeMatch) { filename = typeMatch[1]; type = typeMatch[2]; }

        const splitIndex = Math.max(filename.lastIndexOf('/'), filename.lastIndexOf('\\'));
        if (splitIndex !== -1) {
            subfolder = filename.substring(0, splitIndex);
            filename = filename.substring(splitIndex + 1);
        }

        const params = new URLSearchParams({ filename, type });
        if (subfolder) params.set('subfolder', subfolder);
        // 使用专门支持 HEIC 转换的 /a_my_nodes/view_input 路由
        const url = api.apiURL(`/a_my_nodes/view_input?${params.toString()}`);
        
        // Extract timestamp ONLY if the current path is a clipspace file
        let ts = null;
        if (currentPath.includes("clipspace-painted-masked-")) {
            let tsMatch = currentPath.match(/clipspace-painted-masked-(\d+)\.png/);
            if (tsMatch) ts = tsMatch[1];
        }

        baseImg.onload = () => {
            if (suppressBaseImageOnload) {
                suppressBaseImageOnload = false;
                return;
            }
            if (myLoadId !== currentLoadId) return;
            origW = baseImg.naturalWidth;
            origH = baseImg.naturalHeight;
            
            let w = origW, h = origH;
            if (w > MAX_DIM || h > MAX_DIM) {
                const ratio = Math.min(MAX_DIM / w, MAX_DIM / h);
                w = Math.floor(w * ratio);
                h = Math.floor(h * ratio);
            }
            setCanvasSize(w, h);
            state.cropRect = null;
            state.cropHover = null;
            state.baseImagePath = origPath;
            state.baseImageDirty = false;
            cropAction = null;
            pCtx.clearRect(0, 0, canvasW, canvasH);
            mCtx.clearRect(0, 0, canvasW, canvasH);
            tempCtx.clearRect(0, 0, canvasW, canvasH);

            // Initialize UI state completely before doing anything else
            setTool('brushMask');
            setShape(state.shape);
            setCropPreset(state.cropPreset);
            
            // Force the layer UI update explicitly for the first time
            setLayer('mask');

            // Setup base image thumbnail
            updateBaseThumbnail(baseImg.src);
            
            updateInfoLabel();
            
            // Center view
            resetView();

            let layersLoaded = 0;
            let layersToLoad = ts ? 2 : 0;

            const finalizeLoad = () => {
                state.history = [];
                state.historyIndex = -1;
                saveState();
            };

            if (ts) {
                let mImg = new Image();
                mImg.onload = () => {
                    if (myLoadId !== currentLoadId) return;
                    let tmp = document.createElement('canvas');
                    tmp.width = canvasW; tmp.height = canvasH;
                    let tCtx = tmp.getContext('2d');
                    tCtx.drawImage(mImg, 0, 0, canvasW, canvasH);
                    let imgData = tCtx.getImageData(0, 0, canvasW, canvasH);
                    
                    for(let i=0; i<imgData.data.length; i+=4) {
                        let originalAlpha = imgData.data[i+3]; 
                        let drawnAlpha = 255 - originalAlpha; // Invert to get exact mask opacity
                        if (drawnAlpha > 0) {
                            imgData.data[i] = 0;   // R = 0 (Black mask)
                            imgData.data[i+1] = 0; // G = 0
                            imgData.data[i+2] = 0; // B = 0
                            imgData.data[i+3] = drawnAlpha; // A = exact drawn mask opacity
                        } else {
                            imgData.data[i+3] = 0;
                        }
                    }
                    mCtx.putImageData(imgData, 0, 0);
                    layersLoaded++;
                    if (layersLoaded === layersToLoad) finalizeLoad();
                };
                mImg.onerror = () => { 
                    if (myLoadId !== currentLoadId) return;
                    layersLoaded++; if (layersLoaded === layersToLoad) finalizeLoad(); 
                };
                mImg.src = api.apiURL(`/a_my_nodes/view_input?filename=clipspace-mask-${ts}.png&type=input&subfolder=clipspace`);

                let pImg = new Image();
                pImg.onload = () => {
                    if (myLoadId !== currentLoadId) return;
                    let tmp = document.createElement('canvas');
                    tmp.width = canvasW; tmp.height = canvasH;
                    let tCtx = tmp.getContext('2d');
                    tCtx.drawImage(pImg, 0, 0, canvasW, canvasH);
                    pCtx.clearRect(0, 0, canvasW, canvasH);
                    pCtx.drawImage(tmp, 0, 0);
                    layersLoaded++;
                    if (layersLoaded === layersToLoad) finalizeLoad();
                };
                pImg.onerror = () => { 
                    if (myLoadId !== currentLoadId) return;
                    layersLoaded++; if (layersLoaded === layersToLoad) finalizeLoad(); 
                };
                pImg.src = api.apiURL(`/a_my_nodes/view_input?filename=clipspace-paint-${ts}.png&type=input&subfolder=clipspace`);
            } else {
                finalizeLoad();
            }
        };
        baseImg.src = url;
    };

    saveBtn.onclick = async () => {
        saveBtn.textContent = "保存中...";
        saveBtn.disabled = true;
        
        try {
            let saveBasePath = state.baseImagePath || ((node && node.properties && node.properties.original_image_paths) ? node.properties.original_image_paths[index] : imagePaths[index]);
            if (state.baseImageDirty) {
                let baseBlob = null;
                if (baseImg.src && baseImg.src.startsWith("data:image/")) {
                    baseBlob = await fetch(baseImg.src).then((resp) => resp.blob());
                } else {
                    const baseCanvas = rasterizeBaseImage();
                    baseBlob = await new Promise((resolve, reject) => {
                        baseCanvas.toBlob((blob) => blob ? resolve(blob) : reject(new Error("无法导出裁剪后的基础图")), "image/png");
                    });
                }
                const formData = new FormData();
                const uploadName = `image-editor-base-${Date.now()}.png`;
                formData.append("image", new File([baseBlob], uploadName, { type: "image/png" }), uploadName);
                formData.append("type", "input");
                formData.append("subfolder", "clipspace");
                const uploadResp = await api.fetchApi("/upload/image", { method: "POST", body: formData });
                if (!uploadResp || (uploadResp.status !== 200 && uploadResp.status !== 201)) {
                    throw new Error(uploadResp ? await uploadResp.text() : "上传裁剪后的基础图失败");
                }
                const uploadData = await uploadResp.json();
                saveBasePath = uploadData.subfolder ? `${uploadData.subfolder}/${uploadData.name} [input]` : `${uploadData.name} [input]`;
            }

            // Extract mask directly as transparent PNG (just like official editor)
            let hasMask = false;
            let maskBase64 = "";
            const chkMask = document.getElementById('A_chk_mask');
            const isMaskVisible = chkMask ? chkMask.checked : true;
            
            if (isMaskVisible) {
                const tmpMaskCnv = document.createElement("canvas");
                tmpMaskCnv.width = canvasW; tmpMaskCnv.height = canvasH;
                const tCtx = tmpMaskCnv.getContext("2d");
                tCtx.drawImage(maskCanvas, 0, 0);
                
                const mData = mCtx.getImageData(0, 0, canvasW, canvasH);
                for (let i = 0; i < mData.data.length; i += 4) {
                    if (mData.data[i+3] > 0) {
                        hasMask = true;
                        break;
                    }
                }
                if (hasMask) maskBase64 = tmpMaskCnv.toDataURL("image/png");
            }
            
            // Extract paint directly as transparent PNG
            let hasPaint = false;
            let paintBase64 = "";
            const chkPaint = document.getElementById('A_chk_paint');
            const isPaintVisible = chkPaint ? chkPaint.checked : true;
            
            if (isPaintVisible) {
                const tmpPaintCnv = document.createElement("canvas");
                tmpPaintCnv.width = canvasW; tmpPaintCnv.height = canvasH;
                const pCtx2 = tmpPaintCnv.getContext("2d");
                pCtx2.drawImage(paintCanvas, 0, 0);
                
                const pData = pCtx2.getImageData(0, 0, canvasW, canvasH);
                for (let i = 0; i < pData.data.length; i += 4) {
                    if (pData.data[i+3] > 0) {
                        hasPaint = true;
                        break;
                    }
                }
                if (hasPaint) paintBase64 = tmpPaintCnv.toDataURL("image/png");
            }
            
            // 只要点击了保存，哪怕只是加载了历史记录没有新增绘制，我们也强制重新上传
            // 因为用户可能在别的节点复用了原图，这里手动保存就应该生成一个独立的新实例
            const resp = await api.fetchApi("/a_my_nodes/upload_custom_edited_image", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    image_path: saveBasePath,
                    mask_data: maskBase64,
                    paint_data: paintBase64
                })
            });

            if (resp.status !== 200) throw new Error(await resp.text());
            const data = await resp.json();
            
            if (data.success && data.filepath) {
                let newPath = data.filepath;
                // 只要产生了编辑后的新路径，就统一补上 [input]
                newPath += " [input]";
                imagePaths[index] = newPath;
                if (node) {
                    if (!node.properties) node.properties = {};
                    if (!node.properties.original_image_paths) node.properties.original_image_paths = [];
                    node.properties.original_image_paths[index] = saveBasePath;
                }
                if (onSaveCallback) onSaveCallback(imagePaths);
                cleanup();
            }
        } catch (err) {
            console.error(err);
            showTopNotification("保存失败: " + err.message, "error");
            saveBtn.textContent = "保存";
            saveBtn.disabled = false;
        }
    };

    const cleanup = () => {
        if (cropWheelSaveTimer) clearTimeout(cropWheelSaveTimer);
        window.removeEventListener('mousemove', updateCursorPos);
        window.removeEventListener("keydown", handleKey);
        cursor.remove();
        editor.remove();
    };

    const handleKey = (e) => {
        if (e.key === "Escape") cleanup();
        if (e.ctrlKey && e.key === 'z') undoBtn.click();
        if (e.ctrlKey && e.key === 'y') redoBtn.click();
    };
    window.addEventListener("keydown", handleKey);
    cancelBtn.onclick = cleanup;
    closeBtn.onclick = cleanup;

    document.body.appendChild(editor);
    loadMedia(index);
}
