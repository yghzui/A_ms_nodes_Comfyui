import { api } from "../../../scripts/api.js";

const MAX_DIM = 1536;

export function showImageEditor(imagePaths, currentIndex, node, onSaveCallback) {
    if (!imagePaths || imagePaths.length === 0) return;

    let index = typeof currentIndex === 'number' ? currentIndex : 0;
    if (index < 0) index = 0;
    if (index >= imagePaths.length) index = imagePaths.length - 1;

    const existing = document.getElementById("my-nodes-image-editor");
    if (existing) existing.remove();

    let state = {
        tool: 'brush', // Default to brush
        layer: 'mask', // Default to mask
        shape: 'round', // round, square
        size: 60,
        opacity: 1.0,
        color: '#ff0000', // 画笔默认红色
        maskColor: 'rgba(0, 0, 0, 1)', // 遮罩固定使用黑色绘制
        scale: 1,
        panX: 0,
        panY: 0,
        history: [],
        historyIndex: -1
    };

    let origW = 0, origH = 0, canvasW = 0, canvasH = 0;
    let currentLoadId = 0;
    let isPanning = false, isDrawing = false;
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

    const undoBtn = createBtn("↩️ 撤销");
    const redoBtn = createBtn("↪️ 重做");
    const invertBtn = createBtn("◐ 反转遮罩");
    const clearBtn = createBtn("🗑️ 一键清除");
    const saveBtn = createBtn("√ 保存", "#4CAF50");
    const cancelBtn = createBtn("取消", "#f44336");

    const spacer = document.createElement("div");
    spacer.style.flex = "1";

    const closeBtn = createBtn("✖");
    closeBtn.style.background = "transparent";
    closeBtn.style.padding = "4px 8px";
    closeBtn.style.fontSize = "18px";
    closeBtn.style.marginLeft = "10px";
    closeBtn.onmouseenter = () => closeBtn.style.color = "#f44336";
    closeBtn.onmouseleave = () => closeBtn.style.color = "#fff";

    topBar.append(title, undoBtn, redoBtn, invertBtn, clearBtn, saveBtn, cancelBtn, spacer, closeBtn);
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

    // Shape Toggle
    const shapeDiv = document.createElement("div");
    shapeDiv.style.display = "flex"; shapeDiv.style.gap = "10px";
    const shapeBtns = {
        round: createBtn("⚫ 圆形", "transparent"),
        square: createBtn("⬛ 方形", "transparent")
    };
    shapeBtns.round.style.flex = "1"; shapeBtns.square.style.flex = "1";
    shapeBtns.round.onclick = () => setShape('round');
    shapeBtns.square.onclick = () => setShape('square');
    shapeDiv.append(shapeBtns.round, shapeBtns.square);
    rightBar.appendChild(createSection("笔刷形状", shapeDiv));

    // Color
    const colorInput = document.createElement("input");
    colorInput.type = "color"; colorInput.value = "#ff0000";
    colorInput.style.width = "100%"; colorInput.style.height = "40px";
    colorInput.style.cursor = "pointer"; colorInput.style.border = "none";
    colorInput.style.padding = "0"; colorInput.style.background = "transparent";
    colorInput.oninput = (e) => state.color = e.target.value;
    rightBar.appendChild(createSection("色彩获取 (仅绘制层有效)", colorInput));

    // Size
    const sizeDiv = document.createElement("div");
    const sizeVal = document.createElement("span"); sizeVal.textContent = "60";
    const sizeSlider = document.createElement("input");
    sizeSlider.type = "range"; sizeSlider.min = "1"; sizeSlider.max = "500"; sizeSlider.value = "60";
    sizeSlider.style.width = "100%";
    sizeSlider.oninput = (e) => { state.size = parseInt(e.target.value); sizeVal.textContent = state.size; updateCursorPos(); };
    sizeDiv.append(sizeSlider);
    const sizeSec = createSection("笔刷大小: ", sizeDiv);
    sizeSec.children[0].appendChild(sizeVal);
    rightBar.appendChild(sizeSec);

    // Opacity
    const opDiv = document.createElement("div");
    const opVal = document.createElement("span"); opVal.textContent = "1.00";
    const opSlider = document.createElement("input");
    opSlider.type = "range"; opSlider.min = "0.01"; opSlider.max = "1"; opSlider.step = "0.01"; opSlider.value = "1";
    opSlider.style.width = "100%";
    opSlider.oninput = (e) => { state.opacity = parseFloat(e.target.value); opVal.textContent = state.opacity.toFixed(2); };
    opDiv.append(opSlider);
    const opSec = createSection("不透明度: ", opDiv);
    opSec.children[0].appendChild(opVal);
    rightBar.appendChild(opSec);


    let lastMouseE = null;
    const updateCursorPos = (e = lastMouseE) => {
        if (!e) return;
        lastMouseE = e;
        
        const rect = tempCanvas.getBoundingClientRect();
        const x = (e.clientX - rect.left) / state.scale;
        const y = (e.clientY - rect.top) / state.scale;
        
        // Check if mouse is within the actual image bounds
        const isWithinImage = (x >= 0 && x <= canvasW && y >= 0 && y <= canvasH);

        if (state.tool === 'pan' || !isWithinImage) {
            cursor.style.display = 'none';
            // Only show crosshair for eyedropper if within image
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
        
        // Handle tool selection
        if (tool === 'brushMask') {
            toolBtns.brushMask.style.background = '#4CAF50';
            toolBtns.brushMask.style.border = '1px solid #66bb6a';
            setLayer('mask'); // Automatically switch to mask layer
        } else if (tool === 'brushPaint') {
            toolBtns.brushPaint.style.background = '#4CAF50';
            toolBtns.brushPaint.style.border = '1px solid #66bb6a';
            setLayer('paint'); // Automatically switch to paint layer
        } else if (toolBtns[tool]) {
            toolBtns[tool].style.background = '#4CAF50';
            toolBtns[tool].style.border = '1px solid #66bb6a';
        }
        
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
            maskCanvas.style.opacity = "1";
            paintCanvas.style.opacity = "0.3";
            colorInput.disabled = true;
            colorInput.style.opacity = '0.3';
        } else {
            maskCanvas.style.opacity = "0.5";
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
    setShape('round');

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
        updateCursorPos();
    };

    // --- History ---
    const saveState = () => {
        state.history = state.history.slice(0, state.historyIndex + 1);
        state.history.push({
            paint: paintCanvas.toDataURL(),
            mask: maskCanvas.toDataURL()
        });
        if (state.history.length > 30) state.history.shift();
        else state.historyIndex++;
        
        undoBtn.style.opacity = state.historyIndex > 0 ? "1" : "0.5";
        redoBtn.style.opacity = state.historyIndex < state.history.length - 1 ? "1" : "0.5";
    };

    const restoreState = (hState) => {
        const pImg = new Image();
        pImg.onload = () => { pCtx.clearRect(0, 0, canvasW, canvasH); pCtx.drawImage(pImg, 0, 0); };
        pImg.src = hState.paint;
        
        const mImg = new Image();
        mImg.onload = () => { mCtx.clearRect(0, 0, canvasW, canvasH); mCtx.drawImage(mImg, 0, 0); };
        mImg.src = hState.mask;
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
        const isMask = state.layer === 'mask';
        const targetCtx = isMask ? mCtx : pCtx;
        const targetData = targetCtx.getImageData(0, 0, canvasW, canvasH);
        const tempData = tempCtx.getImageData(0, 0, canvasW, canvasH);
        
        let r = 0, g = 0, b = 0;
        if (!isMask && state.color) {
            const hex = state.color.replace('#', '');
            r = parseInt(hex.substring(0, 2), 16);
            g = parseInt(hex.substring(2, 4), 16);
            b = parseInt(hex.substring(4, 6), 16);
        } // mask uses black (0,0,0)
        
        const targetAlpha = Math.round(state.opacity * 255);
        
        for (let i = 0; i < tempData.data.length; i += 4) {
            if (tempData.data[i+3] > 0) { // If stroke touched this pixel
                if (state.tool === 'eraser') {
                    targetData.data[i+3] = 0; // erase completely
                } else {
                    targetData.data[i] = r;
                    targetData.data[i+1] = g;
                    targetData.data[i+2] = b;
                    targetData.data[i+3] = targetAlpha; // exact replacement
                }
            }
        }
        targetCtx.putImageData(targetData, 0, 0);
        tempCtx.clearRect(0, 0, canvasW, canvasH);
    };

    const drawLine = (ctx, from, to) => {
        ctx.globalCompositeOperation = 'source-over';
        ctx.globalAlpha = 1.0;
        
        if (state.tool === 'eraser') {
            ctx.strokeStyle = 'rgba(255, 255, 255, 1)'; // 橡皮擦显示为白色
        } else {
            ctx.strokeStyle = state.layer === 'paint' ? state.color : state.maskColor;
        }
        ctx.lineWidth = state.size;
        ctx.lineCap = state.shape === 'round' ? 'round' : 'square';
        ctx.lineJoin = state.shape === 'round' ? 'round' : 'miter';

        ctx.beginPath();
        ctx.moveTo(from.x, from.y);
        ctx.lineTo(to.x, to.y);
        ctx.stroke();
    };

    workArea.addEventListener("mousedown", (e) => {
        if (e.button === 1 || (e.button === 0 && (state.tool === 'pan' || e.shiftKey || e.code === "Space"))) {
            isPanning = true;
            panStart.x = e.clientX - state.panX;
            panStart.y = e.clientY - state.panY;
            workArea.style.cursor = "grabbing";
        } else if (e.button === 0) {
            const pos = getPos(e);
            
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
            tempCanvas.style.opacity = state.tool === 'eraser' ? 0.5 : state.opacity;
            drawLine(tempCtx, pos, { x: pos.x + 0.1, y: pos.y + 0.1 });
        }
    });

    window.addEventListener("mousemove", (e) => {
        if (isPanning) {
            state.panX = e.clientX - panStart.x;
            state.panY = e.clientY - panStart.y;
            updateTransform();
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
        if (isDrawing) {
            isDrawing = false;
            mergeTempCanvas();
            saveState(); // push to history
        }
    });

    workArea.addEventListener("wheel", (e) => {
        e.preventDefault();
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
            if (myLoadId !== currentLoadId) return;
            origW = baseImg.naturalWidth;
            origH = baseImg.naturalHeight;
            
            let w = origW, h = origH;
            if (w > MAX_DIM || h > MAX_DIM) {
                const ratio = Math.min(MAX_DIM / w, MAX_DIM / h);
                w = Math.floor(w * ratio);
                h = Math.floor(h * ratio);
            }
            canvasW = w; canvasH = h;
            
            canvasWrapper.style.width = `${canvasW}px`;
            canvasWrapper.style.height = `${canvasH}px`;
            
            pCtx.clearRect(0, 0, canvasW, canvasH);
            mCtx.clearRect(0, 0, canvasW, canvasH);
            tempCtx.clearRect(0, 0, canvasW, canvasH);
            paintCanvas.width = canvasW; paintCanvas.height = canvasH;
            maskCanvas.width = canvasW; maskCanvas.height = canvasH;
            tempCanvas.width = canvasW; tempCanvas.height = canvasH;

            // Initialize UI state completely before doing anything else
            setTool('brushMask');
            setShape('round');
            
            // Force the layer UI update explicitly for the first time
            setLayer('mask');

            // Setup base image thumbnail
            const thumbDiv = document.getElementById("A_base_thumb");
            if (thumbDiv) {
                thumbDiv.style.backgroundImage = `url("${baseImg.src}")`;
            }
            
            infoLabel.innerHTML = `${Math.round((canvasW/origW)*100)}%<br/>${origW}x${origH}`;
            
            // Center view
            const availableW = workArea.clientWidth;
            const availableH = workArea.clientHeight;
            state.scale = Math.min(availableW / canvasW, availableH / canvasH, 1) * 0.9;
            state.panX = (availableW - canvasW * state.scale) / 2;
            state.panY = (availableH - canvasH * state.scale) / 2;
            updateTransform();

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
            
            const origPath = (node && node.properties && node.properties.original_image_paths) ? node.properties.original_image_paths[index] : imagePaths[index];

            // 只要点击了保存，哪怕只是加载了历史记录没有新增绘制，我们也强制重新上传
            // 因为用户可能在别的节点复用了原图，这里手动保存就应该生成一个独立的新实例
            const resp = await api.fetchApi("/a_my_nodes/upload_custom_edited_image", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    image_path: origPath,
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
                if (onSaveCallback) onSaveCallback(imagePaths);
                cleanup();
            }
        } catch (err) {
            console.error(err);
            alert("保存失败: " + err.message);
            saveBtn.textContent = "💾 保存";
            saveBtn.disabled = false;
        }
    };

    const cleanup = () => {
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