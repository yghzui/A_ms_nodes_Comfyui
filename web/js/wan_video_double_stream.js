import { app } from "../../../scripts/app.js";
import { drawNumberWidgetPart, drawRoundedRectangle, drawTogglePart, fitString, isLowQuality } from "./utils/utils_canvas.js";
import { RgthreeBaseWidget, RgthreeBetterButtonWidget } from "./utils/utils_widgets.js";
import { modal } from "./utils/modal.js";
import { moveArrayItem, showTopNotification } from "./utils/shared_utils.js";
import { rgthree } from "./core/rgthree.js";
import { api } from "../../../scripts/api.js";
import { rgthreeApi } from "./core/rgthree_api.js";

// --- Helper Functions ---

// --- Status Label Settings ---
let statusLabelSettings = JSON.parse(localStorage.getItem("wan_video_status_label_settings") || "{}");
statusLabelSettings = {
    fontSize: statusLabelSettings.fontSize || 24,
    textColor: statusLabelSettings.textColor || "#ffffff",
    bgColor: statusLabelSettings.bgColor || "#000000",
    opacity: statusLabelSettings.opacity !== undefined ? statusLabelSettings.opacity : 0.8
};

function saveStatusLabelSettings() {
    localStorage.setItem("wan_video_status_label_settings", JSON.stringify(statusLabelSettings));
    app.canvas.setDirty(true, true);
}

function showStatusLabelSettingsModal() {
    const content = `
        <div style="display:flex; flex-direction:column; gap:10px; color: white;">
            <label>字体大小 (px):</label>
            <input type="number" id="lbl-fontsize" value="${statusLabelSettings.fontSize}" style="background:#333; color:white; border:1px solid #555; padding:5px;">
            <label>文字颜色:</label>
            <input type="color" id="lbl-textcolor" value="${statusLabelSettings.textColor}" style="width:100%; height:30px; padding:0; border:none;">
            <label>背景颜色:</label>
            <input type="color" id="lbl-bgcolor" value="${statusLabelSettings.bgColor}" style="width:100%; height:30px; padding:0; border:none;">
            <label>透明度 (0.0 - 1.0):</label>
            <input type="number" step="0.1" min="0" max="1" id="lbl-opacity" value="${statusLabelSettings.opacity}" style="background:#333; color:white; border:1px solid #555; padding:5px;">
        </div>
    `;

    modal.show({
        title: "全局标签显示设置",
        content: content,
        width: "300px",
        buttons: [
            {
                text: "保存",
                type: "primary",
                onClick: () => {
                    statusLabelSettings.fontSize = parseInt(document.getElementById("lbl-fontsize").value) || 24;
                    statusLabelSettings.textColor = document.getElementById("lbl-textcolor").value;
                    statusLabelSettings.bgColor = document.getElementById("lbl-bgcolor").value;
                    statusLabelSettings.opacity = parseFloat(document.getElementById("lbl-opacity").value);
                    if (isNaN(statusLabelSettings.opacity)) statusLabelSettings.opacity = 0.8;
                    saveStatusLabelSettings();
                    modal.close();
                }
            },
            { text: "取消", onClick: () => modal.close() }
        ]
    });
}


// Helper function to show the LoRA chooser menu (From LoadLoraMerge)
async function showLoraChooser(event, callback, parentMenu, loras, buttonNode) {
    const canvas = app.canvas;
    if (!loras) {
        try {
            const useOfficial = buttonNode?.properties?.useOfficialLoraList ?? false;
            if (useOfficial) {
                const loraFiles = await api.getModels('loras');
                loras = ["None", ...loraFiles.map((l) => l.name)];
            } else {
                loras = ["None", ...(await rgthreeApi.getLoras().then((loras) => loras.map((l) => l.file)))];
            }
        } catch (e) {
            console.error("Failed to fetch LoRAs:", e);
            loras = ["None"];
        }
    }

    const menuItems = loras.map(lora => ({
        content: lora,
        callback: () => callback(lora)
    }));
    
    let menuEvent = event;
    let targetX, targetY;

    if (event && event.clientX !== undefined) {
        targetX = event.clientX;
        targetY = event.clientY;
        menuEvent = new MouseEvent('contextmenu', {
            clientX: targetX,
            clientY: targetY,
            bubbles: true,
            cancelable: true,
            view: window
        });
    }

    const contextMenu = new LiteGraph.ContextMenu(menuItems, {
        event: menuEvent,
        parentMenu: parentMenu || undefined,
        title: "Select LoRA",
        scale: Math.max(1, canvas.ds?.scale || 1),
        className: "dark",
        callback,
    });

    // Simple positioning fix if needed, but LiteGraph usually handles it well with event
    if (contextMenu && contextMenu.root && targetX !== undefined && targetY !== undefined) {
         // Let LiteGraph handle initial positioning, just ensure it stays on screen if we want
         // But the previous complex manual positioning might have been fighting with LiteGraph's own logic
         // or the custom widget's coordinate system.
         // Let's simplify and rely on the mouse event which is robust.
    }
}

// --- Custom Widgets ---

class LabelWidget extends RgthreeBaseWidget {
    constructor(name, label, toggleCallback) {
        super(name);
        this.label = label;
        this.type = "custom";
        this.toggleCallback = toggleCallback;
        this.collapsed = false;
        this.hitAreas = {
            toggle: { bounds: [0, 0], onDown: this.onToggleDown.bind(this) }
        };
    }
    draw(ctx, node, w, posY, height) {
        this.node = node; // Save node reference for interaction
        ctx.save();
        ctx.fillStyle = "#888"; 
        ctx.font = "bold 12px Arial";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(this.label, node.size[0] / 2, posY + height / 2);
        
        if (this.toggleCallback) {
            const icon = this.collapsed ? "🙈" : "👁️";
            const toggleW = 30;
            const toggleX = node.size[0] - toggleW;
            
            ctx.textAlign = "center";
            ctx.fillStyle = this.collapsed ? "#666" : "#aaa";
            ctx.fillText(icon, toggleX + toggleW/2 - 5, posY + height / 2);
            
            this.hitAreas.toggle.bounds = [toggleX, toggleW];
        }
        
        ctx.restore();
    }
    
    onToggleDown() {
        if (this.toggleCallback) {
            this.collapsed = !this.collapsed;
            this.toggleCallback(this.collapsed);
            return true;
        }
        return false;
    }

    setCollapsed(collapsed) {
        if (this.collapsed !== collapsed) {
            this.collapsed = collapsed;
            if (this.toggleCallback) {
                this.toggleCallback(this.collapsed);
            }
        }
    }
}

// Triple Toggle Widget for settings (Copied from LoadLoraMerge)
class LoadLoraMergeTripleToggleWidget extends RgthreeBaseWidget {
    constructor(name, label1, label2, label3, defaultValue1, defaultValue2, defaultValue3) {
        super(name);
        this.type = "custom";
        this.label1 = label1;
        this.label2 = label2;
        this.label3 = label3;
        this.hitAreas = {
            toggle1: { bounds: [0, 0], onDown: this.onToggle1Down.bind(this) },
            toggle2: { bounds: [0, 0], onDown: this.onToggle2Down.bind(this) },
            toggle3: { bounds: [0, 0], onDown: this.onToggle3Down.bind(this) },
        };
        this._value1 = defaultValue1;
        this._value2 = defaultValue2;
        this._value3 = defaultValue3;
    }

    set value(v) {
        if (typeof v === 'object' && v !== null) {
            this._value1 = v.value1 !== undefined ? v.value1 : this._value1;
            this._value2 = v.value2 !== undefined ? v.value2 : this._value2;
            this._value3 = v.value3 !== undefined ? v.value3 : this._value3;
        }
    }

    get value() {
        return {
            value1: this._value1,
            value2: this._value2,
            value3: this._value3,
        };
    }

    draw(ctx, node, w, posY, height) {
        ctx.save();
        const margin = 10, innerMargin = margin * 0.33, lowQuality = isLowQuality(), midY = posY + height * 0.5;
        let posX = margin;
        drawRoundedRectangle(ctx, { pos: [posX, posY], size: [node.size[0] - margin * 2, height] });
        
        // Toggle 1 (Toggle All)
        this.hitAreas.toggle1.bounds = drawTogglePart(ctx, { posX, posY, height, value: this._value1 });
        posX += this.hitAreas.toggle1.bounds[1] + innerMargin;
        if (lowQuality) { ctx.restore(); return; }
        ctx.fillStyle = LiteGraph.WIDGET_TEXT_COLOR;
        ctx.textAlign = "left";
        ctx.textBaseline = "middle";
        ctx.fillText(this.label1, posX, midY);
        posX += ctx.measureText(this.label1).width + innerMargin * 2;

        // Toggle 2 (Low Mem)
        this.hitAreas.toggle2.bounds = drawTogglePart(ctx, { posX, posY, height, value: this._value2 });
        posX += this.hitAreas.toggle2.bounds[1] + innerMargin;
        ctx.fillText(this.label2, posX, midY);
        posX += ctx.measureText(this.label2).width + innerMargin * 2;

        // Toggle 3 (Merge)
        this.hitAreas.toggle3.bounds = drawTogglePart(ctx, { posX, posY, height, value: this._value3 });
        posX += this.hitAreas.toggle3.bounds[1] + innerMargin;
        ctx.fillText(this.label3, posX, midY);
        posX += ctx.measureText(this.label3).width + innerMargin * 2;

        ctx.restore();
    }

    serializeValue(node, index) { return this.value; }
    onToggle1Down() { 
        this._value1 = !this._value1; 
        this.cancelMouseDown(); 
        if (this.properties && typeof this.properties.onToggle1 === 'function') {
            this.properties.onToggle1(this._value1);
        }
        return true; 
    }
    onToggle2Down() { this._value2 = !this._value2; this.cancelMouseDown(); return true; }
    onToggle3Down() { this._value3 = !this._value3; this.cancelMouseDown(); return true; }
}

class WanVideoLoraWidget extends RgthreeBaseWidget {
    constructor(name, streamType) {
        super(name);
        this.streamType = streamType; // "High" or "Low"
        this.type = "custom";
        this.hitAreas = {
            toggle: { bounds: [0, 0], onDown: this.onToggleDown.bind(this) },
            lora: { bounds: [0, 0], onDown: this.onLoraDown.bind(this) },
            strengthDec: { bounds: [0, 0], onDown: this.onStrengthDecDown.bind(this) },
            strengthVal: { bounds: [0, 0], onDown: this.onStrengthValDown.bind(this), onMove: this.onStrengthAnyMove.bind(this) },
            strengthInc: { bounds: [0, 0], onDown: this.onStrengthIncDown.bind(this) },
            strengthAny: { bounds: [0, 0], onMove: this.onStrengthAnyMove.bind(this) },
        };
        this._value = { on: true, lora: null, strength: 1 };
    }

    set value(v) { 
        this._value = v;
        if (typeof this._value !== "object") {
            this._value = { on: true, lora: null, strength: 1 };
        }
    }
    get value() { return this._value; }
    setLora(lora) { this._value.lora = lora; }

    draw(ctx, node, w, posY, height) {
        ctx.save();
        const margin = 10, innerMargin = margin * 0.33, lowQuality = isLowQuality(), midY = posY + height * 0.5;
        let posX = margin;
        
        // Background
        const color = this.streamType === "High" ? "#2a3b4d" : "#4d2a2a"; // Blueish for High, Reddish for Low
        drawRoundedRectangle(ctx, { pos: [posX, posY], size: [node.size[0] - margin * 2, height], color: color, strokeColor: "#555" });
        
        this.hitAreas.toggle.bounds = drawTogglePart(ctx, { posX, posY, height, value: this.value.on });
        posX += this.hitAreas.toggle.bounds[1] + innerMargin;
        
        if (lowQuality) { ctx.restore(); return; }
        if (!this.value.on) ctx.globalAlpha = app.canvas.editor_alpha * 0.4;
        
        ctx.fillStyle = LiteGraph.WIDGET_TEXT_COLOR;
        const [leftArrow, text, rightArrow] = drawNumberWidgetPart(ctx, { posX: node.size[0] - margin - innerMargin, posY, height, value: this.value.strength || 1, direction: -1 });
        this.hitAreas.strengthDec.bounds = leftArrow;
        this.hitAreas.strengthVal.bounds = text;
        this.hitAreas.strengthInc.bounds = rightArrow;
        this.hitAreas.strengthAny.bounds = [leftArrow[0], rightArrow[0] + rightArrow[1] - leftArrow[0]];
        
        const rposX = leftArrow[0] - innerMargin;
        const loraWidth = rposX - posX;
        ctx.textAlign = "left";
        ctx.textBaseline = "middle";
        ctx.fillText(fitString(ctx, String(this.value.lora || "None"), loraWidth), posX, midY);
        this.hitAreas.lora.bounds = [posX, loraWidth];
        
        ctx.globalAlpha = app.canvas.editor_alpha;
        ctx.restore();
    }

    serializeValue(node, index) { return { ...this.value }; }
    onToggleDown() { this.value.on = !this.value.on; this.cancelMouseDown(); return true; }
    onLoraDown(event, pos, node) {
        const targetNode = node || this.node;
        showLoraChooser(event, (value) => {
            if (typeof value === "string") this.value.lora = value;
            targetNode?.setDirtyCanvas(true, true);
        }, null, null, targetNode);
        return true;
    }
    onStrengthDecDown() { this.stepStrength(-1); return true;}
    onStrengthIncDown() { this.stepStrength(1); return true;}
    onStrengthAnyMove(event) { 
        if (event.deltaX) { 
            this.value.strength = (this.value.strength || 1) + event.deltaX * 0.05; 
            this.value.strength = Math.round(this.value.strength * 100) / 100;
        } 
    }
    onStrengthValDown(event) {
        const now = Date.now();
        if (this.lastClickTime && (now - this.lastClickTime < 300)) {
            app.canvas.prompt("Strength", this.value.strength || 1, (v) => { this.value.strength = Number(v); }, event);
            this.lastClickTime = 0;
            return true;
        }
        this.lastClickTime = now;
        return true;
    }
    stepStrength(direction) {
        let step = 0.05;
        this.value.strength = Math.round(((this.value.strength || 1) + step * direction) * 100) / 100;
    }
}

class BigStatusLabelWidget extends RgthreeBaseWidget {
    constructor(name) {
        super(name);
        this.type = "custom";
    }

    computeSize(width) {
        return [width, statusLabelSettings.fontSize + 20];
    }

    draw(ctx, node, w, posY, height) {
        const wKey = node.widgets.find(w => w.name === "key_to_check");
        const wStatus = node.widgets.find(w => w.name === "Check Status");
        
        const keyText = wKey ? wKey.value : "";
        const statusText = wStatus ? wStatus.value : "";
        
        // Clean up status text for cleaner display
        let displayStatus = statusText;
        if (displayStatus.includes("✅")) displayStatus = "✅ 已启用";
        else if (displayStatus.includes("❌")) displayStatus = "❌ 未启用";
        else if (displayStatus.includes("🔒 Force True")) displayStatus = "✅ 强制启用";
        else if (displayStatus.includes("🔒 Force False")) displayStatus = "❌ 强制关闭";
        
        const text = `🔍 ${keyText}  |  ${displayStatus}`;

        ctx.save();
        
        ctx.globalAlpha = statusLabelSettings.opacity;
        
        // Draw background
        drawRoundedRectangle(ctx, { pos: [0, posY], size: [w, height], color: statusLabelSettings.bgColor, radius: 5 });
        
        ctx.globalAlpha = 1.0;
        ctx.fillStyle = statusLabelSettings.textColor;
        ctx.font = `bold ${statusLabelSettings.fontSize}px Arial`;
        ctx.textAlign = "left";
        ctx.textBaseline = "middle";
        
        ctx.fillText(fitString(ctx, text, w - 20), 10, posY + height / 2);
        
        ctx.restore();
        
        this.last_y = posY;
    }
}

class DoubleButtonWidget extends RgthreeBaseWidget {
    constructor(name, label1, label2, callback1, callback2) {
        super(name);
        this.type = "custom";
        this.label1 = label1;
        this.label2 = label2;
        this.callback1 = callback1;
        this.callback2 = callback2;
        this.hitAreas = {
            btn1: { bounds: [0, 0], onDown: this.onBtn1Down.bind(this) },
            btn2: { bounds: [0, 0], onDown: this.onBtn2Down.bind(this) }
        };
    }

    draw(ctx, node, w, posY, height) {
        this.node = node;
        ctx.save();
        
        const margin = 10;
        const innerMargin = 5;
        const btnWidth = (w - margin * 2 - innerMargin) / 2;
        
        // Button 1
        const btn1X = margin;
        ctx.fillStyle = "#333";
        ctx.beginPath();
        ctx.roundRect(btn1X, posY + 2, btnWidth, height - 4, 4);
        ctx.fill();
        ctx.strokeStyle = "#555";
        ctx.stroke();
        
        ctx.fillStyle = "#fff";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.font = "12px Arial";
        ctx.fillText(this.label1, btn1X + btnWidth / 2, posY + height / 2);
        
        this.hitAreas.btn1.bounds = [btn1X, btnWidth];
        
        // Button 2
        const btn2X = margin + btnWidth + innerMargin;
        ctx.fillStyle = "#333";
        ctx.beginPath();
        ctx.roundRect(btn2X, posY + 2, btnWidth, height - 4, 4);
        ctx.fill();
        ctx.strokeStyle = "#555";
        ctx.stroke();
        
        ctx.fillStyle = "#fff";
        ctx.fillText(this.label2, btn2X + btnWidth / 2, posY + height / 2);
        
        this.hitAreas.btn2.bounds = [btn2X, btnWidth];
        
        ctx.restore();
    }

    onBtn1Down(event, pos, node) {
        if (this.callback1) this.callback1(event, pos, node || this.node);
        return true;
    }

    onBtn2Down(event, pos, node) {
        if (this.callback2) this.callback2(event, pos, node || this.node);
        return true;
    }
}

// --- Import/Export Helper Functions ---
function handleExport(node) {
    const data = {
        key_to_check: node.widgets.find(w => w.name === "key_to_check")?.value || "",
        check_mode: node.widgets.find(w => w.name === "check_mode")?.value || "contains",
        high: {
            settings: node.widgets.find(w => w.name === "settings_high")?.value || {},
            loras: node.highLoraWidgets.map(w => w.value)
        },
        low: {
            settings: node.widgets.find(w => w.name === "settings_low")?.value || {},
            loras: node.lowLoraWidgets.map(w => w.value)
        }
    };
    
    modal.show({
        title: "导出节点配置",
        content: "请选择导出方式：",
        buttons: [
            {
                text: "导出为 JSON 文件",
                type: "primary",
                onClick: () => {
                    const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement("a");
                    a.href = url;
                    a.download = "wan_video_node_config.json";
                    a.click();
                    URL.revokeObjectURL(url);
                    modal.close();
                }
            },
            {
                text: "复制到剪贴板",
                type: "secondary",
                onClick: () => {
                    navigator.clipboard.writeText(JSON.stringify(data, null, 2))
                        .then(() => showTopNotification("已复制到剪贴板", "success"))
                        .catch(err => showTopNotification("复制失败: " + err, "error"));
                    modal.close();
                }
            },
            { text: "取消", onClick: () => modal.close() }
        ]
    });
}

function handleImport(node) {
    const content = `
        <div style="display:flex; flex-direction:column; gap:10px;">
            <label>粘贴内容 (JSON):</label>
            <textarea id="import-text" class="custom-modal-textarea" placeholder="在此粘贴..."></textarea>
            <label>或 选择文件:</label>
            <input type="file" id="import-file" accept=".json" class="custom-modal-file-input">
        </div>
    `;

    modal.show({
        title: "导入节点配置 (追加模式)",
        content: content,
        width: "500px",
        buttons: [
            {
                text: "确认导入",
                type: "primary",
                onClick: async () => {
                    const textEl = document.getElementById("import-text");
                    const fileEl = document.getElementById("import-file");
                    let rawData = textEl.value.trim();

                    if (fileEl.files.length > 0) {
                        const file = fileEl.files[0];
                        rawData = await file.text();
                    }

                    if (!rawData) {
                        showTopNotification("请输入内容或选择文件", "warning");
                        return;
                    }

                    processImport(node, rawData);
                    modal.close();
                }
            },
            { text: "取消", onClick: () => modal.close() }
        ]
    });
}

function processImport(node, rawData) {
    try {
        const parsed = JSON.parse(rawData);
        if (!parsed || (typeof parsed !== 'object')) {
            throw new Error("Invalid format");
        }

        // Load key_to_check (Append logic)
        if (parsed.key_to_check || parsed.keyword) {
            const newKey = parsed.key_to_check || parsed.keyword;
            const wKey = node.widgets.find(w => w.name === "key_to_check");
            if (wKey) {
                const currentKeys = wKey.value.split(';').map(k => k.trim()).filter(k => k);
                if (!currentKeys.includes(newKey)) {
                    currentKeys.push(newKey);
                    wKey.value = currentKeys.join(';');
                }
            }
        }
        
        // Mode remains untouched as per requirement "模式不用管"

        // Load High Stream (Append mode)
        const highData = parsed.high || parsed.high_loras;
        if (highData) {
            if (highData.settings) {
                const settingsHigh = node.widgets.find(w => w.name === "settings_high");
                if (settingsHigh) settingsHigh.value = highData.settings;
            }
            const loras = Array.isArray(highData) ? highData : highData.loras;
            if (Array.isArray(loras)) {
                loras.forEach(l => node.addNewLoraWidget("High", l));
            }
        }

        // Load Low Stream (Append mode)
        const lowData = parsed.low || parsed.low_loras;
        if (lowData) {
            if (lowData.settings) {
                const settingsLow = node.widgets.find(w => w.name === "settings_low");
                if (settingsLow) settingsLow.value = lowData.settings;
            }
            const loras = Array.isArray(lowData) ? lowData : lowData.loras;
            if (Array.isArray(loras)) {
                loras.forEach(l => node.addNewLoraWidget("Low", l));
            }
        }

        node.reorderWidgets();
        node.ensureHiddenWidgets();
        node.setDirtyCanvas(true, true);
        showTopNotification("导入成功 (已追加)", "success");
    } catch (e) {
        console.error("Import error:", e);
        showTopNotification("未识别到有效的配置数据，请检查格式。", "error");
    }
}

// --- Main Extension ---

app.registerExtension({
    name: "A_my_nodes.WanVideoDoubleStream.UI",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "WanVideoDoubleStream") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function() {
            if (onNodeCreated) onNodeCreated.apply(this, arguments);

            this.highLoraWidgets = [];
            this.lowLoraWidgets = [];
            this.loraWidgetsCounter = 0;

            // 1. Setup Status Widget
            this.statusWidget = this.addWidget("text", "Check Status", "Pending...", () => {}, {});
            if (this.statusWidget && this.statusWidget.inputEl) {
                this.statusWidget.inputEl.readOnly = true;
                this.statusWidget.inputEl.style.opacity = 0.6;
                this.statusWidget.inputEl.style.textAlign = "center";
            }

            // Helper for toggling
            const toggleWidgets = (names, isHidden) => {
                names.forEach(name => {
                    const w = this.widgets.find(w => w.name === name);
                    if (w) w.hidden = isHidden;
                });
            };

            // 2. Setup High Stream Section
            this.labelHigh = this.addCustomWidget(new LabelWidget("label_high", "🔼 High Stream", (collapsed) => {
                toggleWidgets(["settings_high", "➕ Add High LoRA", "✨ 从资产库选择 (High)"], collapsed);
                this.highLoraWidgets.forEach(w => w.hidden = collapsed);
                const newSize = this.computeSize();
                this.setSize([this.size[0], newSize[1]]);
                this.setDirtyCanvas(true, true);
            }));
            
            // Add settings for High Stream
            const settingsHigh = this.addCustomWidget(new LoadLoraMergeTripleToggleWidget("settings_high", "Toggle All", "Low Mem", "Merge", false, false, true));
            settingsHigh.properties = {
                onToggle1: (value) => {
                    for (const widget of this.highLoraWidgets) {
                        widget.value.on = value;
                    }
                }
            };
            
            this.btnAddHigh = this.addCustomWidget(new RgthreeBetterButtonWidget("➕ Add High LoRA", (e,p,n) => {
                 showLoraChooser(rgthree.lastCanvasMouseEvent || e, (value) => {
                    if (typeof value === "string" && value && value !== "None") this.addNewLoraWidget("High", { on: true, lora: value, strength: 1 });
                 }, null, null, this, this.btnAddHigh);
            }));

            // {{ AURA-X: Add - 资产库全局导入按钮 }}
            this.btnImportAsset = this.addCustomWidget(new DoubleButtonWidget(
                "asset_manager_btns", 
                "✨ 插入模板", 
                "💾 保存到资产库", 
                (e,p,n) => {
                    if (window.AssetManager) {
                        window.AssetManager.showDrawer(this, 'models', (selectedModels) => {
                            selectedModels.forEach(m => {
                                // 1. 追加 key_to_check
                                if (m.keyword) {
                                    const wKey = this.widgets.find(w => w.name === "key_to_check");
                                    if (wKey) {
                                        const currentKeys = wKey.value.split(';').map(k => k.trim()).filter(k => k);
                                        if (!currentKeys.includes(m.keyword)) {
                                            currentKeys.push(m.keyword);
                                            wKey.value = currentKeys.join(';');
                                        }
                                    }
                                }
                                
                                // 2. 注入 High Stream
                                if (Array.isArray(m.high_loras)) {
                                    m.high_loras.forEach(lora => {
                                        if (lora && lora.lora && lora.lora !== "None") {
                                            this.addNewLoraWidget("High", lora);
                                        }
                                    });
                                }
                                
                                // 3. 注入 Low Stream
                                if (Array.isArray(m.low_loras)) {
                                    m.low_loras.forEach(lora => {
                                        if (lora && lora.lora && lora.lora !== "None") {
                                            this.addNewLoraWidget("Low", lora);
                                        }
                                    });
                                }
                            });
                            this.reorderWidgets();
                            this.setDirtyCanvas(true, true);
                        });
                    } else {
                        if (window.AMDialog) {
                            window.AMDialog.alert("资产管理系统未就绪！");
                        } else {
                            alert("资产管理系统未就绪！");
                        }
                    }
                },
                async (e,p,n) => {
                    try {
                        const res = await api.fetchApi("/a_my_nodes/assets/models");
                        const modelsData = await res.json();
                        const groups = modelsData.groups || [];
                        
                        if (groups.length === 0) {
                            alert("资产库中没有模型分组，请先在资产库中创建分组！");
                            return;
                        }
                        
                        const groupOptions = groups.map((g, i) => `<option value="${i}">${g.name}</option>`).join("");
                        
                        const currentKey = this.widgets.find(w => w.name === "key_to_check")?.value || "未命名配置";
                        
                        const content = `
                            <div style="display:flex; flex-direction:column; gap:10px; color:white;">
                                <label>保存到分组:</label>
                                <select id="am-save-group" style="background:#333; color:white; padding:5px; border:1px solid #555; border-radius:4px;">
                                    ${groupOptions}
                                </select>
                                <label>条目标题 (key_to_check):</label>
                                <input type="text" id="am-save-title" value="${currentKey}" style="background:#333; color:white; padding:5px; border:1px solid #555; border-radius:4px;">
                            </div>
                        `;
                        
                        modal.show({
                            title: "保存到资产库",
                            content: content,
                            width: "400px",
                            buttons: [
                                {
                                    text: "保存",
                                    type: "primary",
                                    onClick: async () => {
                                        const groupIdx = parseInt(document.getElementById("am-save-group").value);
                                        const title = document.getElementById("am-save-title").value.trim() || "未命名配置";
                                        
                                        const newItem = {
                                            id: Date.now().toString() + Math.random().toString().slice(2, 6),
                                            keyword: title,
                                            check_mode: this.widgets.find(w => w.name === "check_mode")?.value || "contains",
                                            high_loras: this.highLoraWidgets.map(w => w.value).filter(v => v.lora && v.lora !== "None"),
                                            low_loras: this.lowLoraWidgets.map(w => w.value).filter(v => v.lora && v.lora !== "None"),
                                            preview_image: ""
                                        };
                                        
                                        groups[groupIdx].items.push(newItem);
                                        
                                        await api.fetchApi("/a_my_nodes/assets/models", {
                                            method: "POST",
                                            body: JSON.stringify(modelsData),
                                            headers: { "Content-Type": "application/json" }
                                        });
                                        
                                        if (window.AssetManager) {
                                            window.AssetManager.loadData();
                                        }
                                        
                                        showTopNotification("成功保存到资产库！", "success");
                                        modal.close();
                                    }
                                },
                                { text: "取消", onClick: () => modal.close() }
                            ]
                        });
                        
                    } catch (err) {
                        console.error(err);
                        alert("无法连接到资产库 API！");
                    }
                }
            ));

            // 3. Setup Low Stream Section
            this.labelLow = this.addCustomWidget(new LabelWidget("label_low", "🔽 Low Stream", (collapsed) => {
                toggleWidgets(["settings_low", "➕ Add Low LoRA"], collapsed);
                this.lowLoraWidgets.forEach(w => w.hidden = collapsed);
                const newSize = this.computeSize();
                this.setSize([this.size[0], newSize[1]]);
                this.setDirtyCanvas(true, true);
            }));

            // Add settings for Low Stream
            const settingsLow = this.addCustomWidget(new LoadLoraMergeTripleToggleWidget("settings_low", "Toggle All", "Low Mem", "Merge", false, false, true));
            settingsLow.properties = {
                onToggle1: (value) => {
                    for (const widget of this.lowLoraWidgets) {
                        widget.value.on = value;
                    }
                }
            };

            this.btnAddLow = this.addCustomWidget(new RgthreeBetterButtonWidget("➕ Add Low LoRA", (e,p,n) => {
                 showLoraChooser(rgthree.lastCanvasMouseEvent || e, (value) => {
                    if (typeof value === "string" && value && value !== "None") this.addNewLoraWidget("Low", { on: true, lora: value, strength: 1 });
                 }, null, null, this, this.btnAddLow);
            }));

            // Add Import/Export Buttons
            this.btnImportExport = this.addCustomWidget(new DoubleButtonWidget(
                "import_export_btns", 
                "📥 Import LoRAs", 
                "📤 Export LoRAs", 
                (e,p,n) => handleImport(this),
                (e,p,n) => handleExport(this)
            ));

            // 4. Setup Auto Enable Logic Section (Label only, inputs are standard)
            this.labelAuto = this.addCustomWidget(new LabelWidget("label_auto", "⚙️ Auto Enable Logic", (collapsed) => {
                toggleWidgets(["dict_input", "key_to_check", "check_mode"], collapsed);
                const newSize = this.computeSize();
                this.setSize([this.size[0], newSize[1]]);
                this.setDirtyCanvas(true, true);
            }));

            // Setup Big Status Label
            // this.bigStatusLabel = this.addCustomWidget(new BigStatusLabelWidget("big_status_label"));

            // 5. Initial Layout Reordering
            this.reorderWidgets();

            // 6. Ensure hidden widgets
            this.ensureHiddenWidgets();
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function(message) {
            if (onExecuted) onExecuted.apply(this, arguments);

            if (message && message.text && message.text.length > 0) {
                const w = this.widgets.find(w => w.name === "Check Status");
                if (w) {
                    const statusText = message.text[0];
                    const isEnabled = statusText === "True" || statusText === "Force True";
                    const isForceFalse = statusText === "Force False";
                    
                    if (statusText === "Force True") {
                        w.value = "🔒 Force True";
                    } else if (statusText === "Force False") {
                        w.value = "🔒 Force False";
                    } else {
                        w.value = isEnabled ? "✅ Auto: Enabled" : "❌ Auto: Disabled";
                    }
                    
                    if (w.inputEl) {
                        if (isEnabled) {
                            w.inputEl.style.color = "#4caf50"; // Green
                        } else if (isForceFalse) {
                            w.inputEl.style.color = "#f44336"; // Red
                        } else {
                            w.inputEl.style.color = "#f44336"; // Red
                        }
                        w.inputEl.style.fontWeight = "bold";
                    }
                }
            }
        };

        // Reorder widgets to match desired layout
        nodeType.prototype.reorderWidgets = function() {
            const getW = (name) => this.widgets.find(w => w.name === name);
            
            // Standard Widgets (from Python)
            const wEnable = getW("enable_mode");
            const wDict = getW("dict_input");
            const wKey = getW("key_to_check");
            const wCheck = getW("check_mode");
            const wModelH = getW("model_high");
            const wLoraH = getW("prev_lora_high");
            const wBlocksH = getW("blocks_high");
            const wModelL = getW("model_low");
            const wLoraL = getW("prev_lora_low");
            const wBlocksL = getW("blocks_low");
            const settingsHigh = getW("settings_high");
            const settingsLow = getW("settings_low");
            
            // Helper to move widget to end (building the list)
            const moveToBottom = (w) => {
                if (!w) return;
                const idx = this.widgets.indexOf(w);
                if (idx > -1) {
                    this.widgets.splice(idx, 1);
                    this.widgets.push(w);
                }
            };

            // Rebuild order
            // Top: Control & Status & Logic
            // Moving logic widgets to top ensures their indices are stable (0, 1, 2...)
            // regardless of how many dynamic LoRA widgets are added later.
            // This fixes the issue where dict_input receives a LoRA object value on reload.
            moveToBottom(this.labelAuto);
            moveToBottom(wEnable);
            moveToBottom(this.statusWidget);
            moveToBottom(wDict);
            moveToBottom(wKey);
            moveToBottom(wCheck);

            // High Stream
            moveToBottom(this.labelHigh);
            moveToBottom(wModelH);
            moveToBottom(wLoraH);
            moveToBottom(wBlocksH);
            moveToBottom(settingsHigh);
            this.highLoraWidgets.forEach(w => moveToBottom(w));
            moveToBottom(this.btnAddHigh);

            // Low Stream
            moveToBottom(this.labelLow);
            moveToBottom(wModelL);
            moveToBottom(wLoraL);
            moveToBottom(wBlocksL);
            moveToBottom(settingsLow);
            this.lowLoraWidgets.forEach(w => moveToBottom(w));
            moveToBottom(this.btnAddLow);

            // Import/Export and Asset Manager
            moveToBottom(this.btnImportAsset);
            moveToBottom(this.btnImportExport);

            // Big Status Label at the very bottom
            // moveToBottom(this.bigStatusLabel);

            // Logic Config (Previously at Bottom - Removed from here)
            // moveToBottom(this.labelAuto);
            // moveToBottom(wDict);
            // moveToBottom(wKey);
            // moveToBottom(wCheck);

            // Hidden stuff stays at very bottom (handled by ensureHiddenWidgets usually, but good to push here too)
            // const wInfoH = getW("loras_info_high");
            // const wInfoL = getW("loras_info_low");
            // moveToBottom(wInfoH);
            // moveToBottom(wInfoL);
        };

        nodeType.prototype.addNewLoraWidget = function(streamType, loraData = null) {
            this.loraWidgetsCounter++;
            const name = `LORA_${streamType}_${this.loraWidgetsCounter}`;
            const widget = new WanVideoLoraWidget(name, streamType);
            this.addCustomWidget(widget);
            
            if (loraData) widget.value = loraData;

            if (streamType === "High") {
                this.highLoraWidgets.push(widget);
            } else {
                this.lowLoraWidgets.push(widget);
            }

            this.reorderWidgets(); // Re-sort to put it in correct place
            return widget;
        };

        // Hidden Widgets Logic
        nodeType.prototype.ensureHiddenWidgets = function() {
            const hide = (name) => {
                const w = this.widgets.find(w => w.name === name);
                if (w) {
                    w.hidden = true;
                }
            };
            hide("loras_info_high");
            hide("loras_info_low");
        };

        // Serialization
        const onSerialize = nodeType.prototype.onSerialize;
        nodeType.prototype.onSerialize = function(o) {
            // Update hidden inputs from dynamic widgets
            const updateInfo = (infoName, widgetList, settingsName) => {
                const infoW = this.widgets.find(w => w.name === infoName);
                if (infoW) {
                    const settingsW = this.widgets.find(w => w.name === settingsName);
                    const settingsData = settingsW ? settingsW.value : {};
                    const data = { 
                        loras: widgetList.map(w => w.value),
                        settings: settingsData
                    };
                    infoW.value = JSON.stringify(data);
                }
            };
            
            updateInfo("loras_info_high", this.highLoraWidgets, "settings_high");
            updateInfo("loras_info_low", this.lowLoraWidgets, "settings_low");

            if (onSerialize) onSerialize.apply(this, arguments);
            
            // Save collapsed states
            o.collapsed_states = {
                high: this.labelHigh ? this.labelHigh.collapsed : false,
                low: this.labelLow ? this.labelLow.collapsed : false,
                auto: this.labelAuto ? this.labelAuto.collapsed : false
            };
        };

        // Configuration (Loading)
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function(info) {
            if (onConfigure) onConfigure.apply(this, arguments);
            
            // Clear existing dynamic widgets to avoid duplicates on reload
            const clearWidgets = (list) => {
                list.forEach(w => {
                    const idx = this.widgets.indexOf(w);
                    if (idx > -1) this.widgets.splice(idx, 1);
                });
            };
            clearWidgets(this.highLoraWidgets);
            clearWidgets(this.lowLoraWidgets);
            this.highLoraWidgets = [];
            this.lowLoraWidgets = [];

            // Load from hidden inputs
            const loadFromInfo = (infoName, type, settingsName) => {
                const infoW = this.widgets.find(w => w.name === infoName);
                if (infoW && infoW.value) {
                    try {
                        let data = JSON.parse(infoW.value);
                        
                        // Handle legacy format (direct array)
                        if (Array.isArray(data)) {
                            data = { loras: data };
                        }

                        if (data && Array.isArray(data.loras)) {
                            data.loras.forEach(l => this.addNewLoraWidget(type, l));
                        }

                        // Load settings
                        if (data && data.settings) {
                            const settingsW = this.widgets.find(w => w.name === settingsName);
                            if (settingsW) {
                                // Backward compatibility mapping if needed (e.g. if fields changed)
                                // For now assuming direct mapping as we just added it
                                settingsW.value = data.settings;
                            }
                        }

                    } catch (e) {
                        console.error(`Error loading ${infoName}`, e);
                    }
                }
            };

            loadFromInfo("loras_info_high", "High", "settings_high");
            loadFromInfo("loras_info_low", "Low", "settings_low");
            
            // Restore collapsed states
            if (info.collapsed_states) {
                if (this.labelHigh && info.collapsed_states.high) this.labelHigh.setCollapsed(true);
                if (this.labelLow && info.collapsed_states.low) this.labelLow.setCollapsed(true);
                if (this.labelAuto && info.collapsed_states.auto) this.labelAuto.setCollapsed(true);
            }

            this.reorderWidgets();
            this.ensureHiddenWidgets();
        };
        
        const onDrawForeground = nodeType.prototype.onDrawForeground;
        nodeType.prototype.onDrawForeground = function(ctx) {
            if (onDrawForeground) onDrawForeground.apply(this, arguments);

            const wKey = this.widgets.find(w => w.name === "key_to_check");
            const wStatus = this.widgets.find(w => w.name === "Check Status");
            
            const keyText = wKey ? wKey.value : "";
            const statusText = wStatus ? wStatus.value : "";
            
            let displayStatus = statusText;
            if (displayStatus.includes("✅")) displayStatus = "✅ 已启用";
            else if (displayStatus.includes("❌")) displayStatus = "❌ 未启用";
            else if (displayStatus.includes("🔒 Force True")) displayStatus = "✅ 强制启用";
            else if (displayStatus.includes("🔒 Force False")) displayStatus = "❌ 强制关闭";
            
            const text = `${keyText}  |  ${displayStatus}`;
            
            ctx.save();
            ctx.globalAlpha = statusLabelSettings.opacity;
            
            const margin = 10;
            const height = statusLabelSettings.fontSize + 20;
            const posY = this.size[1] + margin;
            const w = this.size[0];
            
            // Draw background
            drawRoundedRectangle(ctx, { pos: [0, posY], size: [w, height], color: statusLabelSettings.bgColor, radius: 5 });
            
            ctx.globalAlpha = 1.0;
            ctx.fillStyle = statusLabelSettings.textColor;
            ctx.font = `bold ${statusLabelSettings.fontSize}px Arial`;
            ctx.textAlign = "left";
            ctx.textBaseline = "middle";
            
            ctx.fillText(fitString(ctx, text, w - 20), 10, posY + height / 2);
            
            ctx.restore();
            
            // Save bounding box for hit testing
            this.bigLabelBounds = [0, posY, w, height];
        };

        // Context Menu for removal
        const getSlotMenuOptions = nodeType.prototype.getSlotMenuOptions;
        nodeType.prototype.getSlotMenuOptions = function(slot) {
            getSlotMenuOptions?.apply(this, arguments);
            
            if (slot && slot.output && slot.output.type === "STATUS_LABEL_WIDGET") {
                const menuItems = [
                    {
                        content: "⚙️ 设置标签样式 (全局)",
                        callback: () => {
                            showStatusLabelSettingsModal();
                        }
                    }
                ];
                new LiteGraph.ContextMenu(menuItems, { 
                    title: "Label Settings", 
                    event: rgthree.lastCanvasMouseEvent || event, 
                    className: "dark", 
                    scale: Math.max(1, app?.canvas?.ds?.scale || 1) 
                });
                return undefined;
            }

            if (!slot || !slot.widget || !slot.widget.streamType) return null;
            
            const widget = slot.widget;
            const streamList = widget.streamType === "High" ? this.highLoraWidgets : this.lowLoraWidgets;
            const index = streamList.indexOf(widget);
            
            if (index === -1) return null;

            const canMoveUp = index > 0;
            const canMoveDown = index < streamList.length - 1;
            const isFirst = index === 0;
            const isLast = index === streamList.length - 1;
            
            const menuItems = [
                { 
                    content: `${widget.value.on ? "⚫" : "🟢"} Toggle ${widget.value.on ? "Off" : "On"}`, 
                    callback: () => { 
                        widget.value.on = !widget.value.on; 
                        this.setDirtyCanvas(true, true);
                    } 
                },
                { 
                    content: `📋 Copy Path`, 
                    disabled: !widget.value.lora || widget.value.lora === "None", 
                    callback: () => {
                        const text = widget.value.lora;
                        navigator.clipboard.writeText(text).then(() => {
                            showTopNotification(`Copied: ${text}`, 'success');
                        }).catch(err => {
                            // Fallback
                            const textArea = document.createElement('textarea');
                            textArea.value = text;
                            document.body.appendChild(textArea);
                            textArea.select();
                            try {
                                document.execCommand('copy');
                                showTopNotification(`Copied: ${text}`, 'success');
                            } catch (err) {
                                showTopNotification('Copy failed', 'error');
                            }
                            document.body.removeChild(textArea);
                        });
                    } 
                },
                { 
                    content: `⬆️ Move Up`, 
                    disabled: !canMoveUp, 
                    callback: () => {
                        const temp = streamList[index];
                        streamList[index] = streamList[index - 1];
                        streamList[index - 1] = temp;
                        this.reorderWidgets();
                        this.setDirtyCanvas(true, true);
                    } 
                },
                { 
                    content: `⬇️ Move Down`, 
                    disabled: !canMoveDown, 
                    callback: () => {
                        const temp = streamList[index];
                        streamList[index] = streamList[index + 1];
                        streamList[index + 1] = temp;
                        this.reorderWidgets();
                        this.setDirtyCanvas(true, true);
                    } 
                },
                { 
                    content: `⏫ Move to Top`, 
                    disabled: isFirst || streamList.length <= 1, 
                    callback: () => {
                        streamList.splice(index, 1);
                        streamList.unshift(widget);
                        this.reorderWidgets();
                        this.setDirtyCanvas(true, true);
                    }
                },
                { 
                    content: `⏬ Move to Bottom`, 
                    disabled: isLast || streamList.length <= 1, 
                    callback: () => {
                        streamList.splice(index, 1);
                        streamList.push(widget);
                        this.reorderWidgets();
                        this.setDirtyCanvas(true, true);
                    }
                },
                { 
                    content: `🗑️ Delete`, 
                    callback: () => {
                        streamList.splice(index, 1);
                        const wIdx = this.widgets.indexOf(widget);
                        if (wIdx > -1) this.widgets.splice(wIdx, 1);
                        this.reorderWidgets();
                        this.setDirtyCanvas(true, true);
                    } 
                },
                { 
                    content: `🗑️ Clear All ${widget.streamType} Stream`, 
                    callback: () => { 
                        // Remove all widgets of this stream from main widget list
                        streamList.forEach(w => {
                            const wIdx = this.widgets.indexOf(w);
                            if (wIdx > -1) this.widgets.splice(wIdx, 1);
                        });
                        // Clear the tracking array
                        streamList.length = 0;
                        this.reorderWidgets();
                        this.setDirtyCanvas(true, true);
                    } 
                },
                {
                    content: this.properties?.useOfficialLoraList ? "Use Custom Lora List" : "Use Official Lora List",
                    callback: () => {
                        if (!this.properties) this.properties = {};
                        this.properties['useOfficialLoraList'] = !this.properties.useOfficialLoraList;
                    }
                },
            ];

            new LiteGraph.ContextMenu(menuItems, { 
                title: "LoRA Item", 
                event: rgthree.lastCanvasMouseEvent || event, 
                className: "dark", 
                scale: Math.max(1, app?.canvas?.ds?.scale || 1) 
            });
            return undefined;
        };

        const getExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
        nodeType.prototype.getExtraMenuOptions = function(canvas, options) {
            let menu = [];
            if (getExtraMenuOptions) {
                menu = getExtraMenuOptions.apply(this, arguments) || [];
            }
            
            const useOfficial = this.properties?.useOfficialLoraList ?? false;
            menu.push(
                {
                    content: "📥 Import LoRAs",
                    callback: () => handleImport(this)
                },
                {
                    content: "📤 Export LoRAs",
                    callback: () => handleExport(this)
                },
                null, // separator
                {
                    content: useOfficial ? "Use Custom Lora List (Default)" : "Use Official Lora List",
                    callback: () => {
                        if (!this.properties) this.properties = {};
                        this.properties['useOfficialLoraList'] = !useOfficial;
                    }
                }
            );
            
            return menu;
        };
        
        // Add getSlotInPosition for right-click menu support
        nodeType.prototype.getSlotInPosition = function(canvasX, canvasY) {
            let lastWidget = null;
            // Iterate widgets to find which one is under the mouse
            // The canvasY is relative to the node position usually? 
            // LiteGraph's getSlotInPosition is called with node-relative coordinates or canvas coordinates?
            // Wait, looking at LoadLoraMerge: 
            // if (canvasY > this.pos[1] + widget.last_y)
            // This suggests canvasY is in Canvas coordinates.
            
            // Let's verify coordinates. 
            // If getSlotInPosition is called by LiteGraph, it usually passes coordinates.
            // But LoadLoraMerge implementation implies canvasY is absolute canvas coord?
            // "this.pos[1]" is node Y. "widget.last_y" is relative Y.
            
            // Check if mouse is over the external big status label
            if (this.bigLabelBounds) {
                const [lx, ly, lw, lh] = this.bigLabelBounds;
                // canvasX and canvasY passed to getSlotInPosition are node-relative coordinates!
                if (canvasX >= lx && canvasX <= lx + lw && 
                    canvasY >= ly && canvasY <= ly + lh) {
                    return { widget: null, output: { type: "STATUS_LABEL_WIDGET" } };
                }
            }

            for (const widget of this.widgets) {
                if (widget.last_y === undefined) continue;
                
                // Simple hit test based on vertical position
                // Assuming widgets are stacked vertically and take full width
                const widgetY = this.pos[1] + widget.last_y;
                const widgetHeight = widget.computeSize ? widget.computeSize(this.size[0])[1] : LiteGraph.NODE_WIDGET_HEIGHT;
                
                if (canvasY >= widgetY && canvasY < widgetY + widgetHeight) {
                    lastWidget = widget;
                    break;
                }
            }

            if (lastWidget && lastWidget.streamType) {
                return { widget: lastWidget, output: { type: "LORA WIDGET" } };
            }
            return null;
        };
    }
});
