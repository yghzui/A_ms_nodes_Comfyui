import { app } from "../../../../scripts/app.js";
import { api } from "../../../../scripts/api.js";
import { drawNumberWidgetPart, drawRoundedRectangle, drawTogglePart, fitString, isLowQuality } from "../utils/utils_canvas.js";
import { RgthreeBaseWidget } from "../utils/utils_widgets.js";
import { modal } from "../utils/modal.js";
import { rgthreeApi } from "../core/rgthree_api.js";

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

export function showStatusLabelSettingsModal() {
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
export async function showLoraChooser(event, callback, parentMenu, loras, buttonNode) {
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

export class LabelWidget extends RgthreeBaseWidget {
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
        this._drawNode = node; // Save node reference for interaction
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

export class LoraWidget extends RgthreeBaseWidget {
    constructor(name, group, color = "#333") {
        super(name);
        this.color = color;
        this.group = group;
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
        const color = this.color;
        drawRoundedRectangle(ctx, { pos: [posX, posY], size: [node.size[0] - margin * 2, height], color: color, strokeColor: "#555" });

        this.hitAreas.toggle.bounds = drawTogglePart(ctx, { posX, posY, height, value: this.value.on });
        posX += this.hitAreas.toggle.bounds[1] + innerMargin;

        if (lowQuality) { ctx.restore(); return; }
        if (!this.value.on) ctx.globalAlpha = app.canvas.editor_alpha * 0.4;

        ctx.fillStyle = LiteGraph.WIDGET_TEXT_COLOR;
        const [leftArrow, text, rightArrow] = drawNumberWidgetPart(ctx, { posX: node.size[0] - margin - innerMargin, posY, height, value: this.value.strength ?? 1, direction: -1 });
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
            this.value.strength = (this.value.strength ?? 1) + event.deltaX * 0.05;
            this.value.strength = Math.round(this.value.strength * 100) / 100;
        }
    }
    onStrengthValDown(event) {
        const now = Date.now();
        if (this.lastClickTime && (now - this.lastClickTime < 300)) {
            app.canvas.prompt("Strength", this.value.strength ?? 1, (v) => { this.value.strength = Number(v); }, event);
            this.lastClickTime = 0;
            return true;
        }
        this.lastClickTime = now;
        return true;
    }
    stepStrength(direction) {
        let step = 0.05;
        this.value.strength = Math.round(((this.value.strength ?? 1) + step * direction) * 100) / 100;
    }
}

export class StatusLabelWidget extends RgthreeBaseWidget {
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
        const displayStatus = statusText.includes("✅") || statusText.includes("Force True") ? "✅" :
            statusText.includes("❌") || statusText.includes("Force False") ? "❌" : "";

        const text = `${displayStatus} ${keyText}`.trim();

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

export class DoubleButtonWidget extends RgthreeBaseWidget {
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
        this._drawNode = node;
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
        if (this.callback1) this.callback1(event, pos, node || this._drawNode);
        return true;
    }

    onBtn2Down(event, pos, node) {
        if (this.callback2) this.callback2(event, pos, node || this._drawNode);
        return true;
    }
}
