import { app } from "../../../scripts/app.js";
import { drawNumberWidgetPart, drawRoundedRectangle, drawTogglePart, fitString, isLowQuality, } from "./utils_canvas.js";
import { RgthreeBaseWidget, RgthreeBetterButtonWidget, RgthreeDividerWidget, } from "./utils_widgets.js";
import { rgthreeApi } from "./rgthree_api.js";
import { moveArrayItem, removeArrayItem, showTopNotification } from "./shared_utils.js";
import { rgthree } from "./rgthree.js";

console.log("Loaded load_lora_merge.js");

// Helper function to show the LoRA chooser menu
async function showLoraChooser(event, callback, parentMenu, loras, buttonNode, buttonWidget) {
    const canvas = app.canvas;
    if (!loras) {
        loras = ["None", ...(await rgthreeApi.getLoras().then((loras) => loras.map((l) => l.file)))];
    }
    
    const menuItems = loras.map(lora => ({ content: lora, callback: () => callback(lora) }));
    
    let menuEvent = event;
    let targetX, targetY;

    if (buttonNode && buttonWidget) {
        const canvasRect = canvas.canvas.getBoundingClientRect();
        const ds = canvas.ds || { scale: 1, offset: [0, 0] };
        const nodeX = buttonNode.pos[0];
        const nodeY = buttonNode.pos[1];
        const widgetY = buttonWidget.last_y || 0;
        const widgetLeftMargin = 15;
        
        const anchorCanvasX = nodeX + widgetLeftMargin;
        const anchorCanvasY = nodeY + widgetY;
        
        targetX = (anchorCanvasX + ds.offset[0]) * ds.scale + canvasRect.left;
        targetY = (anchorCanvasY + ds.offset[1]) * ds.scale + canvasRect.top;
        
        menuEvent = new MouseEvent('contextmenu', { clientX: targetX, clientY: targetY, bubbles: true, cancelable: true, view: window });
    } else if (event && event.clientX !== undefined) {
        targetX = event.clientX;
        targetY = event.clientY;
        menuEvent = new MouseEvent('contextmenu', { clientX: targetX, clientY: targetY, bubbles: true, cancelable: true, view: window });
    }

    const contextMenu = new LiteGraph.ContextMenu(menuItems, {
        event: menuEvent,
        parentMenu: parentMenu || undefined,
        title: "Select LoRA",
        scale: Math.max(1, canvas.ds?.scale || 1),
        className: "dark",
        callback,
    });

    if (contextMenu && contextMenu.root && targetX !== undefined && targetY !== undefined) {
        requestAnimationFrame(() => {
            const rect = contextMenu.root.getBoundingClientRect();
            const bodyRect = document.body.getBoundingClientRect();
            contextMenu.root.style.left = targetX + 'px';
            let finalY = targetY;
            if (bodyRect.height && targetY + rect.height > bodyRect.height - 10) {
                finalY = Math.max(10, bodyRect.height - rect.height - 10);
            }
            contextMenu.root.style.top = finalY + 'px';
            if (bodyRect.width && targetX + rect.width > bodyRect.width - 10) {
                contextMenu.root.style.left = Math.max(10, bodyRect.width - rect.width - 10) + 'px';
            }
        });
    }
}

// Dual Toggle Widget for settings
class LoadLoraMergeDualToggleWidget extends RgthreeBaseWidget {
    constructor(name, label1, label2, defaultValue1, defaultValue2) {
        super(name);
        this.type = "custom";
        this.label1 = label1;
        this.label2 = label2;
        this.hitAreas = {
            toggle1: { bounds: [0, 0], onDown: this.onToggle1Down.bind(this) },
            toggle2: { bounds: [0, 0], onDown: this.onToggle2Down.bind(this) },
        };
        this._value = { value1: defaultValue1, value2: defaultValue2 };
    }

    set value(v) { this._value = v; }
    get value() { return this._value; }

    draw(ctx, node, w, posY, height) {
        ctx.save();
        const margin = 10, innerMargin = margin * 0.33, lowQuality = isLowQuality(), midY = posY + height * 0.5;
        let posX = margin;
        drawRoundedRectangle(ctx, { pos: [posX, posY], size: [node.size[0] - margin * 2, height] });
        this.hitAreas.toggle1.bounds = drawTogglePart(ctx, { posX, posY, height, value: this.value.value1 });
        posX += this.hitAreas.toggle1.bounds[1] + innerMargin;
        if (lowQuality) { ctx.restore(); return; }
        ctx.fillStyle = LiteGraph.WIDGET_TEXT_COLOR;
        ctx.textAlign = "left";
        ctx.textBaseline = "middle";
        ctx.fillText(this.label1, posX, midY);
        posX += ctx.measureText(this.label1).width + innerMargin * 2;
        this.hitAreas.toggle2.bounds = drawTogglePart(ctx, { posX, posY, height, value: this.value.value2 });
        posX += this.hitAreas.toggle2.bounds[1] + innerMargin;
        ctx.fillText(this.label2, posX, midY);
        ctx.restore();
    }

    serializeValue(node, index) { return this.value; }
    onToggle1Down() { this.value.value1 = !this.value.value1; this.cancelMouseDown(); return true; }
    onToggle2Down() { this.value.value2 = !this.value.value2; this.cancelMouseDown(); return true; }
}

// Main Lora Widget
class LoadLoraMergeWidget extends RgthreeBaseWidget {
    constructor(name) {
        super(name);
        this.type = "custom";
        this.haveMouseMovedStrength = false;
        this.hitAreas = {
            toggle: { bounds: [0, 0], onDown: this.onToggleDown.bind(this) },
            lora: { bounds: [0, 0], onClick: this.onLoraClick.bind(this) },
            strengthDec: { bounds: [0, 0], onClick: (e,p,n) => this.stepStrength(-1) },
            strengthVal: { bounds: [0, 0], onClick: this.onStrengthValUp.bind(this) },
            strengthInc: { bounds: [0, 0], onClick: (e,p,n) => this.stepStrength(1) },
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
        drawRoundedRectangle(ctx, { pos: [posX, posY], size: [node.size[0] - margin * 2, height] });
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

    serializeValue(node, index) { 
        console.log(`[LoadLoraMerge] Serializing widget: ${this.name}, value:`, this.value);
        return { ...this.value }; 
    }
    onToggleDown() { this.value.on = !this.value.on; this.cancelMouseDown(); return true; }
    onLoraClick(event, pos, node) {
        showLoraChooser(rgthree.lastCanvasMouseEvent || event, (value) => {
            if (typeof value === "string") this.value.lora = value;
            node.setDirtyCanvas(true, true);
        }, null, null, node, this);
        this.cancelMouseDown();
    }
    onStrengthDecDown() { this.stepStrength(-1); }
    onStrengthIncDown() { this.stepStrength(1); }
    onStrengthAnyMove(event) { if (event.deltaX) { this.haveMouseMovedStrength = true; this.value.strength = (this.value.strength || 1) + event.deltaX * 0.05; } }
    onStrengthValUp(event) {
        if (this.haveMouseMovedStrength) { this.haveMouseMovedStrength = false; return; }
        app.canvas.prompt("Strength", this.value.strength || 1, (v) => { this.value.strength = Number(v); }, event);
    }
    stepStrength(direction) {
        let step = 0.05;
        this.value.strength = Math.round(((this.value.strength || 1) + step * direction) * 100) / 100;
    }
}

// The main node class
app.registerExtension({
    name: "A_my_nodes.LoadLoraMerge.UI",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "LoadLoraMerge") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function() {
            onNodeCreated?.apply(this, arguments);
            this.loraWidgetsCounter = 0;
            this.serialize_widgets = true; // Important for kwargs
            this.addNonLoraWidgets();
            this.lorasInfoWidget = this.widgets.find(w => w.name === 'loras_info');
        };

        // Main serialization logic
        const onSerialize = nodeType.prototype.onSerialize;
        nodeType.prototype.onSerialize = function(o) {
            onSerialize?.apply(this, arguments);
            const loraWidgets = this.widgets.filter(w => w.name?.startsWith("LORA_"));
            const loraData = loraWidgets.map(w => w.value);
            // Find the hidden loras_info widget and update its value
            const lorasInfoWidget = this.widgets.find(w => w.name === 'loras_info');
            if (lorasInfoWidget) {
                lorasInfoWidget.value = JSON.stringify(loraData);
            }
        };

        // Main configuration/deserialization logic
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function(info) {
            onConfigure?.apply(this, arguments);
            // Clear existing dynamic widgets before loading
            this.widgets = this.widgets.filter(w => !w.name?.startsWith("LORA_"));
            this.loraWidgetsCounter = 0;

            const lorasInfoWidget = this.widgets.find(w => w.name === 'loras_info');
            if (lorasInfoWidget && lorasInfoWidget.value) {
                try {
                    const loraData = JSON.parse(lorasInfoWidget.value);
                    if (Array.isArray(loraData)) {
                        loraData.forEach(loraInfo => {
                            const widget = this.addNewLoraWidget();
                            widget.value = loraInfo;
                        });
                    }
                } catch (e) {
                    console.error("[LoadLoraMerge] Error parsing loras_info:", e);
                }
            }
        };

        // Add the static widgets and the button
        nodeType.prototype.addNonLoraWidgets = function() {
            this.addCustomWidget(new LoadLoraMergeDualToggleWidget("settings", "Low Mem", "Merge", false, true));
            this.addCustomWidget(new RgthreeDividerWidget({ marginTop: 1, marginBottom: 0, thickness: 0 }));
            const addButton = new RgthreeBetterButtonWidget("➕ Add LoRA", (e,p,n) => {
                showLoraChooser(rgthree.lastCanvasMouseEvent || e, (value) => {
                    if (value && value !== "None") this.addNewLoraWidget(value);
                }, null, null, this, addButton);
            });
            addButton.serializeValue = () => undefined;
            this.addCustomWidget(addButton);
        };

        // Add a new LoRA widget
        nodeType.prototype.addNewLoraWidget = function(lora) {
            this.loraWidgetsCounter++;
            const widget = this.addCustomWidget(new LoadLoraMergeWidget("LORA_" + this.loraWidgetsCounter));
            if (lora) widget.setLora(lora);
            const settingsWidget = this.widgets.find(w => w.name === "settings");
            if (settingsWidget) {
                moveArrayItem(this.widgets, widget, this.widgets.indexOf(settingsWidget));
            }
            return widget;
        };

        // Right-click context menu for LoRA widgets
        const getSlotMenuOptions = nodeType.prototype.getSlotMenuOptions;
        nodeType.prototype.getSlotMenuOptions = function(slot) {
            getSlotMenuOptions?.apply(this, arguments);
            if (!slot || !slot.widget?.name?.startsWith("LORA_")) return null;
            
            const widget = slot.widget;
            const index = this.widgets.indexOf(widget);
            const canMoveUp = !!this.widgets[index - 1]?.name?.startsWith("LORA_");
            const canMoveDown = !!this.widgets[index + 1]?.name?.startsWith("LORA_");

            const menuItems = [
                { content: `${widget.value.on ? "⚫" : "🟢"} Toggle ${widget.value.on ? "Off" : "On"}`, callback: () => { widget.value.on = !widget.value.on; } },
                { content: `📋 Copy Path`, disabled: !widget.value.lora || widget.value.lora === "None", callback: () => {
                    navigator.clipboard.writeText(widget.value.lora).then(() => {
                        console.log(`[LoadLoraMerge] Copied model path: ${widget.value.lora}`);
                        showTopNotification(`Copied: ${widget.value.lora}`, 'success');
                    }).catch(err => {
                        console.error('[LoadLoraMerge] Copy failed:', err);
                        // Fallback method
                        try {
                            const textArea = document.createElement('textarea');
                            textArea.value = widget.value.lora;
                            document.body.appendChild(textArea);
                            textArea.select();
                            document.execCommand('copy');
                            document.body.removeChild(textArea);
                            console.log(`[LoadLoraMerge] Copied model path(fallback): ${widget.value.lora}`);
                            showTopNotification(`Copied: ${widget.value.lora}`, 'success');
                        } catch (fallbackErr) {
                            console.error('[LoadLoraMerge] Fallback copy also failed:', fallbackErr);
                            showTopNotification('Copy failed, please copy manually', 'error');
                        }
                    });
                }},
                { content: `⬆️ Move Up`, disabled: !canMoveUp, callback: () => moveArrayItem(this.widgets, widget, index - 1) },
                { content: `⬇️ Move Down`, disabled: !canMoveDown, callback: () => moveArrayItem(this.widgets, widget, index + 1) },
                { content: `🗑️ Delete`, callback: () => removeArrayItem(this.widgets, widget) },
                { content: `🗑️ Clear All`, callback: () => { this.widgets = this.widgets.filter(w => !w.name?.startsWith("LORA_")); } },
            ];

            new LiteGraph.ContextMenu(menuItems, { title: "LoRA Item", event: rgthree.lastCanvasMouseEvent, className: "dark", scale: Math.max(1, app?.canvas?.ds?.scale || 1) });
            return undefined;
        };
        
        // Add getSlotInPosition for right-click menu support
        nodeType.prototype.getSlotInPosition = function(canvasX, canvasY) {
            let lastWidget = null;
            for (const widget of this.widgets) {
                if (!widget.last_y) return;
                if (canvasY > this.pos[1] + widget.last_y) {
                    lastWidget = widget;
                    continue;
                }
                break;
            }
            if (lastWidget && lastWidget.name && lastWidget.name.startsWith("LORA_")) {
                return { widget: lastWidget, output: { type: "LORA WIDGET" } };
            }
            return null;
        };

        // Helper to add custom widgets
        if (!nodeType.prototype.addCustomWidget) {
            nodeType.prototype.addCustomWidget = function(widget) {
                this.widgets = this.widgets || [];
                this.widgets.push(widget);
                return widget;
            };
        }
    },
});