import { app } from "../../../scripts/app.js";
import { drawNumberWidgetPart, drawRoundedRectangle, drawTogglePart, fitString, isLowQuality, } from "./utils_canvas.js";
import { RgthreeBaseWidget, RgthreeBetterButtonWidget, RgthreeDividerWidget, } from "./utils_widgets.js";
import { rgthreeApi } from "./rgthree_api.js";
import { moveArrayItem, removeArrayItem, showTopNotification } from "./shared_utils.js";
import { rgthree } from "./rgthree.js";

console.log("Loaded load_lora_merge.js");

import { api } from "../../../scripts/api.js";

// Helper function to show the LoRA chooser menu
async function showLoraChooser(event, callback, parentMenu, loras, buttonNode, buttonWidget) {
    const canvas = app.canvas;
    if (!loras) {
        try {
            const loraFiles = await api.getModels('loras');
            loras = ["None", ...loraFiles.map((l) => l.name)];
        } catch (e) {
            console.error("[LoadLoraMerge] Failed to fetch LoRAs:", e);
            loras = ["None"];
        }
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
        this._value1 = defaultValue1;
        this._value2 = defaultValue2;
    }

    set value(v) {
        if (typeof v === 'object' && v !== null) {
            this._value1 = v.value1 !== undefined ? v.value1 : this._value1;
            this._value2 = v.value2 !== undefined ? v.value2 : this._value2;
        }
    }

    get value() {
        return {
            value1: this._value1,
            value2: this._value2
        };
    }

    draw(ctx, node, w, posY, height) {
        ctx.save();
        const margin = 10, innerMargin = margin * 0.33, lowQuality = isLowQuality(), midY = posY + height * 0.5;
        let posX = margin;
        drawRoundedRectangle(ctx, { pos: [posX, posY], size: [node.size[0] - margin * 2, height] });
        this.hitAreas.toggle1.bounds = drawTogglePart(ctx, { posX, posY, height, value: this._value1 });
        posX += this.hitAreas.toggle1.bounds[1] + innerMargin;
        if (lowQuality) { ctx.restore(); return; }
        ctx.fillStyle = LiteGraph.WIDGET_TEXT_COLOR;
        ctx.textAlign = "left";
        ctx.textBaseline = "middle";
        ctx.fillText(this.label1, posX, midY);
        posX += ctx.measureText(this.label1).width + innerMargin * 2;
        this.hitAreas.toggle2.bounds = drawTogglePart(ctx, { posX, posY, height, value: this._value2 });
        posX += this.hitAreas.toggle2.bounds[1] + innerMargin;
        ctx.fillText(this.label2, posX, midY);
        ctx.restore();
    }

    serializeValue(node, index) { return this.value; }
    onToggle1Down() { this._value1 = !this._value1; this.cancelMouseDown(); return true; }
    onToggle2Down() { this._value2 = !this._value2; this.cancelMouseDown(); return true; }
}

// Triple Toggle Widget for settings
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
            if (this.lorasInfoWidget) {
                // 重写computeSize以隐藏小部件
                this.lorasInfoWidget.computeSize = () => [0, -4];
            }
        };

        // Main serialization logic
        const onSerialize = nodeType.prototype.onSerialize;
        nodeType.prototype.onSerialize = function(o) {
            const loraWidgets = this.widgets.filter(w => w.name?.startsWith("LORA_"));
            const loraData = loraWidgets.map(w => w.value);
            
            const settingsWidget = this.widgets.find(w => w.name === 'settings');
            const settingsData = settingsWidget ? settingsWidget.value : {};

            const combinedData = {
                loras: loraData,
                settings: settingsData,
            };

            // Find the hidden loras_info widget and update its value before serializing
            const lorasInfoWidget = this.widgets.find(w => w.name === 'loras_info');
            if (lorasInfoWidget) {
                lorasInfoWidget.value = JSON.stringify(combinedData);
            }
            // Now, call the original serialization which will pick up all widget values.
            onSerialize?.apply(this, arguments);
        };

        // Main configuration/deserialization logic
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function(info) {
            onConfigure?.apply(this, arguments);
            // Clear existing dynamic widgets before loading
            this.widgets = this.widgets.filter(w => !w.name?.startsWith("LORA_"));
            this.loraWidgetsCounter = 0;

            const lorasInfoWidget = this.widgets.find(w => w.name === 'loras_info');
            if (lorasInfoWidget && lorasInfoWidget.value && lorasInfoWidget.value !== '[]') {
                try {
                    const combinedData = JSON.parse(lorasInfoWidget.value);
                    
                    // Handle legacy format (just an array of loras)
                    if (Array.isArray(combinedData)) {
                        combinedData.forEach(loraInfo => {
                            const widget = this.addNewLoraWidget();
                            widget.value = loraInfo;
                        });
                    } else if (typeof combinedData === 'object' && combinedData !== null) {
                        // New format
                        if (Array.isArray(combinedData.loras)) {
                            combinedData.loras.forEach(loraInfo => {
                                const widget = this.addNewLoraWidget();
                                widget.value = loraInfo;
                            });
                        }
                        if (combinedData.settings) {
                            const settingsWidget = this.widgets.find(w => w.name === 'settings');
                            if (settingsWidget) {
                                const settings = combinedData.settings;
                                // Handle old data where there were only two toggles (value1: low_mem, value2: merge)
                                // and map to new three-toggle widget (value1: toggle_all, value2: low_mem, value3: merge)
                                if (settings.value3 === undefined) {
                                    settingsWidget.value = {
                                        value1: false, // toggle all - default off for old saves
                                        value2: settings.value1 !== undefined ? settings.value1 : false, // low mem
                                        value3: settings.value2 !== undefined ? settings.value2 : true, // merge
                                    };
                                } else {
                                    settingsWidget.value = settings;
                                }
                            }
                        }
                    }
                } catch (e) {
                    console.error("[LoadLoraMerge] Error parsing loras_info:", e);
                }
            }
        };

        // Add the static widgets and the button
        nodeType.prototype.addNonLoraWidgets = function() {
            const settingsWidget = this.addCustomWidget(new LoadLoraMergeTripleToggleWidget("settings", "Toggle All", "Low Mem", "Merge", false, false, true));
            settingsWidget.properties = {
                onToggle1: (value) => {
                    const loraWidgets = this.widgets.filter(w => w.name?.startsWith("LORA_"));
                    for (const widget of loraWidgets) {
                        widget.value.on = value;
                    }
                }
            };
            this.addCustomWidget(new RgthreeDividerWidget({ marginTop: 1, marginBottom: 0, thickness: 0 }));
            const addButton = new RgthreeBetterButtonWidget("➕ Add LoRA", (e,p,n) => {
                showLoraChooser(rgthree.lastCanvasMouseEvent || e, (value) => {
                    if (typeof value === "string" && value && value !== "None") this.addNewLoraWidget(value);
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

            // 获取所有LoRA widgets的索引范围
            const loraWidgets = this.widgets.filter(w => w.name?.startsWith("LORA_"));
            const firstLoraIndex = this.widgets.findIndex(w => w.name?.startsWith("LORA_"));
            const lastLoraIndex = this.widgets.map((w, i) => w.name?.startsWith("LORA_") ? i : -1).filter(i => i !== -1).pop();
            const isFirst = index === firstLoraIndex;
            const isLast = index === lastLoraIndex;

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
                { content: `⏫ Move to Top`, disabled: isFirst || loraWidgets.length <= 1, callback: () => {
                    // 移动到第一个LoRA widget的位置
                    if (firstLoraIndex !== -1 && index !== firstLoraIndex) {
                        moveArrayItem(this.widgets, widget, firstLoraIndex);
                        showTopNotification('Moved to top', 'success');
                    }
                }},
                { content: `⏬ Move to Bottom`, disabled: isLast || loraWidgets.length <= 1, callback: () => {
                    // 移动到最后一个LoRA widget的位置
                    if (lastLoraIndex !== -1 && index !== lastLoraIndex) {
                        moveArrayItem(this.widgets, widget, lastLoraIndex);
                        showTopNotification('Moved to bottom', 'success');
                    }
                }},
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