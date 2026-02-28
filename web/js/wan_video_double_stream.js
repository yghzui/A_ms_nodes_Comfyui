import { app } from "../../../scripts/app.js";
import { drawNumberWidgetPart, drawRoundedRectangle, drawTogglePart, fitString, isLowQuality } from "./utils_canvas.js";
import { RgthreeBaseWidget, RgthreeBetterButtonWidget } from "./utils_widgets.js";
import { moveArrayItem, showTopNotification } from "./shared_utils.js";
import { rgthree } from "./rgthree.js";
import { api } from "../../../scripts/api.js";
import { rgthreeApi } from "./rgthree_api.js";

// --- Helper Functions ---

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
                toggleWidgets(["settings_high", "➕ Add High LoRA"], collapsed);
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

            // 4. Setup Auto Enable Logic Section (Label only, inputs are standard)
            this.labelAuto = this.addCustomWidget(new LabelWidget("label_auto", "⚙️ Auto Enable Logic", (collapsed) => {
                toggleWidgets(["dict_input", "key_to_check", "check_mode"], collapsed);
                const newSize = this.computeSize();
                this.setSize([this.size[0], newSize[1]]);
                this.setDirtyCanvas(true, true);
            }));

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
        
        // Context Menu for removal
        const getSlotMenuOptions = nodeType.prototype.getSlotMenuOptions;
        nodeType.prototype.getSlotMenuOptions = function(slot) {
            getSlotMenuOptions?.apply(this, arguments);
            
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
                event: rgthree.lastCanvasMouseEvent, 
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
            menu.push({
                content: useOfficial ? "Use Custom Lora List (Default)" : "Use Official Lora List",
                callback: () => {
                    if (!this.properties) this.properties = {};
                    this.properties['useOfficialLoraList'] = !useOfficial;
                }
            });
            
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
