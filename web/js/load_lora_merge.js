import { app } from "../../../scripts/app.js";
import { drawNumberWidgetPart, drawRoundedRectangle, drawTogglePart, fitString, isLowQuality, } from "./utils/utils_canvas.js";
import { RgthreeBaseWidget, RgthreeBetterButtonWidget, RgthreeDividerWidget, } from "./utils/utils_widgets.js";
import { moveArrayItem, removeArrayItem, showTopNotification } from "./utils/shared_utils.js";
import { modal } from "./utils/modal.js";
import { rgthree } from "./core/rgthree.js";
import { rgthreeApi } from "./core/rgthree_api.js";

console.log("Loaded load_lora_merge.js");

import { api } from "../../../scripts/api.js";

const ASSET_SAVE_GROUP_STORAGE_KEY = "load_lora_merge_asset_save_group_name";
const ASSET_SAVE_TARGET_STORAGE_KEY = "load_lora_merge_asset_save_target";
const ASSET_SAVE_MODE_STORAGE_KEY = "load_lora_merge_asset_save_mode";
const ASSET_SAVE_ITEM_STORAGE_KEY = "load_lora_merge_asset_save_item_keyword";

function getStoredValue(storageKey, defaultValue = "") {
    try {
        return localStorage.getItem(storageKey) || defaultValue;
    } catch (error) {
        console.warn("[LoadLoraMerge] Failed to read localStorage:", error);
        return defaultValue;
    }
}

function setStoredValue(storageKey, value) {
    if (value === undefined || value === null || value === "") return;
    try {
        localStorage.setItem(storageKey, value);
    } catch (error) {
        console.warn("[LoadLoraMerge] Failed to write localStorage:", error);
    }
}

function escapeHtml(value) {
    return String(value ?? "")
        .replace(/&/g, "&amp;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#39;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");
}

function getAssetGroupItems(group) {
    return Array.isArray(group?.items) ? group.items : [];
}

function buildAssetItemOptions(items, selectedKeyword = "") {
    if (!items.length) {
        return `<option value="">当前分组暂无可追加模型组</option>`;
    }
    const matchedIndex = items.findIndex(item => (item?.keyword || "") === selectedKeyword);
    const defaultIndex = matchedIndex >= 0 ? matchedIndex : 0;
    return items.map((item, index) => {
        const label = item?.keyword || `未命名模型组 ${index + 1}`;
        return `<option value="${index}"${index === defaultIndex ? " selected" : ""}>${escapeHtml(label)}</option>`;
    }).join("");
}

function mergeUniqueLoras(existingLoras, incomingLoras) {
    const merged = Array.isArray(existingLoras) ? existingLoras.map(item => ({ ...item })) : [];
    const seenLoras = new Set(
        merged
            .map(item => String(item?.lora || "").trim())
            .filter(Boolean)
    );

    for (const item of incomingLoras || []) {
        const loraName = String(item?.lora || "").trim();
        if (!loraName || loraName === "None" || seenLoras.has(loraName)) continue;
        merged.push({ ...item });
        seenLoras.add(loraName);
    }

    return merged;
}

function getCurrentNodeLoraValues(node) {
    return (node.widgets || [])
        .filter(widget => widget.name?.startsWith("LORA_"))
        .map(widget => ({ ...widget.value }))
        .filter(value => value && value.lora && value.lora !== "None");
}

function clearNodeLoraWidgets(node) {
    node.widgets = (node.widgets || []).filter(widget => !widget.name?.startsWith("LORA_"));
    node.loraWidgetsCounter = 0;
}

function addLoraValuesToNode(node, loraValues) {
    for (const loraValue of loraValues || []) {
        const widget = node.addNewLoraWidget();
        widget.value = {
            on: true,
            lora: null,
            strength: 1,
            ...loraValue
        };
    }
    const newSize = node.computeSize?.();
    if (newSize) {
        node.setSize([node.size[0], newSize[1]]);
    }
    node.setDirtyCanvas(true, true);
}

function getImportLorasFromAssetItem(assetItem, sourceType) {
    const highLoras = Array.isArray(assetItem?.high_loras) ? assetItem.high_loras : [];
    const lowLoras = Array.isArray(assetItem?.low_loras) ? assetItem.low_loras : [];

    if (sourceType === "high") return mergeUniqueLoras([], highLoras);
    if (sourceType === "low") return mergeUniqueLoras([], lowLoras);
    return mergeUniqueLoras(highLoras, lowLoras);
}

// Helper function to show the LoRA chooser menu
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
            console.error("[LoadLoraMerge] Failed to fetch LoRAs:", e);
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
        this.node = node; // Save node reference for interaction
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

class LoadLoraMergeButtonRowWidget extends RgthreeBaseWidget {
    constructor(name, buttons) {
        super(name);
        this.type = "custom";
        this.buttons = buttons;
        this.hitAreas = {};
        buttons.forEach((button, index) => {
            this.hitAreas[`btn${index}`] = {
                bounds: [0, 0],
                onDown: (event, pos, node) => {
                    button.onClick?.(event, pos, node || this.node);
                    return true;
                }
            };
        });
    }

    draw(ctx, node, w, posY, height) {
        this.node = node;
        ctx.save();

        const margin = 10;
        const innerMargin = 5;
        const buttonCount = this.buttons.length || 1;
        const buttonWidth = (w - margin * 2 - innerMargin * (buttonCount - 1)) / buttonCount;

        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.font = "11px Arial";

        this.buttons.forEach((button, index) => {
            const btnX = margin + index * (buttonWidth + innerMargin);
            ctx.fillStyle = button.bgColor || "#333";
            ctx.beginPath();
            ctx.roundRect(btnX, posY + 2, buttonWidth, height - 4, 4);
            ctx.fill();

            ctx.strokeStyle = button.borderColor || "#555";
            ctx.stroke();

            ctx.fillStyle = button.textColor || "#fff";
            const label = fitString(ctx, button.label, buttonWidth - 10);
            ctx.fillText(label, btnX + buttonWidth / 2, posY + height / 2);
            this.hitAreas[`btn${index}`].bounds = [btnX, buttonWidth];
        });

        ctx.restore();
    }

    serializeValue() {
        return undefined;
    }
}


// Main Lora Widget
class LoadLoraMergeWidget extends RgthreeBaseWidget {
    constructor(name) {
        super(name);
        this.type = "custom";
        this.haveMouseMovedStrength = false;
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
    onLoraDown(event, pos, node) {
        const targetNode = node || this.node;
        showLoraChooser(event, (value) => {
            if (typeof value === "string") this.value.lora = value;
            targetNode?.setDirtyCanvas(true, true);
        }, null, null, targetNode);
        return true; // Indicate we handled it
    }
    onStrengthDecDown() { this.stepStrength(-1); return true;}
    onStrengthIncDown() { this.stepStrength(1); return true;}
    onStrengthAnyMove(event) { 
        if (event.deltaX) { 
            this.wasDragging = true;
            this.value.strength = (this.value.strength || 1) + event.deltaX * 0.05; 
            this.value.strength = Math.round(this.value.strength * 100) / 100; // Round to avoid float issues
        } 
    }
    onStrengthValDown(event) {
        const now = Date.now();
        // Double click detection: < 300ms and the previous interaction was not a drag
        if (this.lastClickTime && (now - this.lastClickTime < 300) && !this.wasDragging) {
            // Double click - open prompt
            const e = event || rgthree.lastCanvasMouseEvent;
            app.canvas.prompt("Strength", this.value.strength || 1, (v) => { this.value.strength = Number(v); }, e);
            this.lastClickTime = 0; // Reset
            return true;
        }
        
        // Single click / Start of potential drag
        this.lastClickTime = now;
        this.wasDragging = false; // Reset drag flag for this new interaction
        return true;
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
            const handleAddLora = (e) => {
                showLoraChooser(rgthree.lastCanvasMouseEvent || e, (value) => {
                    if (typeof value === "string" && value && value !== "None") this.addNewLoraWidget(value);
                }, null, null, this, this);
            };

            const handleImportAsset = async () => {
                if (!window.AssetManager) {
                    if (window.AMDialog) {
                        window.AMDialog.alert("资产管理系统未就绪！");
                    } else {
                        alert("资产管理系统未就绪！");
                    }
                    return;
                }

                window.AssetManager.showDrawer(this, "models", async (selectedModels) => {
                    const assetItem = Array.isArray(selectedModels) ? selectedModels[0] : null;
                    if (!assetItem) return;

                    const hasHigh = Array.isArray(assetItem.high_loras) && assetItem.high_loras.length > 0;
                    const hasLow = Array.isArray(assetItem.low_loras) && assetItem.low_loras.length > 0;

                    if (!hasHigh && !hasLow) {
                        showTopNotification("所选模型组没有可导入的 LoRA。", "warning");
                        return;
                    }

                    let importSource = "all";
                    if (hasHigh && hasLow) {
                        importSource = await new Promise(resolve => {
                            modal.show({
                                title: "选择导入来源",
                                content: "当前模型组同时包含 High 和 Low LoRA，请选择导入哪一侧：",
                                width: "400px",
                                buttons: [
                                    { text: "导入 High", type: "primary", onClick: () => { resolve("high"); modal.close(); } },
                                    { text: "导入 Low", type: "secondary", onClick: () => { resolve("low"); modal.close(); } },
                                    { text: "全部导入", onClick: () => { resolve("all"); modal.close(); } },
                                    { text: "取消", onClick: () => { resolve("cancel"); modal.close(); } }
                                ]
                            });
                        });
                        if (importSource === "cancel") return;
                    } else {
                        importSource = hasHigh ? "high" : "low";
                    }

                    const importedLoras = getImportLorasFromAssetItem(assetItem, importSource);
                    if (importedLoras.length === 0) {
                        showTopNotification("所选来源中没有可导入的 LoRA。", "warning");
                        return;
                    }

                    const currentLoras = getCurrentNodeLoraValues(this);
                    let importMode = "append";

                    if (currentLoras.length > 0) {
                        importMode = await new Promise(resolve => {
                            modal.show({
                                title: "检测到已有配置",
                                content: "当前节点已有 LoRA 配置，请选择导入模式：<br><br><b>追加</b>：保留现有 LoRA，并自动去重后追加新 LoRA。<br><b>覆盖</b>：清空当前节点 LoRA，然后导入所选模板。",
                                width: "400px",
                                buttons: [
                                    { text: "追加 (保留现有)", type: "primary", onClick: () => { resolve("append"); modal.close(); } },
                                    { text: "覆盖 (清空现有)", type: "secondary", onClick: () => { resolve("override"); modal.close(); } },
                                    { text: "取消", onClick: () => { resolve("cancel"); modal.close(); } }
                                ]
                            });
                        });
                        if (importMode === "cancel") return;
                    }

                    if (importMode === "override") {
                        clearNodeLoraWidgets(this);
                        addLoraValuesToNode(this, importedLoras);
                        showTopNotification(`已覆盖导入 ${importedLoras.length} 个 LoRA`, "success");
                        return;
                    }

                    const mergedLoras = mergeUniqueLoras(currentLoras, importedLoras);
                    const addedLoras = mergedLoras.slice(currentLoras.length);
                    if (addedLoras.length === 0) {
                        showTopNotification("没有新的 LoRA 可导入，已自动去重。", "warning");
                        return;
                    }

                    addLoraValuesToNode(this, addedLoras);
                    showTopNotification(`已追加导入 ${addedLoras.length} 个 LoRA`, "success");
                });
            };

            const handleSaveAsset = async () => {
                const loraWidgets = this.widgets.filter(w => w.name?.startsWith("LORA_"));
                const loraValues = loraWidgets
                    .map(w => ({ ...w.value }))
                    .filter(v => v && v.lora && v.lora !== "None");

                if (loraValues.length === 0) {
                    showTopNotification("请先添加至少一个有效的 LoRA。", "warning");
                    return;
                }

                try {
                    const res = await api.fetchApi("/a_my_nodes/assets/models");
                    const modelsData = await res.json();
                    const groups = modelsData.groups || [];

                    if (groups.length === 0) {
                        alert("资产库中没有模型分组，请先在资产库中创建分组！");
                        return;
                    }

                    const savedGroupName = getStoredValue(ASSET_SAVE_GROUP_STORAGE_KEY);
                    const savedTarget = getStoredValue(ASSET_SAVE_TARGET_STORAGE_KEY, "high");
                    const savedMode = getStoredValue(ASSET_SAVE_MODE_STORAGE_KEY, "create");
                    const savedItemKeyword = getStoredValue(ASSET_SAVE_ITEM_STORAGE_KEY);
                    const savedGroupIdx = groups.findIndex(g => g?.name === savedGroupName);
                    const defaultGroupIdx = savedGroupIdx >= 0 ? savedGroupIdx : 0;
                    const groupOptions = groups.map((g, i) => (
                        `<option value="${i}"${i === defaultGroupIdx ? " selected" : ""}>${escapeHtml(g.name)}</option>`
                    )).join("");
                    const defaultGroupItems = getAssetGroupItems(groups[defaultGroupIdx]);
                    const initialMode = defaultGroupItems.length > 0
                        ? (savedMode === "append" ? "append" : "create")
                        : "create";
                    const existingItemOptions = buildAssetItemOptions(defaultGroupItems, savedItemKeyword);

                    const defaultTitle = loraValues
                        .map(v => String(v.lora || "").split(/[\\/]/).pop()?.replace(/\.[^.]+$/, "") || "")
                        .filter(Boolean)
                        .slice(0, 3)
                        .join(" + ") || "未命名配置";

                    const content = `
                        <div style="display:flex; flex-direction:column; gap:10px; color:white;">
                            <label>保存到分组:</label>
                            <select id="am-save-group" style="background:#333; color:white; padding:5px; border:1px solid #555; border-radius:4px;">
                                ${groupOptions}
                            </select>
                            <label>保存到流类型:</label>
                            <select id="am-save-target" style="background:#333; color:white; padding:5px; border:1px solid #555; border-radius:4px;">
                                <option value="high"${savedTarget === "high" ? " selected" : ""}>High LoRAs</option>
                                <option value="low"${savedTarget === "low" ? " selected" : ""}>Low LoRAs</option>
                            </select>
                            <label>保存方式:</label>
                            <select id="am-save-mode" style="background:#333; color:white; padding:5px; border:1px solid #555; border-radius:4px;">
                                <option value="append"${initialMode === "append" ? " selected" : ""}>追加到已有模型组</option>
                                <option value="create"${initialMode === "create" ? " selected" : ""}>新建模型组</option>
                            </select>
                            <div id="am-existing-item-wrap" style="display:${initialMode === "append" ? "flex" : "none"}; flex-direction:column; gap:6px;">
                                <label>已有模型组名:</label>
                                <select id="am-existing-item" style="background:#333; color:white; padding:5px; border:1px solid #555; border-radius:4px;">
                                    ${existingItemOptions}
                                </select>
                                <div id="am-existing-item-tip" style="font-size:12px; color:#aaa;">
                                    ${defaultGroupItems.length > 0 ? `当前分组共有 ${defaultGroupItems.length} 个模型组可追加` : "当前分组暂无可追加模型组"}
                                </div>
                            </div>
                            <div id="am-new-item-wrap" style="display:${initialMode === "create" ? "flex" : "none"}; flex-direction:column; gap:6px;">
                                <label>模型组名:</label>
                                <input type="text" id="am-save-title" value="${escapeHtml(defaultTitle)}" style="background:#333; color:white; padding:5px; border:1px solid #555; border-radius:4px;">
                            </div>
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
                                    const groupIdx = parseInt(document.getElementById("am-save-group").value, 10);
                                    const targetType = document.getElementById("am-save-target").value === "low" ? "low" : "high";
                                    const saveMode = document.getElementById("am-save-mode").value === "append" ? "append" : "create";
                                    const titleInput = document.getElementById("am-save-title");
                                    const existingItemSelect = document.getElementById("am-existing-item");
                                    const selectedGroup = groups[groupIdx];

                                    if (!selectedGroup) {
                                        showTopNotification("所选分组无效，请重新选择。", "error");
                                        return;
                                    }

                                    selectedGroup.items = getAssetGroupItems(selectedGroup);
                                    const targetField = targetType === "low" ? "low_loras" : "high_loras";
                                    let resultMessage = "成功保存到资产库！";
                                    let storedItemKeyword = "";

                                    if (saveMode === "append") {
                                        const itemIdx = parseInt(existingItemSelect?.value ?? "", 10);
                                        const existingItem = selectedGroup.items[itemIdx];

                                        if (!existingItem) {
                                            showTopNotification("当前分组没有可追加的模型组，请切换为新建模型组。", "warning");
                                            return;
                                        }

                                        const existingTargetLoras = Array.isArray(existingItem[targetField]) ? existingItem[targetField] : [];
                                        const mergedTargetLoras = mergeUniqueLoras(existingTargetLoras, loraValues);
                                        const addedCount = mergedTargetLoras.length - existingTargetLoras.length;

                                        existingItem.high_loras = Array.isArray(existingItem.high_loras) ? existingItem.high_loras : [];
                                        existingItem.low_loras = Array.isArray(existingItem.low_loras) ? existingItem.low_loras : [];
                                        existingItem[targetField] = mergedTargetLoras;

                                        storedItemKeyword = existingItem.keyword || "";
                                        resultMessage = addedCount > 0
                                            ? `成功追加 ${addedCount} 个 LoRA 到已有模型组！`
                                            : "目标模型组中没有新增 LoRA，已自动去重。";
                                    } else {
                                        const title = titleInput?.value.trim() || defaultTitle;
                                        let targetItem = selectedGroup.items.find(item => (item?.keyword || "") === title);

                                        if (!targetItem) {
                                            targetItem = {
                                                id: Date.now().toString() + Math.random().toString().slice(2, 6),
                                                keyword: title,
                                                check_mode: "contains",
                                                high_loras: [],
                                                low_loras: [],
                                                preview_image: ""
                                            };
                                            selectedGroup.items.push(targetItem);
                                            resultMessage = "成功新建模型组并保存到资产库！";
                                        } else {
                                            resultMessage = "检测到同名模型组，已自动合并并去重。";
                                        }

                                        targetItem.high_loras = Array.isArray(targetItem.high_loras) ? targetItem.high_loras : [];
                                        targetItem.low_loras = Array.isArray(targetItem.low_loras) ? targetItem.low_loras : [];
                                        targetItem[targetField] = mergeUniqueLoras(targetItem[targetField], loraValues);
                                        storedItemKeyword = targetItem.keyword || title;
                                    }

                                    setStoredValue(ASSET_SAVE_GROUP_STORAGE_KEY, selectedGroup.name);
                                    setStoredValue(ASSET_SAVE_TARGET_STORAGE_KEY, targetType);
                                    setStoredValue(ASSET_SAVE_MODE_STORAGE_KEY, saveMode);
                                    setStoredValue(ASSET_SAVE_ITEM_STORAGE_KEY, storedItemKeyword);

                                    await api.fetchApi("/a_my_nodes/assets/models", {
                                        method: "POST",
                                        body: JSON.stringify(modelsData),
                                        headers: { "Content-Type": "application/json" }
                                    });

                                    if (window.AssetManager) {
                                        window.AssetManager.loadData();
                                    }

                                    showTopNotification(resultMessage, "success");
                                    modal.close();
                                }
                            },
                            { text: "取消", onClick: () => modal.close() }
                        ]
                    });

                    const groupSelect = document.getElementById("am-save-group");
                    const modeSelect = document.getElementById("am-save-mode");
                    const existingItemWrap = document.getElementById("am-existing-item-wrap");
                    const existingItemSelect = document.getElementById("am-existing-item");
                    const existingItemTip = document.getElementById("am-existing-item-tip");
                    const newItemWrap = document.getElementById("am-new-item-wrap");

                    const refreshExistingItems = () => {
                        const groupIdx = parseInt(groupSelect.value, 10);
                        const selectedGroup = groups[groupIdx];
                        const items = getAssetGroupItems(selectedGroup);
                        existingItemSelect.innerHTML = buildAssetItemOptions(items, savedItemKeyword);
                        existingItemSelect.disabled = items.length === 0;
                        existingItemTip.textContent = items.length > 0
                            ? `当前分组共有 ${items.length} 个模型组可追加`
                            : "当前分组暂无可追加模型组";
                        if (items.length === 0 && modeSelect.value === "append") {
                            modeSelect.value = "create";
                        }
                        updateModeVisibility();
                    };

                    const updateModeVisibility = () => {
                        const isAppendMode = modeSelect.value === "append";
                        existingItemWrap.style.display = isAppendMode ? "flex" : "none";
                        newItemWrap.style.display = isAppendMode ? "none" : "flex";
                    };

                    groupSelect?.addEventListener("change", refreshExistingItems);
                    modeSelect?.addEventListener("change", updateModeVisibility);
                    refreshExistingItems();
                } catch (error) {
                    console.error("[LoadLoraMerge] Failed to save asset:", error);
                    alert("无法连接到资产库 API！");
                }
            };

            this.addCustomWidget(new LoadLoraMergeButtonRowWidget("lora_actions", [
                { label: "➕ Add", onClick: (e) => handleAddLora(e) },
                { label: "✨ 导入", onClick: () => handleImportAsset() },
                { label: "💾 保存", onClick: () => handleSaveAsset() }
            ]));
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
                {
                    content: `${widget.value.on ? "⚫" : "🟢"} Toggle ${widget.value.on ? "Off" : "On"}`,
                    callback: () => { widget.value.on = !widget.value.on; },
                },
                {
                    content: `📋 Copy Path`,
                    disabled: !widget.value.lora || widget.value.lora === "None",
                    callback: () => {
                        if (widget.value.lora && widget.value.lora !== "None") {
                            navigator.clipboard.writeText(widget.value.lora).then(() => {
                                showTopNotification(`Copied: ${widget.value.lora}`, 'success');
                            }).catch(err => {
                                console.error('Copy failed:', err);
                            });
                        }
                    },
                },
                {
                    content: `⬆️ Move Up`,
                    disabled: !canMoveUp,
                    callback: () => { moveArrayItem(this.widgets, widget, index - 1); },
                },
                {
                    content: `⬇️ Move Down`,
                    disabled: !canMoveDown,
                    callback: () => { moveArrayItem(this.widgets, widget, index + 1); },
                },
                {
                    content: `🗑️ Delete`,
                    callback: () => { removeArrayItem(this.widgets, widget); },
                },
                {
                    content: `🗑️ Clear All LoRAs`,
                    callback: () => {
                        this.widgets = this.widgets.filter(widget => !widget.name || !widget.name.startsWith("LORA_"));
                        this.setDirtyCanvas(true, true);
                    },
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
