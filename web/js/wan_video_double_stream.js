import { app } from "../../../scripts/app.js";
import { drawRoundedRectangle, drawTogglePart, isLowQuality } from "./utils/utils_canvas.js";
import { RgthreeBaseWidget, RgthreeBetterButtonWidget, addIconButtonRow } from "./utils/utils_widgets.js";
import { modal } from "./utils/modal.js";
import { moveArrayItem, showTopNotification } from "./utils/shared_utils.js";
import { rgthree } from "./core/rgthree.js";
import { api } from "../../../scripts/api.js";
import { LabelWidget, LoraWidget, DoubleButtonWidget, showLoraChooser, StatusLabelWidget, showStatusLabelSettingsModal, getLoraRowSlot, getLoraRowMenuItems } from "./lora/lora_common.js";

// --- Helper Functions ---

const ASSET_SAVE_GROUP_STORAGE_KEY = "wan_video_double_stream_asset_save_group_name";

function getSavedAssetGroupName() {
    try {
        return localStorage.getItem(ASSET_SAVE_GROUP_STORAGE_KEY) || "";
    } catch (error) {
        console.warn("Failed to read saved asset group:", error);
        return "";
    }
}

function saveAssetGroupName(groupName) {
    if (!groupName) return;
    try {
        localStorage.setItem(ASSET_SAVE_GROUP_STORAGE_KEY, groupName);
    } catch (error) {
        console.warn("Failed to save asset group:", error);
    }
}

// LoRA 行右键菜单配置（保持该节点原有英文文案与双流分组逻辑）
const DOUBLE_STREAM_ROW_MENU = {
    isRow: (widget) => !!widget.group,
    rowList: (node, widget) => (widget.group === "High" ? node.highLoraWidgets : node.lowLoraWidgets) || [],
    labels: {
        toggle: (widget) => `${widget.value.on ? "⚫" : "🟢"} Toggle ${widget.value.on ? "Off" : "On"}`,
        copyPath: "📋 Copy Path",
        moveUp: "⬆️ Move Up",
        moveDown: "⬇️ Move Down",
        delete: "🗑️ Delete",
    },
    moveRow: (node, widget, offset, list) => {
        const index = list.indexOf(widget);
        const target = index + offset;
        if (index < 0 || target < 0 || target >= list.length) return;
        [list[index], list[target]] = [list[target], list[index]];
        node.reorderWidgets();
        node.setDirtyCanvas(true, true);
    },
    deleteRow: (node, widget, list) => {
        const index = list.indexOf(widget);
        if (index >= 0) list.splice(index, 1);
        const wIdx = node.widgets.indexOf(widget);
        if (wIdx > -1) node.widgets.splice(wIdx, 1);
        node.reorderWidgets();
        node.setDirtyCanvas(true, true);
    },
    afterChange: (node) => node.setDirtyCanvas(true, true),
    extraItems: (node, widget, index, list) => [
        {
            content: "⏫ Move to Top",
            disabled: index <= 0 || list.length <= 1,
            callback: () => {
                list.splice(index, 1);
                list.unshift(widget);
                node.reorderWidgets();
                node.setDirtyCanvas(true, true);
            },
        },
        {
            content: "⏬ Move to Bottom",
            disabled: index >= list.length - 1 || list.length <= 1,
            callback: () => {
                list.splice(index, 1);
                list.push(widget);
                node.reorderWidgets();
                node.setDirtyCanvas(true, true);
            },
        },
        {
            content: `🗑️ Clear All ${widget.group} Stream`,
            callback: () => {
                list.forEach(w => {
                    const wIdx = node.widgets.indexOf(w);
                    if (wIdx > -1) node.widgets.splice(wIdx, 1);
                });
                list.length = 0;
                node.reorderWidgets();
                node.setDirtyCanvas(true, true);
            },
        },
        {
            content: node.properties?.useOfficialLoraList ? "Use Custom Lora List" : "Use Official Lora List",
            callback: () => {
                if (!node.properties) node.properties = {};
                node.properties.useOfficialLoraList = !node.properties.useOfficialLoraList;
            },
        },
    ],
};

function normalizeAssetStreamSettings(settings) {
    if (!settings || typeof settings !== "object") {
        return { value1: false, value2: false, value3: false };
    }
    if (settings.value3 === undefined && (settings.value1 !== undefined || settings.value2 !== undefined)) {
        return {
            value1: false,
            value2: settings.value1 === true,
            value3: settings.value2 === true,
        };
    }
    return {
        value1: settings.value1 === true,
        value2: settings.value2 === true,
        value3: settings.value3 === true,
    };
}

// --- Custom Widgets ---

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
            this.finalStatusLabel = new StatusLabelWidget("final_status_label");
            this.assetStreamType = "double";

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
            const settingsHigh = this.addCustomWidget(new LoadLoraMergeTripleToggleWidget("settings_high", "Toggle All", "Low Mem", "Merge", false, false, false));
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
            this.btnImportAsset = new DoubleButtonWidget(
                "asset_manager_btns", 
                "✨ 插入模板", 
                "💾 保存到资产库", 
                (e,p,n) => {
                    if (window.AssetManager) {
                        window.AssetManager.showDrawer(this, 'models', async (selectedModels) => {
                            // 判断当前节点是否已经有模型配置
                            const hasExistingConfig = this.highLoraWidgets.length > 0 || this.lowLoraWidgets.length > 0;
                            let mode = "append";
                            
                            if (hasExistingConfig) {
                                // 弹窗询问覆盖还是追加
                                const userChoice = await new Promise(resolve => {
                                    modal.show({
                                        title: "检测到已有配置",
                                        content: "当前节点已有模型配置，请选择插入模式：<br><br><b>追加</b>：保留现有模型，在末尾添加新模型。<br><b>覆盖</b>：清空当前节点的所有模型配置，然后应用新模板。",
                                        width: "400px",
                                        buttons: [
                                            { text: "覆盖 (清空现有)", type: "secondary", onClick: () => { resolve("override"); modal.close(); } },
                                            { text: "追加 (保留现有)", type: "primary", onClick: () => { resolve("append"); modal.close(); } },
                                            { text: "取消", onClick: () => { resolve("cancel"); modal.close(); } }
                                        ]
                                    });
                                });
                                
                                if (userChoice === "cancel") return;
                                mode = userChoice;
                            }

                            if (mode === "override") {
                                // 清空现有的模型配置
                                this.highLoraWidgets.forEach(w => {
                                    const idx = this.widgets.indexOf(w);
                                    if (idx > -1) this.widgets.splice(idx, 1);
                                });
                                this.lowLoraWidgets.forEach(w => {
                                    const idx = this.widgets.indexOf(w);
                                    if (idx > -1) this.widgets.splice(idx, 1);
                                });
                                this.highLoraWidgets = [];
                                this.lowLoraWidgets = [];
                                this.loraWidgetsCounter = 0;
                            }

                            const shouldApplyAssetSettings = mode === "override" || !hasExistingConfig;

                            selectedModels.forEach(m => {
                                // 1. 处理 key_to_check
                                if (m.keyword) {
                                    const wKey = this.widgets.find(w => w.name === "key_to_check");
                                    if (wKey) {
                                        if (mode === "override") {
                                            wKey.value = m.keyword;
                                        } else {
                                            const currentKeys = wKey.value.split(';').map(k => k.trim()).filter(k => k);
                                            if (!currentKeys.includes(m.keyword)) {
                                                currentKeys.push(m.keyword);
                                                wKey.value = currentKeys.join(';');
                                            }
                                        }
                                    }
                                }

                                if (shouldApplyAssetSettings) {
                                    const settingsHighWidget = this.widgets.find(w => w.name === "settings_high");
                                    const settingsLowWidget = this.widgets.find(w => w.name === "settings_low");
                                    if (settingsHighWidget) {
                                        settingsHighWidget.value = normalizeAssetStreamSettings(m.high_settings);
                                    }
                                    if (settingsLowWidget) {
                                        settingsLowWidget.value = normalizeAssetStreamSettings(m.low_settings);
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
                            const newSize = this.computeSize();
                            this.setSize([this.size[0], newSize[1]]);
                        });
                    } else {
                        showTopNotification("资产管理系统未就绪！", "error");
                    }
                },
                async (e,p,n) => {
                    try {
                        const res = await api.fetchApi("/a_my_nodes/assets/models");
                        const modelsData = await res.json();
                        const groups = modelsData.groups || [];
                        
                        if (groups.length === 0) {
                            showTopNotification("资产库中没有模型分组，请先在资产库中创建分组！", "warning");
                            return;
                        }
                        
                        const savedGroupName = getSavedAssetGroupName();
                        const savedGroupIdx = groups.findIndex(g => g?.name === savedGroupName);
                        const defaultGroupIdx = savedGroupIdx >= 0 ? savedGroupIdx : 0;
                        const groupOptions = groups.map((g, i) => (
                            `<option value="${i}"${i === defaultGroupIdx ? " selected" : ""}>${g.name}</option>`
                        )).join("");
                        
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
                                        const selectedGroup = groups[groupIdx];
                                        
                                        if (!selectedGroup) {
                                            showTopNotification("所选分组无效，请重新选择。", "error");
                                            return;
                                        }
                                        
                                        const newItem = {
                                            id: Date.now().toString() + Math.random().toString().slice(2, 6),
                                            keyword: title,
                                            check_mode: this.widgets.find(w => w.name === "check_mode")?.value || "contains",
                                            high_settings: normalizeAssetStreamSettings(this.widgets.find(w => w.name === "settings_high")?.value),
                                            low_settings: normalizeAssetStreamSettings(this.widgets.find(w => w.name === "settings_low")?.value),
                                            high_loras: this.highLoraWidgets.map(w => w.value).filter(v => v.lora && v.lora !== "None"),
                                            low_loras: this.lowLoraWidgets.map(w => w.value).filter(v => v.lora && v.lora !== "None"),
                                            preview_image: ""
                                        };
                                        
                                        selectedGroup.items = Array.isArray(selectedGroup.items) ? selectedGroup.items : [];
                                        selectedGroup.items.push(newItem);
                                        saveAssetGroupName(selectedGroup.name);
                                        
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
                        showTopNotification("无法连接到资产库 API！", "error");
                    }
                }
            );

            // 3. Setup Low Stream Section
            this.labelLow = this.addCustomWidget(new LabelWidget("label_low", "🔽 Low Stream", (collapsed) => {
                toggleWidgets(["settings_low", "➕ Add Low LoRA"], collapsed);
                this.lowLoraWidgets.forEach(w => w.hidden = collapsed);
                const newSize = this.computeSize();
                this.setSize([this.size[0], newSize[1]]);
                this.setDirtyCanvas(true, true);
            }));

            // Add settings for Low Stream
            const settingsLow = this.addCustomWidget(new LoadLoraMergeTripleToggleWidget("settings_low", "Toggle All", "Low Mem", "Merge", false, false, false));
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
            this.btnImportExport = new DoubleButtonWidget(
                "import_export_btns", 
                "📥 Import LoRAs", 
                "📤 Export LoRAs", 
                (e,p,n) => handleImport(this),
                (e,p,n) => handleExport(this)
            );
            this.actionBar = addIconButtonRow(this, "lora_action_icons", [
                { icon: "✨", title: "插入模板", onClick: this.btnImportAsset.callback1 },
                { icon: "💾", title: "保存到资产库", onClick: this.btnImportAsset.callback2 },
                { icon: "📥", title: "Import LoRAs", onClick: this.btnImportExport.callback1 },
                { icon: "📤", title: "Export LoRAs", onClick: this.btnImportExport.callback2 },
            ]);

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
            moveToBottom(this.actionBar);

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
            // "High" or "Low"
            const widget = new LoraWidget(name, streamType, streamType === "High" ? "#2a3b4d" : "#4d2a2a"); // Blueish for High, Reddish for Low
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
                    if (o.widgets_values) o.widgets_values[this.widgets.indexOf(infoW)] = infoW.value;
                    if (o.widgets_values_named) o.widgets_values_named[infoName] = infoW.value;
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
        const configure = nodeType.prototype.configure;
        nodeType.prototype.configure = function(info) {
            const result = configure.apply(this, arguments);
            
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

                        if (data && typeof data === "object" && !data.loras?.length && info.widgets_values_named) {
                            data.loras = Object.entries(info.widgets_values_named)
                                .filter(([name, value]) => name.startsWith(`LORA_${type}_`) && value && typeof value === "object")
                                .map(([, value]) => value);
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
            const collapsedStates = info.collapsed_states || info.extensions?.collapsed_states;
            if (collapsedStates) {
                if (this.labelHigh && collapsedStates.high) this.labelHigh.setCollapsed(true);
                if (this.labelLow && collapsedStates.low) this.labelLow.setCollapsed(true);
                if (this.labelAuto && collapsedStates.auto) this.labelAuto.setCollapsed(true);
            }

            this.reorderWidgets();
            this.ensureHiddenWidgets();
            return result;
        };
        
        const onDrawForeground = nodeType.prototype.onDrawForeground;
        nodeType.prototype.onDrawForeground = function(ctx) {
            if (onDrawForeground) onDrawForeground.apply(this, arguments);

            const height = this.finalStatusLabel.computeSize(this.size[0])[1];
            const posY = this.size[1] + 10;
            const w = this.size[0];
            this.finalStatusLabel.draw(ctx, this, w, posY, height);

            // Save bounding box for hit testing
            this.bigLabelBounds = [0, posY, w, height];
        };

        // Context Menu for removal
        const getSlotMenuOptions = nodeType.prototype.getSlotMenuOptions;
        nodeType.prototype.getSlotMenuOptions = function(slot) {
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
                    event: rgthree.lastCanvasMouseEvent,
                    className: "dark",
                    scale: Math.max(1, app?.canvas?.ds?.scale || 1)
                });
                return undefined;
            }

            const items = getLoraRowMenuItems(this, slot, DOUBLE_STREAM_ROW_MENU);
            if (items) return items;
            return getSlotMenuOptions?.apply(this, arguments);
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
            // Check if mouse is over the external big status label
            if (this.bigLabelBounds) {
                const [lx, ly, lw, lh] = this.bigLabelBounds;
                if (canvasX >= lx && canvasX <= lx + lw &&
                    canvasY >= ly && canvasY <= ly + lh) {
                    return { widget: null, output: { type: "STATUS_LABEL_WIDGET" } };
                }
            }
            return getLoraRowSlot(this, canvasX, canvasY, DOUBLE_STREAM_ROW_MENU);
        };
    }
});
