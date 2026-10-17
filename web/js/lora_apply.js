import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";
import { modal } from "./utils/modal.js";
import { $el, showTopNotification } from "./utils/shared_utils.js";
import { rgthree } from "./core/rgthree.js";
import { RgthreeBetterButtonWidget, addIconButtonRow } from "./utils/utils_widgets.js";
import { LoraWidget, LabelWidget, DoubleButtonWidget, StatusLabelWidget, showStatusLabelSettingsModal } from "./lora/lora_common.js";

function normalizeLoras(data) {
    const loras = Array.isArray(data) ? data : data?.loras;
    if (!Array.isArray(loras)) throw new Error("配置需要包含 LoRA 列表");
    return loras.filter(value => value && typeof value === "object").map(value => ({
        on: value.on !== false, lora: value.lora || "None", strength: value.strength ?? 1,
    }));
}

function chooseMode(title, choices) {
    return new Promise(resolve => modal.show({
        title, content: "", width: "380px", onClose: () => resolve(null),
        buttons: [...choices.map(([text, value]) => ({
            text, onClick: () => { resolve(value); modal.close(); },
        })), { text: "取消", onClick: () => modal.close() }],
    }));
}

async function resolveLoras(data) {
    if (Array.isArray(data) || Array.isArray(data?.loras)) return normalizeLoras(data);
    if (data?.high_loras || data?.low_loras || data?.high || data?.low) {
        const stream = await chooseMode("选择要导入的双流分支", [["High", "high"], ["Low", "low"]]);
        if (!stream) return null;
        const branch = data[stream];
        return normalizeLoras(data[`${stream}_loras`] || (Array.isArray(branch) ? branch : branch?.loras) || []);
    }
    throw new Error("未找到 LoRA 配置");
}

function showJsonDialog(node, importing) {
    const input = $el("textarea", {
        value: importing ? "" : JSON.stringify(node.getLoraConfig(), null, 2),
        readOnly: !importing,
        style: { width: "100%", height: "240px", boxSizing: "border-box", background: "#222", color: "white" },
    });
    const content = $el("div", {}, [input]);
    if (importing) content.appendChild($el("input", {
        type: "file", accept: ".json", onchange: async event => {
            const file = event.target.files[0];
            if (file) input.value = await file.text();
        },
    }));
    modal.show({
        title: importing ? "Import LoRAs" : "Export LoRAs", content, width: "520px",
        buttons: [{
            text: importing ? "导入" : "复制", type: "primary", onClick: async () => {
                try {
                    if (importing) {
                        const data = JSON.parse(input.value);
                        modal.close();
                        await node.insertLoraConfig(data);
                    } else {
                        await navigator.clipboard.writeText(input.value);
                        showTopNotification("已复制 LoRA 配置", "success");
                    }
                } catch (error) { showTopNotification(error.message, "error"); }
            },
        }, ...(!importing ? [{
            text: "下载 JSON", onClick: () => {
                const url = URL.createObjectURL(new Blob([input.value], { type: "application/json" }));
                const link = $el("a", { href: url, download: "lora_config.json" });
                link.click();
                setTimeout(() => URL.revokeObjectURL(url), 1000);
            },
        }] : []), { text: "关闭", onClick: () => modal.close() }],
    });
}

async function saveAsset(node) {
    try {
        const response = await api.fetchApi("/a_my_nodes/assets/models");
        if (!response.ok) throw new Error("读取资产库失败");
        const data = await response.json();
        const groups = data.groups || [];
        if (!groups.length) throw new Error("请先在资产管理器中创建分组");
        const saved = localStorage.getItem("lora_apply_asset_group");
        const groupSelect = $el("select", { style: { width: "100%" } }, groups.map((group, index) =>
            $el("option", { value: String(index), textContent: group.name, selected: group.name === saved })
        ));
        const title = $el("input", {
            value: node.widgets.find(widget => widget.name === "key_to_check")?.value || "未命名配置",
            style: { width: "100%", boxSizing: "border-box" },
        });
        modal.show({
            title: "保存到资产库", width: "400px",
            content: $el("div", { style: { display: "grid", gap: "8px" } }, [
                $el("label", { textContent: "分组" }), groupSelect,
                $el("label", { textContent: "条目标题" }), title,
            ]),
            buttons: [{ text: "保存", type: "primary", onClick: async () => {
                try {
                    // Re-read before saving so edits made in the manager are preserved.
                    const latestResponse = await api.fetchApi("/a_my_nodes/assets/models");
                    if (!latestResponse.ok) throw new Error("读取资产库失败");
                    const latest = await latestResponse.json();
                    const groupName = groups[Number(groupSelect.value)].name;
                    const group = latest.groups.find(value => value.name === groupName);
                    if (!group) throw new Error("分组已被删除，请重新选择");
                    const config = node.getLoraConfig();
                    group.items ||= [];
                    group.items.push({
                        id: Date.now().toString(36) + Math.random().toString(36).slice(2), stream_type: "single", keyword: title.value.trim() || "未命名配置",
                        check_mode: config.check_mode, loras: config.loras, preview_image: "",
                    });
                    const result = await api.fetchApi("/a_my_nodes/assets/models", {
                        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(latest),
                    });
                    if (!result.ok || !(await result.json()).success) throw new Error("保存资产失败");
                    localStorage.setItem("lora_apply_asset_group", groupName);
                    await window.AssetManager?.loadData();
                    modal.close();
                    showTopNotification("已保存到资产库", "success");
                } catch (error) { showTopNotification(error.message, "error"); }
            } }, { text: "取消", onClick: () => modal.close() }],
        });
    } catch (error) { showTopNotification(error.message, "error"); }
}

app.registerExtension({
    name: "A_my_nodes.LoraApply",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "LoraApply") return;
        const created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = created?.apply(this, arguments);
            this.loraWidgets = [];
            this.loraCounter = 0;
            this.assetStreamType = "single";
            this.infoWidget = this.widgets.find(widget => widget.name === "loras_info");
            this.infoWidget.type = "hidden";
            this.infoWidget.computeSize = () => [0, -4];
            this.infoWidget.serializeValue = () => JSON.stringify(this.getLoraConfig().loras);
            this.autoLabel = this.addCustomWidget(new LabelWidget("auto_label", "Auto Enable Logic", collapsed => {
                for (const name of ["enable_mode", "Check Status", "dict_input", "key_to_check", "check_mode"]) {
                    const widget = this.widgets.find(value => value.name === name);
                    if (widget) widget.hidden = collapsed;
                }
                this.reorderLoras();
            }));
            this.statusWidget = this.addWidget("text", "Check Status", "Pending...", () => {}, {});
            if (this.statusWidget.inputEl) this.statusWidget.inputEl.readOnly = true;
            this.loraLabel = this.addCustomWidget(new LabelWidget("lora_label", "LoRA 列表", () => this.reorderLoras()));
            this.addLoraButton = this.addCustomWidget(new RgthreeBetterButtonWidget("添加 LoRA", (event, pos) => {
                const widget = this.addLora({ on: true, lora: "None", strength: 1 });
                widget.onLoraDown(event, pos, this);
            }));
            this.assetButtons = new DoubleButtonWidget(
                "asset_buttons", "插入模板", "保存到资产库",
                () => {
                    if (!window.AssetManager) return showTopNotification("资产管理器未就绪", "error");
                    window.AssetManager.showDrawer(this, "models", async items => {
                        try { for (const item of items) await this.insertLoraConfig(item); }
                        catch (error) { showTopNotification(error.message, "error"); }
                    });
                }, () => saveAsset(this),
            );
            this.jsonButtons = new DoubleButtonWidget(
                "json_buttons", "Import LoRAs", "Export LoRAs",
                () => showJsonDialog(this, true), () => showJsonDialog(this, false),
            );
            this.actionBar = addIconButtonRow(this, "lora_action_icons", [
                { icon: "✨", title: "插入模板", onClick: this.assetButtons.callback1 },
                { icon: "💾", title: "保存到资产库", onClick: this.assetButtons.callback2 },
                { icon: "📥", title: "Import LoRAs", onClick: this.jsonButtons.callback1 },
                { icon: "📤", title: "Export LoRAs", onClick: this.jsonButtons.callback2 },
            ]);
            this.finalStatusLabel = new StatusLabelWidget("final_status_label");
            for (const widget of [this.autoLabel, this.statusWidget, this.loraLabel, this.addLoraButton]) {
                widget.options.serialize = false;
            }
            this.reorderLoras();
            return result;
        };
        nodeType.prototype.getLoraConfig = function () {
            return {
                stream_type: "single",
                keyword: this.widgets.find(widget => widget.name === "key_to_check")?.value || "",
                check_mode: this.widgets.find(widget => widget.name === "check_mode")?.value || "contains",
                loras: this.loraWidgets.map(widget => ({ ...widget.value })),
            };
        };
        nodeType.prototype.addLora = function (value) {
            const widget = new LoraWidget(`LORA_${++this.loraCounter}`, "Single", "#4d2a2a");
            widget.value = { ...value };
            widget.options.serialize = false;
            this.addCustomWidget(widget);
            this.loraWidgets.push(widget);
            this.reorderLoras();
            return widget;
        };
        nodeType.prototype.reorderLoras = function () {
            const dynamic = new Set([this.loraLabel, ...this.loraWidgets, this.addLoraButton,
                this.actionBar, this.infoWidget]);
            const logic = [this.autoLabel, this.widgets.find(w => w.name === "enable_mode"), this.statusWidget,
                ...["dict_input", "key_to_check", "check_mode"].map(name => this.widgets.find(w => w.name === name))];
            logic.forEach(widget => dynamic.add(widget));
            this.widgets = [...this.widgets.filter(widget => !dynamic.has(widget)), ...logic, this.loraLabel,
                ...this.loraWidgets, this.addLoraButton, this.actionBar, this.infoWidget].filter(Boolean);
            for (const widget of this.loraWidgets) {
                widget.hidden = this.loraLabel.collapsed;
                widget.computeSize = () => [this.size[0], widget.hidden ? -4 : LiteGraph.NODE_WIDGET_HEIGHT];
            }
            this.infoWidget.value = JSON.stringify(this.getLoraConfig().loras);
            this.setSize([Math.max(400, this.size[0]), this.computeSize()[1]]);
            this.setDirtyCanvas(true, true);
        };
        nodeType.prototype.insertLoraConfig = async function (data) {
            const loras = await resolveLoras(data);
            if (loras === null) return;
            const mode = this.loraWidgets.length ? await chooseMode("插入 LoRA 配置", [["追加", "append"], ["替换", "replace"]]) : "replace";
            if (!mode) return;
            if (mode === "replace") {
                this.widgets = this.widgets.filter(widget => !this.loraWidgets.includes(widget));
                this.loraWidgets = [];
            }
            for (const value of loras) this.addLora(value);
            const keyword = data.keyword || data.key_to_check;
            const keyWidget = this.widgets.find(widget => widget.name === "key_to_check");
            if (keyword && keyWidget) {
                keyWidget.value = mode === "replace" ? keyword : [...new Set(
                    [...keyWidget.value.split(";"), keyword].map(value => value.trim()).filter(Boolean)
                )].join(";");
            }
            if (mode === "replace" && data.check_mode) {
                this.widgets.find(widget => widget.name === "check_mode").value = data.check_mode;
            }
            this.reorderLoras();
        };
        const serialize = nodeType.prototype.onSerialize;
        nodeType.prototype.onSerialize = function (info) {
            this.infoWidget.value = JSON.stringify(this.getLoraConfig().loras);
            serialize?.apply(this, arguments);
            info.lora_config = this.getLoraConfig();
            info.loras_collapsed = this.loraLabel.collapsed;
            info.auto_collapsed = this.autoLabel.collapsed;
            if (info.widgets_values) info.widgets_values[this.widgets.indexOf(this.infoWidget)] = this.infoWidget.value;
        };
        const configure = nodeType.prototype.configure;
        nodeType.prototype.configure = function (info) {
            const result = configure.apply(this, arguments);
            this.widgets = this.widgets.filter(widget => !this.loraWidgets.includes(widget));
            this.loraWidgets = [];
            try {
                const config = info.lora_config || info.extensions?.lora_config ||
                    JSON.parse(info.widgets_values_named?.loras_info || this.infoWidget.value || "[]");
                for (const value of normalizeLoras(config)) this.addLora(value);
            } catch (error) { showTopNotification(`LoRA 配置读取失败: ${error.message}`, "error"); }
            this.loraLabel.setCollapsed((info.loras_collapsed ?? info.extensions?.loras_collapsed) === true);
            this.autoLabel.setCollapsed((info.auto_collapsed ?? info.extensions?.auto_collapsed) === true);
            this.reorderLoras();
            return result;
        };
        const executed = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            executed?.apply(this, arguments);
            const active = message.text?.[0] === "True";
            const mode = this.widgets.find(widget => widget.name === "enable_mode")?.value;
            this.statusWidget.value = mode === "Auto" ? `${active ? "✅" : "❌"} Auto: ${active ? "Enabled" : "Disabled"}` : `🔒 Force ${active ? "True" : "False"}`;
            if (this.statusWidget.inputEl) this.statusWidget.inputEl.style.color = active ? "#4caf50" : "#f44336";
            this.setDirtyCanvas(true, true);
        };
        const drawForeground = nodeType.prototype.onDrawForeground;
        nodeType.prototype.onDrawForeground = function (ctx) {
            drawForeground?.apply(this, arguments);
            if (this.flags.collapsed) return;
            const height = this.finalStatusLabel.computeSize(this.size[0])[1];
            this.finalStatusLabel.draw(ctx, this, this.size[0], this.size[1] + 10, height);
            this.bigLabelBounds = [0, this.size[1] + 10, this.size[0], height];
        };
        const slotInPosition = nodeType.prototype.getSlotInPosition;
        nodeType.prototype.getSlotInPosition = function (x, y) {
            const bounds = this.bigLabelBounds;
            if (!this.flags.collapsed && bounds && x >= bounds[0] && x <= bounds[0] + bounds[2] &&
                y >= bounds[1] && y <= bounds[1] + bounds[3]) {
                return { widget: null, output: { type: "STATUS_LABEL_WIDGET" } };
            }
            return slotInPosition?.apply(this, arguments);
        };
        const slotMenu = nodeType.prototype.getSlotMenuOptions;
        nodeType.prototype.getSlotMenuOptions = function (slot) {
            if (slot?.output?.type === "STATUS_LABEL_WIDGET") {
                new LiteGraph.ContextMenu([{ content: "设置标签样式 (全局)", callback: showStatusLabelSettingsModal }], {
                    event: rgthree.lastCanvasMouseEvent, className: "dark", scale: Math.max(1, app.canvas.ds.scale),
                });
                return;
            }
            return slotMenu?.apply(this, arguments);
        };
        const extraMenu = nodeType.prototype.getExtraMenuOptions;
        nodeType.prototype.getExtraMenuOptions = function (canvas, options) {
            extraMenu?.apply(this, arguments);
            for (const [index, widget] of this.loraWidgets.entries()) options.push({
                content: widget.value.lora || "None", submenu: { options: [
                    { content: "上移", disabled: index === 0, callback: () => {
                        this.loraWidgets.splice(index, 1); this.loraWidgets.splice(index - 1, 0, widget); this.reorderLoras();
                    } },
                    { content: "下移", disabled: index === this.loraWidgets.length - 1, callback: () => {
                        this.loraWidgets.splice(index, 1); this.loraWidgets.splice(index + 1, 0, widget); this.reorderLoras();
                    } },
                    { content: "复制路径", callback: () => navigator.clipboard.writeText(widget.value.lora || "") },
                    { content: "删除", callback: () => {
                        this.loraWidgets.splice(index, 1); this.widgets = this.widgets.filter(value => value !== widget); this.reorderLoras();
                    } },
                ] },
            });
            options.push({ content: "全部启用", callback: () => { this.loraWidgets.forEach(w => w.value.on = true); this.reorderLoras(); } },
                { content: "全部禁用", callback: () => { this.loraWidgets.forEach(w => w.value.on = false); this.reorderLoras(); } },
                { content: "清空 LoRA", callback: () => {
                    this.widgets = this.widgets.filter(widget => !this.loraWidgets.includes(widget));
                    this.loraWidgets = []; this.reorderLoras();
                } }, { content: this.properties?.useOfficialLoraList ? "使用自定义 LoRA 列表" : "使用官方 LoRA 列表", callback: () => {
                    this.properties.useOfficialLoraList = !this.properties.useOfficialLoraList;
                } });
        };
    },
});
