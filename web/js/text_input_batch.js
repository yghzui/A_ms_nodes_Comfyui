import { app } from "../../../scripts/app.js";

console.log("Patching node: text_input_batch.js");

function ensureStringsJsonWidget(node) {
    let w = node.widgets?.find(w => w.name === "strings_json");
    if (!w) {
        w = node.addWidget("text", "strings_json", node.properties?._strings || "[]", () => {}, { multiline: true });
        w.name = "strings_json";
        w.disabled = true;
        w.visible = false;
    }
    return w;
}

function collectTextValues(node) {
    const texts = [];
    for (const w of node.widgets || []) {
        if (w && typeof w.name === 'string' && w.name.startsWith("TEXT_")) {
            texts.push(String(w.value ?? ""));
        }
    }
    return texts;
}

function removeAllTextWidgets(node) {
    if (!node.widgets) return;
    node.widgets = node.widgets.filter(w => !(w && typeof w.name === 'string' && w.name.startsWith("TEXT_")));
}

function addTextWidget(node, text = "") {
    const idx = (node.__textItemCounter = (node.__textItemCounter || 0) + 1);
    const w = node.addWidget("text", `TEXT_${idx}`, String(text), (v) => {
        updateStringsJson(node);
    });
    return w;
}

function updateStringsJson(node) {
    const items = collectTextValues(node);
    const json = JSON.stringify(items);
    const hidden = ensureStringsJsonWidget(node);
    hidden.value = json;
    node.properties = node.properties || {};
    node.properties._strings = json;
}

function rebuildTextWidgetsFromJson(node) {
    const hidden = ensureStringsJsonWidget(node);
    let arr = [];
    try {
        arr = JSON.parse(String(hidden.value || node.properties?._strings || "[]"));
        if (!Array.isArray(arr)) arr = [];
    } catch (e) {
        arr = [];
    }
    // 移除旧的 TEXT_ 小部件，仅保留默认/其它控件
    removeAllTextWidgets(node);
    // 逐个添加文本输入控件
    for (const s of arr) addTextWidget(node, String(s ?? ""));
}

app.registerExtension({
    name: "A_my_nodes.TextInputBatch.UI",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "TextInputBatch") return;
        console.log("[TextInputBatch] UI扩展注册");

        const origOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function() {
            if (origOnNodeCreated) origOnNodeCreated.apply(this, arguments);

            // 确保隐藏容器与初始化
            ensureStringsJsonWidget(this);

            // 添加“添加字符串”按钮（一次性）
            if (!this.__addButtonInstalled) {
                const addBtn = this.addWidget("button", "➕ 添加字符串", null, () => {
                    addTextWidget(this, "");
                    updateStringsJson(this);
                    this.setDirtyCanvas(true, true);
                    return true;
                });
                addBtn.options.serialize = false;
                this.__addButtonInstalled = true;
            }

            // 根据已存储 JSON 恢复文本条目
            rebuildTextWidgetsFromJson(this);

            // 初始同步一次
            updateStringsJson(this);
        };

        const origConfigure = nodeType.prototype.configure;
        nodeType.prototype.configure = function(info) {
            if (origConfigure) origConfigure.apply(this, arguments);
            // 不清除默认 widgets，避免丢失 index 等输入

            // 先确保隐藏容器存在，并尽量从 properties 恢复
            ensureStringsJsonWidget(this);
            if (info && info.properties && typeof info.properties._strings === 'string') {
                this.properties = this.properties || {};
                this.properties._strings = info.properties._strings;
                const hidden = ensureStringsJsonWidget(this);
                hidden.value = this.properties._strings;
            }

            // 若“添加字符串”按钮未安装（例如加载工作流时），则补装
            if (!this.__addButtonInstalled) {
                const addBtn = this.addWidget("button", "➕ 添加字符串", null, () => {
                    addTextWidget(this, "");
                    updateStringsJson(this);
                    this.setDirtyCanvas(true, true);
                    return true;
                });
                addBtn.options.serialize = false;
                this.__addButtonInstalled = true;
            }

            // 重建文本输入控件
            rebuildTextWidgetsFromJson(this);

            // 同步一次隐藏 JSON
            updateStringsJson(this);

            // 尺寸刷新
            const size = this.computeSize();
            this.size[0] = Math.max(this.size[0], size[0] || 260);
            this.size[1] = Math.max(this.size[1], size[1] || 160);
        };
    },
}); 