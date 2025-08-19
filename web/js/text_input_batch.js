import { app } from "../../../scripts/app.js";

console.log("Patching node: text_input_batch.js");

function ensureStringsJsonWidget(node) {
    let w = node.widgets?.find(w => w.name === "strings_json");
    if (!w) {
        w = node.addWidget("text", "strings_json", node.properties?._strings || "[]", () => {}, { multiline: true });
        w.name = "strings_json";
    }
    // 前端彻底隐藏：不显示、不绘制、不占空间
    w.disabled = true;
    w.visible = false;
    w.draw = () => {};
    w.computeSize = () => [0, 0];
    return w;
}

function getItems(node) {
    try {
        const hidden = ensureStringsJsonWidget(node);
        const raw = String(hidden.value || node.properties?._strings || "[]");
        const arr = JSON.parse(raw);
        return Array.isArray(arr) ? arr.map(v => String(v ?? "")) : [];
    } catch (e) {
        return [];
    }
}

function setItems(node, arr) {
    const json = JSON.stringify(arr);
    const hidden = ensureStringsJsonWidget(node);
    hidden.value = json;
    node.properties = node.properties || {};
    node.properties._strings = json;
}

function installAddButton(node) {
    if (node.__addButtonInstalled) return;
    const addBtn = node.addWidget("button", "➕ 添加字符串", null, () => {
        const items = getItems(node);
        items.push("");
        setItems(node, items);
        node.setDirtyCanvas(true, true);
        return true;
    });
    addBtn.options.serialize = false;
    node.__addButtonInstalled = true;
}

function getWidgetsBottom(node) {
    // 动态计算当前widgets区域的底部Y，避免重叠
    let bottom = 0;
    if (Array.isArray(node.widgets)) {
        for (const w of node.widgets) {
            if (w && w.visible !== false) {
                const y = (typeof w.last_y === 'number') ? w.last_y : 0;
                const h = (w.type === 'button') ? 26 : 24;
                bottom = Math.max(bottom, y + h);
            }
        }
    }
    return bottom;
}

function ensureTextareas(node, layout, items) {
    const ds = app?.canvas?.ds;
    const canvas = app?.canvas?.canvas;
    if (!ds || !canvas) return;
    const rect = canvas.getBoundingClientRect();

    if (!node.__taEls) node.__taEls = [];

    for (let i = 0; i < items.length; i++) {
        const cell = layout[i];
        if (!cell) continue;
        let ta = node.__taEls[i];
        if (!ta) {
            ta = document.createElement('textarea');
            ta.placeholder = `文本 ${i+1}`;
            ta.spellcheck = false;
            ta.wrap = 'soft';
            ta.value = items[i] || "";
            ta.style.cssText = `position: fixed; z-index: 10000; resize: none; padding: 6px; border-radius: 4px; border: 1px solid #666; background: #1a1a1a; color: #eee; font: 12px/1.4 monospace; box-sizing: border-box; overflow: auto;`;
            ta.addEventListener('input', () => {
                const arr = getItems(node);
                if (i < arr.length) arr[i] = ta.value;
                setItems(node, arr);
            });
            document.body.appendChild(ta);
            node.__taEls[i] = ta;
        }
        const sx = (node.pos[0] + cell.x + ds.offset[0]) * ds.scale + rect.left;
        const sy = (node.pos[1] + cell.y + ds.offset[1]) * ds.scale + rect.top;
        const sw = cell.w * ds.scale;
        const sh = cell.h * ds.scale;
        ta.style.left = `${Math.round(sx)}px`;
        ta.style.top = `${Math.round(sy)}px`;
        ta.style.width = `${Math.max(40, Math.round(sw))}px`;
        ta.style.height = `${Math.max(32, Math.round(sh))}px`;
        ta.style.visibility = 'visible';
    }

    for (let j = items.length; j < (node.__taEls?.length || 0); j++) {
        const el = node.__taEls[j];
        if (el && el.remove) el.remove();
    }
    node.__taEls.length = items.length;
}

function layoutCells(node, items) {
    const PADDING = 8;
    const GAP = 6;
    const MIN_H = 48; // 每项最小高度
    const n = items.length;
    if (n === 0) return [];
    const cols = n > 1 ? 2 : 1;
    const rows = Math.ceil(n / cols);
    const availW = node.size[0] - PADDING * 2;
    const cellW = Math.floor((availW - GAP * (cols - 1)) / cols);
    const startY = PADDING + getWidgetsBottom(node);
    const availH = Math.max(0, node.size[1] - startY - PADDING);
    const cellH = Math.max(MIN_H, Math.floor((availH - GAP * (rows - 1)) / rows));

    const cells = [];
    for (let i = 0; i < n; i++) {
        const r = Math.floor(i / cols);
        const c = i % cols;
        const x = PADDING + c * (cellW + GAP);
        const y = startY + r * (cellH + GAP);
        cells.push({ x, y, w: cellW, h: cellH });
    }
    return cells;
}

function installDrawingHandlers(node) {
    if (node.__drawingInstalled) return;
    node.__drawingInstalled = true;

    const relayoutAndUpdate = (ctx) => {
        const items = getItems(node);
        if (!items.length) return;
        const cells = layoutCells(node, items);
        ensureTextareas(node, cells, items);
    };

    const origDraw = node.onDrawForeground;
    node.onDrawForeground = function(ctx) {
        if (origDraw) origDraw.call(this, ctx);
        relayoutAndUpdate(ctx);
    };

    const origResize = node.onResize;
    node.onResize = function(size) {
        if (origResize) origResize.call(this, size);
        // 触发布局更新
        relayoutAndUpdate();
    };

    const origRemoved = node.onRemoved;
    node.onRemoved = function() {
        if (origRemoved) origRemoved.call(this);
        if (this.__taEls) {
            for (const el of this.__taEls) { try { el.remove(); } catch(e) {} }
            this.__taEls = [];
        }
    };
}

app.registerExtension({
    name: "A_my_nodes.TextInputBatch.UI",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "TextInputBatch") return;
        console.log("[TextInputBatch] UI扩展注册");

        const origOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function() {
            if (origOnNodeCreated) origOnNodeCreated.apply(this, arguments);
            ensureStringsJsonWidget(this);
            installAddButton(this);
            installDrawingHandlers(this);
            setItems(this, getItems(this));
        };

        const origConfigure = nodeType.prototype.configure;
        nodeType.prototype.configure = function(info) {
            if (origConfigure) origConfigure.apply(this, arguments);
            ensureStringsJsonWidget(this);
            installAddButton(this);
            installDrawingHandlers(this);
            if (info && info.properties && typeof info.properties._strings === 'string') {
                this.properties = this.properties || {};
                this.properties._strings = info.properties._strings;
                const hidden = ensureStringsJsonWidget(this);
                hidden.value = this.properties._strings;
            }
            setItems(this, getItems(this));
        };
    },
}); 