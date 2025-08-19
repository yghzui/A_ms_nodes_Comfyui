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

function moveItem(arr, from, to) {
    const n = arr.length;
    if (n === 0) return arr;
    const src = Math.max(0, Math.min(n - 1, from|0));
    let dst = Math.max(0, Math.min(n - 1, to|0));
    if (src === dst) return arr;
    const copy = arr.slice();
    const [it] = copy.splice(src, 1);
    copy.splice(dst, 0, it);
    return copy;
}

function showItemContextMenu(node, index, event) {
    const items = getItems(node);
    const n = items.length;
    const hasUp = index > 0;
    const hasDown = index < n - 1;
    const Lite = window.LiteGraph || window?.app?.canvas?.graph?.constructor; // 尝试拿到LiteGraph

    const doDelete = () => {
        const next = items.slice(0, index).concat(items.slice(index + 1));
        setItems(node, next);
        app.graph.setDirtyCanvas(true, true);
    };
    const doMoveUp = () => {
        if (!hasUp) return;
        const next = moveItem(items, index, index - 1);
        setItems(node, next);
        app.graph.setDirtyCanvas(true, true);
    };
    const doMoveDown = () => {
        if (!hasDown) return;
        const next = moveItem(items, index, index + 1);
        setItems(node, next);
        app.graph.setDirtyCanvas(true, true);
    };
    const doMoveTo = () => {
        let to = prompt(`移动到索引 (0 - ${Math.max(0, n - 1)}):`, String(index));
        if (to == null) return;
        to = Number(to);
        if (!Number.isFinite(to)) return;
        const next = moveItem(items, index, to);
        setItems(node, next);
        app.graph.setDirtyCanvas(true, true);
    };

    // 临时降低触发的 textarea 的指针，避免挡住菜单
    const targetEl = event?.target;
    let prevPointer = null;
    if (targetEl && targetEl.style) {
        prevPointer = targetEl.style.pointerEvents;
        targetEl.style.pointerEvents = 'none';
    }

    const restorePointer = () => {
        if (targetEl && targetEl.style) targetEl.style.pointerEvents = prevPointer || 'auto';
    };

    if (Lite && Lite.ContextMenu) {
        const menu = [
            { content: `🗑️ 删除`, callback: doDelete },
            { content: `⬆️ 上移`, disabled: !hasUp, callback: doMoveUp },
            { content: `⬇️ 下移`, disabled: !hasDown, callback: doMoveDown },
            { content: `↔ 移动到索引…`, callback: doMoveTo },
        ];
        const cm = new Lite.ContextMenu(menu, {
            event,
            title: `文本 ${index+1}`,
            className: "dark",
            scale: Math.max(1, app?.canvas?.ds?.scale || 1),
        });
        // 提升菜单z-index，确保浮在textarea之上
        try {
            const root = cm.root || cm.element || cm.menu || cm;
            if (root && root.style) root.style.zIndex = '10050';
        } catch(e) {}
        // 点击一次任意处后恢复
        setTimeout(() => {
            const once = () => { document.removeEventListener('mousedown', once, true); restorePointer(); };
            document.addEventListener('mousedown', once, true);
        }, 0);
    } else {
        // 简易回退
        const choice = prompt(`操作: d=删除, u=上移, n=下移, m=移动到索引`, "d");
        if (choice === 'd') doDelete();
        else if (choice === 'u') doMoveUp();
        else if (choice === 'n') doMoveDown();
        else if (choice === 'm') doMoveTo();
        restorePointer();
    }
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
            ta.style.cssText = `position: fixed; z-index: 100; resize: none; padding: 6px; border-radius: 4px; border: 1px solid #666; background: #1a1a1a; color: #eee; font: 12px/1.4 monospace; box-sizing: border-box; overflow: auto;`;
            ta.addEventListener('input', () => {
                const arr = getItems(node);
                if (i < arr.length) arr[i] = ta.value;
                setItems(node, arr);
            });
            // 右键菜单
            if (!ta.__ctxInstalled) {
                ta.addEventListener('contextmenu', (e) => {
                    e.preventDefault(); e.stopPropagation();
                    showItemContextMenu(node, i, e);
                });
                ta.__ctxInstalled = true;
            }
            document.body.appendChild(ta);
            node.__taEls[i] = ta;
        } else {
            // 更新现有textarea的值，确保移动后内容正确显示
            ta.value = items[i] || "";
        }
        const sx = (node.pos[0] + cell.x + ds.offset[0]) * ds.scale + rect.left;
        const sy = (node.pos[1] + cell.y + ds.offset[1]) * ds.scale + rect.top;
        const sw = cell.w * ds.scale;
        const sh = cell.h * ds.scale;
        ta.style.left = `${Math.round(sx)}px`;
        ta.style.top = `${Math.round(sy)}px`;
        ta.style.width = `${Math.max(40, Math.round(sw))}px`;
        ta.style.height = `${Math.max(32, Math.round(sh))}px`;
        const fontPx = Math.max(10, Math.round(12 * (ds.scale || 1)));
        ta.style.fontSize = `${fontPx}px`;
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