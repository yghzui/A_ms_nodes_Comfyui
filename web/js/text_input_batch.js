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
    // 保持可序列化（用于工作流保存/恢复）
    // w.options && (w.options.serialize = true); // 走默认即可
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

function showTextareaOverlay(node, initialText, onCommit) {
    // 创建浮动多行输入框
    const ta = document.createElement('textarea');
    ta.value = initialText || "";
    ta.style.cssText = `position: fixed; z-index: 10000; padding: 8px; border-radius: 4px; border: 1px solid #888; background: #111; color: #fff; font: 12px/1.4 monospace; width: 360px; height: 120px;`;
    // 放在鼠标附近
    const lastEvt = app?.canvas?.last_mouse_position || [window.innerWidth/2, window.innerHeight/2];
    ta.style.left = (lastEvt[0] + 10) + 'px';
    ta.style.top = (lastEvt[1] + 10) + 'px';
    document.body.appendChild(ta);
    ta.focus();

    const cleanup = () => { try { ta.remove(); } catch(e) {} };
    const commit = () => { try { onCommit?.(ta.value); } finally { cleanup(); } };
    const cancel = () => cleanup();

    ta.addEventListener('keydown', (e) => {
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            e.preventDefault(); commit();
        } else if (e.key === 'Escape') {
            e.preventDefault(); cancel();
        }
    });
    ta.addEventListener('blur', () => commit());
}

function getWidgetsBottom(node) {
    // 动态计算当前widgets区域的底部Y，避免重叠
    let bottom = 0;
    if (Array.isArray(node.widgets)) {
        for (const w of node.widgets) {
            if (w && w.visible !== false) {
                const y = (typeof w.last_y === 'number') ? w.last_y : 0;
                // 经验补偿：按钮/文本等小部件高度约 22-28，加上间距
                const h = (w.type === 'button') ? 26 : 24;
                bottom = Math.max(bottom, y + h);
            }
        }
    }
    // 不额外添加边距，由外部控制
    return bottom;
}

function installDrawingHandlers(node) {
    if (node.__drawingInstalled) return;
    node.__drawingInstalled = true;
    const PADDING = 8;
    const GAP = 6;
    const ITEM_H = 56; // 每项高度，便于多行

    // 前景绘制：两列排版
    const origDraw = node.onDrawForeground;
    node.onDrawForeground = function(ctx) {
        if (origDraw) origDraw.call(this, ctx);
        const items = getItems(this);
        const n = items.length;
        if (!n) return;
        const cols = n > 1 ? 2 : 1;
        const rows = Math.ceil(n / cols);
        const availW = this.size[0] - PADDING * 2;
        const cellW = Math.floor((availW - GAP * (cols - 1)) / cols);
        const startY = PADDING + getWidgetsBottom(this); // 动态避开上方widgets
        this.__rects = [];
        ctx.save();
        ctx.font = '12px monospace';
        ctx.textAlign = 'left';
        ctx.textBaseline = 'top';
        for (let i = 0; i < n; i++) {
            const r = Math.floor(i / cols);
            const c = i % cols;
            const x = PADDING + c * (cellW + GAP);
            const y = startY + r * (ITEM_H + GAP);
            // 背景
            ctx.fillStyle = '#222';
            ctx.strokeStyle = '#555';
            ctx.lineWidth = 1;
            ctx.fillRect(x, y, cellW, ITEM_H);
            ctx.strokeRect(x, y, cellW, ITEM_H);
            // 文本（显示多行，裁剪）
            const text = String(items[i] || "");
            const lines = text.split(/\r?\n/);
            const maxLines = 3;
            ctx.fillStyle = '#ddd';
            const lineH = 14;
            for (let li = 0; li < Math.min(lines.length, maxLines); li++) {
                const ly = y + 6 + li * lineH;
                const str = lines[li];
                // 粗略裁剪
                let clip = str;
                while (ctx.measureText(clip).width > cellW - 12 && clip.length > 0) {
                    clip = clip.slice(0, clip.length - 1);
                }
                ctx.fillText(clip, x + 6, ly);
            }
            // 记录可点击区域
            this.__rects.push({ x, y, w: cellW, h: ITEM_H });
        }
        ctx.restore();
        // 根据行数动态增高节点以避免重叠
        const needH = startY + rows * (ITEM_H + GAP) + PADDING;
        if (!this.__autoH || this.__autoH !== needH) {
            this.__autoH = needH;
            this.size[1] = Math.max(this.size[1], needH);
        }
    };

    // 点击编辑
    const origDown = node.onMouseDown;
    node.onMouseDown = function(e) {
        if (origDown) origDown.call(this, e);
        const items = getItems(this);
        const rects = this.__rects || [];
        if (!rects.length) return false;
        const nodePos = this.pos;
        for (let i = 0; i < rects.length; i++) {
            const r = rects[i];
            const ax = nodePos[0] + r.x;
            const ay = nodePos[1] + r.y;
            if (e.canvasX >= ax && e.canvasX <= ax + r.w && e.canvasY >= ay && e.canvasY <= ay + r.h) {
                e.preventDefault(); e.stopPropagation();
                showTextareaOverlay(this, items[i] || "", (val) => {
                    const newItems = getItems(this);
                    if (i < newItems.length) newItems[i] = String(val ?? "");
                    setItems(this, newItems);
                    app.graph.setDirtyCanvas(true, true);
                });
                return true;
            }
        }
        return false;
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
            // 初次无数据时也同步
            setItems(this, getItems(this));
        };

        const origConfigure = nodeType.prototype.configure;
        nodeType.prototype.configure = function(info) {
            if (origConfigure) origConfigure.apply(this, arguments);
            ensureStringsJsonWidget(this);
            installAddButton(this);
            installDrawingHandlers(this);
            // 保持 properties._strings 为主
            if (info && info.properties && typeof info.properties._strings === 'string') {
                this.properties = this.properties || {};
                this.properties._strings = info.properties._strings;
                const hidden = ensureStringsJsonWidget(this);
                hidden.value = this.properties._strings;
            }
            // 同步一次
            setItems(this, getItems(this));
        };
    },
}); 