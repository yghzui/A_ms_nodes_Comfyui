import { app } from "../../../scripts/app.js";
import { rgthree } from "./rgthree.js"; // 统一右键菜单定位使用的事件来源

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

// {{ AURA-X: Modify - 更新数据结构支持标题+内容格式，兼容旧版数据. }}
// {{ AURA-X: Modify - 更新getItems函数，确保启用状态只有当前选中索引对应项目为true，其他为false. }}
function getItems(node) {
    try {
        const widget = node.widgets.find(w => w.name === "strings_json");
        if (!widget) return [];
        
        const arr = JSON.parse(widget.value || "[]");
        const currentIndex = getCurrentIndex(node); // 获取当前选中索引
        
        return arr.map((item, index) => {
            if (typeof item === 'object' && item !== null && 'title' in item && 'content' in item) {
                // 新格式数据，根据当前索引设置启用状态
                return {
                    title: String(item.title || `prompt_${index}`),
                    content: String(item.content || ""),
                    enabled: index === currentIndex // 只有当前选中索引的项目启用
                };
            } else {
                // 旧格式数据，自动转换，根据当前索引设置启用状态
                return {
                    title: `prompt_${index}`,
                    content: String(item || ""),
                    enabled: index === currentIndex // 只有当前选中索引的项目启用
                };
            }
        });
    } catch (e) {
        return [];
    }
}

// {{ AURA-X: Modify - 更新setItems函数，在保存数据时根据当前选中索引设置启用状态. }}
function setItems(node, arr) {
    const currentIndex = getCurrentIndex(node); // 获取当前选中索引
    
    // 确保数组中的每个项目都是正确的格式，并设置启用状态
    const formattedArr = arr.map((item, index) => {
        if (typeof item === 'object' && item !== null) {
            return {
                title: String(item.title || `prompt_${index}`),
                content: String(item.content || ""),
                enabled: index === currentIndex // 只有当前选中索引的项目启用
            };
        } else {
            // 兼容旧格式
            return {
                title: `prompt_${index}`,
                content: String(item || ""),
                enabled: index === currentIndex // 只有当前选中索引的项目启用
            };
        }
    });
    
    const json = JSON.stringify(formattedArr);
    const hidden = ensureStringsJsonWidget(node);
    hidden.value = json;
    node.properties = node.properties || {};
    node.properties._strings = json;
}

// 获取当前选中的索引值 - 使用节点自身的index widget
function getCurrentIndex(node) {
    // 查找节点自身的 index widget
    const indexWidget = node.widgets?.find(w => w.name === "index");
    return indexWidget ? indexWidget.value : 0;
}

// 设置节点自身的索引值
function setIndexSelectorValue(node, index) {
    // 查找节点自身的 index widget
    const indexWidget = node.widgets?.find(w => w.name === "index");
    if (indexWidget) {
        // 确保索引值在有效范围内
        const items = getItems(node);
        const maxIndex = Math.max(0, items.length - 1);
        indexWidget.value = Math.max(0, Math.min(index, maxIndex));
        
        // 触发节点更新
        if (node.onWidgetChanged) {
            node.onWidgetChanged("index", indexWidget.value, indexWidget.value, indexWidget);
        }
        app.graph.setDirtyCanvas(true, true);
        return true;
    }
    return false;
}

// 更新文本框样式，高亮当前选中的索引
function updateTextareaStyles(node) {
    if (!node.__taEls) return;
    const currentIndex = getCurrentIndex(node);
    
    node.__taEls.forEach((ta, index) => {
        if (ta && ta.style) {
            if (index === currentIndex) {
                // 当前选中的内容框显示蓝色边框，保持下圆角
                ta.style.border = '2px solid #4a9eff';
                ta.style.borderTop = 'none'; // 保持与标题框的连接
                ta.style.borderRadius = '0 0 6px 6px';
                ta.style.boxShadow = '0 0 8px rgba(74, 158, 255, 0.3)';
            } else {
                // 其他内容框恢复默认样式
                ta.style.border = '1px solid #666';
                ta.style.borderTop = 'none';
                ta.style.borderRadius = '0 0 6px 6px';
                ta.style.boxShadow = 'none';
            }
        }
    });
    
    // 同样更新标题输入框的样式，确保整体性
    if (node.__titleEls) {
        node.__titleEls.forEach((titleEl, index) => {
            if (titleEl && titleEl.style) {
                if (index === currentIndex) {
                    // 当前选中的标题框显示蓝色边框，保持上圆角
                    titleEl.style.border = '2px solid #4a9eff';
                    titleEl.style.borderBottom = 'none'; // 保持与内容框的连接
                    titleEl.style.borderRadius = '6px 6px 0 0';
                    titleEl.style.boxShadow = '0 0 8px rgba(74, 158, 255, 0.3)';
                } else {
                    // 其他标题框恢复默认样式
                    titleEl.style.border = '1px solid #666';
                    titleEl.style.borderBottom = 'none';
                    titleEl.style.borderRadius = '6px 6px 0 0';
                    titleEl.style.boxShadow = 'none';
                }
            }
        });
    }
}

function bindColumnsChange(node) {
    const w = node.widgets?.find(w => w.name === "columns");
    if (!w) return;
    const initialNum = Number(w.value);
    if (!Number.isFinite(initialNum)) {
        const parsed = parseInt(String(w.value), 10);
        w.value = Number.isFinite(parsed) ? Math.max(1, Math.min(8, Math.floor(parsed))) : 2;
    } else {
        w.value = Math.max(1, Math.min(8, Math.floor(initialNum)));
    }
    if (w.__columnsCbInstalled) return;
    const orig = w.callback;
    w.callback = (v) => {
        const num = Number(v);
        const base = Number.isFinite(num) ? num : parseInt(String(v), 10);
        const val = Number.isFinite(base) ? Math.max(1, Math.min(8, Math.floor(base))) : 2;
        w.value = val;
        if (orig) { try { orig(val); } catch(e) {} }
        const items = getItems(node);
        const cells = layoutCells(node, items);
        ensureTextareas(node, cells, items);
        app.graph.setDirtyCanvas(true, true);
        return true;
    };
    w.__columnsCbInstalled = true;
}

function installAddButton(node) {
    if (node.__addButtonInstalled) return;
    const addBtn = node.addWidget("button", "➕ 添加字符串", null, () => {
        const items = getItems(node);
        const newIndex = items.length;
        items.push({
            title: `prompt_${newIndex}`,
            content: ""
        });
        setItems(node, items);
        const layout = layoutCells(node, items);
        ensureTextareas(node, layout, items);
        app.graph.setDirtyCanvas(true, true);
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

// {{ AURA-X: Modify - 更新右键菜单功能，适配新的数据结构. }}
function showItemContextMenu(node, index, event) {
    const items = getItems(node);
    const n = items.length;
    const hasUp = index > 0;
    const hasDown = index < n - 1;
    const Lite = window.LiteGraph || window?.app?.canvas?.graph?.constructor;

    const doDelete = () => {
        const next = items.slice(0, index).concat(items.slice(index + 1));
        setItems(node, next);
        if (node.ensureTextareas) node.ensureTextareas();
        app.graph.setDirtyCanvas(true, true);
    };
    const doMoveUp = () => {
        if (!hasUp) return;
        const next = moveItem(items, index, index - 1);
        setItems(node, next);
        if (node.ensureTextareas) node.ensureTextareas();
        app.graph.setDirtyCanvas(true, true);
    };
    const doMoveDown = () => {
        if (!hasDown) return;
        const next = moveItem(items, index, index + 1);
        setItems(node, next);
        if (node.ensureTextareas) node.ensureTextareas();
        app.graph.setDirtyCanvas(true, true);
    };
    const doMoveTo = () => {
        let to = prompt(`移动到索引 (0 - ${Math.max(0, n - 1)}):`, String(index));
        if (to == null) return;
        to = Number(to);
        if (!Number.isFinite(to)) return;
        const next = moveItem(items, index, to);
        setItems(node, next);
        if (node.ensureTextareas) node.ensureTextareas();
        app.graph.setDirtyCanvas(true, true);
    };

    // 清空/复制/粘贴功能
    const doClear = () => {
        const arr = getItems(node);
        if (index < arr.length) {
            arr[index].content = "";
            setItems(node, arr);
            const ta = node.__taEls?.[index];
            if (ta) ta.value = "";
            app.graph.setDirtyCanvas(true, true);
        }
    };
    const doCopy = async () => {
        try {
            const item = getItems(node)[index];
            const value = item ? item.content : "";
            if (navigator.clipboard?.writeText) {
                await navigator.clipboard.writeText(value);
            } else {
                const tmp = document.createElement('textarea');
                tmp.value = value;
                document.body.appendChild(tmp);
                tmp.select();
                document.execCommand('copy');
                tmp.remove();
            }
        } catch (e) {
            const item = getItems(node)[index];
            prompt('复制失败，请手动复制:', item ? item.content : "");
        }
    };
    const doPaste = async () => {
        let text = "";
        try {
            if (navigator.clipboard?.readText) {
                text = await navigator.clipboard.readText();
            } else {
                text = prompt('粘贴文本:', "") || "";
            }
        } catch (e) {
            text = prompt('粘贴文本:', "") || "";
        }
        const arr = getItems(node);
        if (index < arr.length) {
            arr[index].content = text;
            setItems(node, arr);
            const ta = node.__taEls?.[index];
            if (ta) ta.value = text;
            app.graph.setDirtyCanvas(true, true);
        }
    };

    // 使用该提示词功能
    const doUseThisPrompt = () => {
        const success = setIndexSelectorValue(node, index);
        if (success) {
            setTimeout(() => updateTextareaStyles(node), 50);
        } else {
            alert('未找到节点的 index 控件');
        }
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
            { content: `✨ 使用该提示词`, callback: doUseThisPrompt },
            null, // 分隔线
            { content: `🧹 清空内容`, callback: doClear },
            { content: `📋 复制`, callback: doCopy },
            { content: `📥 粘贴`, callback: doPaste },
            { content: `🗑️ 删除`, callback: doDelete },
            { content: `⬆️ 上移`, disabled: !hasUp, callback: doMoveUp },
            { content: `⬇️ 下移`, disabled: !hasDown, callback: doMoveDown },
            { content: `↔ 移动到索引…`, callback: doMoveTo },
        ];
        const useEvent = rgthree.lastContextMenuEvent || event;
        const cm = new Lite.ContextMenu(menu, {
            event: useEvent,
            title: `文本 ${index+1}`,
            className: "dark",
            scale: Math.max(1, app?.canvas?.ds?.scale || 1),
        });
        try {
            const root = cm.root || cm.element || cm.menu || cm;
            if (root && root.style) root.style.zIndex = '10050';
        } catch(e) {}
        setTimeout(() => {
            const once = () => { document.removeEventListener('mousedown', once, true); restorePointer(); };
            document.addEventListener('mousedown', once, true);
        }, 0);
    } else {
        // 简易回退
        const choice = prompt(`操作: u=使用该提示词, c=清空, y=复制, p=粘贴, d=删除, up=上移, n=下移, m=移动到索引`, "u");
        if (choice === 'u') doUseThisPrompt();
        else if (choice === 'c') doClear();
        else if (choice === 'y') doCopy();
        else if (choice === 'p') doPaste();
        else if (choice === 'd') doDelete();
        else if (choice === 'up') doMoveUp();
        else if (choice === 'n') doMoveDown();
        else if (choice === 'm') doMoveTo();
        restorePointer();
    }
}

// {{ AURA-X: Add - 创建标题和内容的UI元素，在输入框上方添加标题输入. }}
function ensureTextareas(node, layout, items) {
    const ds = app?.canvas?.ds;
    const canvas = app?.canvas?.canvas;
    if (!ds || !canvas) return;
    const rect = canvas.getBoundingClientRect();

    // 初始化元素数组
    if (!node.__taEls) node.__taEls = [];
    if (!node.__titleEls) node.__titleEls = [];

    const currentIndex = getCurrentIndex(node);

    for (let i = 0; i < items.length; i++) {
        const cell = layout[i];
        if (!cell) continue;
        
        const item = items[i];
        const isSelected = i === currentIndex;
        
        // 创建或更新标题输入框
        let titleEl = node.__titleEls[i];
        if (!titleEl) {
            titleEl = document.createElement('input');
            titleEl.type = 'text';
            titleEl.placeholder = `标题 ${i+1}`;
            titleEl.value = item.title || `prompt_${i}`;
            titleEl.style.cssText = `position: fixed; z-index: 1; padding: 4px 6px; border-radius: 6px 6px 0 0; border: 1px solid #666; border-bottom: none; background: #3a3a3a; color: #eee; font: 11px/1.2 monospace; box-sizing: border-box;`;
            
            // 标题输入框事件处理
            titleEl.addEventListener('input', () => {
                const arr = getItems(node);
                if (i < arr.length) {
                    arr[i].title = titleEl.value || `prompt_${i}`;
                    setItems(node, arr);
                }
            });
            
            // 支持Enter键确认，Escape键取消
            titleEl.addEventListener('keydown', (e) => {
                if (e.key === 'Enter') {
                    titleEl.blur();
                } else if (e.key === 'Escape') {
                    titleEl.value = item.title || `prompt_${i}`;
                    titleEl.blur();
                }
            });
            
            document.body.appendChild(titleEl);
            node.__titleEls[i] = titleEl;
        } else {
            // 更新现有标题输入框的值
            titleEl.value = item.title || `prompt_${i}`;
        }

        // 创建或更新内容文本框
        let ta = node.__taEls[i];
        if (!ta) {
            ta = document.createElement('textarea');
            ta.placeholder = `内容 ${i+1}`;
            ta.spellcheck = false;
            ta.wrap = 'soft';
            ta.value = item.content || "";
            ta.style.cssText = `position: fixed; z-index: 1; resize: none; padding: 6px; border-radius: 0 0 6px 6px; border: 1px solid #666; border-top: none; background: #222; color: #eee; font: 12px/1.4 monospace; box-sizing: border-box; overflow: auto;`;
            
            // 内容文本框事件处理
            ta.addEventListener('input', () => {
                const arr = getItems(node);
                if (i < arr.length) {
                    arr[i].content = ta.value;
                    setItems(node, arr);
                }
            });
            
            // 右键菜单
            if (!ta.__ctxInstalled) {
                ta.addEventListener('contextmenu', (e) => {
                    e.preventDefault(); 
                    e.stopPropagation();
                    showItemContextMenu(node, i, e);
                });
                ta.__ctxInstalled = true;
            }
            
            document.body.appendChild(ta);
            node.__taEls[i] = ta;
        } else {
            // 更新现有textarea的值
            ta.value = item.content || "";
        }

        // 计算位置和大小
        const sx = (node.pos[0] + cell.x + ds.offset[0]) * ds.scale + rect.left;
        const sy = (node.pos[1] + cell.y + ds.offset[1]) * ds.scale + rect.top;
        const sw = cell.w * ds.scale;
        const sh = cell.h * ds.scale;
        
        // 标题输入框位置（在内容框上方）
        const titleHeight = 24;
        // 设置标题输入框位置和大小
        titleEl.style.left = `${Math.round(sx)}px`;
        titleEl.style.top = `${Math.round(sy)}px`;
        titleEl.style.width = `${Math.max(40, Math.round(sw))}px`;
        titleEl.style.height = `${Math.round(titleHeight)}px`;
        
        // 设置内容文本框位置和大小 - 移除间距，紧密连接
        ta.style.left = `${Math.round(sx)}px`;
        ta.style.top = `${Math.round(sy + titleHeight)}px`;
        ta.style.width = `${Math.max(40, Math.round(sw))}px`;
        ta.style.height = `${Math.max(32, Math.round(sh - titleHeight))}px`;
        
        // 字体大小缩放
        const fontPx = Math.max(10, Math.round(12 * (ds.scale || 1)));
        const titleFontPx = Math.max(9, Math.round(11 * (ds.scale || 1)));
        titleEl.style.fontSize = `${titleFontPx}px`;
        ta.style.fontSize = `${fontPx}px`;
        
        // 设置可见性 - 显示所有项目，但只有在节点未折叠时
        const shouldShow = node.flags?.collapsed !== true;
        titleEl.style.visibility = shouldShow ? 'visible' : 'hidden';
        ta.style.visibility = shouldShow ? 'visible' : 'hidden';
    }

    // 清理多余的元素
    for (let j = items.length; j < (node.__taEls?.length || 0); j++) {
        const el = node.__taEls[j];
        if (el && el.remove) el.remove();
    }
    for (let j = items.length; j < (node.__titleEls?.length || 0); j++) {
        const el = node.__titleEls[j];
        if (el && el.remove) el.remove();
    }
    
    node.__taEls.length = items.length;
    node.__titleEls.length = items.length;
    
    // 更新样式以反映当前选中的索引
    updateTextareaStyles(node);
}

// {{ AURA-X: Add - 计算布局单元格位置，为标题+内容预留更多垂直空间. }}
function layoutCells(node, items) {
    const PADDING = 8;
    const GAP = 6;
    const MIN_H = 72;
    const n = items.length;
    if (n === 0) return [];

    let cols = 2;
    const wCols = node.widgets?.find(w => w.name === "columns");
    if (wCols) {
        const v = wCols.value;
        const num = Number(v);
        if (Number.isFinite(num)) {
            cols = num;
        } else {
            const parsed = parseInt(String(v), 10);
            cols = Number.isFinite(parsed) ? parsed : 2;
        }
    }
    cols = Math.floor(Math.max(1, Math.min(8, cols)));
    const rows = Math.ceil(n / cols);
    const availW = node.size[0] - PADDING * 2;
    const cellW = Math.floor((availW - GAP * (cols - 1)) / cols);
    const startY = PADDING + getWidgetsBottom(node);

    const requiredH = rows * MIN_H + GAP * Math.max(0, rows - 1);
    const minTotalH = startY + requiredH + PADDING;
    if (node.size[1] < minTotalH) {
        if (typeof node.setSize === 'function') {
            node.setSize([node.size[0], minTotalH]);
        } else {
            node.size[1] = minTotalH;
        }
        app.graph.setDirtyCanvas(true, true);
    }

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
        // 清理标题输入框
        if (this.__titleEls) {
            for (const el of this.__titleEls) { try { el.remove(); } catch(e) {} }
            this.__titleEls = [];
        }
    };
}

// 监听图形变化，当连接的 IndexSelector 节点的索引值改变时更新样式
// {{ AURA-X: Modify - 修改索引变化监听器，确保在索引变化时重新保存数据以更新启用状态. }}
function installIndexChangeListener(node) {
    if (node.__indexListenerInstalled) return;
    node.__indexListenerInstalled = true;

    // 定期检查索引值变化
    let lastIndex = -1;
    const checkIndexChange = () => {
        const currentIndex = getCurrentIndex(node);
        if (currentIndex !== lastIndex) {
            lastIndex = currentIndex;
            updateTextareaStyles(node);
            
            // 重新保存数据以更新启用状态
            const items = getItems(node);
            setItems(node, items);
        }
    };

    // 使用定时器定期检查（更轻量级的方式）
    node.__indexCheckInterval = setInterval(checkIndexChange, 100);

    // 在节点移除时清理定时器
    const origRemoved = node.onRemoved;
    node.onRemoved = function() {
        if (origRemoved) origRemoved.call(this);
        if (this.__indexCheckInterval) {
            clearInterval(this.__indexCheckInterval);
            this.__indexCheckInterval = null;
        }
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
            installIndexChangeListener(this);
            bindColumnsChange(this);
            setItems(this, getItems(this));
        };

        const origConfigure = nodeType.prototype.configure;
        nodeType.prototype.configure = function(info) {
            if (origConfigure) origConfigure.apply(this, arguments);
            ensureStringsJsonWidget(this);
            installAddButton(this);
            installDrawingHandlers(this);
            installIndexChangeListener(this);
            bindColumnsChange(this);
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