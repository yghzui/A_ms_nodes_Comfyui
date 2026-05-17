import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";
import { rgthree } from "./core/rgthree.js"; // 统一右键菜单定位使用的事件来源
import { modal } from "./utils/modal.js";
import { showTopNotification } from "./utils/shared_utils.js";
import { createTextInputBatchActionsApi } from "./text_input_batch/text_input_batch_actions.js";
import { createTextInputBatchMenuApi } from "./text_input_batch/text_input_batch_menu.js";
import { createTextInputBatchRenderCoreApi } from "./text_input_batch/text_input_batch_render_core.js";
import { createTextInputBatchLifecycleApi } from "./text_input_batch/text_input_batch_lifecycle.js";

let layoutCells;
let ensureTextareas;

console.log("Patching node: text_input_batch.js");

// {{ AURA-X: Add - Tooltip工具类，用于显示悬浮提示. }}
const Tooltip = {
    _el: null,
    _timer: null,
    _delay: 500, // 延迟显示时间（毫秒）

    get el() {
        if (!this._el) {
            this._el = document.createElement('div');
            this._el.style.cssText = `
                position: absolute;
                display: none;
                background: white;
                color: black;
                border: 1px solid #ccc;
                padding: 10px;
                z-index: 999999;
                font-family: "Microsoft YaHei", sans-serif;
                font-size: 14px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.2);
                pointer-events: none;
                white-space: pre-wrap;
                max-width: 800px;
                line-height: 1.5;
                border-radius: 6px;
                text-align: left;
            `;
            document.body.appendChild(this._el);
        }
        return this._el;
    },
    show(x, y, title, content) {
        const el = this.el;
        el.innerHTML = `<div style="font-weight:bold; margin-bottom:5px; border-bottom:1px solid #eee; padding-bottom:5px;">${title}</div><div>${content}</div>`;
        el.style.display = 'block';
        el.style.left = (x + 15) + 'px'; // 鼠标右侧显示
        el.style.top = y + 'px';
        
        // 简单的边界检查，防止溢出屏幕右侧和底部
        const rect = el.getBoundingClientRect();
        if (rect.right > window.innerWidth) {
            el.style.left = (window.innerWidth - rect.width - 10) + 'px';
        }
        if (rect.bottom > window.innerHeight) {
            // 如果底部溢出，尝试向上显示
            el.style.top = (y - rect.height) + 'px';
        }
    },
    scheduleShow(x, y, title, content) {
        this.cancelTimer();
        this._timer = setTimeout(() => {
            this.show(x, y, title, content);
        }, this._delay);
    },
    cancelTimer() {
        if (this._timer) {
            clearTimeout(this._timer);
            this._timer = null;
        }
    },
    hide() {
        this.cancelTimer();
        if (this._el) this._el.style.display = 'none';
    }
};

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
// {{ AURA-X: Modify - 更新getItems函数，支持读取enabled状态，默认为true，且强制当前选中项为true. }}
function getItems(node) {
    try {
        const widget = node.widgets.find(w => w.name === "strings_json");
        if (!widget) return [];
        
        const arr = JSON.parse(widget.value || "[]");
        const currentIndex = getCurrentIndex(node); // 获取当前选中索引
        
        // 确保至少有一项
        if (arr.length === 0) {
            arr.push({ title: "prompt_0", content: "", enabled: true });
        }
        
        return arr.map((item, index) => {
            if (typeof item === 'object' && item !== null && 'title' in item && 'content' in item) {
                // 新格式数据
                // 移除强制当前选中项为true的逻辑，保持原始enabled状态
                return {
                    title: String(item.title || `prompt_${index}`),
                    content: String(item.content || ""),
                    enabled: item.enabled !== false
                };
            } else {
                // 旧格式数据，转换为新格式
                const isSelected = index === currentIndex;
                return {
                    title: `prompt_${index}`,
                    content: String(item || ""),
                    enabled: true // 旧数据默认全部启用
                };
            }
        });
    } catch (e) {
        return [];
    }
}

// {{ AURA-X: Add - 标题处理工具函数 }}
function getBaseTitle(title) {
    const str = String(title || "");
    const match = str.match(/^(.*)_\d+$/);
    return match ? match[1] : str;
}

function normalizeTitle(title, index) {
    const base = getBaseTitle(title);
    return `${base}_${index}`;
}

// {{ AURA-X: Modify - 更新setItems函数，保存enabled状态. }}
function setItems(node, arr) {
    const currentIndex = getCurrentIndex(node); // 获取当前选中索引
    
    // 确保数组中的每个项目都是正确的格式，并强制标题格式
    const formattedArr = arr.map((item, index) => {
        // 强制标题格式：base_index
        let title = "";
        let content = "";
        let enabled = true;
        
        if (typeof item === 'object' && item !== null) {
            title = normalizeTitle(item.title || `prompt`, index);
            content = String(item.content || "");
            enabled = item.enabled !== false; // 移除强制逻辑
        } else {
            title = `prompt_${index}`;
            content = String(item || "");
            enabled = true;
        }
        
        return {
            title,
            content,
            enabled
        };
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
    
    // 更新卡片容器样式
    if (node.__cardEls) {
        node.__cardEls.forEach((cardEl, index) => {
            if (cardEl && cardEl.style) {
                if (index === currentIndex) {
                    cardEl.style.border = '2px solid #4a9eff';
                    cardEl.style.background = 'rgba(74, 158, 255, 0.1)';
                    cardEl.style.boxShadow = '0 0 12px rgba(74, 158, 255, 0.4)';
                } else {
                    cardEl.style.border = '1px solid #555';
                    cardEl.style.background = 'rgba(45, 45, 45, 0.95)';
                    cardEl.style.boxShadow = '0 2px 8px rgba(0,0,0,0.3)';
                }
            }
        });
    }
    
    // 更新Index标签样式
    if (node.__suffixEls) {
        node.__suffixEls.forEach((suffixEl, index) => {
            if (suffixEl && suffixEl.style) {
                suffixEl.style.background = index === currentIndex ? '#4a9eff' : '#555';
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

function bindMinHeightChange(node) {
    const w = node.widgets?.find(w => w.name === "cell_min_height");
    if (!w) return;
    
    // 确保值在有效范围内
    const initialNum = Number(w.value);
    if (!Number.isFinite(initialNum)) {
        const parsed = parseInt(String(w.value), 10);
        w.value = Number.isFinite(parsed) ? Math.max(72, Math.min(300, Math.floor(parsed))) : 120;
    } else {
        w.value = Math.max(72, Math.min(300, Math.floor(initialNum)));
    }
    if (w.__minHeightCbInstalled) return;
    const orig = w.callback;
    w.callback = (v) => {
        const num = Number(v);
        const base = Number.isFinite(num) ? num : parseInt(String(v), 10);
        const val = Number.isFinite(base) ? Math.max(72, Math.min(300, Math.floor(base))) : 120;
        w.value = val;
        if (orig) { try { orig(val); } catch(e) {} }
        const items = getItems(node);
        const cells = layoutCells(node, items);
        ensureTextareas(node, cells, items);
        app.graph.setDirtyCanvas(true, true);
        return true;
    };
    w.__minHeightCbInstalled = true;
}

function installSelectionTools(node) {
    if (node.__selectionToolsInstalled) return;
    
    // 统一更新函数
    const updateAll = (newItems) => {
        setItems(node, newItems);
        // 需要更新布局和文本框样式以反映启用状态
        const layout = layoutCells(node, newItems);
        ensureTextareas(node, layout, newItems);
        app.graph.setDirtyCanvas(true, true);
    };

    const selectAllBtn = node.addWidget("button", "全选", null, () => {
        const items = getItems(node);
        items.forEach(item => item.enabled = true);
        updateAll(items);
    });
    selectAllBtn.options.serialize = false;

    const deselectAllBtn = node.addWidget("button", "全不选", null, () => {
        const items = getItems(node);
        items.forEach(item => item.enabled = false);
        updateAll(items);
    });
    deselectAllBtn.options.serialize = false;

    const invertBtn = node.addWidget("button", "反选", null, () => {
        const items = getItems(node);
        items.forEach(item => item.enabled = !item.enabled);
        updateAll(items);
    });
    invertBtn.options.serialize = false;
    
    node.__selectionToolsInstalled = true;
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
        
        // 自动选中新添加的项 (如果是 combo 模式，这很有用)
        setIndexSelectorValue(node, newIndex);

        const layout = layoutCells(node, items);
        ensureTextareas(node, layout, items);
        app.graph.setDirtyCanvas(true, true);
        return true;
    });
    addBtn.options.serialize = false;

    node.__addButtonInstalled = true;
}

// 移除 installExtraButtons



function isHandMode() {
    const canvasWrapper = app?.canvas;
    const canvasEl = canvasWrapper?.canvas;
    if (!canvasEl) return false;
    let cursor = canvasEl.style?.cursor;
    if (!cursor && window.getComputedStyle) {
        try {
            cursor = window.getComputedStyle(canvasEl).cursor;
        } catch(e) {
            cursor = "";
        }
    }
    if (!cursor) return false;
    cursor = String(cursor).toLowerCase();
    return cursor.includes("grab");
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

const actionsApi = createTextInputBatchActionsApi({
    app,
    api,
    modal,
    showTopNotification,
    getItems,
    setItems,
    getBaseTitle,
    setIndexSelectorValue,
    getLayoutCells: () => layoutCells,
    getEnsureTextareas: () => ensureTextareas
});

const menuApi = createTextInputBatchMenuApi({
    app,
    rgthree,
    showTopNotification,
    getItems,
    setItems,
    getCurrentIndex,
    setIndexSelectorValue,
    updateTextareaStyles,
    moveItem,
    fetchPinyinMatches: actionsApi.fetchPinyinMatches,
    openAddToAssetManagerGroupPicker: actionsApi.openAddToAssetManagerGroupPicker
});

const renderCoreApi = createTextInputBatchRenderCoreApi({
    app,
    Tooltip,
    getItems,
    setItems,
    getBaseTitle,
    getCurrentIndex,
    updateTextareaStyles,
    isHandMode,
    getWidgetsBottom,
    handleTextareaCommentShortcut: menuApi.handleTextareaCommentShortcut,
    handleTextareaWheel: menuApi.handleTextareaWheel,
    showItemContextMenu: menuApi.showItemContextMenu,
    showCustomDropdown: menuApi.showCustomDropdown,
    updateCustomDropdownPosition: menuApi.updateCustomDropdownPosition,
    getEnsureTextareas: () => ensureTextareas
});

layoutCells = renderCoreApi.layoutCells;
ensureTextareas = renderCoreApi.ensureTextareas;

const lifecycleApi = createTextInputBatchLifecycleApi({
    app,
    Tooltip,
    ensureStringsJsonWidget,
    getItems,
    getCurrentIndex,
    setItems,
    updateTextareaStyles,
    bindColumnsChange,
    bindMinHeightChange,
    installAddButton,
    getLayoutCells: () => layoutCells,
    getEnsureTextareas: () => ensureTextareas,
    handleImport: actionsApi.handleImport,
    handleExport: actionsApi.handleExport,
    handleBatchDelete: actionsApi.handleBatchDelete
});

ensureTextareas = lifecycleApi.ensureTextareas;
