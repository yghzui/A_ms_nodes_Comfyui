import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";
import { rgthree } from "./core/rgthree.js"; // 统一右键菜单定位使用的事件来源
import { modal } from "./utils/modal.js";
import { showTopNotification } from "./utils/shared_utils.js";

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


function handleExport(node) {
    const items = getItems(node);
    modal.show({
        title: "导出提示词",
        content: "请选择导出方式：",
        buttons: [
            {
                text: "导出为 JSON 文件",
                type: "primary",
                onClick: () => {
                    const blob = new Blob([JSON.stringify(items, null, 2)], { type: "application/json" });
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement("a");
                    a.href = url;
                    a.download = "prompts_export.json";
                    a.click();
                    URL.revokeObjectURL(url);
                    modal.close();
                }
            },
            {
                text: "复制到剪贴板",
                type: "secondary",
                onClick: () => {
                    navigator.clipboard.writeText(JSON.stringify(items, null, 2))
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
            <label>粘贴内容 (JSON 或 文本):</label>
            <textarea id="import-text" class="custom-modal-textarea" placeholder="在此粘贴..."></textarea>
            <label>或 选择文件:</label>
            <input type="file" id="import-file" accept=".json,.txt" class="custom-modal-file-input">
        </div>
    `;

    modal.show({
        title: "导入提示词 (追加模式)",
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
    let newItems = [];
    try {
        // 尝试 JSON 解析
        const parsed = JSON.parse(rawData);
        if (Array.isArray(parsed)) {
            newItems = parsed.map((item, idx) => {
                const content = (typeof item === 'object' && item !== null && 'content' in item) ? item.content : 
                                (typeof item === 'string' ? item : JSON.stringify(item));
                const title = (typeof item === 'object' && item !== null && 'title' in item) ? item.title : `imported_${idx}`;
                const enabled = (typeof item === 'object' && item !== null && 'enabled' in item) ? item.enabled !== false : true;
                
                return {
                    title: String(title),
                    content: String(content),
                    enabled: enabled
                };
            });
        } else if (typeof parsed === 'object' && parsed !== null) {
             // 单个对象
             newItems.push({
                title: String(parsed.title || "imported"),
                content: String(parsed.content || JSON.stringify(parsed)),
                enabled: parsed.enabled !== false
             });
        }
    } catch (e) {
        // 非 JSON，按行分割
        const lines = rawData.split(/\n/);
        newItems = lines.filter(line => line.trim()).map((line, idx) => ({
            title: `line_${idx}`,
            content: line.trim(),
            enabled: true
        }));
    }

    if (newItems.length > 0) {
        const currentItems = getItems(node);
        // 追加模式
        const merged = currentItems.concat(newItems);
        setItems(node, merged);
        
        // 刷新 UI
        const layout = layoutCells(node, merged);
        ensureTextareas(node, layout, merged);
        app.graph.setDirtyCanvas(true, true);
        showTopNotification(`成功导入 ${newItems.length} 条提示词`, "success");
    } else {
        showTopNotification("未识别到有效的提示词数据，请检查格式。支持 JSON 数组或每行一条文本。", "error");
    }
}

function handleBatchDelete(node) {
    const items = getItems(node);
    const toDeleteCount = items.filter(i => i.enabled).length;
    
    if (toDeleteCount === 0) {
        showTopNotification("没有选中的提示词 (Enabled = true)", "warning");
        return;
    }

    modal.show({
        title: "批量删除确认",
        content: `确定要删除 ${toDeleteCount} 个选中的提示词吗？此操作不可撤销。`,
        buttons: [
            {
                text: "确认删除",
                type: "danger",
                onClick: () => {
                    // 保留未选中的 (即 enabled == false 的)
                    let remaining = items.filter(i => !i.enabled);
                    
                    // 至少保留一项
                    if (remaining.length === 0) {
                        remaining.push({ title: "prompt_0", content: "", enabled: true });
                    }
                    
                    setItems(node, remaining);
                    setIndexSelectorValue(node, 0);
                    
                    const layout = layoutCells(node, remaining);
                    ensureTextareas(node, layout, remaining);
                    app.graph.setDirtyCanvas(true, true);
                    
                    modal.close();
                }
            },
            { text: "取消", onClick: () => modal.close() }
        ]
    });
}

async function fetchAssetManagerCollection(endpoint) {
    const res = await api.fetchApi(endpoint);
    if (!res.ok) {
        throw new Error(`请求失败: ${res.status}`);
    }
    const data = await res.json();
    if (!data || typeof data !== "object") {
        return { groups: [] };
    }
    if (!Array.isArray(data.groups)) {
        data.groups = [];
    }
    return data;
}

async function saveAssetManagerCollection(endpoint, data) {
    const res = await api.fetchApi(endpoint, {
        method: "POST",
        body: JSON.stringify(data),
        headers: { "Content-Type": "application/json" }
    });
    if (!res.ok) {
        throw new Error(`保存失败: ${res.status}`);
    }
    const result = await res.json().catch(() => ({ success: true }));
    if (result && result.success === false) {
        throw new Error(result.error || "保存失败");
    }
    return result;
}

function buildLocalIncludesMatches(texts, keyword) {
    const normalized = String(keyword || "").trim().toLowerCase();
    if (!normalized) return texts.map(() => true);
    return texts.map(t => String(t || "").toLowerCase().includes(normalized));
}

async function fetchPinyinMatches(texts, keyword) {
    const normalized = String(keyword || "").trim().toLowerCase();
    if (!normalized) return texts.map(() => true);

    try {
        const res = await api.fetchApi("/a_my_nodes/assets/search_pinyin", {
            method: "POST",
            body: JSON.stringify({ texts, keyword: normalized }),
            headers: { "Content-Type": "application/json" }
        });
        if (!res.ok) {
            throw new Error(`Search API failed: ${res.status}`);
        }
        const data = await res.json();
        if (data && Array.isArray(data.matches) && data.matches.length === texts.length) {
            return data.matches.map(Boolean);
        }
    } catch (e) {
        console.warn("[TextInputBatch] search_pinyin failed, fallback to local search:", e);
    }

    return buildLocalIncludesMatches(texts, normalized);
}

function buildNewPromptAssetItem(item, index) {
    const baseTitle = getBaseTitle(item?.title || `prompt_${index}`).trim();
    return {
        id: `${Date.now()}_${Math.random().toString(16).slice(2, 8)}`,
        title: baseTitle || `prompt_${index}`,
        content: String(item?.content || ""),
        preview_image: ""
    };
}

function escapeHtml(value) {
    return String(value ?? "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#39;");
}

function ensureModelsGroupAlignment(modelsData, targetIndex, targetGroupName) {
    if (!Array.isArray(modelsData.groups)) {
        modelsData.groups = [];
    }
    while (modelsData.groups.length < targetIndex) {
        modelsData.groups.push({
            name: `未命名分组 ${modelsData.groups.length + 1}`,
            items: []
        });
    }
    if (!modelsData.groups[targetIndex]) {
        modelsData.groups[targetIndex] = {
            name: targetGroupName,
            items: []
        };
    } else if (!modelsData.groups[targetIndex].name) {
        modelsData.groups[targetIndex].name = targetGroupName;
    }
}

function getSearchRowDom({ placeholder = "🔍 搜索拼音/原名/首字母...", onChange }) {
    const wrap = document.createElement("div");
    wrap.style.position = "relative";
    wrap.style.display = "flex";
    wrap.style.alignItems = "center";

    const input = document.createElement("input");
    input.type = "text";
    input.placeholder = placeholder;
    input.style.width = "100%";
    input.style.boxSizing = "border-box";
    input.style.padding = "8px 26px 8px 10px";
    input.style.background = "#1a1a1a";
    input.style.border = "1px solid #444";
    input.style.color = "#eee";
    input.style.borderRadius = "4px";
    input.style.outline = "none";

    const clearBtn = document.createElement("span");
    clearBtn.textContent = "✖";
    clearBtn.style.position = "absolute";
    clearBtn.style.right = "8px";
    clearBtn.style.cursor = "pointer";
    clearBtn.style.color = "#aaa";
    clearBtn.style.fontSize = "12px";
    clearBtn.style.display = "none";
    clearBtn.onmouseenter = () => { clearBtn.style.color = "#fff"; };
    clearBtn.onmouseleave = () => { clearBtn.style.color = "#aaa"; };

    const setClearVisible = (visible) => {
        clearBtn.style.display = visible ? "block" : "none";
    };

    let composing = false;
    input.addEventListener("compositionstart", () => { composing = true; });
    input.addEventListener("compositionend", () => {
        composing = false;
        if (onChange) onChange(input.value);
    });

    input.addEventListener("input", () => {
        setClearVisible(!!input.value);
        if (composing) return;
        if (onChange) onChange(input.value);
    });

    clearBtn.addEventListener("click", (e) => {
        e.preventDefault();
        e.stopPropagation();
        input.value = "";
        setClearVisible(false);
        if (onChange) onChange("");
        input.focus();
    });

    wrap.appendChild(input);
    wrap.appendChild(clearBtn);
    return { wrap, input, clearBtn };
}

async function openAddToAssetManagerGroupPicker(node, index) {
    const items = getItems(node);
    const item = items[index];
    if (!item) {
        showTopNotification("未找到要添加的提示词", "error");
        return;
    }

    let promptsData;
    let modelsData;
    try {
        [promptsData, modelsData] = await Promise.all([
            fetchAssetManagerCollection("/a_my_nodes/assets/prompts"),
            fetchAssetManagerCollection("/a_my_nodes/assets/models")
        ]);
    } catch (e) {
        showTopNotification(`读取管理器数据失败: ${e.message || e}`, "error");
        return;
    }

    const promptItem = buildNewPromptAssetItem(item, index);

    const root = document.createElement("div");
    root.style.display = "flex";
    root.style.flexDirection = "column";
    root.style.gap = "10px";

    const tip = document.createElement("div");
    tip.style.color = "#bbb";
    tip.style.fontSize = "12px";
    tip.textContent = `选择要添加到的目录（提示词标题将使用: ${promptItem.title}）`;
    root.appendChild(tip);

    const listContainer = document.createElement("div");
    listContainer.style.border = "1px solid #333";
    listContainer.style.borderRadius = "6px";
    listContainer.style.overflow = "hidden";
    listContainer.style.maxHeight = "360px";
    listContainer.style.overflowY = "auto";

    const groupNames = promptsData.groups.map(g => String(g?.name || "").trim());
    const groupTexts = groupNames.map(n => n || "");

    let debounceTimer = null;
    let latestSeq = 0;
    let lastKeyword = "";

    const createRow = (label, onClick, opts = {}) => {
        const row = document.createElement("div");
        row.style.padding = "10px 12px";
        row.style.cursor = "pointer";
        row.style.borderBottom = "1px solid #2a2a2a";
        row.style.display = "flex";
        row.style.alignItems = "center";
        row.style.justifyContent = "space-between";
        row.style.background = opts.background || "transparent";
        row.style.color = opts.color || "#eee";
        row.onmouseenter = () => { row.style.background = "#333"; };
        row.onmouseleave = () => { row.style.background = opts.background || "transparent"; };

        const left = document.createElement("span");
        left.textContent = label;
        left.style.overflow = "hidden";
        left.style.textOverflow = "ellipsis";
        left.style.whiteSpace = "nowrap";
        row.appendChild(left);

        if (opts.suffix) {
            const right = document.createElement("span");
            right.textContent = opts.suffix;
            right.style.color = "#888";
            right.style.fontSize = "12px";
            row.appendChild(right);
        }

        row.addEventListener("click", (e) => {
            e.preventDefault();
            e.stopPropagation();
            if (onClick) onClick();
        });
        return row;
    };

    const tryAddToGroup = async (targetGroupName) => {
        const cleanName = String(targetGroupName || "").trim();
        if (!cleanName) {
            showTopNotification("请先选择目录或输入目录名", "warning");
            return;
        }

        let targetGroupIndex = promptsData.groups.findIndex(group => String(group?.name || "").trim() === cleanName);
        let shouldSaveModels = false;

        if (targetGroupIndex === -1) {
            promptsData.groups.push({ name: cleanName, items: [] });
            targetGroupIndex = promptsData.groups.length - 1;
            ensureModelsGroupAlignment(modelsData, targetGroupIndex, cleanName);
            shouldSaveModels = true;
        }

        const targetGroup = promptsData.groups[targetGroupIndex];
        if (!Array.isArray(targetGroup.items)) targetGroup.items = [];

        const duplicateExists = targetGroup.items.some(existingItem =>
            String(existingItem?.title || "").trim() === promptItem.title &&
            String(existingItem?.content || "") === promptItem.content
        );
        if (duplicateExists) {
            showTopNotification(`目录 [${cleanName}] 中已存在相同提示词`, "warning");
            return;
        }

        targetGroup.items.push({ ...promptItem });
        try {
            await saveAssetManagerCollection("/a_my_nodes/assets/prompts", promptsData);
            if (shouldSaveModels) {
                await saveAssetManagerCollection("/a_my_nodes/assets/models", modelsData);
            }
            modal.close();
            showTopNotification(`已添加到管理器目录: ${cleanName}`, "success");
        } catch (e) {
            showTopNotification(`保存到管理器失败: ${e.message || e}`, "error");
        }
    };

    const promptCreateGroup = async () => {
        const currentKeyword = String(searchInput?.value || "").trim();
        const groupName = prompt("请输入新分组名:", currentKeyword);
        if (groupName == null) return;
        await tryAddToGroup(groupName);
    };

    const renderGroups = (matches) => {
        listContainer.innerHTML = "";

        const keyword = String(lastKeyword || "").trim();
        const keywordHasValue = !!keyword;
        const hasExact = keywordHasValue && groupNames.some(n => String(n || "").trim() === keyword);

        if (keywordHasValue && !hasExact) {
            listContainer.appendChild(createRow(`➕ 新建分组: ${keyword}`, () => tryAddToGroup(keyword), {
                background: "rgba(42, 109, 181, 0.15)",
                color: "#e6edf3",
                suffix: "Enter"
            }));
        } else {
            listContainer.appendChild(createRow("➕ 新建分组...", () => promptCreateGroup(), {
                background: "rgba(42, 109, 181, 0.08)",
                color: "#e6edf3"
            }));
        }

        let shown = 0;
        for (let i = 0; i < promptsData.groups.length; i++) {
            const name = String(promptsData.groups[i]?.name || "").trim();
            if (!name) continue;
            if (matches && matches[i] === false) continue;
            listContainer.appendChild(createRow(name, () => tryAddToGroup(name), { suffix: `#${i + 1}` }));
            shown += 1;
        }

        if (shown === 0 && !(keywordHasValue && !hasExact)) {
            const empty = document.createElement("div");
            empty.style.padding = "12px";
            empty.style.color = "#888";
            empty.style.textAlign = "center";
            empty.textContent = "无匹配目录";
            listContainer.appendChild(empty);
        }
    };

    const scheduleSearch = (keyword, delay = 180) => {
        lastKeyword = keyword;
        if (debounceTimer) clearTimeout(debounceTimer);
        const seq = ++latestSeq;
        debounceTimer = setTimeout(async () => {
            debounceTimer = null;
            const matches = await fetchPinyinMatches(groupTexts, keyword);
            if (seq !== latestSeq) return; // 丢弃过期请求
            renderGroups(matches);
        }, delay);
    };

    const { wrap: searchWrap, input: searchInput } = getSearchRowDom({
        placeholder: "🔍 搜索目录(拼音/原名/首字母)...",
        onChange: (v) => scheduleSearch(v)
    });
    root.appendChild(searchWrap);
    root.appendChild(listContainer);

    // Enter: 如果输入框里有值，直接走“新建分组: xxx”
    searchInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter") {
            e.preventDefault();
            const keyword = String(searchInput.value || "").trim();
            if (keyword) tryAddToGroup(keyword);
        }
    });

    // 初始渲染
    renderGroups(null);

    modal.show({
        title: "选择管理器目录",
        content: root,
        width: "520px",
        buttons: [{ text: "关闭", onClick: () => modal.close() }]
    });
}

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
function showItemContextMenu(node, index, event) {
    const items = getItems(node);
    const n = items.length;
    const hasUp = index > 0;
    const hasDown = index < n - 1;
    const Lite = window.LiteGraph || window?.app?.canvas?.graph?.constructor;

    const doDelete = () => {
        // 至少保留一项
        if (items.length <= 1) {
            showTopNotification('至少需要保留一个提示词输入！', "warning");
            return;
        }

        const next = items.slice(0, index).concat(items.slice(index + 1));
        setItems(node, next);
        
        // 自动切换到下一项或上一项
        if (next.length > 0) {
            // 如果删除的是当前选中项，或者删除的项在当前选中项之前，需要调整索引
            const currentIndex = Number(getCurrentIndex(node));
            const targetIndex = Number(index);
            
            if (targetIndex === currentIndex) {
                // 删除的是当前选中项
                // 如果删除的是最后一项，选中新的最后一项（即原索引减一）
                // 如果删除的是中间项或第一项，索引不变（即选中了原来的下一项）
                
                if (targetIndex >= next.length) {
                    // 删除了最后一项，选中前一项
                    setIndexSelectorValue(node, Math.max(0, next.length - 1));
                } else {
                    // 删除了中间项或第一项，索引保持不变，即选中下一项
                    // 无需操作，因为索引没变，内容变了
                    setIndexSelectorValue(node, targetIndex);
                }
            } else if (targetIndex < currentIndex) {
                // 删除的项在当前选中项之前，当前选中项的索引需要减一
                setIndexSelectorValue(node, currentIndex - 1);
            }
            // 如果删除的项在当前选中项之后，当前选中项索引不变，无需处理
        } else {
             // 列表为空，重置索引为0
             setIndexSelectorValue(node, 0);
        }

        if (node.ensureTextareas) node.ensureTextareas();
        app.graph.setDirtyCanvas(true, true);
    };
    const doMoveUp = () => {
        if (!hasUp) return;
        const next = moveItem(items, index, index - 1);
        setItems(node, next);
        
        // 如果移动的是当前选中项，跟随移动
        const currentIndex = Number(getCurrentIndex(node));
        const targetIndex = Number(index);
        
        if (targetIndex === currentIndex) {
            setIndexSelectorValue(node, targetIndex - 1);
        } else if (targetIndex - 1 === currentIndex) {
            // 如果移动到了当前选中项的位置（即与选中项交换了位置），选中项索引需要调整
            setIndexSelectorValue(node, targetIndex);
        }
        
        if (node.ensureTextareas) node.ensureTextareas();
        app.graph.setDirtyCanvas(true, true);
    };
    const doMoveDown = () => {
        if (!hasDown) return;
        const next = moveItem(items, index, index + 1);
        setItems(node, next);
        
        // 如果移动的是当前选中项，跟随移动
        const currentIndex = Number(getCurrentIndex(node));
        const targetIndex = Number(index);
        
        if (targetIndex === currentIndex) {
            setIndexSelectorValue(node, targetIndex + 1);
        } else if (targetIndex + 1 === currentIndex) {
            // 如果移动到了当前选中项的位置
            setIndexSelectorValue(node, targetIndex);
        }

        if (node.ensureTextareas) node.ensureTextareas();
        app.graph.setDirtyCanvas(true, true);
    };
    const doMoveTo = () => {
        let to = prompt(`移动到索引 (0 - ${Math.max(0, n - 1)}):`, String(index));
        if (to == null) return;
        to = Number(to);
        if (!Number.isFinite(to)) return;
        
        // 限制范围
        to = Math.max(0, Math.min(n - 1, to));
        if (to === index) return;

        const next = moveItem(items, index, to);
        setItems(node, next);
        
        // 索引跟随逻辑
        const currentIndex = Number(getCurrentIndex(node));
        const targetIndex = Number(index);
        
        if (targetIndex === currentIndex) {
            // 如果移动的是当前选中项，直接更新为目标索引
            setIndexSelectorValue(node, to);
        } else {
            // 如果移动的不是当前选中项，但影响了当前选中项的索引
            // 比如从选中项之前移到之后，或者从之后移到之前
            if (targetIndex < currentIndex && to >= currentIndex) {
                // 从前移到后（跨过选中项），选中项索引减一
                setIndexSelectorValue(node, currentIndex - 1);
            } else if (targetIndex > currentIndex && to <= currentIndex) {
                // 从后移到前（跨过选中项），选中项索引加一
                setIndexSelectorValue(node, currentIndex + 1);
            }
        }

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
                showTopNotification(`已复制内容 ${index + 1}`, "success");
            } else {
                const tmp = document.createElement('textarea');
                tmp.value = value;
                document.body.appendChild(tmp);
                tmp.select();
                document.execCommand('copy');
                tmp.remove();
                showTopNotification(`已复制内容 ${index + 1}`, "success");
            }
        } catch (e) {
            const item = getItems(node)[index];
            showTopNotification("复制失败: " + e, "error");
            // prompt('复制失败，请手动复制:', item ? item.content : "");
        }
    };
    const doPaste = async () => {
        let text = "";
        try {
            if (navigator.clipboard?.readText) {
                text = await navigator.clipboard.readText();
            } else {
                // text = prompt('粘贴文本:', "") || "";
                showTopNotification("无法访问剪贴板，请使用 Ctrl+V", "warning");
                return;
            }
        } catch (e) {
            // text = prompt('粘贴文本:', "") || "";
            showTopNotification("无法访问剪贴板，请使用 Ctrl+V", "warning");
            return;
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
            showTopNotification('未找到节点的 index 控件', "error");
        }
    };
    const doAddToAssetManager = async () => {
        await openAddToAssetManagerGroupPicker(node, index);
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
            { content: `📁 添加到管理器...`, callback: doAddToAssetManager },
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
        const choice = prompt(`操作: u=使用该提示词, a=添加到管理器, c=清空, y=复制, p=粘贴, d=删除, up=上移, n=下移, m=移动到索引`, "u");
        if (choice === 'u') doUseThisPrompt();
        else if (choice === 'a') doAddToAssetManager();
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

function handleTextareaCommentShortcut(e, node, index) {
    // 监听 Ctrl+/ (Mac下可能是 Meta+/)
    // 同时支持 Ctrl+D 作为备用（如果用户习惯了，但可能会触发浏览器收藏）
    // 为了避免冲突，我们优先推荐 Ctrl+/，并拦截 Ctrl+D
    
    const isCtrl = e.ctrlKey || e.metaKey;
    const isSlash = e.key === '/';
    const isD = e.key === 'd' || e.key === 'D' || e.code === 'KeyD';

    if (isCtrl && (isSlash || isD)) {
        e.preventDefault();
        e.stopPropagation();

        const ta = e.target;
        if (!ta) return;

        const start = ta.selectionStart;
        const end = ta.selectionEnd;
        const value = ta.value;

        // 找到选中区域涉及的所有行
        // 1. 找到起始行的开始位置
        let lineStart = value.lastIndexOf('\n', start - 1);
        if (lineStart === -1) lineStart = 0;
        else lineStart += 1; // 跳过换行符

        // 2. 找到结束行的结束位置
        let lineEnd = value.indexOf('\n', end);
        if (lineEnd === -1) lineEnd = value.length;
        
        // 如果选区正好在行尾（选中多行时最后一行可能只选中了开头），需要注意
        // 但这里我们简单处理：只要选区触及到了行，就处理整行
        // 特殊情况：光标在行首，且 selectionStart === selectionEnd，也应该处理该行
        // 上面的逻辑已经覆盖了这种情况：lastIndexOf('\n', start - 1) 会找到上一行的末尾，lineStart 指向当前行首

        // 提取涉及的文本块
        const linesChunk = value.substring(lineStart, lineEnd);
        const lines = linesChunk.split('\n');
        
        // 检查是否所有行都已经被注释（以 # 开头）
        // 注意：这里我们检查的是每一行是否以 # 开头（忽略前导空格）
        // 如果所有选中的行都已经是注释，则执行取消注释；否则执行全部注释
        const allCommented = lines.every(line => line.trim().startsWith('#'));
        
        const newLines = lines.map(line => {
             if (allCommented) {
                 // 取消注释：移除开头的 # 和可选的一个空格
                 // 使用正则替换：仅替换开头的 # 以及其后紧跟的一个空格（如果有）
                 // ^\s*# ? 匹配开头空白、#、可选空格
                 // 但我们要保留原始缩进吗？
                 // 如果原始是 "  # abc"，取消后应该是 "  abc"
                 // 正则：replace(/^(\s*)# ?/, '$1')
                 return line.replace(/^(\s*)# ?/, '$1');
             } else {
                 // 添加注释：在开头添加 # 
                 // 或者在缩进后添加？通常是在最前面或者缩进后。
                 // PyCharm 默认 behavior: Ctrl+/ (Line Comment) usually toggles at the start of line content or start of line.
                 // 这里简单在行首添加 "# "
                 return '# ' + line;
             }
        });

        const newChunk = newLines.join('\n');
        
        // 使用 setRangeText 替换文本并保持选区
        // select mode: 'select' 选中替换后的文本
        try {
            ta.setRangeText(newChunk, lineStart, lineEnd, 'select');
        } catch (err) {
            // Fallback for older browsers
            ta.value = value.substring(0, lineStart) + newChunk + value.substring(lineEnd);
            ta.selectionStart = lineStart;
            ta.selectionEnd = lineStart + newChunk.length;
        }

        // 触发 input 事件以更新节点数据
        const event = new Event('input', { bubbles: true });
        ta.dispatchEvent(event);
    }
}

// {{ AURA-X: Add - 滚轮事件处理函数，用于在 Textarea 中支持边缘滚动穿透到 Canvas }}
function handleTextareaWheel(e) {
    if (e.ctrlKey || e.shiftKey) {
        e.stopPropagation();
        return;
    }
    const el = e.currentTarget;
    if (!el) return;
    
    const deltaY = e.deltaY || 0;
    const scrollingDown = deltaY > 0;
    const scrollingUp = deltaY < 0;
    const maxScrollTop = el.scrollHeight - el.clientHeight;
    const scrollTop = el.scrollTop;
    
    const canScrollDown = scrollTop < maxScrollTop - 1;
    const canScrollUp = scrollTop > 1;
    
    const atBottom = !canScrollDown;
    const atTop = !canScrollUp;
    
    let edgeState = el.__edgeScrollState;
    if (!edgeState) {
        edgeState = { dir: 0, count: 0 };
        el.__edgeScrollState = edgeState;
    }
    
    const dir = scrollingDown ? 1 : (scrollingUp ? -1 : 0);
    
    // 如果没有滚动方向，或者是横向滚动，可能不需要特别处理，但为了保险起见，阻止冒泡以防止缩放
    if (dir === 0) {
        // 如果是纯横向滚动，可能也需要阻止冒泡，除非我们想让画布平移
        // 这里假设主要处理纵向
        return;
    }

    if (!atTop && !atBottom) {
        // 在中间滚动
        edgeState.dir = 0;
        edgeState.count = 0;
        e.stopPropagation(); // 阻止冒泡，让 textarea 自己滚动
        return;
    }
    
    const atEdgeInDir = (dir > 0 && atBottom) || (dir < 0 && atTop);
    
    if (!atEdgeInDir) {
        // 虽然在边缘，但往回滚
        edgeState.dir = 0;
        edgeState.count = 0;
        e.stopPropagation();
        return;
    }
    
    // 在边缘继续往外滚
    if (edgeState.dir !== dir) {
        edgeState.dir = dir;
        edgeState.count = 1;
        e.stopPropagation(); // 第一次到达边缘，停顿一下
        return;
    }
    
    edgeState.count += 1;
    let belowThreshold = false;
    const count = edgeState.count;
    if (count === 1) {
        belowThreshold = true;
    }
    if (belowThreshold) {
        e.stopPropagation(); // 还没达到阈值
        return;
    }
    
    // 达到阈值，不阻止事件冒泡，让 scrollContainer 处理
}

// {{ AURA-X: Add - 自定义下拉菜单，带搜索功能 }}
function showCustomDropdown(node, items) {
    // 如果已存在，先关闭
    if (node.__customDropdown) {
        node.__customDropdown.remove();
        node.__customDropdown = null;
        return;
    }

    const canvas = app?.canvas?.canvas;
    const container = canvas?.parentElement || document.body;
    if (!canvas) return;

    // 创建容器
    const dropdown = document.createElement('div');
    dropdown.className = 'custom-dropdown-menu';
    dropdown.style.cssText = `
        position: absolute;
        z-index: 9999;
        background: #222;
        border: 1px solid #666;
        border-radius: 4px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.5);
        display: flex;
        flex-direction: column;
        overflow: hidden;
        font-family: "Microsoft YaHei", sans-serif;
        font-size: 12px;
        color: #eee;
        min-width: 150px;
    `;
    
    // 阻止事件冒泡
    dropdown.addEventListener('wheel', (e) => e.stopPropagation());
    dropdown.addEventListener('mousedown', (e) => e.stopPropagation());
    dropdown.addEventListener('mouseup', (e) => e.stopPropagation());

    // 搜索框
    const searchContainer = document.createElement('div');
    searchContainer.style.cssText = `
        padding: 6px;
        border-bottom: 1px solid #444;
        background: #333;
        flex-shrink: 0;
    `;
    
    const searchWrap = document.createElement('div');
    searchWrap.style.position = "relative";
    searchWrap.style.display = "flex";
    searchWrap.style.alignItems = "center";

    const searchInput = document.createElement('input');
    searchInput.type = 'text';
    searchInput.placeholder = '🔍 搜索拼音/原名/首字母...';
    searchInput.style.cssText = `
        width: 100%;
        box-sizing: border-box;
        padding: 4px 24px 4px 6px;
        border: 1px solid #555;
        border-radius: 3px;
        background: #111;
        color: #fff;
        font-size: 12px;
        outline: none;
    `;

    const clearBtn = document.createElement("span");
    clearBtn.textContent = "✖";
    clearBtn.style.position = "absolute";
    clearBtn.style.right = "6px";
    clearBtn.style.cursor = "pointer";
    clearBtn.style.color = "#aaa";
    clearBtn.style.fontSize = "12px";
    clearBtn.style.display = "none";
    clearBtn.onmouseenter = () => { clearBtn.style.color = "#fff"; };
    clearBtn.onmouseleave = () => { clearBtn.style.color = "#aaa"; };

    clearBtn.addEventListener("click", (e) => {
        e.preventDefault();
        e.stopPropagation();
        searchInput.value = "";
        clearBtn.style.display = "none";
        scheduleSearchRender("");
        searchInput.focus();
    });

    searchWrap.appendChild(searchInput);
    searchWrap.appendChild(clearBtn);
    searchContainer.appendChild(searchWrap);
    dropdown.appendChild(searchContainer);
    
    // 列表容器
    const listContainer = document.createElement('div');
    listContainer.style.cssText = `
        overflow-y: auto;
        flex: 1;
        max-height: 300px; /* 默认最大高度，会被动态调整覆盖 */
    `;
    
    // {{ AURA-X: Add - 点击列表空白处关闭 }}
    listContainer.addEventListener('click', (e) => {
        if (e.target === listContainer) {
            closeDropdown();
        }
    });
    
    dropdown.appendChild(listContainer);
    
    // 渲染列表函数
    const searchTexts = items.map((it, idx) => {
        const title = String(it?.title || `prompt_${idx}`);
        const content = String(it?.content || "");
        // 给拼音/首字母匹配更多上下文，但别太长
        const merged = `${title} ${content}`.trim();
        return merged.length > 240 ? merged.slice(0, 240) : merged;
    });

    const renderList = (matches, filterText = '') => {
        listContainer.innerHTML = '';
        let hasMatch = false;
        
        items.forEach((item, idx) => {
            const title = item.title || `prompt_${idx}`;
            const content = item.content || '';
            const match = matches ? !!matches[idx] : true;
            
            if (match) {
                hasMatch = true;
                const itemEl = document.createElement('div');
                itemEl.style.cssText = `
                    padding: 6px 10px;
                    cursor: pointer;
                    white-space: nowrap;
                    overflow: hidden;
                    text-overflow: ellipsis;
                    border-bottom: 1px solid #333;
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                `;
                
                // 标题部分
                const titleSpan = document.createElement('span');
                titleSpan.textContent = title;
                titleSpan.style.flex = "1";
                titleSpan.style.overflow = "hidden";
                titleSpan.style.textOverflow = "ellipsis";
                itemEl.appendChild(titleSpan);
                
                // 悬浮提示
                itemEl.title = content.substring(0, 100);
                
                // 选中状态
                if (idx === getCurrentIndex(node)) {
                    itemEl.style.background = '#4a9eff';
                    itemEl.style.color = '#fff';
                } else {
                    itemEl.onmouseenter = () => { itemEl.style.background = '#444'; };
                    itemEl.onmouseleave = () => { itemEl.style.background = 'transparent'; };
                }
                
                // 点击事件
                itemEl.onclick = (e) => {
                    e.stopPropagation();
                    setIndexSelectorValue(node, idx);
                    closeDropdown();
                };
                
                listContainer.appendChild(itemEl);
            }
        });
        
        if (!hasMatch) {
            const emptyEl = document.createElement('div');
            emptyEl.textContent = '无匹配结果';
            emptyEl.style.padding = '10px';
            emptyEl.style.color = '#888';
            emptyEl.style.textAlign = 'center';
            listContainer.appendChild(emptyEl);
        }
    };
    
    // 初始渲染
    renderList(null);
    
    // 搜索事件：API + 防抖
    let debounceTimer = null;
    let composing = false;
    let latestSeq = 0;

    const scheduleSearchRender = (keyword, delay = 150) => {
        if (debounceTimer) clearTimeout(debounceTimer);
        const seq = ++latestSeq;
        debounceTimer = setTimeout(async () => {
            debounceTimer = null;
            const matches = await fetchPinyinMatches(searchTexts, keyword);
            if (seq !== latestSeq) return;
            renderList(matches, keyword);
        }, delay);
    };

    searchInput.addEventListener("compositionstart", () => { composing = true; });
    searchInput.addEventListener("compositionend", () => {
        composing = false;
        clearBtn.style.display = searchInput.value ? "block" : "none";
        scheduleSearchRender(searchInput.value);
    });

    searchInput.addEventListener('input', (e) => {
        clearBtn.style.display = e.target.value ? "block" : "none";
        if (composing) return;
        scheduleSearchRender(e.target.value);
    });
    
    // 关闭函数
    const closeDropdown = () => {
        if (node.__customDropdown) {
            node.__customDropdown.remove();
            node.__customDropdown = null;
        }
        document.removeEventListener('mousedown', onDocClick, true);
        // 移除 window 上的事件（双重保险）
        window.removeEventListener('mousedown', onDocClick, true);
    };
    
    // 点击外部关闭
    const onDocClick = (e) => {
        // 如果点击的是 dropdown 内部，或者是 menuBtn，则不关闭
        if (node.__customDropdown && 
            node.__customDropdown.contains(e.target)) {
            return;
        }
        if (node.__comboMenuBtn && node.__comboMenuBtn.contains(e.target)) {
            return;
        }
        closeDropdown();
    };
    
    document.addEventListener('mousedown', onDocClick, true);
    // 同时监听 window 上的点击（双重保险，应对 Canvas 拦截）
    window.addEventListener('mousedown', onDocClick, true);
    
    // ESC 关闭
    searchInput.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') closeDropdown();
        e.stopPropagation();
    });

    container.appendChild(dropdown);
    node.__customDropdown = dropdown;
    
    // 立即更新一次位置
    updateCustomDropdownPosition(node);
    
    // 聚焦搜索框
    setTimeout(() => searchInput.focus(), 50);
}

// {{ AURA-X: Add - 更新下拉菜单位置 }}
function updateCustomDropdownPosition(node) {
    const dropdown = node.__customDropdown;
    const menuBtn = node.__comboMenuBtn;
    if (!dropdown || !menuBtn) return;
    
    const ds = app?.canvas?.ds;
    if (!ds) return;
    
    const canvas = app.canvas.canvas;
    const container = canvas.parentElement || document.body;
    
    const btnRect = menuBtn.getBoundingClientRect();
    const containerRect = container.getBoundingClientRect();
    
    // 计算相对于 container 的位置
    const left = btnRect.left - containerRect.left;
    const top = btnRect.bottom - containerRect.top;
    
    // 宽度跟随节点，但有最小值和最大值
    const nodeWidthPx = node.size[0] * ds.scale;
    dropdown.style.width = `${Math.max(160, Math.min(nodeWidthPx, 220))}px`;
    
    // 高度限制：不超过节点高度
    const nodeHeightPx = node.size[1] * ds.scale;
    // 考虑到搜索框的高度（约 30px），列表高度应适配
    // 另外要确保最小可用高度，但用户要求不超过节点高度
    // 如果节点高度太小（例如小于100px），这可能会导致显示问题，但我们必须遵守用户要求
    // 增加一个合理的最小值防止完全不可用，比如 100px，除非节点本身小于 100px
    const maxDropdownHeight = Math.max(Math.min(100, nodeHeightPx), nodeHeightPx);
    
    // 检查屏幕底部空间，防止溢出
    const spaceBelow = window.innerHeight - btnRect.bottom - 10;
    const finalMaxHeight = Math.min(maxDropdownHeight, spaceBelow);
    
    dropdown.style.maxHeight = `${finalMaxHeight}px`;
    
    dropdown.style.left = `${left}px`;
    dropdown.style.top = `${top}px`;
}

// {{ AURA-X: Add - 创建标题、内容和开关的UI元素. }}
function ensureTextareas(node, layout, items) {
    const ds = app?.canvas?.ds;
    const canvas = app?.canvas?.canvas;
    if (!ds || !canvas) return;
    const container = canvas.parentElement || document.body;
    const rect = canvas.getBoundingClientRect();
    const parentRect = container.getBoundingClientRect();
    const cs = window.getComputedStyle(container);
    if (cs.position === 'static') container.style.position = 'relative';

    const viewMode = node.properties?._viewMode || "grid";
    const currentIndex = getCurrentIndex(node);

    // 清理辅助函数
    const clearGridElements = () => {
        if (node.__taEls) { node.__taEls.forEach(el => el && el.remove()); node.__taEls = []; }
        if (node.__titleEls) { node.__titleEls.forEach(el => el && el.remove()); node.__titleEls = []; }
        if (node.__suffixEls) { node.__suffixEls.forEach(el => el && el.remove()); node.__suffixEls = []; }
        if (node.__toggleEls) { node.__toggleEls.forEach(el => el && el.remove()); node.__toggleEls = []; }
        if (node.__inputEls) { node.__inputEls.forEach(el => el && el.remove()); node.__inputEls = []; }
        if (node.__cardEls) { node.__cardEls.forEach(el => el && el.remove()); node.__cardEls = []; }
        if (node.__gridScrollContainer) { node.__gridScrollContainer.remove(); node.__gridScrollContainer = null; }
    };
    
    const clearComboElements = () => {
        if (node.__comboSelect) { node.__comboSelect.remove(); node.__comboSelect = null; }
        if (node.__comboTextarea) { node.__comboTextarea.remove(); node.__comboTextarea = null; }
        if (node.__comboInputEl) { node.__comboInputEl.remove(); node.__comboInputEl = null; }
        
        // 新增的 combo UI 元素清理
        if (node.__comboTitleInput) { node.__comboTitleInput.remove(); node.__comboTitleInput = null; }
        if (node.__comboSuffixEl) { node.__comboSuffixEl.remove(); node.__comboSuffixEl = null; }
        if (node.__comboMenuBtn) { node.__comboMenuBtn.remove(); node.__comboMenuBtn = null; }
        if (node.__comboToggleEl) { node.__comboToggleEl.remove(); node.__comboToggleEl = null; } // 清理 Combo 模式下的复选框
        if (node.__customDropdown) { node.__customDropdown.remove(); node.__customDropdown = null; } // 清理下拉菜单
    };

    if (viewMode === 'combo') {
        clearGridElements();
        
        // --- Combo 模式渲染 ---
        if (items.length === 0) {
            clearComboElements();
            return;
        }

        const cell = layout[0]; // Combo 模式只有一个布局单元格
        if (!cell) return;

        // 计算位置和大小
        const sx = (node.pos[0] + cell.x + ds.offset[0]) * ds.scale + rect.left - parentRect.left;
        const sy = (node.pos[1] + cell.y + ds.offset[1]) * ds.scale + rect.top - parentRect.top;
        // 确保宽度和高度至少为正数
        const sw = Math.max(0, cell.w * ds.scale);
        const sh = Math.max(0, cell.h * ds.scale);
        
        const titleHeight = 24;
        const inputBtnWidth = 16;
        const toggleWidth = 20; // 复选框宽度
        const menuBtnWidth = 20; // 下拉菜单按钮宽度
        
        // 1. 下拉菜单按钮 (▼)
        let menuBtn = node.__comboMenuBtn;
        if (!menuBtn) {
            menuBtn = document.createElement('div');
            menuBtn.textContent = "▼";
            menuBtn.style.cssText = `
                position: absolute; 
                z-index: 102; 
                cursor: pointer; 
                margin: 0;
                font-size: 10px;
                line-height: 24px;
                text-align: center;
                user-select: none;
                color: #eee;
                background: #444;
                border: 1px solid #666;
                border-right: none;
                border-bottom: none;
                border-radius: 6px 0 0 0;
                box-sizing: border-box;
            `;
            
            menuBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                e.preventDefault();
                
                // {{ AURA-X: Modify - 使用自定义下拉菜单 }}
                const currentItems = getItems(node);
                showCustomDropdown(node, currentItems);
            });
            
            menuBtn.addEventListener('wheel', (e) => { e.stopPropagation(); });
            
            container.appendChild(menuBtn);
            node.__comboMenuBtn = menuBtn;
        }

        // 2. 标题输入框 (可编辑 baseTitle)
        let titleInput = node.__comboTitleInput;
        const currentItem = items[currentIndex];
        const baseTitle = getBaseTitle(currentItem?.title || `prompt_${currentIndex}`);
        const suffixText = `${currentIndex}`;
        
        // 计算后缀宽度
        const suffixWidth = Math.max(16, Math.ceil(suffixText.length * 7) + 4);

        if (!titleInput) {
            titleInput = document.createElement('input');
            titleInput.type = 'text';
            titleInput.value = baseTitle;
            titleInput.placeholder = "标题";
            titleInput.style.cssText = `
                position: absolute; 
                z-index: 101; 
                padding: 4px 6px; 
                border: 1px solid #666; 
                border-left: none;
                border-right: none;
                border-bottom: none; 
                background: #3a3a3a; 
                color: #eee; 
                font-size: 11px; 
                line-height: 1.2; 
                font-family: "Microsoft YaHei", "SimHei", Arial, monospace; 
                box-sizing: border-box; 
                transform-origin: 0 0;
                outline: none;
            `;
            
            titleInput.addEventListener('input', () => {
                const arr = getItems(node);
                const idx = getCurrentIndex(node);
                if (idx < arr.length) {
                    const currentVal = titleInput.value.trim();
                    const newTitle = currentVal ? `${currentVal}_${idx}` : `prompt_${idx}`;
                    const oldTitle = arr[idx].title;
                    
                    if (oldTitle !== newTitle) {
                        const oldInputName = `insert_${oldTitle}`;
                        const newInputName = `insert_${newTitle}`;
                        if (node.inputs) {
                            const input = node.inputs.find(inp => inp.name === oldInputName);
                            if (input) {
                                input.name = newInputName;
                            }
                        }
                    }
                    
                    arr[idx].title = newTitle;
                    setItems(node, arr);
                }
            });
            
            titleInput.addEventListener('keydown', (e) => {
                if (e.key === 'Enter') {
                    titleInput.blur();
                } else if (e.key === 'Escape') {
                    const curItems = getItems(node);
                    const curItem = curItems[getCurrentIndex(node)];
                    titleInput.value = getBaseTitle(curItem ? curItem.title : `prompt`);
                    titleInput.blur();
                }
                e.stopPropagation();
            });
            
            container.appendChild(titleInput);
            node.__comboTitleInput = titleInput;
        } else {
             if (document.activeElement !== titleInput) {
                 titleInput.value = baseTitle;
             }
        }
        
        // 3. 后缀标签
        let suffixEl = node.__comboSuffixEl;
        if (!suffixEl) {
            suffixEl = document.createElement('div');
            suffixEl.style.cssText = `
                position: absolute; 
                z-index: 101; 
                color: #888; 
                font-size: 11px; 
                line-height: 24px; 
                font-family: "Microsoft YaHei", "SimHei", Arial, monospace; 
                pointer-events: none; 
                text-align: left; 
                padding-left: 2px;
                background: #3a3a3a;
                border-top: 1px solid #666;
            `;
            container.appendChild(suffixEl);
            node.__comboSuffixEl = suffixEl;
        }
        suffixEl.textContent = suffixText;

        // 4. Input 连接点按钮 (针对当前选中项)
        let inputEl = node.__comboInputEl;
        const safeTitle = (currentItem?.title || `prompt_${currentIndex}`).trim();
        const inputName = `insert_${safeTitle}`;
        const hasInput = node.inputs && node.inputs.some(inp => inp.name === inputName);
        
        if (!inputEl) {
            inputEl = document.createElement('div');
            inputEl.textContent = "🔗"; 
            inputEl.style.cssText = `
                position: absolute; 
                z-index: 102; 
                cursor: pointer; 
                margin: 0;
                font-size: 12px;
                line-height: 16px;
                text-align: center;
                user-select: none;
                transition: opacity 0.2s;
                color: #eee;
                background: transparent;
                pointer-events: auto;
            `;
            inputEl.title = "点击切换外部输入连接点";
            
            inputEl.addEventListener('click', (e) => {
                e.stopPropagation();
                e.preventDefault();
                const curIdx = getCurrentIndex(node); // 获取最新的 index
                const curItem = getItems(node)[curIdx];
                const sTitle = (curItem?.title || `prompt_${curIdx}`).trim();
                const iName = `insert_${sTitle}`;
                
                const currentInputIdx = node.inputs ? node.inputs.findIndex(inp => inp.name === iName) : -1;
                
                if (currentInputIdx !== -1) {
                    node.removeInput(currentInputIdx);
                } else {
                    node.addInput(iName, "STRING");
                }
                app.graph.setDirtyCanvas(true, true);
                // 重新渲染以更新状态
                const newItems = getItems(node);
                const newCells = layoutCells(node, newItems);
                ensureTextareas(node, newCells, newItems);
            });
             
            inputEl.addEventListener('wheel', (e) => { e.stopPropagation(); });

            container.appendChild(inputEl);
            node.__comboInputEl = inputEl;
        }
        
        if (hasInput) {
            inputEl.style.opacity = '1.0';
            inputEl.style.filter = 'none';
        } else {
            inputEl.style.opacity = '0.3';
            inputEl.style.filter = 'grayscale(100%)';
        }

        // 5. 复选框 (控制 enabled)
        let toggleEl = node.__comboToggleEl;
        if (!toggleEl) {
            toggleEl = document.createElement('input');
            toggleEl.type = 'checkbox';
            toggleEl.style.cssText = `position: absolute; z-index: 102; cursor: pointer; margin: 0;`;
            
            toggleEl.addEventListener('change', (e) => {
                const arr = getItems(node);
                const idx = getCurrentIndex(node);
                if (idx < arr.length) {
                    arr[idx].enabled = toggleEl.checked;
                    setItems(node, arr);
                    app.graph.setDirtyCanvas(true, true);
                }
            });
            
            toggleEl.addEventListener('wheel', (e) => { e.stopPropagation(); });
            
            container.appendChild(toggleEl);
            node.__comboToggleEl = toggleEl;
        }
        
        // 更新复选框状态
        toggleEl.checked = currentItem.enabled !== false; 
        toggleEl.disabled = false;

        // 6. 创建或更新 Textarea (显示当前选中项内容)
        let ta = node.__comboTextarea;
        if (!ta) {
            ta = document.createElement('textarea');
            ta.placeholder = `内容`;
            ta.spellcheck = false;
            ta.wrap = 'soft';
            ta.style.cssText = `
                position: absolute; 
                z-index: 100; 
                resize: none; 
                padding: 6px; 
                border-radius: 0 0 6px 6px; 
                border: 2px solid #4a9eff; 
                border-top: none; 
                background: #222; 
                color: #eee; 
                font-size: 12px; 
                line-height: 1.4; 
                font-family: "Microsoft YaHei", "SimHei", Arial, monospace; 
                box-sizing: border-box; 
                overflow: auto; 
                transform-origin: 0 0;
                box-shadow: 0 0 8px rgba(74, 158, 255, 0.3);
            `;
            
            ta.addEventListener('input', () => {
                const arr = getItems(node);
                const idx = getCurrentIndex(node);
                if (idx < arr.length) {
                    arr[idx].content = ta.value;
                    setItems(node, arr);
                }
            });
            
            // 右键菜单等事件 (复用之前的逻辑，这里简化)
            ta.addEventListener('keydown', (e) => { e.stopPropagation(); });
            
            // {{ AURA-X: Add - Combo模式下的右键菜单支持 }}
            ta.addEventListener('contextmenu', (e) => {
                e.stopPropagation();
                e.preventDefault();
                const idx = getCurrentIndex(node);
                if (idx !== null && idx >= 0) {
                    showItemContextMenu(node, idx, e);
                }
            });

            // 使用通用的滚轮处理函数，支持边缘穿透
            ta.addEventListener('wheel', handleTextareaWheel);
            
            // {{ AURA-X: Add - 监听 Ctrl+D 快捷键 }}
            ta.addEventListener('keydown', (e) => {
                const idx = getCurrentIndex(node);
                handleTextareaCommentShortcut(e, node, idx);
            });
            
            container.appendChild(ta);
            node.__comboTextarea = ta;
        }
        
        // 更新 Textarea 值
        if (ta.value !== currentItem.content) {
            ta.value = currentItem.content || "";
        }

        // --- 布局设置 ---
        
        // Menu Button (Left)
        menuBtn.style.left = `${sx}px`;
        menuBtn.style.top = `${sy}px`;
        menuBtn.style.width = `${menuBtnWidth * ds.scale}px`;
        menuBtn.style.height = `${titleHeight * ds.scale}px`;
        menuBtn.style.fontSize = `${10 * ds.scale}px`;
        menuBtn.style.lineHeight = `${24 * ds.scale}px`; // Vertically center arrow
        
        // Title Input (Middle)
        // 宽度 = 总宽 - 菜单按钮 - 后缀 - Input按钮 - 复选框 - 间距
        const titleInputAvailableW = Math.max(20, Math.round(cell.w - menuBtnWidth - suffixWidth - inputBtnWidth - toggleWidth - 6));
        
        titleInput.style.left = `${sx + menuBtnWidth * ds.scale}px`;
        titleInput.style.top = `${sy}px`;
        titleInput.style.width = `${titleInputAvailableW}px`;
        titleInput.style.height = `${titleHeight}px`; // 逻辑高度，transform scale 处理
        titleInput.style.transform = `scale(${ds.scale})`;
        
        // Suffix (Right of Title)
        suffixEl.style.left = `${sx + (menuBtnWidth + titleInputAvailableW) * ds.scale}px`;
        suffixEl.style.top = `${sy}px`;
        suffixEl.style.width = `${suffixWidth * ds.scale}px`;
        suffixEl.style.height = `${titleHeight * ds.scale}px`;
        suffixEl.style.fontSize = `${11 * ds.scale}px`;
        suffixEl.style.lineHeight = `${24 * ds.scale}px`;
        
        // Input Button (Right of Suffix)
        inputEl.style.left = `${sx + (menuBtnWidth + titleInputAvailableW + suffixWidth) * ds.scale + 2}px`;
        inputEl.style.top = `${sy + 4 * ds.scale}px`;
        inputEl.style.width = `${16 * ds.scale}px`;
        inputEl.style.height = `${16 * ds.scale}px`;
        inputEl.style.fontSize = `${12 * ds.scale}px`;
        inputEl.style.lineHeight = `${16 * ds.scale}px`;
        inputEl.style.display = 'block';

        // Checkbox (Far Right)
        toggleEl.style.left = `${sx + sw - toggleWidth * ds.scale - 2}px`;
        toggleEl.style.top = `${sy + 4 * ds.scale}px`;
        toggleEl.style.width = `${16 * ds.scale}px`;
        toggleEl.style.height = `${16 * ds.scale}px`;
        toggleEl.style.transform = `scale(${1})`;

        // Textarea
        ta.style.left = `${sx}px`;
        ta.style.top = `${sy + titleHeight * ds.scale}px`;
        ta.style.width = `${Math.round(cell.w)}px`;
        ta.style.height = `${Math.max(32, Math.round(cell.h - titleHeight))}px`;
        ta.style.transform = `scale(${ds.scale})`;
        ta.style.fontSize = `${12}px`;

        // 可见性控制
        const nodeVisibleX = sx + sw > 0 && sx < (parentRect.width || rect.width);
        const nodeVisibleY = sy + sh > 0 && sy < (parentRect.height || rect.height);
        const shouldShow = node.flags?.collapsed !== true && nodeVisibleX && nodeVisibleY;
        const hand = isHandMode();
        
        const visibility = shouldShow ? 'visible' : 'hidden';
        const pointerEvents = shouldShow && !hand ? 'auto' : 'none';
        
        menuBtn.style.visibility = visibility;
        menuBtn.style.pointerEvents = pointerEvents;
        
        titleInput.style.visibility = visibility;
        titleInput.style.pointerEvents = pointerEvents;
        
        suffixEl.style.visibility = visibility;
        
        inputEl.style.visibility = visibility;
        inputEl.style.pointerEvents = pointerEvents;
        
        toggleEl.style.visibility = visibility;
        toggleEl.style.pointerEvents = pointerEvents;
        
        ta.style.visibility = visibility;
        ta.style.pointerEvents = pointerEvents;
        
        // {{ AURA-X: Add - 更新下拉菜单位置 }}
        if (node.__customDropdown) {
            updateCustomDropdownPosition(node);
            node.__customDropdown.style.visibility = visibility;
        }

        return; // Combo 模式处理完毕
    }

    // --- Grid 模式 (清理 Combo 元素) ---
    clearComboElements();

    const startY = 8 + getWidgetsBottom(node);
    const useScrollMode = node.__useScrollMode;
    const totalContentHeight = node.__totalContentHeight || 0;
    const visibleContentHeight = node.__visibleContentHeight || 500;

    let scrollContainer = node.__gridScrollContainer;
    if (useScrollMode) {
        if (!scrollContainer) {
            scrollContainer = document.createElement('div');
            scrollContainer.style.cssText = `
                position: absolute;
                z-index: 99;
                overflow-y: auto;
                overflow-x: hidden;
                background: rgba(30, 30, 30, 0.95);
                border-radius: 6px;
                box-sizing: border-box;
                scrollbar-width: thin;
                scrollbar-color: #555 #333;
                padding-right: 12px;
            `;
            scrollContainer.addEventListener('wheel', function(e) {
                if (e.ctrlKey || e.shiftKey) {
                    e.stopPropagation();
                    return;
                }
                const el = scrollContainer;
                const deltaY = e.deltaY || 0;
                const scrollingDown = deltaY &gt; 0;
                const scrollingUp = deltaY &lt; 0;
                const maxScrollTop = el.scrollHeight - el.clientHeight;
                const scrollTop = el.scrollTop;
                
                const canScrollDown = scrollTop &lt; maxScrollTop - 1;
                const canScrollUp = scrollTop &gt; 1;
                
                const atBottom = !canScrollDown;
                const atTop = !canScrollUp;
                
                let edgeState = el.__edgeScrollState;
                if (!edgeState) {
                    edgeState = { dir: 0, count: 0 };
                    el.__edgeScrollState = edgeState;
                }
                
                const dir = scrollingDown ? 1 : (scrollingUp ? -1 : 0);
                if (dir === 0) {
                    return;
                }
                
                const notAtTop = !atTop;
                const notAtBottom = !atBottom;
                let bothNotAtEdge = false;
                if (notAtTop) {
                    if (notAtBottom) {
                        bothNotAtEdge = true;
                    }
                }
                if (bothNotAtEdge) {
                    edgeState.dir = 0;
                    edgeState.count = 0;
                    e.stopPropagation();
                    return;
                }
                
                const isDown = dir &gt; 0;
                const isUp = dir &lt; 0;
                let isDownAndAtBottom = false;
                if (isDown) {
                    if (atBottom) {
                        isDownAndAtBottom = true;
                    }
                }
                let isUpAndAtTop = false;
                if (isUp) {
                    if (atTop) {
                        isUpAndAtTop = true;
                    }
                }
                const atEdgeInDir = isDownAndAtBottom || isUpAndAtTop;
                if (!atEdgeInDir) {
                    edgeState.dir = 0;
                    edgeState.count = 0;
                    e.stopPropagation();
                    return;
                }
                
                if (edgeState.dir !== dir) {
                    edgeState.dir = dir;
                    edgeState.count = 1;
                    e.stopPropagation();
                    return;
                }
                
                edgeState.count += 1;
                const belowThreshold = edgeState.count &lt; 2;
                if (belowThreshold) {
                    e.stopPropagation();
                    return;
                }
                
                const canvasEl = app?.canvas?.canvas;
                if (!canvasEl || typeof WheelEvent === 'undefined') {
                    return;
                }
                
                const evt = new WheelEvent('wheel', {
                    deltaX: e.deltaX,
                    deltaY: e.deltaY,
                    deltaZ: e.deltaZ,
                    deltaMode: e.deltaMode,
                    clientX: e.clientX,
                    clientY: e.clientY,
                    ctrlKey: e.ctrlKey,
                    shiftKey: e.shiftKey,
                    altKey: e.altKey,
                    metaKey: e.metaKey,
                    buttons: e.buttons,
                    bubbles: true,
                    cancelable: true
                });
                
                canvasEl.dispatchEvent(evt);
                e.preventDefault();
                e.stopPropagation();
            });
            container.appendChild(scrollContainer);
            node.__gridScrollContainer = scrollContainer;
        }
        
        const scrollSx = (node.pos[0] + 8 + ds.offset[0]) * ds.scale + rect.left - parentRect.left;
        const scrollSy = (node.pos[1] + startY + ds.offset[1]) * ds.scale + rect.top - parentRect.top;
        const scrollSw = Math.max(0, (node.size[0] - 16)) * ds.scale;
        const scrollSh = visibleContentHeight * ds.scale;
        
        scrollContainer.style.left = `${scrollSx}px`;
        scrollContainer.style.top = `${scrollSy}px`;
        scrollContainer.style.width = `${scrollSw / ds.scale}px`;
        scrollContainer.style.height = `${scrollSh / ds.scale}px`;
        scrollContainer.style.transform = `scale(${ds.scale})`;
        scrollContainer.style.transformOrigin = '0 0';
        
        const nodeVisibleX = scrollSx + scrollSw > 0 && scrollSx < (parentRect.width || rect.width);
        const nodeVisibleY = scrollSy + scrollSh > 0 && scrollSy < (parentRect.height || rect.height);
        const shouldShow = node.flags?.collapsed !== true && nodeVisibleX && nodeVisibleY;
        const hand = isHandMode();
        
        scrollContainer.style.visibility = shouldShow ? 'visible' : 'hidden';
        scrollContainer.style.pointerEvents = shouldShow && !hand ? 'auto' : 'none';
    } else {
        if (scrollContainer) {
            scrollContainer.remove();
            node.__gridScrollContainer = null;
        }
    }

    // 初始化元素数组
    if (!node.__taEls) node.__taEls = [];
    if (!node.__titleEls) node.__titleEls = [];
    if (!node.__suffixEls) node.__suffixEls = []; // 新增后缀标签数组
    if (!node.__toggleEls) node.__toggleEls = [];
    if (!node.__inputEls) node.__inputEls = []; // 新增输入开关数组

    // const currentIndex = getCurrentIndex(node); // 已在上面定义

    // 初始化卡片容器数组
    if (!node.__cardEls) node.__cardEls = [];

    for (let i = 0; i < items.length; i++) {
        const cell = layout[i];
        if (!cell) continue;
        
        const item = items[i];
        const isSelected = i === currentIndex;
        const baseTitle = getBaseTitle(item.title);
        const suffixText = `${i}`;
        
        // 创建或更新卡片容器
        let cardEl = node.__cardEls[i];
        if (!cardEl) {
            cardEl = document.createElement('div');
            cardEl.style.cssText = `
                position: absolute;
                z-index: 99;
                border-radius: 8px;
                box-sizing: border-box;
                overflow: hidden;
                transition: box-shadow 0.2s, border-color 0.2s;
            `;
            container.appendChild(cardEl);
            node.__cardEls[i] = cardEl;
        }
        
        // 创建或更新开关 (Checkbox)
        let toggleEl = node.__toggleEls[i];
        if (!toggleEl) {
            toggleEl = document.createElement('input');
            toggleEl.type = 'checkbox';
            toggleEl.style.cssText = `position: absolute; z-index: 101; cursor: pointer; margin: 0;`;
            
            // 开关事件处理
            toggleEl.addEventListener('change', (e) => {
                const arr = getItems(node);
                if (i < arr.length) {
                    arr[i].enabled = toggleEl.checked;
                    setItems(node, arr);
                    // 触发重绘以更新样式
                    const newItems = getItems(node);
                    const newCells = layoutCells(node, newItems);
                    ensureTextareas(node, newCells, newItems);
                }
            });
            
            // 防止滚轮事件穿透
            toggleEl.addEventListener('wheel', (e) => { e.stopPropagation(); });
            
            container.appendChild(toggleEl);
            node.__toggleEls[i] = toggleEl;
        }
        
        // 更新开关状态
        toggleEl.checked = item.enabled !== false;
        // 移除强制禁用逻辑，允许自由切换
        toggleEl.disabled = false;

        // {{ AURA-X: Add - 输入连接点开关按钮 }}
        let inputEl = node.__inputEls[i];
        const safeTitle = (item.title || `prompt_${i}`).trim();
        const inputName = `insert_${safeTitle}`;
        
        // --- 迁移旧版本接口名逻辑 ---
        // 之前的版本可能使用 insert_{index} 作为接口名
        // 如果发现节点上有旧格式的接口，且该位置不应该使用旧格式（即 title 产生的名字不同），则将其重命名
        const oldInputName = `insert_${i}`;
        if (inputName !== oldInputName && node.inputs) {
            const oldInput = node.inputs.find(inp => inp.name === oldInputName);
            const newInput = node.inputs.find(inp => inp.name === inputName);
            
            // 只有当旧接口存在，且新接口不存在时，才进行重命名
            // 如果两者都存在，说明可能是用户有意为之，或者是某种冲突，暂时保留新接口（按钮控制新接口）
            if (oldInput && !newInput) {
                oldInput.name = inputName;
            }
        }
        // ---------------------------

        const hasInput = node.inputs && node.inputs.some(inp => inp.name === inputName);
        
        if (!inputEl) {
            inputEl = document.createElement('div');
            inputEl.textContent = "🔗"; // 使用textContent避免innerText的一些问题
            inputEl.style.cssText = `
                position: absolute; 
                z-index: 102; 
                cursor: pointer; 
                margin: 0;
                font-size: 12px;
                line-height: 1;
                text-align: center;
                user-select: none;
                transition: opacity 0.2s;
                color: #eee;
                background: transparent;
                pointer-events: auto;
            `;
            inputEl.title = "点击切换外部输入连接点";
            
            inputEl.addEventListener('click', (e) => {
                e.stopPropagation();
                e.preventDefault();
                
                // 再次检查当前是否有输入
                const currentIdx = node.inputs ? node.inputs.findIndex(inp => inp.name === inputName) : -1;
                
                if (currentIdx !== -1) {
                    // 移除输入
                    node.removeInput(currentIdx);
                } else {
                    // 添加输入
                    node.addInput(inputName, "STRING");
                }
                
                // 触发重绘
                app.graph.setDirtyCanvas(true, true);
                // 重新运行ensureTextareas以更新按钮状态
                const currentItems = getItems(node);
                const currentCells = layoutCells(node, currentItems);
                ensureTextareas(node, currentCells, currentItems);
            });
             
             // 防止滚轮事件穿透
            inputEl.addEventListener('wheel', (e) => { e.stopPropagation(); });

            container.appendChild(inputEl);
            node.__inputEls[i] = inputEl;
        }
        
        // 更新Input按钮状态样式
        if (hasInput) {
            inputEl.style.opacity = '1.0';
            inputEl.style.filter = 'none';
        } else {
            inputEl.style.opacity = '0.3';
            inputEl.style.filter = 'grayscale(100%)';
        }
        
        // 创建或更新后缀标签
        let suffixEl = node.__suffixEls[i];
        if (!suffixEl) {
            suffixEl = document.createElement('div');
            suffixEl.style.cssText = `position: absolute; z-index: 100; color: #888; font-size: 11px; line-height: 1.2; font-family: "Microsoft YaHei", "SimHei", Arial, monospace; pointer-events: none; text-align: left; padding: 5px 0 0 2px;`;
            container.appendChild(suffixEl);
            node.__suffixEls[i] = suffixEl;
        }
        suffixEl.textContent = suffixText;

        // 创建或更新标题输入框
        let titleEl = node.__titleEls[i];
        if (!titleEl) {
            titleEl = document.createElement('input');
            titleEl.type = 'text';
            titleEl.placeholder = `标题`;
            titleEl.value = baseTitle; // 显示不带后缀的标题
            titleEl.style.cssText = `position: absolute; z-index: 100; padding: 4px 6px; border-radius: 4px; border: none; background: transparent; color: #eee; font-size: 15px; line-height: 1.2; font-family: "Microsoft YaHei", "SimHei", Arial, monospace; box-sizing: border-box; transform-origin: 0 0;`;
            
            // 标题输入框事件处理
            titleEl.addEventListener('input', () => {
                const arr = getItems(node);
                if (i < arr.length) {
                    const currentVal = titleEl.value.trim();
                    const newTitle = currentVal ? `${currentVal}_${i}` : `prompt_${i}`; // 强制添加后缀
                    const oldTitle = arr[i].title;
                    
                    if (oldTitle !== newTitle) {
                        // 如果标题改变，同时尝试更新对应的输入连接点名称
                        const oldInputName = `insert_${oldTitle}`;
                        const newInputName = `insert_${newTitle}`;
                        if (node.inputs) {
                            const input = node.inputs.find(inp => inp.name === oldInputName);
                            if (input) {
                                input.name = newInputName;
                            }
                        }
                    }

                    arr[i].title = newTitle;
                    setItems(node, arr);
                }
            });

            // 悬浮显示 Tooltip
            titleEl.addEventListener('mouseenter', (e) => {
                if (document.activeElement === titleEl) return; // 编辑状态不显示
                const currentItems = getItems(node);
                const currentItem = currentItems[i];
                if (currentItem) {
                    Tooltip.scheduleShow(e.clientX, e.clientY, currentItem.title, currentItem.content);
                }
            });
            titleEl.addEventListener('mouseleave', () => {
                Tooltip.hide();
            });
            titleEl.addEventListener('mousedown', () => {
                Tooltip.hide();
            });
            
            // 支持Enter键确认，Escape键取消
            titleEl.addEventListener('keydown', (e) => {
                if (e.key === 'Enter') {
                    titleEl.blur();
                } else if (e.key === 'Escape') {
                    const currentItems = getItems(node);
                    const currentItem = currentItems[i];
                    titleEl.value = getBaseTitle(currentItem ? currentItem.title : `prompt`);
                    titleEl.blur();
                }
                e.stopPropagation();
            });
            
            container.appendChild(titleEl);
            node.__titleEls[i] = titleEl;
        } else {
            // 更新现有标题输入框的值
            // 仅当非焦点状态下更新，或者值确实不匹配时更新，避免打断输入
            if (document.activeElement !== titleEl) {
                 titleEl.value = baseTitle;
            }
        }

        // 创建或更新内容文本框
        let ta = node.__taEls[i];
        if (!ta) {
            ta = document.createElement('textarea');
            ta.placeholder = `内容 ${i+1}`;
            ta.spellcheck = false;
            ta.wrap = 'soft';
            ta.value = item.content || "";
            ta.style.cssText = `position: absolute; z-index: 100; resize: none; padding: 6px; border-radius: 4px; border: none; background: transparent; color: #eee; font-size: 12px; line-height: 1.4; font-family: "Microsoft YaHei", "SimHei", Arial, monospace; box-sizing: border-box; overflow: auto; transform-origin: 0 0;`;
            
            // 内容文本框事件处理
            ta.addEventListener('input', () => {
                const arr = getItems(node);
                if (i < arr.length) {
                    arr[i].content = ta.value;
                    setItems(node, arr);
                }
            });

            // 悬浮显示 Tooltip
            ta.addEventListener('mouseenter', (e) => {
                if (document.activeElement === ta) return; // 编辑状态不显示
                const currentItems = getItems(node);
                const currentItem = currentItems[i];
                if (currentItem) {
                    Tooltip.scheduleShow(e.clientX, e.clientY, currentItem.title, currentItem.content);
                }
            });
            ta.addEventListener('mouseleave', () => {
                Tooltip.hide();
            });
            ta.addEventListener('mousedown', () => {
                Tooltip.hide();
            });
            
            // 右键菜单与滚轮行为
            if (!ta.__ctxInstalled) {
                ta.addEventListener('contextmenu', (e) => {
                    e.preventDefault(); 
                    e.stopPropagation();
                    showItemContextMenu(node, i, e);
                });
                ta.addEventListener('keydown', (e) => { e.stopPropagation(); });
                // 使用通用的滚轮处理函数
                ta.addEventListener('wheel', handleTextareaWheel);
                
                // {{ AURA-X: Add - 监听 Ctrl+D 快捷键 }}
                ta.addEventListener('keydown', (e) => {
                    const idx = getCurrentIndex(node);
                    handleTextareaCommentShortcut(e, node, idx);
                });
                
                ta.__ctxInstalled = true;
            }
            
            container.appendChild(ta);
            node.__taEls[i] = ta;
        } else {
            // 更新现有textarea的值
            ta.value = item.content || "";
        }

        // 计算位置和大小
        let sx, sy, sw, sh;
        const GAP = 6;
        
        if (useScrollMode && scrollContainer) {
            sx = cell.x;
            sy = cell.scrollY;
            sw = cell.w;
            sh = cell.h;
        } else {
            sx = (node.pos[0] + cell.x + ds.offset[0]) * ds.scale + rect.left - parentRect.left;
            sy = (node.pos[1] + startY + cell.scrollY + ds.offset[1]) * ds.scale + rect.top - parentRect.top;
            sw = cell.w * ds.scale;
            sh = cell.h * ds.scale;
        }
        
        const titleHeight = 26;
        const toggleWidth = 18;
        const inputBtnWidth = 18;
        const cardPadding = 4;
        
        const suffixWidth = Math.max(20, Math.ceil(suffixText.length * 8) + 8);

        if (useScrollMode && scrollContainer) {
            // 卡片容器
            cardEl.style.left = `${sx}px`;
            cardEl.style.top = `${sy}px`;
            cardEl.style.width = `${sw}px`;
            cardEl.style.height = `${sh}px`;
            cardEl.style.border = isSelected ? '2px solid #4a9eff' : '1px solid #555';
            cardEl.style.background = isSelected ? 'rgba(74, 158, 255, 0.1)' : 'rgba(45, 45, 45, 0.95)';
            cardEl.style.boxShadow = isSelected ? '0 0 12px rgba(74, 158, 255, 0.4)' : '0 2px 8px rgba(0,0,0,0.3)';
            
            // Index标签
            suffixEl.style.left = `${sx + cardPadding}px`;
            suffixEl.style.top = `${sy + cardPadding}px`;
            suffixEl.style.width = `${suffixWidth}px`;
            suffixEl.style.height = `${titleHeight - 4}px`;
            suffixEl.style.background = isSelected ? '#4a9eff' : '#555';
            suffixEl.style.borderRadius = '4px';
            suffixEl.style.display = 'flex';
            suffixEl.style.alignItems = 'center';
            suffixEl.style.justifyContent = 'center';
            suffixEl.style.color = '#fff';
            suffixEl.style.fontWeight = 'bold';
            suffixEl.style.transform = `scale(1)`;
            
            // 标题输入框
            const titleAvailableW = Math.max(40, Math.round(sw - suffixWidth - toggleWidth - inputBtnWidth - cardPadding * 5));
            titleEl.style.left = `${sx + suffixWidth + cardPadding * 2}px`;
            titleEl.style.top = `${sy + cardPadding}px`;
            titleEl.style.width = `${titleAvailableW}px`;
            titleEl.style.height = `${titleHeight - 4}px`;
            titleEl.style.borderRadius = '4px';
            titleEl.style.border = 'none';
            titleEl.style.background = 'transparent';
            titleEl.style.transform = `scale(1)`;
            
            // 连接点按钮
            if (inputEl) {
                inputEl.style.left = `${sx + sw - toggleWidth - inputBtnWidth - cardPadding * 2}px`;
                inputEl.style.top = `${sy + cardPadding + 2}px`;
                inputEl.style.width = `${inputBtnWidth}px`;
                inputEl.style.height = `${inputBtnWidth}px`;
                inputEl.style.fontSize = `12px`;
                inputEl.style.lineHeight = `${inputBtnWidth}px`;
                inputEl.style.display = 'flex';
                inputEl.style.alignItems = 'center';
                inputEl.style.justifyContent = 'center';
                inputEl.style.borderRadius = '4px';
                inputEl.style.background = hasInput ? '#4a9eff' : '#444';
            }
            
            // 复选框
            toggleEl.style.left = `${sx + sw - toggleWidth - cardPadding}px`;
            toggleEl.style.top = `${sy + cardPadding + 2}px`;
            toggleEl.style.width = `14px`;
            toggleEl.style.height = `14px`;
            toggleEl.style.transform = `scale(1)`;
            
            // 内容区域
            ta.style.left = `${sx + cardPadding}px`;
            ta.style.top = `${sy + titleHeight + cardPadding}px`;
            ta.style.width = `${Math.max(40, Math.round(sw - cardPadding * 2))}px`;
            ta.style.height = `${Math.max(32, Math.round(sh - titleHeight - cardPadding * 2))}px`;
            ta.style.borderRadius = '4px';
            ta.style.border = 'none';
            ta.style.background = 'transparent';
            ta.style.transform = `scale(1)`;
            
            titleEl.style.fontSize = `15px`;
            ta.style.fontSize = `12px`;
            
            // 添加到滚动容器
            if (cardEl.parentElement !== scrollContainer) {
                scrollContainer.appendChild(cardEl);
            }
            if (suffixEl.parentElement !== scrollContainer) {
                scrollContainer.appendChild(suffixEl);
            }
            if (titleEl.parentElement !== scrollContainer) {
                scrollContainer.appendChild(titleEl);
            }
            if (inputEl && inputEl.parentElement !== scrollContainer) {
                scrollContainer.appendChild(inputEl);
            }
            if (toggleEl.parentElement !== scrollContainer) {
                scrollContainer.appendChild(toggleEl);
            }
            if (ta.parentElement !== scrollContainer) {
                scrollContainer.appendChild(ta);
            }
        } else {
            // 非滚动模式
            const scale = ds.scale;
            
            // 卡片容器
            cardEl.style.left = `${sx}px`;
            cardEl.style.top = `${sy}px`;
            cardEl.style.width = `${sw}px`;
            cardEl.style.height = `${sh}px`;
            cardEl.style.border = isSelected ? '2px solid #4a9eff' : '1px solid #555';
            cardEl.style.background = isSelected ? 'rgba(74, 158, 255, 0.1)' : 'rgba(45, 45, 45, 0.95)';
            cardEl.style.boxShadow = isSelected ? '0 0 12px rgba(74, 158, 255, 0.4)' : '0 2px 8px rgba(0,0,0,0.3)';
            
            // Index标签
            suffixEl.style.left = `${sx + cardPadding * scale}px`;
            suffixEl.style.top = `${sy + cardPadding * scale}px`;
            suffixEl.style.width = `${suffixWidth}px`;
            suffixEl.style.height = `${titleHeight - 4}px`;
            suffixEl.style.background = isSelected ? '#4a9eff' : '#555';
            suffixEl.style.borderRadius = '4px';
            suffixEl.style.display = 'flex';
            suffixEl.style.alignItems = 'center';
            suffixEl.style.justifyContent = 'center';
            suffixEl.style.color = '#fff';
            suffixEl.style.fontWeight = 'bold';
            suffixEl.style.transform = `scale(${scale})`;
            suffixEl.style.transformOrigin = '0 0';
            
            // 标题输入框
            const titleAvailableW = Math.max(40, Math.round(sw / scale - suffixWidth - toggleWidth - inputBtnWidth - cardPadding * 5));
            titleEl.style.left = `${sx + (suffixWidth + cardPadding * 2) * scale}px`;
            titleEl.style.top = `${sy + cardPadding * scale}px`;
            titleEl.style.width = `${titleAvailableW}px`;
            titleEl.style.height = `${titleHeight - 4}px`;
            titleEl.style.borderRadius = '4px';
            titleEl.style.border = 'none';
            titleEl.style.background = 'transparent';
            titleEl.style.transform = `scale(${scale})`;
            titleEl.style.transformOrigin = '0 0';
            
            // 连接点按钮
            if (inputEl) {
                inputEl.style.left = `${sx + (sw / scale - toggleWidth - inputBtnWidth - cardPadding * 2) * scale}px`;
                inputEl.style.top = `${sy + (cardPadding + 2) * scale}px`;
                inputEl.style.width = `${inputBtnWidth * scale}px`;
                inputEl.style.height = `${inputBtnWidth * scale}px`;
                inputEl.style.fontSize = `${12 * scale}px`;
                inputEl.style.lineHeight = `${inputBtnWidth * scale}px`;
                inputEl.style.display = 'flex';
                inputEl.style.alignItems = 'center';
                inputEl.style.justifyContent = 'center';
                inputEl.style.borderRadius = '4px';
                inputEl.style.background = hasInput ? '#4a9eff' : '#444';
            }
            
            // 复选框
            toggleEl.style.left = `${sx + (sw / scale - toggleWidth - cardPadding) * scale}px`;
            toggleEl.style.top = `${sy + (cardPadding + 2) * scale}px`;
            toggleEl.style.width = `${14 * scale}px`;
            toggleEl.style.height = `${14 * scale}px`;
            toggleEl.style.transform = `scale(1)`;
            
            // 内容区域
            ta.style.left = `${sx + cardPadding * scale}px`;
            ta.style.top = `${sy + (titleHeight + cardPadding) * scale}px`;
            ta.style.width = `${Math.max(40, Math.round(sw / scale - cardPadding * 2))}px`;
            ta.style.height = `${Math.max(32, Math.round(sh / scale - (titleHeight + cardPadding * 2)))}px`;
            ta.style.borderRadius = '4px';
            ta.style.border = 'none';
            ta.style.background = 'transparent';
            ta.style.transform = `scale(${scale})`;
            ta.style.transformOrigin = '0 0';
            
            titleEl.style.fontSize = `15px`;
            ta.style.fontSize = `12px`;
            
            // 添加到容器
            if (cardEl.parentElement !== container) {
                container.appendChild(cardEl);
            }
            if (suffixEl.parentElement !== container) {
                container.appendChild(suffixEl);
            }
            if (titleEl.parentElement !== container) {
                container.appendChild(titleEl);
            }
            if (inputEl && inputEl.parentElement !== container) {
                container.appendChild(inputEl);
            }
            if (toggleEl.parentElement !== container) {
                container.appendChild(toggleEl);
            }
            if (ta.parentElement !== container) {
                container.appendChild(ta);
            }
        }
        
        // 视觉反馈：如果未启用，降低不透明度
        if (item.enabled === false && !isSelected) {
            cardEl.style.opacity = '0.5';
            titleEl.style.color = '#888';
            ta.style.color = '#888';
        } else {
            cardEl.style.opacity = '1';
            titleEl.style.color = '#eee';
            ta.style.color = '#eee';
        }

        // 设置可见性
        let shouldShow;
        const hand = isHandMode();
        const hidePrompts = !!node.__hidePrompts;
        
        if (useScrollMode && scrollContainer) {
            shouldShow = node.flags?.collapsed !== true;
        } else {
            const nodeVisibleX = sx + sw > 0 && sx < (parentRect.width || rect.width);
            const nodeVisibleY = sy + sh > 0 && sy < (parentRect.height || rect.height);
            shouldShow = node.flags?.collapsed !== true && nodeVisibleX && nodeVisibleY;
        }
        
        cardEl.style.visibility = shouldShow ? 'visible' : 'hidden';
        titleEl.style.visibility = shouldShow ? 'visible' : 'hidden';
        ta.style.visibility = shouldShow && !hidePrompts ? 'visible' : 'hidden';
        toggleEl.style.visibility = shouldShow ? 'visible' : 'hidden';
        if (inputEl) inputEl.style.visibility = shouldShow ? 'visible' : 'hidden';
        if (suffixEl) suffixEl.style.visibility = shouldShow ? 'visible' : 'hidden';
        
        titleEl.style.pointerEvents = shouldShow && !hand ? 'auto' : 'none';
        ta.style.pointerEvents = shouldShow && !hand && !hidePrompts ? 'auto' : 'none';
        toggleEl.style.pointerEvents = shouldShow && !hand ? 'auto' : 'none';
        if (inputEl) inputEl.style.pointerEvents = shouldShow && !hand ? 'auto' : 'none';
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
    for (let j = items.length; j < (node.__suffixEls?.length || 0); j++) {
        const el = node.__suffixEls[j];
        if (el && el.remove) el.remove();
    }
    for (let j = items.length; j < (node.__toggleEls?.length || 0); j++) {
        const el = node.__toggleEls[j];
        if (el && el.remove) el.remove();
    }
    for (let j = items.length; j < (node.__inputEls?.length || 0); j++) {
        const el = node.__inputEls[j];
        if (el && el.remove) el.remove();
    }
    for (let j = items.length; j < (node.__cardEls?.length || 0); j++) {
        const el = node.__cardEls[j];
        if (el && el.remove) el.remove();
    }
    
    node.__taEls.length = items.length;
    node.__titleEls.length = items.length;
    node.__suffixEls.length = items.length;
    node.__toggleEls.length = items.length;
    node.__inputEls.length = items.length;
    node.__cardEls.length = items.length;
    
    // 清理滚动容器（如果不再需要）
    if (!useScrollMode && node.__gridScrollContainer) {
        node.__gridScrollContainer.remove();
        node.__gridScrollContainer = null;
    }
    
    // 更新样式以反映当前选中的索引
    updateTextareaStyles(node);
}

// {{ AURA-X: Add - 计算布局单元格位置，为标题+内容预留更多垂直空间. }}
// {{ AURA-X: Modify - 增大最小高度，添加滚动条支持，限制最大内容区域高度. }}
// {{ AURA-X: Modify - 添加滑条控制最小高度，添加左右边距避免滚动条遮挡. }}
function layoutCells(node, items) {
    const PADDING = 8;
    const GAP = 6;
    const SCROLL_PADDING = 16; // 左右边距
    const n = items.length;
    if (n === 0) return [];

    const viewMode = node.properties?._viewMode || "grid";
    
    // 从widget获取最小高度，默认120
    const minHeightWidget = node.widgets?.find(w => w.name === "cell_min_height");
    const MIN_H = minHeightWidget ? Math.max(72, Math.min(300, Number(minHeightWidget.value))) : 120;
    const HIDE_PROMPT_H = 36; // 隐藏提示词时只显示标题行的高度
    const MAX_CONTENT_H = 500;

    const BUTTON_AREA_H = 70;
    const startY = PADDING + getWidgetsBottom(node);
    
    const hidePrompts = !!node.__hidePrompts;
    const cellH = hidePrompts ? HIDE_PROMPT_H : MIN_H;

    if (node.flags?.collapsed) {
        if (viewMode === "combo") {
             return [{ x: PADDING, y: startY, w: Math.max(0, node.size[0] - PADDING * 2), h: MIN_H }];
        }
        const cells = [];
        for (let i = 0; i < n; i++) {
             cells.push({ x: PADDING, y: startY, w: 10, h: 10 });
        }
        return cells;
    }

    if (viewMode === "combo") {
        const minTotalH = startY + MIN_H + PADDING + BUTTON_AREA_H;
        if (node.size[1] < minTotalH) {
             if (typeof node.setSize === 'function') {
                node.setSize([node.size[0], minTotalH]);
            } else {
                node.size[1] = minTotalH;
            }
            app.graph.setDirtyCanvas(true, true);
        }
        
        const availH = Math.max(MIN_H, node.size[1] - startY - PADDING - BUTTON_AREA_H);
        const availW = node.size[0] - PADDING * 2;
        
        return [{ x: PADDING, y: startY, w: availW, h: availH }];
    }

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
    const SCROLL_CONTAINER_PADDING = 12; // 滚动容器内部右边距
    const availW = node.size[0] - PADDING * 2 - SCROLL_PADDING - SCROLL_CONTAINER_PADDING; // 减去边距和滚动条区域
    const cellW = Math.floor((availW - GAP * (cols - 1)) / cols);
    
    const requiredH = rows * cellH + GAP * Math.max(0, rows - 1);
    const useScrollMode = requiredH > MAX_CONTENT_H;
    
    const contentH = useScrollMode ? MAX_CONTENT_H : requiredH;
    const minTotalH = startY + contentH + PADDING + BUTTON_AREA_H;
    
    // 只在高度不足时自动扩展，不强制固定高度，允许用户手动调整
    if (node.size[1] < minTotalH) {
        if (typeof node.setSize === 'function') {
            node.setSize([node.size[0], minTotalH]);
        } else {
            node.size[1] = minTotalH;
        }
        app.graph.setDirtyCanvas(true, true);
    }

    const totalContentHeight = rows * cellH + GAP * Math.max(0, rows - 1);
    
    // 计算实际可见高度（基于用户调整后的节点高度）
    const actualContentH = Math.max(contentH, node.size[1] - startY - PADDING - BUTTON_AREA_H);

    const cells = [];
    for (let i = 0; i < n; i++) {
        const r = Math.floor(i / cols);
        const c = i % cols;
        const x = PADDING + SCROLL_PADDING / 2 + c * (cellW + GAP); // 添加边距偏移
        const y = r * (cellH + GAP);
        cells.push({ x, y, w: cellW, h: cellH, scrollY: y });
    }
    
    node.__useScrollMode = useScrollMode;
    node.__totalContentHeight = totalContentHeight;
    node.__visibleContentHeight = actualContentH;
    node.__scrollPadding = SCROLL_PADDING;
    
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
        
        // 即使折叠也要更新DOM元素的可见性
        if (this.flags?.collapsed) {
             relayoutAndUpdate(ctx);
             return;
        }
        
        // 绘制自定义按钮
        
        const buttonHeight = 25;
        const buttonSpacing = 10;
        const rowSpacing = 5;
        const startX = 10;
        const nodeWidth = this.size[0];

        const currentViewMode = this.properties?._viewMode || "grid";

        // 按钮定义
        // 顺序：视图切换(左一) -> 全选 -> 全不选 -> 反选 -> 导入 -> 导出 -> 批量删除
        const buttons = [
            { 
                text: currentViewMode === 'combo' ? '切换: 列表' : '切换: 下拉', 
                width: 80,
                callback: () => {
                    this.properties._viewMode = (this.properties._viewMode === 'combo') ? 'grid' : 'combo';
                    const items = getItems(this);
                    const layout = layoutCells(this, items);
                    ensureTextareas(this, layout, items);
                    app.graph.setDirtyCanvas(true, true);
                }
            },
            { 
                text: '全选', 
                width: 50,
                callback: () => {
                    const items = getItems(this);
                    items.forEach(item => item.enabled = true);
                    setItems(this, items);
                    const layout = layoutCells(this, items);
                    ensureTextareas(this, layout, items);
                    app.graph.setDirtyCanvas(true, true);
                }
            },
            { 
                text: '全不选', 
                width: 60,
                callback: () => {
                    const items = getItems(this);
                    items.forEach(item => item.enabled = false);
                    setItems(this, items);
                    const layout = layoutCells(this, items);
                    ensureTextareas(this, layout, items);
                    app.graph.setDirtyCanvas(true, true);
                }
            },
            { 
                text: '反选', 
                width: 50,
                callback: () => {
                    const items = getItems(this);
                    items.forEach(item => item.enabled = !item.enabled);
                    setItems(this, items);
                    const layout = layoutCells(this, items);
                    ensureTextareas(this, layout, items);
                    app.graph.setDirtyCanvas(true, true);
                }
            },
            { 
                text: '📥 导入', 
                width: 70,
                callback: () => handleImport(this)
            },
            { 
                text: '📤 导出', 
                width: 70,
                callback: () => handleExport(this)
            },
            { 
                text: '🗑️ 删除选中', 
                width: 90,
                callback: () => handleBatchDelete(this)
            },
            { 
                text: this.__hidePrompts ? '📝 显示提示词' : '📋 隐藏提示词', 
                width: 100,
                callback: () => {
                    this.__hidePrompts = !this.__hidePrompts;
                    const items = getItems(this);
                    const layout = layoutCells(this, items);
                    ensureTextareas(this, layout, items);
                    app.graph.setDirtyCanvas(true, true);
                }
            }
        ];

        // 计算布局行
        const rows = [];
        let currentRow = [];
        let currentRowWidth = startX;
        
        buttons.forEach(btn => {
            if (currentRowWidth + btn.width + buttonSpacing > nodeWidth - 10) { // -10 padding right
                if (currentRow.length > 0) {
                    rows.push(currentRow);
                    currentRow = [];
                    currentRowWidth = startX;
                }
            }
            currentRow.push(btn);
            currentRowWidth += btn.width + buttonSpacing;
        });
        if (currentRow.length > 0) rows.push(currentRow);

        // 计算起始Y坐标，使按钮组靠底部对齐
        // 假设底部预留区域足够大，我们将按钮组放在底部
        const totalHeight = rows.length * buttonHeight + (rows.length - 1) * rowSpacing;
        // 底部留 5px
        let startY = this.size[1] - totalHeight - 5;
        
        // 清空点击区域缓存
        this._customButtons = [];
        
        const r = 6;
        function drawButton(ctx, x, y, w, h, text, hover) {
            ctx.fillStyle = hover ? 'rgba(235,235,240,0.95)' : 'rgba(235,235,240,0.85)';
            ctx.strokeStyle = hover ? 'rgba(80,80,90,0.9)' : 'rgba(120,120,130,0.8)';
            ctx.lineWidth = hover ? 2 : 1;
            ctx.beginPath();
            
            if (ctx.roundRect) {
                ctx.roundRect(x, y, w, h, r);
            } else {
                ctx.moveTo(x + r, y);
                ctx.lineTo(x + w - r, y);
                ctx.quadraticCurveTo(x + w, y, x + w, y + r);
                ctx.lineTo(x + w, y + h - r);
                ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
                ctx.lineTo(x + r, y + h);
                ctx.quadraticCurveTo(x, y + h, x, y + h - r);
                ctx.lineTo(x, y + r);
                ctx.quadraticCurveTo(x, y, x + r, y);
            }
            
            ctx.closePath();
            ctx.fill();
            ctx.stroke();
            ctx.fillStyle = 'rgba(30,30,35,1)';
            ctx.font = 'bold 12px "Microsoft YaHei", Arial';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(text, x + w / 2, y + h / 2);
        }

        // 绘制每一行
        rows.forEach((row, rowIndex) => {
            let x = startX;
            let y = startY + rowIndex * (buttonHeight + rowSpacing);
            
            row.forEach(btn => {
                const hover = this._customMouseX >= x && this._customMouseX <= x + btn.width &&
                              this._customMouseY >= y && this._customMouseY <= y + buttonHeight;
                
                drawButton(ctx, x, y, btn.width, buttonHeight, btn.text, hover);
                
                // 记录点击区域
                this._customButtons.push({
                    rect: { x, y, w: btn.width, h: buttonHeight },
                    callback: btn.callback
                });
                
                x += btn.width + buttonSpacing;
            });
        });
        
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
        
        // 清理 Grid 模式元素
        if (this.__taEls) {
            this.__taEls.forEach(el => el && el.remove());
            this.__taEls = null;
        }
        if (this.__titleEls) {
            this.__titleEls.forEach(el => el && el.remove());
            this.__titleEls = null;
        }
        if (this.__suffixEls) {
            this.__suffixEls.forEach(el => el && el.remove());
            this.__suffixEls = null;
        }
        if (this.__toggleEls) {
            this.__toggleEls.forEach(el => el && el.remove());
            this.__toggleEls = null;
        }
        if (this.__inputEls) {
            this.__inputEls.forEach(el => el && el.remove());
            this.__inputEls = null;
        }
        
        // 清理 Combo 模式元素
        if (this.__comboSelect) { this.__comboSelect.remove(); this.__comboSelect = null; }
        if (this.__comboTextarea) { this.__comboTextarea.remove(); this.__comboTextarea = null; }
        if (this.__comboInputEl) { this.__comboInputEl.remove(); this.__comboInputEl = null; }
        if (this.__comboTitleInput) { this.__comboTitleInput.remove(); this.__comboTitleInput = null; }
        if (this.__comboSuffixEl) { this.__comboSuffixEl.remove(); this.__comboSuffixEl = null; }
        if (this.__comboMenuBtn) { this.__comboMenuBtn.remove(); this.__comboMenuBtn = null; }
        if (this.__comboToggleEl) { this.__comboToggleEl.remove(); this.__comboToggleEl = null; }
        
        // 清理卡片容器
        if (this.__cardEls) {
            this.__cardEls.forEach(el => el && el.remove());
            this.__cardEls = null;
        }
        
        // 清理滚动容器
        if (this.__gridScrollContainer) { this.__gridScrollContainer.remove(); this.__gridScrollContainer = null; }
        
        // 隐藏 Tooltip
        Tooltip.hide();
        
        // 清理按钮引用
        this._customButtons = null;
    };
    
    // 添加交互事件处理
    node.onMouseDown = function(e) {
        if (this.flags?.collapsed) return false;
        
        const x = e.canvasX - this.pos[0];
        const y = e.canvasY - this.pos[1];
        
        if (this._customButtons) {
            for (const btn of this._customButtons) {
                if (x >= btn.rect.x && x <= btn.rect.x + btn.rect.w &&
                    y >= btn.rect.y && y <= btn.rect.y + btn.rect.h) {
                    if (btn.callback) {
                        btn.callback();
                        return true; // 阻止事件传播
                    }
                }
            }
        }
        
        return false;
    };
    
    // 添加 onMouseMove 以支持悬浮效果
    node.onMouseMove = function(e) {
        // 计算相对于节点的坐标
        const x = e.canvasX - this.pos[0];
        const y = e.canvasY - this.pos[1];
        
        this._customMouseX = x;
        this._customMouseY = y;
        
        // 简单判断是否在按钮区域，触发重绘
        // 为了性能，可以只在进入/离开按钮区域时 setDirty
        // 这里简化处理，只要移动就重绘（注意性能，如果卡顿则需要优化）
        // 由于是 Canvas 绘制，悬浮变色需要重绘
        // 只有当鼠标在底部区域时才重绘
        if (y > this.size[1] - 80) { // 更新为 80 以匹配新的按钮区域
             app.graph.setDirtyCanvas(true, false);
        }
    };
    
    // 鼠标离开节点时清除状态
    node.onMouseLeave = function(e) {
        this._customMouseX = undefined;
        this._customMouseY = undefined;
        app.graph.setDirtyCanvas(true, false);
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

function installViewportSync(node) {
    if (node.__viewportSyncInstalled) return;
    const ds = app?.canvas?.ds;
    const canvas = app?.canvas?.canvas;
    if (!ds || !canvas) return;
    node.__viewportSyncInstalled = true;
    let lastScale = ds.scale, lastX = ds.offset[0], lastY = ds.offset[1];
    let lastLeft = 0, lastTop = 0;
    const tick = () => {
        const rect = canvas.getBoundingClientRect();
        const changed = ds.scale !== lastScale || ds.offset[0] !== lastX || ds.offset[1] !== lastY || rect.left !== lastLeft || rect.top !== lastTop;
        if (changed) {
            lastScale = ds.scale; lastX = ds.offset[0]; lastY = ds.offset[1];
            lastLeft = rect.left; lastTop = rect.top;
            const items = getItems(node);
            const cells = layoutCells(node, items);
            ensureTextareas(node, cells, items);
        }
        node.__rafId = requestAnimationFrame(tick);
    };
    node.__rafId = requestAnimationFrame(tick);
    const hide = () => {
        if (node.__taEls) node.__taEls.forEach(el => { if (el && el.style) { el.style.visibility = 'hidden'; el.style.pointerEvents = 'none'; } });
        if (node.__titleEls) node.__titleEls.forEach(el => { if (el && el.style) { el.style.visibility = 'hidden'; el.style.pointerEvents = 'none'; } });
        if (node.__suffixEls) node.__suffixEls.forEach(el => { if (el && el.style) { el.style.visibility = 'hidden'; el.style.pointerEvents = 'none'; } });
        if (node.__toggleEls) node.__toggleEls.forEach(el => { if (el && el.style) { el.style.visibility = 'hidden'; el.style.pointerEvents = 'none'; } });
        if (node.__inputEls) node.__inputEls.forEach(el => { if (el && el.style) { el.style.visibility = 'hidden'; el.style.pointerEvents = 'none'; } });
        if (node.__cardEls) node.__cardEls.forEach(el => { if (el && el.style) { el.style.visibility = 'hidden'; el.style.pointerEvents = 'none'; } });
        if (node.__gridScrollContainer && node.__gridScrollContainer.style) { node.__gridScrollContainer.style.visibility = 'hidden'; node.__gridScrollContainer.style.pointerEvents = 'none'; }
        
        // Hide Combo elements
        if (node.__comboSelect && node.__comboSelect.style) { node.__comboSelect.style.visibility = 'hidden'; node.__comboSelect.style.pointerEvents = 'none'; }
        if (node.__comboTextarea && node.__comboTextarea.style) { node.__comboTextarea.style.visibility = 'hidden'; node.__comboTextarea.style.pointerEvents = 'none'; }
        if (node.__comboInputEl && node.__comboInputEl.style) { node.__comboInputEl.style.visibility = 'hidden'; node.__comboInputEl.style.pointerEvents = 'none'; }
        
        // Hide new Combo elements
        if (node.__comboTitleInput && node.__comboTitleInput.style) { node.__comboTitleInput.style.visibility = 'hidden'; node.__comboTitleInput.style.pointerEvents = 'none'; }
        if (node.__comboSuffixEl && node.__comboSuffixEl.style) { node.__comboSuffixEl.style.visibility = 'hidden'; node.__comboSuffixEl.style.pointerEvents = 'none'; }
        if (node.__comboMenuBtn && node.__comboMenuBtn.style) { node.__comboMenuBtn.style.visibility = 'hidden'; node.__comboMenuBtn.style.pointerEvents = 'none'; }
        if (node.__comboToggleEl && node.__comboToggleEl.style) { node.__comboToggleEl.style.visibility = 'hidden'; node.__comboToggleEl.style.pointerEvents = 'none'; }
        if (node.__customDropdown && node.__customDropdown.style) { node.__customDropdown.style.visibility = 'hidden'; node.__customDropdown.style.pointerEvents = 'none'; }
    };
    const show = () => {
        const items = getItems(node);
        const cells = layoutCells(node, items);
        ensureTextareas(node, cells, items);
    };
    const canvasEl = canvas;
    node.__onWheel = () => { hide(); requestAnimationFrame(show); };
    node.__onMouseDown = () => { hide(); requestAnimationFrame(show); };

    canvasEl.addEventListener('wheel', node.__onWheel, { passive: true, capture: true });
    canvasEl.addEventListener('mousedown', node.__onMouseDown, { capture: true });

    const origRemoved = node.onRemoved;
    node.onRemoved = function() {
        if (origRemoved) origRemoved.call(this);
        if (node.__rafId) cancelAnimationFrame(node.__rafId);
        canvasEl.removeEventListener('wheel', node.__onWheel, { capture: true });
        canvasEl.removeEventListener('mousedown', node.__onMouseDown, { capture: true });

        node.__viewportSyncInstalled = false;
    };
}

function initDomRefs(node) {
    const props = [
        "__taEls", "__titleEls", "__suffixEls", "__toggleEls", "__inputEls", "__cardEls",
        "__comboSelect", "__comboTextarea", "__comboInputEl",
        "__comboTitleInput", "__comboSuffixEl", "__comboMenuBtn", "__comboToggleEl",
        "__customDropdown", "__gridScrollContainer",
        "__viewportSyncInstalled", "__indexListenerInstalled", "__addButtonInstalled",
        "__drawingInstalled", "__rafId", "__onWheel", "__onMouseDown", "__indexCheckInterval",
        "_customSelectAllButtonRect", "_customDeselectAllButtonRect", 
        "_customInvertSelectionButtonRect", "_customViewModeButtonRect",
        "_customButtons",
        "_customMouseX", "_customMouseY"
    ];
    
    props.forEach(p => {
        if (!Object.getOwnPropertyDescriptor(node, p)) {
            Object.defineProperty(node, p, {
                value: undefined,
                writable: true,
                enumerable: false,
                configurable: true
            });
        }
    });
}

app.registerExtension({
    name: "A_my_nodes.TextInputBatch.UI",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "TextInputBatch") return;
        console.log("[TextInputBatch] UI扩展注册");

        // {{ AURA-X: Add - 强制重置 DOM 引用的清理函数 }}
        function clearDomRefs(node) {
            const propsToClear = [
                "__taEls", "__titleEls", "__suffixEls", "__toggleEls", "__inputEls", "__cardEls",
                "__comboSelect", "__comboTextarea", "__comboInputEl",
                "__comboTitleInput", "__comboSuffixEl", "__comboMenuBtn", "__comboToggleEl",
                "__customDropdown", "__gridScrollContainer",
                "__viewportSyncInstalled", "__indexListenerInstalled", "__addButtonInstalled",
                "__drawingInstalled", "__rafId", "__onWheel", "__onMouseDown", "__indexCheckInterval",
                "_customSelectAllButtonRect", "_customDeselectAllButtonRect", 
                "_customInvertSelectionButtonRect", "_customViewModeButtonRect",
                "_customButtons",
                "_customMouseX", "_customMouseY",
                "__useScrollMode", "__totalContentHeight", "__visibleContentHeight", "__scrollPadding",
                "__hidePrompts"
            ];
            propsToClear.forEach(p => {
                if (node.hasOwnProperty(p)) {
                    const el = node[p];
                    if (el) {
                        if (Array.isArray(el)) {
                            el.forEach(e => { if (e && e.remove) e.remove(); });
                        } else if (el.remove) {
                            el.remove();
                        }
                    }
                    node[p] = undefined;
                }
            });
        }

        // {{ AURA-X: Add - 重写 clone 方法，确保 DOM 引用不会被复制 }}
        nodeType.prototype.clone = function() {
            const newNode = LiteGraph.LGraphNode.prototype.clone.call(this);
            clearDomRefs(newNode);
            
            // 重新初始化节点组件
            initDomRefs(newNode);
            ensureStringsJsonWidget(newNode);
            installAddButton(newNode);
            // installExtraButtons(newNode);
            installDrawingHandlers(newNode);
            installViewportSync(newNode);
            installIndexChangeListener(newNode);
            bindColumnsChange(newNode);
            bindMinHeightChange(newNode);
            setItems(newNode, getItems(newNode));
            
            return newNode;
        };

        const origOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function() {
            if (origOnNodeCreated) origOnNodeCreated.apply(this, arguments);
            initDomRefs(this);
            ensureStringsJsonWidget(this);
            
            // 添加提示词最小高度控制滑条
            if (!this.widgets?.find(w => w.name === "cell_min_height")) {
                const minHeightWidget = this.addWidget("number", "cell_min_height", 120, () => {}, { min: 72, max: 300, step: 8 });
                minHeightWidget.options.serialize = true;
            }
            
            // installSelectionTools(this); // Removed in favor of canvas buttons
            installAddButton(this);
            // installExtraButtons(this);
            installDrawingHandlers(this);
            installViewportSync(this);
            installIndexChangeListener(this);
            bindColumnsChange(this);
            bindMinHeightChange(this);
            
            // 重要：推迟 DOM 创建！
            // 不要在这里调用 setItems(this, getItems(this));
            // 因为 setItems 会调用 layoutCells -> ensureTextareas -> 创建 DOM。
            // 如果我们在 clone 过程中，这会导致创建一套无用的 DOM，然后被 clone 的属性覆盖，导致泄漏。
            
            // 我们只初始化数据，不创建 DOM。
            // DOM 创建延迟到 onAdded 或第一次 onDrawForeground。
        };
        
        // {{ AURA-X: Add - 处理 onAdded 事件，延迟创建 DOM }}
        const origOnAdded = nodeType.prototype.onAdded;
        nodeType.prototype.onAdded = function(graph) {
            if (origOnAdded) origOnAdded.apply(this, arguments);
            
            // 节点被添加到画布时，安全地创建 DOM
            // 此时如果是 clone 操作，属性覆盖已经完成，this.__taEls 可能指向旧节点 DOM（如果有脏引用）
            // 我们需要先清理脏引用，再创建自己的 DOM
            
            clearDomRefs(this); // 确保干净
            setItems(this, getItems(this)); // 创建 DOM
        };
        
        // {{ AURA-X: Add - 处理 onConfigure 事件，确保 DOM 更新 }}
        const origConfigure = nodeType.prototype.configure;
        nodeType.prototype.configure = function(info) {
            if (origConfigure) origConfigure.apply(this, arguments);
            
            // 反序列化后，同样可能带有脏引用（虽然 JSON 不含 DOM，但如果 LiteGraph 内部做了什么奇怪的事）
            // 或者如果是粘贴操作，configure 会在 onNodeCreated 之后调用。
            
            // 清理并重建
            clearDomRefs(this);
            
            if (info && info.properties && typeof info.properties._strings === 'string') {
                this.properties = this.properties || {};
                this.properties._strings = info.properties._strings;
                const hidden = ensureStringsJsonWidget(this);
                hidden.value = this.properties._strings;
            }
            
            // 这里是否应该创建 DOM？
            // 如果节点还没被 add 到 graph，创建 DOM 可能会有问题（parent 不存在？）
            // ensureTextareas 会尝试 append 到 document.body 或 canvas.parentNode。
            // 如果 app.canvas 存在，应该没问题。
            
            // 为了安全，如果节点不在 graph 中，我们可以推迟到 onAdded。
            if (this.graph) {
                setItems(this, getItems(this));
            }
        };
    },
});
// =============================
// DOM 生命周期管理系统
// =============================

// 统一清理函数
function cleanupNodeDom(node) {
    const keys = [
        "__comboTextarea",
        "__comboSelect",
        "__comboTitleInput",
        "__comboSuffixEl",
        "__comboMenuBtn",
        "__comboToggleEl",
        "__comboInputEl",
        "__customDropdown",
        "__gridScrollContainer",
        "__taEls",
        "__titleEls",
        "__suffixEls",
        "__toggleEls",
        "__inputEls",
        "__cardEls"
    ];

    keys.forEach(k => {
        const el = node[k];
        if (!el) return;

        if (Array.isArray(el)) {
            el.forEach(e => {
                if (e && e.remove) e.remove();
            });
        } else {
            if (el.remove) el.remove();
        }

        node[k] = null;
    });
}


// 强制重置 DOM 引用（解决 clone 复制污染）
function resetNodeDomRefs(node) {
    node.__comboTextarea = null;
    node.__comboSelect = null;
    node.__comboTitleInput = null;
    node.__comboSuffixEl = null;
    node.__comboMenuBtn = null;
    node.__comboToggleEl = null;
    node.__comboInputEl = null;
    node.__customDropdown = null;
    node.__gridScrollContainer = null;
    node.__hidePrompts = undefined;

    node.__taEls = [];
    node.__titleEls = [];
    node.__suffixEls = [];
    node.__toggleEls = [];
    node.__inputEls = [];
    node.__cardEls = [];
}


// 安装节点生命周期
function installNodeLifecycle(node) {

    if (node.__lifecycleInstalled) return;
    node.__lifecycleInstalled = true;

    // --- 节点删除 ---
    const origRemoved = node.onRemoved;
    node.onRemoved = function () {
        cleanupNodeDom(this);
        if (origRemoved) origRemoved.apply(this, arguments);
    };

    // --- 节点添加（包括 clone） ---
    const origAdded = node.onAdded;
    node.onAdded = function () {

        // 关键：先清理 clone 遗留 DOM
        cleanupNodeDom(this);

        // 重置引用，避免复制污染
        resetNodeDomRefs(this);

        if (origAdded) origAdded.apply(this, arguments);
    };

    // --- 节点配置（加载json时） ---
    const origConfigure = node.onConfigure;
    node.onConfigure = function () {

        cleanupNodeDom(this);
        resetNodeDomRefs(this);

        if (origConfigure) origConfigure.apply(this, arguments);
    };
}


// =============================
// 防幽灵检测（必须放在 ensureTextareas 开头）
// =============================

const __origEnsureTextareas = ensureTextareas;
ensureTextareas = function(node, layout, items) {

    // 如果节点已经不在 graph 中
    if (!node.graph || !node.graph._nodes || node.graph._nodes.indexOf(node) === -1) {
        cleanupNodeDom(node);
        return;
    }

    return __origEnsureTextareas(node, layout, items);
};


// =============================
// 自动为目标节点安装生命周期
// =============================

app.registerExtension({
    name: "text_input_batch_dom_fix",

    beforeRegisterNodeDef(nodeType, nodeData) {

        // 修改为你的节点名字
        if (nodeData.name !== "Text Input Batch") return;

        const origOnNodeCreated = nodeType.prototype.onNodeCreated;

        nodeType.prototype.onNodeCreated = function () {

            if (origOnNodeCreated)
                origOnNodeCreated.apply(this, arguments);

            // 安装生命周期
            installNodeLifecycle(this);

            // 初始化引用
            resetNodeDomRefs(this);
        };
    }
});
