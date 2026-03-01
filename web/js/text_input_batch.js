import { app } from "../../../scripts/app.js";
import { rgthree } from "./rgthree.js"; // 统一右键菜单定位使用的事件来源

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
            alert('至少需要保留一个提示词输入！');
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
    if (edgeState.count < 3) {
        e.stopPropagation(); // 还没达到阈值
        return;
    }
    
    // 达到阈值，触发 Canvas 滚动
    const canvasEl = app?.canvas?.canvas;
    if (!canvasEl || typeof WheelEvent === 'undefined') {
        return;
    }
    
    // 构造新的事件转发给 canvas
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
                
                // 显示上下文菜单作为下拉列表
                const Lite = window.LiteGraph || window?.app?.canvas?.graph?.constructor;
                if (!Lite || !Lite.ContextMenu) return;
                
                // 重新获取 items 以确保标题最新
                const currentItems = getItems(node);
                
                const menuItems = currentItems.map((item, idx) => {
                    return {
                        content: `${item.title || `prompt_${idx}`}`, // 显示完整标题
                        callback: () => {
                            setIndexSelectorValue(node, idx);
                        },
                        // LiteGraph ContextMenu 默认不高亮当前项，这里可以通过 title 或其他方式辅助
                        // 或者如果支持的话，使用 checked 属性
                        checked: idx === currentIndex
                    };
                });
                
                const cm = new Lite.ContextMenu(menuItems, {
                    event: e,
                    parentMenu: null,
                    node: node
                });

                // {{ AURA-X: Add - 为菜单项添加悬浮提示 }}
                const root = cm.root || cm.element || cm.menu || cm;
                if (root) {
                    const entries = Array.from(root.querySelectorAll('.litemenu-entry'));
                    entries.forEach((entry, idx) => {
                        // 确保索引对应（假设没有分隔符）
                        if (idx < currentItems.length) {
                            const item = currentItems[idx];
                            entry.addEventListener('mouseenter', (evt) => {
                                const rect = entry.getBoundingClientRect();
                                Tooltip.scheduleShow(rect.right + 5, rect.top, item.title, item.content);
                            });
                            entry.addEventListener('mouseleave', () => {
                                Tooltip.hide();
                            });
                        }
                    });
                    
                    // Hook close 方法以确保菜单关闭时隐藏提示
                    const origClose = cm.close;
                    cm.close = function() {
                        Tooltip.hide();
                        if (origClose) origClose.apply(this, arguments);
                    };
                }
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
        
        return; // Combo 模式处理完毕
    }

    // --- Grid 模式 (清理 Combo 元素) ---
    clearComboElements();

    // 初始化元素数组
    if (!node.__taEls) node.__taEls = [];
    if (!node.__titleEls) node.__titleEls = [];
    if (!node.__suffixEls) node.__suffixEls = []; // 新增后缀标签数组
    if (!node.__toggleEls) node.__toggleEls = [];
    if (!node.__inputEls) node.__inputEls = []; // 新增输入开关数组

    // const currentIndex = getCurrentIndex(node); // 已在上面定义

    for (let i = 0; i < items.length; i++) {
        const cell = layout[i];
        if (!cell) continue;
        
        const item = items[i];
        const isSelected = i === currentIndex;
        const baseTitle = getBaseTitle(item.title); // 获取不带后缀的标题
        const suffixText = `${i}`; // 只显示数字，不带下划线
        
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
            titleEl.style.cssText = `position: absolute; z-index: 100; padding: 4px 6px; border-radius: 6px 6px 0 0; border: 1px solid #666; border-bottom: none; background: #3a3a3a; color: #eee; font-size: 11px; line-height: 1.2; font-family: "Microsoft YaHei", "SimHei", Arial, monospace; box-sizing: border-box; transform-origin: 0 0;`;
            
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
            ta.style.cssText = `position: absolute; z-index: 100; resize: none; padding: 6px; border-radius: 0 0 6px 6px; border: 1px solid #666; border-top: none; background: #222; color: #eee; font-size: 12px; line-height: 1.4; font-family: "Microsoft YaHei", "SimHei", Arial, monospace; box-sizing: border-box; overflow: auto; transform-origin: 0 0;`;
            
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
        const sx = (node.pos[0] + cell.x + ds.offset[0]) * ds.scale + rect.left - parentRect.left;
        const sy = (node.pos[1] + cell.y + ds.offset[1]) * ds.scale + rect.top - parentRect.top;
        const sw = cell.w * ds.scale;
        const sh = cell.h * ds.scale;
        
        // 标题输入框位置（在内容框上方）
        const titleHeight = 24;
        const toggleWidth = 20; // 开关宽度
        const inputBtnWidth = 16; // 输入连接点开关宽度
        
        // 计算后缀标签宽度
        // 简单估算：每个字符约 7px + padding
        const suffixWidth = Math.max(16, Math.ceil(suffixText.length * 7) + 4);

        // 设置开关位置和大小
        toggleEl.style.left = `${sx + sw - toggleWidth * ds.scale - 2}px`; // 靠右
        toggleEl.style.top = `${sy + 4 * ds.scale}px`; // 垂直居中微调
        toggleEl.style.width = `${16 * ds.scale}px`;
        toggleEl.style.height = `${16 * ds.scale}px`;
        toggleEl.style.transform = `scale(${1})`; 

        // 设置输入连接点开关位置
        if (inputEl) {
            inputEl.style.left = `${sx + sw - (toggleWidth + inputBtnWidth) * ds.scale - 4}px`; // 开关左侧
            inputEl.style.top = `${sy + 4 * ds.scale}px`;
            inputEl.style.width = `${16 * ds.scale}px`;
            inputEl.style.height = `${16 * ds.scale}px`;
            inputEl.style.fontSize = `${12 * ds.scale}px`;
            inputEl.style.lineHeight = `${16 * ds.scale}px`;
            inputEl.style.display = 'block'; // 强制显示
        }
        
        // 设置标题输入框位置和大小（使用CSS缩放保持与节点比例一致）
        // 标题宽度减少，为开关和后缀留出空间
        titleEl.style.left = `${sx}px`;
        titleEl.style.top = `${sy}px`;
        const titleAvailableW = Math.max(20, Math.round(cell.w - toggleWidth - inputBtnWidth - suffixWidth - 8));
        titleEl.style.width = `${titleAvailableW}px`; // 减去开关宽度、后缀宽度和间距
        titleEl.style.height = `${Math.round(titleHeight)}px`;
        titleEl.style.transform = `scale(${ds.scale})`;
        
        // 设置后缀标签位置
        // 紧跟在标题输入框右侧
        // 注意：titleEl 已经缩放，left 是物理坐标，但 transform-origin 是 0 0
        // titleEl 占据的物理宽度是 titleAvailableW * ds.scale
        suffixEl.style.left = `${sx + titleAvailableW * ds.scale}px`;
        suffixEl.style.top = `${sy}px`;
        suffixEl.style.width = `${suffixWidth * ds.scale}px`;
        suffixEl.style.height = `${titleHeight * ds.scale}px`;
        suffixEl.style.transform = `scale(${ds.scale})`;
        suffixEl.style.transformOrigin = "0 0"; // 确保缩放原点正确
        
        // 设置内容文本框位置和大小（使用CSS缩放保持与节点比例一致）
        ta.style.left = `${sx}px`;
        ta.style.top = `${sy + titleHeight * ds.scale}px`;
        ta.style.width = `${Math.max(40, Math.round(cell.w))}px`;
        ta.style.height = `${Math.max(32, Math.round(cell.h - titleHeight))}px`;
        ta.style.transform = `scale(${ds.scale})`;
        
        const fontPx = 12;
        const titleFontPx = 11;
        titleEl.style.fontSize = `${titleFontPx}px`;
        ta.style.fontSize = `${fontPx}px`;
        
        // 视觉反馈：如果未启用，降低不透明度
        if (item.enabled === false && !isSelected) {
            titleEl.style.opacity = '0.5';
            ta.style.opacity = '0.5';
            titleEl.style.color = '#888';
            ta.style.color = '#888';
        } else {
            titleEl.style.opacity = '1';
            ta.style.opacity = '1';
            titleEl.style.color = '#eee';
            ta.style.color = '#eee';
        }

        // 设置可见性 - 节点未折叠且在容器可视范围内才显示
        const nodeVisibleX = sx + sw > 0 && sx < (parentRect.width || rect.width);
        const nodeVisibleY = sy + sh > 0 && sy < (parentRect.height || rect.height);
        const shouldShow = node.flags?.collapsed !== true && nodeVisibleX && nodeVisibleY;
        const hand = isHandMode();
        
        titleEl.style.visibility = shouldShow ? 'visible' : 'hidden';
        ta.style.visibility = shouldShow ? 'visible' : 'hidden';
        toggleEl.style.visibility = shouldShow ? 'visible' : 'hidden';
        if (inputEl) inputEl.style.visibility = shouldShow ? 'visible' : 'hidden';
        if (suffixEl) suffixEl.style.visibility = shouldShow ? 'visible' : 'hidden';
        
        titleEl.style.pointerEvents = shouldShow && !hand ? 'auto' : 'none';
        ta.style.pointerEvents = shouldShow && !hand ? 'auto' : 'none';
        toggleEl.style.pointerEvents = shouldShow && !hand ? 'auto' : 'none';
        if (inputEl) inputEl.style.pointerEvents = shouldShow && !hand ? 'auto' : 'none';
        // suffixEl 不需要 pointerEvents，它是 none
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
    
    node.__taEls.length = items.length;
    node.__titleEls.length = items.length;
    node.__suffixEls.length = items.length;
    node.__toggleEls.length = items.length;
    node.__inputEls.length = items.length;
    
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

    // 获取视图模式
    const viewMode = node.properties?._viewMode || "grid";

    // 预留底部按钮区域高度
    const BUTTON_AREA_H = 40;
    const startY = PADDING + getWidgetsBottom(node);

    // 如果节点折叠，跳过高度调整
    if (node.flags?.collapsed) {
        // 返回基于当前尺寸的虚拟布局，防止报错
        if (viewMode === "combo") {
             return [{ x: PADDING, y: startY, w: Math.max(0, node.size[0] - PADDING * 2), h: MIN_H }];
        }
        // Grid 模式简单计算
        const cells = [];
        for (let i = 0; i < n; i++) {
             cells.push({ x: PADDING, y: startY, w: 10, h: 10 });
        }
        return cells;
    }

    if (viewMode === "combo") {
        // Combo 模式布局：一个全宽单元格用于显示当前内容
        // 高度自动适应剩余空间，但至少要有 MIN_H
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
        
        // 返回一个单元格，代表当前显示区域
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
    const availW = node.size[0] - PADDING * 2;
    const cellW = Math.floor((availW - GAP * (cols - 1)) / cols);
    
    const requiredH = rows * MIN_H + GAP * Math.max(0, rows - 1);
    const minTotalH = startY + requiredH + PADDING + BUTTON_AREA_H;
    if (node.size[1] < minTotalH) {
        if (typeof node.setSize === 'function') {
            node.setSize([node.size[0], minTotalH]);
        } else {
            node.size[1] = minTotalH;
        }
        app.graph.setDirtyCanvas(true, true);
    }

    const availH = Math.max(0, node.size[1] - startY - PADDING - BUTTON_AREA_H);
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
        
        // 即使折叠也要更新DOM元素的可见性
        if (this.flags?.collapsed) {
             relayoutAndUpdate(ctx);
             return;
        }
        
        // 绘制自定义按钮
        
        const buttonHeight = 25;
        const buttonSpacing = 10;
        // 按钮位置在节点底部，layoutCells 会预留空间
        // 使用实际的高度减去按钮高度和间距
        const buttonY = this.size[1] - buttonHeight - 5;
        
        const selectW = 60;
        const deselectW = 70;
        const invertW = 60;
        const viewModeW = 80; // 视图切换按钮宽度
        
        // 按钮水平排列，居中或靠左？参考 load_image_batch 是靠左
        const startX = 10;
        const selectAllButtonX = startX;
        const deselectAllButtonX = selectAllButtonX + selectW + buttonSpacing;
        const invertSelectionButtonX = deselectAllButtonX + deselectW + buttonSpacing;
        const viewModeButtonX = invertSelectionButtonX + invertW + buttonSpacing;
        
        // 检查鼠标悬浮状态
        const mouseInSelectAllButton = this._customMouseX !== undefined && this._customMouseY !== undefined &&
            this._customMouseX >= selectAllButtonX && this._customMouseX <= selectAllButtonX + selectW &&
            this._customMouseY >= buttonY && this._customMouseY <= buttonY + buttonHeight;
            
        const mouseInDeselectAllButton = this._customMouseX !== undefined && this._customMouseY !== undefined &&
            this._customMouseX >= deselectAllButtonX && this._customMouseX <= deselectAllButtonX + deselectW &&
            this._customMouseY >= buttonY && this._customMouseY <= buttonY + buttonHeight;
            
        const mouseInInvertSelectionButton = this._customMouseX !== undefined && this._customMouseY !== undefined &&
            this._customMouseX >= invertSelectionButtonX && this._customMouseX <= invertSelectionButtonX + invertW &&
            this._customMouseY >= buttonY && this._customMouseY <= buttonY + buttonHeight;

        const mouseInViewModeButton = this._customMouseX !== undefined && this._customMouseY !== undefined &&
            this._customMouseX >= viewModeButtonX && this._customMouseX <= viewModeButtonX + viewModeW &&
            this._customMouseY >= buttonY && this._customMouseY <= buttonY + buttonHeight;
            
        const r = 6;
        function drawButton(x, w, text, hover) {
            const y = buttonY, h = buttonHeight;
            ctx.fillStyle = hover ? 'rgba(235,235,240,0.95)' : 'rgba(235,235,240,0.85)';
            ctx.strokeStyle = hover ? 'rgba(80,80,90,0.9)' : 'rgba(120,120,130,0.8)';
            ctx.lineWidth = hover ? 2 : 1;
            ctx.beginPath();
            ctx.moveTo(x + r, y);
            ctx.lineTo(x + w - r, y);
            ctx.quadraticCurveTo(x + w, y, x + w, y + r);
            ctx.lineTo(x + w, y + h - r);
            ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
            ctx.lineTo(x + r, y + h);
            ctx.quadraticCurveTo(x, y + h, x, y + h - r);
            ctx.lineTo(x, y + r);
            ctx.quadraticCurveTo(x, y, x + r, y);
            ctx.closePath();
            ctx.fill();
            ctx.stroke();
            ctx.fillStyle = 'rgba(30,30,35,1)';
            ctx.font = 'bold 13px "Microsoft YaHei", Arial';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(text, x + w / 2, y + h / 2);
        }
        
        drawButton(selectAllButtonX, selectW, '全选', mouseInSelectAllButton);
        drawButton(deselectAllButtonX, deselectW, '全不选', mouseInDeselectAllButton);
        drawButton(invertSelectionButtonX, invertW, '反选', mouseInInvertSelectionButton);
        
        const currentViewMode = this.properties?._viewMode || "grid";
        drawButton(viewModeButtonX, viewModeW, currentViewMode === 'combo' ? '切换: 列表' : '切换: 下拉', mouseInViewModeButton);
        
        // 保存按钮区域供点击检测
        this._customSelectAllButtonRect = { x: selectAllButtonX, y: buttonY, width: selectW, height: buttonHeight };
        this._customDeselectAllButtonRect = { x: deselectAllButtonX, y: buttonY, width: deselectW, height: buttonHeight };
        this._customInvertSelectionButtonRect = { x: invertSelectionButtonX, y: buttonY, width: invertW, height: buttonHeight };
        this._customViewModeButtonRect = { x: viewModeButtonX, y: buttonY, width: viewModeW, height: buttonHeight };
        
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
        
        // 隐藏 Tooltip
        Tooltip.hide();
    };
    
    // 添加交互事件处理
    node.onMouseDown = function(e) {
        // 保存鼠标位置用于悬浮效果（虽然onMouseDown只在点击时触发，但我们可以借此更新位置）
        // 更好的方式是实现 onMouseMove，但 LiteGraph 默认可能不频繁触发重绘
        // 这里主要处理点击
        
        const nodePos = this.pos;
        // e.canvasX/Y 是画布坐标，我们需要相对于节点的坐标？
        // LiteGraph 的 onMouseDown 传入的 e 包含了 canvasX, canvasY
        // 但我们在 drawButton 中使用的是相对于节点的坐标 (0,0 是节点左上角)
        // 实际上 LiteGraph 的 onDrawForeground 的 ctx 是变换过的，原点在节点左上角
        // 所以我们需要将鼠标坐标转换为节点内坐标
        
        // 修正：LiteGraph 的事件处理通常会把局部坐标传给 onMouseDown?
        // 不，LiteGraph 的 onMouseDown 参数 e 是 MouseEvent 或者是经过处理的对象
        // 通常 e.canvasX 是世界坐标。
        // 但如果我们看 load_image_batch.js，它使用的是 e.canvasX 和 nodePos
        // 让我们参考 load_image_batch.js 的实现
        
        // 在 load_image_batch.js 中：
        // const ax = nodePos[0] + this._customSelectAllButtonRect.x;
        // if (e.canvasX >= ax ...
        
        // 所以我们需要使用 nodePos 加上按钮的相对坐标来检测
        
        if (this.flags?.collapsed) return;
        
        // 统一更新函数
        const updateAll = (newItems) => {
            setItems(this, newItems);
            // 需要更新布局和文本框样式以反映启用状态
            const layout = layoutCells(this, newItems);
            ensureTextareas(this, layout, newItems);
            app.graph.setDirtyCanvas(true, true);
        };

        if (this._customSelectAllButtonRect) {
            const ax = nodePos[0] + this._customSelectAllButtonRect.x;
            const ay = nodePos[1] + this._customSelectAllButtonRect.y;
            if (e.canvasX >= ax && e.canvasX <= ax + this._customSelectAllButtonRect.width &&
                e.canvasY >= ay && e.canvasY <= ay + this._customSelectAllButtonRect.height) {
                const items = getItems(this);
                items.forEach(item => item.enabled = true);
                updateAll(items);
                return true; // 阻止事件传播
            }
        }
        
        if (this._customDeselectAllButtonRect) {
            const ax = nodePos[0] + this._customDeselectAllButtonRect.x;
            const ay = nodePos[1] + this._customDeselectAllButtonRect.y;
            if (e.canvasX >= ax && e.canvasX <= ax + this._customDeselectAllButtonRect.width &&
                e.canvasY >= ay && e.canvasY <= ay + this._customDeselectAllButtonRect.height) {
                const items = getItems(this);
                items.forEach(item => item.enabled = false);
                updateAll(items);
                return true;
            }
        }
        
        if (this._customInvertSelectionButtonRect) {
            const ax = nodePos[0] + this._customInvertSelectionButtonRect.x;
            const ay = nodePos[1] + this._customInvertSelectionButtonRect.y;
            if (e.canvasX >= ax && e.canvasX <= ax + this._customInvertSelectionButtonRect.width &&
                e.canvasY >= ay && e.canvasY <= ay + this._customInvertSelectionButtonRect.height) {
                const items = getItems(this);
                items.forEach(item => item.enabled = !item.enabled);
                updateAll(items);
                return true;
            }
        }
        
        if (this._customViewModeButtonRect) {
            const ax = nodePos[0] + this._customViewModeButtonRect.x;
            const ay = nodePos[1] + this._customViewModeButtonRect.y;
            if (e.canvasX >= ax && e.canvasX <= ax + this._customViewModeButtonRect.width &&
                e.canvasY >= ay && e.canvasY <= ay + this._customViewModeButtonRect.height) {
                
                this.properties._viewMode = (this.properties._viewMode === 'combo') ? 'grid' : 'combo';
                
                // 触发重绘和重新布局
                const items = getItems(this);
                const layout = layoutCells(this, items);
                ensureTextareas(this, layout, items);
                app.graph.setDirtyCanvas(true, true);
                return true;
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
        if (y > this.size[1] - 40) {
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
        
        // Hide Combo elements
        if (node.__comboSelect && node.__comboSelect.style) { node.__comboSelect.style.visibility = 'hidden'; node.__comboSelect.style.pointerEvents = 'none'; }
        if (node.__comboTextarea && node.__comboTextarea.style) { node.__comboTextarea.style.visibility = 'hidden'; node.__comboTextarea.style.pointerEvents = 'none'; }
        if (node.__comboInputEl && node.__comboInputEl.style) { node.__comboInputEl.style.visibility = 'hidden'; node.__comboInputEl.style.pointerEvents = 'none'; }
        
        // Hide new Combo elements
        if (node.__comboTitleInput && node.__comboTitleInput.style) { node.__comboTitleInput.style.visibility = 'hidden'; node.__comboTitleInput.style.pointerEvents = 'none'; }
        if (node.__comboSuffixEl && node.__comboSuffixEl.style) { node.__comboSuffixEl.style.visibility = 'hidden'; node.__comboSuffixEl.style.pointerEvents = 'none'; }
        if (node.__comboMenuBtn && node.__comboMenuBtn.style) { node.__comboMenuBtn.style.visibility = 'hidden'; node.__comboMenuBtn.style.pointerEvents = 'none'; }
        if (node.__comboToggleEl && node.__comboToggleEl.style) { node.__comboToggleEl.style.visibility = 'hidden'; node.__comboToggleEl.style.pointerEvents = 'none'; }
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
        "__taEls", "__titleEls", "__suffixEls", "__toggleEls", "__inputEls",
        "__comboSelect", "__comboTextarea", "__comboInputEl",
        "__comboTitleInput", "__comboSuffixEl", "__comboMenuBtn", "__comboToggleEl",
        "__viewportSyncInstalled", "__indexListenerInstalled", "__addButtonInstalled",
        "__drawingInstalled", "__rafId", "__onWheel", "__onMouseDown", "__indexCheckInterval",
        "_customSelectAllButtonRect", "_customDeselectAllButtonRect", 
        "_customInvertSelectionButtonRect", "_customViewModeButtonRect",
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

        const origOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function() {
            if (origOnNodeCreated) origOnNodeCreated.apply(this, arguments);
            initDomRefs(this);
            ensureStringsJsonWidget(this);
            // installSelectionTools(this); // Removed in favor of canvas buttons
            installAddButton(this);
            installDrawingHandlers(this);
            installViewportSync(this);
            installIndexChangeListener(this);
            bindColumnsChange(this);
            setItems(this, getItems(this));
        };

        const origConfigure = nodeType.prototype.configure;
        nodeType.prototype.configure = function(info) {
            if (origConfigure) origConfigure.apply(this, arguments);
            initDomRefs(this);
            ensureStringsJsonWidget(this);
            // installSelectionTools(this); // Removed in favor of canvas buttons
            installAddButton(this);
            installDrawingHandlers(this);
            installViewportSync(this);
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
