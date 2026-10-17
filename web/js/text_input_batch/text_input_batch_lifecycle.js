// 左侧图标按钮栏的几何参数（预留宽度需与 text_input_batch_render_core.js 保持一致）
const BUTTON_SIZE = 20;
const BUTTON_SPACING = 4;
const RAIL_PADDING_LEFT = 6;
export const BUTTON_RAIL_WIDTH = RAIL_PADDING_LEFT + BUTTON_SIZE;
// 图标栏左外边距：同时作为图标栏与内容区之间的间距，保证左右对称
export const BUTTON_RAIL_MARGIN = RAIL_PADDING_LEFT;
// 8 个图标按钮竖向排列的总高度（需与 installDrawingHandlers 中按钮数量保持一致）
export const BUTTON_RAIL_HEIGHT = 8 * BUTTON_SIZE + 7 * BUTTON_SPACING;

// SVG 图标缓存
const iconImageCache = new Map();
function getIconImage(fileName) {
    let img = iconImageCache.get(fileName);
    if (!img) {
        img = new Image();
        img.src = new URL(`./icons/${fileName}`, import.meta.url).href;
        iconImageCache.set(fileName, img);
    }
    return img;
}

export function createTextInputBatchLifecycleApi({
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
    getLayoutCells,
    getEnsureTextareas,
    getWidgetsBottom,
    handleImport,
    handleExport,
    handleBatchDelete
}) {
    const layoutCells = (...args) => getLayoutCells()(...args);
    const baseEnsureTextareas = getEnsureTextareas();
    let ensureTextareas = (...args) => baseEnsureTextareas(...args);

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
        
        // 绘制自定义按钮（左侧竖向图标栏）
        const currentViewMode = this.properties?._viewMode || "grid";
        const railTop = (typeof getWidgetsBottom === 'function' ? getWidgetsBottom(this) : 0) + 8;

        // 按钮定义
        // 顺序：视图切换 -> 全选 -> 全不选 -> 反选 -> 导入 -> 导出 -> 批量删除 -> 隐藏/显示提示词
        const buttons = [
            {
                icon: 'view_toggle.svg',
                tooltip: currentViewMode === 'combo' ? '切换视图：网格列表' : '切换视图：下拉列表',
                callback: () => {
                    this.properties._viewMode = (this.properties._viewMode === 'combo') ? 'grid' : 'combo';
                    const items = getItems(this);
                    const layout = layoutCells(this, items);
                    ensureTextareas(this, layout, items);
                    app.graph.setDirtyCanvas(true, true);
                }
            },
            {
                icon: 'select_all.svg',
                tooltip: '全选',
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
                icon: 'deselect_all.svg',
                tooltip: '全不选',
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
                icon: 'invert.svg',
                tooltip: '反选',
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
                icon: 'import.svg',
                tooltip: '导入',
                callback: () => handleImport(this)
            },
            {
                icon: 'export.svg',
                tooltip: '导出',
                callback: () => handleExport(this)
            },
            {
                icon: 'delete_selected.svg',
                tooltip: '删除选中',
                callback: () => handleBatchDelete(this)
            },
            {
                icon: this.__hidePrompts ? 'show_prompt.svg' : 'hide_prompt.svg',
                tooltip: this.__hidePrompts ? '显示提示词' : '隐藏提示词',
                active: !!this.__hidePrompts,
                callback: () => {
                    this.__hidePrompts = !this.__hidePrompts;
                    const items = getItems(this);
                    const layout = layoutCells(this, items);
                    ensureTextareas(this, layout, items);
                    app.graph.setDirtyCanvas(true, true);
                }
            }
        ];

        // 清空点击区域缓存
        this._customButtons = [];

        const r = 5;
        function drawIconButton(ctx, x, y, icon, hover, active) {
            ctx.fillStyle = active ? 'rgba(120,170,255,0.92)'
                : (hover ? 'rgba(235,235,240,0.95)' : 'rgba(235,235,240,0.85)');
            ctx.strokeStyle = hover ? 'rgba(80,80,90,0.9)' : 'rgba(120,120,130,0.8)';
            ctx.lineWidth = hover ? 2 : 1;
            ctx.beginPath();

            if (ctx.roundRect) {
                ctx.roundRect(x, y, BUTTON_SIZE, BUTTON_SIZE, r);
            } else {
                ctx.moveTo(x + r, y);
                ctx.lineTo(x + BUTTON_SIZE - r, y);
                ctx.quadraticCurveTo(x + BUTTON_SIZE, y, x + BUTTON_SIZE, y + r);
                ctx.lineTo(x + BUTTON_SIZE, y + BUTTON_SIZE - r);
                ctx.quadraticCurveTo(x + BUTTON_SIZE, y + BUTTON_SIZE, x + BUTTON_SIZE - r, y + BUTTON_SIZE);
                ctx.lineTo(x + r, y + BUTTON_SIZE);
                ctx.quadraticCurveTo(x, y + BUTTON_SIZE, x, y + BUTTON_SIZE - r);
                ctx.lineTo(x, y + r);
                ctx.quadraticCurveTo(x, y, x + r, y);
            }

            ctx.closePath();
            ctx.fill();
            ctx.stroke();

            const img = getIconImage(icon);
            if (img && img.complete && img.naturalWidth > 0) {
                ctx.drawImage(img, x + 3, y + 3, BUTTON_SIZE - 6, BUTTON_SIZE - 6);
            }
        }

        // 左侧竖向排列
        buttons.forEach((btn, index) => {
            const x = RAIL_PADDING_LEFT;
            const y = railTop + index * (BUTTON_SIZE + BUTTON_SPACING);
            const hover = this._customMouseX >= x && this._customMouseX <= x + BUTTON_SIZE &&
                          this._customMouseY >= y && this._customMouseY <= y + BUTTON_SIZE;

            drawIconButton(ctx, x, y, btn.icon, hover, !!btn.active);

            // 记录点击区域
            this._customButtons.push({
                rect: { x, y, w: BUTTON_SIZE, h: BUTTON_SIZE },
                callback: btn.callback,
                tooltip: btn.tooltip
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
        
        // 命中检测：悬浮按钮状态变化时重绘并更新 Tooltip
        let hovered = null;
        if (this._customButtons) {
            for (const btn of this._customButtons) {
                if (x >= btn.rect.x && x <= btn.rect.x + btn.rect.w &&
                    y >= btn.rect.y && y <= btn.rect.y + btn.rect.h) {
                    hovered = btn;
                    break;
                }
            }
        }
        
        const hoverKey = hovered ? (hovered.tooltip || '') : '';
        if (hoverKey !== this.__hoveredButtonTooltip) {
            this.__hoveredButtonTooltip = hoverKey;
            if (hovered && hovered.tooltip) {
                Tooltip.scheduleShow(e.clientX, e.clientY, hovered.tooltip, '');
            } else {
                Tooltip.hide();
            }
            app.graph.setDirtyCanvas(true, false);
        }
    };
    
    // 鼠标离开节点时清除状态
    node.onMouseLeave = function(e) {
        this._customMouseX = undefined;
        this._customMouseY = undefined;
        this.__hoveredButtonTooltip = '';
        Tooltip.hide();
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
        "_customMouseX", "_customMouseY",
        "__hoveredButtonTooltip"
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
                "__hoveredButtonTooltip",
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

    return {
        ensureTextareas
    };
}
