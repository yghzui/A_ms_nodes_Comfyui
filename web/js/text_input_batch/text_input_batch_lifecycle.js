export function createTextInputBatchLifecycleApi({
    app,
    Tooltip,
    ensureStringsJsonWidget,
    getItems,
    setItems,
    bindColumnsChange,
    bindMinHeightChange,
    installAddButton,
    getLayoutCells,
    getEnsureTextareas,
    handleImport,
    handleExport,
    handleBatchDelete
}) {
    const layoutCells = (...args) => getLayoutCells()(...args);
    let ensureTextareas = (...args) => getEnsureTextareas()(...args);

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

    return {
        ensureTextareas
    };
}
