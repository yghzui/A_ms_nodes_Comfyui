export function createTextInputBatchRenderCoreApi({
    app,
    Tooltip,
    getItems,
    setItems,
    getBaseTitle,
    getCurrentIndex,
    updateTextareaStyles,
    isHandMode,
    getWidgetsBottom,
    handleTextareaCommentShortcut,
    handleTextareaWheel,
    showItemContextMenu,
    showCustomDropdown,
    updateCustomDropdownPosition,
    getEnsureTextareas
}) {
    const ensureTextareasRef = (...args) => getEnsureTextareas()(...args);

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
                ensureTextareasRef(node, newCells, newItems);
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
                    ensureTextareasRef(node, newCells, newItems);
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
                ensureTextareasRef(node, currentCells, currentItems);
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

    return {
        ensureTextareas,
        layoutCells
    };
}
