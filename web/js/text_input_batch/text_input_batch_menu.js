export function createTextInputBatchMenuApi({
    app,
    rgthree,
    showTopNotification,
    getItems,
    setItems,
    getCurrentIndex,
    setIndexSelectorValue,
    updateTextareaStyles,
    moveItem,
    fetchPinyinMatches,
    openAddToAssetManagerGroupPicker
}) {
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
        min-height: 0;
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
        min-height: 0;
        max-height: 300px; /* 默认最大高度，会被动态调整覆盖 */
        scrollbar-width: thin;
        scrollbar-color: #666 #222;
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

    return {
        showItemContextMenu,
        handleTextareaCommentShortcut,
        handleTextareaWheel,
        showCustomDropdown,
        updateCustomDropdownPosition
    };
}
