import { $el } from "../utils/shared_utils.js";

export class AssetManagerQuickMenu {
    constructor() {
        this.lastFocusedInput = null;
        this.activeTab = 'prompts'; // 'prompts' or 'models'
        
        // 从 localStorage 恢复记忆的面板位置
        this.savedPanelPos = null;
        try {
            const saved = localStorage.getItem("am_quick_menu_pos");
            if (saved) {
                this.savedPanelPos = JSON.parse(saved);
            }
        } catch(e) {}
        
        this.searchKeyword = ""; // 搜索关键字
        this.createDOM();
        this.initEvents();
    }

    createDOM() {
        // 由于小悬浮球（triggerBtn）已经被移至 asset_manager_fab.js 作为一个整体，
        // 这里不再创建独立的 triggerBtn。
        // 为了向后兼容某些调用，我们可以声明一个空的引用或将其指向主 FAB 的 miniFab
        this.triggerBtn = null;

        // 菜单面板
        this.menuPanel = $el("div", {
            className: "am-quick-menu",
            style: {
                position: "absolute",
                display: "none",
                zIndex: 9999,
                background: "var(--am-bg, #222)",
                border: "1px solid var(--am-border, #444)",
                borderRadius: "6px",
                padding: "8px",
                width: "280px",
                boxShadow: "0 4px 10px rgba(0,0,0,0.5)",
                flexDirection: "column",
                gap: "5px"
            }
        });

        // 顶部拖拽手柄区
        this.dragHandle = $el("div", {
            style: {
                height: "12px",
                background: "var(--am-accent, #444)",
                borderRadius: "3px",
                cursor: "move",
                marginBottom: "4px",
                opacity: "0.5",
                display: "flex",
                justifyContent: "center",
                alignItems: "center"
            },
            onmouseenter: (e) => { e.target.style.opacity = "1"; },
            onmouseleave: (e) => { e.target.style.opacity = "0.5"; }
        });
        
        // 添加三个小点作为拖拽指示
        this.dragHandle.appendChild($el("div", {
            style: { width: "20px", height: "3px", borderTop: "1px solid #fff", borderBottom: "1px solid #fff", opacity: "0.5" }
        }));
        
        // 右上角关闭按钮
        this.closeBtn = $el("div", {
            textContent: "✖",
            title: "关闭面板",
            style: {
                position: "absolute",
                right: "8px",
                top: "8px",
                cursor: "pointer",
                color: "#aaa",
                fontSize: "12px",
                zIndex: 10000
            },
            onmousedown: (e) => e.preventDefault(),
            onmouseenter: (e) => e.target.style.color = "#fff",
            onmouseleave: (e) => e.target.style.color = "#aaa",
            onclick: (e) => {
                e.stopPropagation();
                this.closePanel();
            }
        });

        // 顶部: 分组下拉框
        this.groupSelect = $el("select", {
            style: { 
                width: "100%", padding: "4px", background: "#111", 
                color: "#fff", border: "1px solid #555", borderRadius: "4px" 
            },
            onchange: () => this.renderList()
        });

        // 按钮组 (切换提示词/模型)
        this.tabs = $el("div", { style: { display: "flex", gap: "5px", marginTop: "5px" } }, [
            $el("button", {
                id: "am-qm-tab-prompts",
                textContent: "📝 提示词",
                style: { flex: 1, padding: "4px", background: "var(--am-accent, #555)", color: "#fff", border: "none", borderRadius: "4px", cursor: "pointer" },
                onmousedown: (e) => e.preventDefault(),
                onclick: () => { this.activeTab = 'prompts'; this.updateTabStyles(); this.renderList(); }
            }),
            $el("button", {
                id: "am-qm-tab-models",
                textContent: "🧩 模型组",
                style: { flex: 1, padding: "4px", background: "transparent", color: "#fff", border: "1px solid #555", borderRadius: "4px", cursor: "pointer" },
                onmousedown: (e) => e.preventDefault(),
                onclick: () => { this.activeTab = 'models'; this.updateTabStyles(); this.renderList(); }
            })
        ]);

        // 搜索框容器
        this.searchContainer = $el("div", {
            style: { position: "relative", width: "100%", marginTop: "5px" }
        });

        // 搜索框
        this.searchInput = $el("input", {
            type: "text",
            placeholder: "🔍 搜索...",
            style: {
                width: "100%", padding: "4px 20px 4px 6px", background: "#111", 
                color: "#fff", border: "1px solid #555", borderRadius: "4px", 
                boxSizing: "border-box", outline: "none"
            },
            oninput: (e) => {
                this.searchKeyword = e.target.value.toLowerCase();
                this.searchClearBtn.style.display = this.searchKeyword ? "block" : "none";
                this.renderList();
            }
        });

        // 搜索框快速清除按钮
        this.searchClearBtn = $el("span", {
            textContent: "✖",
            title: "清除搜索",
            style: {
                position: "absolute",
                right: "6px",
                top: "50%",
                transform: "translateY(-50%)",
                cursor: "pointer",
                color: "#aaa",
                fontSize: "12px",
                display: "none" // 默认隐藏
            },
            onmousedown: (e) => e.preventDefault(),
            onmouseenter: (e) => e.target.style.color = "#fff",
            onmouseleave: (e) => e.target.style.color = "#aaa",
            onclick: (e) => {
                e.stopPropagation();
                this.searchInput.value = "";
                this.searchKeyword = "";
                this.searchClearBtn.style.display = "none";
                this.renderList();
                this.searchInput.focus();
            }
        });

        this.searchContainer.append(this.searchInput, this.searchClearBtn);

        // 列表容器
        this.listContainer = $el("div", {
            style: { 
                maxHeight: "350px", overflowY: "auto", display: "flex", 
                flexDirection: "column", gap: "4px", marginTop: "5px" 
            }
        });

        this.menuPanel.append(this.dragHandle, this.closeBtn, this.groupSelect, this.tabs, this.searchContainer, this.listContainer);
        document.body.appendChild(this.menuPanel);

        // 初始化拖拽逻辑
        this.initDrag();
    }

    initDrag() {
        // 1. 面板拖拽逻辑
        let isPanelDragging = false;
        let pStartX, pStartY, pInitialLeft, pInitialTop;

        this.dragHandle.addEventListener("mousedown", (e) => {
            isPanelDragging = true;
            pStartX = e.clientX;
            pStartY = e.clientY;
            
            const rect = this.menuPanel.getBoundingClientRect();
            pInitialLeft = rect.left;
            pInitialTop = rect.top;
            
            e.preventDefault();
            e.stopPropagation();
        });

        // 2. 小悬浮球独立拖拽逻辑已经移除，现在由大悬浮球统一拖动

        // 全局鼠标移动事件
        document.addEventListener("mousemove", (e) => {
            if (isPanelDragging) {
                const dx = e.clientX - pStartX;
                const dy = e.clientY - pStartY;
                
                let newLeft = pInitialLeft + dx;
                let newTop = pInitialTop + dy;
                
                newLeft = Math.max(0, Math.min(newLeft, window.innerWidth - this.menuPanel.offsetWidth));
                newTop = Math.max(0, Math.min(newTop, window.innerHeight - this.menuPanel.offsetHeight));
                
                this.menuPanel.style.left = `${newLeft}px`;
                this.menuPanel.style.top = `${newTop}px`;
                
                // 记忆位置并保存到 localStorage
                this.savedPanelPos = { left: newLeft, top: newTop };
                try {
                    localStorage.setItem("am_quick_menu_pos", JSON.stringify(this.savedPanelPos));
                } catch(e) {}
            }
        });

        // 全局鼠标抬起事件
        document.addEventListener("mouseup", () => {
            isPanelDragging = false;
        });
    }

    updateTabStyles() {
        const btnP = document.getElementById("am-qm-tab-prompts");
        const btnM = document.getElementById("am-qm-tab-models");
        if (this.activeTab === 'prompts') {
            btnP.style.background = "var(--am-accent, #555)";
            btnP.style.border = "none";
            btnM.style.background = "transparent";
            btnM.style.border = "1px solid #555";
        } else {
            btnM.style.background = "var(--am-accent, #555)";
            btnM.style.border = "none";
            btnP.style.background = "transparent";
            btnP.style.border = "1px solid #555";
        }
    }

    initEvents() {
        // 监听焦点进入事件，仅记录最后一次焦点的文本输入框
        document.addEventListener("focusin", (e) => {
            const target = e.target;
            if (target && (target.tagName === 'TEXTAREA' || (target.tagName === 'INPUT' && target.type === 'text'))) {
                // 如果焦点在我们的面板内，则不处理
                if (this.menuPanel.contains(target) || (this.triggerBtn && this.triggerBtn.contains(target))) return;
                // 也要判断是不是在 FAB 的容器里
                const fab = document.getElementById("asset-manager-fab");
                if (fab && fab.contains(target)) return;
                
                // 如果焦点在全局管理器的搜索框里，也不处理
                if (target.id === 'am-search-input') return;

                this.lastFocusedInput = target;
                
                // 给输入框绑定失焦和输入事件，以便在插入时准确知道光标位置和内容
                if (!target._am_bound) {
                    target._am_bound = true;
                    
                    // 我们还监听 mousedown 和 keyup 以实时更新光标
                    const updateCursor = () => {
                        if (target === this.lastFocusedInput) {
                            this.lastInputState = {
                                value: target.value,
                                selectionStart: target.selectionStart,
                                selectionEnd: target.selectionEnd,
                                element: target,
                                timestamp: Date.now() // 记录时间戳
                            };
                        }
                    };
                    target.addEventListener('mouseup', updateCursor);
                    target.addEventListener('keyup', updateCursor);
                    
                    // 标记失焦状态，如果在一定时间内没有其他点击操作，我们认为它彻底失焦了
        target.addEventListener('blur', (blurEvent) => {
            // 当文本框失去焦点时，我们保存它最后的状态（内容和光标位置）
            updateCursor();
            
            // 为了防止用户点击空白处（导致输入框销毁），但在短时间内又点击了插入按钮
            // 我们延迟 100ms 检查，如果 100ms 后焦点仍然没有回到输入框，且没有点击面板，
            // 则认为它是真正的失焦，此时清空记录。
            setTimeout(() => {
                if (document.activeElement !== target && (!this.lastInputState || Date.now() - this.lastInputState.timestamp > 300)) {
                    // 如果超过 300 毫秒没有针对该输入框的新操作（因为键盘、鼠标都会刷新 timestamp）
                    // 那说明它是真的彻底失焦（比如点去了其他节点），就清理掉，杜绝幽灵插入
                    if (this.lastFocusedInput === target) {
                        this.lastFocusedInput = null;
                        this.lastInputState = null;
                    }
                }
            }, 300);
        });
                }
            }
        });

        // 监听全局点击事件，用于判断用户是否点击了空白处或其他不相关的元素
        document.addEventListener("mousedown", (e) => {
            // 如果点击的是我们的面板，不要清空记录（用户准备点击插入）
            if (this.menuPanel.contains(e.target)) return;
            
            // 如果点击的是 FAB (触发按钮)，也不要清空
            const fab = document.getElementById("asset-manager-fab");
            if (fab && fab.contains(e.target)) return;

            // 如果点击的是提示词工具提示面板(Tooltip)，也不要清空
            const tooltip = document.getElementById("am-tooltip");
            if (tooltip && tooltip.contains(e.target)) return;

            // 如果点击的是输入框本身，也不清空（用户在编辑）
            if (e.target && (e.target.tagName === 'TEXTAREA' || (e.target.tagName === 'INPUT' && e.target.type === 'text'))) return;
            
            // 只要不是点击面板、FAB、或者输入框，我们就把之前的焦点记录清空，避免“幽灵插入”
            this.lastFocusedInput = null;
            this.lastInputState = null;
        }, true); // 使用捕获阶段，确保比面板内的事件更早触发或者平行判断

        // 移除点击外部隐藏整个面板和按钮的逻辑，改为仅点击外部隐藏菜单面板（按钮保留）
        document.addEventListener("mousedown", (e) => {
            if (!this.menuPanel.contains(e.target)) {
                // 如果点击的是主 FAB (或其内的伴生按钮)，由 FAB 自身处理
                const fab = document.getElementById("asset-manager-fab");
                if (fab && fab.contains(e.target)) return;

                // 如果点击的是输入框（TEXTAREA 或 type="text" 的 INPUT），则不要关闭面板，
                // 因为用户可能只是在切换焦点准备继续插入。
                if (e.target && (e.target.tagName === 'TEXTAREA' || (e.target.tagName === 'INPUT' && e.target.type === 'text'))) {
                    return;
                }

                // 点击了外部，我们只隐藏菜单面板
                if (this.menuPanel.style.display === "flex") {
                    this.menuPanel.style.display = "none";
                }
            }
        });
    }

    showTriggerNear(element) {
        if (!element) return;
        const rect = element.getBoundingClientRect();
        
        let left, top;

        if (element.id === 'asset-manager-fab') {
            // 如果小球当前是隐藏状态，则伴随主悬浮球显示在左上方偏一点
            // 如果小球已经是显示状态，单击大球则将其隐藏（即手动 toggle 显隐）
            if (this.triggerBtn.style.display === "block") {
                this.hideAll();
                return;
            }
            left = rect.left - 30;
            top = rect.top - 10;
        } else {
            // 此处保留给其他可能的特定定位需求，不过输入框 focus 已经不再调用这里了
            left = rect.right - 20;
            top = rect.bottom - 20;
        }

        // 如果超出屏幕，调整位置
        if (left < 10) left = 10;
        if (left > window.innerWidth - 30) left = window.innerWidth - 30;
        if (top < 10) top = 10;
        if (top > window.innerHeight - 30) top = window.innerHeight - 30;

        this.triggerBtn.style.left = `${left}px`;
        this.triggerBtn.style.top = `${top}px`;
        this.triggerBtn.style.display = "block";
        this.menuPanel.style.display = "none"; // 重置菜单
    }

    toggleMenu() {
        if (this.menuPanel.style.display === "flex") {
            this.menuPanel.style.display = "none";
        } else {
            this.updateGroups();
            this.renderList();
            
            // 如果有记忆的位置，优先使用记忆的位置
            if (this.savedPanelPos) {
                this.menuPanel.style.left = `${this.savedPanelPos.left}px`;
                this.menuPanel.style.top = `${this.savedPanelPos.top}px`;
            } else {
                // 定位面板 (紧贴主 FAB)
                const fab = document.getElementById("asset-manager-fab");
                if (fab) {
                    const rect = fab.getBoundingClientRect();
                    let left = rect.left - 290; // 显示在 FAB 左侧
                    let top = rect.bottom - 300; // 向上对齐
                    
                    // 避免超出屏幕
                    if (left < 10) left = 10;
                    if (top < 10) top = 10;

                    this.menuPanel.style.left = `${left}px`;
                    this.menuPanel.style.top = `${top}px`;
                } else {
                    this.menuPanel.style.left = `100px`;
                    this.menuPanel.style.top = `100px`;
                }
            }

            this.menuPanel.style.display = "flex";
            // 聚焦搜索框（仅当当前没有聚焦其他输入框时，避免抢夺节点文本框焦点导致其被销毁）
            setTimeout(() => {
                if (document.activeElement && (document.activeElement.tagName === 'TEXTAREA' || (document.activeElement.tagName === 'INPUT' && document.activeElement.type === 'text'))) {
                    // 保持原焦点，不聚焦搜索框
                } else {
                    this.searchInput.focus();
                }
            }, 50);
        }
    }

    closePanel() {
        this.menuPanel.style.display = "none";
    }

    hideAll() {
        this.triggerBtn.style.display = "none";
        this.menuPanel.style.display = "none";
    }

    updateGroups() {
        if (!window.AssetManager) return;
        this.groupSelect.innerHTML = "";
        
        // 合并提示词和模型的分组名去重
        const pGroups = window.AssetManager.promptsData?.groups || [];
        const mGroups = window.AssetManager.modelsData?.groups || [];
        const allGroupNames = new Set([
            ...pGroups.map(g => g.name),
            ...mGroups.map(g => g.name)
        ]);

        if (allGroupNames.size === 0) {
            this.groupSelect.appendChild($el("option", { value: "", textContent: "无分组数据" }));
            return;
        }

        allGroupNames.forEach(name => {
            this.groupSelect.appendChild($el("option", { value: name, textContent: name }));
        });
    }

    async renderList() {
        if (!window.AssetManager) return;
        this.listContainer.innerHTML = "";
        
        const groupName = this.groupSelect.value;
        if (!groupName) return;

        const isPrompts = this.activeTab === 'prompts';
        const data = isPrompts ? window.AssetManager.promptsData : window.AssetManager.modelsData;
        const group = data?.groups?.find(g => g.name === groupName);

        if (!group || !group.items || group.items.length === 0) {
            this.listContainer.appendChild($el("div", { 
                textContent: "此分组下无数据", 
                style: { color: "#888", fontSize: "12px", textAlign: "center", padding: "10px" } 
            }));
            return;
        }

        const keyword = this.searchKeyword.toLowerCase().trim();
        let filteredItems = [];
        
        if (keyword) {
            try {
                // 使用和 ui 模块一样的后端拼音搜索，以确保搜索结果完全一致
                const titles = group.items.map(i => isPrompts ? i.title : (i.keyword || ""));
                const res = await window.AssetManagerUI?.prototype?.constructor?.prototype ? fetch("/a_my_nodes/assets/search_pinyin", {
                    method: "POST",
                    body: JSON.stringify({ texts: titles, keyword: keyword }),
                    headers: { "Content-Type": "application/json" }
                }) : null;
                
                // 由于跨域或模块未加载可能导致 res 为 null，我们加上原生的 fetch 直接请求
                let actualRes = res;
                if (!actualRes) {
                    actualRes = await fetch("/a_my_nodes/assets/search_pinyin", {
                        method: "POST",
                        body: JSON.stringify({ texts: titles, keyword: keyword }),
                        headers: { "Content-Type": "application/json" }
                    });
                }
                
                if (actualRes) {
                    const data = await actualRes.json();
                    if (data && data.matches) {
                        filteredItems = group.items.filter((_, idx) => data.matches[idx]);
                    } else {
                        filteredItems = group.items.filter(i => {
                            const t = isPrompts ? i.title : (i.keyword || "");
                            const c = isPrompts ? i.content : "";
                            return t.toLowerCase().includes(keyword) || c.toLowerCase().includes(keyword);
                        });
                    }
                } else {
                    // Fallback
                    filteredItems = group.items.filter(i => {
                        const t = isPrompts ? i.title : (i.keyword || "");
                        const c = isPrompts ? i.content : "";
                        return t.toLowerCase().includes(keyword) || c.toLowerCase().includes(keyword);
                    });
                }
            } catch (e) {
                // Fallback
                filteredItems = group.items.filter(i => {
                    const t = isPrompts ? i.title : (i.keyword || "");
                    const c = isPrompts ? i.content : "";
                    return t.toLowerCase().includes(keyword) || c.toLowerCase().includes(keyword);
                });
            }
        } else {
            filteredItems = group.items;
        }
        
        this.listContainer.innerHTML = ""; // 再次清空，防止异步导致重复
        
        if (filteredItems.length === 0) {
            this.listContainer.appendChild($el("div", { 
                textContent: "无匹配结果", 
                style: { color: "#888", fontSize: "12px", textAlign: "center", padding: "10px" } 
            }));
            return;
        }

        filteredItems.forEach((item, index) => {
            const row = $el("div", {
                style: { 
                    display: "flex", justifyContent: "space-between", alignItems: "center",
                    background: "#1a1a1a", padding: "4px 8px", borderRadius: "4px", border: "1px solid #333"
                },
                onmousedown: (e) => e.preventDefault() // 防止点击行空白处抢夺焦点
            });

            // 标题 (鼠标悬浮预览图)
            const titleEl = $el("span", {
                textContent: isPrompts ? item.title : (item.keyword || "未命名"),
                style: { flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", fontSize: "12px", cursor: "pointer" },
                onmousedown: (e) => e.preventDefault(), // 防止抢夺焦点
                onmouseenter: (e) => window.AssetManager.showTooltip(e, item, this.activeTab),
                onmouseleave: () => window.AssetManager.hideTooltip(),
                onclick: () => {
                    const text = isPrompts ? item.title : item.keyword; // 点击标题写入标题
                    this.insertText(text);
                }
            });

            // 操作按钮
            const actionBtn = $el("button", {
                textContent: isPrompts ? "➕应用" : "✏️编辑",
                style: { 
                    marginLeft: "5px", padding: "2px 6px", fontSize: "10px", 
                    background: "var(--am-accent, #555)", color: "#fff", border: "none", borderRadius: "3px", cursor: "pointer" 
                },
                onmousedown: (e) => e.preventDefault(), // 防止抢夺焦点
                onmouseenter: (e) => {
                    if (isPrompts) {
                        window.AssetManager.showTooltip(e, item, this.activeTab); // 悬浮按钮也显示完整提示词
                    } else {
                        window.AssetManager.showTooltip(e, item, this.activeTab);
                    }
                },
                onmouseleave: () => window.AssetManager.hideTooltip(),
                onclick: () => {
                    if (isPrompts) {
                        this.insertText(item.content);
                    } else {
                        // 跳转到模型组编辑
                        this.hideAll();
                        window.AssetManager.showModal();
                        window.AssetManager.switchTab('models', document.querySelectorAll('.am-tab')[1]); // 假定第二个 tab 是 models
                        // 定位到对应的 group
                        const gIdx = window.AssetManager.modelsData.groups.findIndex(g => g.name === groupName);
                        if (gIdx >= 0) {
                            window.AssetManager.currentGroupIndex = gIdx;
                            window.AssetManager.renderGroups();
                            window.AssetManager.renderItems();
                            // 可以尝试高亮一下被点击的项
                        }
                    }
                }
            });

            row.append(titleEl, actionBtn);
            this.listContainer.appendChild(row);
        });
    }

    async insertText(text) {
        if (!text) return;
        
        // 自动补充英文分号和空格
        let toInsert = text;
        if (!toInsert.endsWith(';')) toInsert += '; ';
        else if (!toInsert.endsWith('; ')) toInsert += ' ';

        let success = false;
        
        // 1. 最高优先级：尝试直接从页面上找当前正在编辑的、属于节点的文本框
        let targetElement = null;
        if (document.activeElement && (document.activeElement.tagName === 'TEXTAREA' || (document.activeElement.tagName === 'INPUT' && document.activeElement.type === 'text'))) {
            // 排除掉我们自己面板的搜索框
            if (document.activeElement.id !== 'am-search-input') {
                targetElement = document.activeElement;
            }
        }
        
        // 1.5 强检查：如果目标元素在 DOM 树里，但它不属于当前 LiteGraph 画布（比如隐藏的游离节点）
        // ComfyUI 中正在编辑的节点文本框一般是挂载在 document.body 下的临时元素，而不是隐藏的元素
        if (targetElement && (!document.body.contains(targetElement) || targetElement.style.display === 'none')) {
            targetElement = null;
        }

        // 如果没有找到正在编辑的（可能因为点击了面板导致它暂时失焦，但因为 preventDefault 它没有被销毁）
        if (!targetElement && this.lastFocusedInput && document.body.contains(this.lastFocusedInput) && this.lastFocusedInput.style.display !== 'none') {
            targetElement = this.lastFocusedInput;
        }

        let selectionStart, selectionEnd, val;

        // 如果找到了存活的元素
        if (targetElement) {
            try {
                if (typeof targetElement.selectionStart === 'number') {
                    selectionStart = targetElement.selectionStart;
                    selectionEnd = targetElement.selectionEnd;
                    val = targetElement.value || "";
                }
            } catch (e) {
                console.warn("[AssetManager] Failed to read selection from active element", e);
            }
        } else if (this.lastInputState && this.lastInputState.element) {
            // 2. 只有在真的没有存活的输入框时，才考虑使用历史记录（且要经过严格校验）
            const timeSinceLastFocus = Date.now() - (this.lastInputState.timestamp || 0);
            
            // 非常严格的条件：
            // A. 必须在非常短的时间内（比如 2 秒内），因为可能是点击按钮瞬间输入框被 LiteGraph 销毁
            // B. 如果时间超过 2 秒，直接放弃
            if (timeSinceLastFocus < 1000 * 2) { 
                targetElement = this.lastInputState.element;
                selectionStart = this.lastInputState.selectionStart;
                selectionEnd = this.lastInputState.selectionEnd;
                val = this.lastInputState.value || "";
            } else {
                // 超时，不信任该记录，清空
                this.lastInputState = null;
                targetElement = null;
            }
        }

        // 3. 最后防线，如果此时找到的 targetElement 不是当前 document.activeElement
        // 且它也不在屏幕上，或者我们仅仅通过 lastInputState 找到了一个离线的元素
        // 尝试判断一下它是不是真的离线且有效
        if (targetElement && !document.body.contains(targetElement)) {
             // 对于离线的，只有距离它上次活跃在 1 秒以内，我们才敢写，否则视为“以前被销毁的幽灵”
             const t = (this.lastInputState && this.lastInputState.timestamp) ? this.lastInputState.timestamp : 0;
             if (Date.now() - t > 1000) {
                 targetElement = null;
                 this.lastInputState = null;
             }
        }

        if (targetElement && typeof selectionStart === 'number') {
            try {
                // 如果发现该元素被销毁（不在DOM里），LiteGraph 可能已经把它原有的 value 同步回了节点
                // 为了防止它将文本塞入未知的节点末尾（幽灵插入），我们必须定位到它的真实宿主节点。
                // 但由于通过 JS 修改游离 input 比较危险，如果有 lastInputState，最好是通过它找节点
                
                const before = val.substring(0, selectionStart);
                const after = val.substring(selectionEnd);
                const newValue = before + toInsert + after;
                
                // 强制写回该对象并触发事件
                targetElement.value = newValue;
                
                // 更新我们自己保存的光标位置以便连续插入
                const newPos = selectionStart + toInsert.length;
                if (this.lastInputState) {
                    this.lastInputState.value = newValue;
                    this.lastInputState.selectionStart = newPos;
                    this.lastInputState.selectionEnd = newPos;
                }

                // 强制焦点回到它身上，如果它还在 DOM 树里
                if (document.body.contains(targetElement)) {
                    targetElement.focus();
                    targetElement.setSelectionRange(newPos, newPos);
                }

                targetElement.dispatchEvent(new Event('input', { bubbles: true }));
                targetElement.dispatchEvent(new Event('change', { bubbles: true }));

                // ComfyUI 特有的节点 widget 更新通知
                if (!document.body.contains(targetElement)) {
                    // 节点如果不在 DOM 里，修改其值往往会无效或者插错地方
                    console.warn("[AssetManager] Target element is no longer in DOM.");
                }
                
                success = true;
            } catch (e) {
                console.warn("[AssetManager] Failed to insert text directly", e);
            }
        }
        
        if (!success) {
            try {
                await navigator.clipboard.writeText(toInsert);
                this.showToast("写入剪贴板成功: " + toInsert);
            } catch (err) {
                this.showToast("写入剪贴板失败，请手动复制");
            }
        } else {
            this.showToast("插入成功: " + toInsert);
        }
        
        // 插入完毕后不再隐藏面板，由用户完全手动控制开启和关闭
    }

    showToast(msg) {
        const toast = $el("div", {
            textContent: msg,
            style: {
                position: "fixed", top: "20px", left: "50%", transform: "translateX(-50%)",
                background: "rgba(0, 0, 0, 0.8)", color: "#fff", padding: "10px 20px",
                borderRadius: "4px", zIndex: 10000, fontSize: "14px",
                transition: "opacity 0.3s"
            }
        });
        document.body.appendChild(toast);
        setTimeout(() => {
            toast.style.opacity = "0";
            setTimeout(() => toast.remove(), 300);
        }, 3000);
    }
}
