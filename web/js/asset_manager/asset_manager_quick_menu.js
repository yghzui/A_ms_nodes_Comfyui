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
                onclick: () => { this.activeTab = 'prompts'; this.updateTabStyles(); this.renderList(); }
            }),
            $el("button", {
                id: "am-qm-tab-models",
                textContent: "🧩 模型组",
                style: { flex: 1, padding: "4px", background: "transparent", color: "#fff", border: "1px solid #555", borderRadius: "4px", cursor: "pointer" },
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
                maxHeight: "200px", overflowY: "auto", display: "flex", 
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
                if (this.menuPanel.contains(target) || this.triggerBtn.contains(target)) return;
                // 如果焦点在全局管理器的搜索框里，也不处理
                if (target.id === 'am-search-input') return;

                this.lastFocusedInput = target;
            }
        });

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
            // 聚焦搜索框
            setTimeout(() => this.searchInput.focus(), 50);
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
                }
            });

            // 标题 (鼠标悬浮预览图)
            const titleEl = $el("span", {
                textContent: isPrompts ? item.title : (item.keyword || "未命名"),
                style: { flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", fontSize: "12px", cursor: "pointer" },
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
        if (this.lastFocusedInput && document.body.contains(this.lastFocusedInput)) {
            try {
                // 仅当该元素支持 selection 时才处理
                if (typeof this.lastFocusedInput.selectionStart === 'number') {
                    const start = this.lastFocusedInput.selectionStart;
                    const end = this.lastFocusedInput.selectionEnd;
                    const val = this.lastFocusedInput.value || "";
                    
                    const before = val.substring(0, start);
                    const after = val.substring(end);
                    
                    this.lastFocusedInput.value = before + toInsert + after;
                    
                    // 触发事件通知框架
                    this.lastFocusedInput.dispatchEvent(new Event('input', { bubbles: true }));
                    this.lastFocusedInput.dispatchEvent(new Event('change', { bubbles: true }));
                    
                    // 更新光标
                    const newPos = start + toInsert.length;
                    this.lastFocusedInput.setSelectionRange(newPos, newPos);
                    this.lastFocusedInput.focus();
                    success = true;
                }
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
