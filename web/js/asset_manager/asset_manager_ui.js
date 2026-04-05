import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";
import { $el } from "../../../scripts/ui.js";
import { AMDialog } from "./am_dialog.js";
import { cssStyles } from "./asset_manager_style.js";

// ================= CSS 注入 =================
const style = document.createElement("style");
style.textContent = cssStyles;
document.head.appendChild(style);

// ================= 全局资产管理核心逻辑 =================
class AssetManagerUI {
    constructor() {
        this.promptsData = { groups: [] };
        this.modelsData = { groups: [] };
        this.currentTab = 'prompts'; // 'prompts' or 'models'
        this.currentGroupIndex = 0;
        this.viewMode = 'grid'; // 'grid', 'list', 'single'
        
        // 全局缓存 LoRA 列表，避免每次点击卡顿
        this.cachedLoraList = null;
        
        this.selectedIndices = new Set();
        this.lastClickedIndex = -1;
        window.amClipboard = window.amClipboard || [];
        
        this.createDOM();
        this.createDrawerDOM();
        this.loadData();
        this.preloadLoraList();
    }

    async preloadLoraList() {
        try {
            const res = await api.fetchApi("/object_info");
            const data = await res.json();
            const loras = data?.LoraLoader?.input?.required?.lora_name[0];
            if (Array.isArray(loras)) {
                this.cachedLoraList = ["None"].concat(loras);
            } else {
                this.cachedLoraList = ["None"];
            }
        } catch(e) {
            console.warn("[AssetManager] Failed to preload lora list", e);
            this.cachedLoraList = ["None"];
        }
    }

    createDrawerDOM() {
        this.drawer = $el("div", { className: "am-drawer", id: "asset-manager-drawer" });
        document.body.appendChild(this.drawer);

        // 点击外部关闭抽屉
        document.addEventListener("mousedown", (e) => {
            if (this.drawer.style.display === "block" && !this.drawer.contains(e.target)) {
                this.hideDrawer();
            }
        });
    }

    createDOM() {
        // 1. 创建悬浮球 (FAB)
        this.fab = $el("div", {
            id: "asset-manager-fab",
            textContent: "📦",
            title: "资产管理 (拖拽移动, 点击打开)"
        });
        document.body.appendChild(this.fab);
        
        // 恢复悬浮球保存的位置
        const savedPos = localStorage.getItem("am_fab_position");
        if (savedPos) {
            try {
                const pos = JSON.parse(savedPos);
                
                // 将保存的字符串值 (例如 "150px") 转换为数字
                const rightVal = parseFloat(pos.right);
                const bottomVal = parseFloat(pos.bottom);
                
                // 进行安全校验，确保位置不会超出当前窗口的边界，导致悬浮球不可见
                if (!isNaN(rightVal) && !isNaN(bottomVal)) {
                    // 如果因为浏览器窗口缩小导致 right 或 bottom 值过大，我们把它限制在安全范围内
                    const safeRight = Math.max(10, Math.min(rightVal, window.innerWidth - 60));
                    const safeBottom = Math.max(10, Math.min(bottomVal, window.innerHeight - 60));
                    
                    this.fab.style.right = `${safeRight}px`;
                    this.fab.style.bottom = `${safeBottom}px`;
                }
            } catch(e) {
                console.error("[AssetManager] Failed to restore FAB position:", e);
                // 出错时清除失效的缓存
                localStorage.removeItem("am_fab_position");
                this.fab.style.right = "30px";
                this.fab.style.bottom = "30px";
            }
        }
        
        // 绑定拖拽和点击逻辑
        let isDragging = false;
        let startX, startY, initialX, initialY;
        let moved = false;

        this.fab.addEventListener("mousedown", (e) => {
            isDragging = true;
            moved = false;
            startX = e.clientX;
            startY = e.clientY;
            
            const rect = this.fab.getBoundingClientRect();
            // 改为使用 right 和 bottom 计算，避免与初始 CSS 冲突
            initialX = window.innerWidth - rect.right;
            initialY = window.innerHeight - rect.bottom;
            
            this.fab.style.transition = "none"; // 拖拽时取消动画
            e.preventDefault(); // 阻止默认选中文本
        });

        document.addEventListener("mousemove", (e) => {
            if (!isDragging) return;
            
            const dx = e.clientX - startX;
            const dy = e.clientY - startY;
            
            if (Math.abs(dx) > 5 || Math.abs(dy) > 5) {
                moved = true; // 标记为已移动，非单纯点击
            }
            
            if (moved) {
                // 更新位置 (使用 right 和 bottom 保持与 CSS 一致)
                let newRight = initialX - dx;
                let newBottom = initialY - dy;
                
                // 边界限制
                newRight = Math.max(0, Math.min(newRight, window.innerWidth - 50));
                newBottom = Math.max(0, Math.min(newBottom, window.innerHeight - 50));
                
                this.fab.style.right = `${newRight}px`;
                this.fab.style.bottom = `${newBottom}px`;
            }
        });

        document.addEventListener("mouseup", (e) => {
            if (!isDragging) return;
            isDragging = false;
            this.fab.style.transition = "transform 0.2s"; // 恢复动画
            
            if (moved) {
                // 拖拽结束，保存位置到 localStorage
                localStorage.setItem("am_fab_position", JSON.stringify({
                    right: this.fab.style.right,
                    bottom: this.fab.style.bottom
                }));
            } else {
                // 如果没有发生明显移动，视为点击
                this.showModal();
            }
        });

        // 2. 创建模态框
        this.modal = $el("div", { id: "asset-manager-modal" }, [
            $el("div", { className: "am-container" }, [
                // Header
                $el("div", { className: "am-header" }, [
                    $el("div", { className: "am-tabs" }, [
                        $el("div", { 
                            className: "am-tab active", 
                            textContent: "📝 提示词管理 (Prompts)",
                            onclick: (e) => this.switchTab('prompts', e.target)
                        }),
                        $el("div", { 
                            className: "am-tab", 
                            textContent: "🧩 模型管理 (Models)",
                            onclick: (e) => this.switchTab('models', e.target)
                        })
                    ]),
                    $el("div", { 
                        className: "am-close", 
                        textContent: "✖",
                        onclick: () => this.hideModal()
                    })
                ]),
                // Body
                $el("div", { className: "am-body" }, [
                    // Sidebar
                    $el("div", { className: "am-sidebar" }, [
                        $el("div", { className: "am-groups", id: "am-group-list" }),
                        $el("div", { className: "am-sidebar-footer" }, [
                            $el("button", { 
                                textContent: "➕ 新建分组",
                                style: { width: "100%", padding: "5px" },
                                onclick: () => this.addGroup()
                            })
                        ])
                    ]),
                    // Content
                    $el("div", { className: "am-content" }, [
                        $el("div", { className: "am-toolbar" }, [
                            $el("button", { textContent: "➕ 新建条目", onclick: () => this.addItem() }),
                            $el("button", { textContent: "📥 导入", onclick: () => this.importData() }),
                            $el("button", { textContent: "📤 导出", onclick: () => this.exportData() }),
                            $el("span", { style: { flex: 1 } }),
                            $el("select", {
                                onchange: (e) => { this.viewMode = e.target.value; this.renderItems(); }
                            }, [
                                $el("option", { value: "grid", textContent: "网格视图" }),
                                $el("option", { value: "list", textContent: "列表视图" })
                            ])
                        ]),
                        $el("div", { className: "am-items-area am-grid", id: "am-item-list" })
                    ])
                ])
            ])
        ]);
        document.body.appendChild(this.modal);
        this.initSelectionAndClipboard();
    }

    async loadData() {
        try {
            const resP = await api.fetchApi("/a_my_nodes/assets/prompts");
            this.promptsData = await resP.json();
            if (!this.promptsData.groups) this.promptsData.groups = [];

            const resM = await api.fetchApi("/a_my_nodes/assets/models");
            this.modelsData = await resM.json();
            if (!this.modelsData.groups) this.modelsData.groups = [];
            
            this.renderGroups();
        } catch (e) {
            console.error("[AssetManager] Failed to load data:", e);
        }
    }

    async saveData() {
        try {
            const endpoint = this.currentTab === 'prompts' ? '/a_my_nodes/assets/prompts' : '/a_my_nodes/assets/models';
            const data = this.currentTab === 'prompts' ? this.promptsData : this.modelsData;
            
            await api.fetchApi(endpoint, {
                method: "POST",
                body: JSON.stringify(data),
                headers: { "Content-Type": "application/json" }
            });
            console.log(`[AssetManager] Saved ${this.currentTab} successfully.`);
        } catch (e) {
            console.error("[AssetManager] Failed to save data:", e);
        }
    }

    switchTab(tab, el) {
        this.currentTab = tab;
        this.currentGroupIndex = 0;
        this.selectedIndices.clear();
        this.lastClickedIndex = -1;
        
        // Update active class on tabs
        const tabs = this.modal.querySelectorAll('.am-tab');
        tabs.forEach(t => t.classList.remove('active'));
        el.classList.add('active');
        
        this.renderGroups();
    }

    renderGroups() {
        const listEl = document.getElementById("am-group-list");
        listEl.innerHTML = "";
        
        const data = this.currentTab === 'prompts' ? this.promptsData : this.modelsData;
        
        data.groups.forEach((group, index) => {
            const el = $el("div", {
                className: `am-group-item ${index === this.currentGroupIndex ? 'active' : ''}`,
                title: "双击重命名, 长按拖拽排序",
                style: { display: "flex", justifyContent: "space-between", alignItems: "center" },
                draggable: true, // 开启拖拽
                ondragstart: (e) => {
                    e.dataTransfer.setData("text/plain", JSON.stringify({ type: "group", index }));
                    e.stopPropagation();
                },
                ondragover: (e) => {
                    e.preventDefault();
                    el.style.borderTop = "2px solid var(--am-accent)";
                },
                ondragleave: (e) => {
                    el.style.borderTop = "";
                },
                ondrop: (e) => this.handleGroupDrop(e, index, el)
            });
            
            // 使用 span 包裹文字，方便替换为 input
            const textSpan = $el("span", { textContent: group.name || `未命名分组 ${index+1}`, style: { flex: 1, overflow: "hidden", textOverflow: "ellipsis" } });
            el.appendChild(textSpan);
            
            // 删除按钮 (仅删除当前Tab的条目)
            const delBtn = $el("span", {
                textContent: "🗑️",
                style: { cursor: "pointer", fontSize: "12px", marginLeft: "5px", opacity: "0.6" },
                onclick: async (e) => {
                    e.stopPropagation();
                    const yes = await this.confirm(`确定要清空/删除当前标签页下的【${group.name}】分组的所有数据吗？`);
                    if (yes) {
                        group.items = []; // 只清空当前侧的内容，不破坏两端对齐的索引
                        this.saveData();
                        this.renderItems();
                    }
                }
            });
            delBtn.onmouseenter = () => delBtn.style.opacity = "1";
            delBtn.onmouseleave = () => delBtn.style.opacity = "0.6";
            el.appendChild(delBtn);
            
            // 单击切换组
            el.onclick = (e) => {
                if (e.target.tagName === 'INPUT') return; // 如果正在编辑，不触发切换
                this.currentGroupIndex = index;
                this.selectedIndices.clear();
                this.lastClickedIndex = -1;
                this.renderGroups(); // re-render to update active class
            };
            
            // 双击重命名 - 绑定在 textSpan 上以避免与父级的 drag 事件冲突
            textSpan.ondblclick = (e) => {
                e.stopPropagation(); // 阻止事件冒泡
                this.enterGroupEditMode(el, group, textSpan, index);
            };

            listEl.appendChild(el);
        });
        
        this.renderItems();
    }

    enterGroupEditMode(el, group, textSpan, index) {
        // 进入编辑模式时，暂时禁用外层的拖拽，防止冲突
        el.draggable = false;
        
        const input = $el("input", {
            type: "text",
            value: group.name,
            style: { width: "100%", background: "var(--am-bg)", color: "white", border: "1px solid var(--am-accent)", padding: "2px" }
        });
        
        const saveName = () => {
            const newName = input.value.trim();
            if (newName && newName !== group.name) {
                group.name = newName;
                this.syncGroupNames(index, newName); // 同步提示词和模型组名
                this.saveData(); // 注意：这里仅保存了当前tab，其实需要都保存
                this.saveOtherData(); // 新增方法保存另一侧
            }
            // 恢复拖拽属性
            el.draggable = true;
            this.renderGroups();
        };

        input.onblur = saveName;
        input.onkeydown = (e) => {
            if (e.key === 'Enter') saveName();
            if (e.key === 'Escape') {
                el.draggable = true;
                this.renderGroups(); // 取消编辑
            }
        };

        // 替换时注意不要覆盖了垃圾桶图标
        el.replaceChild(input, textSpan);
        input.focus();
        input.select();
    }

    syncGroupNames(index, newName) {
        // 同步修改两边的同名或同索引分组
        if (this.promptsData.groups[index]) this.promptsData.groups[index].name = newName;
        if (this.modelsData.groups[index]) this.modelsData.groups[index].name = newName;
    }

    async saveOtherData() {
        try {
            const endpoint = this.currentTab !== 'prompts' ? '/a_my_nodes/assets/prompts' : '/a_my_nodes/assets/models';
            const data = this.currentTab !== 'prompts' ? this.promptsData : this.modelsData;
            await api.fetchApi(endpoint, {
                method: "POST",
                body: JSON.stringify(data),
                headers: { "Content-Type": "application/json" }
            });
        } catch (e) {
            console.error("[AssetManager] Failed to save other data:", e);
        }
    }

    // --- 多选、复制、粘贴、框选 ---
    initSelectionAndClipboard() {
        const area = this.modal.querySelector('.am-content');
        let isSelecting = false, startX, startY, selBox = null, initialSel = new Set();

        area.addEventListener('mousedown', (e) => {
            if (!e.ctrlKey || e.button !== 0) return;
            if (['INPUT', 'BUTTON', 'SELECT', 'TEXTAREA'].includes(e.target.tagName)) return;
            isSelecting = true;
            startX = e.clientX; startY = e.clientY;
            initialSel = new Set(this.selectedIndices);
            selBox = $el("div", { className: "am-selection-box" });
            document.body.appendChild(selBox);
            this.updateSelectionBoxRect(selBox, startX, startY, startX, startY);
            e.preventDefault();
        });

        document.addEventListener('mousemove', (e) => {
            if (!isSelecting) return;
            this.updateSelectionBoxRect(selBox, startX, startY, e.clientX, e.clientY);
            const boxRect = selBox.getBoundingClientRect();
            const listEl = document.getElementById("am-item-list");
            if (!listEl) return;
            
            const newSel = new Set(initialSel);
            Array.from(listEl.children).forEach((card, idx) => {
                const r = card.getBoundingClientRect();
                if (!(r.right < boxRect.left || r.left > boxRect.right || r.bottom < boxRect.top || r.top > boxRect.bottom)) {
                    newSel.add(idx);
                }
            });
            this.selectedIndices = newSel;
            this.updateSelectionUI();
        });

        document.addEventListener('mouseup', () => {
            if (isSelecting) {
                isSelecting = false;
                if (selBox && selBox.parentNode) selBox.parentNode.removeChild(selBox);
                selBox = null;
            }
        });

        document.addEventListener('keydown', (e) => {
            if (this.modal.style.display !== "flex") return;
            if (document.activeElement && ['INPUT', 'TEXTAREA'].includes(document.activeElement.tagName)) return;
            
            if (e.ctrlKey && e.key.toLowerCase() === 'a') { e.preventDefault(); this.selectAll(); }
            else if (e.key === 'Delete' || e.key === 'Backspace') { this.deleteSelected(); }
            else if (e.ctrlKey && e.key.toLowerCase() === 'c') { this.copySelected(); }
            else if (e.ctrlKey && e.key.toLowerCase() === 'v') { this.pasteClipboard(); }
        });

        area.addEventListener('contextmenu', (e) => {
            if (['INPUT', 'BUTTON', 'TEXTAREA'].includes(e.target.tagName)) return;
            e.preventDefault();
            this.showContextMenu(e.clientX, e.clientY);
        });

        document.addEventListener('click', () => {
            if (this.contextMenu && this.contextMenu.parentNode) {
                this.contextMenu.parentNode.removeChild(this.contextMenu);
                this.contextMenu = null;
            }
        });
    }

    updateSelectionBoxRect(box, x1, y1, x2, y2) {
        box.style.left = Math.min(x1, x2) + 'px';
        box.style.top = Math.min(y1, y2) + 'px';
        box.style.width = Math.abs(x1 - x2) + 'px';
        box.style.height = Math.abs(y1 - y2) + 'px';
    }

    updateSelectionUI() {
        const listEl = document.getElementById("am-item-list");
        if (!listEl) return;
        Array.from(listEl.children).forEach((card, idx) => {
            if (this.selectedIndices.has(idx)) card.classList.add('selected');
            else card.classList.remove('selected');
        });
    }

    selectAll() {
        const data = this.currentTab === 'prompts' ? this.promptsData : this.modelsData;
        const group = data.groups[this.currentGroupIndex];
        if (!group || !group.items) return;
        for (let i = 0; i < group.items.length; i++) this.selectedIndices.add(i);
        this.updateSelectionUI();
    }

    async deleteSelected() {
        if (this.selectedIndices.size === 0) return;
        const yes = await this.confirm(`确定删除选中的 ${this.selectedIndices.size} 个项目吗？`);
        if (!yes) return;
        const data = this.currentTab === 'prompts' ? this.promptsData : this.modelsData;
        const group = data.groups[this.currentGroupIndex];
        const indices = Array.from(this.selectedIndices).sort((a, b) => b - a);
        indices.forEach(idx => group.items.splice(idx, 1));
        this.selectedIndices.clear();
        this.saveData();
        this.renderItems();
    }

    copySelected() {
        if (this.selectedIndices.size === 0) return;
        const data = this.currentTab === 'prompts' ? this.promptsData : this.modelsData;
        const group = data.groups[this.currentGroupIndex];
        window.amClipboard = Array.from(this.selectedIndices).sort((a, b) => a - b).map(idx => JSON.parse(JSON.stringify(group.items[idx])));
    }

    pasteClipboard() {
        if (!window.amClipboard || window.amClipboard.length === 0) return;
        const data = this.currentTab === 'prompts' ? this.promptsData : this.modelsData;
        const group = data.groups[this.currentGroupIndex];
        if (!group) return;
        
        const sample = window.amClipboard[0];
        const isPrompt = sample.hasOwnProperty('content');
        if ((this.currentTab === 'prompts' && !isPrompt) || (this.currentTab === 'models' && isPrompt)) {
            this.alert("剪贴板数据类型与当前标签页不匹配！");
            return;
        }
        
        window.amClipboard.forEach(item => {
            const newItem = JSON.parse(JSON.stringify(item));
            newItem.id = Date.now().toString() + Math.random().toString().slice(2, 6);
            group.items.push(newItem);
        });
        this.saveData();
        this.renderItems();
    }

    showContextMenu(x, y) {
        if (this.contextMenu && this.contextMenu.parentNode) this.contextMenu.parentNode.removeChild(this.contextMenu);
        this.contextMenu = $el("div", { className: "am-context-menu", style: { left: x + 'px', top: y + 'px' } });
        
        const addMenuItem = (text, onClick, disabled = false) => {
            const item = $el("div", {
                className: "am-context-menu-item", textContent: text,
                onclick: (e) => {
                    e.stopPropagation();
                    if (!disabled) onClick();
                    if (this.contextMenu.parentNode) this.contextMenu.parentNode.removeChild(this.contextMenu);
                }
            });
            if (disabled) { item.style.opacity = "0.5"; item.style.pointerEvents = "none"; }
            this.contextMenu.appendChild(item);
        };
        
        if (this.selectedIndices.size > 0) {
            addMenuItem(`📋 复制选中 (${this.selectedIndices.size})`, () => this.copySelected());
            addMenuItem(`🗑️ 删除选中 (${this.selectedIndices.size})`, () => this.deleteSelected());
            this.contextMenu.appendChild($el("div", { className: "am-context-menu-divider" }));
        }
        
        const clipLen = window.amClipboard ? window.amClipboard.length : 0;
        addMenuItem(`📥 粘贴 (${clipLen})`, () => this.pasteClipboard(), clipLen === 0);
        
        this.contextMenu.appendChild($el("div", { className: "am-context-menu-divider" }));
        addMenuItem("✅ 全选", () => this.selectAll());
        
        document.body.appendChild(this.contextMenu);
    }

    renderItems() {
        const listEl = document.getElementById("am-item-list");
        listEl.innerHTML = "";
        listEl.className = `am-items-area am-${this.viewMode}`;
        
        const data = this.currentTab === 'prompts' ? this.promptsData : this.modelsData;
        const group = data.groups[this.currentGroupIndex];
        
        if (!group || !group.items) return;
        
        group.items.forEach((item, index) => {
            const card = $el("div", {
                className: "am-card" + (this.selectedIndices.has(index) ? " selected" : ""),
                draggable: true,
                title: "单击选择, Ctrl/Shift多选, 双击编辑",
                onclick: (e) => {
                    if (['INPUT', 'BUTTON', 'SELECT', 'TEXTAREA'].includes(e.target.tagName)) return;
                    if (e.ctrlKey) {
                        if (this.selectedIndices.has(index)) this.selectedIndices.delete(index);
                        else this.selectedIndices.add(index);
                        this.lastClickedIndex = index;
                    } else if (e.shiftKey) {
                        if (this.lastClickedIndex === -1) this.lastClickedIndex = 0;
                        const start = Math.min(this.lastClickedIndex, index);
                        const end = Math.max(this.lastClickedIndex, index);
                        this.selectedIndices.clear();
                        for(let i = start; i <= end; i++) this.selectedIndices.add(i);
                    } else {
                        this.selectedIndices.clear();
                        this.selectedIndices.add(index);
                        this.lastClickedIndex = index;
                    }
                    this.updateSelectionUI();
                },
                ondragstart: (e) => {
                    if (!this.selectedIndices.has(index)) {
                        this.selectedIndices.clear();
                        this.selectedIndices.add(index);
                        this.updateSelectionUI();
                    }
                    const indices = Array.from(this.selectedIndices);
                    e.dataTransfer.setData("text/plain", JSON.stringify({ indices, tab: this.currentTab, type: "items" }));
                },
                ondragover: (e) => e.preventDefault(),
                ondrop: (e) => this.handleDrop(e, index),
                ondblclick: (e) => {
                    e.stopPropagation();
                    this.enterItemEditMode(card, item, index);
                }
            });

            // Preview Image
            let imgSrc = "";
            if (item.preview_image) {
                imgSrc = `/a_my_nodes/assets/view_preview?path=${encodeURIComponent(item.preview_image)}`;
            } else if (this.currentTab === 'models') {
                let firstLora = "";
                if (item.high_loras && item.high_loras.length > 0 && item.high_loras[0].lora) {
                    firstLora = item.high_loras[0].lora;
                } else if (item.low_loras && item.low_loras.length > 0 && item.low_loras[0].lora) {
                    firstLora = item.low_loras[0].lora;
                }
                
                if (firstLora && firstLora !== "None") {
                    imgSrc = `/a_my_nodes/assets/view_preview?fallback_lora=${encodeURIComponent(firstLora)}`;
                }
            }

            if (imgSrc) {
                const imgEl = $el("img", { className: "am-card-img", src: imgSrc });
                imgEl.onerror = () => { imgEl.style.display = "none"; };
                card.appendChild(imgEl);
            }

            const contentWrapper = $el("div", { className: "am-card-content" });

            // Title
            contentWrapper.appendChild($el("div", { className: "am-card-title", textContent: item.title || item.keyword || "未命名" }));
            
            // Description / Content
            if (this.currentTab === 'prompts') {
                contentWrapper.appendChild($el("div", { className: "am-card-desc", textContent: item.content || "" }));
            } else {
                let displayPath = "";
                let displayStr = "";
                
                // 从 high_loras 或 low_loras 中提取第一个有效的模型作为摘要展示
                if (item.high_loras && item.high_loras.length > 0 && item.high_loras[0].lora && item.high_loras[0].lora !== "None") {
                    displayPath = item.high_loras[0].lora;
                    displayStr = item.high_loras[0].strength !== undefined ? item.high_loras[0].strength : 1.0;
                } else if (item.low_loras && item.low_loras.length > 0 && item.low_loras[0].lora && item.low_loras[0].lora !== "None") {
                    displayPath = item.low_loras[0].lora;
                    displayStr = item.low_loras[0].strength !== undefined ? item.low_loras[0].strength : 1.0;
                }
                
                let descText = "";
                if (displayPath) {
                    descText = `路径: ${displayPath}\n强度: ${displayStr}`;
                    const totalLoras = (item.high_loras?.length || 0) + (item.low_loras?.length || 0);
                    if (totalLoras > 1) descText += ` (等 ${totalLoras} 个模型)`;
                } else {
                    descText = "空配置 (请双击添加模型)";
                }
                
                contentWrapper.appendChild($el("div", { 
                    className: "am-card-desc", 
                    textContent: descText,
                    style: { whiteSpace: "pre-wrap", wordBreak: "break-all" } 
                }));
            }
            
            card.appendChild(contentWrapper);
            listEl.appendChild(card);
        });
    }

    handleDrop(e, targetIndex) {
        e.preventDefault();
        try {
            const dataStr = e.dataTransfer.getData("text/plain");
            const dragData = JSON.parse(dataStr);
            if (dragData.type === "group") return; // 忽略组拖拽事件
            if (dragData.tab !== this.currentTab) return; // 不允许跨tab拖拽
            
            const data = this.currentTab === 'prompts' ? this.promptsData : this.modelsData;
            const items = data.groups[this.currentGroupIndex].items;

            let indices = dragData.indices;
            if (!indices && dragData.index !== undefined) indices = [dragData.index]; // 兼容旧数据
            if (!indices || indices.includes(targetIndex)) return;

            // 提取被移动的元素，按索引降序删除，避免下标错乱
            const sortedIndices = indices.slice().sort((a, b) => b - a);
            const movedItems = [];
            sortedIndices.forEach(idx => {
                movedItems.push(items.splice(idx, 1)[0]);
            });
            // 因为我们是降序删除，提取出的元素也是反向的，需要再反转回来
            movedItems.reverse();

            // 重新计算插入点 (如果插入点在被删除元素之后，由于数组变短，需要修正 targetIndex)
            let shift = 0;
            sortedIndices.forEach(idx => { if (idx < targetIndex) shift++; });
            const finalTargetIndex = targetIndex - shift;

            items.splice(finalTargetIndex, 0, ...movedItems);
            
            // 更新选中状态的索引
            this.selectedIndices.clear();
            for(let i=0; i<movedItems.length; i++) this.selectedIndices.add(finalTargetIndex + i);
            
            this.renderItems();
            this.saveData(); // 自动保存排序
        } catch (err) {
            console.error("Drop parsing error", err);
        }
    }

    handleGroupDrop(e, targetIndex, targetEl) {
        e.preventDefault();
        e.stopPropagation();
        targetEl.style.borderTop = ""; // 清除高亮
        
        try {
            const dataStr = e.dataTransfer.getData("text/plain");
            const dragData = JSON.parse(dataStr);
            
            if (dragData.type !== "group") return; // 忽略非组拖拽
            
            const sourceIndex = dragData.index;
            if (sourceIndex === targetIndex) return;

            // 同步调整两边（提示词和模型）的分组顺序，保持结构一致性
            const pGroup = this.promptsData.groups.splice(sourceIndex, 1)[0];
            this.promptsData.groups.splice(targetIndex, 0, pGroup);
            
            const mGroup = this.modelsData.groups.splice(sourceIndex, 1)[0];
            this.modelsData.groups.splice(targetIndex, 0, mGroup);
            
            // 更新当前选中的索引
            if (this.currentGroupIndex === sourceIndex) {
                this.currentGroupIndex = targetIndex;
            } else if (sourceIndex < this.currentGroupIndex && targetIndex >= this.currentGroupIndex) {
                this.currentGroupIndex--;
            } else if (sourceIndex > this.currentGroupIndex && targetIndex <= this.currentGroupIndex) {
                this.currentGroupIndex++;
            }
            
            this.renderGroups();
            
            // 保存两边的数据
            this.saveData();
            this.saveOtherData();
            
        } catch (err) {
            console.error("Group drop parsing error", err);
        }
    }

    addGroup() {
        const name = "未命名分组 " + (this.promptsData.groups.length + 1);
        
        // 提示词和模型组同步添加空分组，保持索引和名称一致
        this.promptsData.groups.push({ name, items: [] });
        this.modelsData.groups.push({ name, items: [] });
        
        this.currentGroupIndex = this.promptsData.groups.length - 1;
        
        // 保存两端数据
        this.saveData();
        this.saveOtherData();
        
        this.renderGroups();
        
        // 自动触发新分组的编辑模式
        setTimeout(() => {
            const listEl = document.getElementById("am-group-list");
            const newGroupEl = listEl.lastElementChild;
            if (newGroupEl) {
                const textSpan = newGroupEl.querySelector("span");
                const data = this.currentTab === 'prompts' ? this.promptsData : this.modelsData;
                this.enterGroupEditMode(newGroupEl, data.groups[this.currentGroupIndex], textSpan, this.currentGroupIndex);
            }
        }, 50);
    }

    async addItem() {
        const data = this.currentTab === 'prompts' ? this.promptsData : this.modelsData;
        if (!data.groups || data.groups.length === 0) {
            await this.alert("请先创建一个分组！");
            return;
        }
        
        let newItem;
        if (this.currentTab === 'prompts') {
            newItem = {
                id: Date.now().toString(),
                title: "新提示词模板",
                content: "",
                preview_image: "" 
            };
        } else {
            newItem = {
                id: Date.now().toString(),
                keyword: "新模型检查词",
                check_mode: "contains",
                high_loras: [],
                low_loras: [],
                preview_image: ""
            };
        }
        
        data.groups[this.currentGroupIndex].items.push(newItem);
        this.saveData();
        this.renderItems();
        
        // 自动触发刚刚添加的这个条目的编辑模式
        setTimeout(() => {
            const listEl = document.getElementById("am-item-list");
            const newCardEl = listEl.lastElementChild;
            if (newCardEl) {
                const itemIndex = data.groups[this.currentGroupIndex].items.length - 1;
                this.enterItemEditMode(newCardEl, newItem, itemIndex);
            }
        }, 50);
    }

    enterItemEditMode(cardEl, item, index) {
        const originalItemStr = JSON.stringify(item);
        
        // 记录原始尺寸，作为占位符，防止网格塌陷或跳跃
        const rect = cardEl.getBoundingClientRect();
        cardEl.style.minHeight = `${rect.height}px`;
        cardEl.style.minWidth = `${rect.width}px`;
        
        cardEl.draggable = false;
        cardEl.classList.add("edit-mode");
        cardEl.innerHTML = ""; 

        // 创建绝对定位的悬浮编辑容器
        const editWrapper = $el("div", { 
            className: "am-edit-wrapper",
            style: {
                position: "absolute",
                top: "-1px",
                width: "450px",
                background: "var(--am-panel-bg)",
                border: "1px solid var(--am-accent)",
                borderRadius: "6px",
                padding: "10px",
                boxShadow: "0 4px 20px rgba(0,0,0,0.8)",
                display: "flex",
                flexDirection: "column",
                gap: "5px",
                zIndex: 100,
                cursor: "default"
            }
        });
        
        // 智能定位：如果靠右边，则向左展开，防止溢出屏幕
        const listRect = document.getElementById("am-item-list").getBoundingClientRect();
        if (rect.left + 450 > listRect.right) {
            editWrapper.style.right = "-1px";
        } else {
            editWrapper.style.left = "-1px";
        }
        
        cardEl.appendChild(editWrapper);

        // 点击外部取消编辑逻辑
        const outsideClickListener = (e) => {
            if (e.target.closest('.litegraph.litecontextmenu')) return;
            
            if (!editWrapper.contains(e.target)) {
                const originalItem = JSON.parse(originalItemStr);
                const data = this.currentTab === 'prompts' ? this.promptsData : this.modelsData;
                data.groups[this.currentGroupIndex].items[index] = originalItem;
                
                document.removeEventListener('mousedown', outsideClickListener);
                this.renderItems();
            }
        };

        // 延迟绑定，防止双击事件直接触发了 mousedown
        setTimeout(() => {
            document.addEventListener('mousedown', outsideClickListener);
        }, 100);

        // 辅助创建输入框
        const createInput = (label, value, key, isTextarea = false) => {
            const wrapper = $el("div", { style: { display: "flex", flexDirection: "column" } });
            wrapper.appendChild($el("label", { textContent: label, style: { fontSize: "12px", color: "#aaa", marginBottom: "2px" } }));
            
            const input = $el(isTextarea ? "textarea" : "input", {
                value: value,
                style: { 
                    width: "100%", background: "#111", color: "white", 
                    border: "1px solid var(--am-border)", padding: "4px", borderRadius: "4px",
                    resize: isTextarea ? "vertical" : "none",
                    minHeight: isTextarea ? "60px" : "auto",
                    fontFamily: isTextarea ? "monospace" : "inherit"
                }
            });
            
            input.dataset.key = key; // 保存 key 到 dataset 以便保存时取值
            
            input.onchange = (e) => { 
                if (key !== 'high_loras' && key !== 'low_loras') item[key] = e.target.value; 
            };
            wrapper.appendChild(input);
            return { wrapper, input };
        };

        const createSelect = (label, value, key, options) => {
            const wrapper = $el("div", { style: { display: "flex", flexDirection: "column", marginBottom: "5px" } });
            wrapper.appendChild($el("label", { textContent: label, style: { fontSize: "12px", color: "#aaa", marginBottom: "2px" } }));
            
            const select = $el("select", {
                style: { 
                    width: "100%", background: "#111", color: "white", 
                    border: "1px solid var(--am-border)", padding: "4px", borderRadius: "4px"
                }
            });
            
            options.forEach(opt => {
                const isSelected = value === opt;
                select.appendChild($el("option", { value: opt, textContent: opt, selected: isSelected }));
            });
            
            select.dataset.key = key;
            select.onchange = (e) => { item[key] = e.target.value; };
            wrapper.appendChild(select);
            return { wrapper, input: select };
        };

        const formElements = [];

        if (this.currentTab === 'prompts') {
            formElements.push(createInput("标题 (Title)", item.title, "title"));
            formElements.push(createInput("提示词内容 (Content)", item.content, "content", true));
            formElements.push(createInput("预览图路径 (可选)", item.preview_image || "", "preview_image"));
            formElements.forEach(fe => editWrapper.appendChild(fe.wrapper));
        } else {
            formElements.push(createInput("要检查的字符串 (key_to_check) [此为该条目的标题]", item.keyword, "keyword"));
            
            const checkModeOptions = ["absolute", "start_with", "contains", "regex", "absolute_invert", "start_with_invert", "contains_invert", "regex_invert"];
            formElements.push(createSelect("匹配模式 (check_mode)", item.check_mode || "contains", "check_mode", checkModeOptions));
            
            formElements.forEach(fe => editWrapper.appendChild(fe.wrapper));
            
            // --- Custom UI for High/Low Streams ---
            const createStreamEditor = (streamName, loraArray) => {
                const wrapper = $el("div", { style: { border: "1px solid var(--am-border)", borderRadius: "6px", padding: "10px", marginTop: "10px", background: "var(--am-bg)" } });
                const header = $el("div", { style: { display: "flex", justifyContent: "space-between", marginBottom: "10px", alignItems: "center" } });
                
                header.appendChild($el("strong", { 
                    textContent: `${streamName === "High" ? "🔼" : "🔽"} ${streamName} Stream`, 
                    style: { fontSize: "14px", color: "var(--am-text)" } 
                }));
                
                header.appendChild($el("button", {
                    textContent: "➕",
                    style: { background: "var(--am-accent)", color: "white", padding: "4px 8px", borderRadius: "4px", border: "none", cursor: "pointer", fontSize: "12px" },
                    onclick: () => {
                        loraArray.push({ lora: "", strength: 1.0, on: true });
                        renderList();
                    }
                }));
                wrapper.appendChild(header);

                const listContainer = $el("div", { style: { display: "flex", flexDirection: "column", gap: "5px" } });
                wrapper.appendChild(listContainer);

                const renderList = () => {
                    listContainer.innerHTML = "";
                    if (loraArray.length === 0) {
                        listContainer.appendChild($el("div", { textContent: "暂无模型，请点击右上角添加", style: { color: "#888", fontSize: "12px", textAlign: "center", padding: "5px" } }));
                        return;
                    }
                    loraArray.forEach((loraItem, idx) => {
                        const row = $el("div", { 
                            className: "am-lora-item",
                            draggable: true, // 恢复为 true，由下面控制
                            title: "拖拽左侧的把手 ☰ 排序模型",
                            style: { 
                                display: "flex", 
                                flexDirection: "column",
                                gap: "5px", 
                                background: "var(--am-panel-bg)", 
                                padding: "5px 8px", 
                                borderRadius: "4px", 
                                border: "1px solid #333"
                            },
                            ondragstart: (e) => {
                                // 拦截：只有从把手触发
                                if (!row.dataset.isHandleDrag) {
                                    e.preventDefault();
                                    return;
                                }
                                e.dataTransfer.effectAllowed = 'move';
                                // 设置两种格式，提高浏览器兼容性
                                const payload = JSON.stringify({ index: idx, stream: streamName, type: "lora_item" });
                                e.dataTransfer.setData('text/plain', payload);
                                e.dataTransfer.setData('application/json', payload);
                                
                                setTimeout(() => row.classList.add("dragging"), 0);
                            },
                            ondragend: () => {
                                row.classList.remove("dragging");
                                row.dataset.isHandleDrag = ""; // 清理状态
                            },
                            ondragover: (e) => {
                                e.preventDefault();
                                // 只响应模型条目的拖拽，不响应外部卡片拖拽
                                if (e.dataTransfer.types.includes('application/json') || e.dataTransfer.types.includes('text/plain')) {
                                    row.classList.add("drag-over");
                                }
                            },
                            ondragleave: () => {
                                row.classList.remove("drag-over");
                            },
                            ondrop: (e) => {
                                e.preventDefault();
                                e.stopPropagation(); // 阻止冒泡到外层卡片的 handleDrop
                                row.classList.remove("drag-over");
                                try {
                                    let dataStr = e.dataTransfer.getData("application/json");
                                    if (!dataStr) dataStr = e.dataTransfer.getData("text/plain");
                                    if (!dataStr) return;
                                    
                                    const dragData = JSON.parse(dataStr);
                                    
                                    // 确保这是模型内部排序，而不是其他类型的拖拽
                                    if (dragData.type !== "lora_item") return;
                                    if (dragData.stream !== streamName) return; 
                                    
                                    const sourceIndex = dragData.index;
                                    if (sourceIndex === idx || sourceIndex === undefined) return;

                                    // 执行数组交换
                                    const [movedItem] = loraArray.splice(sourceIndex, 1);
                                    loraArray.splice(idx, 0, movedItem);
                                    
                                    // 为了防止输入框数据丢失，必须在重新渲染前同步 DOM 的数据到 item
                                    formElements.forEach(fe => {
                                        let key = fe.input.dataset.key;
                                        if(key && key !== 'high_loras' && key !== 'low_loras') {
                                            item[key] = fe.input.value;
                                        }
                                    });

                                    renderList();
                                } catch (err) {
                                    console.error("Lora reorder drop error", err);
                                }
                            }
                        });

                        const topRow = $el("div", {
                            style: { display: "flex", gap: "5px", alignItems: "center", width: "100%", position: "relative" }
                        });

                        // Drag Handle - 鼠标按住时激活标志位
                        const handle = $el("span", {
                            textContent: "☰",
                            style: { cursor: "grab", color: "#666", padding: "0 4px", userSelect: "none", flexShrink: 0 },
                            onmousedown: () => { row.dataset.isHandleDrag = "true"; },
                            onmouseup: () => { row.dataset.isHandleDrag = ""; },
                            onmouseleave: () => { row.dataset.isHandleDrag = ""; }
                        });
                        topRow.appendChild(handle);

                        // Checkbox
                        const cb = $el("input", { type: "checkbox", checked: loraItem.on !== false, title: "是否启用", style: { flexShrink: 0 } });
                        cb.onchange = (e) => loraItem.on = e.target.checked;
                        topRow.appendChild(cb);

                        // Lora Name Input - 支持展开全称或超出边界查看
                        const nameInput = $el("input", {
                            type: "text",
                            className: "am-lora-input",
                            value: loraItem.lora,
                            placeholder: "点击右侧按钮选择...",
                            title: loraItem.lora || "", // hover 时显示全称
                            style: { 
                                flex: "1", 
                                background: "#111", 
                                color: "white", 
                                border: "1px solid var(--am-border)", 
                                padding: "4px", 
                                borderRadius: "3px",
                                minWidth: "0",
                                textOverflow: "ellipsis" // 超长时显示省略号
                            }
                        });
                        nameInput.onchange = (e) => {
                            loraItem.lora = e.target.value;
                            nameInput.title = e.target.value;
                        };
                        topRow.appendChild(nameInput);
                        
                        // Select Lora Button
                        const chooseBtn = $el("button", {
                            textContent: "📂",
                            title: "打开高级模型选择器",
                            style: { background: "var(--am-panel-bg)", border: "1px solid var(--am-border)", cursor: "pointer", padding: "4px", borderRadius: "3px", flexShrink: 0 },
                            onclick: (e) => {
                                e.stopPropagation();
                                this.showNativeLoraChooser(e, (value) => {
                                    if (typeof value === "string" && value && value !== "None") {
                                        nameInput.value = value;
                                        nameInput.title = value; // 更新title
                                        loraItem.lora = value;
                                    }
                                });
                            }
                        });
                        topRow.appendChild(chooseBtn);

                        const bottomRow = $el("div", {
                            style: { display: "flex", gap: "5px", alignItems: "center", justifyContent: "space-between", width: "100%" }
                        });
                        const strContainer = $el("div", { style: { display: "flex", alignItems: "center", gap: "5px" } });
                        strContainer.appendChild($el("span", { textContent: "强度:", style: { fontSize: "12px", color: "#aaa" } }));
                        
                        const strInput = $el("input", {
                            type: "number",
                            step: "0.01",
                            value: loraItem.strength ?? 1.0,
                            title: "模型强度 (Strength)",
                            style: { width: "60px", background: "#111", color: "white", border: "1px solid var(--am-border)", padding: "4px", borderRadius: "3px" }
                        });
                        strInput.onchange = (e) => loraItem.strength = parseFloat(e.target.value);
                        strContainer.appendChild(strInput);
                        bottomRow.appendChild(strContainer);

                        // Delete
                        const delBtn = $el("button", {
                            textContent: "❌",
                            title: "移除该模型",
                            style: { background: "transparent", border: "1px solid #f44336", borderRadius: "3px", cursor: "pointer", color: "#f44336", padding: "2px 5px", fontSize: "12px" },
                            onclick: () => {
                                loraArray.splice(idx, 1);
                                renderList();
                            }
                        });
                        bottomRow.appendChild(delBtn);

                        row.appendChild(topRow);
                        row.appendChild(bottomRow);

                        listContainer.appendChild(row);
                    });
                };

                renderList();
                return wrapper;
            };

            if (!Array.isArray(item.high_loras)) item.high_loras = [];
            if (!Array.isArray(item.low_loras)) item.low_loras = [];

            editWrapper.appendChild(createStreamEditor("High", item.high_loras));
            editWrapper.appendChild(createStreamEditor("Low", item.low_loras));

            const previewFe = createInput("预览图路径 (可选)", item.preview_image || "", "preview_image");
            formElements.push(previewFe);
            editWrapper.appendChild(previewFe.wrapper);
        }

        // 按钮区
        const btnRow = $el("div", { style: { display: "flex", justifyContent: "space-between", marginTop: "15px" } });
        
        const saveBtn = $el("button", {
            textContent: "💾 保存",
            style: { background: "var(--am-accent)", color: "white", flex: 1, marginRight: "5px", padding: "8px", borderRadius: "4px", border: "none", cursor: "pointer" },
            onclick: async () => {
                try {
                    formElements.forEach(fe => {
                        let key = fe.input.dataset.key;
                        let val = fe.input.value;
                        item[key] = val; // 原生赋值，UI列表直接修改了数组，无需序列化
                    });
                } catch (e) {
                    await this.alert("保存失败！\n" + e.message);
                    return;
                }
                
                document.removeEventListener('mousedown', outsideClickListener);
                this.saveData();
                this.renderItems();
            }
        });

        const delBtn = $el("button", {
            textContent: "🗑️ 删除",
            style: { background: "#a00", color: "white" },
            onclick: async () => {
                const yes = await this.confirm("确定要删除这个条目吗？");
                if(yes) {
                    document.removeEventListener('mousedown', outsideClickListener);
                    const data = this.currentTab === 'prompts' ? this.promptsData : this.modelsData;
                    data.groups[this.currentGroupIndex].items.splice(index, 1);
                    this.saveData();
                    this.renderItems();
                }
            }
        });

        btnRow.appendChild(saveBtn);
        btnRow.appendChild(delBtn);
        editWrapper.appendChild(btnRow);

        // 自动聚焦第一个输入框
        if (formElements.length > 0) {
            formElements[0].input.focus();
            formElements[0].input.select();
        }
    }

    importData() {
        const data = this.currentTab === 'prompts' ? this.promptsData : this.modelsData;
        if (!data.groups || data.groups.length === 0) {
            alert("请先创建一个分组！");
            return;
        }

        const overlay = $el("div", {
            style: { position: "fixed", top: 0, left: 0, right: 0, bottom: 0, background: "rgba(0,0,0,0.8)", zIndex: 11000, display: "flex", justifyContent: "center", alignItems: "center" }
        });

        let fileContent = "";
        
        const processImportContent = (jsonStr) => {
            try {
                const parsed = JSON.parse(jsonStr);
                const currentItems = data.groups[this.currentGroupIndex].items;
                let addCount = 0;
                let skipCount = 0;

                const resolveConflict = (newItem, isPrompt) => {
                    let titleKey = isPrompt ? 'title' : 'keyword';
                    
                    // 由于模型配置比较复杂，冲突判断主要基于 title/keyword 
                    // 如果名字相同，则直接追加编号，因为很难判断深层配置是否"完全一致"
                    let finalTitle = newItem[titleKey];
                    let isExactMatch = false;

                    if (isPrompt) {
                        const exactMatch = currentItems.find(i => i.title === newItem.title && i.content === newItem.content);
                        if (exactMatch) isExactMatch = true;
                    } else {
                        // 对于模型配置，我们做一个粗略的深度匹配（如果标题一样且内容序列化一样）
                        const currentJSON = JSON.stringify(newItem);
                        const exactMatch = currentItems.find(i => {
                            if (i.keyword !== newItem.keyword) return false;
                            // 临时剔除 id 等不参与比较的字段进行比对
                            const iCopy = { ...i };
                            delete iCopy.id;
                            const nCopy = { ...newItem };
                            delete nCopy.id;
                            return JSON.stringify(iCopy) === JSON.stringify(nCopy);
                        });
                        if (exactMatch) isExactMatch = true;
                    }

                    if (isExactMatch) {
                        skipCount++;
                        return;
                    }

                    // 标题相同但内容不同，自动编号
                    let counter = 1;
                    while (currentItems.find(i => i[titleKey] === finalTitle)) {
                        finalTitle = `${newItem[titleKey]} (${counter})`;
                        counter++;
                    }
                    
                    newItem[titleKey] = finalTitle;
                    newItem.id = Date.now().toString() + Math.random().toString().slice(2, 6);
                    currentItems.push(newItem);
                    addCount++;
                };

                if (this.currentTab === 'prompts') {
                    // 兼容 text_input_batch.js 数组格式
                    if (Array.isArray(parsed)) {
                        parsed.forEach(item => {
                            if (item.title !== undefined || item.content !== undefined) {
                                resolveConflict({
                                    title: item.title || "未命名",
                                    content: item.content || "",
                                    preview_image: ""
                                }, true);
                            }
                        });
                    }
                } else {
                    // 兼容 wan_video_double_stream.js 导出的节点级配置
                    // 如果它是一个对象且包含 high/low/key_to_check，这就是一个完整的条目
                    if (!Array.isArray(parsed) && (parsed.high || parsed.low || parsed.key_to_check)) {
                        resolveConflict({
                            keyword: parsed.key_to_check || parsed.keyword || "未命名配置",
                            check_mode: parsed.check_mode || "contains",
                            high_loras: parsed.high?.loras || [],
                            low_loras: parsed.low?.loras || [],
                            preview_image: ""
                        }, false);
                    } else if (Array.isArray(parsed)) {
                        // 兼容之前导出的是纯 lora 数组的旧情况
                        const highLoras = [];
                        parsed.forEach(loraItem => {
                            if (loraItem.lora && loraItem.lora !== "None") {
                                highLoras.push({
                                    lora: loraItem.lora,
                                    strength: loraItem.strength || 1.0,
                                    on: loraItem.on !== false
                                });
                            }
                        });
                        if (highLoras.length > 0) {
                            resolveConflict({
                                keyword: "批量导入的旧模型",
                                check_mode: "contains",
                                high_loras: highLoras,
                                low_loras: [],
                                preview_image: ""
                            }, false);
                        }
                    }
                }

                this.saveData();
                this.renderItems();
                alert(`导入完成！\n新增: ${addCount} 条\n跳过(完全重复): ${skipCount} 条`);
                document.body.removeChild(overlay);
            } catch (err) {
                alert("JSON 解析失败，请检查格式是否正确。\n" + err.message);
            }
        };

        const dialog = $el("div", {
            style: { background: "var(--am-panel-bg)", padding: "20px", borderRadius: "8px", border: "1px solid var(--am-border)", color: "white", width: "500px" }
        }, [
            $el("h3", { textContent: `导入到 [${data.groups[this.currentGroupIndex].name}]`, style: { marginTop: 0 } }),
            $el("div", { style: { marginBottom: "10px" } }, [
                $el("label", { textContent: "粘贴 JSON 文本:", style: { display: "block", marginBottom: "5px" } }),
                $el("textarea", { 
                    id: "am-import-textarea",
                    placeholder: "在此粘贴 JSON 内容...", 
                    style: { width: "100%", height: "150px", background: "#111", color: "white", border: "1px solid #444", padding: "5px" } 
                })
            ]),
            $el("div", { style: { marginBottom: "20px" } }, [
                $el("label", { textContent: "或 选择 JSON 文件:", style: { display: "block", marginBottom: "5px" } }),
                $el("input", { 
                    type: "file", accept: ".json",
                    onchange: (e) => {
                        const file = e.target.files[0];
                        if (file) {
                            const reader = new FileReader();
                            reader.onload = (re) => { document.getElementById("am-import-textarea").value = re.target.result; };
                            reader.readAsText(file);
                        }
                    }
                })
            ]),
            $el("div", { style: { display: "flex", gap: "10px", justifyContent: "flex-end" } }, [
                $el("button", { textContent: "✔️ 确认导入", style: { background: "var(--am-accent)", color: "white" }, onclick: () => {
                    const text = document.getElementById("am-import-textarea").value.trim();
                    if (!text) { alert("请先粘贴内容或选择文件"); return; }
                    processImportContent(text);
                } }),
                $el("button", { textContent: "取消", onclick: () => document.body.removeChild(overlay) })
            ])
        ]);

        overlay.appendChild(dialog);
        document.body.appendChild(overlay);
    }

    exportData() {
        const data = this.currentTab === 'prompts' ? this.promptsData : this.modelsData;
        const group = data.groups[this.currentGroupIndex];
        if (!group || !group.items || group.items.length === 0) {
            alert("当前分组为空，没有可导出的数据！");
            return;
        }

        // 构建兼容导出格式
        let exportObj;
        if (this.currentTab === 'prompts') {
            // 兼容 text_input_batch.js
            exportObj = group.items.map(item => ({
                title: item.title,
                content: item.content,
                enabled: true
            }));
        } else {
            // 兼容 wan_video_double_stream.js (导出为 single stream list)
            exportObj = group.items.map(item => ({
                lora: item.model_path,
                strength: item.strength,
                on: item.on !== false
            }));
            // 或者导出为带 high/low 的对象，这里统一导出为一个数组便于分享
        }

        const jsonStr = JSON.stringify(exportObj, null, 2);
        
        // 创建一个简单的弹窗来选择导出方式
        const overlay = $el("div", {
            style: { position: "fixed", top: 0, left: 0, right: 0, bottom: 0, background: "rgba(0,0,0,0.8)", zIndex: 11000, display: "flex", justifyContent: "center", alignItems: "center" }
        });
        
        const dialog = $el("div", {
            style: { background: "var(--am-panel-bg)", padding: "20px", borderRadius: "8px", border: "1px solid var(--am-border)", color: "white", width: "400px" }
        }, [
            $el("h3", { textContent: `导出 [${group.name}]`, style: { marginTop: 0 } }),
            $el("textarea", { value: jsonStr, readOnly: true, style: { width: "100%", height: "200px", background: "#111", color: "#ddd", border: "1px solid #444", marginBottom: "15px" } }),
            $el("div", { style: { display: "flex", gap: "10px", justifyContent: "flex-end" } }, [
                $el("button", { textContent: "📋 复制到剪贴板", onclick: () => {
                    navigator.clipboard.writeText(jsonStr).then(() => {
                        alert("已复制到剪贴板！");
                        document.body.removeChild(overlay);
                    });
                } }),
                $el("button", { textContent: "💾 导出为 JSON 文件", onclick: () => {
                    const blob = new Blob([jsonStr], { type: "application/json" });
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement("a");
                    a.href = url;
                    a.download = `${this.currentTab}_${group.name}.json`;
                    a.click();
                    URL.revokeObjectURL(url);
                    document.body.removeChild(overlay);
                } }),
                $el("button", { textContent: "取消", onclick: () => document.body.removeChild(overlay) })
            ])
        ]);
        
        overlay.appendChild(dialog);
        document.body.appendChild(overlay);
    }

    showModal() {
        this.loadData(); // 每次打开时重新加载最新数据
        this.modal.style.display = "flex";
    }

    hideModal() {
        this.modal.style.display = "none";
    }

    async showNativeLoraChooser(event, callback) {
        // 使用全局缓存的列表，避免每次点击发起网络请求造成卡顿
        let loras = this.cachedLoraList;
        if (!loras) {
            await this.preloadLoraList();
            loras = this.cachedLoraList || ["None"];
        }

        // 构建 LiteGraph 菜单项
        const menuItems = loras.map(lora => ({
            content: lora,
            callback: () => callback(lora)
        }));

        // 构造一个伪造的鼠标事件给 LiteGraph，强制它在点击按钮的位置弹出
        let targetX = event.clientX;
        let targetY = event.clientY;
        const menuEvent = new MouseEvent('contextmenu', {
            clientX: targetX,
            clientY: targetY,
            bubbles: true,
            cancelable: true,
            view: window
        });

        // 临时隐藏我们的弹窗或提升层级，以免 ContextMenu 被遮挡
        // LiteGraph 的菜单 z-index 通常比较低
        const originalZIndex = this.modal.style.zIndex;
        this.modal.style.zIndex = "100"; 
        
        // 调用 LiteGraph 原生菜单
        const contextMenu = new LiteGraph.ContextMenu(menuItems, {
            event: menuEvent,
            title: "选择 LoRA 模型",
            className: "dark",
            callback: () => {
                // 菜单关闭后恢复弹窗层级
                setTimeout(() => { this.modal.style.zIndex = originalZIndex; }, 100);
            }
        });
        
        // 如果用户点击了外部导致菜单关闭，也需要恢复层级
        const oldClose = contextMenu.close;
        contextMenu.close = function(...args) {
            window.AssetManager.modal.style.zIndex = originalZIndex;
            return oldClose.apply(this, args);
        };
    }
    async alert(msg) {
        return AMDialog.alert(msg);
    }

    async confirm(msg) {
        return AMDialog.confirm(msg);
    }

    // ========== 抽屉 (Drawer) 相关 ==========
    async showDrawer(node, tabType, callback) {
        await this.loadData();
        
        // 计算节点在屏幕上的位置
        const canvas = app.canvas;
        const rect = canvas.canvas.getBoundingClientRect();
        
        // node.pos is [x, y] in canvas coordinates
        const scale = canvas.ds.scale;
        const offset = canvas.ds.offset;
        
        const screenX = (node.pos[0] + offset[0]) * scale + rect.left;
        // 放在节点底部
        const screenY = (node.pos[1] + node.size[1] + offset[1]) * scale + rect.top;
        
        this.drawer.style.left = `${screenX}px`;
        this.drawer.style.top = `${screenY}px`;
        this.drawer.style.display = "block";
        this.drawer.innerHTML = ""; // 清空
        
        // 顶部添加一个明显的标题和关闭按钮栏
        const drawerHeader = $el("div", { 
            style: { 
                display: "flex", 
                justifyContent: "space-between", 
                alignItems: "center",
                borderBottom: "1px solid var(--am-accent)", 
                marginBottom: "10px",
                paddingBottom: "5px"
            } 
        });
        drawerHeader.appendChild($el("span", { textContent: tabType === 'prompts' ? "选择提示词模板" : "选择模型配置", style: { fontWeight: "bold", fontSize: "14px" } }));
        drawerHeader.appendChild($el("button", {
            textContent: "✖",
            title: "关闭",
            style: { background: "transparent", border: "none", color: "white", cursor: "pointer", fontSize: "16px" },
            onclick: () => this.hideDrawer()
        }));
        this.drawer.appendChild(drawerHeader);
        
        // 绑定点击外部隐藏抽屉事件（使用 pointerdown 并绑定到 window 层级进行全局捕获）
        if (this._drawerOutsideClickListener) {
            window.removeEventListener('pointerdown', this._drawerOutsideClickListener, true);
        }
        this._drawerOutsideClickListener = (e) => {
            if (this.drawer && this.drawer.style.display === "block") {
                // 如果点击不在抽屉内，且不是触发按钮
                if (!this.drawer.contains(e.target)) {
                    // 放宽判断条件，避开按钮上的文字或 svg 等子元素
                    const isBtn = e.target.closest('.rgthree-better-button') || 
                                  (e.target.textContent && e.target.textContent.includes('插入模板'));
                    if (!isBtn) {
                        this.hideDrawer();
                    }
                }
            }
        };
        // 延迟绑定避免立即触发，第三个参数 true 开启捕获模式
        setTimeout(() => {
            window.addEventListener('pointerdown', this._drawerOutsideClickListener, true);
        }, 100);
        
        // 渲染抽屉内容
        const data = tabType === 'prompts' ? this.promptsData : this.modelsData;
        
        if (!data.groups || data.groups.length === 0) {
            this.drawer.appendChild($el("div", { style: { padding: "10px" }, textContent: "暂无数据，请先在资产管理中添加" }));
            return;
        }

        // 按组渲染
        data.groups.forEach(group => {
            const groupEl = $el("div", { style: { marginBottom: "10px" } });
            groupEl.appendChild($el("div", { 
                style: { fontWeight: "bold", borderBottom: "1px solid var(--am-border)", marginBottom: "5px" },
                textContent: `📁 ${group.name}`
            }));
            
            if (group.items && group.items.length > 0) {
                group.items.forEach(item => {
                    const itemEl = $el("div", { 
                        className: "am-drawer-item",
                        style: { display: "flex", alignItems: "center", gap: "5px" },
                        onmouseenter: (e) => this.showTooltip(e, item, tabType),
                        onmouseleave: () => this.hideTooltip()
                    });

                    if (tabType === 'prompts') {
                        // 提示词：点击直接回调插入
                        itemEl.textContent = item.title;
                        itemEl.onclick = () => {
                            callback(item.content);
                            this.hideDrawer();
                        };
                    } else {
                        // 模型配置模板：直接点击该条目应用（不再使用复选框多选）
                        itemEl.appendChild($el("span", { textContent: item.keyword || "未命名配置" }));
                        
                        itemEl.onclick = () => {
                            // 由于原 callback 期望的是数组，我们把它包装成单元素数组返回
                            callback([item]);
                            this.hideDrawer();
                        };
                    }
                    
                    groupEl.appendChild(itemEl);
                });
            } else {
                groupEl.appendChild($el("div", { style: { color: "#888", fontSize: "12px" }, textContent: "空分组" }));
            }
            this.drawer.appendChild(groupEl);
        });
    }

    hideDrawer() {
        if (this.drawer) this.drawer.style.display = "none";
        this.hideTooltip();
        if (this._drawerOutsideClickListener) {
            window.removeEventListener('pointerdown', this._drawerOutsideClickListener, true);
            this._drawerOutsideClickListener = null;
        }
    }

    toggleDrawerCheckboxes(check) {
        const cbs = this.drawer.querySelectorAll(".am-model-cb");
        cbs.forEach(cb => {
            if (check === false) {
                cb.checked = !cb.checked; // 反选
            } else {
                cb.checked = true; // 全选
            }
        });
    }

    getDrawerSelectedModels() {
        const cbs = this.drawer.querySelectorAll(".am-model-cb");
        const selected = [];
        cbs.forEach(cb => {
            if (cb.checked) {
                const item = JSON.parse(cb.dataset.modelInfo);
                selected.push({
                    lora: item.model_path,
                    strength: item.strength || 1.0,
                    on: true
                });
            }
        });
        return selected;
    }

    // 悬浮预览图 (Tooltip)
    showTooltip(e, item, type) {
        if (!this.tooltip) {
            this.tooltip = $el("div", {
                style: {
                    position: "fixed",
                    background: "rgba(20,20,20,0.95)",
                    border: "1px solid var(--am-accent)",
                    color: "white",
                    padding: "10px",
                    borderRadius: "6px",
                    zIndex: 3000,
                    pointerEvents: "none",
                    maxWidth: "250px",
                    boxShadow: "0 5px 15px rgba(0,0,0,0.5)"
                }
            });
            document.body.appendChild(this.tooltip);
        }
        
        this.tooltip.innerHTML = "";
        
        let imgSrc = "";
        if (item.preview_image) {
            imgSrc = `/a_my_nodes/assets/view_preview?path=${encodeURIComponent(item.preview_image)}`;
        } else if (type === 'models') {
            let firstLora = "";
            if (item.high_loras && item.high_loras.length > 0 && item.high_loras[0].lora) {
                firstLora = item.high_loras[0].lora;
            } else if (item.low_loras && item.low_loras.length > 0 && item.low_loras[0].lora) {
                firstLora = item.low_loras[0].lora;
            }
            
            if (firstLora && firstLora !== "None") {
                imgSrc = `/a_my_nodes/assets/view_preview?fallback_lora=${encodeURIComponent(firstLora)}`;
            }
        }
        
        if (imgSrc) {
            const imgEl = $el("img", { 
                src: imgSrc, 
                style: { width: "100%", borderRadius: "4px", marginBottom: "5px", background: "black" }
            });
            imgEl.onerror = () => { imgEl.style.display = "none"; };
            this.tooltip.appendChild(imgEl);
        }
        
        if (type === 'prompts') {
            this.tooltip.appendChild($el("div", { style: { fontSize: "12px", whiteSpace: "pre-wrap" }, textContent: item.content }));
        } else {
            this.tooltip.appendChild($el("div", { style: { fontSize: "12px" }, textContent: `强度: ${item.strength}` }));
        }
        
        const rect = e.target.getBoundingClientRect();
        this.tooltip.style.left = `${rect.right + 10}px`;
        this.tooltip.style.top = `${rect.top}px`;
        this.tooltip.style.display = "block";
    }

    hideTooltip() {
        if (this.tooltip) this.tooltip.style.display = "none";
    }
}

// ================= 注册扩展 =================
app.registerExtension({
    name: "A_my_nodes.AssetManager",
    setup() {
        // 延迟初始化全局实例，确保 document.body 和其他 UI 元素已经准备完毕
        setTimeout(() => {
            if (!window.AssetManager) {
                window.AssetManager = new AssetManagerUI();
            }
        }, 1000); // 延迟1秒注入，避开 ComfyUI 初始化的 DOM 重绘风暴
        
        // 挂载右键菜单
        const originalGetCanvasMenuOptions = LGraphCanvas.prototype.getCanvasMenuOptions;
        LGraphCanvas.prototype.getCanvasMenuOptions = function () {
            const options = originalGetCanvasMenuOptions.apply(this, arguments) || [];
            options.push(null); // separator
            options.push({
                content: "✨ 资产管理 (Asset Manager)",
                callback: () => {
                    window.AssetManager.showModal();
                }
            });
            return options;
        };
    }
});
