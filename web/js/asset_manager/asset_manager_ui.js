import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";
import { $el } from "../../../scripts/ui.js";
import { AMDialog } from "./am_dialog.js";
import { cssStyles } from "./asset_manager_style.js";
import { PreviewHandler } from "./asset_manager_preview_handler.js";
import { DataHandler } from "./asset_manager_data_handler.js";
import { DragSelectHandler } from "./asset_manager_drag_select.js";

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
    initSelectionAndClipboard() { DragSelectHandler.initSelectionAndClipboard(this, this.modal.querySelector('.am-content')); }
    updateSelectionBoxRect(box, x1, y1, x2, y2) { DragSelectHandler.updateSelectionBoxRect(box, x1, y1, x2, y2); }
    updateSelectionUI() { DragSelectHandler.updateSelectionUI(this); }
    selectAll() { DragSelectHandler.selectAll(this); }
    deleteSelected() { DragSelectHandler.deleteSelected(this); }
    copySelected() { DragSelectHandler.copySelected(this); }
    pasteClipboard() { DragSelectHandler.pasteClipboard(this); }
    showContextMenu(x, y) { DragSelectHandler.showContextMenu(this, x, y); }

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
                ondragover: (e) => {
                    e.preventDefault();
                    if (e.dataTransfer.types && e.dataTransfer.types.includes("Files")) {
                        card.style.border = "2px dashed var(--am-accent)";
                    }
                },
                ondragleave: (e) => {
                    card.style.border = "";
                },
                ondrop: (e) => {
                    card.style.border = "";
                    
                    const isFiles = e.dataTransfer.files && e.dataTransfer.files.length > 0;
                    const textData = e.dataTransfer.getData("text/plain") || e.dataTransfer.getData("text/uri-list");
                    const isPathText = textData && (textData.includes(":\\") || textData.startsWith("file://") || textData.startsWith("/") || textData.match(/\.(png|jpg|jpeg|webp)$/i));
                    
                    // 排除内部卡片排序拖拽（我们在 ondragstart 中设置了 type="items" 或 type="group" 等 JSON）
                    let isInternalDrag = false;
                    try {
                        const dragData = JSON.parse(textData);
                        if (dragData.type === "items" || dragData.type === "group") isInternalDrag = true;
                    } catch(err) {}

                    // 如果不是内部排序拖拽，且有文件或路径，则触发更换预览图逻辑
                    if (!isInternalDrag && (isFiles || isPathText)) {
                        e.preventDefault();
                        e.stopPropagation();
                        this.handlePreviewDrop(e, index);
                        return;
                    }
                    
                    // 否则走正常的排序逻辑
                    this.handleDrop(e, index);
                },
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

    handlePreviewDrop(e, targetIndex) { DragSelectHandler.handlePreviewDrop(this, e, targetIndex); }
    handleDrop(e, targetIndex) { DragSelectHandler.handleDrop(this, e, targetIndex); }
    handleGroupDrop(e, targetIndex, targetEl) { DragSelectHandler.handleGroupDrop(this, e, targetIndex, targetEl); }

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

    importData() { DataHandler.importData(this); }
    exportData() { DataHandler.exportData(this); }

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
