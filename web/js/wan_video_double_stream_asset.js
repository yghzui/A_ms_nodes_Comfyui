import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { showTopNotification } from "./utils/shared_utils.js";

// 生成UUID的简单实现
function generateUUID() {
    return Date.now().toString(36) + Math.random().toString(36).substring(2);
}

app.registerExtension({
    name: "A_my_nodes.video.WanVideoDoubleStreamAsset",
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "WanVideoDoubleStreamAsset") {
            
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                this.assetsData = []; // 存储所有可用的模型资产
                this.groupsList = []; // 存储所有组名
                this.currentGroupFilter = "All"; // 当前选择的组过滤器
                this.viewMode = "list"; // 'list' 或 'grid'
                this.searchKeyword = ""; // 当前搜索关键字
                this.searchDebounceTimer = null; // 搜索防抖定时器
                this.assetDisplayOrder = []; // 资产显示顺序，支持拖拽后真实改变位置
                this.hoverPreviewEl = null; // 当前悬浮预览图
                try {
                    const savedViewMode = localStorage.getItem("wan_video_double_stream_asset_view_mode");
                    if (savedViewMode === "list" || savedViewMode === "grid") {
                        this.viewMode = savedViewMode;
                    }
                } catch (e) {}
                this.selectedAssets = []; // 存储当前选中的资产，保持顺序
                this.hitStatus = {}; // 存储运行后的命中状态 {id: true/false}

                // 找到隐藏的 selected_assets 和 current_group widget，如果在后端没有被完美隐藏，在前端强制隐藏
                this.selectedWidget = this.widgets?.find(w => w.name === "selected_assets");
                if (!this.selectedWidget) {
                    this.selectedWidget = this.addWidget("string", "selected_assets", "[]", () => {}, { hidden: true });
                }
                this.selectedWidget.type = "hidden";
                this.selectedWidget.computeSize = () => [0, -4];
                
                this.currentGroupWidget = this.widgets?.find(w => w.name === "current_group");
                if (!this.currentGroupWidget) {
                    this.currentGroupWidget = this.addWidget("string", "current_group", "All", () => {}, { hidden: true });
                }
                this.currentGroupWidget.type = "hidden";
                this.currentGroupWidget.computeSize = () => [0, -4];

                // 添加 DOM Widget 用于显示列表
                this.domWidget = this.addDOMWidget("AssetList", "div", document.createElement("div"), {
                    serialize: false,
                    hideOnZoom: false
                });

                // 设置一个合理的初始最小尺寸
                if (!this.size || this.size[0] < 400 || this.size[1] < 400) {
                    this.size = [450, 450];
                }

                const container = this.domWidget.element;
                container.style.width = "100%";
                container.style.height = "100%";
                container.style.display = "flex";
                container.style.flexDirection = "column";
                container.style.minHeight = "0";
                container.style.overflow = "hidden";
                container.style.backgroundColor = "var(--bg-color, #222)";
                container.style.border = "1px solid var(--border-color, #444)";
                container.style.padding = "5px";
                container.style.color = "var(--fg-color, #fff)";
                container.style.fontFamily = "sans-serif";
                container.style.fontSize = "12px";

                this.renderContainer = container;
                this._handleHoverPreviewWindowBlur = () => {
                    this.removeHoverPreview();
                };
                window.addEventListener("blur", this._handleHoverPreviewWindowBlur);

                // execution_start 是全局事件，这里用“节点实例级监听”确保 this 指向当前节点
                // 并避免整表重渲染：优先局部更新
                this._handleExecutionStart = () => {
                    if (this.hitStatus && Object.keys(this.hitStatus).length > 0) {
                        this.hitStatus = {};
                        if (!this.refreshVisibleItemStates()) {
                            this.renderList();
                        }
                    }
                };
                api.addEventListener("execution_start", this._handleExecutionStart);

                // 初始化时加载数据并还原选中状态
                this.loadAssetsData().then(() => {
                    this.restoreSelection();
                    this.renderList();
                });

                return r;
            };

            // 监听执行完毕事件，更新命中状态
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                if (onExecuted) onExecuted.apply(this, arguments);
                
                // 清除旧状态
                this.hitStatus = {};

                if (message && message.hit_status) {
                    message.hit_status.forEach(status => {
                        if (status.id) {
                            this.hitStatus[status.id] = status.hit;
                        }
                    });
                }
                if (!this.refreshVisibleItemStates()) {
                    this.renderList();
                }
            };

            // 节点移除时清理事件监听，避免泄漏
            const onRemoved = nodeType.prototype.onRemoved;
            nodeType.prototype.onRemoved = function () {
                try {
                    if (this._handleHoverPreviewWindowBlur) {
                        window.removeEventListener("blur", this._handleHoverPreviewWindowBlur);
                        this._handleHoverPreviewWindowBlur = null;
                    }
                    if (this._handleExecutionStart) {
                        api.removeEventListener("execution_start", this._handleExecutionStart);
                        this._handleExecutionStart = null;
                    }
                    if (this.removeHoverPreview) this.removeHoverPreview();
                } catch (e) {}

                if (onRemoved) return onRemoved.apply(this, arguments);
            };

                nodeType.prototype.loadAssetsData = async function () {
                try {
                    const res = await api.fetchApi("/a_my_nodes/assets/models");
                    const data = await res.json();
                    this.assetsData = [];
                    this.groupsList = [];
                    if (data.groups) {
                        data.groups.forEach(group => {
                            if (group.name && !this.groupsList.includes(group.name)) {
                                this.groupsList.push(group.name);
                            }
                            if (group.items) {
                                group.items.forEach(item => {
                                    this.assetsData.push({
                                        ...item,
                                        groupName: group.name,
                                        // 确保有唯一 ID
                                        id: item.id || generateUUID()
                                    });
                                });
                            }
                        });
                    }
                    this.syncAssetDisplayOrder();
                } catch (e) {
                    console.error("[WanVideoDoubleStreamAsset] Failed to load models data:", e);
                }
            };

            nodeType.prototype.restoreSelection = function () {
                try {
                    const saved = JSON.parse(this.selectedWidget.value);
                    if (Array.isArray(saved)) {
                        this.selectedAssets = saved;
                    }
                } catch (e) {
                    this.selectedAssets = [];
                }
                
                if (this.currentGroupWidget && this.currentGroupWidget.value) {
                    this.currentGroupFilter = this.currentGroupWidget.value;
                }
            };

            nodeType.prototype.updateWidgetValue = function () {
                if (this.selectedWidget) {
                    this.selectedWidget.value = JSON.stringify(this.selectedAssets);
                }
                if (typeof this.setDirtyCanvas === "function") {
                    this.setDirtyCanvas(true, true);
                }
                if (this.graph && typeof this.graph.setDirtyCanvas === "function") {
                    this.graph.setDirtyCanvas(true, true);
                } else if (app.graph && typeof app.graph.setDirtyCanvas === "function") {
                    app.graph.setDirtyCanvas(true, true);
                }
            };

            nodeType.prototype.getPreviewImageSrc = function (item) {
                if (!item) return "";
                if (item.preview_image) {
                    return `/a_my_nodes/assets/view_preview?path=${encodeURIComponent(item.preview_image)}`;
                }

                let firstLora = "";
                if (item.high_loras && item.high_loras.length > 0 && item.high_loras[0].lora) {
                    firstLora = item.high_loras[0].lora;
                } else if (item.low_loras && item.low_loras.length > 0 && item.low_loras[0].lora) {
                    firstLora = item.low_loras[0].lora;
                }

                if (firstLora && firstLora !== "None") {
                    return `/a_my_nodes/assets/view_preview?fallback_lora=${encodeURIComponent(firstLora)}`;
                }

                return "";
            };

            nodeType.prototype.removeHoverPreview = function () {
                if (this.hoverPreviewEl && this.hoverPreviewEl.parentNode) {
                    this.hoverPreviewEl.remove();
                }
                this.hoverPreviewEl = null;
            };

            nodeType.prototype.updateHoverPreviewPosition = function (clientX, clientY) {
                if (!this.hoverPreviewEl) return;
                this.hoverPreviewEl.style.left = (clientX + 15) + "px";
                this.hoverPreviewEl.style.top = (clientY + 15) + "px";
            };

            nodeType.prototype.buildSelectedAssetMap = function () {
                const map = new Map();
                this.selectedAssets.forEach((sel, selectedIndex) => {
                    map.set(sel.id, {
                        enable_mode: sel.enable_mode || "Auto",
                        selectedIndex
                    });
                });
                return map;
            };

            nodeType.prototype.syncAssetDisplayOrder = function () {
                const currentIds = this.assetsData.map(asset => asset.id);
                const currentIdSet = new Set(currentIds);
                const existingOrder = Array.isArray(this.assetDisplayOrder) ? this.assetDisplayOrder : [];
                const nextOrder = existingOrder.filter(id => currentIdSet.has(id));
                const orderedIdSet = new Set(nextOrder);

                currentIds.forEach(id => {
                    if (!orderedIdSet.has(id)) {
                        nextOrder.push(id);
                        orderedIdSet.add(id);
                    }
                });

                this.assetDisplayOrder = nextOrder;
                return nextOrder;
            };

            nodeType.prototype.getOrderedAssetsData = function () {
                const orderedIds = this.syncAssetDisplayOrder();
                const assetMap = new Map(this.assetsData.map(asset => [asset.id, asset]));
                return orderedIds.map(id => assetMap.get(id)).filter(Boolean);
            };

            nodeType.prototype.refreshVisibleItemStates = function () {
                const container = this.renderContainer;
                if (!container) return false;
                const listContainer = container.querySelector(".am-list-container");
                if (!listContainer) return false;

                const latestSelectedMap = this.buildSelectedAssetMap();
                let refreshed = false;
                listContainer.querySelectorAll(".am-asset-item").forEach(itemEl => {
                    const selectedInfo = latestSelectedMap.get(itemEl.dataset.assetId) || null;
                    if (typeof itemEl._applySelectionState === "function") {
                        itemEl._applySelectionState(selectedInfo);
                        refreshed = true;
                    }
                });
                return refreshed;
            };

            nodeType.prototype.reorderVisibleItemsInDom = function (listContainer) {
                const itemsContainer = listContainer?.querySelector(".am-items");
                if (!itemsContainer) return false;

                const itemNodes = Array.from(itemsContainer.querySelectorAll(".am-asset-item"));
                if (itemNodes.length === 0) return false;

                const nodeMap = new Map(itemNodes.map(node => [node.dataset.assetId, node]));
                const orderedVisibleNodes = this.syncAssetDisplayOrder()
                    .filter(id => nodeMap.has(id))
                    .map(id => nodeMap.get(id));

                if (orderedVisibleNodes.length !== itemNodes.length) return false;

                orderedVisibleNodes.forEach(node => itemsContainer.appendChild(node));
                return true;
            };

            nodeType.prototype.getSearchTextForAsset = function (asset) {
                return [
                    asset?.keyword || "",
                    asset?.groupName || "",
                    asset?.check_mode || ""
                ].join(" ");
            };

            nodeType.prototype.getLocalSearchMatchSet = function (normalizedSearch) {
                const matchedIds = new Set();
                if (!normalizedSearch) {
                    this.assetsData.forEach(asset => matchedIds.add(asset.id));
                    return matchedIds;
                }

                this.assetsData.forEach(asset => {
                    const searchText = this.getSearchTextForAsset(asset).toLowerCase();
                    if (searchText.includes(normalizedSearch)) {
                        matchedIds.add(asset.id);
                    }
                });
                return matchedIds;
            };

            nodeType.prototype.getSearchMatchSet = async function (normalizedSearch) {
                if (!normalizedSearch) {
                    return this.getLocalSearchMatchSet("");
                }

                try {
                    const texts = this.assetsData.map(asset => this.getSearchTextForAsset(asset));
                    const res = await api.fetchApi("/a_my_nodes/assets/search_pinyin", {
                        method: "POST",
                        body: JSON.stringify({ texts, keyword: normalizedSearch }),
                        headers: { "Content-Type": "application/json" }
                    });
                    const data = await res.json();
                    if (data && Array.isArray(data.matches) && data.matches.length === this.assetsData.length) {
                        const matchedIds = new Set();
                        data.matches.forEach((matched, idx) => {
                            if (matched && this.assetsData[idx]) {
                                matchedIds.add(this.assetsData[idx].id);
                            }
                        });
                        return matchedIds;
                    }
                } catch (e) {
                    console.warn("[WanVideoDoubleStreamAsset] Search API failed, fallback to local search:", e);
                }

                return this.getLocalSearchMatchSet(normalizedSearch);
            };

            nodeType.prototype.renderList = async function () {
                const container = this.renderContainer;
                if (!container) return;

                this.removeHoverPreview();
                container.innerHTML = "";

                const normalizedSearch = (this.searchKeyword || "").trim().toLowerCase();
                const currentRenderSeq = (this._renderListSeq || 0) + 1;
                this._renderListSeq = currentRenderSeq;
                const matchedAssetIds = await this.getSearchMatchSet(normalizedSearch);
                if (this._renderListSeq !== currentRenderSeq) return;

                const matchesSearch = (asset) => matchedAssetIds.has(asset.id);

                // --- 渲染工具栏 ---
                const toolbar = document.createElement("div");
                toolbar.style.display = "flex";
                toolbar.style.alignItems = "center";
                toolbar.style.justifyContent = "space-between";
                toolbar.style.marginBottom = "8px";
                toolbar.style.paddingBottom = "5px";
                toolbar.style.borderBottom = "1px solid #555";
                toolbar.style.flexShrink = "0"; // 防止工具栏被压缩
                
                // 左侧操作按钮组
                const btnGroup = document.createElement("div");
                btnGroup.style.display = "flex";
                btnGroup.style.gap = "5px";
                btnGroup.style.alignItems = "center";
                btnGroup.style.flexWrap = "wrap";

                const btnSelectAll = document.createElement("button");
                btnSelectAll.textContent = "全选";
                btnSelectAll.style.fontSize = "10px";
                btnSelectAll.style.cursor = "pointer";
                btnSelectAll.onclick = () => {
                    this.assetsData.forEach(asset => {
                        if ((this.currentGroupFilter === "All" || asset.groupName === this.currentGroupFilter) && matchesSearch(asset)) {
                            if (!this.selectedAssets.find(s => s.id === asset.id)) {
                                this.selectedAssets.push({
                                    id: asset.id,
                                    enable_mode: "Auto"
                                });
                            }
                        }
                    });
                    this.updateWidgetValue();
                    if (!this.refreshVisibleItemStates()) {
                        this.renderList();
                    }
                };

                const btnInvertSelect = document.createElement("button");
                btnInvertSelect.textContent = "反选";
                btnInvertSelect.style.fontSize = "10px";
                btnInvertSelect.style.cursor = "pointer";
                btnInvertSelect.onclick = () => {
                    const toAdd = [];
                    const toRemove = [];
                    
                    this.assetsData.forEach(asset => {
                        if ((this.currentGroupFilter === "All" || asset.groupName === this.currentGroupFilter) && matchesSearch(asset)) {
                            const isSelected = !!this.selectedAssets.find(s => s.id === asset.id);
                            if (isSelected) {
                                toRemove.push(asset.id);
                            } else {
                                toAdd.push({
                                    id: asset.id,
                                    enable_mode: "Auto"
                                });
                            }
                        }
                    });

                    this.selectedAssets = this.selectedAssets.filter(s => !toRemove.includes(s.id));
                    this.selectedAssets.push(...toAdd);
                    this.updateWidgetValue();
                    if (!this.refreshVisibleItemStates()) {
                        this.renderList();
                    }
                };

                btnGroup.appendChild(btnSelectAll);
                btnGroup.appendChild(btnInvertSelect);
                
                // 增加刷新按钮
                const btnRefresh = document.createElement("button");
                btnRefresh.textContent = "刷新";
                btnRefresh.style.fontSize = "10px";
                btnRefresh.style.cursor = "pointer";
                btnRefresh.onclick = async () => {
                    btnRefresh.textContent = "加载中...";
                    btnRefresh.disabled = true;
                    await this.loadAssetsData();
                    this.renderList();
                    btnRefresh.textContent = "刷新";
                    btnRefresh.disabled = false;
                };
                btnGroup.appendChild(btnRefresh);

                const searchWrapper = document.createElement("div");
                searchWrapper.style.position = "relative";
                searchWrapper.style.display = "flex";
                searchWrapper.style.alignItems = "center";

                const searchInput = document.createElement("input");
                searchInput.type = "text";
                searchInput.className = "am-search-input";
                searchInput.placeholder = "搜索标题/分组";
                searchInput.value = this.searchKeyword || "";
                searchInput.style.width = "180px";
                searchInput.style.padding = "3px 22px 3px 6px";
                searchInput.style.fontSize = "10px";
                searchInput.style.color = "#fff";
                searchInput.style.background = "#333";
                searchInput.style.border = "1px solid #555";
                searchInput.style.borderRadius = "3px";
                searchInput.style.outline = "none";
                searchInput.title = "按标题、分组或模式搜索";
                let isSearchComposing = false;
                const rerenderSearchInput = (selectionStart, selectionEnd) => {
                    this.renderList();
                    requestAnimationFrame(() => {
                        const nextInput = this.renderContainer?.querySelector(".am-search-input");
                        if (nextInput) {
                            nextInput.focus();
                            if (typeof selectionStart === "number" && typeof selectionEnd === "number") {
                                nextInput.setSelectionRange(selectionStart, selectionEnd);
                            }
                        }
                    });
                };
                const scheduleSearchRender = (selectionStart, selectionEnd, delay = 150) => {
                    if (this.searchDebounceTimer) {
                        clearTimeout(this.searchDebounceTimer);
                    }
                    this.searchDebounceTimer = setTimeout(() => {
                        this.searchDebounceTimer = null;
                        rerenderSearchInput(selectionStart, selectionEnd);
                    }, delay);
                };
                searchInput.addEventListener("compositionstart", () => {
                    isSearchComposing = true;
                    if (this.searchDebounceTimer) {
                        clearTimeout(this.searchDebounceTimer);
                        this.searchDebounceTimer = null;
                    }
                });
                searchInput.addEventListener("compositionend", (e) => {
                    isSearchComposing = false;
                    this.searchKeyword = e.target.value;
                    const end = e.target.value.length;
                    scheduleSearchRender(end, end);
                });
                searchInput.oninput = (e) => {
                    if (isSearchComposing) return;
                    const target = e.target;
                    const selectionStart = target.selectionStart ?? target.value.length;
                    const selectionEnd = target.selectionEnd ?? target.value.length;
                    this.searchKeyword = target.value;
                    scheduleSearchRender(selectionStart, selectionEnd);
                };

                const clearSearchBtn = document.createElement("button");
                clearSearchBtn.textContent = "x";
                clearSearchBtn.style.position = "absolute";
                clearSearchBtn.style.right = "4px";
                clearSearchBtn.style.top = "50%";
                clearSearchBtn.style.transform = "translateY(-50%)";
                clearSearchBtn.style.width = "14px";
                clearSearchBtn.style.height = "14px";
                clearSearchBtn.style.padding = "0";
                clearSearchBtn.style.border = "none";
                clearSearchBtn.style.borderRadius = "50%";
                clearSearchBtn.style.background = this.searchKeyword ? "#666" : "transparent";
                clearSearchBtn.style.color = "#fff";
                clearSearchBtn.style.cursor = this.searchKeyword ? "pointer" : "default";
                clearSearchBtn.style.display = this.searchKeyword ? "inline-flex" : "none";
                clearSearchBtn.style.alignItems = "center";
                clearSearchBtn.style.justifyContent = "center";
                clearSearchBtn.title = "清除搜索";
                clearSearchBtn.onclick = () => {
                    if (!this.searchKeyword) return;
                    if (this.searchDebounceTimer) {
                        clearTimeout(this.searchDebounceTimer);
                        this.searchDebounceTimer = null;
                    }
                    this.searchKeyword = "";
                    this.renderList();
                    requestAnimationFrame(() => {
                        const nextInput = this.renderContainer?.querySelector(".am-search-input");
                        if (nextInput) nextInput.focus();
                    });
                };

                searchWrapper.appendChild(searchInput);
                searchWrapper.appendChild(clearSearchBtn);
                btnGroup.appendChild(searchWrapper);
                
                toolbar.appendChild(btnGroup);

                // 右侧过滤器和视图切换
                const rightControls = document.createElement("div");
                rightControls.style.display = "flex";
                rightControls.style.gap = "5px";
                rightControls.style.alignItems = "center";
                rightControls.style.flexWrap = "wrap";

                const groupSelect = document.createElement("select");
                groupSelect.style.background = "#333";
                groupSelect.style.color = "#fff";
                groupSelect.style.border = "1px solid #555";
                groupSelect.style.fontSize = "10px";
                
                const allOpt = document.createElement("option");
                allOpt.value = "All";
                allOpt.textContent = "所有分组";
                groupSelect.appendChild(allOpt);
                
                this.groupsList.forEach(gName => {
                    const opt = document.createElement("option");
                    opt.value = gName;
                    opt.textContent = gName;
                    groupSelect.appendChild(opt);
                });
                
                groupSelect.value = this.currentGroupFilter;
                groupSelect.onchange = (e) => {
                    this.currentGroupFilter = e.target.value;
                    // 同步更新后端的隐藏字段 current_group
                    const cgWidget = this.widgets?.find(w => w.name === "current_group");
                    if (cgWidget) cgWidget.value = this.currentGroupFilter;
                    this.renderList();
                };

                const viewSelect = document.createElement("select");
                viewSelect.style.background = "#333";
                viewSelect.style.color = "#fff";
                viewSelect.style.border = "1px solid #555";
                viewSelect.style.fontSize = "10px";
                ["list", "grid"].forEach(v => {
                    const opt = document.createElement("option");
                    opt.value = v;
                    opt.textContent = v === "list" ? "列表" : "网格";
                    viewSelect.appendChild(opt);
                });
                viewSelect.value = this.viewMode;
                viewSelect.onchange = (e) => {
                    this.viewMode = e.target.value;
                    try {
                        localStorage.setItem("wan_video_double_stream_asset_view_mode", this.viewMode);
                    } catch (err) {}
                    this.renderList();
                };

                rightControls.appendChild(groupSelect);
                rightControls.appendChild(viewSelect);
                toolbar.appendChild(rightControls);
                
                container.appendChild(toolbar);

                // 统一列表：保持原始资产顺序，避免勾选后条目跳动
                const selectedAssetMap = this.buildSelectedAssetMap();

                const orderedAssets = this.getOrderedAssetsData();
                const visibleItems = [];
                orderedAssets.forEach(asset => {
                    if ((this.currentGroupFilter === "All" || asset.groupName === this.currentGroupFilter) && matchesSearch(asset)) {
                        const selectedInfo = selectedAssetMap.get(asset.id);
                        visibleItems.push({
                            ...asset,
                            enable_mode: selectedInfo?.enable_mode || "Auto",
                            selected: !!selectedInfo,
                            selectedIndex: selectedInfo?.selectedIndex ?? -1
                        });
                    }
                });

                const renderItem = (item, isSelected, selectedIndex) => {
                    const div = document.createElement("div");
                    div.className = "am-asset-item";
                    div.dataset.assetId = item.id;
                    div.style.display = this.viewMode === "list" ? "flex" : "inline-flex";
                    div.style.flexDirection = this.viewMode === "list" ? "row" : "column";
                    if (this.viewMode === "grid") {
                        div.style.width = "100px";
                        div.style.height = "120px";
                        div.style.margin = "5px";
                        div.style.verticalAlign = "top";
                        div.style.position = "relative";
                        div.style.justifyContent = "space-between";
                    } else {
                        div.style.alignItems = "center";
                        div.style.padding = "4px";
                        div.style.marginBottom = "2px";
                    }
                    div.style.border = "1px solid #555";
                    div.style.borderRadius = "4px";
                    div.style.cursor = "default";

                    // 右键菜单支持
                    div.addEventListener("contextmenu", (e) => {
                        e.preventDefault();
                        e.stopPropagation();
                        // 移除已有的菜单
                        const existingMenu = document.getElementById("am-asset-context-menu");
                        if (existingMenu) existingMenu.remove();

                        const menu = document.createElement("div");
                        menu.id = "am-asset-context-menu";
                        menu.style.position = "fixed";
                        menu.style.left = e.clientX + "px";
                        menu.style.top = e.clientY + "px";
                        menu.style.background = "#222";
                        menu.style.border = "1px solid #555";
                        menu.style.padding = "5px";
                        menu.style.zIndex = "10000";
                        menu.style.boxShadow = "0 4px 6px rgba(0,0,0,0.5)";

                        const editBtn = document.createElement("div");
                        editBtn.textContent = "✏️ 在资产管理中编辑";
                        editBtn.style.padding = "5px 10px";
                        editBtn.style.cursor = "pointer";
                        editBtn.onmouseenter = () => editBtn.style.background = "#444";
                        editBtn.onmouseleave = () => editBtn.style.background = "transparent";
                        editBtn.onclick = async () => {
                            menu.remove();
                            if (window.AssetManager) {
                                await window.AssetManager.showModal();
                                window.AssetManager.switchTab('models', window.AssetManager.modal.querySelector('.am-tabs').children[1]);
                                
                                // 查找组索引和条目索引
                                const gIndex = window.AssetManager.modelsData.groups.findIndex(g => g.name === item.groupName);
                                if (gIndex !== -1) {
                                    window.AssetManager.currentGroupIndex = gIndex;
                                    window.AssetManager.renderGroups();
                                    
                                    const iIndex = window.AssetManager.modelsData.groups[gIndex].items.findIndex(i => i.id === item.id);
                                    if (iIndex !== -1) {
                                        window.AssetManager.selectedIndices.clear();
                                        window.AssetManager.selectedIndices.add(iIndex);
                                        window.AssetManager.renderItems();
                                        
                                        // 触发编辑模式
                                        setTimeout(() => {
                                            const listEl = document.getElementById("am-item-list");
                                            const cardEl = listEl.children[iIndex];
                                            if (cardEl) {
                                                cardEl.scrollIntoView({ block: "center" });
                                                window.AssetManager.enterItemEditMode(cardEl, window.AssetManager.modelsData.groups[gIndex].items[iIndex], iIndex);
                                            }
                                        }, 100);
                                    }
                                }
                            } else {
                                showTopNotification("资产管理器未加载", "error");
                            }
                        };
                        menu.appendChild(editBtn);

                        const copyBtn = document.createElement("div");
                        copyBtn.textContent = "📋 复制标题名";
                        copyBtn.style.padding = "5px 10px";
                        copyBtn.style.cursor = "pointer";
                        copyBtn.onmouseenter = () => copyBtn.style.background = "#444";
                        copyBtn.onmouseleave = () => copyBtn.style.background = "transparent";
                        copyBtn.onclick = () => {
                            menu.remove();
                            navigator.clipboard.writeText(item.keyword || '未命名').then(() => {
                                // 可以在这里添加一个小的提示，但通常浏览器自带反馈
                            }).catch(err => {
                                console.error('复制失败', err);
                            });
                        };
                        menu.appendChild(copyBtn);

                        document.body.appendChild(menu);

                        const closeMenu = () => {
                            if (menu.parentNode) menu.remove();
                            document.removeEventListener("click", closeMenu);
                        };
                        setTimeout(() => document.addEventListener("click", closeMenu), 0);
                    });

                    const dragHandle = document.createElement("span");
                    dragHandle.textContent = "☰";
                    dragHandle.style.color = "#ccc";
                    if (this.viewMode === "grid") {
                        dragHandle.style.position = "absolute";
                        dragHandle.style.top = "2px";
                        dragHandle.style.right = "2px";
                        dragHandle.style.background = "rgba(0,0,0,0.5)";
                        dragHandle.style.padding = "2px";
                        dragHandle.style.borderRadius = "2px";
                    } else {
                        dragHandle.style.width = "16px";
                        dragHandle.style.display = "inline-block";
                        dragHandle.style.marginRight = "5px";
                    }
                    dragHandle.addEventListener("dragstart", (e) => {
                        if (parseInt(div.dataset.selectedIndex || "-1", 10) < 0) {
                            e.preventDefault();
                            return;
                        }
                        e.dataTransfer.setData("text/plain", item.id);
                        e.dataTransfer.effectAllowed = "move";
                        div.style.opacity = "0.5";
                    });

                    dragHandle.addEventListener("dragend", () => {
                        div.style.opacity = "1";
                    });

                    div.addEventListener("dragover", (e) => {
                        e.preventDefault();
                        e.dataTransfer.dropEffect = "move";
                        if (this.viewMode === "list") {
                            const rect = div.getBoundingClientRect();
                            const dropAfter = (e.clientY - rect.top) > (rect.height / 2);
                            div.style.borderTop = dropAfter ? "1px solid transparent" : "2px solid #007bff";
                            div.style.borderBottom = dropAfter ? "2px solid #007bff" : "1px solid transparent";
                        } else {
                            div.style.border = "2px solid #007bff";
                        }
                    });

                    div.addEventListener("dragleave", () => {
                        if (this.viewMode === "list") {
                            div.style.borderTop = "1px solid transparent";
                            div.style.borderBottom = "1px solid transparent";
                        }
                        else if (div.dataset.selectedIndex && div.dataset.selectedIndex !== "-1") div.style.border = "1px solid #007bff";
                    });

                    div.addEventListener("drop", (e) => {
                        e.preventDefault();
                        div.style.borderTop = "1px solid transparent";
                        div.style.borderBottom = "1px solid transparent";
                        const sourceAssetId = e.dataTransfer.getData("text/plain");
                        if (!sourceAssetId) return;
                        if (sourceAssetId === item.id) return;

                        const rect = div.getBoundingClientRect();
                        const dropAfter = (e.clientY - rect.top) > (rect.height / 2);
                        const newDisplayOrder = this.syncAssetDisplayOrder().slice();
                        const sourcePos = newDisplayOrder.indexOf(sourceAssetId);
                        const targetPos = newDisplayOrder.indexOf(item.id);
                        if (sourcePos === -1 || targetPos === -1) return;

                        let insertPos = targetPos + (dropAfter ? 1 : 0);
                        if (sourcePos < insertPos) {
                            insertPos -= 1;
                        }
                        if (sourcePos === insertPos) return;

                        newDisplayOrder.splice(sourcePos, 1);
                        newDisplayOrder.splice(insertPos, 0, sourceAssetId);
                        this.assetDisplayOrder = newDisplayOrder;

                        const selectedAssetMap = new Map(this.selectedAssets.map(sel => [sel.id, { ...sel }]));
                        this.selectedAssets = newDisplayOrder
                            .filter(id => selectedAssetMap.has(id))
                            .map(id => selectedAssetMap.get(id));

                        this.updateWidgetValue();
                        const listContainer = div.closest(".am-list-container");
                        const reordered = listContainer ? this.reorderVisibleItemsInDom(listContainer) : false;
                        const refreshed = this.refreshVisibleItemStates();
                        if (!reordered && !refreshed) {
                            this.renderList();
                        }
                    });

                    div.appendChild(dragHandle);

                    // 预览图（如果有且是网格模式）
                    if (this.viewMode === "grid") {
                        const imgSrc = this.getPreviewImageSrc(item);
                        
                        const imgContainer = document.createElement("div");
                        imgContainer.style.width = "100%";
                        imgContainer.style.height = "60px";
                        imgContainer.style.background = "#111";
                        imgContainer.style.overflow = "hidden";
                        imgContainer.style.display = "flex";
                        imgContainer.style.alignItems = "center";
                        imgContainer.style.justifyContent = "center";

                        if (imgSrc) {
                            const img = document.createElement("img");
                            img.src = imgSrc;
                            img.style.maxWidth = "100%";
                            img.style.maxHeight = "100%";
                            img.style.objectFit = "cover";
                            img.onerror = () => { img.style.display = "none"; };
                            imgContainer.appendChild(img);
                        } else {
                            imgContainer.textContent = "无图";
                            imgContainer.style.color = "#555";
                        }
                        div.appendChild(imgContainer);
                    }

                    div.addEventListener("mouseenter", (e) => {
                        if (this.viewMode !== "list") return;
                        const imgSrc = this.getPreviewImageSrc(item);
                        if (!imgSrc) return;

                        this.removeHoverPreview();
                        this.hoverPreviewEl = document.createElement("img");
                        this.hoverPreviewEl.src = imgSrc;
                        this.hoverPreviewEl.style.position = "fixed";
                        this.hoverPreviewEl.style.maxWidth = "200px";
                        this.hoverPreviewEl.style.maxHeight = "200px";
                        this.hoverPreviewEl.style.objectFit = "cover";
                        this.hoverPreviewEl.style.border = "2px solid #555";
                        this.hoverPreviewEl.style.borderRadius = "4px";
                        this.hoverPreviewEl.style.zIndex = "10001";
                        this.hoverPreviewEl.style.pointerEvents = "none";
                        this.hoverPreviewEl.style.boxShadow = "0 4px 8px rgba(0,0,0,0.5)";
                        this.hoverPreviewEl.style.background = "#111";
                        this.updateHoverPreviewPosition(e.clientX, e.clientY);
                        document.body.appendChild(this.hoverPreviewEl);
                    });

                    div.addEventListener("mousemove", (e) => {
                        this.updateHoverPreviewPosition(e.clientX, e.clientY);
                    });

                    div.addEventListener("mouseleave", () => {
                        this.removeHoverPreview();
                    });

                    // 控制区 (复选框 + 名称)
                    const controlRow = document.createElement("div");
                    controlRow.style.display = "flex";
                    controlRow.style.alignItems = "center";
                    if (this.viewMode === "grid") {
                        controlRow.style.padding = "2px";
                        controlRow.style.flex = "1";
                        controlRow.style.overflow = "hidden";
                    } else {
                        controlRow.style.flex = "1";
                        controlRow.style.overflow = "hidden";
                    }

                    // 复选框
                    const checkbox = document.createElement("input");
                    checkbox.type = "checkbox";
                    checkbox.style.marginRight = "4px";
                    if (this.viewMode === "grid") {
                        checkbox.style.position = "absolute";
                        checkbox.style.top = "2px";
                        checkbox.style.left = "2px";
                        checkbox.style.zIndex = "10";
                    }
                    checkbox.addEventListener("change", (e) => {
                        const isChecked = e.target.checked;
                        if (isChecked) {
                            this.selectedAssets.push({
                                id: item.id,
                                enable_mode: "Auto" // 仅保存必须的状态：id和enable_mode
                            });
                        } else {
                            this.selectedAssets = this.selectedAssets.filter(sel => sel.id !== item.id);
                        }
                        this.updateWidgetValue();
                        if (!this.refreshVisibleItemStates()) {
                            this.renderList();
                        }
                    });
                    
                    if (this.viewMode === "list") controlRow.appendChild(checkbox);
                    else div.appendChild(checkbox); // 网格模式下复选框绝对定位

                    const orderBadge = document.createElement("span");
                    orderBadge.style.display = "none";
                    orderBadge.style.minWidth = "16px";
                    orderBadge.style.padding = "0 4px";
                    orderBadge.style.marginRight = "4px";
                    orderBadge.style.borderRadius = "8px";
                    orderBadge.style.background = "#4b6280";
                    orderBadge.style.color = "#e6edf3";
                    orderBadge.style.fontSize = "8px";
                    orderBadge.style.lineHeight = "14px";
                    orderBadge.style.textAlign = "center";
                    orderBadge.style.flexShrink = "0";
                    controlRow.appendChild(orderBadge);

                    // 名称与组名
                    const titleDiv = document.createElement("div");
                    titleDiv.style.flex = "1";
                    titleDiv.style.overflow = "hidden";
                    titleDiv.style.textOverflow = "ellipsis";
                    if (this.viewMode === "grid") {
                        titleDiv.style.display = "-webkit-box";
                        titleDiv.style.webkitLineClamp = "2";
                        titleDiv.style.webkitBoxOrient = "vertical";
                        titleDiv.style.whiteSpace = "normal";
                        titleDiv.style.fontSize = "10px";
                        titleDiv.style.lineHeight = "1.2";
                    } else {
                        titleDiv.style.whiteSpace = "nowrap";
                    }
                    
                    const hitBadge = (isSelected && this.hitStatus[item.id] === true) ? `<span style="color:#28a745; font-weight:bold; margin-right:4px;">✅</span>` : '';
                    if (this.viewMode === "grid") {
                        titleDiv.innerHTML = `${hitBadge}<strong>${item.keyword || '未命名'}</strong>`;
                    } else {
                        titleDiv.innerHTML = `${hitBadge}<strong>${item.keyword || '未命名'}</strong> <span style="color:#888; font-size:10px;">[${item.groupName}]</span>`;
                    }
                    titleDiv.title = `触发词: ${item.keyword}\n模式: ${item.check_mode}\n分组: ${item.groupName}`;
                    controlRow.appendChild(titleDiv);
                    
                    div.appendChild(controlRow);

                    const select = document.createElement("select");
                    select.style.background = "#333";
                    select.style.color = "#fff";
                    select.style.border = "1px solid #555";
                    select.style.padding = "1px";
                    if (this.viewMode === "grid") {
                        select.style.width = "100%";
                        select.style.fontSize = "9px";
                        select.style.marginTop = "2px";
                    } else {
                        select.style.marginLeft = "5px";
                        select.style.fontSize = "10px";
                    }
                    
                    ["Auto", "True", "False"].forEach(opt => {
                        const option = document.createElement("option");
                        option.value = opt;
                        option.textContent = opt;
                        select.appendChild(option);
                    });
                    
                    select.addEventListener("change", (e) => {
                        const selItem = this.selectedAssets.find(s => s.id === item.id);
                        if (selItem) {
                            selItem.enable_mode = e.target.value;
                            this.updateWidgetValue();
                        }
                    });
                    
                    if (this.viewMode === "grid") div.appendChild(select);
                    else div.appendChild(select);

                    div._applySelectionState = (selectedInfo) => {
                        const selected = !!selectedInfo;
                        div.dataset.selectedIndex = selected ? String(selectedInfo.selectedIndex) : "-1";
                        checkbox.checked = selected;
                        select.value = selected ? selectedInfo.enable_mode : "Auto";
                        select.style.display = selected ? "" : "none";

                        if (selected && this.hitStatus[item.id] === true) {
                            div.style.backgroundColor = "rgba(40, 167, 69, 0.3)";
                            div.style.borderColor = "#28a745";
                        } else if (selected && this.hitStatus[item.id] === false) {
                            div.style.backgroundColor = "rgba(108, 117, 125, 0.2)";
                            div.style.borderColor = "#555";
                        } else if (selected) {
                            div.style.backgroundColor = "rgba(0, 123, 255, 0.1)";
                            div.style.borderColor = "#007bff";
                        } else {
                            div.style.backgroundColor = "transparent";
                            div.style.borderColor = "#555";
                        }

                        if (this.viewMode === "grid") {
                            dragHandle.style.display = selected ? "inline-block" : "none";
                            dragHandle.style.cursor = selected ? "grab" : "default";
                        } else {
                            dragHandle.style.cursor = selected ? "grab" : "default";
                            dragHandle.draggable = selected;
                            dragHandle.textContent = selected ? "☰" : "";
                        }
                        dragHandle.draggable = selected;
                        dragHandle.title = selected ? `拖拽调整已选顺序 (${selectedInfo.selectedIndex + 1})` : "";

                        const hitBadge = (selected && this.hitStatus[item.id] === true) ? `<span style="color:#28a745; font-weight:bold; margin-right:4px;">✅</span>` : "";
                        if (this.viewMode === "grid") {
                            titleDiv.innerHTML = `${hitBadge}<strong>${item.keyword || '未命名'}</strong>`;
                        } else {
                            titleDiv.innerHTML = `${hitBadge}<strong>${item.keyword || '未命名'}</strong> <span style="color:#888; font-size:10px;">[${item.groupName}]</span>`;
                        }

                        orderBadge.style.display = selected ? "inline-block" : "none";
                        orderBadge.textContent = selected ? `${selectedInfo.selectedIndex + 1}` : "";
                    };

                    div._applySelectionState(isSelected ? { enable_mode: item.enable_mode, selectedIndex } : null);

                    return div;
                };

                // --- 渲染列表容器 ---
                const listContainer = document.createElement("div");
                listContainer.className = "am-list-container";
                listContainer.style.flex = "1";
                listContainer.style.minHeight = "0";
                listContainer.style.height = "0";
                listContainer.style.overflowX = "hidden";
                listContainer.style.overflowY = "scroll";
                listContainer.style.scrollbarWidth = "thin";
                listContainer.style.scrollbarColor = "#666 #222";
                listContainer.addEventListener("scroll", () => this.removeHoverPreview(), { passive: true });
                listContainer.addEventListener("wheel", () => this.removeHoverPreview(), { passive: true });
                listContainer.addEventListener("mouseleave", () => this.removeHoverPreview());

                const ensureListScrollbarStyle = () => {
                    if (document.getElementById("wan-video-double-stream-asset-scroll-style")) return;
                    const styleEl = document.createElement("style");
                    styleEl.id = "wan-video-double-stream-asset-scroll-style";
                    styleEl.textContent = `
                        .am-list-container::-webkit-scrollbar {
                            width: 10px;
                        }
                        .am-list-container::-webkit-scrollbar-track {
                            background: #222;
                        }
                        .am-list-container::-webkit-scrollbar-thumb {
                            background: #666;
                            border-radius: 999px;
                            border: 2px solid #222;
                        }
                        .am-list-container::-webkit-scrollbar-thumb:hover {
                            background: #888;
                        }
                    `;
                    document.head.appendChild(styleEl);
                };
                ensureListScrollbarStyle();
                
                if (this._renderListToken) {
                    cancelAnimationFrame(this._renderListToken);
                }
                const currentPass = {};
                this._renderListPass = currentPass;

                const itemsContainer = document.createElement("div");
                itemsContainer.className = "am-items";
                if (this.viewMode === "grid") {
                    itemsContainer.style.display = "flex";
                    itemsContainer.style.flexWrap = "wrap";
                    itemsContainer.style.justifyContent = "center";
                }
                listContainer.appendChild(itemsContainer);

                container.appendChild(listContainer);

                const CHUNK_SIZE = 15;
                let visibleIndex = 0;

                const renderNextChunk = () => {
                    if (this._renderListPass !== currentPass) return;

                    const fragment = document.createDocumentFragment();
                    let count = 0;

                    while (visibleIndex < visibleItems.length && count < CHUNK_SIZE) {
                        const item = visibleItems[visibleIndex];
                        fragment.appendChild(renderItem(item, item.selected, item.selectedIndex));
                        visibleIndex++;
                        count++;
                    }

                    if (fragment.childNodes.length > 0) {
                        itemsContainer.appendChild(fragment);
                    }

                    if (visibleIndex < visibleItems.length) {
                        this._renderListToken = requestAnimationFrame(renderNextChunk);
                    }
                };

                this._renderListToken = requestAnimationFrame(renderNextChunk);
            };
        }
    }
});
