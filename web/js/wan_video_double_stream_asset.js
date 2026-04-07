import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

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
                container.style.backgroundColor = "var(--bg-color, #222)";
                container.style.border = "1px solid var(--border-color, #444)";
                container.style.padding = "5px";
                container.style.color = "var(--fg-color, #fff)";
                container.style.fontFamily = "sans-serif";
                container.style.fontSize = "12px";

                this.renderContainer = container;

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
                // 重新渲染以显示命中效果
                this.renderList();
            };

            // 监听执行开始事件，清除状态
            api.addEventListener("execution_start", () => {
                if (this.hitStatus && Object.keys(this.hitStatus).length > 0) {
                    this.hitStatus = {};
                    this.renderList();
                }
            });

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
            };

            nodeType.prototype.renderList = function () {
                const container = this.renderContainer;
                if (!container) return;

                container.innerHTML = "";

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

                const btnSelectAll = document.createElement("button");
                btnSelectAll.textContent = "全选";
                btnSelectAll.style.fontSize = "10px";
                btnSelectAll.style.cursor = "pointer";
                btnSelectAll.onclick = () => {
                    this.assetsData.forEach(asset => {
                        if (this.currentGroupFilter === "All" || asset.groupName === this.currentGroupFilter) {
                            if (!this.selectedAssets.find(s => s.id === asset.id)) {
                                this.selectedAssets.push({
                                    id: asset.id,
                                    enable_mode: "Auto"
                                });
                            }
                        }
                    });
                    this.updateWidgetValue();
                    this.renderList();
                };

                const btnInvertSelect = document.createElement("button");
                btnInvertSelect.textContent = "反选";
                btnInvertSelect.style.fontSize = "10px";
                btnInvertSelect.style.cursor = "pointer";
                btnInvertSelect.onclick = () => {
                    const toAdd = [];
                    const toRemove = [];
                    
                    this.assetsData.forEach(asset => {
                        if (this.currentGroupFilter === "All" || asset.groupName === this.currentGroupFilter) {
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
                    this.renderList();
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
                
                toolbar.appendChild(btnGroup);

                // 右侧过滤器和视图切换
                const rightControls = document.createElement("div");
                rightControls.style.display = "flex";
                rightControls.style.gap = "5px";

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

                // 将数据分为两部分：已选中的（按选中顺序） 和 未选中的
                const selectedItems = [];
                const unselectedItems = [];

                // 遍历保存的选中列表，从 assetsData 中找出最新数据，并保持顺序和设置
                this.selectedAssets.forEach(sel => {
                    const found = this.assetsData.find(a => a.id === sel.id);
                    if (found) {
                        // 过滤器：如果当前分组不是 All，且当前条目不属于该分组，依然要显示已选中的条目吗？
                        // 通常已选中的条目最好一直显示，以保持排序可见，或者也可以过滤。这里选择不隐藏已选中项。
                        selectedItems.push({
                            ...found,
                            enable_mode: sel.enable_mode || "Auto", // 保留之前设置的模式
                            selected: true
                        });
                    }
                });

                // 找出未选中的，应用分组过滤
                this.assetsData.forEach(asset => {
                    if (!this.selectedAssets.find(sel => sel.id === asset.id)) {
                        if (this.currentGroupFilter === "All" || asset.groupName === this.currentGroupFilter) {
                            unselectedItems.push({
                                ...asset,
                                enable_mode: "Auto",
                                selected: false
                            });
                        }
                    }
                });

                const renderItem = (item, isSelected, index) => {
                    const div = document.createElement("div");
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
                    
                    // 命中状态视觉反馈
                    if (isSelected && this.hitStatus[item.id] === true) {
                        div.style.backgroundColor = "rgba(40, 167, 69, 0.3)"; // 绿色高亮
                        div.style.borderColor = "#28a745";
                    } else if (isSelected && this.hitStatus[item.id] === false) {
                        div.style.backgroundColor = "rgba(108, 117, 125, 0.2)"; // 灰色
                    } else if (isSelected) {
                        div.style.backgroundColor = "rgba(0, 123, 255, 0.1)"; // 蓝色（普通选中）
                        div.style.borderColor = "#007bff";
                    } else {
                        div.style.backgroundColor = "transparent";
                    }

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
                        editBtn.onclick = () => {
                            menu.remove();
                            if (window.AssetManager) {
                                window.AssetManager.showModal();
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
                                alert("资产管理器未加载");
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

                    // 拖拽手柄 (仅选中的支持排序)
                    if (isSelected) {
                        const dragHandle = document.createElement("span");
                        dragHandle.textContent = "☰";
                        dragHandle.style.cursor = "grab";
                        dragHandle.style.color = "#ccc";
                        dragHandle.draggable = true;
                        
                        if (this.viewMode === "grid") {
                            dragHandle.style.position = "absolute";
                            dragHandle.style.top = "2px";
                            dragHandle.style.right = "2px";
                            dragHandle.style.background = "rgba(0,0,0,0.5)";
                            dragHandle.style.padding = "2px";
                            dragHandle.style.borderRadius = "2px";
                        } else {
                            dragHandle.style.marginRight = "5px";
                        }

                        dragHandle.addEventListener("dragstart", (e) => {
                            e.dataTransfer.setData("text/plain", index.toString());
                            e.dataTransfer.effectAllowed = "move";
                            div.style.opacity = "0.5";
                        });

                        dragHandle.addEventListener("dragend", () => {
                            div.style.opacity = "1";
                        });

                        div.addEventListener("dragover", (e) => {
                            e.preventDefault();
                            e.dataTransfer.dropEffect = "move";
                            if (this.viewMode === "list") div.style.borderTop = "2px solid #007bff";
                            else div.style.border = "2px solid #007bff";
                        });

                        div.addEventListener("dragleave", () => {
                            if (this.viewMode === "list") div.style.borderTop = "1px solid transparent";
                            else div.style.border = "1px solid #007bff"; // 保持选中颜色
                        });

                        div.addEventListener("drop", (e) => {
                            e.preventDefault();
                            div.style.borderTop = "1px solid transparent";
                            const sourceIndex = parseInt(e.dataTransfer.getData("text/plain"), 10);
                            if (isNaN(sourceIndex) || sourceIndex === index) return;

                            // 重新排序
                            const movedItem = this.selectedAssets.splice(sourceIndex, 1)[0];
                            this.selectedAssets.splice(index, 0, movedItem);
                            this.updateWidgetValue();
                            this.renderList();
                        });

                        div.appendChild(dragHandle);
                    } else if (this.viewMode === "list") {
                        // 占位
                        const spacer = document.createElement("span");
                        spacer.style.width = "16px";
                        spacer.style.display = "inline-block";
                        div.appendChild(spacer);
                    }

                    // 预览图（如果有且是网格模式）
                    if (this.viewMode === "grid") {
                        let imgSrc = "";
                        if (item.preview_image) {
                            imgSrc = `/a_my_nodes/assets/view_preview?path=${encodeURIComponent(item.preview_image)}`;
                        } else {
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

                    // 悬浮预览图容器 (仅列表模式)
                    let hoverPreview = null;

                    div.addEventListener("mouseenter", (e) => {
                        if (this.viewMode === "list") {
                            let imgSrc = "";
                            if (item.preview_image) {
                                imgSrc = `/a_my_nodes/assets/view_preview?path=${encodeURIComponent(item.preview_image)}`;
                            } else {
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
                                hoverPreview = document.createElement("img");
                                hoverPreview.src = imgSrc;
                                hoverPreview.style.position = "fixed";
                                hoverPreview.style.maxWidth = "200px";
                                hoverPreview.style.maxHeight = "200px";
                                hoverPreview.style.objectFit = "cover";
                                hoverPreview.style.border = "2px solid #555";
                                hoverPreview.style.borderRadius = "4px";
                                hoverPreview.style.zIndex = "10001";
                                hoverPreview.style.pointerEvents = "none";
                                hoverPreview.style.boxShadow = "0 4px 8px rgba(0,0,0,0.5)";
                                hoverPreview.style.background = "#111";
                                
                                // 根据鼠标位置调整预览图位置
                                const x = e.clientX + 15;
                                const y = e.clientY + 15;
                                hoverPreview.style.left = x + "px";
                                hoverPreview.style.top = y + "px";
                                
                                document.body.appendChild(hoverPreview);
                            }
                        }
                    });

                    div.addEventListener("mousemove", (e) => {
                        if (hoverPreview) {
                            hoverPreview.style.left = (e.clientX + 15) + "px";
                            hoverPreview.style.top = (e.clientY + 15) + "px";
                        }
                    });

                    div.addEventListener("mouseleave", () => {
                        if (hoverPreview) {
                            hoverPreview.remove();
                            hoverPreview = null;
                        }
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
                    checkbox.checked = isSelected;
                    checkbox.style.marginRight = "4px";
                    if (this.viewMode === "grid") {
                        checkbox.style.position = "absolute";
                        checkbox.style.top = "2px";
                        checkbox.style.left = "2px";
                        checkbox.style.zIndex = "10";
                    }
                    checkbox.addEventListener("change", (e) => {
                        if (e.target.checked) {
                            this.selectedAssets.push({
                                id: item.id,
                                enable_mode: "Auto" // 仅保存必须的状态：id和enable_mode
                            });
                        } else {
                            this.selectedAssets = this.selectedAssets.filter(sel => sel.id !== item.id);
                        }
                        this.updateWidgetValue();
                        this.renderList();
                    });
                    
                    if (this.viewMode === "list") controlRow.appendChild(checkbox);
                    else div.appendChild(checkbox); // 网格模式下复选框绝对定位

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

                    // 启动模式下拉框 (仅选中显示)
                    if (isSelected) {
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
                        
                        select.value = item.enable_mode;
                        
                        select.addEventListener("change", (e) => {
                            const selItem = this.selectedAssets.find(s => s.id === item.id);
                            if (selItem) {
                                selItem.enable_mode = e.target.value;
                                this.updateWidgetValue();
                            }
                        });
                        
                        if (this.viewMode === "grid") div.appendChild(select);
                        else div.appendChild(select);
                    }

                    return div;
                };

                // --- 渲染列表容器 ---
                const listContainer = document.createElement("div");
                listContainer.style.flex = "1";
                listContainer.style.overflowY = "auto";
                
                if (this._renderListToken) {
                    cancelAnimationFrame(this._renderListToken);
                }
                const currentPass = {};
                this._renderListPass = currentPass;

                let selItemsContainer = null;
                // 渲染已选中区域标题
                if (selectedItems.length > 0) {
                    const selTitle = document.createElement("div");
                    selTitle.textContent = `已选模型 (${selectedItems.length}) - 拖拽排序 / 右键编辑`;
                    selTitle.style.fontWeight = "bold";
                    selTitle.style.marginBottom = "5px";
                    selTitle.style.borderBottom = "1px solid #555";
                    listContainer.appendChild(selTitle);

                    selItemsContainer = document.createElement("div");
                    if (this.viewMode === "grid") {
                        selItemsContainer.style.display = "flex";
                        selItemsContainer.style.flexWrap = "wrap";
                        selItemsContainer.style.justifyContent = "center";
                    }
                    listContainer.appendChild(selItemsContainer);
                }

                let unselItemsContainer = null;
                // 渲染未选中区域标题
                if (unselectedItems.length > 0) {
                    const unselTitle = document.createElement("div");
                    unselTitle.textContent = `未选模型 (${unselectedItems.length})`;
                    unselTitle.style.fontWeight = "bold";
                    unselTitle.style.marginTop = "10px";
                    unselTitle.style.marginBottom = "5px";
                    unselTitle.style.borderBottom = "1px solid #555";
                    unselTitle.style.color = "#aaa";
                    listContainer.appendChild(unselTitle);

                    unselItemsContainer = document.createElement("div");
                    if (this.viewMode === "grid") {
                        unselItemsContainer.style.display = "flex";
                        unselItemsContainer.style.flexWrap = "wrap";
                        unselItemsContainer.style.justifyContent = "center";
                    }
                    listContainer.appendChild(unselItemsContainer);
                }

                container.appendChild(listContainer);

                const CHUNK_SIZE = 15;
                let selIndex = 0;
                let unselIndex = 0;

                const renderNextChunk = () => {
                    if (this._renderListPass !== currentPass) return;

                    const fragmentSel = document.createDocumentFragment();
                    const fragmentUnsel = document.createDocumentFragment();
                    let count = 0;

                    while (selIndex < selectedItems.length && count < CHUNK_SIZE) {
                        fragmentSel.appendChild(renderItem(selectedItems[selIndex], true, selIndex));
                        selIndex++;
                        count++;
                    }

                    while (unselIndex < unselectedItems.length && count < CHUNK_SIZE) {
                        fragmentUnsel.appendChild(renderItem(unselectedItems[unselIndex], false, -1));
                        unselIndex++;
                        count++;
                    }

                    if (selItemsContainer && fragmentSel.childNodes.length > 0) {
                        selItemsContainer.appendChild(fragmentSel);
                    }
                    if (unselItemsContainer && fragmentUnsel.childNodes.length > 0) {
                        unselItemsContainer.appendChild(fragmentUnsel);
                    }

                    if (selIndex < selectedItems.length || unselIndex < unselectedItems.length) {
                        this._renderListToken = requestAnimationFrame(renderNextChunk);
                    }
                };

                this._renderListToken = requestAnimationFrame(renderNextChunk);
            };
        }
    }
});
