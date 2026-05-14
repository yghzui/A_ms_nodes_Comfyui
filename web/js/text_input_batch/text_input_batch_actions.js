export function createTextInputBatchActionsApi({
    app,
    api,
    modal,
    showTopNotification,
    getItems,
    setItems,
    getBaseTitle,
    setIndexSelectorValue,
    getLayoutCells,
    getEnsureTextareas
}) {
    const layoutCells = (...args) => getLayoutCells()(...args);
    const ensureTextareas = (...args) => getEnsureTextareas()(...args);

function handleExport(node) {
    const items = getItems(node);
    modal.show({
        title: "导出提示词",
        content: "请选择导出方式：",
        buttons: [
            {
                text: "导出为 JSON 文件",
                type: "primary",
                onClick: () => {
                    const blob = new Blob([JSON.stringify(items, null, 2)], { type: "application/json" });
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement("a");
                    a.href = url;
                    a.download = "prompts_export.json";
                    a.click();
                    URL.revokeObjectURL(url);
                    modal.close();
                }
            },
            {
                text: "复制到剪贴板",
                type: "secondary",
                onClick: () => {
                    navigator.clipboard.writeText(JSON.stringify(items, null, 2))
                        .then(() => showTopNotification("已复制到剪贴板", "success"))
                        .catch(err => showTopNotification("复制失败: " + err, "error"));
                    modal.close();
                }
            },
            { text: "取消", onClick: () => modal.close() }
        ]
    });
}

function handleImport(node) {
    const content = `
        <div style="display:flex; flex-direction:column; gap:10px;">
            <label>粘贴内容 (JSON 或 文本):</label>
            <textarea id="import-text" class="custom-modal-textarea" placeholder="在此粘贴..."></textarea>
            <label>或 选择文件:</label>
            <input type="file" id="import-file" accept=".json,.txt" class="custom-modal-file-input">
        </div>
    `;

    modal.show({
        title: "导入提示词 (追加模式)",
        content: content,
        width: "500px",
        buttons: [
            {
                text: "确认导入",
                type: "primary",
                onClick: async () => {
                    const textEl = document.getElementById("import-text");
                    const fileEl = document.getElementById("import-file");
                    let rawData = textEl.value.trim();

                    if (fileEl.files.length > 0) {
                        const file = fileEl.files[0];
                        rawData = await file.text();
                    }

                    if (!rawData) {
                        showTopNotification("请输入内容或选择文件", "warning");
                        return;
                    }

                    processImport(node, rawData);
                    modal.close();
                }
            },
            { text: "取消", onClick: () => modal.close() }
        ]
    });
}

function processImport(node, rawData) {
    let newItems = [];
    try {
        // 尝试 JSON 解析
        const parsed = JSON.parse(rawData);
        if (Array.isArray(parsed)) {
            newItems = parsed.map((item, idx) => {
                const content = (typeof item === 'object' && item !== null && 'content' in item) ? item.content : 
                                (typeof item === 'string' ? item : JSON.stringify(item));
                const title = (typeof item === 'object' && item !== null && 'title' in item) ? item.title : `imported_${idx}`;
                const enabled = (typeof item === 'object' && item !== null && 'enabled' in item) ? item.enabled !== false : true;
                
                return {
                    title: String(title),
                    content: String(content),
                    enabled: enabled
                };
            });
        } else if (typeof parsed === 'object' && parsed !== null) {
             // 单个对象
             newItems.push({
                title: String(parsed.title || "imported"),
                content: String(parsed.content || JSON.stringify(parsed)),
                enabled: parsed.enabled !== false
             });
        }
    } catch (e) {
        // 非 JSON，按行分割
        const lines = rawData.split(/\n/);
        newItems = lines.filter(line => line.trim()).map((line, idx) => ({
            title: `line_${idx}`,
            content: line.trim(),
            enabled: true
        }));
    }

    if (newItems.length > 0) {
        const currentItems = getItems(node);
        // 追加模式
        const merged = currentItems.concat(newItems);
        setItems(node, merged);
        
        // 刷新 UI
        const layout = layoutCells(node, merged);
        ensureTextareas(node, layout, merged);
        app.graph.setDirtyCanvas(true, true);
        showTopNotification(`成功导入 ${newItems.length} 条提示词`, "success");
    } else {
        showTopNotification("未识别到有效的提示词数据，请检查格式。支持 JSON 数组或每行一条文本。", "error");
    }
}

function handleBatchDelete(node) {
    const items = getItems(node);
    const toDeleteCount = items.filter(i => i.enabled).length;
    
    if (toDeleteCount === 0) {
        showTopNotification("没有选中的提示词 (Enabled = true)", "warning");
        return;
    }

    modal.show({
        title: "批量删除确认",
        content: `确定要删除 ${toDeleteCount} 个选中的提示词吗？此操作不可撤销。`,
        buttons: [
            {
                text: "确认删除",
                type: "danger",
                onClick: () => {
                    // 保留未选中的 (即 enabled == false 的)
                    let remaining = items.filter(i => !i.enabled);
                    
                    // 至少保留一项
                    if (remaining.length === 0) {
                        remaining.push({ title: "prompt_0", content: "", enabled: true });
                    }
                    
                    setItems(node, remaining);
                    setIndexSelectorValue(node, 0);
                    
                    const layout = layoutCells(node, remaining);
                    ensureTextareas(node, layout, remaining);
                    app.graph.setDirtyCanvas(true, true);
                    
                    modal.close();
                }
            },
            { text: "取消", onClick: () => modal.close() }
        ]
    });
}

async function fetchAssetManagerCollection(endpoint) {
    const res = await api.fetchApi(endpoint);
    if (!res.ok) {
        throw new Error(`请求失败: ${res.status}`);
    }
    const data = await res.json();
    if (!data || typeof data !== "object") {
        return { groups: [] };
    }
    if (!Array.isArray(data.groups)) {
        data.groups = [];
    }
    return data;
}

async function saveAssetManagerCollection(endpoint, data) {
    const res = await api.fetchApi(endpoint, {
        method: "POST",
        body: JSON.stringify(data),
        headers: { "Content-Type": "application/json" }
    });
    if (!res.ok) {
        throw new Error(`保存失败: ${res.status}`);
    }
    const result = await res.json().catch(() => ({ success: true }));
    if (result && result.success === false) {
        throw new Error(result.error || "保存失败");
    }
    return result;
}

function buildLocalIncludesMatches(texts, keyword) {
    const normalized = String(keyword || "").trim().toLowerCase();
    if (!normalized) return texts.map(() => true);
    return texts.map(t => String(t || "").toLowerCase().includes(normalized));
}

async function fetchPinyinMatches(texts, keyword) {
    const normalized = String(keyword || "").trim().toLowerCase();
    if (!normalized) return texts.map(() => true);

    try {
        const res = await api.fetchApi("/a_my_nodes/assets/search_pinyin", {
            method: "POST",
            body: JSON.stringify({ texts, keyword: normalized }),
            headers: { "Content-Type": "application/json" }
        });
        if (!res.ok) {
            throw new Error(`Search API failed: ${res.status}`);
        }
        const data = await res.json();
        if (data && Array.isArray(data.matches) && data.matches.length === texts.length) {
            return data.matches.map(Boolean);
        }
    } catch (e) {
        console.warn("[TextInputBatch] search_pinyin failed, fallback to local search:", e);
    }

    return buildLocalIncludesMatches(texts, normalized);
}

function buildNewPromptAssetItem(item, index) {
    const baseTitle = getBaseTitle(item?.title || `prompt_${index}`).trim();
    return {
        id: `${Date.now()}_${Math.random().toString(16).slice(2, 8)}`,
        title: baseTitle || `prompt_${index}`,
        content: String(item?.content || ""),
        preview_image: ""
    };
}

function escapeHtml(value) {
    return String(value ?? "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#39;");
}

function ensureModelsGroupAlignment(modelsData, targetIndex, targetGroupName) {
    if (!Array.isArray(modelsData.groups)) {
        modelsData.groups = [];
    }
    while (modelsData.groups.length < targetIndex) {
        modelsData.groups.push({
            name: `未命名分组 ${modelsData.groups.length + 1}`,
            items: []
        });
    }
    if (!modelsData.groups[targetIndex]) {
        modelsData.groups[targetIndex] = {
            name: targetGroupName,
            items: []
        };
    } else if (!modelsData.groups[targetIndex].name) {
        modelsData.groups[targetIndex].name = targetGroupName;
    }
}

function getSearchRowDom({ placeholder = "🔍 搜索拼音/原名/首字母...", onChange }) {
    const wrap = document.createElement("div");
    wrap.style.position = "relative";
    wrap.style.display = "flex";
    wrap.style.alignItems = "center";

    const input = document.createElement("input");
    input.type = "text";
    input.placeholder = placeholder;
    input.style.width = "100%";
    input.style.boxSizing = "border-box";
    input.style.padding = "8px 26px 8px 10px";
    input.style.background = "#1a1a1a";
    input.style.border = "1px solid #444";
    input.style.color = "#eee";
    input.style.borderRadius = "4px";
    input.style.outline = "none";

    const clearBtn = document.createElement("span");
    clearBtn.textContent = "✖";
    clearBtn.style.position = "absolute";
    clearBtn.style.right = "8px";
    clearBtn.style.cursor = "pointer";
    clearBtn.style.color = "#aaa";
    clearBtn.style.fontSize = "12px";
    clearBtn.style.display = "none";
    clearBtn.onmouseenter = () => { clearBtn.style.color = "#fff"; };
    clearBtn.onmouseleave = () => { clearBtn.style.color = "#aaa"; };

    const setClearVisible = (visible) => {
        clearBtn.style.display = visible ? "block" : "none";
    };

    let composing = false;
    input.addEventListener("compositionstart", () => { composing = true; });
    input.addEventListener("compositionend", () => {
        composing = false;
        if (onChange) onChange(input.value);
    });

    input.addEventListener("input", () => {
        setClearVisible(!!input.value);
        if (composing) return;
        if (onChange) onChange(input.value);
    });

    clearBtn.addEventListener("click", (e) => {
        e.preventDefault();
        e.stopPropagation();
        input.value = "";
        setClearVisible(false);
        if (onChange) onChange("");
        input.focus();
    });

    wrap.appendChild(input);
    wrap.appendChild(clearBtn);
    return { wrap, input, clearBtn };
}

async function openAddToAssetManagerGroupPicker(node, index) {
    const items = getItems(node);
    const item = items[index];
    if (!item) {
        showTopNotification("未找到要添加的提示词", "error");
        return;
    }

    let promptsData;
    let modelsData;
    try {
        [promptsData, modelsData] = await Promise.all([
            fetchAssetManagerCollection("/a_my_nodes/assets/prompts"),
            fetchAssetManagerCollection("/a_my_nodes/assets/models")
        ]);
    } catch (e) {
        showTopNotification(`读取管理器数据失败: ${e.message || e}`, "error");
        return;
    }

    const promptItem = buildNewPromptAssetItem(item, index);

    const root = document.createElement("div");
    root.style.display = "flex";
    root.style.flexDirection = "column";
    root.style.gap = "10px";

    const tip = document.createElement("div");
    tip.style.color = "#bbb";
    tip.style.fontSize = "12px";
    tip.textContent = `选择要添加到的目录（提示词标题将使用: ${promptItem.title}）`;
    root.appendChild(tip);

    const listContainer = document.createElement("div");
    listContainer.style.border = "1px solid #333";
    listContainer.style.borderRadius = "6px";
    listContainer.style.overflow = "hidden";
    listContainer.style.maxHeight = "360px";
    listContainer.style.overflowY = "auto";

    const groupNames = promptsData.groups.map(g => String(g?.name || "").trim());
    const groupTexts = groupNames.map(n => n || "");

    let debounceTimer = null;
    let latestSeq = 0;
    let lastKeyword = "";

    const createRow = (label, onClick, opts = {}) => {
        const row = document.createElement("div");
        row.style.padding = "10px 12px";
        row.style.cursor = "pointer";
        row.style.borderBottom = "1px solid #2a2a2a";
        row.style.display = "flex";
        row.style.alignItems = "center";
        row.style.justifyContent = "space-between";
        row.style.background = opts.background || "transparent";
        row.style.color = opts.color || "#eee";
        row.onmouseenter = () => { row.style.background = "#333"; };
        row.onmouseleave = () => { row.style.background = opts.background || "transparent"; };

        const left = document.createElement("span");
        left.textContent = label;
        left.style.overflow = "hidden";
        left.style.textOverflow = "ellipsis";
        left.style.whiteSpace = "nowrap";
        row.appendChild(left);

        if (opts.suffix) {
            const right = document.createElement("span");
            right.textContent = opts.suffix;
            right.style.color = "#888";
            right.style.fontSize = "12px";
            row.appendChild(right);
        }

        row.addEventListener("click", (e) => {
            e.preventDefault();
            e.stopPropagation();
            if (onClick) onClick();
        });
        return row;
    };

    const tryAddToGroup = async (targetGroupName) => {
        const cleanName = String(targetGroupName || "").trim();
        if (!cleanName) {
            showTopNotification("请先选择目录或输入目录名", "warning");
            return;
        }

        let targetGroupIndex = promptsData.groups.findIndex(group => String(group?.name || "").trim() === cleanName);
        let shouldSaveModels = false;

        if (targetGroupIndex === -1) {
            promptsData.groups.push({ name: cleanName, items: [] });
            targetGroupIndex = promptsData.groups.length - 1;
            ensureModelsGroupAlignment(modelsData, targetGroupIndex, cleanName);
            shouldSaveModels = true;
        }

        const targetGroup = promptsData.groups[targetGroupIndex];
        if (!Array.isArray(targetGroup.items)) targetGroup.items = [];

        const duplicateExists = targetGroup.items.some(existingItem =>
            String(existingItem?.title || "").trim() === promptItem.title &&
            String(existingItem?.content || "") === promptItem.content
        );
        if (duplicateExists) {
            showTopNotification(`目录 [${cleanName}] 中已存在相同提示词`, "warning");
            return;
        }

        targetGroup.items.push({ ...promptItem });
        try {
            await saveAssetManagerCollection("/a_my_nodes/assets/prompts", promptsData);
            if (shouldSaveModels) {
                await saveAssetManagerCollection("/a_my_nodes/assets/models", modelsData);
            }
            modal.close();
            showTopNotification(`已添加到管理器目录: ${cleanName}`, "success");
        } catch (e) {
            showTopNotification(`保存到管理器失败: ${e.message || e}`, "error");
        }
    };

    const promptCreateGroup = async () => {
        const currentKeyword = String(searchInput?.value || "").trim();
        const groupName = prompt("请输入新分组名:", currentKeyword);
        if (groupName == null) return;
        await tryAddToGroup(groupName);
    };

    const renderGroups = (matches) => {
        listContainer.innerHTML = "";

        const keyword = String(lastKeyword || "").trim();
        const keywordHasValue = !!keyword;
        const hasExact = keywordHasValue && groupNames.some(n => String(n || "").trim() === keyword);

        if (keywordHasValue && !hasExact) {
            listContainer.appendChild(createRow(`➕ 新建分组: ${keyword}`, () => tryAddToGroup(keyword), {
                background: "rgba(42, 109, 181, 0.15)",
                color: "#e6edf3",
                suffix: "Enter"
            }));
        } else {
            listContainer.appendChild(createRow("➕ 新建分组...", () => promptCreateGroup(), {
                background: "rgba(42, 109, 181, 0.08)",
                color: "#e6edf3"
            }));
        }

        let shown = 0;
        for (let i = 0; i < promptsData.groups.length; i++) {
            const name = String(promptsData.groups[i]?.name || "").trim();
            if (!name) continue;
            if (matches && matches[i] === false) continue;
            listContainer.appendChild(createRow(name, () => tryAddToGroup(name), { suffix: `#${i + 1}` }));
            shown += 1;
        }

        if (shown === 0 && !(keywordHasValue && !hasExact)) {
            const empty = document.createElement("div");
            empty.style.padding = "12px";
            empty.style.color = "#888";
            empty.style.textAlign = "center";
            empty.textContent = "无匹配目录";
            listContainer.appendChild(empty);
        }
    };

    const scheduleSearch = (keyword, delay = 180) => {
        lastKeyword = keyword;
        if (debounceTimer) clearTimeout(debounceTimer);
        const seq = ++latestSeq;
        debounceTimer = setTimeout(async () => {
            debounceTimer = null;
            const matches = await fetchPinyinMatches(groupTexts, keyword);
            if (seq !== latestSeq) return; // 丢弃过期请求
            renderGroups(matches);
        }, delay);
    };

    const { wrap: searchWrap, input: searchInput } = getSearchRowDom({
        placeholder: "🔍 搜索目录(拼音/原名/首字母)...",
        onChange: (v) => scheduleSearch(v)
    });
    root.appendChild(searchWrap);
    root.appendChild(listContainer);

    // Enter: 如果输入框里有值，直接走“新建分组: xxx”
    searchInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter") {
            e.preventDefault();
            const keyword = String(searchInput.value || "").trim();
            if (keyword) tryAddToGroup(keyword);
        }
    });

    // 初始渲染
    renderGroups(null);

    modal.show({
        title: "选择管理器目录",
        content: root,
        width: "520px",
        buttons: [{ text: "关闭", onClick: () => modal.close() }]
    });
}

    return {
        handleExport,
        handleImport,
        processImport,
        handleBatchDelete,
        fetchPinyinMatches,
        openAddToAssetManagerGroupPicker
    };
}
