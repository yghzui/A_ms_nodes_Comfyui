import { $el } from "../../../scripts/ui.js";

export class DataHandler {
    static importData(manager) {
        const data = manager.currentTab === 'prompts' ? manager.promptsData : manager.modelsData;
        if (!data.groups || data.groups.length === 0) {
            alert("请先创建一个分组！");
            return;
        }

        const overlay = $el("div", {
            style: { position: "fixed", top: 0, left: 0, right: 0, bottom: 0, background: "rgba(0,0,0,0.8)", zIndex: 11000, display: "flex", justifyContent: "center", alignItems: "center" }
        });

        const processImportContent = (jsonStr) => {
            try {
                const parsed = JSON.parse(jsonStr);
                const currentItems = data.groups[manager.currentGroupIndex].items;
                let addCount = 0;
                let skipCount = 0;

                const resolveConflict = (newItem, isPrompt) => {
                    let titleKey = isPrompt ? 'title' : 'keyword';
                    let finalTitle = newItem[titleKey];
                    let isExactMatch = false;

                    if (isPrompt) {
                        const exactMatch = currentItems.find(i => i.title === newItem.title && i.content === newItem.content);
                        if (exactMatch) isExactMatch = true;
                    } else {
                        const currentJSON = JSON.stringify(newItem);
                        const exactMatch = currentItems.find(i => {
                            if (i.keyword !== newItem.keyword) return false;
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

                if (manager.currentTab === 'prompts') {
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
                    if (!Array.isArray(parsed) && (parsed.high || parsed.low || parsed.key_to_check)) {
                        resolveConflict({
                            keyword: parsed.key_to_check || parsed.keyword || "未命名配置",
                            check_mode: parsed.check_mode || "contains",
                            high_loras: parsed.high?.loras || [],
                            low_loras: parsed.low?.loras || [],
                            preview_image: ""
                        }, false);
                    } else if (Array.isArray(parsed)) {
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

                manager.saveData();
                manager.renderItems();
                alert(`导入完成！\n新增: ${addCount} 条\n跳过(完全重复): ${skipCount} 条`);
                document.body.removeChild(overlay);
            } catch (err) {
                alert("JSON 解析失败，请检查格式是否正确。\n" + err.message);
            }
        };

        const dialog = $el("div", {
            style: { background: "var(--am-panel-bg)", padding: "20px", borderRadius: "8px", border: "1px solid var(--am-border)", color: "white", width: "500px" }
        }, [
            $el("h3", { textContent: `导入到 [${data.groups[manager.currentGroupIndex].name}]`, style: { marginTop: 0 } }),
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

    static exportData(manager) {
        const data = manager.currentTab === 'prompts' ? manager.promptsData : manager.modelsData;
        const group = data.groups[manager.currentGroupIndex];
        if (!group || !group.items || group.items.length === 0) {
            alert("当前分组为空，没有可导出的数据！");
            return;
        }

        let exportObj;
        if (manager.currentTab === 'prompts') {
            exportObj = group.items.map(item => ({
                title: item.title,
                content: item.content,
                enabled: true
            }));
        } else {
            exportObj = group.items.map(item => ({
                lora: item.model_path,
                strength: item.strength,
                on: item.on !== false
            }));
        }

        const jsonStr = JSON.stringify(exportObj, null, 2);
        
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
                    a.download = `${manager.currentTab}_${group.name}.json`;
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
}
