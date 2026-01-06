import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";
console.log("正在为节点 resolutionpreset 应用UI逻辑");
app.registerExtension({
    name: "A_my_nodes.ResolutionPresetNode.UI",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "ResolutionPresetNode") {
            return;
        }

        const ensureResolutionPresetUI = (node) => {
            if (!node || !node.widgets) {
                return;
            }
            if (node._resolutionPresetInit) {
                return;
            }

            const presetWidget = node.widgets.find(w => w.name === "preset");
            const customPresetsWidget = node.widgets.find(w => w.name === "custom_presets");
            const mirrorWidget = node.widgets.find(w => w.name === "mirror");

            if (!presetWidget || !customPresetsWidget) {
                console.error("[ResolutionPresetNode] UI: 无法找到必要控件。");
                return;
            }

            customPresetsWidget.computeSize = () => [0, -4];

            const builtinPresets = [
                { id: "512x768", w: 512, h: 768 },
                { id: "1024x1440", w: 1024, h: 1440 },
                { id: "1280x1980", w: 1280, h: 1980 },
            ];

            let customPresetCache = null;
            let widthWidget = null;
            let heightWidget = null;
            let stepWidget = null; // 新增步长控件

            const parseCustomPresetsFromWidget = () => {
                let raw = customPresetsWidget.value;
                if (typeof raw !== "string") {
                    raw = "";
                }
                if (!raw) {
                    return {};
                }
                try {
                    const data = JSON.parse(raw);
                    if (!data || typeof data !== "object") {
                        return {};
                    }
                    const result = {};
                    for (const key of Object.keys(data)) {
                        const item = data[key] || {};
                        const w = parseInt(item.w);
                        const h = parseInt(item.h);
                        if (!Number.isFinite(w) || !Number.isFinite(h) || w <= 0 || h <= 0) {
                            continue;
                        }
                        const choose = !!item.choose;
                        result[key] = { w, h, choose };
                    }
                    return result;
                } catch (e) {
                    console.error("[ResolutionPresetNode] UI: 解析 custom_presets 失败", e);
                    return {};
                }
            };

            const getCustomPresets = () => {
                if (customPresetCache) {
                    return customPresetCache;
                }
                customPresetCache = parseCustomPresetsFromWidget();
                return customPresetCache;
            };

            const setCustomPresetsLocal = (obj) => {
                let data = obj || {};
                try {
                    const json = JSON.stringify(data);
                    customPresetCache = data;
                    customPresetsWidget.value = json;
                    if (customPresetsWidget.element) {
                        customPresetsWidget.element.value = json;
                    }
                } catch (e) {
                    customPresetCache = {};
                    customPresetsWidget.value = "";
                    if (customPresetsWidget.element) {
                        customPresetsWidget.element.value = "";
                    }
                    console.error("[ResolutionPresetNode] UI: 保存 custom_presets 失败", e);
                }
            };

            const saveCustomPresetsToServer = async (obj) => {
                try {
                    const resp = await api.fetchApi("/a_my_nodes/resolution_presets", {
                        method: "POST",
                        headers: {
                            "Content-Type": "application/json",
                        },
                        body: JSON.stringify({ presets: obj || {} }),
                    });
                    if (resp.status !== 200) {
                        try {
                            const errorData = await resp.json();
                            alert(`保存预设失败: ${resp.status} - ${errorData.error || '未知错误'}`);
                        } catch (e) {
                            alert(`保存预设失败: ${resp.status}`);
                        }
                        console.error("[ResolutionPresetNode] UI: 服务器返回错误", resp);
                    }
                } catch (e) {
                    console.error("[ResolutionPresetNode] UI: 保存全局预设到服务器失败", e);
                    alert("保存预设请求失败: " + e);
                }
            };

            const loadCustomPresetsFromServer = async () => {
                if (node._resolutionPresetLoadedFromServer) {
                    return;
                }
                node._resolutionPresetLoadedFromServer = true;
                try {
                    console.log("[ResolutionPresetNode] UI: 正在从服务器加载预设...");
                    const resp = await api.fetchApi("/a_my_nodes/resolution_presets");
                    if (resp && resp.ok) {
                        const data = await resp.json();
                        console.log("[ResolutionPresetNode] UI: 服务器返回预设数据", data);
                        if (data && typeof data === "object" && data.presets && typeof data.presets === "object") {
                            setCustomPresetsLocal(data.presets);
                            // 强制同步一次，确保UI更新
                            syncPresetOptions();
                        }
                    } else {
                        console.error("[ResolutionPresetNode] UI: 服务器响应异常", resp.status);
                    }
                } catch (e) {
                    console.error("[ResolutionPresetNode] UI: 从服务器加载全局预设失败", e);
                }
            };

            const ensureWidthHeightWidgets = () => {
                // 先尝试查找已存在的 widget
                if (!stepWidget && node.widgets) {
                    stepWidget = node.widgets.find(w => w.name === "步长");
                }
                if (!widthWidget && node.widgets) {
                    widthWidget = node.widgets.find(w => w.name === "宽");
                }
                if (!heightWidget && node.widgets) {
                    heightWidget = node.widgets.find(w => w.name === "高");
                }

                // 如果发现有 widget 但配置不正确（比如还是浮点数），则移除重建
                // 这一步非常关键，因为 ComfyUI 会缓存 widget 状态
                if (stepWidget && stepWidget.options && stepWidget.options.precision !== 0) {
                     node.widgets.splice(node.widgets.indexOf(stepWidget), 1);
                     stepWidget = null;
                }
                if (widthWidget && widthWidget.options && widthWidget.options.precision !== 0) {
                     node.widgets.splice(node.widgets.indexOf(widthWidget), 1);
                     widthWidget = null;
                }
                if (heightWidget && heightWidget.options && heightWidget.options.precision !== 0) {
                     node.widgets.splice(node.widgets.indexOf(heightWidget), 1);
                     heightWidget = null;
                }

                // 如果三个控件都存在且配置正确，直接返回
                if (widthWidget && heightWidget && stepWidget) {
                     // 再次确保 options 设置正确
                     if (widthWidget.options) widthWidget.options.precision = 0;
                     if (heightWidget.options) heightWidget.options.precision = 0;
                     if (stepWidget.options) stepWidget.options.precision = 0;
                     return;
                }
                
                // 添加步长控件
                if (!stepWidget) {
                    stepWidget = node.addWidget("number", "步长", 8, (v) => {
                         // 当步长改变时，更新宽高控件的 step 属性，并立即对齐当前宽高
                         const step = Math.max(1, parseInt(v, 10) || 1);
                         if (widthWidget) {
                             widthWidget.options.step = step * 10;
                             // 立即对齐宽度
                             const w = parseInt(widthWidget.value, 10);
                             if (w % step !== 0) {
                                 widthWidget.value = Math.round(w / step) * step;
                             }
                         }
                         if (heightWidget) {
                             heightWidget.options.step = step * 10;
                             // 立即对齐高度
                             const h = parseInt(heightWidget.value, 10);
                             if (h % step !== 0) {
                                 heightWidget.value = Math.round(h / step) * step;
                             }
                         }
                    });
                    // 显式设置 stepWidget 的选项，确保显示为整数
                    stepWidget.options = stepWidget.options || {};
                    stepWidget.options.min = 1;
                    stepWidget.options.max = 128;
                    stepWidget.options.step = 1;
                    stepWidget.options.precision = 0; // 强制整数
                }

                if (!widthWidget) {
                    widthWidget = node.addWidget("number", "宽", 0, (v) => {
                         // 宽高改变时自动对齐步长
                         const step = stepWidget ? (parseInt(stepWidget.value, 10) || 1) : 8;
                         const val = parseInt(v, 10);
                         if (val % step !== 0) {
                             widthWidget.value = Math.round(val / step) * step;
                         }
                    });
                }
                
                if (!heightWidget) {
                    heightWidget = node.addWidget("number", "高", 0, (v) => {
                         // 宽高改变时自动对齐步长
                         const step = stepWidget ? (parseInt(stepWidget.value, 10) || 1) : 8;
                         const val = parseInt(v, 10);
                         if (val % step !== 0) {
                             heightWidget.value = Math.round(val / step) * step;
                         }
                    });
                }

                if (!widthWidget.options) {
                    widthWidget.options = {};
                }
                if (!heightWidget.options) {
                    heightWidget.options = {};
                }
                
                const currentStep = stepWidget ? (parseInt(stepWidget.value, 10) || 1) : 8;
                widthWidget.options.min = currentStep;
                widthWidget.options.step = currentStep * 10;
                widthWidget.options.precision = 0; // 强制整数
                
                heightWidget.options.min = currentStep;
                heightWidget.options.step = currentStep * 10;
                heightWidget.options.precision = 0; // 强制整数
            };

            const buildOptions = (customMap) => {
                const options = builtinPresets.map(p => p.id);
                for (const key of Object.keys(customMap)) {
                    if (!options.includes(key)) {
                        options.push(key);
                    }
                }
                return options;
            };

            const syncPresetOptions = () => {
                const custom = getCustomPresets();
                const options = buildOptions(custom);
                console.log("[ResolutionPresetNode] UI: 同步预设选项", options);

                presetWidget.options = presetWidget.options || {};
                presetWidget.options.values = options;

                let selected = presetWidget.value;
                
                // 只有当当前选中的值无效（不在选项列表中）时，才尝试使用默认值
                if (!selected || !options.includes(selected)) {
                    const chosenKey = Object.keys(custom).find(k => custom[k].choose);
                    if (chosenKey && options.includes(chosenKey)) {
                        selected = chosenKey;
                    } else {
                        selected = options[0];
                    }
                    console.log(`[ResolutionPresetNode] UI: 当前值无效，重置为: ${selected}`);
                }

                if (selected !== presetWidget.value) {
                    presetWidget.value = selected;
                    if (presetWidget.element) {
                        presetWidget.element.value = selected;
                    }
                }

                ensureWidthHeightWidgets();
                const selectedKey = presetWidget.value;
                let widthValue = null;
                let heightValue = null;
                const builtin = builtinPresets.find(p => p.id === selectedKey);
                if (builtin) {
                    widthValue = builtin.w;
                    heightValue = builtin.h;
                } else if (custom[selectedKey]) {
                    widthValue = custom[selectedKey].w;
                    heightValue = custom[selectedKey].h;
                }
                if (widthValue !== null && heightValue !== null) {
                    let w = widthValue;
                    let h = heightValue;
                    if (mirrorWidget && mirrorWidget.value) {
                        const t = w;
                        w = h;
                        h = t;
                    }
                    widthWidget.value = w;
                    heightWidget.value = h;
                    if (widthWidget.element) {
                        widthWidget.element.value = w;
                    }
                    if (heightWidget.element) {
                        heightWidget.element.value = h;
                    }
                }

                node.setDirtyCanvas(true, false);
                if (node.onResize) {
                    node.onResize();
                }
            };

            const applyCurrentPresetToWidthHeight = () => {
                syncPresetOptions();
            };

            const saveCurrentWidthHeightAsPreset = async () => {
                ensureWidthHeightWidgets();
                const custom = getCustomPresets();
                const w = parseInt(widthWidget.value, 10);
                const h = parseInt(heightWidget.value, 10);
                if (!Number.isFinite(w) || !Number.isFinite(h) || w <= 0 || h <= 0) {
                    alert("宽高必须为正整数。");
                    return;
                }
                let selectedId = presetWidget.value;
                const isBuiltin = builtinPresets.some(p => p.id === selectedId);
                let targetId = selectedId;
                if (isBuiltin || !custom[selectedId]) {
                    let baseId = `${w}x${h}`;
                    targetId = baseId;
                    let suffix = 1;
                    while (custom[targetId] && (custom[targetId].w !== w || custom[targetId].h !== h)) {
                        targetId = `${baseId}_${suffix++}`;
                    }
                }
                const setDefault = false; // 移除弹窗，默认不改变默认设置
                // if (setDefault) {
                //     for (const k of Object.keys(custom)) {
                //         custom[k].choose = false;
                //     }
                // }
                const choose = setDefault ? true : (custom[targetId] && !!custom[targetId].choose);
                custom[targetId] = { w, h, choose };
                setCustomPresetsLocal(custom);
                await saveCustomPresetsToServer(custom);
                presetWidget.value = targetId;
                if (presetWidget.element) {
                    presetWidget.element.value = targetId;
                }
                syncPresetOptions();
                // alert(`预设 ${targetId} 保存成功！`); 
            };

            const deleteCurrentPreset = async () => {
                const custom = getCustomPresets();
                const selectedId = presetWidget.value;
                if (!selectedId) {
                    alert("当前没有选择预设。");
                    return;
                }
                const isBuiltin = builtinPresets.some(p => p.id === selectedId);
                if (isBuiltin) {
                    alert("内置预设不能删除。");
                    return;
                }
                if (!custom[selectedId]) {
                    alert("当前预设不是自定义预设。");
                    return;
                }
                if (!window.confirm(`确定删除预设 ${selectedId} 吗?`)) {
                    return;
                }
                delete custom[selectedId];
                setCustomPresetsLocal(custom);
                await saveCustomPresetsToServer(custom);
                syncPresetOptions();
            };

            node.addWidget("button", "保存/更新预设", null, () => {
                saveCurrentWidthHeightAsPreset();
            }, { serialize: false });

            node.addWidget("button", "删除当前预设", null, () => {
                deleteCurrentPreset();
            }, { serialize: false });

            if (presetWidget) {
                const originalPresetCallback = presetWidget.callback;
                presetWidget.callback = (value, ...args) => {
                    if (originalPresetCallback) {
                        originalPresetCallback.apply(presetWidget, [value, ...args]);
                    }
                    syncPresetOptions();
                };
            }

            if (mirrorWidget) {
                const originalMirrorCallback = mirrorWidget.callback;
                mirrorWidget.callback = (value, ...args) => {
                    if (originalMirrorCallback) {
                        originalMirrorCallback.apply(mirrorWidget, [value, ...args]);
                    }
                    syncPresetOptions();
                };
            }

            node._resolutionPresetSync = syncPresetOptions;
            node._resolutionPresetInit = true;

            loadCustomPresetsFromServer().then(() => {
                syncPresetOptions();
            });
        };

        const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            originalOnNodeCreated?.apply(this, arguments);
            ensureResolutionPresetUI(this);
        };

        const originalOnConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            originalOnConfigure?.apply(this, arguments);
            ensureResolutionPresetUI(this);
            if (this._resolutionPresetSync) {
                this._resolutionPresetSync();
            }
        };
    },
});
