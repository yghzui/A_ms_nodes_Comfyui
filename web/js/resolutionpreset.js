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
                    await api.fetchApi("/a_my_nodes/resolution_presets", {
                        method: "POST",
                        headers: {
                            "Content-Type": "application/json",
                        },
                        body: JSON.stringify({ presets: obj || {} }),
                    });
                } catch (e) {
                    console.error("[ResolutionPresetNode] UI: 保存全局预设到服务器失败", e);
                }
            };

            const loadCustomPresetsFromServer = async () => {
                if (node._resolutionPresetLoadedFromServer) {
                    return;
                }
                node._resolutionPresetLoadedFromServer = true;
                try {
                    const resp = await api.fetchApi("/a_my_nodes/resolution_presets");
                    if (resp && resp.ok) {
                        const data = await resp.json();
                        if (data && typeof data === "object" && data.presets && typeof data.presets === "object") {
                            setCustomPresetsLocal(data.presets);
                        }
                    }
                } catch (e) {
                    console.error("[ResolutionPresetNode] UI: 从服务器加载全局预设失败", e);
                }
            };

            const ensureWidthHeightWidgets = () => {
                if (widthWidget && heightWidget) {
                    return;
                }
                widthWidget = node.addWidget("number", "宽", 0, () => {
                });
                heightWidget = node.addWidget("number", "高", 0, () => {
                });
                if (!widthWidget.options) {
                    widthWidget.options = {};
                }
                if (!heightWidget.options) {
                    heightWidget.options = {};
                }
                widthWidget.options.min = 16;
                widthWidget.options.step = 16;
                heightWidget.options.min = 16;
                heightWidget.options.step = 16;
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

                presetWidget.options = presetWidget.options || {};
                presetWidget.options.values = options;

                let selected = presetWidget.value;
                const chosenKey = Object.keys(custom).find(k => custom[k].choose);
                if (chosenKey) {
                    selected = chosenKey;
                } else {
                    if (!selected || !options.includes(selected)) {
                        selected = options[0];
                    }
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
                const setDefault = window.confirm("是否将此预设设为默认? 取消则不会改变默认设置。");
                if (setDefault) {
                    for (const k of Object.keys(custom)) {
                        custom[k].choose = false;
                    }
                }
                const choose = setDefault ? true : (custom[targetId] && !!custom[targetId].choose);
                custom[targetId] = { w, h, choose };
                setCustomPresetsLocal(custom);
                await saveCustomPresetsToServer(custom);
                presetWidget.value = targetId;
                if (presetWidget.element) {
                    presetWidget.element.value = targetId;
                }
                syncPresetOptions();
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

            node.addWidget("button", "使用当前预设宽高", null, () => {
                applyCurrentPresetToWidthHeight();
            }, { serialize: false });

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
