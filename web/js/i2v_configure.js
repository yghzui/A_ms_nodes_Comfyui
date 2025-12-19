import { app } from "../../../scripts/app.js";

console.log("正在为节点 I2VConfigureNode 应用UI逻辑 (i2v_configure.js) - v5_compat_fix");

app.registerExtension({
    name: "A_my_nodes.I2VConfigureNode.UI",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "I2VConfigureNode") {
            return;
        }

        const originalOnConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            originalOnConfigure?.apply(this, arguments);
            const node = this;

            const useSecondsWidget = node.widgets.find(w => w.name === "use_seconds_for_length");
            const secondsWidget = node.widgets.find(w => w.name === "seconds");
            const fpsWidget = node.widgets.find(w => w.name === "fps");
            const lengthWidget = node.widgets.find(w => w.name === "length");
            const stepsWidget = node.widgets.find(w => w.name === "steps");
            const middleStepsWidget = node.widgets.find(w => w.name === "middle_steps");

            if (!useSecondsWidget || !secondsWidget || !fpsWidget || !lengthWidget || !stepsWidget || !middleStepsWidget) {
                console.error("[I2VConfigureNode] UI: 无法找到所有必要的控件。");
                return;
            }

            const updateLengthState = () => {
                const useSeconds = useSecondsWidget.value;
                const oldLength = lengthWidget.value;

                let newLength = oldLength;
                if (useSeconds) {
                    const seconds = secondsWidget.value;
                    const fps = fpsWidget.value;
                    // 根据公式计算新的帧数：秒数 * 帧率 + 1
                    newLength = Math.floor(seconds * fps + 1);
                    lengthWidget.value = newLength;
                    
                    // 使用只读属性而不是disabled，确保数值显示
                    if (lengthWidget.element) {
                        lengthWidget.element.readOnly = true;
                        lengthWidget.element.value = newLength;
                        lengthWidget.element.style.backgroundColor = "#f0f0f0"; // 设置背景色显示只读状态
                        lengthWidget.element.style.color = "#666"; // 设置文字颜色
                        lengthWidget.element.style.cursor = "not-allowed"; // 设置鼠标样式
                    }
                    
                    // 阻止用户输入事件
                    lengthWidget._originalCallback = lengthWidget.callback;
                    lengthWidget.callback = () => {
                        // 在禁用状态下，重新设置为计算值
                        lengthWidget.value = newLength;
                        if (lengthWidget.element) {
                            lengthWidget.element.value = newLength;
                        }
                    };
                } else {
                    // 启用状态下恢复正常
                    if (lengthWidget.element) {
                        lengthWidget.element.readOnly = false;
                        lengthWidget.element.style.backgroundColor = "";
                        lengthWidget.element.style.color = "";
                        lengthWidget.element.style.cursor = "";
                    }
                    
                    // 恢复原始回调函数
                    if (lengthWidget._originalCallback) {
                        lengthWidget.callback = lengthWidget._originalCallback;
                        delete lengthWidget._originalCallback;
                    }
                }
                
                // 强制更新UI显示状态
                if (oldLength !== newLength) {
                    // 标记画布需要重绘
                    node.setDirtyCanvas(true, false);
                    // 强制触发节点更新
                    if (node.onResize) {
                        node.onResize();
                    }
                }
            };

            const updateMiddleStepsState = () => {
                if (!stepsWidget || !middleStepsWidget) {
                    return;
                }
                const steps = parseInt(stepsWidget.value ?? 0) || 0;
                let middle = parseInt(middleStepsWidget.value ?? 0) || 0;

                if (steps <= 1) {
                    middle = 1;
                } else {
                    if (middle < 1) {
                        middle = 1;
                    }
                    if (middle >= steps) {
                        middle = Math.floor(steps / 2);
                    }
                }

                if (middleStepsWidget.element) {
                    if (typeof middleStepsWidget.element.min !== "undefined") {
                        middleStepsWidget.element.min = 1;
                    }
                    if (typeof middleStepsWidget.element.max !== "undefined") {
                        middleStepsWidget.element.max = steps > 1 ? steps - 1 : 1;
                    }
                }

                if (middle !== middleStepsWidget.value) {
                    middleStepsWidget.value = middle;
                    if (middleStepsWidget.element) {
                        middleStepsWidget.element.value = middle;
                    }
                    node.setDirtyCanvas(true, false);
                    if (node.onResize) {
                        node.onResize();
                    }
                }
            };
            
            [useSecondsWidget, secondsWidget, fpsWidget, stepsWidget, middleStepsWidget].forEach(widget => {
                const originalCallback = widget.callback;
                widget.callback = (value, ...args) => {
                    if(originalCallback) {
                       // 修复：使用 widget 作为 `this` 上下文来调用原始回调，以兼容其他扩展
                       originalCallback.apply(widget, [value, ...args]);
                    }
                    // 延迟执行状态更新，确保所有控件值都已更新
                    setTimeout(() => {
                        updateLengthState();
                        updateMiddleStepsState();
                    }, 1);
                };
            });
            
            // 初始化时设置正确的状态
             setTimeout(() => {
                 updateLengthState();
                 updateMiddleStepsState();
                 // 强制重绘节点以确保UI状态正确显示
         if (node.graph && node.graph.canvas) {
                     node.graph.canvas.setDirty(true, false);
                 }
             }, 10);
        };
    },
});

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

            const parseCustomPresets = () => {
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

            const saveCustomPresets = (obj) => {
                try {
                    const json = JSON.stringify(obj);
                    customPresetsWidget.value = json;
                    if (customPresetsWidget.element) {
                        customPresetsWidget.element.value = json;
                    }
                } catch (e) {
                    console.error("[ResolutionPresetNode] UI: 保存 custom_presets 失败", e);
                }
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
                const custom = parseCustomPresets();
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

                node.setDirtyCanvas(true, false);
                if (node.onResize) {
                    node.onResize();
                }
            };

            const openPresetManager = () => {
                const custom = parseCustomPresets();
                const keys = Object.keys(custom);
                let message = "当前自定义宽高预设：\n";
                if (keys.length === 0) {
                    message += "  (无)\n";
                } else {
                    keys.forEach((key, index) => {
                        const item = custom[key];
                        const flag = item.choose ? " *默认" : "";
                        message += `${index}: ${key} => ${item.w}x${item.h}${flag}\n`;
                    });
                }
                message += "\n输入操作：a=新增, e=编辑, d=删除, c=取消";
                const op = window.prompt(message, "a");
                if (!op) {
                    return;
                }
                const action = op.trim().toLowerCase();

                if (action === "a") {
                    const wStr = window.prompt("输入宽度", "512");
                    if (!wStr) {
                        return;
                    }
                    const hStr = window.prompt("输入高度", "768");
                    if (!hStr) {
                        return;
                    }
                    const w = parseInt(wStr, 10);
                    const h = parseInt(hStr, 10);
                    if (!Number.isFinite(w) || !Number.isFinite(h) || w <= 0 || h <= 0) {
                        alert("宽高必须为正整数。");
                        return;
                    }
                    let id = `${w}x${h}`;
                    let suffix = 1;
                    while (custom[id]) {
                        id = `${w}x${h}_${suffix++}`;
                    }
                    let choose = false;
                    if (window.confirm("是否将此预设设为默认?")) {
                        choose = true;
                        for (const k of Object.keys(custom)) {
                            custom[k].choose = false;
                        }
                    }
                    custom[id] = { w, h, choose };
                    saveCustomPresets(custom);
                    syncPresetOptions();
                } else if (action === "d") { 
                    if (keys.length === 0) {
                        alert("当前没有自定义预设可删除。");
                        return;
                    }
                    const indexStr = window.prompt("输入要删除的预设索引", "0");
                    if (!indexStr) {
                        return;
                    }
                    const idx = parseInt(indexStr, 10);
                    if (!Number.isFinite(idx) || idx < 0 || idx >= keys.length) {
                        alert("索引无效。");
                        return;
                    }
                    const key = keys[idx];
                    if (!window.confirm(`确定删除预设 ${key} 吗?`)) {
                        return;
                    }
                    delete custom[key];
                    saveCustomPresets(custom);
                    syncPresetOptions();
                } else if (action === "e") {
                    if (keys.length === 0) {
                        alert("当前没有自定义预设可编辑。");
                        return;
                    }
                    const indexStr = window.prompt("输入要编辑的预设索引", "0");
                    if (!indexStr) {
                        return;
                    }
                    const idx = parseInt(indexStr, 10);
                    if (!Number.isFinite(idx) || idx < 0 || idx >= keys.length) {
                        alert("索引无效。");
                        return;
                    }
                    const key = keys[idx];
                    const item = custom[key];
                    const wStr = window.prompt("编辑宽度", String(item.w));
                    if (!wStr) {
                        return;
                    }
                    const hStr = window.prompt("编辑高度", String(item.h));
                    if (!hStr) {
                        return;
                    }
                    const w = parseInt(wStr, 10);
                    const h = parseInt(hStr, 10);
                    if (!Number.isFinite(w) || !Number.isFinite(h) || w <= 0 || h <= 0) {
                        alert("宽高必须为正整数。");
                        return;
                    }
                    let newId = `${w}x${h}`;
                    let suffix = 1;
                    if (newId !== key) {
                        while (custom[newId] && newId !== key) {
                            newId = `${w}x${h}_${suffix++}`;
                        }
                    } else {
                        newId = key;
                    }
                    const setDefault = window.confirm("是否将此预设设为默认? 取消则保留原来默认设置。");
                    if (setDefault) {
                        for (const k of Object.keys(custom)) {
                            custom[k].choose = false;
                        }
                    }
                    const choose = setDefault ? true : !!item.choose;
                    if (newId !== key) {
                        delete custom[key];
                    }
                    custom[newId] = { w, h, choose };
                    saveCustomPresets(custom);
                    syncPresetOptions();
                }
            };

            node.addWidget("button", "管理自定义宽高", null, () => {
                openPresetManager();
            }, { serialize: false });

            node._resolutionPresetSync = syncPresetOptions;
            node._resolutionPresetInit = true;

            syncPresetOptions();
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
