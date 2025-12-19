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
