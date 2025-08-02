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

            if (!useSecondsWidget || !secondsWidget || !fpsWidget || !lengthWidget) {
                console.error("[I2VConfigureNode] UI: 无法找到所有必要的控件。");
                return;
            }

            const updateLengthState = () => {
                const useSeconds = useSecondsWidget.value;
                const oldDisabled = lengthWidget.disabled;
                const oldLength = lengthWidget.value;

                lengthWidget.disabled = useSeconds;
                
                let newLength = oldLength;
                if (useSeconds) {
                    const seconds = secondsWidget.value;
                    const fps = fpsWidget.value;
                    newLength = Math.floor(seconds * fps + 1);
                    lengthWidget.value = newLength;
                    
                    if (lengthWidget.callback) {
                        lengthWidget.callback(newLength);
                    }
                }
                
                if (oldDisabled !== lengthWidget.disabled || oldLength !== newLength) {
                    node.setDirtyCanvas(true, false);
                }
            };
            
            [useSecondsWidget, secondsWidget, fpsWidget].forEach(widget => {
                const originalCallback = widget.callback;
                widget.callback = (value, ...args) => {
                    if(originalCallback) {
                       // 修复：使用 widget 作为 `this` 上下文来调用原始回调，以兼容其他扩展
                       originalCallback.apply(widget, [value, ...args]);
                    }
                    updateLengthState();
                };
            });
            
            setTimeout(() => updateLengthState(), 10);
        };
    },
}); 