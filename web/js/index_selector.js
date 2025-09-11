import { app } from "../../../scripts/app.js";

/**
 * A custom node that dynamically generates boolean outputs based on an input number 'n'.
 * It takes 'n' and 'index' as inputs. It will create 'n' output slots.
 * The output slot at 'index' will be true, and all others will be false.
 */
app.registerExtension({
    name: "A_my_nodes.IndexSelector",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // 检查节点名称是否为 "IndexSelector"
        if (nodeData.name === "IndexSelector") {

            // 包装 onExecuted 方法以处理输出来分配
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                onExecuted?.apply(this, arguments);
            };

            // 包装 onConfigure，以便在加载工作流时更新输出
            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function () {
                onConfigure?.apply(this, arguments);
                if (this.updateOutputs) {
                    // 使用 setTimeout 以确保在加载工作流时小部件值已更新
                    const nWidget = this.widgets.find((w) => w.name === "n");
                    if (nWidget) {
                        setTimeout(() => this.updateOutputs(nWidget.value), 10);
                    }
                }
            };

            // 包装 onNodeCreated 方法以初始化和设置动态输出
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                onNodeCreated?.apply(this, arguments);

                // 找到 'n' 和 'index' 的小部件
                const nWidget = this.widgets.find((w) => w.name === "n");
                const indexWidget = this.widgets.find((w) => w.name === "index");

                // 定义一个函数来根据 'n' 的值更新输出槽
                this.updateOutputs = (n) => {
                    // 确保 n 是一个有效的数字
                    if (n === undefined || n === null || isNaN(n)) {
                        return;
                    }

                    const currentOutputs = this.outputs ? this.outputs.length : 0;

                    // 如果 'n' 大于当前输出数量，则添加新的输出槽
                    if (n > currentOutputs) {
                        for (let i = currentOutputs; i < n; i++) {
                            this.addOutput(`output_${i}`, "BOOLEAN");
                        }
                    } 
                    // 如果 'n' 小于当前输出数量，则移除多余的输出槽
                    else if (n < currentOutputs) {
                        for (let i = currentOutputs - 1; i >= n; i--) {
                            // 移除输出前先断开所有连接
                            if (this.outputs[i] && this.outputs[i].links && this.outputs[i].links.length > 0) {
                                this.disconnectOutput(i);
                            }
                            this.removeOutput(i);
                        }
                    }

                    // 动态调整 'index' 小部件的最大值以防止越界
                    if (indexWidget) {
                        indexWidget.options.max = n > 0 ? n - 1 : 0;
                        // 确保当前 'index' 值不会超过新的最大值
                        indexWidget.value = Math.min(indexWidget.value, indexWidget.options.max);
                    }

                    // 强制节点重绘
                    this.setSize(this.computeSize());
                    app.graph.setDirtyCanvas(true, true);
                };

                // 节点创建时立即执行一次以设置初始状态
                if (nWidget) {
                    this.updateOutputs(nWidget.value);
                }

                // 包装 'n' 小部件的回调函数，以便在值更改时更新输出
                if (nWidget) {
                    const originalCallback = nWidget.callback;
                    nWidget.callback = (value) => {
                        // 使用 .call() 来确保原始回调函数中的 'this' 指向小部件本身
                        originalCallback?.call(nWidget, value);
                        this.updateOutputs(value);
                    };
                }
            };
        }
    },
});