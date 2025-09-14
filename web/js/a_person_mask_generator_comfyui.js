import { app } from "../../../scripts/app.js";

/**
 * A custom node that dynamically shows/hides output slots based on mask type selections.
 * When a mask type is enabled, its corresponding output slot becomes visible.
 * When disabled, the output slot is hidden and returns None.
 */
app.registerExtension({
    name: "A_my_nodes.APersonMaskGenerator",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // 检查节点名称是否为 "APersonMaskGenerator"
        if (nodeData.name === "APersonMaskGenerator") {

            // 包装 onExecuted 方法以处理输出分配
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
                    setTimeout(() => this.updateOutputs(), 10);
                }
            };

            // 包装 onNodeCreated 方法以初始化和设置动态输出
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                onNodeCreated?.apply(this, arguments);

                // 找到各个遮罩类型的小部件
                const faceWidget = this.widgets.find((w) => w.name === "face_mask");
                const backgroundWidget = this.widgets.find((w) => w.name === "background_mask");
                const hairWidget = this.widgets.find((w) => w.name === "hair_mask");
                const bodyWidget = this.widgets.find((w) => w.name === "body_mask");
                const clothesWidget = this.widgets.find((w) => w.name === "clothes_mask");

                // 输出槽位映射：输出索引对应的遮罩类型
                const outputMapping = {
                    0: { name: "merged_mask", widget: null, alwaysVisible: true }, // 合并遮罩始终可见
                    1: { name: "face_mask", widget: faceWidget, alwaysVisible: false },
                    2: { name: "background_mask", widget: backgroundWidget, alwaysVisible: false },
                    3: { name: "hair_mask", widget: hairWidget, alwaysVisible: false },
                    4: { name: "body_mask", widget: bodyWidget, alwaysVisible: false },
                    5: { name: "clothes_mask", widget: clothesWidget, alwaysVisible: false }
                };

                // 定义一个函数来根据小部件状态更新输出槽的可见性
                this.updateOutputs = () => {
                    if (!this.outputs) return;

                    const activeOutputs = [0]; // 第一个输出始终是合并遮罩
                    const requiredOutputNames = ["合并遮罩"]; // 对应的输出名称

                    // 检查每个遮罩类型是否启用
                    for (let i = 1; i < 6; i++) {
                        const mapping = outputMapping[i];
                        if (mapping && mapping.widget && mapping.widget.value === true) {
                            activeOutputs.push(i);
                            requiredOutputNames.push(mapping.name);
                        }
                    }

                    // 获取当前输出数量
                    const currentOutputs = this.outputs ? this.outputs.length : 0;
                    const requiredOutputs = requiredOutputNames.length;

                    // 如果当前输出数量大于需要的数量，移除多余的输出
                    if (currentOutputs > requiredOutputs) {
                        for (let i = currentOutputs - 1; i >= requiredOutputs; i--) {
                            if (this.outputs[i] && this.outputs[i].links && this.outputs[i].links.length > 0) {
                                // 只有在真正需要移除时才断开连接
                                const linksToRemove = [...this.outputs[i].links];
                                linksToRemove.forEach(linkId => {
                                    app.graph.removeLink(linkId);
                                });
                            }
                            this.removeOutput(i);
                        }
                    }
                    // 如果当前输出数量小于需要的数量，添加缺少的输出
                    else if (currentOutputs < requiredOutputs) {
                        for (let i = currentOutputs; i < requiredOutputs; i++) {
                            this.addOutput(requiredOutputNames[i], "MASK");
                        }
                    }

                    // 更新现有输出的名称（确保名称正确）
                    for (let i = 0; i < Math.min(currentOutputs, requiredOutputs); i++) {
                        if (this.outputs[i] && this.outputs[i].name !== requiredOutputNames[i]) {
                            this.outputs[i].name = requiredOutputNames[i];
                        }
                    }

                    // 强制节点重绘和画布更新
                    this.setSize(this.computeSize());
                    if (this.graph && this.graph.canvas) {
                        this.graph.canvas.setDirty(true, false);
                    }
                    app.graph.setDirtyCanvas(true, true);
                };

                // 初始化时设置正确的状态
                setTimeout(() => {
                    this.updateOutputs();
                    // 强制重绘节点以确保UI状态正确显示
                    if (this.graph && this.graph.canvas) {
                        this.graph.canvas.setDirty(true, false);
                    }
                }, 10);

                // 为每个遮罩类型小部件添加回调函数
                const widgets = [faceWidget, backgroundWidget, hairWidget, bodyWidget, clothesWidget];
                widgets.forEach(widget => {
                    if (widget) {
                        const originalCallback = widget.callback;
                        widget.callback = (value, ...args) => {
                            // 调用原始回调函数，使用widget作为this上下文以兼容其他扩展
                            if (originalCallback) {
                                originalCallback.apply(widget, [value, ...args]);
                            }
                            // 延迟执行状态更新，确保所有控件值都已更新
                            setTimeout(() => {
                                this.updateOutputs();
                                // 强制重绘节点以确保UI状态正确显示
                                if (this.graph && this.graph.canvas) {
                                    this.graph.canvas.setDirty(true, false);
                                }
                            }, 1);
                        };
                    }
                });
            };

            // 重写 onExecuted 以处理来自后端的数据
            const originalOnExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function(data) {
                originalOnExecuted?.apply(this, arguments);
                
                // 确保输出接口与当前启用的遮罩类型匹配
                if (this.updateOutputs) {
                    setTimeout(() => this.updateOutputs(), 1);
                }
            };
        }
    },
});