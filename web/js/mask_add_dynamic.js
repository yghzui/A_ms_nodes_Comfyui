// 在脚本顶部添加日志，以便在浏览器控制台中确认脚本是否被加载
console.log("Loading custom node: A_my_nodes/web/js/mask_add_dynamic.js");

import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "A_my_nodes.MaskAdd.DynamicInputs",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        // 只对 MaskAdd 节点进行操作
        if (nodeData.name === "MaskAdd") {
            console.log(`Patching node: ${nodeData.name} for dynamic inputs`);

            // 保存原始的 onConnectionsChange 方法
            const onConnectionsChange = nodeType.prototype.onConnectionsChange;
            
            // 重写 onConnectionsChange 方法
            nodeType.prototype.onConnectionsChange = function(connectionType, slotIndex, isConnected, linkInfo, inputInfo) {
                // 调用原始方法
                if (onConnectionsChange) {
                    onConnectionsChange.apply(this, arguments);
                }
                
                // 只处理输入连接
                if (connectionType !== 1) { // 1 表示输入连接
                    return;
                }
                
                // 获取输入名称
                const inputName = inputInfo ? inputInfo.name : null;
                
                // 检查是否是 mask 输入
                if (inputName && inputName.startsWith("mask_")) {
                    if (isConnected) {
                        // 当连接时，检查是否需要添加新的输入
                        const currentIndex = parseInt(inputName.split("_")[1]);
                        const nextIndex = currentIndex + 1;
                        
                        // 检查下一个输入是否已存在
                        const nextInputName = `mask_${nextIndex}`;
                        if (!this.inputs.find(input => input.name === nextInputName) && nextIndex <= 20) {
                            // 添加下一个输入
                            this.addInput(nextInputName, "MASK");
                            
                            // 更新节点的大小
                            this.setSize([this.size[0], this.computeSize()[1]]);
                        }
                    }
                }
            };
            
            // 扩展 getExtraMenuOptions 方法，添加清除所有连接的选项
            const getExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
            nodeType.prototype.getExtraMenuOptions = function(_, options) {
                if (getExtraMenuOptions) {
                    getExtraMenuOptions.apply(this, arguments);
                }
                
                options.push({
                    content: "清除未连接的 Mask 输入",
                    callback: () => {
                        // 获取所有 mask 输入
                        const maskInputs = this.inputs.filter(input => input.name.startsWith("mask_"));
                        
                        // 收集未连接的输入索引（从后向前）
                        const unconnectedInputs = [];
                        for (let i = maskInputs.length - 1; i > 0; i--) {
                            const inputIndex = this.inputs.indexOf(maskInputs[i]);
                            const linkId = this.getInputLink(inputIndex);
                            if (linkId === null) {
                                unconnectedInputs.push(inputIndex);
                            }
                        }
                        
                        // 移除未连接的输入（从后向前，以避免索引变化问题）
                        for (const inputIndex of unconnectedInputs) {
                            this.removeInput(inputIndex);
                        }
                        
                        // 更新节点的大小
                        this.setSize([this.size[0], this.computeSize()[1]]);
                    }
                });
            };
        }
    }
}); 