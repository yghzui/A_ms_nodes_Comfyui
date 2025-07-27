// 在脚本顶部添加日志，以便在浏览器控制台中确认脚本是否被加载
console.log("Loading custom node: A_my_nodes/web/js/mask_add_dynamic.js");

import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "A_my_nodes.MaskAdd.DynamicInputs",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        // 只对 MaskAdd 节点进行操作
        if (nodeData.name === "MaskAdd") {
            console.log(`Patching node: ${nodeData.name} for dynamic inputs`);

            // 保存原始的 onNodeCreated 方法
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                // 调用原始方法
                if (onNodeCreated) {
                    onNodeCreated.apply(this, arguments);
                }
                
                // 节点创建后，设置一个标志，表示这是一个新节点
                this.isNewNode = true;
            };
            
            // 保存原始的 onConfigure 方法
            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function(info) {
                // 在应用配置前，先记录这是一个已有节点
                this.isNewNode = false;
                
                // 调用原始方法
                if (onConfigure) {
                    onConfigure.apply(this, arguments);
                }
                
                // 在配置应用后，延迟执行清理
                setTimeout(() => {
                    this.cleanupMaskInputs();
                }, 100);
            };
            
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
                    } else {
                        // 当断开连接时，自动触发清理功能
                        // 延迟执行，确保断开连接的操作已完成
                        setTimeout(() => {
                            this.cleanupMaskInputs();
                        }, 10);
                    }
                }
            };
            
            // 添加清理 Mask 输入的方法
            nodeType.prototype.cleanupMaskInputs = function() {
                // 获取所有 mask 输入
                const maskInputs = this.inputs.filter(input => input.name.startsWith("mask_"));
                
                // 找出最后一个已连接的输入索引
                let lastConnectedIndex = -1;
                for (let i = 0; i < maskInputs.length; i++) {
                    const inputIndex = this.inputs.indexOf(maskInputs[i]);
                    const linkId = this.getInputLink(inputIndex);
                    if (linkId !== null) {
                        lastConnectedIndex = i;
                    }
                }
                
                // 如果没有连接，保留第一个输入
                if (lastConnectedIndex === -1) {
                    lastConnectedIndex = 0;
                }
                
                // 要保留的输入数量（最后一个已连接的输入 + 1个空白输入）
                const keepCount = lastConnectedIndex + 2;
                
                // 移除多余的输入（从后向前，以避免索引变化问题）
                for (let i = maskInputs.length - 1; i >= keepCount; i--) {
                    const inputIndex = this.inputs.indexOf(maskInputs[i]);
                    this.removeInput(inputIndex);
                }
                
                // 确保有一个空白输入在最后
                if (maskInputs.length < keepCount) {
                    const nextIndex = maskInputs.length + 1;
                    const nextInputName = `mask_${nextIndex}`;
                    this.addInput(nextInputName, "MASK");
                }
                
                // 更新节点的大小
                this.setSize([this.size[0], this.computeSize()[1]]);
            };
            
            // 扩展 getExtraMenuOptions 方法，添加清除功能的选项
            const getExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
            nodeType.prototype.getExtraMenuOptions = function(_, options) {
                if (getExtraMenuOptions) {
                    getExtraMenuOptions.apply(this, arguments);
                }
                
                options.push({
                    content: "清理 Mask 输入",
                    callback: () => {
                        this.cleanupMaskInputs();
                    }
                });
            };
        }
    }
}); 