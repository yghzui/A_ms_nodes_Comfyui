import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "A_my_nodes.AnyBatchAccumulator.DynamicInputs",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "AnyBatchAccumulator" && nodeData.name !== "AnyBatchListConverter") {
            return;
        }
        const isConverter = nodeData.name === "AnyBatchListConverter";

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function() {
            if (onNodeCreated) {
                onNodeCreated.apply(this, arguments);
            }
            this.isNewNode = true;
            this.sortInputs();
            this.updateOutputs();
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function(info) {
            this.isNewNode = false;
            if (onConfigure) {
                onConfigure.apply(this, arguments);
            }
            setTimeout(() => {
                this.cleanupDataInputs();
                this.sortInputs();
                this.updateOutputs();
            }, 100);
        };

        const onConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function(connectionType, slotIndex, isConnected, linkInfo, inputInfo) {
            if (onConnectionsChange) {
                onConnectionsChange.apply(this, arguments);
            }
            if (connectionType !== 1) {
                return;
            }
            const inputName = (inputInfo && inputInfo.name) ? inputInfo.name : (this.inputs && this.inputs[slotIndex] ? this.inputs[slotIndex].name : null);
            
            // Check if input is "data" or "data_X"
            let currentIndex = -1;
            if (inputName === "data") {
                currentIndex = 0;
            } else if (inputName && inputName.startsWith("data_")) {
                currentIndex = parseInt(inputName.split("_")[1]);
            }

            if (currentIndex !== -1) {
                if (isConnected) {
                    const nextIndex = currentIndex + 1;
                    const nextInputName = `data_${nextIndex}`;
                    if (!this.inputs.find(input => input.name === nextInputName) && nextIndex <= 7) {
                        this.addInput(nextInputName, "*");
                        this.setSize([this.size[0], this.computeSize()[1]]);
                    }
                    this.sortInputs();
                    this.updateOutputs();
                } else {
                    setTimeout(() => {
                        this.cleanupDataInputs();
                        this.sortInputs();
                        this.updateOutputs();
                    }, 10);
                }
            }
        };

        nodeType.prototype.sortInputs = function() {
            if (!this.inputs) return;
            
            const batchManager = [];
            const dataInputs = [];
            const others = [];
            
            this.inputs.forEach(input => {
                if (input.name === "batch_manager") {
                    batchManager.push(input);
                } else if (input.name === "data" || input.name.startsWith("data_")) {
                    dataInputs.push(input);
                } else {
                    others.push(input);
                }
            });
            
            dataInputs.sort((a, b) => {
                const getIndex = (name) => {
                    if (name === "data") return 0;
                    return parseInt(name.split("_")[1]);
                };
                return getIndex(a.name) - getIndex(b.name);
            });
            
            this.inputs = [...batchManager, ...dataInputs, ...others];
            this.setSize([this.size[0], this.computeSize()[1]]);
        };

        nodeType.prototype.cleanupDataInputs = function() {
            const dataInputs = this.inputs.filter(input => input.name === "data" || input.name.startsWith("data_"));
            
            // Sort to ensure we check in order: data, data_1, data_2...
            dataInputs.sort((a, b) => {
                const getIndex = (name) => {
                    if (name === "data") return 0;
                    return parseInt(name.split("_")[1]);
                };
                return getIndex(a.name) - getIndex(b.name);
            });

            let lastConnectedIndex = -1;
            for (let i = 0; i < dataInputs.length; i++) {
                const inputIndex = this.inputs.indexOf(dataInputs[i]);
                const linkId = this.getInputLink(inputIndex);
                if (linkId !== null) {
                    lastConnectedIndex = i;
                }
            }
            if (lastConnectedIndex === -1) {
                lastConnectedIndex = 0;
            }
            let keepCount = lastConnectedIndex + 2; // +1 for next empty slot
            if (keepCount > 8) {
                keepCount = 8;
            }
            
            // Remove extra inputs
            // We iterate from end to keepCount
            for (let i = dataInputs.length - 1; i >= keepCount; i--) {
                const inputIndex = this.inputs.indexOf(dataInputs[i]);
                this.removeInput(inputIndex);
            }
            
            // Add missing inputs if needed
            if (dataInputs.length < keepCount) {
                // Determine the next index to add based on the last existing one
                let lastExistingIndex = -1;
                if (dataInputs.length > 0) {
                    const lastName = dataInputs[dataInputs.length - 1].name;
                    lastExistingIndex = lastName === "data" ? 0 : parseInt(lastName.split("_")[1]);
                }
                
                const nextIndex = lastExistingIndex + 1;
                if (nextIndex <= 7) {
                    const nextInputName = `data_${nextIndex}`;
                    this.addInput(nextInputName, "*");
                }
            }
            this.sortInputs();
            this.setSize([this.size[0], this.computeSize()[1]]);
        };

        nodeType.prototype.moveBatchManagerToTop = function() {
           this.sortInputs();
        };

        nodeType.prototype.updateOutputs = function() {
            if (!this.outputs) {
                return;
            }
            // 如果节点尚未加入图(graph)，无法获取连接信息，直接返回
            if (!this.graph) {
                return;
            }
            const dataInputs = this.inputs.filter(input => input.name === "data" || input.name.startsWith("data_"));
            const requiredCount = Math.min(8, dataInputs.length);
            const requiredNames = [];
            
            // Generate names: index 0 -> data_out, index 1 -> data_out_1, etc.
            for (let i = 0; i < requiredCount; i++) {
                if (i === 0) {
                    requiredNames.push("data_out");
                } else {
                    requiredNames.push(`data_out_${i}`);
                }
            }
            
            const currentOutputs = this.outputs ? this.outputs.length : 0;
            const graph = this.graph || app.graph;
            if (currentOutputs > requiredCount) {
                for (let i = currentOutputs - 1; i >= requiredCount; i--) {
                    if (this.outputs[i] && this.outputs[i].links && this.outputs[i].links.length > 0) {
                        const linksToRemove = [...this.outputs[i].links];
                        linksToRemove.forEach(linkId => {
                            if (graph) {
                                graph.removeLink(linkId);
                            }
                        });
                    }
                    this.removeOutput(i);
                }
            } else if (currentOutputs < requiredCount) {
                for (let i = currentOutputs; i < requiredCount; i++) {
                    this.addOutput(requiredNames[i], "*");
                }
            }
            for (let i = 0; i < Math.min(currentOutputs, requiredCount); i++) {
                if (this.outputs[i] && this.outputs[i].name !== requiredNames[i]) {
                    this.outputs[i].name = requiredNames[i];
                }
            }
            this.setSize(this.computeSize());
            if (this.graph && this.graph.canvas) {
                this.graph.canvas.setDirty(true, false);
            }
            if (graph) {
                graph.setDirtyCanvas(true, true);
            }
        };

        const getExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
        nodeType.prototype.getExtraMenuOptions = function(_, options) {
            if (getExtraMenuOptions) {
                getExtraMenuOptions.apply(this, arguments);
            }
            options.push({
                content: "清理 Data 输入",
                callback: () => {
                    this.cleanupDataInputs();
                    this.updateOutputs();
                }
            });
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function(data) {
            if (onExecuted) {
                onExecuted.apply(this, arguments);
            }
            if (this.updateOutputs) {
                setTimeout(() => this.updateOutputs(), 1);
            }
        };
    }
});
