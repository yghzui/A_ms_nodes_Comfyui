import { app } from "../../../scripts/app.js";

const NODE_NAME = "MultiInputStateMapper";
const MAX_INPUTS = 12;

app.registerExtension({
    name: "A_my_nodes.MultiInputStateMapper.DynamicInputs",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) {
            return;
        }

        const getDynamicInputs = (node) => {
            return (node.inputs || []).filter((input) => /^value_\d+$/.test(input.name));
        };

        const getInputNumber = (inputName) => {
            const matched = /^value_(\d+)$/.exec(inputName || "");
            return matched ? Number(matched[1]) : -1;
        };

        const isInputConnected = (node, input) => {
            if (!input) {
                return false;
            }
            if (input.link !== undefined && input.link !== null) {
                return true;
            }
            const inputIndex = node.inputs.indexOf(input);
            if (inputIndex === -1 || !node.graph) {
                return false;
            }
            return node.getInputLink(inputIndex) !== null;
        };

        const sortDynamicInputs = (node) => {
            if (!node.inputs) {
                return;
            }

            const dynamicInputs = [];
            const others = [];
            node.inputs.forEach((input) => {
                if (/^value_\d+$/.test(input.name)) {
                    dynamicInputs.push(input);
                } else {
                    others.push(input);
                }
            });

            dynamicInputs.sort((a, b) => getInputNumber(a.name) - getInputNumber(b.name));
            node.inputs = [...dynamicInputs, ...others];
        };

        const syncDynamicInputs = (node) => {
            sortDynamicInputs(node);

            const dynamicInputs = getDynamicInputs(node);
            let lastConnectedIndex = -1;

            dynamicInputs.forEach((input, index) => {
                if (isInputConnected(node, input)) {
                    lastConnectedIndex = index;
                }
            });

            let keepCount = Math.max(1, lastConnectedIndex + 2);
            keepCount = Math.min(MAX_INPUTS, keepCount);

            while (dynamicInputs.length > keepCount) {
                const input = dynamicInputs.pop();
                const inputIndex = node.inputs.indexOf(input);
                if (inputIndex !== -1) {
                    node.removeInput(inputIndex);
                }
            }

            while (dynamicInputs.length < keepCount) {
                const nextIndex = dynamicInputs.length + 1;
                node.addInput(`value_${nextIndex}`, "*");
                dynamicInputs.push(node.inputs[node.inputs.length - 1]);
            }

            dynamicInputs.forEach((input, index) => {
                const inputNumber = index + 1;
                input.name = `value_${inputNumber}`;
                input.type = "*";
                if (!input.extra_info) {
                    input.extra_info = {};
                }
                input.extra_info.tooltip = `第 ${inputNumber} 个动态输入，支持连接 INT、FLOAT、BOOLEAN。仅前两个输入参与判断。`;
            });

            sortDynamicInputs(node);
            node.setSize([node.size[0], node.computeSize()[1]]);
            node.graph?.setDirtyCanvas(true, true);
        };

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function() {
            onNodeCreated?.apply(this, arguments);
            setTimeout(() => syncDynamicInputs(this), 0);
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function(info) {
            onConfigure?.apply(this, arguments);
            setTimeout(() => syncDynamicInputs(this), 50);
        };

        const onConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function(connectionType, slotIndex, isConnected, linkInfo, slot) {
            onConnectionsChange?.apply(this, arguments);
            if (connectionType !== 1) {
                return;
            }

            const inputName = slot?.name || this.inputs?.[slotIndex]?.name;
            if (!/^value_\d+$/.test(inputName || "")) {
                return;
            }

            if (isConnected) {
                syncDynamicInputs(this);
            } else {
                setTimeout(() => syncDynamicInputs(this), 10);
            }
        };

        const getExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
        nodeType.prototype.getExtraMenuOptions = function(_, options) {
            getExtraMenuOptions?.apply(this, arguments);
            options.push({
                content: "清理状态映射输入",
                callback: () => syncDynamicInputs(this),
            });
        };
    },
});
