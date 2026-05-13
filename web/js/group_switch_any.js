import { app } from "../../../scripts/app.js";

const NODE_NAME = "GroupSwitchAny";
const MAX_OUTPUTS = 5;
const INPUT_PATTERN = /^input_(\d+)_(\d+)$/;

app.registerExtension({
    name: "A_my_nodes.GroupSwitchAny.DynamicIO",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) {
            return;
        }

        const getGroupSizeWidget = (node) => {
            return node.widgets?.find((widget) => widget.name === "group_size");
        };

        const getSafeGroupSize = (node) => {
            const widget = getGroupSizeWidget(node);
            const raw = widget ? Number(widget.value) : 2;
            if (!Number.isFinite(raw)) {
                return 2;
            }
            return Math.max(1, Math.min(MAX_OUTPUTS, Math.floor(raw)));
        };

        const getDynamicInputs = (node) => {
            return (node.inputs || []).filter((input) => INPUT_PATTERN.test(input.name));
        };

        const getInputNameByFlatIndex = (flatIndex, groupSize) => {
            const groupIndex = Math.floor(flatIndex / groupSize) + 1;
            const slotIndex = (flatIndex % groupSize) + 1;
            return `input_${groupIndex}_${slotIndex}`;
        };

        const isInputConnected = (node, input) => {
            if (!input) {
                return false;
            }
            if (input.link !== undefined && input.link !== null) {
                return true;
            }
            if (!node.graph) {
                return false;
            }
            const inputIndex = node.inputs.indexOf(input);
            return inputIndex !== -1 && node.getInputLink(inputIndex) !== null;
        };

        const ensureDynamicInputs = (node) => {
            const groupSize = getSafeGroupSize(node);
            const dynamicInputs = getDynamicInputs(node);

            let lastConnectedIndex = -1;
            dynamicInputs.forEach((input, flatIndex) => {
                if (isInputConnected(node, input)) {
                    lastConnectedIndex = flatIndex;
                }
            });

            const keepCount = Math.max(1, lastConnectedIndex + 2);

            while (dynamicInputs.length > keepCount) {
                const input = dynamicInputs.pop();
                const inputIndex = node.inputs.indexOf(input);
                if (inputIndex !== -1) {
                    node.removeInput(inputIndex);
                }
            }

            while (dynamicInputs.length < keepCount) {
                const flatIndex = dynamicInputs.length;
                const inputName = getInputNameByFlatIndex(flatIndex, groupSize);
                node.addInput(inputName, "*");
                dynamicInputs.push(node.inputs[node.inputs.length - 1]);
            }

            dynamicInputs.forEach((input, flatIndex) => {
                input.name = getInputNameByFlatIndex(flatIndex, groupSize);
                input.type = "*";
                if (!input.extra_info) {
                    input.extra_info = {};
                }
                input.extra_info.tooltip = `第 ${Math.floor(flatIndex / groupSize) + 1} 组第 ${(flatIndex % groupSize) + 1} 个输入`;
            });

            node.setSize([node.size[0], node.computeSize()[1]]);
        };

        const updateOutputs = (node) => {
            const requiredCount = getSafeGroupSize(node);
            const graph = node.graph || app.graph;
            const currentOutputs = node.outputs ? node.outputs.length : 0;

            if (currentOutputs > requiredCount) {
                for (let i = currentOutputs - 1; i >= requiredCount; i--) {
                    if (node.outputs[i]?.links?.length) {
                        const linksToRemove = [...node.outputs[i].links];
                        linksToRemove.forEach((linkId) => graph?.removeLink(linkId));
                    }
                    node.removeOutput(i);
                }
            } else if (currentOutputs < requiredCount) {
                for (let i = currentOutputs; i < requiredCount; i++) {
                    node.addOutput(`out${i + 1}`, "*");
                }
            }

            for (let i = 0; i < requiredCount; i++) {
                if (node.outputs[i]) {
                    node.outputs[i].name = `out${i + 1}`;
                    node.outputs[i].type = "*";
                }
            }

            node.setSize(node.computeSize());
            graph?.setDirtyCanvas(true, true);
        };

        const syncNodeLayout = (node) => {
            ensureDynamicInputs(node);
            updateOutputs(node);
        };

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function() {
            onNodeCreated?.apply(this, arguments);
            setTimeout(() => syncNodeLayout(this), 0);

            const groupSizeWidget = getGroupSizeWidget(this);
            if (groupSizeWidget) {
                const originalCallback = groupSizeWidget.callback;
                groupSizeWidget.callback = (value) => {
                    originalCallback?.call(groupSizeWidget, value);
                    syncNodeLayout(this);
                };
            }
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function(info) {
            onConfigure?.apply(this, arguments);
            setTimeout(() => syncNodeLayout(this), 50);
        };

        const onConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function(connectionType, slotIndex, isConnected, linkInfo, slot) {
            onConnectionsChange?.apply(this, arguments);
            if (connectionType !== 1) {
                return;
            }

            const inputName = slot?.name || this.inputs?.[slotIndex]?.name;
            if (!inputName || !INPUT_PATTERN.test(inputName)) {
                return;
            }

            if (isConnected) {
                syncNodeLayout(this);
            } else {
                setTimeout(() => syncNodeLayout(this), 10);
            }
        };

        const getExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
        nodeType.prototype.getExtraMenuOptions = function(_, options) {
            getExtraMenuOptions?.apply(this, arguments);
            options.push({
                content: "清理 GroupSwitchAny 输入",
                callback: () => syncNodeLayout(this),
            });
        };
    },
});
