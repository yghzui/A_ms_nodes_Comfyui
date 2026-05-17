import { app } from "../../../scripts/app.js";

const NODE_NAME = "GroupSwitchAny";
const MAX_OUTPUTS = 8;
const INPUT_PATTERN = /^input_(\d+)_(\d+)$/;
const INPUT_CLEANUP_DELAY_MS = 200;
const OUTPUT_CLEANUP_DELAY_MS = 200;

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

        const cancelPendingInputCleanup = (node) => {
            if (node.__groupSwitchAnyInputCleanupTimer) {
                clearTimeout(node.__groupSwitchAnyInputCleanupTimer);
                node.__groupSwitchAnyInputCleanupTimer = null;
            }
        };

        const cancelPendingOutputCleanup = (node) => {
            if (node.__groupSwitchAnyOutputCleanupTimer) {
                clearTimeout(node.__groupSwitchAnyOutputCleanupTimer);
                node.__groupSwitchAnyOutputCleanupTimer = null;
            }
        };

        const getRequiredInputCount = (node) => {
            const dynamicInputs = getDynamicInputs(node);
            let highestConnectedIndex = -1;

            dynamicInputs.forEach((input, flatIndex) => {
                if (isInputConnected(node, input)) {
                    highestConnectedIndex = flatIndex;
                }
            });

            return Math.max(1, highestConnectedIndex + 2);
        };

        const pruneTrailingUnusedInputs = (node) => {
            const keepCount = getRequiredInputCount(node);
            const dynamicInputs = getDynamicInputs(node);
            while (dynamicInputs.length > keepCount) {
                const input = dynamicInputs.pop();
                const inputIndex = node.inputs.indexOf(input);
                if (inputIndex !== -1) {
                    node.removeInput(inputIndex);
                }
            }
        };

        const ensureDynamicInputs = (node) => {
            const groupSize = getSafeGroupSize(node);
            const dynamicInputs = getDynamicInputs(node);
            const keepCount = getRequiredInputCount(node);

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

            cancelPendingInputCleanup(node);
            if (dynamicInputs.length > keepCount) {
                node.__groupSwitchAnyInputCleanupTimer = setTimeout(() => {
                    node.__groupSwitchAnyInputCleanupTimer = null;
                    pruneTrailingUnusedInputs(node);
                    ensureDynamicInputs(node);
                    (node.graph || app.graph)?.setDirtyCanvas(true, true);
                }, INPUT_CLEANUP_DELAY_MS);
            }

            node.setSize([node.size[0], node.computeSize()[1]]);
        };

        const isOutputConnected = (node, output, outputIndex) => {
            if (!output) {
                return false;
            }
            if (Array.isArray(output.links) && output.links.length > 0) {
                return true;
            }
            const graph = node.graph || app.graph;
            if (!graph || !Array.isArray(graph.links) && typeof graph.links !== "object") {
                return false;
            }
            const links = output.links || [];
            return Array.isArray(links) && links.some((linkId) => !!graph.links?.[linkId]) || false;
        };

        const getRequiredOutputCount = (node) => {
            const outputs = node.outputs || [];
            let highestConnectedIndex = -1;
            outputs.forEach((output, outputIndex) => {
                if (isOutputConnected(node, output, outputIndex)) {
                    highestConnectedIndex = outputIndex;
                }
            });
            return Math.max(getSafeGroupSize(node), highestConnectedIndex + 1, 1);
        };

        const pruneTrailingUnusedOutputs = (node) => {
            const graph = node.graph || app.graph;
            const keepCount = getRequiredOutputCount(node);
            while ((node.outputs?.length || 0) > keepCount) {
                const lastIndex = node.outputs.length - 1;
                const lastOutput = node.outputs[lastIndex];
                if (isOutputConnected(node, lastOutput, lastIndex)) {
                    break;
                }
                node.removeOutput(lastIndex);
            }
            node.setSize(node.computeSize());
            graph?.setDirtyCanvas(true, true);
        };

        const syncOutputMetadata = (node) => {
            const outputs = node.outputs || [];
            for (let i = 0; i < outputs.length; i++) {
                if (outputs[i]) {
                    outputs[i].name = `out${i + 1}`;
                    outputs[i].type = "*";
                    if (!outputs[i].extra_info) {
                        outputs[i].extra_info = {};
                    }
                    outputs[i].extra_info.tooltip = `第 ${i + 1} 个输出。当前 group_size 小于该位置时会输出空值。`;
                }
            }
        };

        const updateOutputs = (node, options = {}) => {
            const { bootstrap = false } = options;
            const graph = node.graph || app.graph;
            const requiredCount = bootstrap ? MAX_OUTPUTS : getRequiredOutputCount(node);
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

            cancelPendingOutputCleanup(node);
            syncOutputMetadata(node);

            if (bootstrap) {
                node.__groupSwitchAnyOutputCleanupTimer = setTimeout(() => {
                    node.__groupSwitchAnyOutputCleanupTimer = null;
                    pruneTrailingUnusedOutputs(node);
                }, OUTPUT_CLEANUP_DELAY_MS);
            }

            node.setSize(node.computeSize());
            graph?.setDirtyCanvas(true, true);
        };

        const syncNodeLayout = (node, options = {}) => {
            ensureDynamicInputs(node);
            updateOutputs(node, options);
        };

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function() {
            onNodeCreated?.apply(this, arguments);
            setTimeout(() => syncNodeLayout(this, { bootstrap: true }), 0);

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
            cancelPendingInputCleanup(this);
            cancelPendingOutputCleanup(this);
            setTimeout(() => syncNodeLayout(this, { bootstrap: true }), 50);
        };

        const onConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function(connectionType, slotIndex, isConnected, linkInfo, slot) {
            onConnectionsChange?.apply(this, arguments);
            if (connectionType === 2) {
                cancelPendingOutputCleanup(this);
                setTimeout(() => updateOutputs(this), 10);
                return;
            }
            if (connectionType !== 1) {
                return;
            }

            const inputName = slot?.name || this.inputs?.[slotIndex]?.name;
            if (!inputName || !INPUT_PATTERN.test(inputName)) {
                return;
            }

            if (isConnected) {
                cancelPendingInputCleanup(this);
                syncNodeLayout(this);
            } else {
                cancelPendingInputCleanup(this);
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
