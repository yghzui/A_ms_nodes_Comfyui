import { app } from "../../../scripts/app.js";

const NODE_NAME = "MultiImageConditionReference";
const INPUT_PREFIX = "image_";
const MAX_IMAGES = 12;

app.registerExtension({
    name: "A_my_nodes.MultiImageConditionReference.DynamicInputs",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) {
            return;
        }

        const getImageInputs = (node) => {
            return (node.inputs || []).filter((input) => input.name?.startsWith(INPUT_PREFIX));
        };

        const getInputNumber = (name) => {
            const index = Number(String(name || "").replace(INPUT_PREFIX, ""));
            return Number.isFinite(index) ? index : 0;
        };

        const ensureInputOrder = (node) => {
            const imageInputs = getImageInputs(node).sort((a, b) => getInputNumber(a.name) - getInputNumber(b.name));
            const others = (node.inputs || []).filter((input) => !input.name?.startsWith(INPUT_PREFIX));
            node.inputs = [...others, ...imageInputs];
        };

        const cleanupImageInputs = (node) => {
            ensureInputOrder(node);
            const imageInputs = getImageInputs(node);

            let lastConnectedIndex = -1;
            for (let i = 0; i < imageInputs.length; i++) {
                const inputIndex = node.inputs.indexOf(imageInputs[i]);
                if (inputIndex !== -1 && node.getInputLink(inputIndex) !== null) {
                    lastConnectedIndex = i;
                }
            }

            const keepCount = Math.min(MAX_IMAGES, Math.max(1, lastConnectedIndex + 2));

            while (imageInputs.length > keepCount) {
                const input = imageInputs.pop();
                const inputIndex = node.inputs.indexOf(input);
                if (inputIndex !== -1) {
                    node.removeInput(inputIndex);
                }
            }

            while (imageInputs.length < keepCount && imageInputs.length < MAX_IMAGES) {
                const nextIndex = imageInputs.length + 1;
                node.addInput(`image_${nextIndex}`, "IMAGE", {
                    tooltip: `第 ${nextIndex} 张参考图。`,
                });
                imageInputs.push(node.inputs[node.inputs.length - 1]);
            }

            imageInputs.forEach((input, index) => {
                input.name = `image_${index + 1}`;
                input.type = "IMAGE";
                if (!input.extra_info) {
                    input.extra_info = {};
                }
                input.extra_info.tooltip = `第 ${index + 1} 张参考图。`;
            });

            ensureInputOrder(node);
            node.setSize([node.size[0], node.computeSize()[1]]);
        };

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function() {
            onNodeCreated?.apply(this, arguments);
            setTimeout(() => cleanupImageInputs(this), 0);
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function(info) {
            onConfigure?.apply(this, arguments);
            setTimeout(() => cleanupImageInputs(this), 50);
        };

        const onConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function(connectionType, slotIndex, isConnected, linkInfo, inputInfo) {
            onConnectionsChange?.apply(this, arguments);
            if (connectionType !== 1) {
                return;
            }

            const inputName = inputInfo?.name || this.inputs?.[slotIndex]?.name;
            if (!inputName?.startsWith(INPUT_PREFIX)) {
                return;
            }

            if (isConnected) {
                cleanupImageInputs(this);
            } else {
                setTimeout(() => cleanupImageInputs(this), 10);
            }
        };

        const getExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
        nodeType.prototype.getExtraMenuOptions = function(_, options) {
            getExtraMenuOptions?.apply(this, arguments);
            options.push({
                content: "清理参考图输入",
                callback: () => cleanupImageInputs(this),
            });
        };
    },
});
