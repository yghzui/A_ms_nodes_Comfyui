import { ComfyWidgets } from "../../../scripts/widgets.js";
import { LogLevel, rgthree } from "./rgthree.js";
import { defineProperty, moveArrayItem } from "./shared_utils.js";

export class RgthreeBaseNode extends LGraphNode {
    constructor(title = RgthreeBaseNode.title, skipOnConstructedCall = true) {
        super(title);
        this.comfyClass = "__NEED_COMFY_CLASS__";
        this.nickname = "rgthree";
        this.isVirtualNode = false;
        this.isDropEnabled = false;
        this.removed = false;
        this.configuring = false;
        this._tempWidth = 0;
        this.__constructed__ = false;
        this.helpDialog = null;
        if (title == "__NEED_CLASS_TITLE__") {
            throw new Error("RgthreeBaseNode needs overrides.");
        }
        this.widgets = this.widgets || [];
        this.properties = this.properties || {};
        setTimeout(() => {
            if (this.comfyClass == "__NEED_COMFY_CLASS__") {
                throw new Error("RgthreeBaseNode needs a comfy class override.");
            }
            if (this.constructor.type == "__NEED_CLASS_TYPE__") {
                throw new Error("RgthreeBaseNode needs overrides.");
            }
            this.checkAndRunOnConstructed();
        });
        defineProperty(this, "mode", {
            get: () => {
                return this.rgthree_mode;
            },
            set: (mode) => {
                if (this.rgthree_mode != mode) {
                    const oldMode = this.rgthree_mode;
                    this.rgthree_mode = mode;
                    this.onModeChange(oldMode, mode);
                }
            },
        });
    }
    checkAndRunOnConstructed() {
        if (!this.__constructed__) {
            this.onConstructed();
        }
        return this.__constructed__;
    }
    onConstructed() {
        if (this.__constructed__)
            return false;
        this.type = this.type || undefined;
        this.__constructed__ = true;
        return this.__constructed__;
    }
    configure(info) {
        this.configuring = true;
        super.configure(info);
        for (const w of this.widgets || []) {
            w.last_y = w.last_y || 0;
        }
        this.configuring = false;
    }
    clone() {
        const cloned = super.clone();
        if (cloned?.properties && !!window.structuredClone) {
            cloned.properties = structuredClone(cloned.properties);
        }
        return cloned;
    }
    onModeChange(from, to) {
    }
    async handleAction(action) {
        action;
    }
    removeWidget(widgetOrSlot) {
        if (!this.widgets) {
            return;
        }
        const widget = typeof widgetOrSlot === "number" ? this.widgets[widgetOrSlot] : widgetOrSlot;
        if (widget) {
            const index = this.widgets.indexOf(widget);
            if (index > -1) {
                this.widgets.splice(index, 1);
            }
            widget.onRemove?.call(widget);
        }
    }
    replaceWidget(widgetOrSlot, newWidget) {
        let index = null;
        if (widgetOrSlot) {
            index = typeof widgetOrSlot === 'number' ? widgetOrSlot : this.widgets.indexOf(widgetOrSlot);
            this.removeWidget(index);
        }
        index = index != null ? index : this.widgets.length - 1;
        if (this.widgets.includes(newWidget)) {
            moveArrayItem(this.widgets, newWidget, index);
        }
        else {
            this.widgets.splice(index, 0, newWidget);
        }
    }
    defaultGetSlotMenuOptions(slot) {
        const menu_info = [];
        if (slot?.output?.links?.length) {
            menu_info.push({ content: "Disconnect Links", slot });
        }
        let inputOrOutput = slot.input || slot.output;
        if (inputOrOutput) {
            if (inputOrOutput.removable) {
                menu_info.push(inputOrOutput.locked ? { content: "Cannot remove" } : { content: "Remove Slot", slot });
            }
            if (!inputOrOutput.nameLocked) {
                menu_info.push({ content: "Rename Slot", slot });
            }
        }
        return menu_info;
    }
    onRemoved() {
        super.onRemoved?.call(this);
        this.removed = true;
    }
    static setUp(...args) {
    }
    getHelp() {
        return "";
    }
    showHelp() {
        const help = this.getHelp() || this.constructor.help;
        if (help) {
            console.log("Help:", help);
        }
    }
    onKeyDown(event) {
        if (event.key == "?" && !this.helpDialog) {
            this.showHelp();
        }
    }
    onKeyUp(event) {
    }
    getExtraMenuOptions(canvas, options) {
        if (super.getExtraMenuOptions) {
            super.getExtraMenuOptions.apply(this, [canvas, options]);
        }
        else if (this.constructor.nodeType?.prototype?.getExtraMenuOptions) {
            this.constructor.nodeType.prototype.getExtraMenuOptions.apply(this, [
                canvas,
                options,
            ]);
        }
        const help = this.getHelp() || this.constructor.help;
        if (help) {
            console.log("Help:", help);
        }
        return [];
    }
}

RgthreeBaseNode.exposedActions = [];
RgthreeBaseNode.title = "__NEED_CLASS_TITLE__";
RgthreeBaseNode.type = "__NEED_CLASS_TYPE__";
RgthreeBaseNode.category = "rgthree";
RgthreeBaseNode._category = "rgthree";

export class RgthreeBaseVirtualNode extends RgthreeBaseNode {
    constructor(title = RgthreeBaseNode.title) {
        super(title, false);
        this.isVirtualNode = true;
    }
    static setUp() {
        if (!this.type) {
            throw new Error(`Missing type for RgthreeBaseVirtualNode: ${this.title}`);
        }
        LiteGraph.registerNodeType(this.type, this);
        if (this._category) {
            this.category = this._category;
        }
    }
}

export class RgthreeBaseServerNode extends RgthreeBaseNode {
    constructor(title) {
        super(title, true);
        this.isDropEnabled = true;
        this.serialize_widgets = true;
        this.setupFromServerNodeData();
        this.onConstructed();
    }
    getWidgets() {
        return ComfyWidgets;
    }
    async setupFromServerNodeData() {
        const nodeData = this.constructor.nodeData;
        if (!nodeData) {
            throw Error("No node data");
        }
        this.comfyClass = nodeData.name;
        let inputs = nodeData["input"]["required"];
        if (nodeData["input"]["optional"] != undefined) {
            inputs = Object.assign({}, inputs, nodeData["input"]["optional"]);
        }
        const WIDGETS = this.getWidgets();
        const config = {
            minWidth: 1,
            minHeight: 1,
            widget: null,
        };
        for (const inputName in inputs) {
            const inputData = inputs[inputName];
            const type = inputData[0];
            if (inputData[1]?.forceInput) {
                this.addInput(inputName, type);
            }
            else {
                let widgetCreated = true;
                if (Array.isArray(type)) {
                    Object.assign(config, WIDGETS.COMBO(this, inputName, inputData, app) || {});
                }
                else if (`${type}:${inputName}` in WIDGETS) {
                    Object.assign(config, WIDGETS[`${type}:${inputName}`](this, inputName, inputData, app) || {});
                }
                else if (type in WIDGETS) {
                    Object.assign(config, WIDGETS[type](this, inputName, inputData, app) || {});
                }
                else {
                    this.addInput(inputName, type);
                    widgetCreated = false;
                }
                if (widgetCreated && inputData[1]?.forceInput && config?.widget) {
                    if (!config.widget.options)
                        config.widget.options = {};
                    config.widget.options.forceInput = inputData[1].forceInput;
                }
                if (widgetCreated && inputData[1]?.defaultInput && config?.widget) {
                    if (!config.widget.options)
                        config.widget.options = {};
                    config.widget.options.defaultInput = inputData[1].defaultInput;
                }
            }
        }
        for (const o in nodeData["output"]) {
            let output = nodeData["output"][o];
            if (output instanceof Array)
                output = "COMBO";
            const outputName = nodeData["output_name"][o] || output;
            const outputShape = nodeData["output_is_list"][o]
                ? LiteGraph.GRID_SHAPE
                : LiteGraph.CIRCLE_SHAPE;
            this.addOutput(outputName, output, { shape: outputShape });
        }
        const s = this.computeSize();
        s[0] = Math.max(config.minWidth || 1, s[0] * 1.5);
        s[1] = Math.max(config.minHeight || 1, s[1]);
        this.size = s;
        this.serialize_widgets = true;
    }
    static registerForOverride(comfyClass, nodeData, rgthreeClass) {
        if (OVERRIDDEN_SERVER_NODES.has(comfyClass)) {
            throw Error(`Already have a class to override ${comfyClass.type || comfyClass.name || comfyClass.title}`);
        }
        OVERRIDDEN_SERVER_NODES.set(comfyClass, rgthreeClass);
        if (!rgthreeClass.__registeredForOverride__) {
            rgthreeClass.__registeredForOverride__ = true;
            rgthreeClass.nodeType = comfyClass;
            rgthreeClass.nodeData = nodeData;
            rgthreeClass.onRegisteredForOverride(comfyClass, rgthreeClass);
        }
    }
    static onRegisteredForOverride(comfyClass, rgthreeClass) {
    }
}

RgthreeBaseServerNode.nodeType = null;
RgthreeBaseServerNode.nodeData = null;
RgthreeBaseServerNode.__registeredForOverride__ = false;

const OVERRIDDEN_SERVER_NODES = new Map();
const oldregisterNodeType = LiteGraph.registerNodeType;
LiteGraph.registerNodeType = async function (nodeId, baseClass) {
    const clazz = OVERRIDDEN_SERVER_NODES.get(baseClass) || baseClass;
    if (clazz !== baseClass) {
        const classLabel = clazz.type || clazz.name || clazz.title;
        console.log(`${nodeId}: replacing default ComfyNode implementation with custom ${classLabel} class.`);
    }
    return oldregisterNodeType.call(LiteGraph, nodeId, clazz);
};
