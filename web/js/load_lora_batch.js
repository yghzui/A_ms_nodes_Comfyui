import { app } from "../../../scripts/app.js";
import { drawNumberWidgetPart, drawRoundedRectangle, drawTogglePart, fitString, isLowQuality, } from "./utils/utils_canvas.js";
import { RgthreeBaseWidget, RgthreeDividerWidget, addIconButtonRow } from "./utils/utils_widgets.js";
import { rgthreeApi } from "./core/rgthree_api.js";
import { moveArrayItem } from "./utils/shared_utils.js";
import { rgthree } from "./core/rgthree.js";
import { getLoraRowSlot, getLoraRowMenuItems } from "./lora/lora_common.js";

console.log("Patching node: load_lora_batch.js");
console.log("Loaded load_lora_batch.js");

// 改进的LoRA选择器，基于官方实现修复位置定位问题
async function showLoraChooser(event, callback, parentMenu, loras, buttonNode, buttonWidget) {
    const canvas = app.canvas;
    if (!loras) {
        loras = ["None", ...(await rgthreeApi.getLoras().then((loras) => loras.map((l) => l.file)))];
    }

    const menuItems = loras.map(lora => ({
        content: lora,
        callback: () => callback(lora)
    }));

    let menuEvent = event;
    let targetX, targetY;

    // DOM 按钮控件（带 element）用点击事件的屏幕坐标定位，只有画布绘制的控件才用 last_y 换算
    if (buttonNode && buttonWidget && !buttonWidget.element) {
        const canvasRect = canvas.canvas.getBoundingClientRect();
        const ds = canvas.ds || { scale: 1, offset: [0, 0] };

        const nodeX = buttonNode.pos[0];
        const nodeY = buttonNode.pos[1];
        const widgetY = buttonWidget.last_y || 0;
        const widgetLeftMargin = 15; // 与按钮绘制保持一致

        const anchorCanvasX = nodeX + widgetLeftMargin;
        const anchorCanvasY = nodeY + widgetY;

        targetX = (anchorCanvasX + ds.offset[0]) * ds.scale + canvasRect.left;
        targetY = (anchorCanvasY + ds.offset[1]) * ds.scale + canvasRect.top;

        menuEvent = new MouseEvent('contextmenu', {
            clientX: targetX,
            clientY: targetY,
            bubbles: true,
            cancelable: true,
            view: window
        });
    } else if (event && event.clientX !== undefined) {
        targetX = event.clientX;
        targetY = event.clientY;
        menuEvent = new MouseEvent('contextmenu', {
            clientX: targetX,
            clientY: targetY,
            bubbles: true,
            cancelable: true,
            view: window
        });
    }

    const contextMenu = new LiteGraph.ContextMenu(menuItems, {
        event: menuEvent,
        parentMenu: parentMenu || undefined,
        title: "选择LoRA",
        scale: Math.max(1, canvas.ds?.scale || 1),
        className: "dark",
        callback,
    });

    if (contextMenu && contextMenu.root && targetX !== undefined && targetY !== undefined) {
        requestAnimationFrame(() => {
            const rect = contextMenu.root.getBoundingClientRect();
            const bodyRect = document.body.getBoundingClientRect();

            contextMenu.root.style.left = targetX + 'px';

            let finalY = targetY;
            if (bodyRect.height && targetY + rect.height > bodyRect.height - 10) {
                finalY = Math.max(10, bodyRect.height - rect.height - 10);
            }
            contextMenu.root.style.top = finalY + 'px';

            if (bodyRect.width && targetX + rect.width > bodyRect.width - 10) {
                const adjustedX = Math.max(10, bodyRect.width - rect.width - 10);
                contextMenu.root.style.left = adjustedX + 'px';
            }
        });
    }
}

// LoadLoraBatch节点类
class LoadLoraBatchNode extends LGraphNode {
    constructor(title = "LoadLoraBatch") {
        super(title);
        this.serialize_widgets = true;
        this.loraWidgetsCounter = 0;
        this.widgetButtonSpacer = null;
        this.widgets = this.widgets || [];
        this.properties = this.properties || {};
        rgthreeApi.getLoras();
    }

    addCustomWidget(widget) {
        this.widgets.push(widget);
        return widget;
    }

    setDirtyCanvas(flag, flag2) {
        if (app.canvas) {
            app.canvas.setDirty(flag, flag2);
        }
    }

    computeSize() {
        return [240, 120];
    }

    configure(info) {
        console.log("[LoadLoraBatch] 配置节点", info);
        console.log("[LoadLoraBatch] widgets_values:", info.widgets_values);
        
        // 清除现有的widgets
        // HTML 按钮控件（DOM 控件）由前端 store 注册管理，只清空数组会让元素残留在画布上，故保留它，只清理画布绘制的控件
        if (this.widgets) {
            const keptWidget = this.widgets.find((w) => w && w.name === "lora_actions");
            this.widgets.length = 0;
            if (keptWidget) {
                this.widgets.push(keptWidget);
            }
        }
        this.widgetButtonSpacer = null;
        
        // 调用父类的configure
        super.configure(info);
        
        // 保存临时尺寸
        this._tempWidth = this.size[0];
        this._tempHeight = this.size[1];
        
        // 恢复LoRA控件
        for (const widgetValue of info.widgets_values || []) {
            if (widgetValue && typeof widgetValue === 'object' && widgetValue.lora !== undefined) {
                console.log("[LoadLoraBatch] 恢复LoRA控件:", widgetValue);
                const widget = this.addNewLoraWidget();
                widget.value = { ...widgetValue };
            }
        }
        
        // 添加非LoRA控件
        this.addNonLoraWidgets();
        
        // 恢复尺寸
        this.size[0] = this._tempWidth;
        this.size[1] = Math.max(this._tempHeight, this.computeSize()[1]);
    }

    onNodeCreated() {
        this.addNonLoraWidgets();
        const computed = this.computeSize();
        this.size = this.size || [0, 0];
        this.size[0] = Math.max(this.size[0], computed[0]);
        this.size[1] = Math.max(this.size[1], computed[1]);
        this.setDirtyCanvas(true, true);
    }

    addNewLoraWidget(lora) {
        this.loraWidgetsCounter++;
        const widget = this.addCustomWidget(new LoadLoraBatchWidget("LORA_" + this.loraWidgetsCounter));
        if (lora) {
            widget.setLora(lora);
        }
        if (this.widgetButtonSpacer && this.widgets.indexOf(this.widgetButtonSpacer) !== -1) {
            moveArrayItem(this.widgets, widget, this.widgets.indexOf(this.widgetButtonSpacer));
        }
        return widget;
    }

    addNonLoraWidgets() {
        // 确保widgets数组存在
        if (!this.widgets) {
            this.widgets = [];
        }
        
        // 添加分隔线
        const divider1 = this.addCustomWidget(new RgthreeDividerWidget({ marginTop: 1, marginBottom: 0, thickness: 0 }));
        
        // 添加按钮分隔线
        //this.widgetButtonSpacer = this.addCustomWidget(new RgthreeDividerWidget({ marginTop: 2, marginBottom: 0, thickness: 0 }));
        
        // 添加增加LoRA按钮（若已存在则复用，避免重复创建 DOM 控件）
        let addButton = this.widgets.find((w) => w && w.name === "lora_actions");
        if (!addButton) {
            addButton = addIconButtonRow(this, "lora_actions", [{ icon: "+", title: "添加 LoRA", onClick: (event, pos, node) => {
                // 传入节点和按钮控件信息以实现正确的位置定位
                showLoraChooser(event || rgthree.lastCanvasMouseEvent, (value) => {
                    if (typeof value === "string") {
                        if (value !== "NONE") {
                            this.addNewLoraWidget(value);
                            const computed = this.computeSize();
                            this.size[1] = Math.max(this.size[1] || 15, computed[1]);
                            this.setDirtyCanvas(true, true);
                        }
                    }
                }, null, null, this, addButton); // 传入节点和按钮控件信息
                return true;
            } }]);
            addButton.serializeValue = () => undefined; // 阻止按钮被序列化
        }
        // 按钮始终排在控件列表末尾
        moveArrayItem(this.widgets, addButton, this.widgets.length - 1);
    }

    getSlotInPosition(canvasX, canvasY) {
        return getLoraRowSlot(this, canvasX, canvasY);
    }

    getSlotMenuOptions(slot, event) {
        return getLoraRowMenuItems(this, slot, {
            extraItems: (node) => [{
                content: "🗑️ 清空所有LoRA",
                callback: () => {
                    node.widgets = node.widgets.filter(widget => !widget.name || !widget.name.startsWith("LORA_"));
                    node.setDirtyCanvas(true, true);
                },
            }],
        });
    }

    hasLoraWidgets() {
        return !!this.widgets.find((w) => w.name && w.name.startsWith("LORA_"));
    }

    // 添加全局右键菜单
    getExtraMenuOptions(canvas, options) {
        // 移除右键菜单中的添加和清空LoRA选项
        return [];
    }
}

// LoadLoraBatch控件类
class LoadLoraBatchWidget extends RgthreeBaseWidget {
    constructor(name) {
        super(name);
        this.type = "custom";
        this.haveMouseMovedStrength = false;
        this.hitAreas = {
            toggle: { bounds: [0, 0], onDown: this.onToggleDown.bind(this) },
            lora: { bounds: [0, 0], onClick: this.onLoraClick.bind(this) },
            strengthDec: { bounds: [0, 0], onClick: this.onStrengthDecDown.bind(this) },
            strengthVal: { bounds: [0, 0], onClick: this.onStrengthValUp.bind(this) },
            strengthInc: { bounds: [0, 0], onClick: this.onStrengthIncDown.bind(this) },
            strengthAny: { bounds: [0, 0], onMove: this.onStrengthAnyMove.bind(this) },
        };
        this._value = {
            on: true,
            lora: null,
            strength: 1,
        };
    }

    set value(v) {
        this._value = v;
        if (typeof this._value !== "object") {
            this._value = { on: true, lora: null, strength: 1 };
        }
    }

    get value() {
        return this._value;
    }

    setLora(lora) {
        this._value.lora = lora;
    }

    draw(ctx, node, w, posY, height) {
        ctx.save();
        const margin = 10;
        const innerMargin = margin * 0.33;
        const lowQuality = isLowQuality();
        const midY = posY + height * 0.5;
        let posX = margin;
        
        drawRoundedRectangle(ctx, { pos: [posX, posY], size: [node.size[0] - margin * 2, height] });
        this.hitAreas.toggle.bounds = drawTogglePart(ctx, { posX, posY, height, value: this.value.on });
        posX += this.hitAreas.toggle.bounds[1] + innerMargin;
        
        if (lowQuality) {
            ctx.restore();
            return;
        }
        
        if (!this.value.on) {
            ctx.globalAlpha = app.canvas.editor_alpha * 0.4;
        }
        
        ctx.fillStyle = LiteGraph.WIDGET_TEXT_COLOR;
        let rposX = node.size[0] - margin - innerMargin - innerMargin;
        
        const [leftArrow, text, rightArrow] = drawNumberWidgetPart(ctx, {
            posX: node.size[0] - margin - innerMargin - innerMargin,
            posY,
            height,
            value: this.value.strength || 1,
            direction: -1,
        });
        
        this.hitAreas.strengthDec.bounds = leftArrow;
        this.hitAreas.strengthVal.bounds = text;
        this.hitAreas.strengthInc.bounds = rightArrow;
        this.hitAreas.strengthAny.bounds = [leftArrow[0], rightArrow[0] + rightArrow[1] - leftArrow[0]];
        rposX = leftArrow[0] - innerMargin;
        
        const loraWidth = rposX - posX;
        ctx.textAlign = "left";
        ctx.textBaseline = "middle";
        const loraLabel = String(this.value.lora || "None");
        ctx.fillText(fitString(ctx, loraLabel, loraWidth), posX, midY);
        this.hitAreas.lora.bounds = [posX, loraWidth];
        posX += loraWidth + innerMargin;
        
        ctx.globalAlpha = app.canvas.editor_alpha;
        ctx.restore();
    }

    serializeValue(node, index) {
        console.log(`[LoadLoraBatch] 序列化widget: ${this.name}, 值:`, this.value);
        return { ...this.value };
    }

    onToggleDown(event, pos, node) {
        this.value.on = !this.value.on;
        this.cancelMouseDown();
        return true;
    }

    onLoraClick(event, pos, node) {
        // 传入节点和控件信息以实现正确的位置定位
        showLoraChooser(rgthree.lastCanvasMouseEvent || event, (value) => {
            if (typeof value === "string") {
                this.value.lora = value;
            }
            node.setDirtyCanvas(true, true);
        }, null, null, node, this);
        this.cancelMouseDown();
    }

    onStrengthDecDown(event, pos, node) {
        this.stepStrength(-1);
    }

    onStrengthIncDown(event, pos, node) {
        this.stepStrength(1);
    }

    onStrengthAnyMove(event, pos, node) {
        if (event.deltaX) {
            this.haveMouseMovedStrength = true;
            this.value.strength = (this.value.strength || 1) + event.deltaX * 0.05;
        }
    }

    onStrengthValUp(event, pos, node) {
        if (this.haveMouseMovedStrength) {
            this.haveMouseMovedStrength = false;
            return;
        }
        
        const canvas = app.canvas;
        canvas.prompt("强度值", this.value.strength || 1, (v) => {
            this.value.strength = Number(v);
        }, event);
    }

    stepStrength(direction) {
        let step = 0.05;
        let strength = (this.value.strength || 1) + step * direction;
        this.value.strength = Math.round(strength * 100) / 100;
    }


}

// 注册扩展
app.registerExtension({
    name: "A_my_nodes.LoadLoraBatch.UI",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "LoadLoraBatch") return;
        console.log("[LoadLoraBatch] UI扩展注册, 节点名:", nodeData.name);

        // 覆盖节点类
        const origOnNodeCreated = nodeType.prototype.onNodeCreated;
        const origConfigure = nodeType.prototype.configure;
        
        nodeType.prototype.onNodeCreated = function() {
            console.log("[LoadLoraBatch] 节点UI初始化", this);
            if (origOnNodeCreated) origOnNodeCreated.apply(this, arguments);
            
            // 创建LoadLoraBatchNode实例并复制方法
            const loraNode = new LoadLoraBatchNode();
            this.addNewLoraWidget = loraNode.addNewLoraWidget.bind(this);
            this.addNonLoraWidgets = loraNode.addNonLoraWidgets.bind(this);
            this.getSlotInPosition = loraNode.getSlotInPosition.bind(this);
            this.getSlotMenuOptions = loraNode.getSlotMenuOptions.bind(this);
            this.hasLoraWidgets = loraNode.hasLoraWidgets.bind(this);
            this.configure = loraNode.configure.bind(this);
            
            // 初始化
            this.loraWidgetsCounter = 0;
            this.widgetButtonSpacer = null;
            this.serialize_widgets = true;
            
            this.addNonLoraWidgets();
            const computed = this.computeSize();
            this.size = this.size || [0, 0];
            this.size[0] = Math.max(this.size[0], computed[0]);
            this.size[1] = Math.max(this.size[1], computed[1]);
            this.setDirtyCanvas(true, true);
        };
    }
});
