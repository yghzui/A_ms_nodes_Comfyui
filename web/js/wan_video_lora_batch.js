import { app } from "../../../scripts/app.js";
import { drawNumberWidgetPart, drawRoundedRectangle, drawTogglePart, fitString, isLowQuality, } from "./utils_canvas.js";
import { RgthreeBaseWidget, RgthreeBetterButtonWidget, RgthreeDividerWidget, } from "./utils_widgets.js";
import { rgthreeApi } from "./rgthree_api.js";
import { moveArrayItem, removeArrayItem } from "./shared_utils.js";
import { rgthree } from "./rgthree.js";

console.log("Patching node: wan_video_lora_batch.js");
console.log("Loaded wan_video_lora_batch.js");

// 改进的LoRA选择器，基于官方实现修复位置定位问题
async function showLoraChooser(event, callback, parentMenu, loras, buttonNode, buttonWidget) {
    const canvas = app.canvas;
    if (!loras) {
        loras = ["None", ...(await rgthreeApi.getLoras().then((loras) => loras.map((l) => l.file)))];
    }
    
    // 转换为LiteGraph.ContextMenu需要的格式
    const menuItems = loras.map(lora => ({
        content: lora,
        callback: () => callback(lora)
    }));
    
    // 计算菜单位置，按用户要求：左侧与点击点x轴对齐，顶部以点击点为基准
    let menuEvent = event;
    let targetX, targetY;
    
    if (buttonNode && buttonWidget) {
        // 获取画布的变换信息
        const canvasRect = canvas.canvas.getBoundingClientRect();
        const ds = canvas.ds || { scale: 1, offset: [0, 0] };
        
        // 节点与控件在画布坐标系中的位置
        const nodeX = buttonNode.pos[0];
        const nodeY = buttonNode.pos[1];
        const widgetY = buttonWidget.last_y || 0;
        const widgetLeftMargin = 15; // 按钮绘制时使用的左边距，与utils_widgets中drawWidgetButton一致
        
        // 使用官方公式：client = (canvasPos + offset) * scale + canvasBoundingLeft/Top
        const anchorCanvasX = nodeX + widgetLeftMargin; // 以按钮左侧作为x锚点
        const anchorCanvasY = nodeY + widgetY;          // 以控件顶部作为y锚点（搜索框顶端）
        
        targetX = (anchorCanvasX + ds.offset[0]) * ds.scale + canvasRect.left;
        targetY = (anchorCanvasY + ds.offset[1]) * ds.scale + canvasRect.top;
        
        // 创建标准的MouseEvent对象，确保ContextMenu按该位置定位
        menuEvent = new MouseEvent('contextmenu', {
            clientX: targetX,
            clientY: targetY,
            bubbles: true,
            cancelable: true,
            view: window
        });
    } else if (event && event.clientX !== undefined) {
        // 对于重新选择的情况，直接使用真实鼠标事件的client坐标，避免二次换算误差
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
    
    // 创建上下文菜单，使用官方推荐的选项格式
    const contextMenu = new LiteGraph.ContextMenu(menuItems, {
        event: menuEvent,
        parentMenu: parentMenu || undefined,
        title: "选择LoRA",
        scale: Math.max(1, canvas.ds?.scale || 1),
        className: "dark",
        callback,
    });
    
    // 应用用户要求的位置调整逻辑：左侧与点击点x轴对齐，顶部以点击点为基准
    if (contextMenu && contextMenu.root && targetX !== undefined && targetY !== undefined) {
        requestAnimationFrame(() => {
            const rect = contextMenu.root.getBoundingClientRect();
            const bodyRect = document.body.getBoundingClientRect();
            
            // 设置左侧与点击点x轴对齐
            contextMenu.root.style.left = targetX + 'px';
            
            // 设置顶部以点击点为基准，但如果选项较多超出屏幕底部，则向上调整
            let finalY = targetY;
            if (bodyRect.height && targetY + rect.height > bodyRect.height - 10) {
                // 选项较多时，向上调整，确保搜索框顶部仍以初始点为基准
                finalY = Math.max(10, bodyRect.height - rect.height - 10);
            }
            contextMenu.root.style.top = finalY + 'px';
            
            // 如果菜单超出屏幕右侧，向左调整
            if (bodyRect.width && targetX + rect.width > bodyRect.width - 10) {
                const adjustedX = Math.max(10, bodyRect.width - rect.width - 10);
                contextMenu.root.style.left = adjustedX + 'px';
            }
        });
    }
}

// WanVideoLoraBatchDualToggleWidget控件类 - 用于在同一行显示两个开关
class WanVideoLoraBatchDualToggleWidget extends RgthreeBaseWidget {
    constructor(name, label1, label2, defaultValue1, defaultValue2) {
        super(name);
        this.type = "custom";
        this.label1 = label1;
        this.label2 = label2;
        this.hitAreas = {
            toggle1: { bounds: [0, 0], onDown: this.onToggle1Down.bind(this) },
            toggle2: { bounds: [0, 0], onDown: this.onToggle2Down.bind(this) },
        };
        this._value1 = defaultValue1;
        this._value2 = defaultValue2;
    }

    set value(v) {
        if (typeof v === 'object' && v !== null) {
            this._value1 = v.value1 !== undefined ? v.value1 : this._value1;
            this._value2 = v.value2 !== undefined ? v.value2 : this._value2;
        }
    }

    get value() {
        return {
            value1: this._value1,
            value2: this._value2
        };
    }

    draw(ctx, node, w, posY, height) {
        ctx.save();
        const margin = 10;
        const innerMargin = margin * 0.33;
        const lowQuality = isLowQuality();
        const midY = posY + height * 0.5;
        let posX = margin;
        
        drawRoundedRectangle(ctx, { pos: [posX, posY], size: [node.size[0] - margin * 2, height] });
        
        // 第一个开关
        this.hitAreas.toggle1.bounds = drawTogglePart(ctx, { posX, posY, height, value: this._value1 });
        posX += this.hitAreas.toggle1.bounds[1] + innerMargin;
        
        if (lowQuality) {
            ctx.restore();
            return;
        }
        
        // 第一个标签
        ctx.fillStyle = LiteGraph.WIDGET_TEXT_COLOR;
        ctx.textAlign = "left";
        ctx.textBaseline = "middle";
        ctx.fillText(this.label1, posX, midY);
        posX += ctx.measureText(this.label1).width + innerMargin * 2;
        
        // 第二个开关
        this.hitAreas.toggle2.bounds = drawTogglePart(ctx, { posX, posY, height, value: this._value2 });
        posX += this.hitAreas.toggle2.bounds[1] + innerMargin;
        
        // 第二个标签
        ctx.fillText(this.label2, posX, midY);
        
        ctx.restore();
    }

    serializeValue(node, index) {
        console.log(`[WanVideoLoraBatch] 序列化dual toggle widget: ${this.name}, 值:`, this.value);
        return this.value;
    }

    onToggle1Down(event, pos, node) {
        this._value1 = !this._value1;
        this.cancelMouseDown();
        return true;
    }

    onToggle2Down(event, pos, node) {
        this._value2 = !this._value2;
        this.cancelMouseDown();
        return true;
    }
}

// WanVideoLoraBatchToggleWidget控件类 - 用于low_mem_load和merge_loras
class WanVideoLoraBatchToggleWidget extends RgthreeBaseWidget {
    constructor(name, label, defaultValue) {
        super(name);
        this.type = "custom";
        this.label = label;
        this.hitAreas = {
            toggle: { bounds: [0, 0], onDown: this.onToggleDown.bind(this) },
        };
        this._value = defaultValue;
    }

    set value(v) {
        this._value = v;
    }

    get value() {
        return this._value;
    }

    draw(ctx, node, w, posY, height) {
        ctx.save();
        const margin = 10;
        const innerMargin = margin * 0.33;
        const lowQuality = isLowQuality();
        const midY = posY + height * 0.5;
        let posX = margin;
        
        drawRoundedRectangle(ctx, { pos: [posX, posY], size: [node.size[0] - margin * 2, height] });
        this.hitAreas.toggle.bounds = drawTogglePart(ctx, { posX, posY, height, value: this.value });
        posX += this.hitAreas.toggle.bounds[1] + innerMargin;
        
        if (lowQuality) {
            ctx.restore();
            return;
        }
        
        ctx.fillStyle = LiteGraph.WIDGET_TEXT_COLOR;
        ctx.textAlign = "left";
        ctx.textBaseline = "middle";
        ctx.fillText(this.label, posX, midY);
        
        ctx.restore();
    }

    serializeValue(node, index) {
        console.log(`[WanVideoLoraBatch] 序列化toggle widget: ${this.name}, 值:`, this.value);
        return this.value;
    }

    onToggleDown(event, pos, node) {
        this.value = !this.value;
        this.cancelMouseDown();
        return true;
    }
}

// WanVideoLoraBatch节点类
class WanVideoLoraBatchNode extends LGraphNode {
    constructor(title = "WanVideoLoraBatch") {
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
        console.log("[WanVideoLoraBatch] 配置节点", info);
        console.log("[WanVideoLoraBatch] widgets_values:", info.widgets_values);
        
        // 清除现有的widgets
        if (this.widgets) {
            this.widgets.length = 0;
        }
        this.widgetButtonSpacer = null;
        
        // 调用父类的configure
        super.configure(info);
        
        // 保存临时尺寸
        this._tempWidth = this.size[0];
        this._tempHeight = this.size[1];
        
        // 先添加非LoRA控件（设置控件等）
        this.addNonLoraWidgets();
        
        // 恢复LoRA控件
        for (const widgetValue of info.widgets_values || []) {
            if (widgetValue && typeof widgetValue === 'object' && widgetValue.lora !== undefined) {
                console.log("[WanVideoLoraBatch] 恢复LoRA控件:", widgetValue);
                const widget = this.addNewLoraWidget();
                widget.value = { ...widgetValue };
            }
        }
        
        // 恢复设置控件的值
        const settingsWidget = this.widgets.find(w => w.name === "settings");
        if (settingsWidget) {
            const settingsValue = info.widgets_values?.find(w => w && typeof w === 'object' && w.value1 !== undefined && w.value2 !== undefined);
            if (settingsValue) {
                console.log("[WanVideoLoraBatch] 恢复设置控件:", settingsValue);
                settingsWidget.value = settingsValue;
            }
        }
        
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
        const widget = this.addCustomWidget(new WanVideoLoraBatchWidget("LORA_" + this.loraWidgetsCounter));
        if (lora) {
            widget.setLora(lora);
        }
        
        // 找到设置控件的位置，将新LoRA插入到设置控件之后
        const settingsWidgetIndex = this.widgets.findIndex(w => w.name === "settings");
        if (settingsWidgetIndex !== -1) {
            // 将新LoRA移动到设置控件之后
            moveArrayItem(this.widgets, widget, settingsWidgetIndex + 1);
        }
        
        return widget;
    }

    addNonLoraWidgets() {
        // 确保widgets数组存在
        if (!this.widgets) {
            this.widgets = [];
        }
        
        // 添加双开关控件（low_mem_load和merge_loras）
        this.addCustomWidget(new WanVideoLoraBatchDualToggleWidget("settings", "低内存加载", "合并LoRA", false, true));
        
        // 添加分隔线
        const divider1 = this.addCustomWidget(new RgthreeDividerWidget({ marginTop: 4, marginBottom: 0, thickness: 0 }));
        
        // 添加按钮分隔线
        this.widgetButtonSpacer = this.addCustomWidget(new RgthreeDividerWidget({ marginTop: 4, marginBottom: 0, thickness: 0 }));
        
        // 添加增加LoRA按钮
        const addButton = new RgthreeBetterButtonWidget("➕ 增加LoRA", (event, pos, node) => {
            // 传入节点和按钮控件信息以实现正确的位置定位
            showLoraChooser(rgthree.lastCanvasMouseEvent || event, (value) => {
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
        });
        addButton.serializeValue = () => undefined; // 阻止按钮被序列化
        this.addCustomWidget(addButton);
    }

    getSlotInPosition(canvasX, canvasY) {
        let lastWidget = null;
        for (const widget of this.widgets) {
            if (!widget.last_y) return;
            if (canvasY > this.pos[1] + widget.last_y) {
                lastWidget = widget;
                continue;
            }
            break;
        }
        if (lastWidget && lastWidget.name && lastWidget.name.startsWith("LORA_")) {
            return { widget: lastWidget, output: { type: "LORA WIDGET" } };
        }
        return null;
    }

    getSlotMenuOptions(slot) {
        if (slot && slot.widget && slot.widget.name && slot.widget.name.startsWith("LORA_")) {
            const widget = slot.widget;
            const index = this.widgets.indexOf(widget);
            const canMoveUp = !!(this.widgets[index - 1]?.name?.startsWith("LORA_"));
            const canMoveDown = !!(this.widgets[index + 1]?.name?.startsWith("LORA_"));
            
            const menuItems = [
                {
                    content: `${widget.value.on ? "⚫" : "🟢"} 切换 ${widget.value.on ? "关闭" : "开启"}`,
                    callback: () => {
                        widget.value.on = !widget.value.on;
                    },
                },
                {
                    content: `⬆️ 上移`,
                    disabled: !canMoveUp,
                    callback: () => {
                        moveArrayItem(this.widgets, widget, index - 1);
                    },
                },
                {
                    content: `⬇️ 下移`,
                    disabled: !canMoveDown,
                    callback: () => {
                        moveArrayItem(this.widgets, widget, index + 1);
                    },
                },
                {
                    content: `🗑️ 删除`,
                    callback: () => {
                        removeArrayItem(this.widgets, widget);
                    },
                },
                {
                    content: `🗑️ 清空所有LoRA`,
                    callback: () => {
                        // 移除所有LoRA控件
                        this.widgets = this.widgets.filter(widget => !widget.name || !widget.name.startsWith("LORA_"));
                        this.setDirtyCanvas(true, true);
                    },
                },
            ];
            
            // 计算正确的右键菜单位置，确保显示在鼠标附近
            const canvas = app.canvas;
            const canvasRect = canvas.canvas.getBoundingClientRect();
            const transform = canvas.ds || { scale: 1, offset: [0, 0] };
            
            // 获取当前鼠标位置或使用最后记录的位置
            const mouseEvent = rgthree.lastCanvasMouseEvent;
            let menuEvent = mouseEvent;
            
            if (mouseEvent && mouseEvent.clientX !== undefined) {
                // 确保菜单显示在鼠标右侧，避免位置跳动
                menuEvent = {
                    clientX: mouseEvent.clientX + 10, // 向右偏移10像素
                    clientY: mouseEvent.clientY,
                    preventDefault: () => {},
                    stopPropagation: () => {}
                };
            }
            
            new LiteGraph.ContextMenu(menuItems, {
                title: "LORA WIDGET",
                event: menuEvent,
            });
            return undefined;
        }
        return null;
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

// WanVideoLoraBatch控件类
class WanVideoLoraBatchWidget extends RgthreeBaseWidget {
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
        console.log(`[WanVideoLoraBatch] 序列化widget: ${this.name}, 值:`, this.value);
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
    name: "A_my_nodes.WanVideoLoraBatch.UI",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "WanVideoLoraBatch") return;
        console.log("[WanVideoLoraBatch] UI扩展注册, 节点名:", nodeData.name);

        // 覆盖节点类
        const origOnNodeCreated = nodeType.prototype.onNodeCreated;
        const origConfigure = nodeType.prototype.configure;
        
        nodeType.prototype.onNodeCreated = function() {
            console.log("[WanVideoLoraBatch] 节点UI初始化", this);
            if (origOnNodeCreated) origOnNodeCreated.apply(this, arguments);
            
            // 创建WanVideoLoraBatchNode实例并复制方法
            const loraNode = new WanVideoLoraBatchNode();
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