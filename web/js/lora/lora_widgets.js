// LoRA控件公共库
// 包含LoRA相关的通用控件类和工具函数

import { app } from "../../../../scripts/app.js";
import { RgthreeBaseWidget } from "../utils/utils_widgets.js";
import { showTopNotification, moveArrayItem, removeArrayItem } from "../utils/shared_utils.js";
import {
    drawRoundedRectangle,
    drawTogglePart,
    drawNumberWidgetPart,
    isLowQuality,
    fitString,
} from "../utils/utils_canvas.js";

/**
 * 显示LoRA选择器对话框
 * @param {Event} event - 触发事件
 * @param {Function} callback - 选择回调函数
 * @param {*} filter - 过滤器（未使用）
 * @param {*} opts - 选项（未使用）
 * @param {Object} node - 节点对象
 * @param {Object} widget - 控件对象
 */
export function showLoraChooser(event, callback, filter, opts, node, widget) {
    const input = node.widgets.find((w) => w.name === "lora_name");
    const inputEl = document.createElement("input");
    inputEl.className = "comfy-multiline-input";
    inputEl.value = input.value;
    inputEl.placeholder = "List of lora names, one per line";

    const widgets = node.widgets.filter((w) => w.name && w.name.startsWith("LORA_"));
    if (widgets?.length) {
        inputEl.value = widgets.map((w) => `${w.value.lora}:${w.value.strength}`).join("\n");
    }

    app.ui.dialog.show(inputEl);
    inputEl.focus();

    const onInput = (e) => {
        if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
            e.preventDefault();
            e.stopPropagation();
            app.ui.dialog.close();
            processInput();
        }
    };

    const processInput = () => {
        const lines = inputEl.value.split("\n").map((line) => line.trim()).filter((line) => line);
        
        // 清除现有的LoRA控件
        node.widgets = node.widgets.filter((w) => !w.name || !w.name.startsWith("LORA_"));
        
        // 添加新的LoRA控件
        lines.forEach((line) => {
            const parts = line.split(":");
            const loraName = parts[0]?.trim();
            const strength = parts[1] ? parseFloat(parts[1].trim()) : 1.0;
            
            if (loraName) {
                node.addNewLoraWidget({ lora: loraName, strength: strength, on: true });
            }
        });
        
        node.setDirtyCanvas(true, true);
        if (callback) callback(inputEl.value);
    };

    inputEl.addEventListener("keydown", onInput);
}

/**
 * 获取LoRA控件的右键菜单选项
 * @param {Object} slot - 插槽对象
 * @param {Event} event - 事件对象
 * @param {Object} node - 节点对象
 * @returns {undefined|null} 菜单选项或null
 */
export function getLoraSlotMenuOptions(slot, event, node) {
    if (slot && slot.widget && slot.widget.name && slot.widget.name.startsWith("LORA_")) {
        const widget = slot.widget;
        const index = node.widgets.indexOf(widget);
        const canMoveUp = !!(node.widgets[index - 1]?.name?.startsWith("LORA_"));
        const canMoveDown = !!(node.widgets[index + 1]?.name?.startsWith("LORA_"));
        
        const menuItems = [
            {
                content: `${widget.value.on ? "⚫" : "🟢"} 切换 ${widget.value.on ? "关闭" : "开启"}`,
                callback: () => {
                    widget.value.on = !widget.value.on;
                },
            },
            {
                content: `📋 复制模型路径`,
                disabled: !widget.value.lora || widget.value.lora === "None",
                callback: () => {
                    if (widget.value.lora && widget.value.lora !== "None") {
                        // 复制LoRA模型路径到剪贴板
                        navigator.clipboard.writeText(widget.value.lora).then(() => {
                            console.log(`[LoraWidgets] 已复制模型路径: ${widget.value.lora}`);
                            // 显示顶部提示信息
                            showTopNotification(`已复制模型路径: ${widget.value.lora}`, 'success');
                        }).catch(err => {
                            console.error('[LoraWidgets] 复制失败:', err);
                            // 降级方案：使用旧的复制方法
                            try {
                                const textArea = document.createElement('textarea');
                                textArea.value = widget.value.lora;
                                document.body.appendChild(textArea);
                                textArea.select();
                                document.execCommand('copy');
                                document.body.removeChild(textArea);
                                console.log(`[LoraWidgets] 已复制模型路径(降级): ${widget.value.lora}`);
                                showTopNotification(`已复制模型路径: ${widget.value.lora}`, 'success');
                            } catch (fallbackErr) {
                                console.error('[LoraWidgets] 降级复制也失败:', fallbackErr);
                                showTopNotification('复制失败，请手动复制', 'error');
                            }
                        });
                    }
                },
            },
            {
                content: `⬆️ 上移`,
                disabled: !canMoveUp,
                callback: () => {
                    moveArrayItem(node.widgets, widget, index - 1);
                },
            },
            {
                content: `⬇️ 下移`,
                disabled: !canMoveDown,
                callback: () => {
                    moveArrayItem(node.widgets, widget, index + 1);
                },
            },
            {
                content: `🗑️ 删除`,
                callback: () => {
                    removeArrayItem(node.widgets, widget);
                },
            },
            {
                content: `🗑️ 清空所有LoRA`,
                callback: () => {
                    // 移除所有LoRA控件
                    node.widgets = node.widgets.filter(widget => !widget.name || !widget.name.startsWith("LORA_"));
                    node.setDirtyCanvas(true, true);
                },
            },
        ];
        
        // 直接使用LiteGraph.ContextMenu，参考rgthree官方实现
        const menu = new LiteGraph.ContextMenu(menuItems, {
            title: "LORA WIDGET",
            event: rgthree.lastCanvasMouseEvent,
        });
        return undefined;
    }
    return null;
}

/**
 * LoRA控件基类
 * 包含LoRA控件的通用绘制和交互逻辑
 */
export class BaseLoraWidget extends RgthreeBaseWidget {
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
        this.hitAreas.strengthAny.bounds = [leftArrow[0], posY, rightArrow[0] + rightArrow[1] - leftArrow[0], height];
        rposX = leftArrow[0] - innerMargin;
        
        const loraWidth = rposX - posX;
        ctx.textAlign = "left";
        ctx.textBaseline = "middle";
        const loraLabel = String(this.value.lora || "None");
        ctx.fillText(fitString(ctx, loraLabel, loraWidth), posX, midY);
        this.hitAreas.lora.bounds = [posX, posY, loraWidth, height];
        posX += loraWidth + innerMargin;
        
        ctx.globalAlpha = app.canvas.editor_alpha;
        ctx.restore();
    }

    serializeValue(node, index) {
        console.log(`[LoraWidgets] 序列化widget: ${this.name}, 值:`, this.value);
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