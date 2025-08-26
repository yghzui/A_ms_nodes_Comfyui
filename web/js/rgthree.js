import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
export const LogLevel = {
    DEBUG: "debug",
    DEV: "dev",
    INFO: "info",
    WARN: "warn",
    ERROR: "error"
};

export const rgthree = {
    logger: {
        logParts: (level, message) => {
            return [level, [message]];
        }
    },
    // 记录最近一次鼠标事件（任意类型）
    lastCanvasMouseEvent: null,
    // 记录最近一次右键(contextmenu)事件，优先用于定位右键菜单
    lastContextMenuEvent: null,
    invokeExtensionsAsync: async (event, data) => {
        // 简化实现
    }
};

// 重写LGraphCanvas的adjustMouseEvent方法来正确记录画布坐标
if (typeof LGraphCanvas !== 'undefined') {
    const originalAdjustMouseEvent = LGraphCanvas.prototype.adjustMouseEvent;
    LGraphCanvas.prototype.adjustMouseEvent = function(e) {
        // 调用原始方法进行坐标调整
        originalAdjustMouseEvent.apply(this, arguments);
        // 记录调整后的事件
        rgthree.lastCanvasMouseEvent = e;
    };
} else {
    // 如果LGraphCanvas还未加载，延迟执行
    document.addEventListener('DOMContentLoaded', () => {
        if (typeof LGraphCanvas !== 'undefined') {
            const originalAdjustMouseEvent = LGraphCanvas.prototype.adjustMouseEvent;
            LGraphCanvas.prototype.adjustMouseEvent = function(e) {
                originalAdjustMouseEvent.apply(this, arguments);
                rgthree.lastCanvasMouseEvent = e;
            };
        }
    });
}

// 右键菜单事件的备用记录（用于lastContextMenuEvent）
document.addEventListener('contextmenu', (event) => {
    rgthree.lastContextMenuEvent = event;
}, true);