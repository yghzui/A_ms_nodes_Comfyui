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

// 使用捕获阶段，确保即便内部 stopPropagation 也能尽量拿到事件
// 左键/中键/右键按下
document.addEventListener('mousedown', (event) => {
    rgthree.lastCanvasMouseEvent = event;
}, true);

// 松开
document.addEventListener('mouseup', (event) => {
    rgthree.lastCanvasMouseEvent = event;
}, true);

// 右键菜单触发：作为右键菜单定位的首选事件
document.addEventListener('contextmenu', (event) => {
    rgthree.lastCanvasMouseEvent = event;
    rgthree.lastContextMenuEvent = event;
}, true);