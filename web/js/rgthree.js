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
    lastCanvasMouseEvent: null,
    invokeExtensionsAsync: async (event, data) => {
        // 简化实现
    }
};

// 监听鼠标事件来更新lastCanvasMouseEvent
document.addEventListener('mousedown', (event) => {
    rgthree.lastCanvasMouseEvent = event;
});

document.addEventListener('mouseup', (event) => {
    rgthree.lastCanvasMouseEvent = event;
});

document.addEventListener('contextmenu', (event) => {
    rgthree.lastCanvasMouseEvent = event;
}); 