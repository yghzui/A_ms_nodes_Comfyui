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