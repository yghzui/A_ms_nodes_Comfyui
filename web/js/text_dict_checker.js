import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "A_my_nodes.TextDictChecker.UI",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "TextDictChecker") return;

        // 1. 在节点创建时添加显示控件
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function() {
            if (onNodeCreated) onNodeCreated.apply(this, arguments);

            // 创建一个只读的 Text Widget 用于显示状态
            // 使用 "display_status" 作为名称
            const w = this.addWidget("text", "Check Status", "Pending...", () => {}, {});
            if (w && w.inputEl) {
                w.inputEl.readOnly = true;
                w.inputEl.style.opacity = 0.6;
                w.inputEl.style.textAlign = "center";
            }
        };

        // 2. 监听执行完成事件，更新控件的值
        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function(message) {
            if (onExecuted) onExecuted.apply(this, arguments);

            // message 对应 Python 返回字典中的 "ui" 字段
            // 现在我们使用 text 字段传递结果列表
            if (message && message.text && message.text.length > 0) {
                const w = this.widgets.find(w => w.name === "Check Status");
                if (w) {
                    const statusText = message.text[0]; // 获取第一个返回值
                    const isEnabled = statusText === "True";
                    
                    w.value = isEnabled ? "✅ Enabled" : "❌ Disabled";
                    
                    if (w.inputEl) {
                        w.inputEl.style.color = isEnabled ? "#4caf50" : "#f44336";
                        w.inputEl.style.fontWeight = "bold";
                    }
                }
            }
        };
    },
});
