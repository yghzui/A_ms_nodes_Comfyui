// 为ShowResultLast节点添加动态文本显示功能
console.log("Loading ShowResultLast.js");
import { app } from "../../../scripts/app.js";
console.log("Patching node: ShowResultLast1");
import { ComfyWidgets } from "../../../scripts/widgets.js";
console.log("Patching node: ShowResultLast2");

app.registerExtension({
    name: "A_my_nodes.ShowResultLast.UI",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // console.log("检查节点:", nodeData.name);
        if (nodeData.name !== "ShowResultLast") {
            return;
        }
        console.log("注册Patching node: ShowResultLast3");
        
        function populate(textList) {
            if (this.widgets) {
                const pos = this.widgets.findIndex((w) => w.name === "text");
                if (pos !== -1) {
                    for (let i = pos; i < this.widgets.length; i++) {
                        this.widgets[i].onRemove?.();
                    }
                    this.widgets.length = pos;
                }
            }

            // 为列表中的每个元素创建一个文本框，显示全部内容
            for (const item of textList) {
                const w = ComfyWidgets["STRING"](this, "text", ["STRING", { multiline: true }], app).widget;
                w.inputEl.readOnly = true;
                w.inputEl.style.opacity = 0.6;
                w.value = item;
            }

            requestAnimationFrame(() => {
                const sz = this.computeSize();
                if (sz[0] < this.size[0]) {
                    sz[0] = this.size[0];
                }
                if (sz[1] < this.size[1]) {
                    sz[1] = this.size[1];
                }
                this.onResize?.(sz);
                app.graph.setDirtyCanvas(true, false);
            });
        }

        // 监听节点输入数据变化
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated?.apply(this, arguments);
            console.log("ShowResultLast 节点创建完成");
            
            // 监听输入连接变化
            const onConnectionsChange = this.onConnectionsChange;
            this.onConnectionsChange = function (type, index, connected, link_info, output) {
                onConnectionsChange?.apply(this, arguments);
                console.log("连接变化:", type, index, connected, link_info, output);
                
                if (type === "input" && index === 0 && connected) {
                    // 当Filenames输入连接时，尝试获取数据
                    setTimeout(() => {
                        if (this.inputs && this.inputs[0] && this.inputs[0].link) {
                            console.log("检测到Filenames输入连接");
                            // 这里可以尝试获取连接的数据
                        }
                    }, 100);
                }
            };
        };

        // 处理节点执行完成
        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);
            console.log("ShowResultLast onExecuted 被调用，message:", message);
            
            // 处理Python返回的数据
            if (message && message.text) {
                console.log("收到Python数据:", message.text);
                populate.call(this, message.text);
            } else {
                console.log("没有收到数据");
                populate.call(this, ["等待数据..."]);
            }
        };
    }
}); 