// 在脚本顶部添加日志，以便在浏览器控制台中确认脚本是否被加载
console.log("Loading custom node: A_my_nodes/web/js/analyze_mask.js");

import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "A_my_nodes.AnalyzeMask.JS",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        // 只对我们的目标节点进行操作
        if (nodeData.name === "AnalyzeMask") {
            console.log(`Patching node: ${nodeData.name}`);

            // 扩展节点的 onNodeCreated 方法
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                if (onNodeCreated) {
                    onNodeCreated.apply(this, arguments);
                }

                // 创建显示结果的容器
                const resultContainer = document.createElement("div");
                Object.assign(resultContainer.style, {
                    backgroundColor: "#1a1a1a",
                    padding: "8px",
                    margin: "5px",
                    borderRadius: "4px",
                    border: "1px solid #444",
                    color: "#fff",
                    fontSize: "12px",
                    fontFamily: "monospace",
                    whiteSpace: "pre-wrap",
                    wordBreak: "break-word",
                    maxHeight: "300px",  // 添加最大高度
                    overflowY: "auto"    // 添加垂直滚动条
                });
                resultContainer.textContent = "等待分析结果...";

                // 添加到节点的 DOM 元素中
                const widget = this.addDOMWidget("analyze_result", "div", resultContainer);
                widget.options.serialize = false;

                // 存储容器引用以便后续更新
                this.resultContainer = resultContainer;
            };

            // 扩展节点的 onExecuted 方法来处理执行结果
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function(message) {
                if (onExecuted) {
                    onExecuted.apply(this, arguments);
                }

                // 检查是否有分析结果
                if (message?.text) {
                    const resultText = message.text[0];
                    if (this.resultContainer) {
                        // 清空容器
                        this.resultContainer.innerHTML = '';
                        
                        // 分割多行结果
                        const lines = resultText.split('\n');
                        
                        // 为每一行创建单独的元素并设置样式
                        lines.forEach(line => {
                            const lineElement = document.createElement('div');
                            lineElement.style.marginBottom = '4px';
                            
                            // 根据结果类型设置不同的样式
                            const isBinary = line.includes("二值型");
                            const isRange = line.includes("范围型");
                            const isError = line.includes("空遮罩") || line.includes("无效");

                            // 设置文本颜色
                            lineElement.style.color = isError ? "#ff6b6b" : 
                                                    isBinary ? "#87ceeb" : 
                                                    isRange ? "#98fb98" : "#fff";

                            lineElement.textContent = line;
                            this.resultContainer.appendChild(lineElement);
                        });
                    }
                }
            };

            // 当节点大小改变时调整容器大小
            const onResize = nodeType.prototype.onResize;
            nodeType.prototype.onResize = function(size) {
                if (onResize) {
                    onResize.apply(this, arguments);
                }

                if (this.resultContainer) {
                    // 确保容器宽度适应节点大小
                    this.resultContainer.style.width = `${size[0] - 20}px`;
                }
            };
        }
    }
}); 