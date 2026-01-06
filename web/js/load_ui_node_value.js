import { app } from "../../../scripts/app.js";

console.log("Patching node: GetNodeInputValue.js");

// 为节点添加“选择目标节点和输入选项”的下拉菜单
function installNodeSelector(node) {
    // 1. 添加“目标节点”下拉框
    let nodeSelectWidget = node.widgets?.find(w => w.name === "target_node_id");
    if (!nodeSelectWidget) {
        // 如果后端定义了 target_node_id 为 optional，它可能已经存在（如果是 combo 类型？）
        // 后端定义的是 STRING，所以默认会创建一个 text widget。
        // 我们需要把它变成 combo，或者如果它已经是 text，我们可以替换它或者利用它。
        // 但 ComfyUI 前端通常会把 STRING 渲染为 text input。
        // 为了变成下拉框，我们通常是先 remove 原来的 widget，再 add 一个新的 combo widget。
        // 或者如果后端定义是 STRING，我们可以直接覆盖它的类型？
        
        // 检查现有的 widget
        const existingWidget = node.widgets?.find(w => w.name === "target_node_id");
        if (existingWidget) {
            // 如果存在且不是 combo，可能需要替换
            if (existingWidget.type !== "combo") {
                // 移除旧的 text widget
                const index = node.widgets.indexOf(existingWidget);
                if (index > -1) node.widgets.splice(index, 1);
                
                // 添加新的 combo widget
                nodeSelectWidget = node.addWidget("combo", "target_node_id", "", (value) => {
                    updateInputOptions(node, value);
                });
            } else {
                nodeSelectWidget = existingWidget;
                nodeSelectWidget.callback = (value) => {
                    updateInputOptions(node, value);
                };
            }
        } else {
            // 如果不存在（比如是 hidden），新建
            nodeSelectWidget = node.addWidget("combo", "target_node_id", "", (value) => {
                updateInputOptions(node, value);
            });
        }
        nodeSelectWidget.name = "target_node_id";
        nodeSelectWidget.tooltip = "选择要获取值的目标节点";
    }

    // 2. 添加“目标输入选项”下拉框
    let inputSelectWidget = node.widgets?.find(w => w.name === "target_input_name");
    if (!inputSelectWidget) {
        const existingWidget = node.widgets?.find(w => w.name === "target_input_name");
        if (existingWidget) {
             if (existingWidget.type !== "combo") {
                const index = node.widgets.indexOf(existingWidget);
                if (index > -1) node.widgets.splice(index, 1);
                
                inputSelectWidget = node.addWidget("combo", "target_input_name", "", (value) => {
                    updateCapturedValue(node, value);
                });
            } else {
                inputSelectWidget = existingWidget;
                inputSelectWidget.callback = (value) => {
                    updateCapturedValue(node, value);
                };
            }
        } else {
            inputSelectWidget = node.addWidget("combo", "target_input_name", "", (value) => {
                updateCapturedValue(node, value);
            });
        }
        inputSelectWidget.name = "target_input_name";
        inputSelectWidget.tooltip = "选择目标节点的输入选项";
    }

    // 新增：捕获并更新值
    const updateCapturedValue = (node, selectedInputName) => {
        const targetNodeIdWidget = node.widgets?.find(w => w.name === "target_node_id");
        if (!targetNodeIdWidget) return;
        const targetNodeId = targetNodeIdWidget.value;
        
        const graph = node.graph || app.graph;
        const targetNode = graph.nodes?.find(n => String(n.id) === targetNodeId);
        const capturedValueWidget = node.widgets?.find(w => w.name === "captured_value");
        
        if (!capturedValueWidget) return;

        let capturedValue = "";
        
        if (targetNode) {
            // 尝试从 widgets 中查找
            if (targetNode.widgets) {
                const targetWidget = targetNode.widgets.find(w => w.name === selectedInputName);
                if (targetWidget) {
                    const val = targetWidget.value;
                    if (typeof val === 'object' && val !== null) {
                        try {
                            capturedValue = JSON.stringify(val);
                        } catch(e) {
                            capturedValue = String(val);
                        }
                    } else {
                        capturedValue = String(val);
                    }
                }
            }
        }
        
        // 更新显示
        if (capturedValueWidget.value !== capturedValue) {
            capturedValueWidget.value = capturedValue;
            // 如果 captured_value widget 有 callback，可能需要触发它？
            if (capturedValueWidget.callback) {
                capturedValueWidget.callback(capturedValue);
            }
        }
    };

    // 4. 初始化下拉选项
    const initNodeOptions = () => {
        const graph = node.graph || app.graph;
        const nodes = graph.nodes || [];
        const options = nodes.map(n => ({
            value: String(n.id),
            text: `${n.id} - ${n.title || n.type}` 
        }));
        nodeSelectWidget.options.values = options.map(o => o.value);
        nodeSelectWidget.options.labels = options.map(o => o.text);
        
        // 尝试恢复之前的选择并更新值
        if (nodeSelectWidget.value && inputSelectWidget.value) {
            updateCapturedValue(node, inputSelectWidget.value);
        }
    };

    // 5. 更新输入选项的下拉列表
    const updateInputOptions = (node, targetNodeId) => {
        const graph = node.graph || app.graph;
        const targetNode = graph.nodes?.find(n => String(n.id) === targetNodeId);
        if (!targetNode) {
            inputSelectWidget.options.values = [];
            inputSelectWidget.options.labels = [];
            return;
        }

        const inputOptions = [];
        if (targetNode.widgets) {
            targetNode.widgets.forEach(w => {
                // 排除不需要的 widget 类型（如按钮）
                if (w.type !== 'button') {
                    inputOptions.push({
                        value: w.name,
                        text: w.name
                    });
                }
            });
        }
        // inputs 通常不作为 widget 值获取源，除非它们是 converted widgets
        // 这里简化逻辑，只获取 widgets

        inputSelectWidget.options.values = inputOptions.map(o => o.value);
        inputSelectWidget.options.labels = inputOptions.map(o => o.text);
        
        // 如果当前值在新列表中不存在，则重置
        if (!inputOptions.some(o => o.value === inputSelectWidget.value)) {
            inputSelectWidget.value = inputOptions.length > 0 ? inputOptions[0].value : "";
        }
        
        // 更新值
        updateCapturedValue(node, inputSelectWidget.value);
    };

    // 初始化
    initNodeOptions();
    
    // 监听鼠标进入事件，动态更新节点列表
    const originalOnMouseEnter = node.onMouseEnter;
    node.onMouseEnter = function(event, pos, canvas) {
        initNodeOptions();
        // 每次鼠标进入也尝试更新一下值（以防目标节点值变了）
        if (inputSelectWidget.value) {
            updateCapturedValue(node, inputSelectWidget.value);
        }
        
        if (originalOnMouseEnter) {
            return originalOnMouseEnter.call(this, event, pos, canvas);
        }
    };
}


// 注册节点UI扩展
app.registerExtension({
    name: "A_my_nodes.GetNodeInputValue.UI",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "GetNodeInputValue") return;

        // 节点创建时初始化UI
        const origOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function() {
            if (origOnNodeCreated) origOnNodeCreated.apply(this, arguments);
            
            // 找到 captured_value 控件并设为只读
            // 注意：widgets 可能还未完全初始化，或者顺序问题
            // 但通常在 onNodeCreated 后，widgets 应该有了
            
            // 安装节点选择器
            installNodeSelector(this);
            
            // 调整 captured_value 的外观
            const capturedWidget = this.widgets?.find(w => w.name === "captured_value");
            if (capturedWidget) {
                // 如果是 customtext (textarea)，可以通过 inputEl 设置
                if (capturedWidget.inputEl) {
                    capturedWidget.inputEl.readOnly = true;
                    capturedWidget.inputEl.style.opacity = 0.8;
                    capturedWidget.inputEl.style.backgroundColor = "#222";
                }
                // 移到最后，或者指定位置？
                // 默认位置由 backend 定义决定
            }
        };
    },
});
