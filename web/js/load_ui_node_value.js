import { app } from "../../../scripts/app.js";

console.log("Patching node: GetNodeInputValue.js");

// 核心逻辑
function installNodeSelector(node) {
    // 1. 获取原有的 Widget (保持它们为 Text 输入框)
    const targetNodeIdWidget = node.widgets?.find(w => w.name === "target_node_id");
    const targetInputNameWidget = node.widgets?.find(w => w.name === "target_input_name");
    const capturedValueWidget = node.widgets?.find(w => w.name === "captured_value");

    if (!targetNodeIdWidget || !targetInputNameWidget) return;

    // 2. 创建或获取辅助 Combo Widget
    // 命名使用 helper 后缀，标签使用中文提示
    let nodeSelector = node.widgets.find(w => w.name === "select_node_helper");
    if (!nodeSelector) {
        nodeSelector = node.addWidget("combo", "select_node_helper", "", (value) => {
            // 当选择节点时
            const realId = extractId(value);
            // 1. 填入原来的输入框
            if (targetNodeIdWidget) {
                targetNodeIdWidget.value = realId;
            }
            // 2. 更新第二个选择框
            updateInputOptions(realId);
        }, { values: [] });
        nodeSelector.label = "🔍 选择节点 (Select Node)";
    }

    let inputSelector = node.widgets.find(w => w.name === "select_input_helper");
    if (!inputSelector) {
        inputSelector = node.addWidget("combo", "select_input_helper", "", (value) => {
            // 当选择参数时
            // 1. 填入原来的输入框
            if (targetInputNameWidget) {
                targetInputNameWidget.value = value;
            }
            // 2. 触发值获取
            updateCapturedValue();
        }, { values: [] });
        inputSelector.label = "🔍 选择参数 (Select Input)";
    }

    // 3. 调整 Widget 顺序
    // 期望顺序: 
    // 1. select_node_helper
    // 2. target_node_id
    // 3. select_input_helper
    // 4. target_input_name
    // 5. captured_value
    
    const desiredOrder = [
        "select_node_helper",
        "target_node_id",
        "select_input_helper",
        "target_input_name",
        "captured_value"
    ];
    
    node.widgets.sort((a, b) => {
        const ia = desiredOrder.indexOf(a.name);
        const ib = desiredOrder.indexOf(b.name);
        // 如果不在列表里，放到最后
        if (ia === -1) return 1;
        if (ib === -1) return -1;
        return ia - ib;
    });


    // 4. 定义功能函数

    // 提取 ID (从 "10 - NodeName" 格式中)
    const extractId = (val) => {
        if (!val) return "";
        const match = String(val).match(/^(\d+)\s*-?/);
        return match ? match[1] : val;
    };

    // 更新捕获的值
    const updateCapturedValue = () => {
        // 使用实际输入框的值，这样即使用户手动输入也能工作
        const targetNodeId = targetNodeIdWidget.value;
        const selectedInputName = targetInputNameWidget.value;
        
        if (!capturedValueWidget) return;

        const graph = node.graph || app.graph;
        if (!graph) return;

        const targetNode = graph.nodes?.find(n => String(n.id) === targetNodeId);
        
        let capturedValue = "";
        
        if (targetNode && targetNode.widgets) {
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
        
        if (capturedValueWidget.value !== capturedValue) {
            capturedValueWidget.value = capturedValue;
            if (capturedValueWidget.callback) {
                capturedValueWidget.callback(capturedValue);
            }
        }
    };

    // 更新参数选择框的选项
    const updateInputOptions = (targetNodeId) => {
        const graph = node.graph || app.graph;
        if (!graph) return;

        const targetNode = graph.nodes?.find(n => String(n.id) === targetNodeId);
        
        const inputOptions = [];
        if (targetNode && targetNode.widgets) {
            targetNode.widgets.forEach(w => {
                if (w.type !== 'button') {
                    inputOptions.push(w.name);
                }
            });
        }

        inputSelector.options.values = inputOptions;
        
        // 如果当前 helper 的值不在新列表中，清空
        if (inputSelector.value && !inputOptions.includes(inputSelector.value)) {
            inputSelector.value = "";
        }
        
        // 尝试自动更新一下值（针对手动输入 ID 的情况）
        updateCapturedValue();
    };

    // 初始化节点选择框的选项
    const initNodeOptions = () => {
        const graph = node.graph || app.graph;
        if (!graph) return;

        const nodes = graph.nodes || [];
        // 过滤：只保留有非按钮类型 widget 的节点
        const validNodes = nodes.filter(n => {
            // 排除自己（防止循环引用，虽然只是读取值，但逻辑上不应该选自己）
            if (n.id === node.id) return false;
            
            // 检查是否有有效的 widget (非 button)
            if (!n.widgets || n.widgets.length === 0) return false;
            return n.widgets.some(w => w.type !== 'button');
        });

        const options = validNodes.map(n => ({
            id: String(n.id),
            text: `${n.id} - ${n.title || n.type}` 
        }));
        
        options.sort((a, b) => parseInt(a.id) - parseInt(b.id));

        nodeSelector.options.values = options.map(o => o.text);
    };

    // 5. 事件监听

    // 监听手动输入：当用户手动修改 Text Widget 时，也要触发逻辑
    // 实现双向绑定：文本框修改 -> 更新下拉框选中状态
    const originalNodeIdCallback = targetNodeIdWidget.callback;
    targetNodeIdWidget.callback = function(value) {
        if (originalNodeIdCallback) originalNodeIdCallback(value);
        
        // 1. 尝试在下拉框中找到对应的项并选中
        if (nodeSelector && nodeSelector.options && nodeSelector.options.values) {
            // 选项格式可能是 "10 - Title"
            const match = nodeSelector.options.values.find(opt => {
                const optId = extractId(opt);
                return String(optId) === String(value);
            });
            if (match) {
                nodeSelector.value = match;
            } else {
                // 如果找不到匹配项（可能是无效ID），可以清空下拉或保持原样
                // nodeSelector.value = ""; 
            }
        }

        // 2. 更新参数列表
        updateInputOptions(value);
    };

    const originalInputNameCallback = targetInputNameWidget.callback;
    targetInputNameWidget.callback = function(value) {
        if (originalInputNameCallback) originalInputNameCallback(value);
        
        // 1. 同步下拉框
        if (inputSelector && inputSelector.options && inputSelector.options.values) {
            if (inputSelector.options.values.includes(value)) {
                inputSelector.value = value;
            }
        }

        // 2. 触发值获取
        updateCapturedValue();
    };

    // 鼠标进入时刷新列表
    const originalOnMouseEnter = node.onMouseEnter;
    node.onMouseEnter = function(event, pos, canvas) {
        initNodeOptions();
        
        // 确保参数列表也是最新的
        if (targetNodeIdWidget.value) {
            updateInputOptions(targetNodeIdWidget.value);
        }

        if (originalOnMouseEnter) {
            return originalOnMouseEnter.call(this, event, pos, canvas);
        }
    };

    // 初始化执行一次
    initNodeOptions();
    if (targetNodeIdWidget.value) {
        updateInputOptions(targetNodeIdWidget.value);
    }

    // 6. 隐藏原始输入框
    // 它们仍然存在于 node.widgets 中以保证数据被保存，但不可见
    // 我们在最后隐藏它们，以免影响上面的逻辑
    const hideWidget = (w) => {
        if (!w) return;
        w.computeSize = () => [0, -4]; // 尽可能不占用空间
        w.type = "converted-widget";   // 更改类型以防止默认绘制
        w.draw = () => {};             // 空绘制函数
    };

    hideWidget(targetNodeIdWidget);
    hideWidget(targetInputNameWidget);
}

// 注册扩展
app.registerExtension({
    name: "A_my_nodes.GetNodeInputValue.UI",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "GetNodeInputValue") return;

        const origOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function() {
            if (origOnNodeCreated) origOnNodeCreated.apply(this, arguments);
            
            installNodeSelector(this);
            
            // 设置 captured_value 只读
            const capturedWidget = this.widgets?.find(w => w.name === "captured_value");
            if (capturedWidget && capturedWidget.inputEl) {
                capturedWidget.inputEl.readOnly = true;
                capturedWidget.inputEl.style.opacity = 0.8;
            }
        };
        
        // 处理 Reload 后的恢复
        const origOnConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function() {
            if (origOnConfigure) origOnConfigure.apply(this, arguments);
            setTimeout(() => {
                installNodeSelector(this);
            }, 100);
        };
    },
});
