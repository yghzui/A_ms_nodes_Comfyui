import { app } from "../../../scripts/app.js";

/**
 * LoadLoraBatch节点的前端扩展
 * 实现在节点上直接添加LoRA控件,从索引2开始
 */
app.registerExtension({
    name: "A_my_nodes.LoadLoraBatch.UI",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "LoadLoraBatch") return;
        
        console.log("[LoadLoraBatch] 初始化节点UI");
        
        // 从节点定义中获取LoRA列表
        const loras = nodeData.input.required.lora_name[0];
        if (!Array.isArray(loras) || loras.length === 0) {
            console.error("[LoadLoraBatch] 无法获取LoRA列表");
            return;
        }
        
        // 当节点创建时初始化UI
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function() {
            if (onNodeCreated) {
                onNodeCreated.apply(this, arguments);
            }
            
            // 初始化节点
            this.additionalLoraWidgets = [];
            this.loraCounter = 1; // 从1开始,因为0是主模板
            this.serialize_widgets = true; // 确保控件值能被保存
            
            // 添加控制按钮
            this.addControlButtons();
            
            // 恢复保存的数据(如果有)
            setTimeout(() => {
                this.loadSavedData();
            }, 100);
        };
        
        // 添加控制按钮
        nodeType.prototype.addControlButtons = function() {
            const addButton = this.addWidget("button", "增加LoRA", null, () => {
                this.addLoraSet(this.loraCounter + 1);
                this.updateBatchData();
            });
            
            const cleanButton = this.addWidget("button", "清除多余", null, () => {
                this.cleanUnusedLoras();
                this.updateBatchData();
            });
            
            // 设置按钮样式
            if (addButton.options) {
                addButton.options.className = "lora-batch-button";
            }
            if (cleanButton.options) {
                cleanButton.options.className = "lora-batch-button";
            }
        };
        
        // 添加一组LoRA控件
        nodeType.prototype.addLoraSet = function(index) {
            // 索引从2开始,因为1是主模板
            if (index < 2) index = 2;
            this.loraCounter = Math.max(this.loraCounter, index);
            
            const groupDiv = document.createElement("div");
            groupDiv.className = "lora-group";
            groupDiv.dataset.index = index;
            
            // 创建启用开关
            const enabledWidget = this.addWidget("toggle", `enabled_${index}`, true, (value) => {
                this.updateBatchData();
            }, { on: "启用", off: "禁用" });
            
            // 创建LoRA选择下拉框
            const nameWidget = this.addWidget("combo", `lora_name_${index}`, "None", (value) => {
                this.updateBatchData();
            }, { values: loras });
            
            // 创建强度滑块
            const strengthWidget = this.addWidget("slider", `strength_model_${index}`, 1.0, (value) => {
                this.updateBatchData();
            }, { min: -10.0, max: 10.0, step: 0.01 });
            
            // 记录这组控件
            this.additionalLoraWidgets.push({
                index: index,
                widgets: [enabledWidget, nameWidget, strengthWidget]
            });
            
            // 更新节点大小
            this.setSize(this.computeSize());
        };
        
        // 清理未使用的LoRA
        nodeType.prototype.cleanUnusedLoras = function() {
            // 找出所有值为"None"的LoRA
            const toRemove = [];
            for (const group of this.additionalLoraWidgets) {
                const nameWidget = group.widgets.find(w => w.name === `lora_name_${group.index}`);
                if (nameWidget && nameWidget.value === "None") {
                    toRemove.push(group);
                }
            }
            
            // 如果所有LoRA都是"None",至少保留一个
            if (toRemove.length === this.additionalLoraWidgets.length && toRemove.length > 0) {
                toRemove.pop(); // 移除最后一个元素,保留一个
            }
            
            // 移除未使用的LoRA控件
            for (const group of toRemove) {
                this.removeLoraSet(group.index);
            }
            
            // 更新节点大小
            this.setSize(this.computeSize());
        };
        
        // 移除一组LoRA控件
        nodeType.prototype.removeLoraSet = function(index) {
            const groupIndex = this.additionalLoraWidgets.findIndex(g => g.index === index);
            if (groupIndex === -1) return;
            
            // 移除控件
            const group = this.additionalLoraWidgets[groupIndex];
            for (const widget of group.widgets) {
                const widgetIndex = this.widgets.indexOf(widget);
                if (widgetIndex !== -1) {
                    this.widgets.splice(widgetIndex, 1);
                }
            }
            
            // 从列表中移除
            this.additionalLoraWidgets.splice(groupIndex, 1);
        };
        
        // 更新批量数据
        nodeType.prototype.updateBatchData = function() {
            // 收集所有额外LoRA的数据
            const batchData = [];
            for (const group of this.additionalLoraWidgets) {
                const enabledWidget = group.widgets.find(w => w.name === `enabled_${group.index}`);
                const nameWidget = group.widgets.find(w => w.name === `lora_name_${group.index}`);
                const strengthWidget = group.widgets.find(w => w.name === `strength_model_${group.index}`);
                
                if (enabledWidget && nameWidget && strengthWidget) {
                    batchData.push({
                        enabled: enabledWidget.value,
                        lora_name: nameWidget.value,
                        strength_model: strengthWidget.value
                    });
                }
            }
            
            // 更新隐藏输入
            const batchDataJson = JSON.stringify(batchData);
            const batchInput = this.widgets.find(w => w.name === "lora_batch_data");
            if (!batchInput) {
                // 如果不存在则创建
                const newWidget = this.addWidget("text", "lora_batch_data", batchDataJson, () => {}, { multiline: true });
                newWidget.inputEl.style.display = "none"; // 隐藏输入框
            } else {
                batchInput.value = batchDataJson;
            }
            
            // 触发画布更新
            app.graph.setDirtyCanvas(true);
        };
        
        // 加载保存的数据
        nodeType.prototype.loadSavedData = function() {
            // 查找保存的数据
            const batchInput = this.widgets.find(w => w.name === "lora_batch_data");
            if (!batchInput || !batchInput.value) {
                // 没有保存的数据,添加一个默认的LoRA
                this.addLoraSet(2);
                return;
            }
            
            try {
                // 解析保存的数据
                const savedData = JSON.parse(batchInput.value);
                if (!Array.isArray(savedData)) {
                    throw new Error("无效的数据格式");
                }
                
                // 恢复所有LoRA
                for (let i = 0; i < savedData.length; i++) {
                    const data = savedData[i];
                    const index = i + 2; // 从索引2开始
                    
                    // 添加控件
                    this.addLoraSet(index);
                    
                    // 设置值
                    const group = this.additionalLoraWidgets.find(g => g.index === index);
                    if (group) {
                        const enabledWidget = group.widgets.find(w => w.name === `enabled_${index}`);
                        const nameWidget = group.widgets.find(w => w.name === `lora_name_${index}`);
                        const strengthWidget = group.widgets.find(w => w.name === `strength_model_${index}`);
                        
                        if (enabledWidget) enabledWidget.value = data.enabled !== undefined ? data.enabled : true;
                        if (nameWidget) nameWidget.value = data.lora_name || "None";
                        if (strengthWidget) strengthWidget.value = data.strength_model !== undefined ? data.strength_model : 1.0;
                    }
                }
                
                // 如果没有数据,添加一个默认的LoRA
                if (savedData.length === 0) {
                    this.addLoraSet(2);
                }
                
            } catch (error) {
                console.error("[LoadLoraBatch] 加载保存数据失败:", error);
                // 出错时添加一个默认的LoRA
                this.addLoraSet(2);
            }
            
            // 更新批量数据
            this.updateBatchData();
        };
        
        // 重写computeSize方法,确保节点大小正确
        const computeSize = nodeType.prototype.computeSize;
        nodeType.prototype.computeSize = function() {
            if (computeSize) {
                const size = computeSize.apply(this, arguments);
                // 确保节点有足够的高度显示所有控件
                const minHeight = 120 + (this.additionalLoraWidgets?.length || 0) * 80;
                size[1] = Math.max(size[1], minHeight);
                return size;
            }
            return [240, 120];
        };
    }
}); 