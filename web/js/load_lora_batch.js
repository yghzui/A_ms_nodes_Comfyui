import { app } from "../../../scripts/app.js";

/**
 * LoadLoraBatch节点的前端扩展
 * 实现在节点上直接添加LoRA控件,使用与模板相同的原理
 */
app.registerExtension({
    name: "A_my_nodes.LoadLoraBatch.UI",
	async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "LoadLoraBatch") return;
        
        console.log("[LoadLoraBatch] 初始化节点UI");
        
        // 获取所有可用的LoRA列表
        let loras = ["None"];
        try {
            // 从节点定义中获取LoRA列表
            if (nodeData.input && nodeData.input.optional && nodeData.input.optional.lora_name_0) {
                loras = nodeData.input.optional.lora_name_0[0];
                console.log("[LoadLoraBatch] 从模板获取到LoRA列表,数量:", loras.length);
            } else if (app.canvas && folder_paths && folder_paths.loras) {
                // 备用方案:尝试从folder_paths获取LoRA列表
                loras = ["None", ...folder_paths.loras];
                console.log("[LoadLoraBatch] 从folder_paths获取到LoRA列表,数量:", loras.length);
            } else {
                console.warn("[LoadLoraBatch] 无法获取LoRA列表,使用默认值");
            }
        } catch (error) {
            console.error("[LoadLoraBatch] 获取LoRA列表出错:", error);
        }
        
        // 重写computeSize方法,确保节点大小正确
        const computeSize = nodeType.prototype.computeSize;
        nodeType.prototype.computeSize = function() {
            if (computeSize) {
                const size = computeSize.apply(this, arguments);
                // 确保节点有足够的高度显示所有控件
                // 安全检查this.loraWidgets是否存在
                const widgetsCount = this.loraWidgets ? this.loraWidgets.length : 0;
                const minHeight = 100 + Math.ceil(widgetsCount / 3) * 40;
                size[1] = Math.max(size[1], minHeight);
                return size;
            }
            return [240, 120];
        };
        
        // 当节点创建时初始化UI
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function() {
            if (onNodeCreated) {
                onNodeCreated.apply(this, arguments);
            }
            
            // 初始化节点
            this.loraWidgets = [];
            this.nextIndex = 1; // 从1开始命名
            this.serialize_widgets = true; // 确保控件值能被保存
            
            // 添加控制按钮
            this.addControlButtons();
            
            // 添加第一个LoRA选项
            this.addLoraOption(1);
            
            // 恢复保存的数据(如果有)
            setTimeout(() => {
                this.loadSavedData();
            }, 100);
        };
        
        // 添加控制按钮
        nodeType.prototype.addControlButtons = function() {
            const addButton = this.addWidget("button", "增加LoRA", null, () => {
                // 查找最后一个LoRA选项的索引,并加1
                let lastIndex = 0;
                if (this.loraWidgets && this.loraWidgets.length > 0) {
                    const lastWidget = this.loraWidgets[this.loraWidgets.length - 1];
                    const match = lastWidget.name.match(/_(\d+)$/);
                    if (match) {
                        lastIndex = parseInt(match[1]);
                    }
                }
                
                // 添加新的LoRA选项
                this.addLoraOption(lastIndex + 1);
                this.updateLoraData();
            });
            
            const cleanButton = this.addWidget("button", "清除多余", null, () => {
                this.cleanUnusedLoras();
                this.updateLoraData();
            });
        };
        
        // 添加一个LoRA选项
        nodeType.prototype.addLoraOption = function(index) {
            // 确保loraWidgets已初始化
            if (!this.loraWidgets) {
                this.loraWidgets = [];
            }
            
            // 创建启用开关
            const enabledWidget = this.addWidget("toggle", `enabled_${index}`, true, () => {
                this.updateLoraData();
            }, { on: "启用", off: "禁用" });
            
            // 创建LoRA选择下拉框
            const nameWidget = this.addWidget("combo", `lora_name_${index}`, "None", () => {
                this.updateLoraData();
            }, { values: loras });
            
            // 创建强度输入框
            const strengthWidget = this.addWidget("number", `strength_model_${index}`, 1.0, () => {
                this.updateLoraData();
            }, { min: -10.0, max: 10.0, step: 0.01, precision: 2 });
            
            // 记录这个LoRA选项
            this.loraWidgets.push(enabledWidget, nameWidget, strengthWidget);
            
            // 更新节点大小
            this.setSize(this.computeSize());
        };
        
        // 清理未使用的LoRA
        nodeType.prototype.cleanUnusedLoras = function() {
            // 确保loraWidgets已初始化
            if (!this.loraWidgets) {
                this.loraWidgets = [];
                return;
            }
            
            // 按索引分组
            const groups = {};
            for (const widget of this.loraWidgets) {
                const match = widget.name.match(/^(enabled|lora_name|strength_model)_(\d+)$/);
                if (match) {
                    const type = match[1];
                    const index = match[2];
                    if (!groups[index]) {
                        groups[index] = {};
                    }
                    groups[index][type] = widget;
                }
            }
            
            // 找出所有值为"None"的LoRA
            const toRemove = [];
            for (const index in groups) {
                const group = groups[index];
                if (group.lora_name && group.lora_name.value === "None") {
                    toRemove.push(index);
                }
            }
            
            // 如果所有LoRA都是"None",至少保留一个
            if (toRemove.length === Object.keys(groups).length && toRemove.length > 0) {
                toRemove.pop();
            }
            
            // 移除未使用的LoRA控件
            for (const index of toRemove) {
                const group = groups[index];
                for (const type in group) {
                    const widget = group[type];
                    const widgetIndex = this.widgets.indexOf(widget);
                    if (widgetIndex !== -1) {
                        this.widgets.splice(widgetIndex, 1);
                        const loraWidgetIndex = this.loraWidgets.indexOf(widget);
                        if (loraWidgetIndex !== -1) {
                            this.loraWidgets.splice(loraWidgetIndex, 1);
                        }
                    }
                }
            }
            
            // 更新节点大小
            this.setSize(this.computeSize());
        };
        
        // 更新LoRA数据
        nodeType.prototype.updateLoraData = function() {
            // 确保loraWidgets已初始化
            if (!this.loraWidgets) {
                this.loraWidgets = [];
                return;
            }
            
            // 按索引分组
            const groups = {};
            for (const widget of this.loraWidgets) {
                const match = widget.name.match(/^(enabled|lora_name|strength_model)_(\d+)$/);
                if (match) {
                    const type = match[1];
                    const index = match[2];
                    if (!groups[index]) {
                        groups[index] = {};
                    }
                    groups[index][type] = widget;
                }
            }
            
            // 收集所有LoRA的数据
            const loraData = [];
            for (const index in groups) {
                const group = groups[index];
                if (group.enabled && group.lora_name && group.strength_model) {
                    loraData.push({
                        enabled: group.enabled.value,
                        lora_name: group.lora_name.value,
                        strength_model: parseFloat(group.strength_model.value)
                    });
                }
            }
            
            // 更新隐藏输入
            const loraJson = JSON.stringify(loraData);
            let jsonWidget = this.widgets.find(w => w.name === "lora_json");
            if (!jsonWidget) {
                jsonWidget = this.addWidget("text", "lora_json", loraJson, () => {}, { multiline: true });
                jsonWidget.inputEl.style.display = "none"; // 隐藏输入框
            } else {
                jsonWidget.value = loraJson;
            }
            
            // 触发画布更新
            app.graph.setDirtyCanvas(true);
        };
        
        // 加载保存的数据
        nodeType.prototype.loadSavedData = function() {
            // 确保loraWidgets已初始化
            if (!this.loraWidgets) {
                this.loraWidgets = [];
            }
            
            // 查找保存的数据
            const jsonWidget = this.widgets.find(w => w.name === "lora_json");
            if (!jsonWidget || !jsonWidget.value) {
                return;
            }
            
            try {
                // 解析保存的数据
                const savedData = JSON.parse(jsonWidget.value);
                if (!Array.isArray(savedData)) {
                    throw new Error("无效的数据格式");
                }
                
                // 清除默认添加的第一个LoRA选项
                while (this.loraWidgets.length > 0) {
                    const widget = this.loraWidgets[0];
                    const widgetIndex = this.widgets.indexOf(widget);
                    if (widgetIndex !== -1) {
                        this.widgets.splice(widgetIndex, 1);
                    }
                    this.loraWidgets.shift();
                }
                
                // 恢复所有LoRA
                for (let i = 0; i < savedData.length; i++) {
                    const data = savedData[i];
                    const index = i + 1; // 从索引1开始
                    
                    // 添加控件
                    this.addLoraOption(index);
                    
                    // 设置值
                    const enabledWidget = this.widgets.find(w => w.name === `enabled_${index}`);
                    const nameWidget = this.widgets.find(w => w.name === `lora_name_${index}`);
                    const strengthWidget = this.widgets.find(w => w.name === `strength_model_${index}`);
                    
                    if (enabledWidget) enabledWidget.value = data.enabled !== undefined ? data.enabled : true;
                    if (nameWidget) nameWidget.value = data.lora_name || "None";
                    if (strengthWidget) strengthWidget.value = data.strength_model !== undefined ? data.strength_model : 1.0;
                }
                
                // 如果没有数据,添加一个默认的LoRA
                if (savedData.length === 0) {
                    this.addLoraOption(1);
                }
                
            } catch (error) {
                console.error("[LoadLoraBatch] 加载保存数据失败:", error);
            }
            
            // 更新LoRA数据
            this.updateLoraData();
        };
    }
}); 