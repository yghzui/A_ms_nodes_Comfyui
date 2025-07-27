console.log("Loading custom node: A_my_nodes/web/js/load_lora_batch.js");
import { app } from "../../../scripts/app.js";

// 获取所有lora
const loras = ["None", ...folder_paths.get_filename_list("loras")];

app.registerExtension({
	name: "A_my_nodes.LoadLoraBatch.Dynamic",
	async beforeRegisterNodeDef(nodeType, nodeData) {
		if (nodeData.name === "LoadLoraBatch") {
			console.log(`Patching node: ${nodeData.name} for dynamic inputs`);
			const onConnectionsChange = nodeType.prototype.onConnectionsChange;
			nodeType.prototype.onConnectionsChange = function (
				type,
				index,
				connected,
				link_info
			) {
				if (onConnectionsChange) {
					onConnectionsChange.apply(this, arguments);
				}

				// 查找所有lora相关的输入
				const loraInputs = this.inputs?.filter((i) =>
					i.name.startsWith("lora_name")
				);

				if (!loraInputs) {
					return;
				}

				const lastInput = loraInputs[loraInputs.length - 1];

				// 如果最后一个lora输入被连接，则添加一个新的
				if (
					lastInput &&
					this.getInputLink(this.inputs.indexOf(lastInput)) !== null
				) {
					const nextIndex = loraInputs.length + 1;
					this.addInput(`enabled_${nextIndex}`, "BOOLEAN", {
						default: true,
						label_on: "启用",
						label_off: "禁用",
					});
					this.addInput(`lora_name_${nextIndex}`, "COMBO", {
						values: loras,
						default: "None",
					});
					this.addInput(`strength_model_${nextIndex}`, "FLOAT", {
						default: 1.0,
						min: -10.0,
						max: 10.0,
						step: 0.01,
					});
				}

				// 清理未连接的输入
				this.cleanupLoraInputs();
			};

			nodeType.prototype.cleanupLoraInputs = function () {
				let lastConnectedIndex = -1;
				const loraInputs = this.inputs.filter((i) =>
					i.name.startsWith("lora_name")
				);
				
				// 找到最后一个连接的lora输入的索引
				for (let i = 0; i < loraInputs.length; i++) {
					if (this.getInputLink(this.inputs.indexOf(loraInputs[i])) !== null) {
						lastConnectedIndex = i;
					}
				}

				// 保留一个未连接的输入
				const keepCount = lastConnectedIndex + 2;

				// 从后向前移除多余的输入
				for (let i = loraInputs.length - 1; i >= keepCount; i--) {
					const loraInput = loraInputs[i];
					const enabledInput = this.inputs.find(
						(j) => j.name === loraInput.name.replace("lora_name", "enabled")
					);
					const strengthInput = this.inputs.find(
						(j) => j.name === loraInput.name.replace("lora_name", "strength_model")
					);

					if (enabledInput) {
						this.removeInput(this.inputs.indexOf(enabledInput));
					}
					if (loraInput) {
						this.removeInput(this.inputs.indexOf(loraInput));
					}
					if (strengthInput) {
						this.removeInput(this.inputs.indexOf(strengthInput));
					}
				}
			};
			
			const onNodeCreated = nodeType.prototype.onNodeCreated;
			nodeType.prototype.onNodeCreated = function() {
				if(onNodeCreated) {
					onNodeCreated.apply(this, arguments);
				}
				// 在节点创建后，确保初始状态正确
				setTimeout(() => this.cleanupLoraInputs(), 10);
			};
			
			const getExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
			nodeType.prototype.getExtraMenuOptions = function(_, options) {
				if(getExtraMenuOptions) {
					getExtraMenuOptions.apply(this, arguments);
				}
				options.push({
					content: "清理LoRA输入",
					callback: () => this.cleanupLoraInputs(),
				});
			};
		}
	},
}); 