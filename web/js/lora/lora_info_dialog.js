import { LORA_INFO_SERVICE } from "./lora_info_service.js";

// LoRA信息对话框
export class LoraInfoDialog {
    constructor(loraName) {
        this.loraName = loraName;
        this.dialog = null;
        this.dirty = false;
    }

    show() {
        this.createDialog();
        return this.dialog;
    }

    createDialog() {
        // 创建对话框容器
        this.dialog = document.createElement("div");
        this.dialog.className = "lora-info-dialog";
        this.dialog.style.cssText = `
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: #2a2a2a;
            border: 1px solid #666;
            border-radius: 8px;
            padding: 20px;
            min-width: 400px;
            max-width: 600px;
            max-height: 80vh;
            overflow-y: auto;
            z-index: 10000;
            color: white;
            font-family: Arial, sans-serif;
        `;

        // 创建标题
        const title = document.createElement("h3");
        title.textContent = `LoRA信息: ${this.loraName}`;
        title.style.cssText = `
            margin: 0 0 15px 0;
            color: #fff;
            border-bottom: 1px solid #666;
            padding-bottom: 10px;
        `;

        // 创建内容区域
        const content = document.createElement("div");
        content.id = "lora-info-content";
        content.innerHTML = "加载中...";

        // 创建按钮区域
        const buttonArea = document.createElement("div");
        buttonArea.style.cssText = `
            margin-top: 15px;
            text-align: right;
            border-top: 1px solid #666;
            padding-top: 10px;
        `;

        const closeButton = document.createElement("button");
        closeButton.textContent = "关闭";
        closeButton.style.cssText = `
            background: #666;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            cursor: pointer;
            margin-left: 10px;
        `;
        closeButton.onclick = () => this.close();

        const refreshButton = document.createElement("button");
        refreshButton.textContent = "刷新";
        refreshButton.style.cssText = `
            background: #4CAF50;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            cursor: pointer;
        `;
        refreshButton.onclick = () => this.refresh();

        buttonArea.appendChild(refreshButton);
        buttonArea.appendChild(closeButton);

        // 组装对话框
        this.dialog.appendChild(title);
        this.dialog.appendChild(content);
        this.dialog.appendChild(buttonArea);

        // 添加到页面
        document.body.appendChild(this.dialog);

        // 加载信息
        this.loadInfo();

        // 添加事件监听器
        this.dialog.addEventListener("click", (e) => {
            if (e.target === this.dialog) {
                this.close();
            }
        });

        return this.dialog;
    }

    async loadInfo() {
        const content = document.getElementById("lora-info-content");
        
        try {
            const info = await LORA_INFO_SERVICE.getInfo(this.loraName, true, true);
            
            if (info) {
                content.innerHTML = `
                    <div style="margin-bottom: 10px;">
                        <strong>文件名:</strong> ${info.name}
                    </div>
                    <div style="margin-bottom: 10px;">
                        <strong>大小:</strong> ${this.formatFileSize(info.size)}
                    </div>
                    <div style="margin-bottom: 10px;">
                        <strong>描述:</strong> ${info.description}
                    </div>
                    <div style="margin-bottom: 10px;">
                        <strong>强度范围:</strong> ${info.strengthMin} ~ ${info.strengthMax}
                    </div>
                `;
            } else {
                content.innerHTML = `
                    <div style="color: #ff6b6b;">
                        无法获取LoRA信息: ${this.loraName}
                    </div>
                `;
            }
        } catch (error) {
            content.innerHTML = `
                <div style="color: #ff6b6b;">
                    加载失败: ${error.message}
                </div>
            `;
        }
    }

    formatFileSize(bytes) {
        if (!bytes) return "未知";
        const sizes = ['B', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(1024));
        return Math.round(bytes / Math.pow(1024, i) * 100) / 100 + ' ' + sizes[i];
    }

    refresh() {
        this.dirty = true;
        this.loadInfo();
    }

    close() {
        if (this.dialog) {
            document.body.removeChild(this.dialog);
            this.dialog = null;
            
            // 触发关闭事件
            const event = new CustomEvent("close", {
                detail: { dirty: this.dirty }
            });
            this.dialog?.dispatchEvent(event);
        }
    }
} 