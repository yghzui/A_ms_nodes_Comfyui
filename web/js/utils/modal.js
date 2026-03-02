// 简单的通用模态框组件
// 支持标题、内容、按钮和关闭回调
// 样式注入

export class CustomModal {
    constructor() {
        this.element = null;
        this.overlay = null;
        this.injectStyles();
    }

    injectStyles() {
        if (document.getElementById('custom-modal-styles')) return;
        const style = document.createElement('style');
        style.id = 'custom-modal-styles';
        style.textContent = `
            .custom-modal-overlay {
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: rgba(0, 0, 0, 0.5);
                z-index: 10000;
                display: flex;
                justify-content: center;
                align-items: center;
                backdrop-filter: blur(2px);
            }
            .custom-modal {
                background: #222;
                border: 1px solid #444;
                border-radius: 8px;
                box-shadow: 0 4px 20px rgba(0,0,0,0.5);
                min-width: 300px;
                max-width: 80%;
                max-height: 90%;
                display: flex;
                flex-direction: column;
                color: #eee;
                font-family: "Microsoft YaHei", sans-serif;
                animation: modal-fade-in 0.2s ease-out;
            }
            @keyframes modal-fade-in {
                from { opacity: 0; transform: scale(0.95); }
                to { opacity: 1; transform: scale(1); }
            }
            .custom-modal-header {
                padding: 12px 16px;
                border-bottom: 1px solid #333;
                font-weight: bold;
                font-size: 16px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            .custom-modal-close {
                cursor: pointer;
                font-size: 20px;
                line-height: 1;
                color: #888;
                transition: color 0.2s;
            }
            .custom-modal-close:hover {
                color: #fff;
            }
            .custom-modal-content {
                padding: 16px;
                overflow-y: auto;
                font-size: 14px;
                line-height: 1.5;
            }
            .custom-modal-footer {
                padding: 12px 16px;
                border-top: 1px solid #333;
                display: flex;
                justify-content: flex-end;
                gap: 10px;
            }
            .custom-modal-btn {
                padding: 6px 12px;
                border-radius: 4px;
                border: 1px solid #555;
                background: #333;
                color: #eee;
                cursor: pointer;
                font-size: 13px;
                transition: background 0.2s;
            }
            .custom-modal-btn:hover {
                background: #444;
            }
            .custom-modal-btn.primary {
                background: #2a6db5;
                border-color: #1e5a9c;
            }
            .custom-modal-btn.primary:hover {
                background: #3a7dc5;
            }
            .custom-modal-btn.danger {
                background: #b52a2a;
                border-color: #9c1e1e;
            }
            .custom-modal-btn.danger:hover {
                background: #c53a3a;
            }
            /* 输入框样式 */
            .custom-modal-textarea {
                width: 100%;
                min-height: 100px;
                background: #1a1a1a;
                border: 1px solid #444;
                color: #eee;
                padding: 8px;
                border-radius: 4px;
                resize: vertical;
                font-family: monospace;
                box-sizing: border-box;
            }
            .custom-modal-textarea:focus {
                border-color: #2a6db5;
                outline: none;
            }
            .custom-modal-file-input {
                margin-top: 10px;
                display: block;
                width: 100%;
                padding: 8px;
                background: #1a1a1a;
                border: 1px solid #444;
                border-radius: 4px;
                box-sizing: border-box;
            }
        `;
        document.head.appendChild(style);
    }

    show({ title = '提示', content = '', buttons = [], width = '400px', onClose = null }) {
        this.close(); // 关闭已有弹窗

        this.overlay = document.createElement('div');
        this.overlay.className = 'custom-modal-overlay';
        
        this.element = document.createElement('div');
        this.element.className = 'custom-modal';
        this.element.style.width = width;
        
        // Header
        const header = document.createElement('div');
        header.className = 'custom-modal-header';
        header.innerHTML = `<span>${title}</span><span class="custom-modal-close">&times;</span>`;
        header.querySelector('.custom-modal-close').onclick = () => this.close();
        this.element.appendChild(header);

        // Content
        const contentDiv = document.createElement('div');
        contentDiv.className = 'custom-modal-content';
        if (typeof content === 'string') {
            contentDiv.innerHTML = content;
        } else if (content instanceof HTMLElement) {
            contentDiv.appendChild(content);
        }
        this.element.appendChild(contentDiv);

        // Footer
        if (buttons.length > 0) {
            const footer = document.createElement('div');
            footer.className = 'custom-modal-footer';
            
            buttons.forEach(btnConfig => {
                const btn = document.createElement('button');
                btn.className = `custom-modal-btn ${btnConfig.type || ''}`;
                btn.textContent = btnConfig.text;
                btn.onclick = (e) => {
                    if (btnConfig.onClick) {
                        btnConfig.onClick(e, this);
                    } else {
                        this.close();
                    }
                };
                footer.appendChild(btn);
            });
            this.element.appendChild(footer);
        }

        this.overlay.appendChild(this.element);
        document.body.appendChild(this.overlay);

        // 点击遮罩关闭
        this.overlay.onclick = (e) => {
            if (e.target === this.overlay) {
                this.close();
            }
        };
        
        // 绑定 onClose
        this.onCloseCallback = onClose;

        // 自动聚焦第一个输入框（如果有）
        const firstInput = this.element.querySelector('input, textarea');
        if (firstInput) setTimeout(() => firstInput.focus(), 50);
    }

    close() {
        if (this.overlay) {
            this.overlay.remove();
            this.overlay = null;
            this.element = null;
            if (this.onCloseCallback) {
                this.onCloseCallback();
                this.onCloseCallback = null;
            }
        }
    }
}

// 导出单例以便直接使用
export const modal = new CustomModal();
