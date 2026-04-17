import { $el } from "../utils/shared_utils.js";

// ================= CSS 注入 =================
const dialogCssStyles = `
/* Custom Dialogs */
.am-dialog-overlay {
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background: rgba(0,0,0,0.8);
    z-index: 12000;
    display: flex;
    justify-content: center;
    align-items: center;
}
.am-dialog-box {
    background: #2d2d2d;
    border: 1px solid #444;
    border-radius: 8px;
    padding: 20px;
    color: #eee;
    min-width: 300px;
    max-width: 500px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.8);
    font-family: sans-serif;
}
.am-dialog-buttons {
    display: flex;
    justify-content: flex-end;
    gap: 10px;
    margin-top: 20px;
}
.am-dialog-btn {
    padding: 8px 16px;
    border-radius: 4px;
    cursor: pointer;
    border: 1px solid #444;
    background: #3d3d3d;
    color: white;
}
.am-dialog-btn:hover {
    background: #4d4d4d;
}
.am-dialog-btn.primary {
    background: #007acc;
    border-color: #007acc;
}
.am-dialog-btn.primary:hover {
    background: #005f9e;
}
`;

const style = document.createElement("style");
style.textContent = dialogCssStyles;
document.head.appendChild(style);

// ================= 弹窗模块 (AMDialog) =================
export const AMDialog = {
    /**
     * 弹出一个提示框 (Alert)
     * @param {string} msg 提示内容
     * @returns {Promise<void>} 用户点击确定后 resolve
     */
    async alert(msg) {
        return new Promise(resolve => {
            const overlay = $el("div", { className: "am-dialog-overlay" });
            const box = $el("div", { className: "am-dialog-box" }, [
                $el("div", { style: { whiteSpace: "pre-wrap", lineHeight: "1.5" }, textContent: msg }),
                $el("div", { className: "am-dialog-buttons" }, [
                    $el("button", { 
                        className: "am-dialog-btn primary",
                        textContent: "确定", 
                        onclick: () => { document.body.removeChild(overlay); resolve(); } 
                    })
                ])
            ]);
            overlay.appendChild(box);
            document.body.appendChild(overlay);
        });
    },

    /**
     * 弹出一个确认框 (Confirm)
     * @param {string} msg 确认内容
     * @returns {Promise<boolean>} 用户点击确定返回 true，取消返回 false
     */
    async confirm(msg) {
        return new Promise(resolve => {
            const overlay = $el("div", { className: "am-dialog-overlay" });
            const box = $el("div", { className: "am-dialog-box" }, [
                $el("div", { style: { whiteSpace: "pre-wrap", lineHeight: "1.5" }, textContent: msg }),
                $el("div", { className: "am-dialog-buttons" }, [
                    $el("button", { 
                        className: "am-dialog-btn",
                        textContent: "取消", 
                        onclick: () => { document.body.removeChild(overlay); resolve(false); } 
                    }),
                    $el("button", { 
                        className: "am-dialog-btn primary",
                        textContent: "确定", 
                        onclick: () => { document.body.removeChild(overlay); resolve(true); } 
                    })
                ])
            ]);
            overlay.appendChild(box);
            document.body.appendChild(overlay);
        });
    }
};

// 挂载到 window 对象，方便在没有 import 的原生/旧脚本中使用 (如 wan_video_double_stream.js)
window.AMDialog = AMDialog;
