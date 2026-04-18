import { $el } from "../utils/shared_utils.js";

export class AssetManagerTooltip {
    constructor() {
        this.tooltip = null;
    }

    show(e, item, type) {
        if (!this.tooltip) {
            this.tooltip = $el("div", {
                style: {
                    position: "fixed",
                    background: "rgba(20,20,20,0.95)",
                    border: "1px solid var(--am-accent, #444)",
                    color: "white",
                    padding: "10px",
                    borderRadius: "6px",
                    zIndex: 10000, // 高于一切
                    pointerEvents: "none",
                    maxWidth: "250px",
                    boxShadow: "0 5px 15px rgba(0,0,0,0.5)"
                }
            });
            document.body.appendChild(this.tooltip);
        }
        
        this.tooltip.innerHTML = "";
        
        let imgSrc = "";
        if (item.preview_image) {
            imgSrc = `/a_my_nodes/assets/view_preview?path=${encodeURIComponent(item.preview_image)}`;
        } else if (type === 'models') {
            let firstLora = "";
            if (item.high_loras && item.high_loras.length > 0 && item.high_loras[0].lora) {
                firstLora = item.high_loras[0].lora;
            } else if (item.low_loras && item.low_loras.length > 0 && item.low_loras[0].lora) {
                firstLora = item.low_loras[0].lora;
            }
            
            if (firstLora && firstLora !== "None") {
                imgSrc = `/a_my_nodes/assets/view_preview?fallback_lora=${encodeURIComponent(firstLora)}`;
            }
        }
        
        if (imgSrc) {
            const imgEl = $el("img", { 
                src: imgSrc, 
                style: { width: "100%", borderRadius: "4px", marginBottom: "5px", background: "black" }
            });
            imgEl.onerror = () => { imgEl.style.display = "none"; };
            this.tooltip.appendChild(imgEl);
        }
        
        if (type === 'prompts') {
            this.tooltip.appendChild($el("div", { style: { fontSize: "12px", whiteSpace: "pre-wrap" }, textContent: item.content }));
        } else {
            this.tooltip.appendChild($el("div", { style: { fontSize: "12px" }, textContent: `强度: ${item.strength || 1.0}` }));
        }
        
        const rect = e.target.getBoundingClientRect();
        
        // 智能定位，防止超出屏幕
        let left = rect.right + 10;
        let top = rect.top;
        
        if (left + 250 > window.innerWidth) {
            left = rect.left - 260; // 显示在左侧
        }
        if (top + 200 > window.innerHeight) {
            top = window.innerHeight - 210; // 向上移
        }

        this.tooltip.style.left = `${left}px`;
        this.tooltip.style.top = `${top}px`;
        this.tooltip.style.display = "block";
    }

    hide() {
        if (this.tooltip) this.tooltip.style.display = "none";
    }
}

// 导出一个单例
export const tooltipManager = new AssetManagerTooltip();
