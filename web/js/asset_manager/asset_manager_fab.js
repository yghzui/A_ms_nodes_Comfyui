import { $el } from "../utils/shared_utils.js";

export class AssetManagerFAB {
    constructor(managerUI) {
        this.managerUI = managerUI;
        this.fab = null;
        this.clickTimeout = null;
        this.createDOM();
        this.initEvents();
    }

    createDOM() {
        this.fab = $el("div", {
            id: "asset-manager-fab",
            textContent: "📦",
            title: "资产管理 (单击: 快捷菜单, 双击: 打开管理, 拖拽移动)",
            style: {
                position: "fixed",
                right: "30px",
                bottom: "30px",
                width: "50px",
                height: "50px",
                borderRadius: "50%",
                background: "var(--am-accent, #444)",
                color: "white",
                display: "flex",
                justifyContent: "center",
                alignItems: "center",
                fontSize: "24px",
                cursor: "pointer",
                boxShadow: "0 4px 10px rgba(0,0,0,0.5)",
                zIndex: 9997,
                userSelect: "none",
                transition: "transform 0.2s"
            }
        });

        // 将小悬浮球作为子元素添加到大悬浮球中，实现整体跟随
        this.miniFab = $el("div", {
            id: "asset-manager-mini-fab",
            textContent: "🪄",
            title: "快捷插入 (Quick Insert)",
            style: {
                position: "absolute",
                left: "-15px",
                top: "-15px",
                width: "24px",
                height: "24px",
                borderRadius: "50%",
                background: "var(--am-accent, #555)",
                color: "white",
                display: "flex",
                justifyContent: "center",
                alignItems: "center",
                fontSize: "14px",
                cursor: "pointer",
                boxShadow: "0 2px 5px rgba(0,0,0,0.5)",
                zIndex: 9998, // 比大球高一层
                transition: "transform 0.2s"
            },
            onclick: (e) => {
                e.stopPropagation();
                // 触发快捷菜单的切换
                if (window.AssetManagerQuickMenu) {
                    window.AssetManagerQuickMenu.toggleMenu();
                }
            },
            onmouseenter: (e) => { e.target.style.transform = "scale(1.1)"; },
            onmouseleave: (e) => { e.target.style.transform = "scale(1)"; }
        });
        
        this.fab.appendChild(this.miniFab);
        document.body.appendChild(this.fab);
        
        // 恢复悬浮球保存的位置
        const savedPos = localStorage.getItem("am_fab_position");
        if (savedPos) {
            try {
                const pos = JSON.parse(savedPos);
                const rightVal = parseFloat(pos.right);
                const bottomVal = parseFloat(pos.bottom);
                
                if (!isNaN(rightVal) && !isNaN(bottomVal)) {
                    const safeRight = Math.max(10, Math.min(rightVal, window.innerWidth - 60));
                    const safeBottom = Math.max(10, Math.min(bottomVal, window.innerHeight - 60));
                    this.fab.style.right = `${safeRight}px`;
                    this.fab.style.bottom = `${safeBottom}px`;
                }
            } catch(e) {
                console.error("[AssetManager] Failed to restore FAB position:", e);
                localStorage.removeItem("am_fab_position");
                this.fab.style.right = "30px";
                this.fab.style.bottom = "30px";
            }
        }
    }

    initEvents() {
        let isDragging = false;
        let startX, startY, initialX, initialY;
        let moved = false;

        this.fab.addEventListener("mousedown", (e) => {
            isDragging = true;
            moved = false;
            startX = e.clientX;
            startY = e.clientY;
            
            const rect = this.fab.getBoundingClientRect();
            initialX = window.innerWidth - rect.right;
            initialY = window.innerHeight - rect.bottom;
            
            this.fab.style.transition = "none";
            e.preventDefault();
        });

        document.addEventListener("mousemove", (e) => {
            if (!isDragging) return;
            
            const dx = e.clientX - startX;
            const dy = e.clientY - startY;
            
            if (Math.abs(dx) > 5 || Math.abs(dy) > 5) {
                moved = true;
            }
            
            if (moved) {
                let newRight = initialX - dx;
                let newBottom = initialY - dy;
                
                newRight = Math.max(0, Math.min(newRight, window.innerWidth - 50));
                newBottom = Math.max(0, Math.min(newBottom, window.innerHeight - 50));
                
                this.fab.style.right = `${newRight}px`;
                this.fab.style.bottom = `${newBottom}px`;
            }
        });

        document.addEventListener("mouseup", (e) => {
            if (!isDragging) return;
            isDragging = false;
            this.fab.style.transition = "transform 0.2s";
            
            if (moved) {
                localStorage.setItem("am_fab_position", JSON.stringify({
                    right: this.fab.style.right,
                    bottom: this.fab.style.bottom
                }));
            } else {
                // 处理点击 (防抖区分单双击)
                if (this.clickTimeout) {
                    clearTimeout(this.clickTimeout);
                    this.clickTimeout = null;
                    // 双击: 打开模态框
                    this.managerUI.showModal();
                } else {
                    this.clickTimeout = setTimeout(() => {
                        this.clickTimeout = null;
                        // 单击大球时：我们不再执行伴生逻辑，因为小球已经始终存在了
                        // 如果要实现单击大球也可以 toggle 菜单，可以放开下面代码：
                        /*
                        if (window.AssetManagerQuickMenu) {
                            window.AssetManagerQuickMenu.toggleMenu();
                        }
                        */
                    }, 250); // 250ms threshold
                }
            }
        });
        
        // 悬浮特效
        this.fab.onmouseenter = () => { this.fab.style.transform = "scale(1.1)"; };
        this.fab.onmouseleave = () => { this.fab.style.transform = "scale(1)"; };
    }
}
