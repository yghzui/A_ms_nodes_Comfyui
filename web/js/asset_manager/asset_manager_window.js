export class AssetManagerWindowController {
    constructor(modal, container, header, body) {
        this.modal = modal;
        this.container = container;
        this.header = header;
        this.body = body;
        this.storageKey = "am_manager_window_bounds";
        this.stateStorageKey = "am_manager_window_state";
        this.dragState = null;
        this.resizeObserver = null;
        this.normalBounds = null;
        this.lastAppliedBounds = null;
        this.isPinned = false;
        this.isMaximized = false;
        this.isMinimized = false;
        this.init();
    }

    init() {
        this.restoreBounds();
        this.restoreState();
        this.initDrag();
        this.initResizePersistence();
        this.initViewportClamp();
        this.initHeaderActions();
        this.container.addEventListener("mousedown", () => this.bringToFront());
        this.updateWindowState();
    }

    bringToFront() {
        this.modal.style.zIndex = this.isPinned ? "11000" : "10000";
    }

    getDefaultBounds() {
        const width = Math.min(Math.max(window.innerWidth * 0.7, 720), window.innerWidth - 40);
        const height = Math.min(Math.max(window.innerHeight * 0.7, 480), window.innerHeight - 40);
        return {
            width,
            height,
            left: Math.max(20, (window.innerWidth - width) / 2),
            top: Math.max(20, (window.innerHeight - height) / 2)
        };
    }

    isValidBounds(bounds) {
        return !!bounds
            && Number.isFinite(bounds.width)
            && Number.isFinite(bounds.height)
            && Number.isFinite(bounds.left)
            && Number.isFinite(bounds.top)
            && bounds.width > 0
            && bounds.height > 0;
    }

    getClampedBounds(bounds) {
        const minWidth = 720;
        const minHeight = 480;
        const maxWidth = Math.max(minWidth, window.innerWidth - 20);
        const maxHeight = Math.max(minHeight, window.innerHeight - 20);
        const width = Math.min(Math.max(bounds.width || minWidth, minWidth), maxWidth);
        const height = Math.min(Math.max(bounds.height || minHeight, minHeight), maxHeight);
        const left = Math.min(Math.max(bounds.left ?? 20, 0), Math.max(0, window.innerWidth - width));
        const top = Math.min(Math.max(bounds.top ?? 20, 0), Math.max(0, window.innerHeight - height));
        return { width, height, left, top };
    }

    applyBounds(bounds) {
        const safeBounds = this.getClampedBounds(bounds);
        this.lastAppliedBounds = { ...safeBounds };
        this.container.style.width = `${safeBounds.width}px`;
        this.container.style.height = `${safeBounds.height}px`;
        this.container.style.left = `${safeBounds.left}px`;
        this.container.style.top = `${safeBounds.top}px`;
    }

    readCurrentBounds() {
        const rect = this.container.getBoundingClientRect();
        const computed = window.getComputedStyle(this.container);
        const styleWidth = parseFloat(computed.width);
        const styleHeight = parseFloat(computed.height);
        const styleLeft = parseFloat(computed.left);
        const styleTop = parseFloat(computed.top);
        return {
            width: Math.round(styleWidth || rect.width || this.container.offsetWidth || 0),
            height: Math.round(styleHeight || rect.height || this.container.offsetHeight || 0),
            left: Math.round(styleLeft || rect.left || 0),
            top: Math.round(styleTop || rect.top || 0)
        };
    }

    saveBounds(force = false) {
        if ((this.isMaximized || this.isMinimized) && !force) return;
        const rawBounds = (this.isMaximized || this.isMinimized) && this.normalBounds
            ? { ...this.normalBounds }
            : this.readCurrentBounds();
        const bounds = this.getClampedBounds(this.isValidBounds(rawBounds) ? rawBounds : (this.lastAppliedBounds || this.getDefaultBounds()));
        this.normalBounds = { ...bounds };
        try {
            localStorage.setItem(this.storageKey, JSON.stringify(bounds));
        } catch (error) {
            console.warn("[AssetManager] Failed to save window bounds:", error);
        }
    }

    restoreBounds() {
        try {
            const saved = localStorage.getItem(this.storageKey);
            if (saved) {
                const parsedBounds = JSON.parse(saved);
                this.normalBounds = this.isValidBounds(parsedBounds) ? parsedBounds : this.getDefaultBounds();
                this.applyBounds(this.normalBounds);
                return;
            }
        } catch (error) {
            console.warn("[AssetManager] Failed to restore window bounds:", error);
        }
        this.normalBounds = this.getDefaultBounds();
        this.applyBounds(this.normalBounds);
    }

    saveState() {
        try {
            localStorage.setItem(this.stateStorageKey, JSON.stringify({
                pinned: this.isPinned,
                maximized: this.isMaximized,
                minimized: this.isMinimized
            }));
        } catch (error) {
            console.warn("[AssetManager] Failed to save window state:", error);
        }
    }

    restoreState() {
        try {
            const saved = localStorage.getItem(this.stateStorageKey);
            if (!saved) return;
            const state = JSON.parse(saved);
            this.isPinned = !!state.pinned;
            this.isMaximized = !!state.maximized;
            this.isMinimized = !!state.minimized;
        } catch (error) {
            console.warn("[AssetManager] Failed to restore window state:", error);
        }
    }

    getMaximizedBounds() {
        return {
            left: 10,
            top: 10,
            width: Math.max(720, window.innerWidth - 20),
            height: Math.max(480, window.innerHeight - 20)
        };
    }

    updateWindowState() {
        this.container.classList.toggle("am-window-pinned", this.isPinned);
        this.container.classList.toggle("am-window-maximized", this.isMaximized);
        this.container.classList.toggle("am-window-minimized", this.isMinimized);
        if (this.body) {
            this.body.style.display = this.isMinimized ? "none" : "flex";
        }
        this.container.style.resize = this.isMaximized || this.isMinimized ? "none" : "both";
        this.bringToFront();
    }

    setPinned(value) {
        this.isPinned = !!value;
        this.updateWindowState();
        this.saveState();
    }

    togglePinned() {
        this.setPinned(!this.isPinned);
        return this.isPinned;
    }

    setMinimized(value) {
        const nextValue = !!value;
        if (nextValue === this.isMinimized) return this.isMinimized;
        if (nextValue && !this.isMaximized) this.saveBounds();
        this.isMinimized = nextValue;
        if (this.isMinimized) {
            const headerHeight = this.header.offsetHeight || 50;
            this.container.style.height = `${headerHeight}px`;
        } else if (this.isMaximized) {
            this.applyBounds(this.getMaximizedBounds());
        } else {
            this.applyBounds(this.normalBounds || this.getDefaultBounds());
        }
        this.updateWindowState();
        this.saveState();
        return this.isMinimized;
    }

    toggleMinimized() {
        return this.setMinimized(!this.isMinimized);
    }

    setMaximized(value) {
        const nextValue = !!value;
        if (nextValue === this.isMaximized) return this.isMaximized;
        if (nextValue) {
            if (!this.isMinimized) this.saveBounds();
            this.isMaximized = true;
            this.isMinimized = false;
            this.applyBounds(this.getMaximizedBounds());
        } else {
            this.isMaximized = false;
            this.applyBounds(this.normalBounds || this.getDefaultBounds());
        }
        this.updateWindowState();
        this.saveState();
        return this.isMaximized;
    }

    toggleMaximized() {
        return this.setMaximized(!this.isMaximized);
    }

    updateControlButtonState(buttonName, active) {
        const button = this.header.querySelector(`[data-am-window-action="${buttonName}"]`);
        if (!button) return;
        button.classList.toggle("active", !!active);
        if (buttonName === "pin") button.title = active ? "取消置顶" : "置顶";
        if (buttonName === "maximize") button.title = active ? "还原窗口" : "最大化";
        if (buttonName === "minimize") button.title = active ? "还原窗口" : "最小化";
    }

    refreshControlStates() {
        this.updateControlButtonState("pin", this.isPinned);
        this.updateControlButtonState("maximize", this.isMaximized);
        this.updateControlButtonState("minimize", this.isMinimized);
    }

    initHeaderActions() {
        const actions = {
            pin: () => this.togglePinned(),
            minimize: () => this.toggleMinimized(),
            maximize: () => this.toggleMaximized()
        };
        this.header.querySelectorAll("[data-am-window-action]").forEach(button => {
            button.addEventListener("click", (event) => {
                const action = button.dataset.amWindowAction;
                actions[action]?.();
                this.refreshControlStates();
                event.stopPropagation();
            });
        });
        this.header.addEventListener("dblclick", (event) => {
            if (event.target.closest(".am-window-controls, .am-tab, .am-close")) return;
            this.toggleMaximized();
            this.refreshControlStates();
        });
        this.refreshControlStates();
    }

    initDrag() {
        this.header.addEventListener("mousedown", (event) => {
            if (event.button !== 0) return;
            if (event.target.closest("button, input, select, textarea, .am-tab, .am-close")) return;
            if (this.isMaximized || this.isMinimized) return;
            const rect = this.container.getBoundingClientRect();
            this.dragState = {
                startX: event.clientX,
                startY: event.clientY,
                left: rect.left,
                top: rect.top
            };
            document.body.style.userSelect = "none";
            event.preventDefault();
        });

        window.addEventListener("mousemove", (event) => {
            if (!this.dragState) return;
            const dx = event.clientX - this.dragState.startX;
            const dy = event.clientY - this.dragState.startY;
            this.applyBounds({
                width: this.container.offsetWidth,
                height: this.container.offsetHeight,
                left: this.dragState.left + dx,
                top: this.dragState.top + dy
            });
        });

        window.addEventListener("mouseup", () => {
            if (!this.dragState) return;
            this.dragState = null;
            document.body.style.userSelect = "";
            this.saveBounds();
        });
    }

    initResizePersistence() {
        if (typeof ResizeObserver === "undefined") return;
        let resizeTimer = null;
        this.resizeObserver = new ResizeObserver(() => {
            clearTimeout(resizeTimer);
            resizeTimer = setTimeout(() => {
                if (this.isMaximized || this.isMinimized) return;
                this.applyBounds(this.readCurrentBounds());
                this.saveBounds(true);
            }, 80);
        });
        this.resizeObserver.observe(this.container);
    }

    initViewportClamp() {
        window.addEventListener("resize", () => {
            if (this.isMaximized) {
                this.applyBounds(this.getMaximizedBounds());
            } else if (!this.isMinimized) {
                this.applyBounds(this.lastAppliedBounds || this.readCurrentBounds());
                this.saveBounds(true);
            }
        });
    }

    onShow() {
        if (this.isMaximized) {
            this.applyBounds(this.getMaximizedBounds());
        } else if (this.isMinimized) {
            const bounds = this.normalBounds || this.getDefaultBounds();
            this.applyBounds(bounds);
            this.container.style.height = `${this.header.offsetHeight || 50}px`;
        } else {
            this.restoreBounds();
        }
        this.updateWindowState();
        this.refreshControlStates();
        this.bringToFront();
    }

    onHide() {
        this.saveBounds(true);
        this.saveState();
    }
}
