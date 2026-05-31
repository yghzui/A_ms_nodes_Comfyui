import { $el } from "../utils/shared_utils.js";
import { PreviewHandler } from "./asset_manager_preview_handler.js";

export class DragSelectHandler {
    static initSelectionAndClipboard(manager, area) {
        let isSelecting = false, startX, startY, selBox = null, initialSel = new Set();

        area.addEventListener('mousedown', (e) => {
            if (!e.ctrlKey || e.button !== 0) return;
            if (['INPUT', 'BUTTON', 'SELECT', 'TEXTAREA'].includes(e.target.tagName)) return;
            isSelecting = true;
            startX = e.clientX; startY = e.clientY;
            initialSel = new Set(manager.selectedIndices);
            selBox = $el("div", { className: "am-selection-box" });
            document.body.appendChild(selBox);
            DragSelectHandler.updateSelectionBoxRect(selBox, startX, startY, startX, startY);
            e.preventDefault();
        });

        document.addEventListener('mousemove', (e) => {
            if (!isSelecting) return;
            DragSelectHandler.updateSelectionBoxRect(selBox, startX, startY, e.clientX, e.clientY);
            const boxRect = selBox.getBoundingClientRect();
            const listEl = document.getElementById("am-item-list");
            if (!listEl) return;
            
            const newSel = new Set(initialSel);
            Array.from(listEl.children).forEach((card, idx) => {
                const r = card.getBoundingClientRect();
                if (!(r.right < boxRect.left || r.left > boxRect.right || r.bottom < boxRect.top || r.top > boxRect.bottom)) {
                    newSel.add(idx);
                }
            });
            manager.selectedIndices = newSel;
            DragSelectHandler.updateSelectionUI(manager);
        });

        document.addEventListener('mouseup', () => {
            if (isSelecting) {
                isSelecting = false;
                if (selBox && selBox.parentNode) selBox.parentNode.removeChild(selBox);
                selBox = null;
            }
        });

        document.addEventListener('keydown', (e) => {
            if (manager.modal.style.display !== "flex") return;
            if (document.activeElement && ['INPUT', 'TEXTAREA'].includes(document.activeElement.tagName)) return;
            
            if (e.ctrlKey && e.key.toLowerCase() === 'a') { e.preventDefault(); DragSelectHandler.selectAll(manager); }
            else if (e.key === 'Delete' || e.key === 'Backspace') { DragSelectHandler.deleteSelected(manager); }
            else if (e.ctrlKey && e.key.toLowerCase() === 'c') { DragSelectHandler.copySelected(manager); }
            else if (e.ctrlKey && e.key.toLowerCase() === 'v') { 
                if (manager.selectedIndices.size === 1) return;
                DragSelectHandler.pasteClipboard(manager); 
            }
        });

        document.addEventListener('paste', async (e) => {
            if (manager.modal.style.display !== "flex") return;
            if (document.activeElement && ['INPUT', 'TEXTAREA'].includes(document.activeElement.tagName)) return;
            
            if (manager.selectedIndices.size === 1) {
                e.preventDefault();
                e.stopPropagation();
                try {
                    const uri = await PreviewHandler.handlePasteEvent(e);
                    if (uri) {
                        const data = manager.currentTab === 'prompts' ? manager.promptsData : manager.modelsData;
                        const group = data.groups[manager.currentGroupIndex];
                        const idx = Array.from(manager.selectedIndices)[0];
                        group.items[idx].preview_image = uri;
                        manager.saveData();
                        manager.renderItems();
                    }
                } catch (err) {
                    if (err.message.includes("没有发现图片文件")) {
                        DragSelectHandler.pasteClipboard(manager);
                    } else {
                        manager.alert(err.message, "error");
                    }
                }
            } else if (manager.selectedIndices.size === 0) {
                DragSelectHandler.pasteClipboard(manager);
            }
        });

        area.addEventListener('contextmenu', (e) => {
            if (['INPUT', 'BUTTON', 'TEXTAREA'].includes(e.target.tagName)) return;
            e.preventDefault();
            DragSelectHandler.showContextMenu(manager, e.clientX, e.clientY);
        });

        document.addEventListener('click', () => {
            if (manager.contextMenu && manager.contextMenu.parentNode) {
                manager.contextMenu.parentNode.removeChild(manager.contextMenu);
                manager.contextMenu = null;
            }
        });
    }

    static updateSelectionBoxRect(box, x1, y1, x2, y2) {
        box.style.left = Math.min(x1, x2) + 'px';
        box.style.top = Math.min(y1, y2) + 'px';
        box.style.width = Math.abs(x1 - x2) + 'px';
        box.style.height = Math.abs(y1 - y2) + 'px';
    }

    static updateSelectionUI(manager) {
        const listEl = document.getElementById("am-item-list");
        if (!listEl) return;
        Array.from(listEl.children).forEach((card, idx) => {
            if (manager.selectedIndices.has(idx)) card.classList.add('selected');
            else card.classList.remove('selected');
        });
    }

    static selectAll(manager) {
        const data = manager.currentTab === 'prompts' ? manager.promptsData : manager.modelsData;
        const group = data.groups[manager.currentGroupIndex];
        if (!group || !group.items) return;
        for (let i = 0; i < group.items.length; i++) manager.selectedIndices.add(i);
        DragSelectHandler.updateSelectionUI(manager);
    }

    static async deleteSelected(manager) {
        if (manager.selectedIndices.size === 0) return;
        const yes = await manager.confirm(`确定删除选中的 ${manager.selectedIndices.size} 个项目吗？`);
        if (!yes) return;
        const data = manager.currentTab === 'prompts' ? manager.promptsData : manager.modelsData;
        const group = data.groups[manager.currentGroupIndex];
        const indices = Array.from(manager.selectedIndices).sort((a, b) => b - a);
        indices.forEach(idx => group.items.splice(idx, 1));
        manager.selectedIndices.clear();
        manager.saveData();
        manager.renderItems();
    }

    static copySelected(manager) {
        if (manager.selectedIndices.size === 0) return;
        const data = manager.currentTab === 'prompts' ? manager.promptsData : manager.modelsData;
        const group = data.groups[manager.currentGroupIndex];
        window.amClipboard = Array.from(manager.selectedIndices).sort((a, b) => a - b).map(idx => JSON.parse(JSON.stringify(group.items[idx])));
    }

    static pasteClipboard(manager) {
        if (!window.amClipboard || window.amClipboard.length === 0) return;
        const data = manager.currentTab === 'prompts' ? manager.promptsData : manager.modelsData;
        const group = data.groups[manager.currentGroupIndex];
        if (!group) return;
        
        const sample = window.amClipboard[0];
        const isPrompt = sample.hasOwnProperty('content');
        if ((manager.currentTab === 'prompts' && !isPrompt) || (manager.currentTab === 'models' && isPrompt)) {
            manager.alert("剪贴板数据类型与当前标签页不匹配！", "warning");
            return;
        }
        
        window.amClipboard.forEach(item => {
            const newItem = JSON.parse(JSON.stringify(item));
            newItem.id = Date.now().toString() + Math.random().toString().slice(2, 6);
            group.items.push(newItem);
        });
        manager.saveData();
        manager.renderItems();
    }

    static showContextMenu(manager, x, y) {
        if (manager.contextMenu && manager.contextMenu.parentNode) manager.contextMenu.parentNode.removeChild(manager.contextMenu);
        manager.contextMenu = $el("div", { className: "am-context-menu", style: { left: x + 'px', top: y + 'px' } });
        
        const addMenuItem = (text, onClick, disabled = false) => {
            const item = $el("div", {
                className: "am-context-menu-item", textContent: text,
                onclick: (e) => {
                    e.stopPropagation();
                    if (!disabled) onClick();
                    if (manager.contextMenu.parentNode) manager.contextMenu.parentNode.removeChild(manager.contextMenu);
                }
            });
            if (disabled) { item.style.opacity = "0.5"; item.style.pointerEvents = "none"; }
            manager.contextMenu.appendChild(item);
        };

        if (manager.selectedIndices.size > 0) {
            addMenuItem(`📋 复制选中 (${manager.selectedIndices.size})`, () => DragSelectHandler.copySelected(manager));
            addMenuItem(`🗑️ 删除选中 (${manager.selectedIndices.size})`, () => DragSelectHandler.deleteSelected(manager));
            manager.contextMenu.appendChild($el("div", { className: "am-context-menu-divider" }));
        }
        
        const clipLen = window.amClipboard ? window.amClipboard.length : 0;
        addMenuItem(`📥 粘贴 (${clipLen})`, () => DragSelectHandler.pasteClipboard(manager), clipLen === 0);
        
        manager.contextMenu.appendChild($el("div", { className: "am-context-menu-divider" }));
        addMenuItem("✅ 全选", () => DragSelectHandler.selectAll(manager));
        
        document.body.appendChild(manager.contextMenu);
    }

    static async handlePreviewDrop(manager, e, targetIndex) {
        let uri = null;
        try {
            let textPath = e.dataTransfer.getData("text/plain") || e.dataTransfer.getData("text/uri-list");
            if (textPath) {
                textPath = textPath.trim().replace(/^"|"$/g, '');
                if (textPath.startsWith("file:///")) {
                    textPath = decodeURI(textPath.replace("file:///", ""));
                    if (textPath.match(/^[a-zA-Z]:\//)) {
                        textPath = textPath.replace(/\//g, "\\");
                    } else {
                        textPath = "/" + textPath;
                    }
                }
                if (textPath.includes(":\\") || textPath.startsWith("/") || textPath.startsWith("models://")) {
                    uri = await PreviewHandler.processPreviewSource(textPath, null);
                }
            }
            
            if (!uri && e.dataTransfer.files && e.dataTransfer.files.length > 0) {
                const file = e.dataTransfer.files[0];
                if (!file.type.startsWith("image/")) {
                    manager.alert("请拖拽有效的图片文件！", "warning");
                    return;
                }
                uri = await PreviewHandler.processPreviewSource(null, file);
            }
            
            if (uri) {
                const data = manager.currentTab === 'prompts' ? manager.promptsData : manager.modelsData;
                const group = data.groups[manager.currentGroupIndex];
                group.items[targetIndex].preview_image = uri;
                manager.saveData();
                manager.renderItems();
            }
        } catch (err) {
            manager.alert(err.message, "error");
        }
    }

    static handleDrop(manager, e, targetIndex) {
        e.preventDefault();
        try {
            const dataStr = e.dataTransfer.getData("text/plain");
            const dragData = JSON.parse(dataStr);
            if (dragData.type === "group") return;
            if (dragData.tab !== manager.currentTab) return;
            
            const data = manager.currentTab === 'prompts' ? manager.promptsData : manager.modelsData;
            const items = data.groups[manager.currentGroupIndex].items;

            let indices = dragData.indices;
            if (!indices && dragData.index !== undefined) indices = [dragData.index];
            if (!indices || indices.includes(targetIndex)) return;

            const sortedIndices = indices.slice().sort((a, b) => b - a);
            const movedItems = [];
            sortedIndices.forEach(idx => {
                movedItems.push(items.splice(idx, 1)[0]);
            });
            movedItems.reverse();

            let shift = 0;
            sortedIndices.forEach(idx => { if (idx < targetIndex) shift++; });
            const finalTargetIndex = targetIndex - shift;

            items.splice(finalTargetIndex, 0, ...movedItems);
            
            manager.selectedIndices.clear();
            for(let i=0; i<movedItems.length; i++) manager.selectedIndices.add(finalTargetIndex + i);
            
            manager.renderItems();
            manager.saveData();
        } catch (err) {
            console.error("Drop parsing error", err);
        }
    }

    static handleGroupDrop(manager, e, targetIndex, targetEl) {
        e.preventDefault();
        e.stopPropagation();
        targetEl.style.borderTop = "";
        
        try {
            const dataStr = e.dataTransfer.getData("text/plain");
            const dragData = JSON.parse(dataStr);
            
            if (dragData.type !== "group") return;
            
            const sourceIndex = dragData.index;
            if (sourceIndex === targetIndex) return;

            const pGroup = manager.promptsData.groups.splice(sourceIndex, 1)[0];
            manager.promptsData.groups.splice(targetIndex, 0, pGroup);
            
            const mGroup = manager.modelsData.groups.splice(sourceIndex, 1)[0];
            manager.modelsData.groups.splice(targetIndex, 0, mGroup);
            
            if (manager.currentGroupIndex === sourceIndex) {
                manager.currentGroupIndex = targetIndex;
            } else if (sourceIndex < manager.currentGroupIndex && targetIndex >= manager.currentGroupIndex) {
                manager.currentGroupIndex--;
            } else if (sourceIndex > manager.currentGroupIndex && targetIndex <= manager.currentGroupIndex) {
                manager.currentGroupIndex++;
            }
            
            manager.renderGroups();
            manager.saveData();
            manager.saveOtherData();
        } catch (err) {
            console.error("Group drop parsing error", err);
        }
    }
}
