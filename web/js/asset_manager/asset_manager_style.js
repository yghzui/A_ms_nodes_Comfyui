export const cssStyles = `
/* Asset Manager 全局样式 */
:root {
    --am-bg: #1e1e1e;
    --am-panel-bg: #2d2d2d;
    --am-border: #444;
    --am-text: #eee;
    --am-accent: #007acc;
    --am-hover: #3d3d3d;
}

#asset-manager-fab {
    position: fixed;
    right: 30px;
    bottom: 30px;
    width: 50px;
    height: 50px;
    border-radius: 25px;
    background-color: var(--am-accent);
    color: white;
    font-size: 24px;
    display: flex;
    justify-content: center;
    align-items: center;
    cursor: pointer;
    box-shadow: 0 4px 10px rgba(0,0,0,0.5);
    z-index: 9999;
    transition: transform 0.2s;
    user-select: none;
}
#asset-manager-fab:hover {
    transform: scale(1.1);
}

#asset-manager-modal {
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background: rgba(0,0,0,0.7);
    z-index: 10000;
    display: none;
    justify-content: center;
    align-items: center;
    backdrop-filter: blur(5px);
}

.am-container {
    width: 80vw;
    height: 80vh;
    background: var(--am-bg);
    border-radius: 10px;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    box-shadow: 0 10px 30px rgba(0,0,0,0.8);
    border: 1px solid var(--am-border);
    color: var(--am-text);
    font-family: sans-serif;
    min-height: 0;
}

.am-header {
    height: 50px;
    flex-shrink: 0;
    background: var(--am-panel-bg);
    display: flex;
    align-items: center;
    padding: 0 20px;
    border-bottom: 1px solid var(--am-border);
}

.am-tabs {
    display: flex;
    gap: 10px;
    flex: 1;
}

.am-tab {
    padding: 8px 16px;
    cursor: pointer;
    border-radius: 4px;
    background: transparent;
}
.am-tab.active {
    background: var(--am-accent);
}

.am-close {
    cursor: pointer;
    font-size: 20px;
    padding: 10px;
}

.am-body {
    display: flex;
    flex: 1;
    min-height: 0;
    overflow: hidden;
}

.am-sidebar {
    width: 200px;
    flex-shrink: 0;
    background: var(--am-panel-bg);
    border-right: 1px solid var(--am-border);
    display: flex;
    flex-direction: column;
    min-height: 0;
}

.am-groups {
    flex: 1;
    height: 0;
    min-height: 0;
    overflow-x: hidden;
    overflow-y: scroll;
    scrollbar-gutter: stable;
    padding: 10px;
}

.am-group-item {
    padding: 10px;
    cursor: pointer;
    border-radius: 4px;
    margin-bottom: 5px;
}
.am-group-item:hover { background: var(--am-hover); }
.am-group-item.active { background: var(--am-accent); }

.am-sidebar-footer {
    padding: 10px;
    flex-shrink: 0;
    border-top: 1px solid var(--am-border);
}

.am-content {
    flex: 1;
    display: flex;
    flex-direction: column;
    min-height: 0;
    overflow: hidden;
    background: var(--am-bg);
}

.am-toolbar {
    height: 40px;
    flex-shrink: 0;
    display: flex;
    align-items: center;
    padding: 0 10px;
    border-bottom: 1px solid var(--am-border);
    gap: 10px;
}

.am-items-area {
    flex: 1;
    height: 0;
    min-height: 0;
    overflow-x: hidden;
    overflow-y: scroll;
    scrollbar-gutter: stable;
    overscroll-behavior: contain;
    padding: 20px;
}

.am-groups::-webkit-scrollbar,
.am-items-area::-webkit-scrollbar,
.am-card-desc::-webkit-scrollbar,
.am-drawer::-webkit-scrollbar {
    width: 10px;
}

.am-groups::-webkit-scrollbar-track,
.am-items-area::-webkit-scrollbar-track,
.am-card-desc::-webkit-scrollbar-track,
.am-drawer::-webkit-scrollbar-track {
    background: #222;
}

.am-groups::-webkit-scrollbar-thumb,
.am-items-area::-webkit-scrollbar-thumb,
.am-card-desc::-webkit-scrollbar-thumb,
.am-drawer::-webkit-scrollbar-thumb {
    background: #666;
    border-radius: 999px;
    border: 2px solid #222;
}

.am-groups::-webkit-scrollbar-thumb:hover,
.am-items-area::-webkit-scrollbar-thumb:hover,
.am-card-desc::-webkit-scrollbar-thumb:hover,
.am-drawer::-webkit-scrollbar-thumb:hover {
    background: #888;
}

/* Grid View */
.am-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 20px;
    align-content: start;
}
.am-grid .am-card {
    display: flex;
    flex-direction: column;
    height: 100%;
}
.am-grid .am-card-content {
    flex: 1; /* 让内容区域自动撑满剩余空间，把总结推到底部 */
}
.am-grid .am-card-img {
    width: 100%;
    max-height: 150px;
    height: auto;
    object-fit: contain;
    background: #000;
    border-radius: 4px;
    margin-bottom: 10px;
}

/* List View */
.am-list {
    display: flex;
    flex-direction: column;
    gap: 10px;
}
.am-list .am-card {
    display: flex;
    flex-direction: row;
    align-items: flex-start;
    gap: 15px;
}
.am-list .am-card-img {
    width: 120px;
    height: 120px;
    object-fit: contain;
    background: #000;
    border-radius: 4px;
    flex-shrink: 0;
}

.am-card {
    background: var(--am-panel-bg);
    border: 1px solid var(--am-border);
    border-radius: 6px;
    padding: 10px;
    cursor: grab;
    position: relative;
    transition: border-color 0.2s;
}
.am-card:active { cursor: grabbing; }
.am-card:hover { border-color: var(--am-accent); }

.am-card-content {
    flex: 1;
    display: flex;
    flex-direction: column;
    overflow: hidden;
}

.am-card-title {
    font-weight: bold;
    margin-bottom: 5px;
}

.am-card-desc {
    font-size: 12px;
    color: #aaa;
    white-space: pre-wrap;
    max-height: 100px;
    overflow-y: auto;
}

.am-list .am-card {
    display: flex;
    flex-direction: row;
    align-items: flex-start;
    gap: 15px;
    height: auto;
}
.am-list .am-card-img {
    width: 120px;
    max-height: 120px;
    object-fit: contain;
    background: #000;
    border-radius: 4px;
    flex-shrink: 0;
}
.am-drawer {
    position: absolute;
    background: rgba(30, 30, 30, 0.95);
    border: 1px solid var(--am-accent);
    border-radius: 8px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.8);
    z-index: 2000;
    padding: 10px;
    color: white;
    display: none;
    backdrop-filter: blur(10px);
    width: 300px;
    max-height: 400px;
    overflow-y: auto;
}
.am-drawer-item {
    padding: 8px;
    border-bottom: 1px solid var(--am-border);
    cursor: pointer;
}
.am-drawer-item:hover {
    background: var(--am-hover);
}
/* Lora item dragging */
.am-lora-item {
    transition: transform 0.2s, box-shadow 0.2s;
}
.am-lora-item.dragging {
    opacity: 0.5;
    background: var(--am-accent) !important;
}
.am-lora-item.drag-over {
    border-top: 2px solid var(--am-accent) !important;
}

.am-card.selected{border-color:var(--am-accent);box-shadow:0 0 0 2px var(--am-accent);}
.am-selection-box{position:fixed;border:1px dashed #007acc;background:rgba(0,122,204,0.2);z-index:10001;pointer-events:none;}
.am-context-menu{position:fixed;background:var(--am-panel-bg);border:1px solid var(--am-border);border-radius:4px;box-shadow:0 4px 10px rgba(0,0,0,0.5);z-index:10002;display:flex;flex-direction:column;padding:5px 0;min-width:120px;}
.am-context-menu-item{padding:8px 15px;cursor:pointer;color:var(--am-text);font-size:14px;}
.am-context-menu-item:hover{background:var(--am-accent);}
.am-context-menu-divider{height:1px;background:var(--am-border);margin:5px 0;}

/* Expanding input (已弃用内部输入框展开，改用整个卡片展开) */
.am-lora-input {
    z-index: 1;
}

/* Edit Mode Expansion in Grid (Absolute overlay to prevent jumping) */
.am-card.edit-mode {
    z-index: 100;
    overflow: visible;
}
`;
