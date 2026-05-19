import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";
import { CustomModal } from "./utils/modal.js";

const NODE_NAME = "WorkflowGroupPresetManager";
const ENDPOINT = "/a_my_nodes/workflow_groups";
const STYLE_ID = "a-my-nodes-workflow-group-modern-style";
const PANEL_OFFSET_Y = 14;
const confirmModal = new CustomModal();
const DEFAULT_DB = { version: 1, groups: [] };
const MANAGER_PANEL_STORAGE_KEY = "a_my_nodes_workflow_group_manager_panel_v1";
const DEFAULT_MANAGER_PANEL_LAYOUT = { left: 120, top: 80, width: 980, height: 720 };
const VALUE_TYPE_OPTIONS = ["INT", "FLOAT", "BOOLEAN", "STRING", "COMBO", "JSON_STRING"];
const LEGACY_WIDGET_NAMES = new Set([
    "group_name_helper",
    "workflow_group_hint",
    "workflow_group_create",
    "workflow_group_duplicate",
    "workflow_group_rename",
    "workflow_group_delete",
    "workflow_group_refresh",
    "workflow_group_capture",
    "workflow_group_apply",
]);

function injectStyles() {
    if (document.getElementById(STYLE_ID)) {
        return;
    }

    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
        .wg-shell {
            box-sizing: border-box;
            width: 100%;
            height: 100%;
            padding: 8px 10px 10px;
            display: flex;
            flex-direction: column;
            gap: 8px;
            color: #edf2ff;
            font-family: "Segoe UI", "Microsoft YaHei", sans-serif;
            background:
                radial-gradient(circle at top right, rgba(93, 132, 245, 0.18), transparent 34%),
                linear-gradient(180deg, rgba(29, 34, 48, 0.98), rgba(17, 20, 30, 0.98));
            border: 1px solid rgba(128, 145, 180, 0.22);
            border-radius: 14px;
            overflow: hidden;
        }
        .wg-shell.compact {
            padding: 8px 10px;
            gap: 6px;
        }
        .wg-header, .wg-toolbar, .wg-row, .wg-actions, .wg-item-row, .wg-inline {
            display: flex;
            gap: 8px;
            align-items: center;
        }
        .wg-header {
            justify-content: space-between;
        }
        .wg-title {
            display: flex;
            flex-direction: column;
            gap: 3px;
        }
        .wg-title strong {
            font-size: 14px;
            color: #ffffff;
        }
        .wg-title span, .wg-note, .wg-meta, .wg-empty, .wg-status {
            font-size: 11px;
            color: rgba(224, 232, 255, 0.72);
        }
        .wg-badge {
            padding: 4px 10px;
            border-radius: 999px;
            border: 1px solid rgba(132, 149, 186, 0.26);
            background: rgba(255, 255, 255, 0.05);
            font-size: 11px;
            color: #eef3ff;
            white-space: nowrap;
        }
        .wg-compact-summary {
            display: inline-flex;
            align-items: center;
            width: fit-content;
            max-width: 100%;
            padding: 0;
            border: none;
            background: transparent;
            color: #eef3ff;
            font-size: 12px;
            font-weight: 600;
            line-height: 1.4;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .wg-section {
            display: flex;
            flex-direction: column;
            gap: 8px;
            padding: 10px;
            border-radius: 12px;
            border: 1px solid rgba(140, 154, 184, 0.18);
            background: rgba(255, 255, 255, 0.05);
        }
        .wg-section.collapsed > :not(.wg-section-head) {
            display: none;
        }
        .wg-section-head {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 8px;
        }
        .wg-section-head strong {
            font-size: 12px;
            color: #ffffff;
        }
        .wg-collapse-btn {
            appearance: none;
            border: none;
            background: transparent;
            color: rgba(224, 232, 255, 0.86);
            cursor: pointer;
            font-size: 16px;
            line-height: 1;
            padding: 0 2px;
        }
        .wg-toolbar {
            display: grid;
            grid-template-columns: minmax(0, 1.2fr) auto auto;
        }
        .wg-grid, .wg-config-grid {
            display: grid;
            gap: 8px;
        }
        .wg-config-grid {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
        .wg-grid {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
        .wg-field {
            display: flex;
            flex-direction: column;
            gap: 5px;
            min-width: 0;
        }
        .wg-field.full {
            grid-column: 1 / -1;
        }
        .wg-field label {
            font-size: 11px;
            color: rgba(224, 232, 255, 0.8);
        }
        .wg-input, .wg-select, .wg-textarea {
            width: 100%;
            box-sizing: border-box;
            border-radius: 10px;
            border: 1px solid rgba(134, 149, 181, 0.28);
            background: rgba(7, 11, 20, 0.88);
            color: #eef3ff;
            height: 34px;
            min-height: 34px;
            padding: 6px 10px;
            font-size: 11px;
        }
        .wg-input-wrap {
            position: relative;
            width: 100%;
        }
        .wg-input-wrap .wg-input {
            padding-right: 28px;
        }
        .wg-input-clear {
            position: absolute;
            right: 7px;
            top: 50%;
            transform: translateY(-50%);
            appearance: none;
            border: none;
            background: transparent;
            color: rgba(224, 232, 255, 0.52);
            width: 18px;
            height: 18px;
            padding: 0;
            border-radius: 50%;
            cursor: pointer;
            opacity: 0;
            pointer-events: none;
            transition: opacity 0.15s ease, background 0.15s ease, color 0.15s ease;
            font-size: 12px;
            line-height: 18px;
            text-align: center;
        }
        .wg-input-wrap:hover .wg-input-clear,
        .wg-input-wrap:focus-within .wg-input-clear {
            opacity: 1;
            pointer-events: auto;
        }
        .wg-input-clear:hover {
            background: rgba(255, 255, 255, 0.08);
            color: #ffffff;
        }
        .wg-input-clear.hidden {
            opacity: 0;
            pointer-events: none;
        }
        .wg-select {
            appearance: none;
            -webkit-appearance: none;
            -moz-appearance: none;
            padding-right: 30px;
            background-image:
                linear-gradient(45deg, transparent 50%, rgba(224, 232, 255, 0.82) 50%),
                linear-gradient(135deg, rgba(224, 232, 255, 0.82) 50%, transparent 50%);
            background-position:
                calc(100% - 16px) calc(50% - 2px),
                calc(100% - 11px) calc(50% - 2px);
            background-size: 5px 5px, 5px 5px;
            background-repeat: no-repeat;
        }
        .wg-textarea {
            height: auto;
            min-height: 72px;
            resize: vertical;
            font-family: Consolas, monospace;
        }
        .wg-input:focus, .wg-select:focus, .wg-textarea:focus {
            outline: none;
            border-color: rgba(124, 168, 255, 0.9);
            box-shadow: 0 0 0 2px rgba(124, 168, 255, 0.15);
        }
        .wg-btn {
            appearance: none;
            border: 1px solid rgba(134, 149, 181, 0.28);
            background: rgba(255, 255, 255, 0.05);
            color: #eef3ff;
            border-radius: 10px;
            height: 34px;
            min-height: 34px;
            padding: 6px 10px;
            font-size: 11px;
            cursor: pointer;
            transition: all 0.15s ease;
            box-sizing: border-box;
        }
        .wg-btn:hover {
            border-color: rgba(124, 168, 255, 0.82);
            background: rgba(82, 124, 229, 0.16);
        }
        .wg-btn.primary {
            background: linear-gradient(180deg, rgba(89, 136, 255, 0.34), rgba(53, 94, 179, 0.42));
            border-color: rgba(124, 168, 255, 0.62);
        }
        .wg-btn.danger {
            background: rgba(195, 70, 70, 0.14);
            border-color: rgba(234, 124, 124, 0.28);
        }
        .wg-btn.ghost {
            background: transparent;
        }
        .wg-btn[disabled] {
            opacity: 0.6;
            cursor: not-allowed;
        }
        .wg-list {
            display: flex;
            flex-direction: column;
            gap: 6px;
            max-height: 190px;
            overflow-y: auto;
            padding-right: 4px;
        }
        .wg-section.wg-fill {
            flex: 1;
            min-height: 0;
        }
        .wg-section.wg-fill .wg-list {
            flex: 1;
            min-height: 0;
            max-height: none;
        }
        .wg-item-row {
            align-items: flex-start;
            padding: 8px 10px;
            border-radius: 10px;
            border: 1px solid rgba(132, 149, 186, 0.18);
            background: rgba(255, 255, 255, 0.04);
            cursor: pointer;
        }
        .wg-item-row.active {
            border-color: rgba(124, 168, 255, 0.82);
            background: rgba(93, 132, 245, 0.18);
        }
        .wg-item-main {
            flex: 1;
            min-width: 0;
            display: flex;
            flex-direction: column;
            gap: 3px;
        }
        .wg-item-title {
            font-size: 12px;
            color: #ffffff;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .wg-item-sub {
            font-size: 11px;
            color: rgba(224, 232, 255, 0.72);
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .wg-item-value {
            font-size: 11px;
            color: rgba(190, 219, 255, 0.88);
            word-break: break-all;
        }
        .wg-check {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            font-size: 11px;
            color: rgba(224, 232, 255, 0.84);
        }
        .wg-status.success {
            color: #b8f5c2;
        }
        .wg-status.error {
            color: #ffbbbb;
        }
        .wg-status.warning {
            color: #ffe09f;
        }
        .wg-empty {
            padding: 10px 12px;
            border-radius: 10px;
            border: 1px dashed rgba(132, 149, 186, 0.28);
            background: rgba(255, 255, 255, 0.04);
        }
        .wg-compact-status {
            padding: 8px 10px;
            border-radius: 10px;
            border: 1px solid rgba(132, 149, 186, 0.18);
            background: rgba(255, 255, 255, 0.04);
            font-size: 11px;
            color: rgba(224, 232, 255, 0.82);
            line-height: 1.5;
        }
        .wg-compact-grid {
            display: grid;
            grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
            gap: 8px;
        }
        .wg-manager-root {
            display: flex;
            flex-direction: column;
            gap: 10px;
            width: 100%;
            height: 100%;
            min-height: 0;
            overflow: hidden;
        }
        .wg-floating-panel {
            position: fixed;
            left: 120px;
            top: 80px;
            width: 980px;
            height: 720px;
            min-width: 560px;
            min-height: 360px;
            display: flex;
            flex-direction: column;
            gap: 0;
            padding: 12px;
            box-sizing: border-box;
            border-radius: 16px;
            border: 1px solid rgba(128, 145, 180, 0.24);
            background:
                radial-gradient(circle at top right, rgba(93, 132, 245, 0.14), transparent 34%),
                linear-gradient(180deg, rgba(24, 28, 40, 0.98), rgba(15, 18, 27, 0.98));
            box-shadow: 0 16px 40px rgba(0, 0, 0, 0.42);
            z-index: 9999;
            resize: both;
            overflow: hidden;
        }
        .wg-floating-panel-head {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
            padding: 0 0 10px;
            margin-bottom: 10px;
            border-bottom: 1px solid rgba(128, 145, 180, 0.18);
            cursor: move;
            user-select: none;
        }
        .wg-floating-panel-title {
            display: flex;
            flex-direction: column;
            gap: 2px;
            min-width: 0;
        }
        .wg-floating-panel-title strong {
            font-size: 13px;
            color: #ffffff;
        }
        .wg-floating-panel-title span {
            font-size: 11px;
            color: rgba(224, 232, 255, 0.66);
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .wg-floating-panel-tools {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            flex-shrink: 0;
        }
        .wg-floating-panel-body {
            flex: 1;
            min-height: 0;
            overflow: hidden;
        }
        .wg-panel-tool-btn {
            appearance: none;
            border: none;
            background: transparent;
            color: rgba(238, 243, 255, 0.88);
            border-radius: 8px;
            min-width: 28px;
            min-height: 28px;
            padding: 0 8px;
            font-size: 12px;
            cursor: pointer;
            transition: background 0.15s ease, color 0.15s ease;
        }
        .wg-panel-tool-btn:hover {
            background: rgba(255, 255, 255, 0.08);
            color: #ffffff;
        }
        .wg-manager-main {
            display: grid;
            grid-template-columns: minmax(320px, 0.9fr) minmax(420px, 1.1fr);
            gap: 10px;
            min-height: 0;
            overflow: hidden;
        }
        .wg-manager-col {
            display: flex;
            flex-direction: column;
            gap: 10px;
            min-height: 0;
            overflow-y: auto;
            padding-right: 4px;
        }
        .wg-manager-toolbar {
            display: grid;
            grid-template-columns: minmax(0, 1fr) auto auto;
            gap: 8px;
            align-items: center;
        }
        .wg-stack {
            display: flex;
            flex-direction: column;
            gap: 8px;
        }
        .wg-value-list {
            display: flex;
            flex-direction: column;
            gap: 8px;
        }
        .wg-value-row {
            display: grid;
            grid-template-columns: minmax(0, 1.2fr) 110px minmax(0, 1.4fr) auto auto;
            gap: 8px;
            align-items: center;
            padding: 8px;
            border-radius: 10px;
            border: 1px solid rgba(132, 149, 186, 0.18);
            background: rgba(255, 255, 255, 0.04);
        }
        .wg-inline-note {
            font-size: 10px;
            color: rgba(224, 232, 255, 0.62);
        }
        .wg-item-row.invalid, .wg-value-row.invalid {
            border-color: rgba(255, 132, 132, 0.72);
            background: rgba(160, 40, 40, 0.16);
        }
        .wg-item-row.invalid .wg-item-title, .wg-value-row.invalid .wg-inline-note {
            color: #ffd3d3;
        }
        .wg-item-tools, .wg-value-tools {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            flex-wrap: wrap;
        }
        .wg-mini-btn {
            appearance: none;
            border: none;
            background: transparent;
            color: rgba(238, 243, 255, 0.9);
            border-radius: 6px;
            min-width: 24px;
            min-height: 24px;
            padding: 0 4px;
            font-size: 13px;
            cursor: pointer;
            transition: background 0.15s ease, color 0.15s ease;
        }
        .wg-mini-btn:hover:not(:disabled) {
            background: rgba(255, 255, 255, 0.08);
            color: #ffffff;
        }
        .wg-mini-btn:disabled {
            opacity: 0.4;
            cursor: not-allowed;
        }
        .wg-mini-btn.danger {
            color: rgba(255, 170, 170, 0.92);
        }
        .wg-mini-btn.danger:hover:not(:disabled) {
            background: rgba(255, 90, 90, 0.14);
            color: #ffd7d7;
        }
    `;
    document.head.appendChild(style);
}

function randomSuffix() {
    return Math.random().toString(16).slice(2, 8);
}

function makeGroupId() {
    return `group_${Date.now().toString(16)}_${randomSuffix()}`;
}

function makeItemId() {
    return `item_${Date.now().toString(16)}_${randomSuffix()}`;
}

function makeValueId() {
    return `value_${Date.now().toString(16)}_${randomSuffix()}`;
}

function setWidgetValue(widget, value) {
    if (!widget) {
        return;
    }
    widget.value = value;
    if (typeof widget.callback === "function") {
        widget.callback(value);
    }
}

function hideWidget(widget) {
    if (!widget) {
        return;
    }
    widget.computeSize = () => [0, 0];
    widget.type = "hidden";
    widget.hidden = true;
    widget.options = { ...(widget.options || {}), hidden: true };
}

function removeLegacyWidgets(node) {
    if (!Array.isArray(node.widgets)) {
        return;
    }
    for (let index = node.widgets.length - 1; index >= 0; index -= 1) {
        if (LEGACY_WIDGET_NAMES.has(node.widgets[index]?.name)) {
            node.widgets.splice(index, 1);
        }
    }
}

function normalizeDb(db) {
    const groups = Array.isArray(db?.groups) ? db.groups : [];
    return {
        version: Number(db?.version) || 1,
        groups: groups.map((group, groupIndex) => ({
            id: String(group?.id || makeGroupId()),
            name: String(group?.name || `未命名组_${groupIndex + 1}`),
            enabled: group?.enabled !== false,
            version: Number(group?.version) || 1,
            items: Array.isArray(group?.items)
                ? group.items.map((item) => ({
                    id: String(item?.id || makeItemId()),
                    target_node_id: String(item?.target_node_id || ""),
                    target_node_title: String(item?.target_node_title || ""),
                    target_class_type: String(item?.target_class_type || ""),
                    apply_mode: String(item?.apply_mode || "set_widget_values"),
                    node_state: String(item?.node_state || "normal"),
                    enabled: item?.enabled !== false,
                    note: String(item?.note || ""),
                    values: Array.isArray(item?.values)
                        ? item.values.map((valueEntry) => ({
                            id: String(valueEntry?.id || makeValueId()),
                            target_input_name: String(valueEntry?.target_input_name || ""),
                            value_type: String(valueEntry?.value_type || "STRING").toUpperCase(),
                            value: valueEntry?.value,
                            enabled: valueEntry?.enabled !== false,
                            note: String(valueEntry?.note || ""),
                        }))
                        : item?.target_input_name
                            ? [{
                                id: makeValueId(),
                                target_input_name: String(item?.target_input_name || ""),
                                value_type: String(item?.value_type || "STRING").toUpperCase(),
                                value: item?.value,
                                enabled: item?.enabled !== false,
                                note: String(item?.note || ""),
                            }]
                            : [],
                }))
                : [],
        })),
    };
}

async function fetchWorkflowGroupsDb() {
    const response = await api.fetchApi(ENDPOINT);
    if (!response.ok) {
        throw new Error(`读取分组失败: ${response.status}`);
    }
    return normalizeDb(await response.json());
}

async function saveWorkflowGroupsDb(db) {
    const response = await api.fetchApi(ENDPOINT, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(normalizeDb(db)),
    });
    let payload = {};
    try {
        payload = await response.json();
    } catch (error) {
        payload = {};
    }
    if (!response.ok) {
        throw new Error(payload?.error || `保存分组失败: ${response.status}`);
    }
    return normalizeDb(payload.data || DEFAULT_DB);
}

function getGraph(node) {
    return node?.graph || app.graph;
}

function getSelectedNodes(node) {
    const graph = getGraph(node);
    if (!graph) {
        return [];
    }
    const selectedMap = app.canvas?.selected_nodes || {};
    return Object.values(selectedMap).filter((item) => item && item !== node);
}

function getWidget(node, name) {
    return node.widgets?.find((widget) => widget.name === name);
}

function getState(node) {
    return node?._workflowGroupState || null;
}

function normalizeManagerPanelLayout(layout) {
    if (!layout || typeof layout !== "object") {
        return null;
    }
    const left = Number(layout.left);
    const top = Number(layout.top);
    const width = Number(layout.width);
    const height = Number(layout.height);
    if (![left, top, width, height].every((value) => Number.isFinite(value))) {
        return null;
    }
    return {
        left,
        top,
        width,
        height,
    };
}

function loadManagerPanelLayout() {
    try {
        const raw = localStorage.getItem(MANAGER_PANEL_STORAGE_KEY);
        if (!raw) {
            return null;
        }
        return normalizeManagerPanelLayout(JSON.parse(raw));
    } catch {
        return null;
    }
}

function saveManagerPanelLayout(layout) {
    const normalized = normalizeManagerPanelLayout(layout);
    if (!normalized) {
        return;
    }
    try {
        localStorage.setItem(MANAGER_PANEL_STORAGE_KEY, JSON.stringify(normalized));
    } catch {
        // ignore storage failures
    }
}

function getManagerPanelLayout(panelElement) {
    if (!panelElement) {
        return null;
    }
    return {
        left: panelElement.offsetLeft,
        top: panelElement.offsetTop,
        width: panelElement.offsetWidth,
        height: panelElement.offsetHeight,
    };
}

function persistManagerPanelLayout(state) {
    const layout = getManagerPanelLayout(state?.managerPanelElement);
    if (layout) {
        saveManagerPanelLayout(layout);
        state.managerPanelLayout = layout;
    }
}

function closeManagerPanel(node) {
    const state = getState(node);
    if (!state?.managerModalVisible) {
        return;
    }
    persistManagerPanelLayout(state);
    state.managerPanelCleanup?.();
    if (state.managerPanelElement?.parentNode) {
        state.managerPanelElement.parentNode.removeChild(state.managerPanelElement);
    }
    state.managerPanelCleanup = null;
    state.managerPanelElement = null;
    state.managerModalVisible = false;
    state.managerModalContainer = null;
}

function attachManagerPanelInteractions(node, panelElement, dragHandle) {
    const state = getState(node);
    if (!state || !panelElement || !dragHandle) {
        return () => {};
    }

    let dragState = null;
    const onPointerMove = (event) => {
        if (!dragState) {
            return;
        }
        panelElement.style.left = `${dragState.startLeft + (event.clientX - dragState.startX)}px`;
        panelElement.style.top = `${dragState.startTop + (event.clientY - dragState.startY)}px`;
    };
    const onPointerUp = () => {
        if (!dragState) {
            return;
        }
        dragState = null;
        document.removeEventListener("pointermove", onPointerMove);
        document.removeEventListener("pointerup", onPointerUp);
        persistManagerPanelLayout(state);
    };
    const onPointerDown = (event) => {
        if (event.button !== 0) {
            return;
        }
        if (event.target.closest("button, input, select, textarea, label")) {
            return;
        }
        dragState = {
            startX: event.clientX,
            startY: event.clientY,
            startLeft: panelElement.offsetLeft,
            startTop: panelElement.offsetTop,
        };
        document.addEventListener("pointermove", onPointerMove);
        document.addEventListener("pointerup", onPointerUp);
    };
    dragHandle.addEventListener("pointerdown", onPointerDown);

    const resizeObserver = typeof ResizeObserver === "function"
        ? new ResizeObserver(() => {
            persistManagerPanelLayout(state);
        })
        : null;
    resizeObserver?.observe(panelElement);

    return () => {
        dragHandle.removeEventListener("pointerdown", onPointerDown);
        document.removeEventListener("pointermove", onPointerMove);
        document.removeEventListener("pointerup", onPointerUp);
        resizeObserver?.disconnect();
    };
}

function getSelectableNodes(node) {
    const graph = getGraph(node);
    if (!graph) {
        return [];
    }
    const nodes = graph.nodes || [];
    return nodes
        .filter((entry) => {
            if (!entry || entry.id === node.id) {
                return false;
            }
            if (!entry.widgets || entry.widgets.length === 0) {
                return false;
            }
            return entry.widgets.some((widget) => isEditableWidget(widget));
        })
        .sort((a, b) => Number(a.id) - Number(b.id));
}

function getTargetNodeById(node, targetNodeId) {
    const graph = getGraph(node);
    if (!graph) {
        return null;
    }
    return graph._nodes_by_id?.[targetNodeId] || graph.nodes?.find((entry) => String(entry.id) === String(targetNodeId)) || null;
}

function getEditableInputNamesFromNode(targetNode) {
    if (!targetNode?.widgets) {
        return [];
    }
    return targetNode.widgets
        .filter((widget) => isEditableWidget(widget))
        .map((widget) => widget.name);
}

function getComboOptionValues(widget) {
    const values = widget?.options?.values;
    if (Array.isArray(values)) {
        return values.map((entry) => {
            if (entry && typeof entry === "object") {
                if ("value" in entry) {
                    return entry.value;
                }
                if ("content" in entry) {
                    return entry.content;
                }
            }
            return entry;
        });
    }
    if (values && typeof values === "object") {
        return Object.keys(values);
    }
    return [];
}

function getNumericWidgetOptions(widget, valueType) {
    const options = widget?.options || {};
    const next = {};
    if (options.min != null && Number.isFinite(Number(options.min))) {
        next.min = String(options.min);
    }
    if (options.max != null && Number.isFinite(Number(options.max))) {
        next.max = String(options.max);
    }
    if (options.step != null && Number.isFinite(Number(options.step))) {
        next.step = String(options.step);
    } else if (valueType === "INT") {
        next.step = "1";
    } else if (valueType === "FLOAT") {
        next.step = "0.01";
    }
    return next;
}

function createClearableInput(input, onClear) {
    const wrapper = document.createElement("div");
    wrapper.className = "wg-input-wrap";
    const clearBtn = document.createElement("button");
    clearBtn.type = "button";
    clearBtn.className = "wg-input-clear hidden";
    clearBtn.textContent = "x";
    const syncClearBtn = () => {
        clearBtn.classList.toggle("hidden", !String(input.value ?? ""));
    };
    clearBtn.onclick = () => {
        input.value = "";
        if (typeof onClear === "function") {
            onClear();
        } else {
            input.dispatchEvent(new Event("input", { bubbles: true }));
            input.dispatchEvent(new Event("change", { bubbles: true }));
        }
        syncClearBtn();
        input.focus();
    };
    input.addEventListener("input", syncClearBtn);
    input.addEventListener("change", syncClearBtn);
    wrapper.appendChild(input);
    wrapper.appendChild(clearBtn);
    syncClearBtn();
    return wrapper;
}

function getItemInvalidReason(node, item) {
    if (!item?.target_node_id) {
        return "未设置目标节点 ID";
    }
    const targetNode = getTargetNodeById(node, item.target_node_id);
    if (!targetNode) {
        return `目标节点不存在: ${item.target_node_id}`;
    }
    return "";
}

function getValueEntryInvalidReason(node, item, valueEntry) {
    if (!item?.target_node_id) {
        return "节点条目未设置目标节点";
    }
    const targetNode = getTargetNodeById(node, item.target_node_id);
    if (!targetNode) {
        return `目标节点不存在: ${item.target_node_id}`;
    }
    const inputName = String(valueEntry?.target_input_name || "").trim();
    if (!inputName) {
        return "未设置输入项";
    }
    const targetWidget = findTargetWidgetByName(targetNode, inputName);
    if (!targetWidget) {
        return `输入项不存在: ${inputName}`;
    }
    return "";
}

function matchesKeyword(text, keyword) {
    const normalizedKeyword = String(keyword || "").trim().toLowerCase();
    if (!normalizedKeyword) {
        return true;
    }
    return String(text || "").toLowerCase().includes(normalizedKeyword);
}

function getCurrentGroup(state) {
    return state?.db?.groups?.find((group) => group.name === state.selectedGroupName) || null;
}

function getCurrentItem(state) {
    const currentGroup = getCurrentGroup(state);
    return currentGroup?.items?.find((item) => item.id === state.selectedItemId) || null;
}

function ensureSelection(state) {
    const currentGroup = getCurrentGroup(state);
    if (!currentGroup) {
        state.selectedItemId = "";
        return;
    }
    if (!currentGroup.items.some((item) => item.id === state.selectedItemId)) {
        state.selectedItemId = currentGroup.items[0]?.id || "";
    }
}

function setStatus(node, type, text) {
    const state = getState(node);
    if (!state) {
        return;
    }
    state.message = text ? { type, text } : null;
    renderNodePanel(node);
}

function triggerNodeRefresh(node) {
    node?.setDirtyCanvas?.(true, false);
    node?.graph?.setDirtyCanvas?.(true, false);
}

function isEditableWidget(widget) {
    if (!widget || !widget.name) {
        return false;
    }
    if (widget.type === "button" || widget.type === "hidden" || widget.type === "converted-widget") {
        return false;
    }
    if (widget.name.startsWith("wg_") || widget.name.endsWith("_helper")) {
        return false;
    }
    return true;
}

function inferValueType(widget, value) {
    if (Array.isArray(widget?.options?.values)) {
        return "COMBO";
    }
    if (typeof value === "boolean") {
        return "BOOLEAN";
    }
    if (typeof value === "number") {
        return Number.isInteger(value) ? "INT" : "FLOAT";
    }
    if (typeof value === "string") {
        return "STRING";
    }
    return "JSON_STRING";
}

function serializeWidgetValue(widget) {
    const value = widget?.value;
    const valueType = inferValueType(widget, value);
    if (valueType === "JSON_STRING") {
        try {
            return {
                value_type: valueType,
                value: JSON.stringify(value),
            };
        } catch (error) {
            return {
                value_type: "STRING",
                value: String(value),
            };
        }
    }
    return {
        value_type: valueType,
        value,
    };
}

function deserializeValue(item, currentValue) {
    const valueType = String(item?.value_type || "STRING").toUpperCase();
    const rawValue = item?.value;
    if (valueType === "INT") {
        return parseInt(rawValue, 10);
    }
    if (valueType === "FLOAT") {
        return parseFloat(rawValue);
    }
    if (valueType === "BOOLEAN") {
        if (typeof rawValue === "boolean") {
            return rawValue;
        }
        return ["1", "true", "yes", "on"].includes(String(rawValue).toLowerCase());
    }
    if (valueType === "JSON_STRING") {
        if (typeof currentValue === "object" && currentValue !== null && typeof rawValue === "string") {
            try {
                return JSON.parse(rawValue);
            } catch (error) {
                return rawValue;
            }
        }
        return rawValue;
    }
    return rawValue == null ? "" : rawValue;
}

function extractNodeIdFromText(text) {
    if (!text) {
        return "";
    }
    const match = String(text).match(/^(\d+)\s*-?/);
    return match ? match[1] : String(text).trim();
}

function findTargetWidgetByName(targetNode, inputName) {
    if (!targetNode?.widgets || !inputName) {
        return null;
    }
    return targetNode.widgets.find((widget) => widget.name === inputName) || null;
}

function applyValueToWidget(targetNode, targetWidget, nextValue) {
    if (!targetNode || !targetWidget) {
        return false;
    }

    targetWidget.value = nextValue;

    if (targetWidget.inputEl) {
        targetWidget.inputEl.value = nextValue == null ? "" : String(nextValue);
    }

    if (typeof targetWidget.callback === "function") {
        targetWidget.callback(nextValue);
    }

    if (typeof targetNode.onWidgetChanged === "function") {
        targetNode.onWidgetChanged(targetWidget.name, nextValue, nextValue, targetWidget);
    }

    targetNode.setDirtyCanvas?.(true, true);
    return true;
}

function formatItemValue(item) {
    const values = Array.isArray(item?.values)
        ? item.values.filter((entry) => entry?.enabled !== false)
        : [];
    if (!values.length) {
        return "";
    }
    const summary = values.slice(0, 3).map((entry) => `${entry.target_input_name}=${entry.value}`).join(" | ");
    return values.length > 3 ? `${summary} ... 共 ${values.length} 项` : summary;
}

function defaultItemFromSelection(node) {
    const selectedNode = getSelectedNodes(node)[0];
    return {
        id: makeItemId(),
        target_node_id: String(selectedNode?.id || ""),
        target_node_title: String(selectedNode?.title || selectedNode?.type || ""),
        target_class_type: String(selectedNode?.type || ""),
        apply_mode: "set_widget_values",
        node_state: "normal",
        enabled: true,
        note: "",
        values: [],
    };
}

function createEmptyValueEntry(inputName = "") {
    return {
        id: makeValueId(),
        target_input_name: String(inputName || ""),
        value_type: "STRING",
        value: "",
        enabled: true,
        note: "",
    };
}

function createValueEntryFromWidget(widget) {
    const serialized = serializeWidgetValue(widget);
    return {
        id: makeValueId(),
        target_input_name: String(widget?.name || ""),
        value_type: serialized.value_type,
        value: serialized.value,
        enabled: true,
        note: "",
    };
}

function countEnabledValueEntries(item) {
    const values = Array.isArray(item?.values) ? item.values : [];
    return values.filter((entry) => entry?.enabled !== false).length;
}

function countGroupValueEntries(group) {
    return (group?.items || []).reduce((total, item) => total + countEnabledValueEntries(item), 0);
}

function parseValueByType(valueType, rawText) {
    const text = String(rawText ?? "");
    if (valueType === "INT") {
        const parsed = parseInt(text, 10);
        if (!Number.isFinite(parsed)) {
            throw new Error("INT 值无效");
        }
        return parsed;
    }
    if (valueType === "FLOAT") {
        const parsed = parseFloat(text);
        if (!Number.isFinite(parsed)) {
            throw new Error("FLOAT 值无效");
        }
        return parsed;
    }
    if (valueType === "BOOLEAN") {
        const lowered = text.trim().toLowerCase();
        if (["true", "1", "yes", "on"].includes(lowered)) {
            return true;
        }
        if (["false", "0", "no", "off"].includes(lowered)) {
            return false;
        }
        throw new Error("BOOLEAN 仅支持 true/false/1/0/yes/no/on/off");
    }
    if (valueType === "JSON_STRING") {
        if (text.trim()) {
            JSON.parse(text);
        }
        return text;
    }
    return text;
}

function collectItemsFromSelectedNodes(node) {
    const selectedNodes = getSelectedNodes(node);
    const items = [];

    selectedNodes.forEach((selectedNode) => {
        const editableWidgets = (selectedNode.widgets || []).filter(isEditableWidget);
        if (!editableWidgets.length) {
            return;
        }
        items.push({
            id: `item_${selectedNode.id}_${randomSuffix()}`,
            target_node_id: String(selectedNode.id),
            target_node_title: String(selectedNode.title || selectedNode.type || ""),
            target_class_type: String(selectedNode.type || ""),
            apply_mode: "set_widget_values",
            node_state: "normal",
            enabled: true,
            note: "",
            values: editableWidgets.map((widget) => createValueEntryFromWidget(widget)),
        });
    });

    return items;
}

function updateEditorDraftFromSelection(state) {
    const currentItem = getCurrentItem(state);
    if (!currentItem) {
        state.editor = defaultItemFromSelection(state.node);
        return;
    }
    state.editor = JSON.parse(JSON.stringify(currentItem));
}

function moveArrayItem(list, fromIndex, toIndex) {
    if (!Array.isArray(list)) {
        return list;
    }
    if (fromIndex < 0 || fromIndex >= list.length || toIndex < 0 || toIndex >= list.length || fromIndex === toIndex) {
        return list;
    }
    const nextList = [...list];
    const [moved] = nextList.splice(fromIndex, 1);
    nextList.splice(toIndex, 0, moved);
    return nextList;
}

function ensureCollapseState(state) {
    if (!state.sectionCollapsed) {
        state.sectionCollapsed = {
            config: false,
            groups: false,
            actions: false,
            items: false,
            editor: false,
        };
    }
}

function syncBackingWidgets(node) {
    const state = getState(node);
    if (!state) {
        return;
    }

    ensureSelection(state);
    setWidgetValue(state.groupNameWidget, state.selectedGroupName || "");
    setWidgetValue(state.autoApplyWidget, !!state.autoApply);
    setWidgetValue(state.applyScopeWidget, String(state.applyScope || "values_only"));
    setWidgetValue(state.fallbackModeWidget, String(state.fallbackMode || "warn_missing"));
    setWidgetValue(state.syncUiPreviewWidget, !!state.syncUiPreview);
}

async function refreshGroups(node, options = {}) {
    const { silent = false } = options;
    const state = getState(node);
    if (!state) {
        return DEFAULT_DB;
    }

    try {
        state.loading = true;
        renderNodePanel(node);

        state.db = await fetchWorkflowGroupsDb();
        if (!state.selectedGroupName || !state.db.groups.some((group) => group.name === state.selectedGroupName)) {
            state.selectedGroupName = state.groupNameWidget.value || state.db.groups[0]?.name || "";
        }
        ensureSelection(state);
        updateEditorDraftFromSelection(state);
        state.message = silent ? state.message : { type: "success", text: "已从后端刷新分组数据。" };
        syncBackingWidgets(node);
        return state.db;
    } catch (error) {
        console.error("[WorkflowGroupPresetManager] 刷新分组失败", error);
        if (!silent) {
            state.message = { type: "error", text: error.message || "刷新分组失败" };
        }
        return state.db;
    } finally {
        state.loading = false;
        renderNodePanel(node);
    }
}

async function persistDb(node, db, successText = "") {
    const state = getState(node);
    if (!state) {
        return;
    }

    try {
        state.loading = true;
        renderNodePanel(node);
        state.db = await saveWorkflowGroupsDb(db);
        if (!state.db.groups.some((group) => group.name === state.selectedGroupName)) {
            state.selectedGroupName = state.db.groups[0]?.name || "";
        }
        ensureSelection(state);
        updateEditorDraftFromSelection(state);
        state.message = successText ? { type: "success", text: successText } : null;
        syncBackingWidgets(node);
    } catch (error) {
        console.error("[WorkflowGroupPresetManager] 保存分组失败", error);
        state.message = { type: "error", text: error.message || "保存分组失败" };
    } finally {
        state.loading = false;
        renderNodePanel(node);
    }
}

async function withUpdatedDb(node, updater, successText = "") {
    const state = getState(node);
    if (!state) {
        return;
    }
    const draftDb = normalizeDb(JSON.parse(JSON.stringify(state.db || DEFAULT_DB)));
    const changed = await updater(draftDb);
    if (!changed) {
        return;
    }
    await persistDb(node, draftDb, successText);
}

function showConfirmDialog({ title, content, onConfirm, danger = false }) {
    confirmModal.show({
        title,
        content: `<div style="font-size:14px;line-height:1.6;">${content}</div>`,
        width: "420px",
        buttons: [
            { text: "取消", onClick: () => confirmModal.close() },
            {
                text: "确认",
                type: danger ? "danger" : "primary",
                onClick: async () => {
                    await onConfirm?.();
                    confirmModal.close();
                },
            },
        ],
    });
}

async function createGroup(node) {
    const state = getState(node);
    const name = String(state?.groupDraftName || "").trim();
    if (!name) {
        setStatus(node, "warning", "请先输入新组名。");
        return;
    }

    await withUpdatedDb(node, (db) => {
        if (db.groups.some((group) => group.name === name)) {
            setStatus(node, "error", `组名已存在: ${name}`);
            return false;
        }
        db.groups.push({
            id: makeGroupId(),
            name,
            enabled: true,
            version: 1,
            items: [],
        });
        state.selectedGroupName = name;
        state.groupDraftName = name;
        state.selectedItemId = "";
        state.editor = defaultItemFromSelection(node);
        return true;
    }, `已创建组「${name}」。`);
}

async function renameCurrentGroup(node) {
    const state = getState(node);
    const currentGroup = getCurrentGroup(state);
    if (!currentGroup) {
        setStatus(node, "warning", "请先选择一个组。");
        return;
    }

    const nextName = String(state.groupDraftName || "").trim();
    if (!nextName) {
        setStatus(node, "warning", "请输入新的组名。");
        return;
    }
    if (nextName === currentGroup.name) {
        setStatus(node, "warning", "组名未发生变化。");
        return;
    }

    await withUpdatedDb(node, (db) => {
        if (db.groups.some((group) => group.name === nextName)) {
            setStatus(node, "error", `组名已存在: ${nextName}`);
            return false;
        }
        const group = db.groups.find((item) => item.id === currentGroup.id);
        if (!group) {
            setStatus(node, "error", "当前组已不存在，请先刷新。");
            return false;
        }
        group.name = nextName;
        group.version = Number(group.version || 1) + 1;
        state.selectedGroupName = nextName;
        state.groupDraftName = nextName;
        return true;
    }, `已重命名为「${nextName}」。`);
}

async function duplicateCurrentGroup(node) {
    const state = getState(node);
    const currentGroup = getCurrentGroup(state);
    if (!currentGroup) {
        setStatus(node, "warning", "请先选择一个组。");
        return;
    }

    const nextName = String(state.groupDraftName || "").trim() || `${currentGroup.name}_副本`;
    await withUpdatedDb(node, (db) => {
        if (db.groups.some((group) => group.name === nextName)) {
            setStatus(node, "error", `组名已存在: ${nextName}`);
            return false;
        }
        const clone = JSON.parse(JSON.stringify(currentGroup));
        clone.id = makeGroupId();
        clone.name = nextName;
        clone.version = 1;
        clone.items = clone.items.map((item) => ({ ...item, id: makeItemId() }));
        db.groups.push(clone);
        state.selectedGroupName = nextName;
        state.groupDraftName = nextName;
        state.selectedItemId = clone.items[0]?.id || "";
        return true;
    }, `已复制组为「${nextName}」。`);
}

async function deleteCurrentGroup(node) {
    const state = getState(node);
    const currentGroup = getCurrentGroup(state);
    if (!currentGroup) {
        setStatus(node, "warning", "请先选择一个组。");
        return;
    }

    showConfirmDialog({
        title: "删除组",
        content: `确认删除组「${currentGroup.name}」吗？此操作会删除该组下所有条目。`,
        danger: true,
        onConfirm: async () => {
            await withUpdatedDb(node, (db) => {
                const beforeLength = db.groups.length;
                db.groups = db.groups.filter((group) => group.id !== currentGroup.id);
                if (db.groups.length === beforeLength) {
                    return false;
                }
                state.selectedGroupName = db.groups[0]?.name || "";
                state.groupDraftName = state.selectedGroupName;
                state.selectedItemId = "";
                state.editor = defaultItemFromSelection(node);
                return true;
            }, `已删除组「${currentGroup.name}」。`);
        },
    });
}

async function captureSelectedNodesToCurrentGroup(node, replaceItems = true) {
    const state = getState(node);
    const currentGroup = getCurrentGroup(state);
    if (!currentGroup) {
        setStatus(node, "warning", "请先创建并选择一个组。");
        return;
    }

    const nextItems = collectItemsFromSelectedNodes(node);
    if (!nextItems.length) {
        setStatus(node, "warning", "请先在画布上选中至少一个含可编辑参数的目标节点。");
        return;
    }

    await withUpdatedDb(node, (db) => {
        const group = db.groups.find((item) => item.id === currentGroup.id);
        if (!group) {
            setStatus(node, "error", "当前组已不存在，请先刷新。");
            return false;
        }

        if (replaceItems) {
            group.items = nextItems;
        } else {
            const merged = [...group.items];
            nextItems.forEach((incoming) => {
                const index = merged.findIndex((item) => item.target_node_id === incoming.target_node_id);
                if (index >= 0) {
                    const existing = merged[index];
                    const nextValueMap = new Map(
                        (existing.values || []).map((valueEntry) => [valueEntry.target_input_name, { ...valueEntry }])
                    );
                    (incoming.values || []).forEach((valueEntry) => {
                        const existingEntry = nextValueMap.get(valueEntry.target_input_name);
                        nextValueMap.set(
                            valueEntry.target_input_name,
                            existingEntry ? { ...existingEntry, ...valueEntry, id: existingEntry.id } : valueEntry
                        );
                    });
                    merged[index] = {
                        ...existing,
                        ...incoming,
                        id: existing.id,
                        values: Array.from(nextValueMap.values()),
                    };
                } else {
                    merged.push(incoming);
                }
            });
            group.items = merged;
        }

        group.version = Number(group.version || 1) + 1;
        state.selectedItemId = group.items[0]?.id || "";
        updateEditorDraftFromSelection(state);
        return true;
    }, replaceItems
        ? `已覆盖采集到组「${currentGroup.name}」，共 ${nextItems.length} 个条目。`
        : `已追加/更新 ${nextItems.length} 个条目到组「${currentGroup.name}」。`);
}

async function createBlankItem(node) {
    const state = getState(node);
    const currentGroup = getCurrentGroup(state);
    if (!currentGroup) {
        setStatus(node, "warning", "请先创建并选择一个组。");
        return;
    }

    await withUpdatedDb(node, (db) => {
        const group = db.groups.find((item) => item.id === currentGroup.id);
        if (!group) {
            return false;
        }
        const newItem = defaultItemFromSelection(node);
        group.items.push(newItem);
        group.version = Number(group.version || 1) + 1;
        state.selectedItemId = newItem.id;
        state.editor = JSON.parse(JSON.stringify(newItem));
        return true;
    }, "已新增空白条目。");
}

function moveEditorValueEntry(node, valueEntryId, direction) {
    const state = getState(node);
    if (!state?.editor || !Array.isArray(state.editor.values)) {
        return;
    }
    const currentIndex = state.editor.values.findIndex((entry) => entry.id === valueEntryId);
    if (currentIndex === -1) {
        return;
    }
    const nextIndex = direction === "up" ? currentIndex - 1 : currentIndex + 1;
    state.editor.values = moveArrayItem(state.editor.values, currentIndex, nextIndex);
    renderNodePanel(node);
}

function resyncEditorValuesFromCurrentNode(node) {
    const state = getState(node);
    const editor = state?.editor;
    if (!editor?.target_node_id) {
        setStatus(node, "warning", "请先选择目标节点，再执行重同步。");
        return;
    }

    const targetNode = getTargetNodeById(node, editor.target_node_id);
    if (!targetNode) {
        setStatus(node, "error", `找不到目标节点: ${editor.target_node_id}`);
        return;
    }

    const editableWidgets = (targetNode.widgets || []).filter(isEditableWidget);
    if (!editableWidgets.length) {
        setStatus(node, "warning", "目标节点没有可同步的参数值。");
        return;
    }

    editor.target_node_title = String(targetNode.title || targetNode.type || "");
    editor.target_class_type = String(targetNode.type || "");
    editor.values = editableWidgets.map((widget) => createValueEntryFromWidget(widget));
    setStatus(node, "success", `已从节点 ${editor.target_node_id} 重同步 ${editor.values.length} 个参数值。`);
    renderNodePanel(node);
}

async function saveCurrentItem(node) {
    const state = getState(node);
    const currentGroup = getCurrentGroup(state);
    if (!currentGroup) {
        setStatus(node, "warning", "请先选择一个组。");
        return;
    }

    const editor = state.editor || defaultItemFromSelection(node);
    if (!String(editor.target_node_id || "").trim()) {
        setStatus(node, "warning", "请先选择目标节点。");
        return;
    }
    if (!Array.isArray(editor.values) || !editor.values.length) {
        setStatus(node, "warning", "当前节点条目还没有任何参数值。");
        return;
    }

    const normalizedValues = [];
    for (const valueEntry of editor.values) {
        const inputName = String(valueEntry?.target_input_name || "").trim();
        if (!inputName) {
            setStatus(node, "warning", "存在未填写输入名的参数值。");
            return;
        }
        let parsedValue;
        try {
            parsedValue = parseValueByType(valueEntry.value_type, valueEntry.value);
        } catch (error) {
            setStatus(node, "error", `${inputName}: ${error.message}`);
            return;
        }
        normalizedValues.push({
            id: valueEntry.id || makeValueId(),
            target_input_name: inputName,
            value_type: String(valueEntry.value_type || "STRING").toUpperCase(),
            value: parsedValue,
            enabled: valueEntry.enabled !== false,
            note: String(valueEntry.note || ""),
        });
    }

    await withUpdatedDb(node, (db) => {
        const group = db.groups.find((item) => item.id === currentGroup.id);
        if (!group) {
            return false;
        }
        const normalizedItem = {
            ...editor,
            id: editor.id || makeItemId(),
            target_node_id: String(editor.target_node_id || "").trim(),
            target_node_title: String(editor.target_node_title || "").trim(),
            target_class_type: String(editor.target_class_type || "").trim(),
            apply_mode: "set_widget_values",
            node_state: String(editor.node_state || "normal"),
            enabled: editor.enabled !== false,
            note: String(editor.note || ""),
            values: normalizedValues,
        };

        const index = group.items.findIndex((item) => item.id === normalizedItem.id);
        if (index >= 0) {
            group.items[index] = normalizedItem;
        } else {
            group.items.push(normalizedItem);
        }
        group.version = Number(group.version || 1) + 1;
        state.selectedItemId = normalizedItem.id;
        state.editor = JSON.parse(JSON.stringify(normalizedItem));
        return true;
    }, "已保存当前条目。");
}

async function removeCurrentItem(node, itemId = null) {
    const state = getState(node);
    const currentGroup = getCurrentGroup(state);
    if (!currentGroup) {
        setStatus(node, "warning", "请先选择一个组。");
        return;
    }

    const targetId = itemId || state.selectedItemId;
    if (!targetId) {
        setStatus(node, "warning", "请先选择一个条目。");
        return;
    }

    showConfirmDialog({
        title: "删除条目",
        content: "确认删除当前条目吗？",
        danger: true,
        onConfirm: async () => {
            await withUpdatedDb(node, (db) => {
                const group = db.groups.find((item) => item.id === currentGroup.id);
                if (!group) {
                    return false;
                }
                const beforeLength = group.items.length;
                group.items = group.items.filter((item) => item.id !== targetId);
                if (group.items.length === beforeLength) {
                    return false;
                }
                group.version = Number(group.version || 1) + 1;
                state.selectedItemId = group.items[0]?.id || "";
                updateEditorDraftFromSelection(state);
                return true;
            }, "已删除条目。");
        },
    });
}

async function toggleItemEnabled(node, itemId, enabled) {
    const state = getState(node);
    const currentGroup = getCurrentGroup(state);
    if (!currentGroup) {
        return;
    }

    await withUpdatedDb(node, (db) => {
        const group = db.groups.find((item) => item.id === currentGroup.id);
        const targetItem = group?.items?.find((item) => item.id === itemId);
        if (!targetItem) {
            return false;
        }
        targetItem.enabled = !!enabled;
        group.version = Number(group.version || 1) + 1;
        if (state.selectedItemId === itemId) {
            state.editor = JSON.parse(JSON.stringify(targetItem));
        }
        return true;
    }, enabled ? "已启用条目。" : "已禁用条目。");
}

function setSelectedGroup(node, groupName) {
    const state = getState(node);
    if (!state) {
        return;
    }
    const previousGroupName = state.selectedGroupName;
    state.selectedGroupName = groupName;
    state.groupDraftName = groupName;
    ensureSelection(state);
    updateEditorDraftFromSelection(state);
    syncBackingWidgets(node);
    renderNodePanel(node);

    if (
        groupName &&
        groupName !== previousGroupName &&
        state.autoApply &&
        state.syncUiPreview
    ) {
        applyCurrentGroupToCanvas(node);
    }
}

function setSelectedItem(node, itemId) {
    const state = getState(node);
    if (!state) {
        return;
    }
    state.selectedItemId = itemId;
    updateEditorDraftFromSelection(state);
    renderNodePanel(node);
}

async function applyCurrentGroupToCanvas(node) {
    const state = getState(node);
    const currentGroup = getCurrentGroup(state);
    if (!currentGroup) {
        setStatus(node, "warning", "请先选择一个组。");
        return;
    }

    const graph = getGraph(node);
    if (!graph) {
        setStatus(node, "error", "当前画布未就绪。");
        return;
    }

    let appliedCount = 0;
    let missingCount = 0;
    const missingReasons = [];
    currentGroup.items.forEach((item) => {
        if (!item.enabled) {
            return;
        }
        const targetNode =
            graph._nodes_by_id?.[item.target_node_id] ||
            graph.nodes?.find((entry) => String(entry.id) === String(item.target_node_id));
        if (!targetNode) {
            missingCount += 1;
            if (missingReasons.length < 5) {
                missingReasons.push(`节点不存在: ${item.target_node_id}`);
            }
            return;
        }

        const valueEntries = Array.isArray(item.values) ? item.values : [];
        valueEntries.forEach((valueEntry) => {
            if (valueEntry?.enabled === false) {
                return;
            }

            const targetWidget = findTargetWidgetByName(targetNode, valueEntry.target_input_name);
            if (!targetWidget) {
                missingCount += 1;
                if (missingReasons.length < 5) {
                    missingReasons.push(`输入不存在: ${item.target_node_id}.${valueEntry.target_input_name}`);
                }
                return;
            }

            const nextValue = deserializeValue(valueEntry, targetWidget.value);
            const applied = applyValueToWidget(targetNode, targetWidget, nextValue);
            if (!applied) {
                missingCount += 1;
                if (missingReasons.length < 5) {
                    missingReasons.push(`应用失败: ${item.target_node_id}.${valueEntry.target_input_name}`);
                }
                return;
            }

            appliedCount += 1;
        });
    });

    graph.setDirtyCanvas(true, true);
    const reasonText = missingReasons.length ? ` 原因示例: ${missingReasons.join("；")}` : "";
    setStatus(
        node,
        missingCount > 0 && appliedCount === 0 ? "warning" : "success",
        `已应用组「${currentGroup.name}」到画布，成功 ${appliedCount} 项，缺失/跳过 ${missingCount} 项。${reasonText}`
    );
}

function createButton(label, className, onClick, disabled = false) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = className;
    button.textContent = label;
    button.disabled = disabled;
    button.onclick = onClick;
    return button;
}

function createMiniButton(label, onClick, disabled = false, className = "wg-mini-btn") {
    const button = document.createElement("button");
    button.type = "button";
    button.className = className;
    button.textContent = label;
    button.disabled = disabled;
    button.onclick = onClick;
    return button;
}

function createField(labelText, element, isFull = false) {
    const wrapper = document.createElement("div");
    wrapper.className = `wg-field ${isFull ? "full" : ""}`.trim();
    const label = document.createElement("label");
    label.textContent = labelText;
    wrapper.appendChild(label);
    wrapper.appendChild(element);
    return wrapper;
}

function makeCollapsibleSection(node, state, key, section) {
    ensureCollapseState(state);
    const head = section.querySelector(".wg-section-head");
    if (!head) {
        return section;
    }

    const toggle = document.createElement("button");
    toggle.type = "button";
    toggle.className = "wg-collapse-btn";
    toggle.textContent = state.sectionCollapsed[key] ? "▸" : "▾";
    toggle.title = state.sectionCollapsed[key] ? "展开" : "折叠";
    toggle.onclick = (event) => {
        event.preventDefault();
        event.stopPropagation();
        state.sectionCollapsed[key] = !state.sectionCollapsed[key];
        renderNodePanel(node);
    };
    head.appendChild(toggle);

    if (key === "items") {
        section.classList.add("wg-fill");
    }

    if (state.sectionCollapsed[key]) {
        section.classList.add("collapsed");
    }
    return section;
}

function buildCompactPanel(node, state, currentGroup) {
    const root = document.createElement("div");
    root.className = "wg-shell compact";

    const compactSection = document.createElement("div");
    compactSection.className = "wg-section";

    const toolbar = document.createElement("div");
    toolbar.className = "wg-manager-toolbar";

    const groupSelect = document.createElement("select");
    groupSelect.className = "wg-select";
    const emptyOption = document.createElement("option");
    emptyOption.value = "";
    emptyOption.textContent = state.db.groups.length ? "请选择组" : "暂无组";
    groupSelect.appendChild(emptyOption);
    state.db.groups.forEach((group) => {
        const option = document.createElement("option");
        option.value = group.name;
        option.textContent = group.name;
        option.selected = group.name === state.selectedGroupName;
        groupSelect.appendChild(option);
    });
    groupSelect.onchange = () => setSelectedGroup(node, groupSelect.value);

    const manageBtn = createButton("打开管理界面", "wg-btn primary", () => openManagerModal(node), state.loading);
    const applyBtn = createButton("应用到画布", "wg-btn", () => applyCurrentGroupToCanvas(node), state.loading || !currentGroup);

    toolbar.appendChild(groupSelect);
    toolbar.appendChild(manageBtn);
    toolbar.appendChild(applyBtn);

    const configGrid = document.createElement("div");
    configGrid.className = "wg-compact-grid";

    const autoApply = document.createElement("label");
    autoApply.className = "wg-check";
    const autoApplyInput = document.createElement("input");
    autoApplyInput.type = "checkbox";
    autoApplyInput.checked = !!state.autoApply;
    autoApplyInput.onchange = () => {
        state.autoApply = autoApplyInput.checked;
        syncBackingWidgets(node);
        renderNodePanel(node);
    };
    autoApply.appendChild(autoApplyInput);
    autoApply.appendChild(document.createTextNode("auto_apply"));

    const syncPreview = document.createElement("label");
    syncPreview.className = "wg-check";
    const syncPreviewInput = document.createElement("input");
    syncPreviewInput.type = "checkbox";
    syncPreviewInput.checked = !!state.syncUiPreview;
    syncPreviewInput.onchange = () => {
        state.syncUiPreview = syncPreviewInput.checked;
        syncBackingWidgets(node);
        renderNodePanel(node);
    };
    syncPreview.appendChild(syncPreviewInput);
    syncPreview.appendChild(document.createTextNode("sync_ui_preview"));

    configGrid.appendChild(autoApply);
    configGrid.appendChild(syncPreview);

    const status = document.createElement("div");
    status.className = "wg-compact-status";
    const badgeText = currentGroup
        ? `${currentGroup.name} · ${currentGroup.items.length} 项 · v${currentGroup.version}`
        : `共 ${state.db.groups.length} 个组`;
    status.innerHTML = `
        <div><span class="wg-compact-summary">${badgeText}</span></div>
        <div>节点条目: ${currentGroup ? currentGroup.items.length : 0} / 参数值: ${currentGroup ? countGroupValueEntries(currentGroup) : 0}</div>
        <div>scope: ${state.applyScope} · fallback: ${state.fallbackMode}</div>
        <div>完整编辑请点“打开管理界面”。</div>
    `;

    compactSection.appendChild(toolbar);
    compactSection.appendChild(configGrid);
    compactSection.appendChild(status);
    root.appendChild(compactSection);

    return root;
}

function buildConfigSection(node, state) {
    const section = document.createElement("div");
    section.className = "wg-section";

    const head = document.createElement("div");
    head.className = "wg-section-head";
    head.innerHTML = `<strong>运行配置</strong><span class="wg-note">这些值会同步回隐藏 widget，后端仍按原节点协议执行。</span>`;

    const grid = document.createElement("div");
    grid.className = "wg-config-grid";

    const groupSelect = document.createElement("select");
    groupSelect.className = "wg-select";
    const emptyOption = document.createElement("option");
    emptyOption.value = "";
    emptyOption.textContent = state.db.groups.length ? "请选择组" : "暂无组";
    groupSelect.appendChild(emptyOption);
    state.db.groups.forEach((group) => {
        const option = document.createElement("option");
        option.value = group.name;
        option.textContent = group.name;
        option.selected = group.name === state.selectedGroupName;
        groupSelect.appendChild(option);
    });
    groupSelect.onchange = () => setSelectedGroup(node, groupSelect.value);

    const applyScope = document.createElement("select");
    applyScope.className = "wg-select";
    ["values_only", "values_and_state"].forEach((value) => {
        const option = document.createElement("option");
        option.value = value;
        option.textContent = value;
        option.selected = value === state.applyScope;
        applyScope.appendChild(option);
    });
    applyScope.onchange = () => {
        state.applyScope = applyScope.value;
        syncBackingWidgets(node);
    };

    const fallbackMode = document.createElement("select");
    fallbackMode.className = "wg-select";
    ["warn_missing", "ignore_missing", "error_missing"].forEach((value) => {
        const option = document.createElement("option");
        option.value = value;
        option.textContent = value;
        option.selected = value === state.fallbackMode;
        fallbackMode.appendChild(option);
    });
    fallbackMode.onchange = () => {
        state.fallbackMode = fallbackMode.value;
        syncBackingWidgets(node);
    };

    const autoApply = document.createElement("label");
    autoApply.className = "wg-check";
    const autoApplyInput = document.createElement("input");
    autoApplyInput.type = "checkbox";
    autoApplyInput.checked = !!state.autoApply;
    autoApplyInput.onchange = () => {
        state.autoApply = autoApplyInput.checked;
        syncBackingWidgets(node);
    };
    autoApply.appendChild(autoApplyInput);
    autoApply.appendChild(document.createTextNode("运行时自动应用当前组"));

    const syncUiPreview = document.createElement("label");
    syncUiPreview.className = "wg-check";
    const syncUiInput = document.createElement("input");
    syncUiInput.type = "checkbox";
    syncUiInput.checked = !!state.syncUiPreview;
    syncUiInput.onchange = () => {
        state.syncUiPreview = syncUiInput.checked;
        syncBackingWidgets(node);
    };
    syncUiPreview.appendChild(syncUiInput);
    syncUiPreview.appendChild(document.createTextNode("允许前端同步应用到画布"));

    const refreshBtn = createButton("刷新后端分组", "wg-btn", () => refreshGroups(node), state.loading);

    grid.appendChild(createField("激活组", groupSelect, true));
    grid.appendChild(createField("apply_scope", applyScope));
    grid.appendChild(createField("fallback_mode", fallbackMode));
    grid.appendChild(createField("auto_apply", autoApply));
    grid.appendChild(createField("sync_ui_preview", syncUiPreview));
    grid.appendChild(createField("同步操作", refreshBtn));

    section.appendChild(head);
    section.appendChild(grid);
    return section;
}

function buildGroupSection(node, state, currentGroup) {
    const section = document.createElement("div");
    section.className = "wg-section";

    const head = document.createElement("div");
    head.className = "wg-section-head";
    head.innerHTML = `<strong>组管理</strong><span class="wg-note">输入框既可用于新建组，也可用于重命名/复制当前组。</span>`;

    const toolbar = document.createElement("div");
    toolbar.className = "wg-toolbar";

    const nameInput = document.createElement("input");
    nameInput.className = "wg-input";
    nameInput.placeholder = "输入组名";
    nameInput.value = state.groupDraftName || "";
    nameInput.oninput = () => {
        state.groupDraftName = nameInput.value;
    };
    const nameInputWrap = createClearableInput(nameInput);

    const createBtn = createButton("新建", "wg-btn primary", () => createGroup(node), state.loading);
    const renameBtn = createButton("重命名", "wg-btn", () => renameCurrentGroup(node), state.loading || !currentGroup);
    const duplicateBtn = createButton("复制", "wg-btn", () => duplicateCurrentGroup(node), state.loading || !currentGroup);
    const deleteBtn = createButton("删除", "wg-btn danger", () => deleteCurrentGroup(node), state.loading || !currentGroup);

    toolbar.appendChild(nameInputWrap);
    toolbar.appendChild(createBtn);
    toolbar.appendChild(renameBtn);
    toolbar.appendChild(duplicateBtn);
    toolbar.appendChild(deleteBtn);

    section.appendChild(head);
    section.appendChild(toolbar);
    return section;
}

function buildActionSection(node, state, currentGroup) {
    const section = document.createElement("div");
    section.className = "wg-section";

    const head = document.createElement("div");
    head.className = "wg-section-head";
    head.innerHTML = `<strong>采集与应用</strong><span class="wg-note">支持覆盖采集、追加采集、直接应用到画布。</span>`;

    const row = document.createElement("div");
    row.className = "wg-actions";
    row.style.flexWrap = "wrap";

    row.appendChild(createButton("覆盖采集选中节点", "wg-btn", () => captureSelectedNodesToCurrentGroup(node, true), state.loading || !currentGroup));
    row.appendChild(createButton("追加选中节点到组", "wg-btn", () => captureSelectedNodesToCurrentGroup(node, false), state.loading || !currentGroup));
    row.appendChild(createButton("应用当前组到画布", "wg-btn primary", () => applyCurrentGroupToCanvas(node), state.loading || !currentGroup));

    const note = document.createElement("div");
    note.className = "wg-note";
    note.textContent = "覆盖采集会替换当前组全部条目；追加采集会按 节点ID + 输入名 合并更新。";

    section.appendChild(head);
    section.appendChild(row);
    section.appendChild(note);
    return section;
}

function buildItemListSection(node, state, currentGroup) {
    const section = document.createElement("div");
    section.className = "wg-section";

    const head = document.createElement("div");
    head.className = "wg-section-head";
    const headTitle = document.createElement("strong");
    headTitle.textContent = "组内节点条目";
    const headNote = document.createElement("span");
    headNote.className = "wg-note";
    headNote.textContent = currentGroup ? `当前 ${currentGroup.items.length} 个节点 / ${countGroupValueEntries(currentGroup)} 个参数值` : "请先创建组";
    const headTools = document.createElement("div");
    headTools.className = "wg-item-tools";
    headTools.appendChild(headNote);
    headTools.appendChild(createButton("新建节点条目", "wg-btn", () => createBlankItem(node), state.loading || !currentGroup));
    head.appendChild(headTitle);
    head.appendChild(headTools);
    section.appendChild(head);

    if (!currentGroup) {
        const empty = document.createElement("div");
        empty.className = "wg-empty";
        empty.textContent = "还没有组，请先创建一个组。";
        section.appendChild(empty);
        return section;
    }

    if (!currentGroup.items.length) {
        const empty = document.createElement("div");
        empty.className = "wg-empty";
        empty.textContent = "当前组还没有条目，可用“新增空白条目”或“采集选中节点”生成。";
        section.appendChild(empty);
        return section;
    }

    const searchInput = document.createElement("input");
    searchInput.className = "wg-input";
    searchInput.placeholder = "搜索节点标题 / 节点ID / 输入名 / 值";
    searchInput.value = state.itemSearchKeyword || "";
    searchInput.oninput = () => {
        state.itemSearchKeyword = searchInput.value;
        renderNodePanel(node);
    };
    section.appendChild(createClearableInput(searchInput, () => {
        state.itemSearchKeyword = "";
        renderNodePanel(node);
    }));

    const filteredItems = currentGroup.items.filter((item) => {
        const haystack = [
            item.target_node_title,
            item.target_class_type,
            item.target_node_id,
            (item.values || []).map((entry) => entry.target_input_name).join(" "),
            (item.values || []).map((entry) => entry.value_type).join(" "),
            formatItemValue(item),
            item.note,
        ].join(" | ");
        return matchesKeyword(haystack, state.itemSearchKeyword);
    });

    if (!filteredItems.length) {
        const empty = document.createElement("div");
        empty.className = "wg-empty";
        empty.textContent = "当前搜索条件下没有匹配条目。";
        section.appendChild(empty);
        return section;
    }

    const list = document.createElement("div");
    list.className = "wg-list";
    filteredItems.forEach((item) => {
        const row = document.createElement("div");
        const itemInvalidReason = getItemInvalidReason(node, item);
        row.className = `wg-item-row ${item.id === state.selectedItemId ? "active" : ""} ${itemInvalidReason ? "invalid" : ""}`.trim();
        row.onclick = () => setSelectedItem(node, item.id);

        const checkboxWrap = document.createElement("label");
        checkboxWrap.className = "wg-check";
        const checkbox = document.createElement("input");
        checkbox.type = "checkbox";
        checkbox.checked = item.enabled !== false;
        checkbox.onclick = (event) => event.stopPropagation();
        checkbox.onchange = () => toggleItemEnabled(node, item.id, checkbox.checked);
        checkboxWrap.appendChild(checkbox);

        const main = document.createElement("div");
        main.className = "wg-item-main";

        const title = document.createElement("div");
        title.className = "wg-item-title";
        title.textContent = `${item.target_node_title || item.target_class_type || item.target_node_id || "未绑定节点"}`;

        const sub = document.createElement("div");
        sub.className = "wg-item-sub";
        sub.textContent = `node_id=${item.target_node_id || "-"} · 参数值=${countEnabledValueEntries(item)} · ${item.enabled !== false ? "启用" : "禁用"}${itemInvalidReason ? ` · 失效: ${itemInvalidReason}` : ""}`;

        const value = document.createElement("div");
        value.className = "wg-item-value";
        value.textContent = formatItemValue(item);

        main.appendChild(title);
        main.appendChild(sub);
        main.appendChild(value);

        const deleteBtn = createButton("删", "wg-btn danger", (event) => {
            event.stopPropagation();
            removeCurrentItem(node, item.id);
        }, state.loading);
        deleteBtn.style.minWidth = "32px";
        deleteBtn.style.padding = "4px 8px";

        row.appendChild(checkboxWrap);
        row.appendChild(main);
        row.appendChild(deleteBtn);
        list.appendChild(row);
    });

    section.appendChild(list);
    return section;
}

function buildEditorSection(node, state, currentGroup) {
    const section = document.createElement("div");
    section.className = "wg-section";

    const head = document.createElement("div");
    head.className = "wg-section-head";
    const headTitle = document.createElement("strong");
    headTitle.textContent = "节点条目编辑器";
    const headNote = document.createElement("span");
    headNote.className = "wg-note";
    headNote.textContent = "一个节点条目可以包含多个参数值，适合整节点快照管理。";
    head.appendChild(headTitle);
    head.appendChild(headNote);
    section.appendChild(head);

    if (!currentGroup) {
        const empty = document.createElement("div");
        empty.className = "wg-empty";
        empty.textContent = "请先创建并选择一个组。";
        section.appendChild(empty);
        return section;
    }

    const editor = state.editor || defaultItemFromSelection(node);
    const allSelectableNodes = getSelectableNodes(node);
    const selectedTargetNode = getTargetNodeById(node, editor.target_node_id);
    const itemInvalidReason = getItemInvalidReason(node, editor);
    const grid = document.createElement("div");
    grid.className = "wg-grid";

    const nodePicker = document.createElement("input");
    nodePicker.className = "wg-input";
    const nodeListId = `wg-node-list-${node.id}`;
    nodePicker.setAttribute("list", nodeListId);
    nodePicker.placeholder = "选择或输入目标节点，例如 12 - KSampler";
    nodePicker.value = editor.target_node_id
        ? `${editor.target_node_id} - ${editor.target_node_title || editor.target_class_type || ""}`.trim()
        : "";
    const nodeDatalist = document.createElement("datalist");
    nodeDatalist.id = nodeListId;
    allSelectableNodes.forEach((targetNode) => {
        const option = document.createElement("option");
        option.value = `${targetNode.id} - ${targetNode.title || targetNode.type}`;
        nodeDatalist.appendChild(option);
    });
    nodePicker.onchange = () => {
        const nextNodeId = extractNodeIdFromText(nodePicker.value);
        const targetNode = getTargetNodeById(node, nextNodeId);
        state.editor.target_node_id = nextNodeId;
        state.editor.target_node_title = String(targetNode?.title || targetNode?.type || "");
        state.editor.target_class_type = String(targetNode?.type || "");
        renderNodePanel(node);
    };
    const nodePickerClearable = createClearableInput(nodePicker, () => {
        state.editor.target_node_id = "";
        state.editor.target_node_title = "";
        state.editor.target_class_type = "";
        renderNodePanel(node);
    });

    const addInputPicker = document.createElement("input");
    addInputPicker.className = "wg-input";
    const inputListId = `wg-input-list-${node.id}`;
    addInputPicker.setAttribute("list", inputListId);
    addInputPicker.placeholder = "选择或输入要添加的输入项";
    const inputDatalist = document.createElement("datalist");
    inputDatalist.id = inputListId;
    getEditableInputNamesFromNode(selectedTargetNode).forEach((inputName) => {
        const option = document.createElement("option");
        option.value = inputName;
        inputDatalist.appendChild(option);
    });
    const addInputWrap = document.createElement("div");
    addInputWrap.className = "wg-stack";
    addInputWrap.appendChild(createClearableInput(addInputPicker));
    addInputWrap.appendChild(inputDatalist);
    addInputWrap.appendChild(
        createButton("添加该输入项", "wg-btn", () => {
            const inputName = String(addInputPicker.value || "").trim();
            if (!inputName) {
                setStatus(node, "warning", "请先选择要添加的输入项。");
                return;
            }
            const existingValues = Array.isArray(state.editor.values) ? state.editor.values : [];
            if (existingValues.some((entry) => entry.target_input_name === inputName)) {
                setStatus(node, "warning", `该输入项已存在: ${inputName}`);
                return;
            }
            const targetWidget = selectedTargetNode?.widgets?.find((widget) => widget.name === inputName);
            const nextEntry = targetWidget ? createValueEntryFromWidget(targetWidget) : createEmptyValueEntry(inputName);
            state.editor.values = [...existingValues, nextEntry];
            renderNodePanel(node);
        }, !editor.target_node_id)
    );

    const nodeIdInput = document.createElement("input");
    nodeIdInput.className = "wg-input";
    nodeIdInput.value = editor.target_node_id || "";
    nodeIdInput.oninput = () => {
        state.editor.target_node_id = nodeIdInput.value;
    };
    const nodeIdInputWrap = createClearableInput(nodeIdInput);

    const nodeTitleInput = document.createElement("input");
    nodeTitleInput.className = "wg-input";
    nodeTitleInput.value = editor.target_node_title || "";
    nodeTitleInput.oninput = () => {
        state.editor.target_node_title = nodeTitleInput.value;
    };
    const nodeTitleInputWrap = createClearableInput(nodeTitleInput);

    const classTypeInput = document.createElement("input");
    classTypeInput.className = "wg-input";
    classTypeInput.value = editor.target_class_type || "";
    classTypeInput.oninput = () => {
        state.editor.target_class_type = classTypeInput.value;
    };
    const classTypeInputWrap = createClearableInput(classTypeInput);

    const enabledCheck = document.createElement("label");
    enabledCheck.className = "wg-check";
    const enabledInput = document.createElement("input");
    enabledInput.type = "checkbox";
    enabledInput.checked = editor.enabled !== false;
    enabledInput.onchange = () => {
        state.editor.enabled = enabledInput.checked;
    };
    enabledCheck.appendChild(enabledInput);
    enabledCheck.appendChild(document.createTextNode("启用该条目"));

    const noteInput = document.createElement("input");
    noteInput.className = "wg-input";
    noteInput.value = editor.note || "";
    noteInput.oninput = () => {
        state.editor.note = noteInput.value;
    };
    const noteInputWrap = createClearableInput(noteInput);

    const valueTextarea = document.createElement("textarea");
    valueTextarea.className = "wg-textarea";
    valueTextarea.value = formatItemValue(editor);
    valueTextarea.readOnly = true;

    const nodePickerWrap = document.createElement("div");
    nodePickerWrap.className = "wg-stack";
    nodePickerWrap.appendChild(nodePickerClearable);
    nodePickerWrap.appendChild(nodeDatalist);
    if (itemInvalidReason) {
        const invalidText = document.createElement("div");
        invalidText.className = "wg-inline-note";
        invalidText.textContent = `当前节点条目失效: ${itemInvalidReason}`;
        nodePickerWrap.appendChild(invalidText);
    }

    grid.appendChild(createField("可编辑搜索下拉: 目标节点", nodePickerWrap, true));
    grid.appendChild(createField("可编辑搜索下拉: 添加输入项", addInputWrap, true));
    grid.appendChild(createField("目标节点 ID", nodeIdInputWrap));
    grid.appendChild(createField("目标节点标题", nodeTitleInputWrap));
    grid.appendChild(createField("目标节点类型", classTypeInputWrap));
    grid.appendChild(createField("启用状态", enabledCheck));
    grid.appendChild(createField("备注", noteInputWrap, true));
    grid.appendChild(createField("当前值摘要", valueTextarea, true));

    const valuesSection = document.createElement("div");
    valuesSection.className = "wg-stack";
    const valuesHeader = document.createElement("div");
    valuesHeader.className = "wg-section-head";
    const valuesTitle = document.createElement("strong");
    valuesTitle.textContent = `参数值列表 (${Array.isArray(editor.values) ? editor.values.length : 0})`;
    const valuesTools = document.createElement("div");
    valuesTools.className = "wg-value-tools";
    valuesTools.appendChild(createButton("从当前节点重同步全部参数值", "wg-btn", () => resyncEditorValuesFromCurrentNode(node), state.loading || !editor.target_node_id));
    valuesHeader.appendChild(valuesTitle);
    valuesHeader.appendChild(valuesTools);
    valuesSection.appendChild(valuesHeader);

    const valuesList = document.createElement("div");
    valuesList.className = "wg-value-list";
    (Array.isArray(editor.values) ? editor.values : []).forEach((valueEntry) => {
        const valueInvalidReason = getValueEntryInvalidReason(node, editor, valueEntry);
        const sourceWidget = selectedTargetNode?.widgets?.find((widget) => widget.name === valueEntry.target_input_name);
        const comboOptions = getComboOptionValues(sourceWidget);
        const numericOptions = getNumericWidgetOptions(sourceWidget, valueEntry.value_type);
        const useComboEditor = valueEntry.value_type === "COMBO" && comboOptions.length > 0;
        const useBooleanEditor = valueEntry.value_type === "BOOLEAN";
        const useNumberEditor = valueEntry.value_type === "INT" || valueEntry.value_type === "FLOAT";
        const row = document.createElement("div");
        row.className = `wg-value-row ${valueInvalidReason ? "invalid" : ""}`.trim();

        const inputNameInput = document.createElement("input");
        inputNameInput.className = "wg-input";
        inputNameInput.value = valueEntry.target_input_name || "";
        inputNameInput.oninput = () => {
            valueEntry.target_input_name = inputNameInput.value;
        };
        const inputNameInputWrap = createClearableInput(inputNameInput);

        const valueTypeSelect = document.createElement("select");
        valueTypeSelect.className = "wg-select";
        VALUE_TYPE_OPTIONS.forEach((valueType) => {
            const option = document.createElement("option");
            option.value = valueType;
            option.textContent = valueType;
            option.selected = valueType === valueEntry.value_type;
            valueTypeSelect.appendChild(option);
        });
        valueTypeSelect.onchange = () => {
            valueEntry.value_type = valueTypeSelect.value;
        };

        let valueEditor = null;
        if (useComboEditor) {
            const comboInput = document.createElement("input");
            comboInput.className = "wg-input";
            const comboListId = `wg-value-combo-${node.id}-${valueEntry.id}`;
            comboInput.setAttribute("list", comboListId);
            comboInput.placeholder = "搜索或直接输入值";
            comboInput.value = valueEntry.value == null ? "" : String(valueEntry.value);
            const comboDatalist = document.createElement("datalist");
            comboDatalist.id = comboListId;
            comboOptions.forEach((optionValue) => {
                const option = document.createElement("option");
                option.value = String(optionValue);
                comboDatalist.appendChild(option);
            });
            comboInput.oninput = () => {
                valueEntry.value = comboInput.value;
            };
            const comboWrap = document.createElement("div");
            comboWrap.className = "wg-stack";
            comboWrap.appendChild(createClearableInput(comboInput));
            comboWrap.appendChild(comboDatalist);
            valueEditor = comboWrap;
        } else if (useBooleanEditor) {
            const valueToggle = document.createElement("label");
            valueToggle.className = "wg-check";
            const valueToggleInput = document.createElement("input");
            valueToggleInput.type = "checkbox";
            valueToggleInput.checked = valueEntry.value === true || String(valueEntry.value).toLowerCase() === "true";
            valueToggleInput.onchange = () => {
                valueEntry.value = valueToggleInput.checked;
            };
            valueToggle.appendChild(valueToggleInput);
            valueToggle.appendChild(document.createTextNode(valueToggleInput.checked ? "true" : "false"));
            valueToggleInput.onchange = () => {
                valueEntry.value = valueToggleInput.checked;
                valueToggle.lastChild.textContent = valueToggleInput.checked ? "true" : "false";
            };
            valueEditor = valueToggle;
        } else if (useNumberEditor) {
            const valueInput = document.createElement("input");
            valueInput.type = "number";
            valueInput.className = "wg-input";
            valueInput.value = valueEntry.value == null ? "" : String(valueEntry.value);
            if (numericOptions.min) {
                valueInput.min = numericOptions.min;
            }
            if (numericOptions.max) {
                valueInput.max = numericOptions.max;
            }
            if (numericOptions.step) {
                valueInput.step = numericOptions.step;
            }
            valueInput.oninput = () => {
                if (valueInput.value === "") {
                    valueEntry.value = "";
                    return;
                }
                const parsedValue = valueEntry.value_type === "INT"
                    ? parseInt(valueInput.value, 10)
                    : parseFloat(valueInput.value);
                valueEntry.value = Number.isFinite(parsedValue) ? parsedValue : valueInput.value;
            };
            valueEditor = createClearableInput(valueInput);
        } else {
            const valueInput = document.createElement("input");
            valueInput.className = "wg-input";
            valueInput.value = valueEntry.value == null ? "" : String(valueEntry.value);
            valueInput.oninput = () => {
                valueEntry.value = valueInput.value;
            };
            valueEditor = createClearableInput(valueInput);
        }

        const valueEnabled = document.createElement("label");
        valueEnabled.className = "wg-check";
        const valueEnabledInput = document.createElement("input");
        valueEnabledInput.type = "checkbox";
        valueEnabledInput.checked = valueEntry.enabled !== false;
        valueEnabledInput.onchange = () => {
            valueEntry.enabled = valueEnabledInput.checked;
        };
        valueEnabled.appendChild(valueEnabledInput);
        valueEnabled.appendChild(document.createTextNode("启用"));

        const deleteValueBtn = createMiniButton("删除", () => {
            state.editor.values = (state.editor.values || []).filter((entry) => entry.id !== valueEntry.id);
            renderNodePanel(node);
        }, state.loading, "wg-mini-btn danger");

        const sortTools = document.createElement("div");
        sortTools.className = "wg-value-tools";
        sortTools.appendChild(createMiniButton("↑", () => moveEditorValueEntry(node, valueEntry.id, "up"), state.loading));
        sortTools.appendChild(createMiniButton("↓", () => moveEditorValueEntry(node, valueEntry.id, "down"), state.loading));
        sortTools.appendChild(deleteValueBtn);

        const metaStack = document.createElement("div");
        metaStack.className = "wg-stack";
        metaStack.appendChild(valueEditor);
        if (useBooleanEditor) {
            const booleanHint = document.createElement("div");
            booleanHint.className = "wg-inline-note";
            booleanHint.textContent = "BOOLEAN 使用开关编辑";
            metaStack.appendChild(booleanHint);
        }
        if (valueInvalidReason) {
            const invalidText = document.createElement("div");
            invalidText.className = "wg-inline-note";
            invalidText.textContent = valueInvalidReason;
            metaStack.appendChild(invalidText);
        }

        row.appendChild(inputNameInputWrap);
        row.appendChild(valueTypeSelect);
        row.appendChild(metaStack);
        row.appendChild(valueEnabled);
        row.appendChild(sortTools);
        valuesList.appendChild(row);
    });
    valuesSection.appendChild(valuesList);

    const actions = document.createElement("div");
    actions.className = "wg-actions";
    actions.style.flexWrap = "wrap";
    actions.appendChild(createButton("保存当前条目", "wg-btn primary", () => saveCurrentItem(node), state.loading));
    actions.appendChild(createButton("删除当前条目", "wg-btn danger", () => removeCurrentItem(node), state.loading || !state.selectedItemId));
    actions.appendChild(createButton("重置编辑器", "wg-btn ghost", () => {
        updateEditorDraftFromSelection(state);
        renderNodePanel(node);
    }, state.loading));

    section.appendChild(grid);
    section.appendChild(valuesSection);
    section.appendChild(actions);
    return section;
}

function buildManagerModalContent(node, state, currentGroup) {
    const root = document.createElement("div");
    root.className = "wg-manager-root";

    root.appendChild(makeCollapsibleSection(node, state, "config", buildConfigSection(node, state)));

    const main = document.createElement("div");
    main.className = "wg-manager-main";

    const leftCol = document.createElement("div");
    leftCol.className = "wg-manager-col";
    leftCol.appendChild(makeCollapsibleSection(node, state, "groups", buildGroupSection(node, state, currentGroup)));
    leftCol.appendChild(makeCollapsibleSection(node, state, "actions", buildActionSection(node, state, currentGroup)));
    leftCol.appendChild(makeCollapsibleSection(node, state, "items", buildItemListSection(node, state, currentGroup)));

    const rightCol = document.createElement("div");
    rightCol.className = "wg-manager-col";
    rightCol.appendChild(makeCollapsibleSection(node, state, "editor", buildEditorSection(node, state, currentGroup)));

    main.appendChild(leftCol);
    main.appendChild(rightCol);
    root.appendChild(main);

    return root;
}

function renderManagerModal(node) {
    const state = getState(node);
    if (!state?.managerModalVisible || !state.managerModalContainer) {
        return;
    }
    const currentGroup = getCurrentGroup(state);
    state.managerModalContainer.innerHTML = "";
    state.managerModalContainer.appendChild(buildManagerModalContent(node, state, currentGroup));
}

function openManagerModal(node) {
    const state = getState(node);
    if (!state) {
        return;
    }

    ensureCollapseState(state);
    if (state.managerModalVisible && state.managerPanelElement) {
        state.managerPanelElement.style.display = "flex";
        state.managerPanelElement.style.zIndex = "9999";
        return;
    }

    const layout = state.managerPanelLayout || loadManagerPanelLayout() || DEFAULT_MANAGER_PANEL_LAYOUT;
    const panelElement = document.createElement("div");
    panelElement.className = "wg-floating-panel";
    panelElement.style.left = `${layout.left}px`;
    panelElement.style.top = `${layout.top}px`;
    panelElement.style.width = `${layout.width}px`;
    panelElement.style.height = `${layout.height}px`;

    const panelHead = document.createElement("div");
    panelHead.className = "wg-floating-panel-head";
    const titleWrap = document.createElement("div");
    titleWrap.className = "wg-floating-panel-title";
    titleWrap.innerHTML = "<strong>Workflow Group 管理界面</strong><span>非阻塞浮窗，可在打开时继续切换并选择画布节点。</span>";
    const panelTools = document.createElement("div");
    panelTools.className = "wg-floating-panel-tools";
    const closeBtn = document.createElement("button");
    closeBtn.type = "button";
    closeBtn.className = "wg-panel-tool-btn";
    closeBtn.textContent = "关闭";
    closeBtn.onclick = () => closeManagerPanel(node);
    panelTools.appendChild(closeBtn);
    panelHead.appendChild(titleWrap);
    panelHead.appendChild(panelTools);

    const panelBody = document.createElement("div");
    panelBody.className = "wg-floating-panel-body";

    panelElement.appendChild(panelHead);
    panelElement.appendChild(panelBody);
    document.body.appendChild(panelElement);

    state.managerModalVisible = true;
    state.managerModalContainer = panelBody;
    state.managerPanelElement = panelElement;
    state.managerPanelLayout = layout;
    state.managerPanelCleanup = attachManagerPanelInteractions(node, panelElement, panelHead);
    renderManagerModal(node);
}

function renderNodePanel(node) {
    const state = getState(node);
    if (!state?.container) {
        return;
    }

    syncBackingWidgets(node);
    const currentGroup = getCurrentGroup(state);
    state.container.innerHTML = "";

    const root = buildCompactPanel(node, state, currentGroup);

    if (state.message?.text) {
        const status = document.createElement("div");
        status.className = `wg-status ${state.message.type || ""}`.trim();
        status.textContent = state.message.text;
        root.appendChild(status);
    }

    state.container.appendChild(root);
    renderManagerModal(node);
    triggerNodeRefresh(node);
}

function ensureWorkflowGroupUI(node) {
    if (!node || !node.widgets) {
        return;
    }

    const groupNameWidget = getWidget(node, "group_name");
    const autoApplyWidget = getWidget(node, "auto_apply");
    const applyScopeWidget = getWidget(node, "apply_scope");
    const fallbackModeWidget = getWidget(node, "fallback_mode");
    const syncUiPreviewWidget = getWidget(node, "sync_ui_preview");

    if (!groupNameWidget || !autoApplyWidget || !applyScopeWidget || !fallbackModeWidget || !syncUiPreviewWidget) {
        console.error("[WorkflowGroupPresetManager] 缺少必要控件。");
        return;
    }

    removeLegacyWidgets(node);
    [groupNameWidget, autoApplyWidget, applyScopeWidget, fallbackModeWidget, syncUiPreviewWidget].forEach(hideWidget);

    if (!node._workflowGroupState) {
        const container = document.createElement("div");
        container.style.width = "100%";
        container.style.height = `calc(100% + ${PANEL_OFFSET_Y}px)`;
        container.style.boxSizing = "border-box";
        container.style.marginTop = `-${PANEL_OFFSET_Y}px`;

        node._workflowGroupState = {
            node,
            groupNameWidget,
            autoApplyWidget,
            applyScopeWidget,
            fallbackModeWidget,
            syncUiPreviewWidget,
            selectedGroupName: String(groupNameWidget.value || ""),
            groupDraftName: String(groupNameWidget.value || ""),
            selectedItemId: "",
            autoApply: !!autoApplyWidget.value,
            applyScope: String(applyScopeWidget.value || "values_only"),
            fallbackMode: String(fallbackModeWidget.value || "warn_missing"),
            syncUiPreview: !!syncUiPreviewWidget.value,
            db: normalizeDb(DEFAULT_DB),
            loading: false,
            message: null,
            editor: defaultItemFromSelection(node),
            itemSearchKeyword: "",
            nodeSearchKeyword: "",
            inputSearchKeyword: "",
            sectionCollapsed: {
                config: false,
                groups: false,
                actions: false,
                items: false,
                editor: false,
            },
            managerModalVisible: false,
            managerModalContainer: null,
            managerPanelElement: null,
            managerPanelCleanup: null,
            managerPanelLayout: loadManagerPanelLayout() || DEFAULT_MANAGER_PANEL_LAYOUT,
            container,
        };

        if (typeof node.addDOMWidget === "function") {
            node.addDOMWidget("WorkflowGroupPresetModernUI", "div", container, {
                serialize: false,
                hideOnZoom: false,
            });
        }
    }

    const state = node._workflowGroupState;
    state.groupNameWidget = groupNameWidget;
    state.autoApplyWidget = autoApplyWidget;
    state.applyScopeWidget = applyScopeWidget;
    state.fallbackModeWidget = fallbackModeWidget;
    state.syncUiPreviewWidget = syncUiPreviewWidget;
    state.autoApply = !!autoApplyWidget.value;
    state.applyScope = String(applyScopeWidget.value || state.applyScope || "values_only");
    state.fallbackMode = String(fallbackModeWidget.value || state.fallbackMode || "warn_missing");
    state.syncUiPreview = !!syncUiPreviewWidget.value;
    if (!state.selectedGroupName) {
        state.selectedGroupName = String(groupNameWidget.value || "");
        state.groupDraftName = state.selectedGroupName;
    }

    if (!node.size || node.size[0] < 360 || node.size[1] < 245) {
        node.size = [360, 245];
    }

    ensureSelection(state);
    updateEditorDraftFromSelection(state);
    renderNodePanel(node);

    if (!node._workflowGroupInitialFetchStarted) {
        node._workflowGroupInitialFetchStarted = true;
        refreshGroups(node, { silent: true });
    }
}

injectStyles();

app.registerExtension({
    name: "A_my_nodes.WorkflowGroupPresetManager.DOMUI",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) {
            return;
        }

        const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function() {
            const result = originalOnNodeCreated?.apply(this, arguments);
            ensureWorkflowGroupUI(this);
            return result;
        };

        const originalOnConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function() {
            const result = originalOnConfigure?.apply(this, arguments);
            setTimeout(() => ensureWorkflowGroupUI(this), 50);
            return result;
        };

        const originalGetExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
        nodeType.prototype.getExtraMenuOptions = function(_, options) {
            originalGetExtraMenuOptions?.apply(this, arguments);
            options.push(
                { content: "刷新 Workflow Groups", callback: () => refreshGroups(this) },
                { content: "覆盖采集选中节点", callback: () => captureSelectedNodesToCurrentGroup(this, true) },
                { content: "追加选中节点到组", callback: () => captureSelectedNodesToCurrentGroup(this, false) },
                { content: "应用当前组到画布", callback: () => applyCurrentGroupToCanvas(this) }
            );
        };
    },
});
