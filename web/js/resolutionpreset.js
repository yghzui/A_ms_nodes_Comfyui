import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";
import { modal } from "./utils/modal.js";

const ENDPOINT = "/a_my_nodes/resolution_presets";
const STYLE_ID = "a-my-nodes-resolutionpreset-modern-style";
const LEGACY_WIDGET_NAMES = new Set(["使用当前预设宽高", "保存/更新预设", "删除当前预设", "宽", "高", "步长"]);
const PANEL_OFFSET_Y = 14;
const DEFAULT_CONSTRAINTS = {
    min: 64,
    max: 8192,
    default_step: 8,
    max_name_length: 80,
};
const FALLBACK_BUILTINS = {
    "512x768": { w: 512, h: 768 },
    "1024x1440": { w: 1024, h: 1440 },
    "1280x1980": { w: 1280, h: 1980 },
};

function injectStyles() {
    if (document.getElementById(STYLE_ID)) {
        return;
    }
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
        .rp-shell {
            box-sizing: border-box;
            width: 100%;
            height: 100%;
            padding: 6px 10px 8px;
            display: flex;
            flex-direction: column;
            gap: 6px;
            color: #edf2ff;
            font-family: "Segoe UI", "Microsoft YaHei", sans-serif;
            background:
                radial-gradient(circle at top right, rgba(93, 132, 245, 0.18), transparent 38%),
                linear-gradient(180deg, rgba(27, 33, 49, 0.98), rgba(17, 20, 30, 0.98));
            border: 1px solid rgba(128, 145, 180, 0.22);
            border-radius: 14px;
            overflow: hidden;
        }
        .rp-header, .rp-toolbar, .rp-actions, .rp-editor-grid, .rp-chip-list {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
        }
        .rp-header {
            align-items: center;
            justify-content: space-between;
        }
        .rp-title {
            display: flex;
            flex-direction: column;
            gap: 3px;
        }
        .rp-title strong {
            font-size: 14px;
            color: #ffffff;
        }
        .rp-title span, .rp-note, .rp-meta, .rp-status {
            font-size: 11px;
            color: rgba(224, 232, 255, 0.72);
        }
        .rp-badge {
            padding: 4px 10px;
            border-radius: 999px;
            border: 1px solid rgba(132, 149, 186, 0.26);
            background: rgba(255, 255, 255, 0.05);
            font-size: 11px;
            color: #eef3ff;
        }
        .rp-toolbar {
            display: grid;
            grid-template-columns: minmax(0, 1fr) auto auto auto;
            align-items: center;
        }
        .rp-select, .rp-input {
            width: 100%;
            box-sizing: border-box;
            border-radius: 10px;
            border: 1px solid rgba(134, 149, 181, 0.28);
            background: rgba(7, 11, 20, 0.88);
            color: #eef3ff;
            min-height: 34px;
            padding: 6px 10px;
            font-size: 11px;
        }
        .rp-select:focus, .rp-input:focus {
            outline: none;
            border-color: rgba(124, 168, 255, 0.9);
            box-shadow: 0 0 0 2px rgba(124, 168, 255, 0.15);
        }
        .rp-btn {
            appearance: none;
            border: 1px solid rgba(134, 149, 181, 0.28);
            background: rgba(255, 255, 255, 0.05);
            color: #eef3ff;
            border-radius: 10px;
            min-height: 34px;
            padding: 6px 10px;
            font-size: 11px;
            cursor: pointer;
            transition: all 0.15s ease;
        }
        .rp-btn.compact {
            min-width: 34px;
            padding: 6px 0;
            min-width: 0;
            font-size: 11px;
        }
        .rp-btn.icon-only {
            min-height: 34px;
            min-width: 24px;
            padding: 0 2px;
            border: none;
            background: transparent;
            box-shadow: none;
            font-size: 18px;
            line-height: 1;
            color: rgba(238, 243, 255, 0.88);
        }
        .rp-btn.icon-only:hover {
            border: none;
            background: transparent;
            color: #ffffff;
        }
        .rp-btn:hover {
            border-color: rgba(124, 168, 255, 0.82);
            background: rgba(82, 124, 229, 0.16);
        }
        .rp-btn.primary {
            background: linear-gradient(180deg, rgba(89, 136, 255, 0.34), rgba(53, 94, 179, 0.42));
            border-color: rgba(124, 168, 255, 0.62);
        }
        .rp-btn.danger {
            background: rgba(195, 70, 70, 0.14);
            border-color: rgba(234, 124, 124, 0.28);
        }
        .rp-btn.toggle-on {
            border-color: rgba(124, 168, 255, 0.8);
            background: rgba(93, 132, 245, 0.2);
        }
        .rp-btn[disabled] {
            opacity: 0.6;
            cursor: not-allowed;
        }
        .rp-preview {
            padding: 8px 10px;
            border-radius: 12px;
            border: 1px solid rgba(140, 154, 184, 0.18);
            background: rgba(255, 255, 255, 0.05);
            display: flex;
            flex-direction: column;
            gap: 2px;
        }
        .rp-size {
            font-size: 20px;
            font-weight: 700;
            color: #ffffff;
            line-height: 1;
        }
        .rp-preview .rp-meta {
            font-size: 10px;
        }
        .rp-section {
            display: flex;
            flex-direction: column;
            gap: 8px;
        }
        .rp-section-head {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 8px;
        }
        .rp-section-head strong {
            font-size: 12px;
            color: #ffffff;
        }
        .rp-editor-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
        .rp-field {
            display: flex;
            flex-direction: column;
            gap: 5px;
        }
        .rp-field.full {
            grid-column: 1 / -1;
        }
        .rp-field label {
            font-size: 11px;
            color: rgba(224, 232, 255, 0.8);
        }
        .rp-check {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 12px;
            color: rgba(224, 232, 255, 0.86);
        }
        .rp-chip-list {
            max-height: 96px;
            overflow-y: auto;
            padding-right: 4px;
        }
        .rp-chip {
            padding: 6px 10px;
            border-radius: 999px;
            border: 1px solid rgba(132, 149, 186, 0.28);
            background: rgba(255, 255, 255, 0.05);
            color: #eef3ff;
            cursor: pointer;
            font-size: 11px;
        }
        .rp-chip.active {
            border-color: rgba(124, 168, 255, 0.9);
            background: rgba(93, 132, 245, 0.24);
        }
        .rp-message {
            font-size: 12px;
        }
        .rp-message.error {
            color: #ffbbbb;
        }
        .rp-message.success {
            color: #b8f5c2;
        }
        .rp-empty {
            padding: 10px 12px;
            border-radius: 10px;
            border: 1px dashed rgba(132, 149, 186, 0.28);
            color: rgba(224, 232, 255, 0.7);
            font-size: 12px;
            background: rgba(255, 255, 255, 0.04);
        }
    `;
    document.head.appendChild(style);
}

function isPlainObject(value) {
    return value && typeof value === "object" && !Array.isArray(value);
}

function normalizeConstraints(value) {
    const constraints = { ...DEFAULT_CONSTRAINTS };
    if (!isPlainObject(value)) {
        return constraints;
    }
    const min = Number.parseInt(value.min, 10);
    const max = Number.parseInt(value.max, 10);
    const step = Number.parseInt(value.default_step, 10);
    const maxNameLength = Number.parseInt(value.max_name_length, 10);
    if (Number.isFinite(min) && min > 0) {
        constraints.min = min;
    }
    if (Number.isFinite(max) && max >= constraints.min) {
        constraints.max = max;
    }
    if (Number.isFinite(step) && step > 0) {
        constraints.default_step = step;
    }
    if (Number.isFinite(maxNameLength) && maxNameLength > 0) {
        constraints.max_name_length = maxNameLength;
    }
    return constraints;
}

function normalizeStep(step, constraints) {
    let value = Number.parseInt(step, 10);
    if (!Number.isFinite(value) || value <= 0) {
        value = constraints.default_step;
    }
    return value;
}

function normalizeDimension(value, step, constraints) {
    let number = Number.parseInt(value, 10);
    if (!Number.isFinite(number) || number <= 0) {
        number = constraints.min;
    }
    const stepValue = normalizeStep(step, constraints);
    number = Math.max(constraints.min, Math.min(constraints.max, number));
    if (stepValue > 1) {
        number = Math.round(number / stepValue) * stepValue;
        number = Math.max(constraints.min, Math.min(constraints.max, number));
    }
    return number;
}

function normalizePresetName(name, constraints) {
    return String(name ?? "").trim().slice(0, constraints.max_name_length);
}

function normalizePresetMap(source, constraints, allowBuiltinNames = true) {
    const presets = {};
    if (!isPlainObject(source)) {
        return presets;
    }
    let defaultName = "";
    Object.entries(source).forEach(([name, item]) => {
        if (!isPlainObject(item)) {
            return;
        }
        const normalizedName = normalizePresetName(name, constraints);
        if (!normalizedName) {
            return;
        }
        if (!allowBuiltinNames && FALLBACK_BUILTINS[normalizedName]) {
            return;
        }
        presets[normalizedName] = {
            w: normalizeDimension(item.w, constraints.default_step, constraints),
            h: normalizeDimension(item.h, constraints.default_step, constraints),
            choose: !!item.choose,
        };
        if (!defaultName && item.choose) {
            defaultName = normalizedName;
        }
    });
    Object.keys(presets).forEach((name) => {
        presets[name].choose = name === defaultName;
    });
    return presets;
}

function parseCustomPresetsFromWidget(widget, constraints) {
    const raw = typeof widget?.value === "string" ? widget.value : "";
    if (!raw) {
        return {};
    }
    try {
        return normalizePresetMap(JSON.parse(raw), constraints, false);
    } catch (error) {
        console.warn("[ResolutionPresetNode] 解析 custom_presets 失败", error);
        return {};
    }
}

function serializeCustomPresets(customPresets) {
    return JSON.stringify(customPresets || {});
}

function setWidgetValue(widget, value) {
    if (!widget) {
        return;
    }
    widget.value = value;
    if (widget.element) {
        if (widget.element.type === "checkbox") {
            widget.element.checked = !!value;
        } else {
            widget.element.value = value;
        }
    }
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

function getDefaultCustomName(customPresets) {
    return Object.keys(customPresets).find((name) => customPresets[name]?.choose) || "";
}

function getAllPresets(state) {
    return {
        ...state.builtinPresets,
        ...state.customPresets,
    };
}

function suggestPresetName(state, width, height) {
    const base = `${width}x${height}`;
    const all = getAllPresets(state);
    if (!all[base]) {
        return base;
    }
    if (state.customPresets[base] && state.customPresets[base].w === width && state.customPresets[base].h === height) {
        return base;
    }
    let index = 1;
    let candidate = `${base}_${index}`;
    while (all[candidate]) {
        index += 1;
        candidate = `${base}_${index}`;
    }
    return candidate;
}

function syncBackingWidgets(node) {
    const state = node._resolutionPresetState;
    if (!state) {
        return;
    }
    const { presetWidget, mirrorWidget, customPresetsWidget } = state;
    const all = getAllPresets(state);
    const names = Object.keys(all);
    presetWidget.options = presetWidget.options || {};
    presetWidget.options.values = names;
    if (!all[state.selectedPreset]) {
        state.selectedPreset = getDefaultCustomName(state.customPresets) || names[0] || "";
    }
    setWidgetValue(presetWidget, state.selectedPreset);
    setWidgetValue(mirrorWidget, !!state.mirror);
    setWidgetValue(customPresetsWidget, serializeCustomPresets(state.customPresets));
}

function triggerNodeRefresh(node) {
    node?.setDirtyCanvas?.(true, false);
    node?.graph?.setDirtyCanvas?.(true, false);
}

async function fetchPresetPayload(options = {}) {
    const response = await api.fetchApi(ENDPOINT, options);
    let payload = {};
    try {
        payload = await response.json();
    } catch (error) {
        payload = {};
    }
    if (!response.ok) {
        throw new Error(payload?.error || `请求失败 (${response.status})`);
    }
    return payload;
}

function applyPayloadToState(node, payload) {
    const state = node._resolutionPresetState;
    state.constraints = normalizeConstraints(payload?.constraints);
    state.builtinPresets = normalizePresetMap(payload?.builtin_presets || FALLBACK_BUILTINS, state.constraints, true);
    state.customPresets = normalizePresetMap(payload?.custom_presets, state.constraints, false);
    state.defaultCustomPreset = payload?.default_custom_preset || getDefaultCustomName(state.customPresets);
    if (!state.selectedPreset || !getAllPresets(state)[state.selectedPreset]) {
        state.selectedPreset = state.defaultCustomPreset || Object.keys(getAllPresets(state))[0] || "";
    }
    syncBackingWidgets(node);
}

function getSelectedPresetInfo(state) {
    const preset = getAllPresets(state)[state.selectedPreset];
    if (!preset) {
        return {
            name: "",
            baseWidth: 0,
            baseHeight: 0,
            width: 0,
            height: 0,
            isBuiltin: false,
            isDefault: false,
        };
    }
    return {
        name: state.selectedPreset,
        baseWidth: preset.w,
        baseHeight: preset.h,
        width: state.mirror ? preset.h : preset.w,
        height: state.mirror ? preset.w : preset.h,
        isBuiltin: !!state.builtinPresets[state.selectedPreset],
        isDefault: !!state.customPresets[state.selectedPreset]?.choose,
    };
}

function resetEditorFromSelected(node) {
    const state = node._resolutionPresetState;
    const info = getSelectedPresetInfo(state);
    const sourceWidth = info.baseWidth || state.constraints.min;
    const sourceHeight = info.baseHeight || state.constraints.min;
    state.editor = {
        name: info.isBuiltin ? suggestPresetName(state, sourceWidth, sourceHeight) : info.name,
        width: sourceWidth,
        height: sourceHeight,
        step: state.constraints.default_step,
        choose: info.isDefault,
    };
}

function createEmptyEditor(node) {
    const state = node._resolutionPresetState;
    const info = getSelectedPresetInfo(state);
    const width = info.baseWidth || 720;
    const height = info.baseHeight || 1024;
    state.editor = {
        name: suggestPresetName(state, width, height),
        width,
        height,
        step: state.constraints.default_step,
        choose: false,
    };
}

function updateEditorField(node, key, value) {
    const state = node._resolutionPresetState;
    state.editor[key] = value;
}

function openManagerModal(node) {
    const state = node._resolutionPresetState;
    if (!state) {
        return;
    }

    const content = document.createElement("div");
    content.style.display = "flex";
    content.style.flexDirection = "column";
    content.style.gap = "12px";
    content.style.minWidth = "560px";
    content.style.maxWidth = "100%";

    const renderModalContent = () => {
        const allPresets = getAllPresets(state);
        const presetNames = Object.keys(allPresets);
        const info = getSelectedPresetInfo(state);
        const editorWidth = normalizeDimension(state.editor.width, state.editor.step, state.constraints);
        const editorHeight = normalizeDimension(state.editor.height, state.editor.step, state.constraints);
        const editorStep = normalizeStep(state.editor.step, state.constraints);

        content.innerHTML = "";

        const preview = document.createElement("div");
        preview.className = "rp-preview";
        preview.innerHTML = `
            <div>
                <div class="rp-size">${info.width || "--"} x ${info.height || "--"}</div>
                <div class="rp-meta">当前预设 ${info.name || "未选择"} · ${info.isBuiltin ? "内置" : "自定义"}${info.isDefault ? " · 默认" : ""}</div>
            </div>
            <div class="rp-status">
                <div>基础 ${info.baseWidth || "--"} x ${info.baseHeight || "--"}</div>
                <div>镜像 ${state.mirror ? "开启" : "关闭"}</div>
            </div>
        `;
        content.appendChild(preview);

        const editorSection = document.createElement("div");
        editorSection.className = "rp-section";
        const editorHead = document.createElement("div");
        editorHead.className = "rp-section-head";
        editorHead.innerHTML = `
            <strong>预设编辑</strong>
            <span class="rp-note">宽高会自动限制到 ${state.constraints.min}-${state.constraints.max}，并按步长对齐。</span>
        `;
        editorSection.appendChild(editorHead);

        const editorGrid = document.createElement("div");
        editorGrid.className = "rp-editor-grid";

        const nameField = document.createElement("div");
        nameField.className = "rp-field full";
        const nameInput = document.createElement("input");
        nameInput.className = "rp-input";
        nameInput.maxLength = state.constraints.max_name_length;
        nameInput.value = state.editor.name || "";
        nameInput.oninput = () => updateEditorField(node, "name", nameInput.value);
        nameField.innerHTML = "<label>预设名称</label>";
        nameField.appendChild(nameInput);

        const widthField = document.createElement("div");
        widthField.className = "rp-field";
        const widthInput = document.createElement("input");
        widthInput.className = "rp-input";
        widthInput.type = "number";
        widthInput.value = String(editorWidth);
        widthInput.oninput = () => updateEditorField(node, "width", widthInput.value);
        widthField.innerHTML = "<label>宽</label>";
        widthField.appendChild(widthInput);

        const heightField = document.createElement("div");
        heightField.className = "rp-field";
        const heightInput = document.createElement("input");
        heightInput.className = "rp-input";
        heightInput.type = "number";
        heightInput.value = String(editorHeight);
        heightInput.oninput = () => updateEditorField(node, "height", heightInput.value);
        heightField.innerHTML = "<label>高</label>";
        heightField.appendChild(heightInput);

        const stepField = document.createElement("div");
        stepField.className = "rp-field";
        const stepInput = document.createElement("input");
        stepInput.className = "rp-input";
        stepInput.type = "number";
        stepInput.value = String(editorStep);
        stepInput.oninput = () => updateEditorField(node, "step", stepInput.value);
        stepField.innerHTML = "<label>步长</label>";
        stepField.appendChild(stepInput);

        const chooseField = document.createElement("div");
        chooseField.className = "rp-field";
        const chooseLabel = document.createElement("label");
        chooseLabel.textContent = "默认预设";
        const chooseRow = document.createElement("label");
        chooseRow.className = "rp-check";
        const chooseInput = document.createElement("input");
        chooseInput.type = "checkbox";
        chooseInput.checked = !!state.editor.choose;
        chooseInput.onchange = () => updateEditorField(node, "choose", chooseInput.checked);
        chooseRow.appendChild(chooseInput);
        chooseRow.appendChild(document.createTextNode("保存时设为默认自定义预设"));
        chooseField.appendChild(chooseLabel);
        chooseField.appendChild(chooseRow);

        [nameField, widthField, heightField, stepField, chooseField].forEach((field) => editorGrid.appendChild(field));
        editorSection.appendChild(editorGrid);

        const actionRow = document.createElement("div");
        actionRow.className = "rp-actions";

        const newBtn = document.createElement("button");
        newBtn.type = "button";
        newBtn.className = "rp-btn";
        newBtn.textContent = "新建";
        newBtn.onclick = () => {
            createEmptyEditor(node);
            state.message = "";
            renderModalContent();
        };

        const alignBtn = document.createElement("button");
        alignBtn.type = "button";
        alignBtn.className = "rp-btn";
        alignBtn.textContent = "对齐宽高";
        alignBtn.onclick = () => {
            state.editor.width = editorWidth;
            state.editor.height = editorHeight;
            state.editor.step = editorStep;
            renderModalContent();
        };

        const saveBtn = document.createElement("button");
        saveBtn.type = "button";
        saveBtn.className = "rp-btn primary";
        saveBtn.textContent = "保存 / 更新";
        saveBtn.disabled = state.loading;
        saveBtn.onclick = async () => {
            state.loading = true;
            state.message = "";
            renderModalContent();
            try {
                const targetName = normalizePresetName(state.editor.name, state.constraints);
                if (!targetName) {
                    throw new Error("请输入预设名称。");
                }
                const payload = await fetchPresetPayload({
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        action: "upsert",
                        name: targetName,
                        w: editorWidth,
                        h: editorHeight,
                        step: editorStep,
                        choose: !!state.editor.choose,
                    }),
                });
                applyPayloadToState(node, payload);
                state.selectedPreset = targetName;
                resetEditorFromSelected(node);
                state.message = { type: "success", text: `预设 ${targetName} 已保存。` };
                renderNodePanel(node);
                renderModalContent();
            } catch (error) {
                state.message = { type: "error", text: error.message };
                renderModalContent();
            } finally {
                state.loading = false;
                renderNodePanel(node);
                renderModalContent();
            }
        };

        const deleteBtn = document.createElement("button");
        deleteBtn.type = "button";
        deleteBtn.className = "rp-btn danger";
        deleteBtn.textContent = "删除当前";
        deleteBtn.disabled = state.loading || !state.customPresets[state.selectedPreset];
        deleteBtn.onclick = async () => {
            if (!state.customPresets[state.selectedPreset]) {
                return;
            }
            if (!window.confirm(`确定删除自定义预设 "${state.selectedPreset}" 吗？`)) {
                return;
            }
            state.loading = true;
            state.message = "";
            renderModalContent();
            try {
                const payload = await fetchPresetPayload({
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        action: "delete",
                        name: state.selectedPreset,
                    }),
                });
                applyPayloadToState(node, payload);
                resetEditorFromSelected(node);
                state.message = { type: "success", text: "已删除当前自定义预设。" };
                renderNodePanel(node);
                renderModalContent();
            } catch (error) {
                state.message = { type: "error", text: error.message };
                renderModalContent();
            } finally {
                state.loading = false;
                renderNodePanel(node);
                renderModalContent();
            }
        };

        actionRow.appendChild(newBtn);
        actionRow.appendChild(alignBtn);
        actionRow.appendChild(saveBtn);
        actionRow.appendChild(deleteBtn);
        editorSection.appendChild(actionRow);
        content.appendChild(editorSection);

        const quickSection = document.createElement("div");
        quickSection.className = "rp-section";
        const quickHead = document.createElement("div");
        quickHead.className = "rp-section-head";
        quickHead.innerHTML = `
            <strong>快速切换</strong>
            <span class="rp-note">内置 ${Object.keys(state.builtinPresets).length} 项 / 自定义 ${Object.keys(state.customPresets).length} 项</span>
        `;
        quickSection.appendChild(quickHead);
        if (presetNames.length === 0) {
            const empty = document.createElement("div");
            empty.className = "rp-empty";
            empty.textContent = "当前没有可用预设。";
            quickSection.appendChild(empty);
        } else {
            const chipList = document.createElement("div");
            chipList.className = "rp-chip-list";
            presetNames.forEach((name) => {
                const chip = document.createElement("button");
                chip.type = "button";
                chip.className = `rp-chip ${name === state.selectedPreset ? "active" : ""}`.trim();
                chip.textContent = state.customPresets[name]?.choose ? `${name} · 默认` : name;
                chip.onclick = () => {
                    state.selectedPreset = name;
                    resetEditorFromSelected(node);
                    state.message = "";
                    syncBackingWidgets(node);
                    renderNodePanel(node);
                    renderModalContent();
                };
                chipList.appendChild(chip);
            });
            quickSection.appendChild(chipList);
        }
        content.appendChild(quickSection);

        const message = document.createElement("div");
        message.className = `rp-message ${state.message?.type || ""}`.trim();
        message.textContent = state.message?.text || "";
        content.appendChild(message);
    };

    renderModalContent();
    modal.show({
        title: "分辨率预设管理",
        content,
        width: "760px",
        buttons: [{ text: "关闭", onClick: () => modal.close() }],
    });
}

function renderNodePanel(node) {
    const state = node._resolutionPresetState;
    if (!state?.container) {
        return;
    }

    syncBackingWidgets(node);
    const info = getSelectedPresetInfo(state);
    const allPresets = getAllPresets(state);
    const presetNames = Object.keys(allPresets);

    state.container.innerHTML = "";

    const root = document.createElement("div");
    root.className = "rp-shell";

    const toolbar = document.createElement("div");
    toolbar.className = "rp-toolbar";
    const select = document.createElement("select");
    select.className = "rp-select";
    presetNames.forEach((name) => {
        const option = document.createElement("option");
        option.value = name;
        option.textContent = name;
        option.selected = name === state.selectedPreset;
        select.appendChild(option);
    });
    select.onchange = () => {
        state.selectedPreset = select.value;
        resetEditorFromSelected(node);
        state.message = "";
        syncBackingWidgets(node);
        renderNodePanel(node);
    };
    const mirrorBtn = document.createElement("button");
    mirrorBtn.type = "button";
    mirrorBtn.className = `rp-btn ${state.mirror ? "toggle-on" : ""}`.trim();
    mirrorBtn.textContent = state.mirror ? "镜像已开" : "镜像关闭";
    mirrorBtn.onclick = () => {
        state.mirror = !state.mirror;
        syncBackingWidgets(node);
        renderNodePanel(node);
    };
    const refreshBtn = document.createElement("button");
    refreshBtn.type = "button";
    refreshBtn.className = "rp-btn icon-only";
    refreshBtn.textContent = "↻";
    refreshBtn.disabled = state.loading;
    refreshBtn.onclick = async () => {
        state.loading = true;
        state.message = "";
        renderNodePanel(node);
        try {
            const payload = await fetchPresetPayload();
            applyPayloadToState(node, payload);
            resetEditorFromSelected(node);
            state.message = { type: "success", text: "已从后端重新加载预设。" };
        } catch (error) {
            state.message = { type: "error", text: error.message };
        } finally {
            state.loading = false;
            renderNodePanel(node);
        }
    };
    const manageBtn = document.createElement("button");
    manageBtn.type = "button";
    manageBtn.className = "rp-btn primary";
    manageBtn.textContent = "管理";
    manageBtn.onclick = () => openManagerModal(node);
    toolbar.appendChild(select);
    toolbar.appendChild(mirrorBtn);
    toolbar.appendChild(manageBtn);
    toolbar.appendChild(refreshBtn);
    root.appendChild(toolbar);

    const preview = document.createElement("div");
    preview.className = "rp-preview";
    const previewMeta = state.mirror && info.baseWidth && info.baseHeight
        ? `基础 ${info.baseWidth} x ${info.baseHeight}`
        : (info.isDefault ? "默认自定义预设" : "");
    preview.innerHTML = `
        <div class="rp-size">${info.width || "--"} x ${info.height || "--"}</div>
        ${previewMeta ? `<div class="rp-meta">${previewMeta}</div>` : ""}
    `;
    root.appendChild(preview);

    if (state.message?.text) {
        const message = document.createElement("div");
        message.className = `rp-message ${state.message?.type || ""}`.trim();
        message.textContent = state.message.text;
        root.appendChild(message);
    }

    state.container.appendChild(root);
    triggerNodeRefresh(node);
}

function ensureResolutionPresetUI(node) {
    if (!node || !node.widgets || node._resolutionPresetModernInit) {
        return;
    }

    const presetWidget = node.widgets.find((widget) => widget.name === "preset");
    const mirrorWidget = node.widgets.find((widget) => widget.name === "mirror");
    const customPresetsWidget = node.widgets.find((widget) => widget.name === "custom_presets");
    if (!presetWidget || !mirrorWidget || !customPresetsWidget) {
        console.error("[ResolutionPresetNode] 缺少必要控件。");
        return;
    }

    removeLegacyWidgets(node);
    [presetWidget, mirrorWidget, customPresetsWidget].forEach((widget) => {
        widget.computeSize = () => [0, 0];
        widget.type = "hidden";
        widget.hidden = true;
        widget.options = { ...(widget.options || {}), hidden: true };
    });

    const state = {
        presetWidget,
        mirrorWidget,
        customPresetsWidget,
        constraints: { ...DEFAULT_CONSTRAINTS },
        builtinPresets: normalizePresetMap(FALLBACK_BUILTINS, DEFAULT_CONSTRAINTS, true),
        customPresets: parseCustomPresetsFromWidget(customPresetsWidget, DEFAULT_CONSTRAINTS),
        selectedPreset: String(presetWidget.value || ""),
        mirror: !!mirrorWidget.value,
        loading: false,
        message: "",
        container: null,
        editor: null,
    };
    node._resolutionPresetState = state;

    const container = document.createElement("div");
    container.style.width = "100%";
    container.style.height = `calc(100% + ${PANEL_OFFSET_Y}px)`;
    container.style.boxSizing = "border-box";
    container.style.marginTop = `-${PANEL_OFFSET_Y}px`;
    state.container = container;

    if (typeof node.addDOMWidget === "function") {
        node.addDOMWidget("ResolutionPresetModernUI", "div", container, {
            serialize: false,
            hideOnZoom: false,
        });
    }

    if (!node.size || node.size[0] < 360 || node.size[1] < 122) {
        node.size = [360, 122];
    }

    if (!state.selectedPreset || !getAllPresets(state)[state.selectedPreset]) {
        state.selectedPreset = getDefaultCustomName(state.customPresets) || Object.keys(getAllPresets(state))[0] || "";
    }
    resetEditorFromSelected(node);
    syncBackingWidgets(node);
    renderNodePanel(node);

    node._resolutionPresetModernInit = true;
    node._resolutionPresetRender = () => renderNodePanel(node);

    fetchPresetPayload()
        .then((payload) => {
            applyPayloadToState(node, payload);
            state.mirror = !!mirrorWidget.value;
            resetEditorFromSelected(node);
            renderNodePanel(node);
        })
        .catch((error) => {
            state.message = { type: "error", text: `读取后端预设失败: ${error.message}` };
            renderNodePanel(node);
        });
}

injectStyles();

app.registerExtension({
    name: "A_my_nodes.ResolutionPresetNode.ModernUI",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "ResolutionPresetNode") {
            return;
        }

        const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = originalOnNodeCreated?.apply(this, arguments);
            ensureResolutionPresetUI(this);
            return result;
        };

        const originalOnConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = originalOnConfigure?.apply(this, arguments);
            ensureResolutionPresetUI(this);
            if (this._resolutionPresetState) {
                this._resolutionPresetState.selectedPreset = String(this._resolutionPresetState.presetWidget.value || "");
                this._resolutionPresetState.mirror = !!this._resolutionPresetState.mirrorWidget.value;
                this._resolutionPresetState.customPresets = parseCustomPresetsFromWidget(
                    this._resolutionPresetState.customPresetsWidget,
                    this._resolutionPresetState.constraints
                );
                resetEditorFromSelected(this);
                this._resolutionPresetRender?.();
            }
            return result;
        };
    },
});
