console.log("Loading custom node: A_my_nodes/web/js/bg_removal_colorize.js");

import { app } from "../../../scripts/app.js";
import { chainCallback } from "./utils/common.js";

const NODE_NAME = "BackgroundRemovalColorize";

function getWidget(node, widgetName) {
    return node.widgets?.find((widget) => widget.name === widgetName);
}

function clampColorChannel(value) {
    return Math.max(0, Math.min(255, value));
}

function rgbToHex(r, g, b) {
    return `#${[r, g, b].map((item) => clampColorChannel(item).toString(16).padStart(2, "0")).join("")}`.toUpperCase();
}

function normalizeColorString(value) {
    if (typeof value !== "string") {
        return null;
    }

    const trimmed = value.trim();
    if (!trimmed) {
        return null;
    }

    const hex = trimmed.startsWith("#") ? trimmed.slice(1) : trimmed;
    if (/^[0-9a-fA-F]{6}$/.test(hex)) {
        return `#${hex.toUpperCase()}`;
    }

    const rgbParts = trimmed.match(/\d+/g);
    if (rgbParts?.length === 3) {
        const [r, g, b] = rgbParts.map((item) => Number.parseInt(item, 10));
        if ([r, g, b].every((item) => Number.isFinite(item))) {
            return rgbToHex(r, g, b);
        }
    }

    return null;
}

function getContrastTextColor(hexColor) {
    const hex = hexColor.slice(1);
    const r = Number.parseInt(hex.slice(0, 2), 16);
    const g = Number.parseInt(hex.slice(2, 4), 16);
    const b = Number.parseInt(hex.slice(4, 6), 16);
    const luminance = (0.299 * r) + (0.587 * g) + (0.114 * b);
    return luminance > 160 ? "#111111" : "#F5F5F5";
}

function setSourceWidgetValue(node, widgetName, colorValue) {
    const sourceWidget = getWidget(node, widgetName);
    if (!sourceWidget) {
        return;
    }

    const normalized = normalizeColorString(colorValue) || colorValue;
    sourceWidget.value = normalized;

    if (sourceWidget.inputEl) {
        sourceWidget.inputEl.value = normalized;
    }

    node.setDirtyCanvas(true, true);
}

function ensureValidColorValue(node, widgetName, fallbackColor) {
    const sourceWidget = getWidget(node, widgetName);
    if (!sourceWidget) {
        return fallbackColor;
    }

    const normalized = normalizeColorString(sourceWidget.value) || fallbackColor;
    if (sourceWidget.value !== normalized) {
        setSourceWidgetValue(node, widgetName, normalized);
    }
    return normalized;
}

function buildColorPickerElement(node, widgetName, label, fallbackColor) {
    const sourceWidget = getWidget(node, widgetName);
    if (!sourceWidget) {
        return null;
    }

    const wrapper = document.createElement("div");
    Object.assign(wrapper.style, {
        position: "relative",
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        gap: "10px",
        width: "100%",
        minHeight: "24px",
        height: "24px",
        padding: "2px 8px",
        boxSizing: "border-box",
        borderRadius: "6px",
        border: "1px solid rgba(255,255,255,0.16)",
        cursor: "default",
        userSelect: "none",
        overflow: "hidden",
    });

    const labelEl = document.createElement("span");
    Object.assign(labelEl.style, {
        fontSize: "12px",
        fontWeight: "600",
        opacity: "0.92",
        lineHeight: "1",
    });
    labelEl.textContent = label;

    const valueEl = document.createElement("span");
    Object.assign(valueEl.style, {
        fontSize: "12px",
        fontFamily: "monospace",
        fontWeight: "700",
        lineHeight: "1",
    });

    const input = document.createElement("input");
    input.type = "color";
    Object.assign(input.style, {
        position: "absolute",
        inset: "0",
        width: "100%",
        height: "100%",
        opacity: "0",
        pointerEvents: "auto",
        cursor: "pointer",
        border: "none",
        padding: "0",
        margin: "0",
    });

    const syncView = () => {
        const colorValue = ensureValidColorValue(node, widgetName, fallbackColor);
        wrapper.style.background = colorValue;
        wrapper.style.boxShadow = "inset 0 0 0 1px rgba(0,0,0,0.10)";
        wrapper.title = `当前颜色: ${colorValue}`;
        labelEl.style.color = getContrastTextColor(colorValue);
        valueEl.style.color = getContrastTextColor(colorValue);
        valueEl.textContent = colorValue;
        input.value = colorValue;
    };

    input.addEventListener("input", () => {
        setSourceWidgetValue(node, widgetName, input.value);
        syncView();
    });

    wrapper.appendChild(labelEl);
    wrapper.appendChild(valueEl);
    wrapper.appendChild(input);
    syncView();

    return { wrapper, syncView };
}

function ensureColorDomWidgets(node) {
    if (node.__amyColorDomInstalled) {
        node.__amyColorDomSyncFns?.forEach((fn) => fn());
        return;
    }

    const backgroundWidget = getWidget(node, "background_color");
    const fillWidget = getWidget(node, "fill_color");
    if (!backgroundWidget || !fillWidget || typeof node.addDOMWidget !== "function") {
        return;
    }

    ensureValidColorValue(node, "background_color", "#FFFFFF");
    ensureValidColorValue(node, "fill_color", "#000000");

    backgroundWidget.hidden = true;
    fillWidget.hidden = true;

    const bgPicker = buildColorPickerElement(node, "background_color", "背景色", "#FFFFFF");
    const fillPicker = buildColorPickerElement(node, "fill_color", "填充色", "#000000");
    if (!bgPicker || !fillPicker) {
        return;
    }

    const container = document.createElement("div");
    Object.assign(container.style, {
        display: "flex",
        flexDirection: "column",
        gap: "3px",
        width: "100%",
        margin: "0",
        padding: "0",
        boxSizing: "border-box",
    });
    container.appendChild(bgPicker.wrapper);
    container.appendChild(fillPicker.wrapper);

    const colorDomWidget = node.addDOMWidget("color_picker_group_dom", "div", container, {
        serialize: false,
        hideOnZoom: false,
    });
    colorDomWidget.options.serialize = false;

    node.__amyColorDomInstalled = true;
    node.__amyColorDomSyncFns = [bgPicker.syncView, fillPicker.syncView];

    console.log(`[A_my_nodes] Patched node UI: ${node.type}`);
}

app.registerExtension({
    name: "A_my_nodes.BackgroundRemovalColorize.JS",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) {
            return;
        }

        console.log(`Patching node: ${nodeData.name}`);

        chainCallback(nodeType.prototype, "onNodeCreated", function() {
            ensureColorDomWidgets(this);
        });

        chainCallback(nodeType.prototype, "onConfigure", function() {
            setTimeout(() => ensureColorDomWidgets(this), 0);
        });
    },
});
