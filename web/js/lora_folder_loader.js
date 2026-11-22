import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";
import { LoraWidget } from "./lora/lora_common.js";
import { RgthreeBaseWidget, addIconButtonRow } from "./utils/utils_widgets.js";
import { showTopNotification } from "./utils/shared_utils.js";

const NODE_NAME = "LoraFolderLoader";
const ROW_PREFIX = "LORA_";
const ACTION_WIDGET_NAME = "lora_folder_actions";
const SCROLLBAR_WIDGET_NAME = "lora_scrollbar";
const LIST_ROUTE = "/a_my_nodes/lora_folder/list";
const MIN_NODE_WIDTH = 420;
const MIN_NODE_HEIGHT = 130;
const DEFAULT_STRENGTH = 1.0;
const MAX_VISIBLE_ROWS = 10;
const SCROLLBAR_WIDTH = 5;
const SCROLLBAR_MARGIN = 3;
const FILTER_IGNORED_TOKENS = new Set(["存在"]);

function isRowWidget(widget) {
    return !!(widget && typeof widget.name === "string" && widget.name.startsWith(ROW_PREFIX));
}

function getRowWidgets(node) {
    return (node.widgets || []).filter(isRowWidget);
}

// 完整行模型：包含被过滤隐藏的行，控件只渲染其中匹配的一部分
function getSourceRows(node) {
    if (!node) return [];
    if (!Array.isArray(node._loraRows)) {
        node._loraRows = [];
    }
    return node._loraRows;
}

function normalizeKey(value) {
    return String(value || "").replace(/\\/g, "/").toLowerCase();
}

function getMode(node) {
    return ((node && node.widgets) || []).find((widget) => widget.name === "mode")?.value || "多选";
}

function getKeywordFilter(node) {
    return String(((node && node.widgets) || []).find((widget) => widget.name === "keyword_filter")?.value || "");
}

function normalizeRow(row) {
    return {
        on: !!row?.on,
        lora: String(row?.lora || row?.name || ""),
        strength: Number.isFinite(Number(row?.strength)) ? Number(row.strength) : DEFAULT_STRENGTH,
        path: String(row?.path || ""),
    };
}

// 过滤表达式解析：空格=并(OR)，且=AND，并=OR；“且”优先于“并”
function parseFilterExpression(keywordFilter) {
    const tokens = String(keywordFilter || "")
        .replace(/且/g, " 且 ")
        .replace(/并/g, " 并 ")
        .split(/\s+/)
        .filter(Boolean);

    const orGroups = [];
    let andGroups = [];
    let terms = [];

    const flushTerms = () => {
        if (terms.length) {
            andGroups.push(terms);
            terms = [];
        }
    };
    const flushAndGroup = () => {
        flushTerms();
        if (andGroups.length) {
            orGroups.push(andGroups);
            andGroups = [];
        }
    };

    for (const token of tokens) {
        if (token === "并") {
            flushAndGroup();
        } else if (token === "且") {
            flushTerms();
        } else if (!FILTER_IGNORED_TOKENS.has(token)) {
            terms.push(token.toLowerCase());
        }
    }
    flushAndGroup();
    return orGroups;
}

function matchesFilter(name, keywordFilter) {
    const groups = parseFilterExpression(keywordFilter);
    if (!groups.length) return true;
    const haystack = String(name || "").toLowerCase();
    return groups.some((andGroups) =>
        andGroups.every((terms) => terms.some((term) => haystack.includes(term)))
    );
}

function getFilteredRows(node) {
    const keywordFilter = getKeywordFilter(node);
    return getSourceRows(node).filter((row) => matchesFilter(row.lora || row.path, keywordFilter));
}

function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
}

function getScrollState(node) {
    const total = getFilteredRows(node).length;
    const maxStart = Math.max(0, total - MAX_VISIBLE_ROWS);
    return {
        total,
        maxStart,
        start: clamp(Number(node?._loraVisibleStart) || 0, 0, maxStart),
        visible: Math.min(total, MAX_VISIBLE_ROWS),
    };
}

// 文件夹扫描出来的 LoRA 行：名称固定，不弹出官方 LoRA 选择器
class FolderLoraRowWidget extends LoraWidget {
    constructor(name) {
        super(name);
        this._value = { on: false, lora: "", strength: DEFAULT_STRENGTH, path: "" };
    }

    mouse(event, pos, node) {
        // 滚动条覆盖在行右侧，优先消费这里的指针事件
        if (handleScrollbarMouse(event, pos, node)) {
            return true;
        }
        return super.mouse(event, pos, node);
    }

    onLoraDown() {
        return true;
    }

    onToggleDown(event, pos, node) {
        const next = !this.value.on;
        this.value.on = next;
        if (next && getMode(node) === "单选") {
            for (const row of getSourceRows(node)) {
                if (row !== this.value) row.on = false;
            }
        }
        this.cancelMouseDown();
        node?.setDirtyCanvas?.(true, true);
        syncLorasInfo(node);
        return true;
    }

    onStrengthDecDown(event, pos, node) {
        super.onStrengthDecDown(event, pos, node);
        syncLorasInfo(node);
    }

    onStrengthIncDown(event, pos, node) {
        super.onStrengthIncDown(event, pos, node);
        syncLorasInfo(node);
    }

    onStrengthAnyMove(event, pos, node) {
        super.onStrengthAnyMove(event, pos, node);
        syncLorasInfo(node);
    }

    onStrengthValDown(event, pos, node) {
        const now = Date.now();
        if (this.lastClickTime && (now - this.lastClickTime < 300)) {
            app.canvas.prompt("强度值", this.value.strength ?? DEFAULT_STRENGTH, (value) => {
                const parsed = Number(value);
                if (Number.isFinite(parsed)) {
                    this.value.strength = parsed;
                }
                syncLorasInfo(node);
                node?.setDirtyCanvas?.(true, true);
            }, event);
            this.lastClickTime = 0;
            return true;
        }
        this.lastClickTime = now;
        return true;
    }
}

// 只负责绘制/拖动滚动条，不占行高
class LoraScrollbarWidget extends RgthreeBaseWidget {
    constructor() {
        super(SCROLLBAR_WIDGET_NAME);
        this.type = "custom";
        this.value = "";
        this.serialize = false;
        this.options = { serialize: false };
    }

    computeSize() {
        return [0, -4];
    }

    serializeValue() {
        return undefined;
    }

    draw(ctx, node) {
        drawScrollbar(ctx, node);
    }

    mouse(event, pos, node) {
        return handleScrollbarMouse(event, pos, node);
    }
}

// 记录已知行的状态，避免关闭“列出子文件夹”后刷新导致隐藏行状态丢失
function getStateCache(node) {
    if (!(node._loraStateCache instanceof Map)) {
        node._loraStateCache = new Map();
    }
    return node._loraStateCache;
}

function rememberRowStates(node) {
    const cache = getStateCache(node);
    for (const row of getSourceRows(node)) {
        const key = normalizeKey(row.path);
        if (!key) continue;
        cache.set(key, {
            on: !!row.on,
            strength: Number.isFinite(Number(row.strength)) ? Number(row.strength) : DEFAULT_STRENGTH,
        });
    }
}

// 把全部行状态（含被过滤隐藏的行）写入隐藏控件，随工作流保存；后端也读取该字段
function syncLorasInfo(node) {
    if (!node) return;
    rememberRowStates(node);
    const infoWidget = (node.widgets || []).find((widget) => widget.name === "loras_info");
    if (!infoWidget) return;
    const folderWidget = (node.widgets || []).find((widget) => widget.name === "folder");
    const serialized = JSON.stringify({
        folder: folderWidget?.value || "",
        filter: getKeywordFilter(node),
        rows: getSourceRows(node).map((row) => ({ ...row })),
    });
    if (infoWidget.value === serialized) return;
    infoWidget.value = serialized;
    // 直接改 value 不会触发前端的变更跟踪，这里补一次，确保自动保存和缓存失效
    if (!node._loraConfiguring && node.graph && typeof node.graph.change === "function") {
        node.graph.change();
    }
}

function ensureInfoWidget(node) {
    let widget = (node.widgets || []).find((w) => w.name === "loras_info");
    if (!widget && typeof node.addWidget === "function") {
        widget = node.addWidget("text", "loras_info", "{}", () => {}, { serialize: true });
    }
    if (widget) {
        // 不占行高、不绘制，但必须保留为可序列化、可发送给后端的控件
        widget.computeSize = () => [0, -4];
        widget.draw = () => {};
        widget.serialize = true;
        if (widget.options) widget.options.serialize = true;
    }
    return widget;
}

function ensureScrollbarWidget(node) {
    let widget = (node.widgets || []).find((w) => w.name === SCROLLBAR_WIDGET_NAME);
    if (!widget) {
        widget = new LoraScrollbarWidget();
        node.widgets.push(widget);
    }
    return widget;
}

function addRefreshButton(node) {
    if ((node.widgets || []).some((widget) => widget.name === ACTION_WIDGET_NAME)) return;
    const widget = addIconButtonRow(node, ACTION_WIDGET_NAME, [
        { icon: "🔄", title: "刷新 LoRA 列表", onClick: () => refreshRows(node) },
    ]);
    if (widget) widget.serialize = false;
}

function enforceSingleMode(node) {
    if (getMode(node) !== "单选") return;
    let kept = false;
    for (const row of getSourceRows(node)) {
        if (!row.on) continue;
        if (kept) row.on = false;
        else kept = true;
    }
}

function hookModeWidget(node) {
    const modeWidget = (node.widgets || []).find((widget) => widget.name === "mode");
    if (!modeWidget || modeWidget._loraFolderHooked) return;
    modeWidget._loraFolderHooked = true;
    const originalCallback = modeWidget.callback;
    modeWidget.callback = function (value, ...rest) {
        originalCallback?.apply(this, [value, ...rest]);
        if (node._loraConfiguring) return;
        requestWidgetStateSync(node, false);
    };
}

function hookFilterWidget(node) {
    const filterWidget = (node.widgets || []).find((widget) => widget.name === "keyword_filter");
    if (!filterWidget || filterWidget._loraFolderHooked) return;
    filterWidget._loraFolderHooked = true;
    const originalCallback = filterWidget.callback;
    filterWidget.callback = function (value, ...rest) {
        originalCallback?.apply(this, [value, ...rest]);
        if (node._loraConfiguring) return;
        requestWidgetStateSync(node, true);
    };
}

function hookWidgets(node) {
    hookModeWidget(node);
    hookFilterWidget(node);
}

function createRowWidget(node, row) {
    node.loraRowCounter = (node.loraRowCounter || 0) + 1;
    const widget = new FolderLoraRowWidget(ROW_PREFIX + node.loraRowCounter);
    // 共享引用：控件上的开关/强度改动会直接写回完整行模型
    widget.value = row;
    const scrollbarIndex = (node.widgets || []).findIndex((w) => w.name === SCROLLBAR_WIDGET_NAME);
    if (scrollbarIndex >= 0) {
        node.widgets.splice(scrollbarIndex, 0, widget);
    } else {
        node.widgets.push(widget);
    }
    return widget;
}

function updateListGeometry(node) {
    const rows = getRowWidgets(node);
    if (!rows.length) {
        node._loraListRect = null;
        node._loraScrollThumb = null;
        return;
    }
    const rowHeight = (typeof LiteGraph !== "undefined" && LiteGraph.NODE_WIDGET_HEIGHT) || 20;
    const top = rows[0].y ?? rows[0].last_y ?? 0;
    node._loraListRect = {
        x: 10,
        y: top,
        w: Math.max(20, (node.size?.[0] || 0) - 20),
        h: rows.length * (rowHeight + 4) - 4,
    };
}

function resizeToFit(node) {
    if (!node) return;
    const computed = typeof node.computeSize === "function" ? node.computeSize() : null;
    const rowHeight = ((typeof LiteGraph !== "undefined" && LiteGraph.NODE_WIDGET_HEIGHT) || 20) + 4;
    // 固定控件预留高度，再按当前可见行数估算；窗口化后高度不会再无限增长
    const estimatedHeight = 110 + getRowWidgets(node).length * rowHeight;
    const width = Math.max(node.size?.[0] || 0, computed?.[0] || 0, MIN_NODE_WIDTH);
    const height = Math.max(computed?.[1] || 0, estimatedHeight, MIN_NODE_HEIGHT);
    node.setSize?.([width, height]);
    node.setDirtyCanvas?.(true, true);
}

// 只渲染滚动窗口内的匹配行
function renderVisibleRows(node) {
    const matched = getFilteredRows(node);
    const maxStart = Math.max(0, matched.length - MAX_VISIBLE_ROWS);
    node._loraVisibleStart = clamp(Number(node._loraVisibleStart) || 0, 0, maxStart);

    node.widgets = (node.widgets || []).filter((widget) => !isRowWidget(widget));
    node.loraRowCounter = 0;
    const start = node._loraVisibleStart;
    for (const row of matched.slice(start, start + MAX_VISIBLE_ROWS)) {
        createRowWidget(node, row);
    }
    ensureScrollbarWidget(node);

    if (typeof node.arrange === "function") {
        try {
            node.arrange();
        } catch (error) {
            console.warn("[LoraFolderLoader] 重新排列控件失败:", error);
        }
    }
    updateListGeometry(node);
    resizeToFit(node);
    // 记录本次渲染使用的控件值，供绘制时变化检测使用
    node._loraLastFilter = getKeywordFilter(node);
    node._loraLastMode = getMode(node);
}

// 统一调度一次重渲染；callback、onWidgetChanged、绘制兜底共用，避免重复执行
function requestWidgetStateSync(node, resetScroll = false) {
    if (!node || node._loraConfiguring || node._loraSyncScheduled) return;
    if (resetScroll) {
        node._loraVisibleStart = 0;
    }
    node._loraSyncScheduled = true;
    setTimeout(() => {
        node._loraSyncScheduled = false;
        if (!node.widgets) return;
        enforceSingleMode(node);
        syncLorasInfo(node);
        renderVisibleRows(node);
    }, 0);
}

// 原生控件的 callback 在新前端里并不总是可靠触发，这里在绘制时兜底检测值变化
function scheduleWidgetStateSync(node) {
    if (!node || node._loraConfiguring) return;
    const keywordFilter = getKeywordFilter(node);
    const mode = getMode(node);
    const changed = node._loraLastFilter !== undefined &&
        (keywordFilter !== node._loraLastFilter || mode !== node._loraLastMode);
    if (!changed) return;
    requestWidgetStateSync(node, keywordFilter !== node._loraLastFilter);
}

function setScrollStart(node, value) {
    const state = getScrollState(node);
    const next = clamp(Math.round(value), 0, state.maxStart);
    if (next === state.start) {
        node.setDirtyCanvas?.(true, true);
        return;
    }
    node._loraVisibleStart = next;
    renderVisibleRows(node);
    node.setDirtyCanvas?.(true, true);
}

function drawScrollbar(ctx, node) {
    const rect = node?._loraListRect;
    const state = getScrollState(node);
    if (!rect || state.total <= MAX_VISIBLE_ROWS) {
        node._loraScrollThumb = null;
        return;
    }

    const barX = (node.size?.[0] || 0) - SCROLLBAR_WIDTH - SCROLLBAR_MARGIN;
    const trackY = rect.y + 2;
    const trackH = Math.max(24, rect.h - 4);
    const thumbH = Math.max(18, trackH * state.visible / state.total);
    const progress = state.maxStart ? state.start / state.maxStart : 0;
    const thumbY = trackY + (trackH - thumbH) * progress;
    const radius = SCROLLBAR_WIDTH / 2;

    ctx.save();
    ctx.beginPath();
    ctx.fillStyle = "rgba(255, 255, 255, 0.10)";
    ctx.roundRect(barX, trackY, SCROLLBAR_WIDTH, trackH, radius);
    ctx.fill();
    ctx.beginPath();
    ctx.fillStyle = node._loraScrollDrag
        ? "rgba(255, 255, 255, 0.70)"
        : "rgba(255, 255, 255, 0.35)";
    ctx.roundRect(barX, thumbY, SCROLLBAR_WIDTH, thumbH, radius);
    ctx.fill();
    ctx.restore();

    node._loraScrollThumb = {
        x: barX,
        y: thumbY,
        w: SCROLLBAR_WIDTH,
        h: thumbH,
        trackY,
        trackH,
    };
}

function handleScrollbarMouse(event, pos, node) {
    if (!node || !pos) return false;
    const dragging = !!node._loraScrollDrag;
    const thumb = node._loraScrollThumb;
    const state = getScrollState(node);

    // 拖动中要在命中检测之前处理，避免拖出轨道后状态卡住
    if (event.type === "pointerup") {
        if (dragging) {
            node._loraScrollDrag = null;
            node.setDirtyCanvas?.(true, true);
            return true;
        }
        return false;
    }
    if (event.type === "pointermove" && dragging) {
        if (!thumb) return false;
        const travel = Math.max(1, thumb.trackH - thumb.h);
        const deltaRows = Math.round((pos[1] - node._loraScrollDrag.startY) / travel * state.maxStart);
        setScrollStart(node, node._loraScrollDrag.startRow + deltaRows);
        return true;
    }

    if (!thumb || state.total <= MAX_VISIBLE_ROWS) return false;
    const hitX = pos[0] >= thumb.x - 6 && pos[0] <= thumb.x + thumb.w + 6;
    const inTrackY = pos[1] >= thumb.trackY && pos[1] <= thumb.trackY + thumb.trackH;
    if (!hitX || !inTrackY) return false;

    if (event.type === "pointerdown") {
        if (pos[1] >= thumb.y && pos[1] <= thumb.y + thumb.h) {
            node._loraScrollDrag = { startY: pos[1], startRow: state.start };
        } else {
            setScrollStart(node, state.start + (pos[1] < thumb.y ? -MAX_VISIBLE_ROWS : MAX_VISIBLE_ROWS));
        }
        return true;
    }
    return false;
}

// 在列表区域滚轮滚动（画布本身不向节点转发 wheel，这里用捕获监听拦截）
function handleGlobalWheel(event) {
    const canvas = app.canvas;
    const mouse = canvas?.graph_mouse;
    if (!canvas || !mouse) return;
    const nodes = canvas.graph?._nodes || canvas.graph?.nodes || [];
    for (const node of nodes) {
        if (node.comfyClass !== NODE_NAME && node.type !== NODE_NAME) continue;
        const rect = node._loraListRect;
        if (!rect) continue;
        const localX = mouse[0] - node.pos[0];
        const localY = mouse[1] - node.pos[1];
        if (localX < 6 || localX > (node.size?.[0] || 0) - 6) continue;
        if (localY < rect.y - 2 || localY > rect.y + rect.h + 2) continue;

        const state = getScrollState(node);
        if (state.total <= MAX_VISIBLE_ROWS) continue;
        const step = event.deltaY > 0 ? 1 : event.deltaY < 0 ? -1 : 0;
        if (!step) continue;
        event.preventDefault();
        event.stopPropagation();
        setScrollStart(node, state.start + step * (event.shiftKey ? MAX_VISIBLE_ROWS : 1));
        return;
    }
}

if (!window.__loraFolderWheelHooked) {
    window.__loraFolderWheelHooked = true;
    window.addEventListener("wheel", handleGlobalWheel, { capture: true, passive: false });
}

// 显示后端执行结果（匹配/应用数量、无行数据等）
if (!window.__loraFolderNotifyHooked) {
    window.__loraFolderNotifyHooked = true;
    api.addEventListener("a_my_nodes.lora_folder_loader", (event) => {
        const detail = event?.detail || {};
        if (detail.message) {
            showTopNotification(detail.message, detail.level || "info");
        }
    });
}

function parseRowsPayload(value) {
    if (!value) return [];
    let data = value;
    if (typeof value === "string") {
        try {
            data = JSON.parse(value);
        } catch (error) {
            return [];
        }
    }
    if (data && Array.isArray(data.rows)) return data.rows;
    if (Array.isArray(data)) return data;
    return [];
}

function readSavedRows(node, info) {
    const infoWidget = (node.widgets || []).find((widget) => widget.name === "loras_info");
    let rows = parseRowsPayload(infoWidget?.value);
    if (rows.length) return rows;

    // 新版前端可能按名称保存控件值
    rows = parseRowsPayload(info?.widgets_values_named?.loras_info);
    if (rows.length) return rows;

    // 兜底：位置序列化的值里找 loras_info 的 JSON 字符串
    for (const value of (info?.widgets_values || [])) {
        rows = parseRowsPayload(value);
        if (rows.length) return rows;
    }
    return [];
}

// 兼容隐藏控件未生成/旧数据：从 widgets_values 或按名保存的值中捞出被序列化的行对象
function isRowPayload(value) {
    return !!(value && typeof value === "object" && !Array.isArray(value) &&
        typeof value.lora === "string" && typeof value.path === "string");
}

function extractRowsFromWidgetsValues(info) {
    let rows = (info?.widgets_values || []).filter(isRowPayload);
    if (!rows.length) {
        rows = Object.values(info?.widgets_values_named || {}).filter(isRowPayload);
    }
    return rows;
}

async function refreshRows(node) {
    if (node?._loraConfiguring) return;
    const folderWidget = (node.widgets || []).find((widget) => widget.name === "folder");
    const folder = String(folderWidget?.value || "").trim();
    if (!folder) {
        showTopNotification("请先填写 LoRA 文件夹路径", "warning");
        return;
    }

    let data;
    try {
        const includeSubfolders = (node.widgets || [])
            .find((widget) => widget.name === "include_subfolders")?.value !== false;
        const response = await api.fetchApi(
            `${LIST_ROUTE}?path=${encodeURIComponent(folder)}&recursive=${includeSubfolders ? "1" : "0"}`,
            { cache: "no-store" }
        );
        data = await response.json().catch(() => ({}));
        if (!response.ok) {
            throw new Error(data?.error || `HTTP ${response.status}`);
        }
    } catch (error) {
        console.error("[LoraFolderLoader] 扫描文件夹失败:", error);
        showTopNotification(`扫描失败：${error.message || error}`, "error");
        return;
    }

    // 先按路径（其次按文件名）保留已有行状态，包括当前被过滤隐藏的行和缓存中的历史行
    rememberRowStates(node);
    const previous = new Map(getStateCache(node));
    for (const row of getSourceRows(node)) {
        const state = {
            on: !!row.on,
            strength: Number.isFinite(Number(row.strength)) ? Number(row.strength) : DEFAULT_STRENGTH,
        };
        for (const key of [row.path, row.lora]) {
            const normalized = normalizeKey(key);
            if (normalized && !previous.has(normalized)) previous.set(normalized, state);
        }
    }

    const files = Array.isArray(data.files) ? data.files : [];
    node._loraRows = files.map((item) => {
        const saved = previous.get(normalizeKey(item.path)) || previous.get(normalizeKey(item.name));
        return normalizeRow({
            lora: item.name,
            path: item.path,
            on: saved ? saved.on : false,
            strength: saved ? saved.strength : DEFAULT_STRENGTH,
        });
    });
    node._loraVisibleStart = 0;

    enforceSingleMode(node);
    syncLorasInfo(node);
    renderVisibleRows(node);

    const matchedCount = getFilteredRows(node).length;
    showTopNotification(`已加载 ${files.length} 个 LoRA，当前匹配 ${matchedCount} 个`, "success");
}

app.registerExtension({
    name: "A_my_nodes.LoraFolderLoader.UI",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated?.apply(this, arguments);
            this.serialize_widgets = true;
            this.loraRowCounter = 0;
            this._loraRows = [];
            this._loraVisibleStart = 0;
            this._loraConfiguring = false;
            this._loraLastFilter = getKeywordFilter(this);
            this._loraLastMode = getMode(this);
            ensureInfoWidget(this);
            addRefreshButton(this);
            ensureScrollbarWidget(this);
            hookWidgets(this);
            resizeToFit(this);
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            // 配置/复制节点期间禁止自动刷新，避免扫描结果覆盖刚恢复的行状态
            this._loraConfiguring = true;
            try {
                onConfigure?.apply(this, arguments);
                this.widgets = (this.widgets || []).filter((widget) => !isRowWidget(widget));
                this.loraRowCounter = 0;
                this.serialize_widgets = true;
                this._loraVisibleStart = 0;
                // 旧工作流没有该布尔控件时，可能被位移成对象，这里纠正为默认勾选
                const subfoldersWidget = (this.widgets || []).find((widget) => widget.name === "include_subfolders");
                if (subfoldersWidget && typeof subfoldersWidget.value !== "boolean") {
                    subfoldersWidget.value = true;
                }
                ensureInfoWidget(this);
                addRefreshButton(this);
                ensureScrollbarWidget(this);
                hookWidgets(this);

                // 依次从控件值、按名保存的值、位置值、行对象中恢复
                let rows = readSavedRows(this, info);
                if (!rows.length) rows = extractRowsFromWidgetsValues(info);
                this._loraRows = rows.map(normalizeRow);
                enforceSingleMode(this);
                syncLorasInfo(this);
                renderVisibleRows(this);
            } finally {
                this._loraConfiguring = false;
            }
        };

        const onSerialize = nodeType.prototype.onSerialize;
        nodeType.prototype.onSerialize = function (info) {
            syncLorasInfo(this);
            onSerialize?.apply(this, arguments);
        };

        const onDrawForeground = nodeType.prototype.onDrawForeground;
        nodeType.prototype.onDrawForeground = function () {
            onDrawForeground?.apply(this, arguments);
            scheduleWidgetStateSync(this);
        };

        const onWidgetChanged = nodeType.prototype.onWidgetChanged;
        nodeType.prototype.onWidgetChanged = function (name) {
            onWidgetChanged?.apply(this, arguments);
            if (this._loraConfiguring) return;
            if (name === "keyword_filter") {
                requestWidgetStateSync(this, true);
            } else if (name === "mode") {
                requestWidgetStateSync(this, false);
            } else if (name === "include_subfolders") {
                const folder = String(((this.widgets || []).find((widget) => widget.name === "folder")?.value) || "").trim();
                if (folder) {
                    refreshRows(this);
                }
            }
        };
    },
});
