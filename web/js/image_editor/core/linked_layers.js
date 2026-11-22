// 本文件由主项目 tools/sync_image_editor_core.py 自动生成，禁止手改。
// 源文件: apps/web_apps/static/image_editor/linked_layers.js
const hexColor = value => {
    const normalized = /^#[0-9a-f]{6}$/i.test(String(value || "")) ? value.slice(1) : "ff0000";
    return [0, 2, 4].map(index => Number.parseInt(normalized.slice(index, index + 2), 16));
};

/** 从唯一可编辑的源图层派生另一图层；派生层不保留独立笔划。 */
export function deriveLinkedLayerData(sourceData, direction, paintColor = '#ff0000', opacity = 1) {
    const source = sourceData?.data || sourceData;
    if (!source || source.length % 4 !== 0) throw new Error("图层像素数据无效");
    const result = new Uint8ClampedArray(source.length);
    const factor = Math.max(0, Math.min(1, Number(opacity) || 0));
    const [red, green, blue] = hexColor(paintColor);
    for (let index = 0; index < source.length; index += 4) {
        const alpha = Math.round(source[index + 3] * factor);
        if (!alpha) continue;
        if (direction === 'mask_to_paint') {
            result[index] = red;
            result[index + 1] = green;
            result[index + 2] = blue;
        }
        result[index + 3] = alpha;
    }
    return result;
}
