// 本文件由主项目 tools/sync_image_editor_core.py 自动生成，禁止手改。
// 源文件: apps/web_apps/static/image_editor/composite.js
/** 图层合成策略：默认覆盖型，保留用户设定的不透明度，不因重复落笔累积加深。 */
export function compositeCoverageData(baseData, overlayData, opacity = 1) {
    const base = baseData?.data || baseData;
    const overlay = overlayData?.data || overlayData;
    if (!base || !overlay || base.length !== overlay.length) {
        throw new Error("图层合成数据尺寸不一致");
    }
    const factor = Math.max(0, Math.min(1, Number(opacity) || 0));
    for (let index = 0; index < base.length; index += 4) {
        const overlayAlpha = (overlay[index + 3] / 255) * factor;
        const existingAlpha = base[index + 3] / 255;
        if (overlayAlpha <= existingAlpha) continue;
        base[index] = overlay[index];
        base[index + 1] = overlay[index + 1];
        base[index + 2] = overlay[index + 2];
        base[index + 3] = Math.round(overlayAlpha * 255);
    }
    return baseData;
}

export function compositeLayerCanvas(targetContext, overlayCanvas, opacity = 1, allowOverlap = false) {
    const { width, height } = targetContext.canvas;
    if (!width || !height) return;
    targetContext.save();
    if (allowOverlap) {
        targetContext.globalCompositeOperation = "source-over";
        targetContext.globalAlpha = opacity;
        targetContext.drawImage(overlayCanvas, 0, 0);
        targetContext.restore();
        return;
    }
    const overlayContext = overlayCanvas.getContext("2d", { willReadFrequently: true });
    const baseData = targetContext.getImageData(0, 0, width, height);
    const overlayData = overlayContext.getImageData(0, 0, width, height);
    compositeCoverageData(baseData, overlayData, opacity);
    targetContext.putImageData(baseData, 0, 0);
    targetContext.restore();
}
