// 本文件由主项目 tools/sync_image_editor_core.py 自动生成，禁止手改。
// 源文件: apps/web_apps/static/image_editor/enclosed_fill.js
/**
 * 仅以最后一笔的像素作为边界，找出不与画布边缘连通的区域。
 * 返回值为与画布同尺寸的二值数组，1 表示该笔独立围出的可填充内部。
 */
export function findStrokeEnclosedPixels(strokeData, width, height, barrierAlpha = 1) {
    const total = Math.max(0, Number(width) * Number(height));
    if (!strokeData || strokeData.length !== total * 4 || !total) return new Uint8Array(total);
    const reachable = new Uint8Array(total);
    const queue = new Int32Array(total);
    let head = 0;
    let tail = 0;
    const isBarrier = index => strokeData[index * 4 + 3] >= barrierAlpha;
    const enqueue = index => {
        if (!reachable[index] && !isBarrier(index)) {
            reachable[index] = 1;
            queue[tail++] = index;
        }
    };
    for (let x = 0; x < width; x++) {
        enqueue(x);
        enqueue((height - 1) * width + x);
    }
    for (let y = 1; y < height - 1; y++) {
        enqueue(y * width);
        enqueue(y * width + width - 1);
    }
    while (head < tail) {
        const index = queue[head++];
        const x = index % width;
        const y = Math.floor(index / width);
        if (x > 0) enqueue(index - 1);
        if (x + 1 < width) enqueue(index + 1);
        if (y > 0) enqueue(index - width);
        if (y + 1 < height) enqueue(index + width);
    }
    const enclosed = new Uint8Array(total);
    for (let index = 0; index < total; index++) {
        if (!isBarrier(index) && !reachable[index]) enclosed[index] = 1;
    }
    return enclosed;
}

/** 将闭环内部向其自身边界扩展一像素，消除抗锯齿边缘与填充之间的细缝。 */
export function includeEnclosureBoundary(enclosed, strokeData, width, height) {
    const result = new Uint8Array(enclosed);
    const isBarrier = index => strokeData[index * 4 + 3] > 0;
    for (let index = 0; index < enclosed.length; index++) {
        if (!enclosed[index]) continue;
        const x = index % width;
        const y = Math.floor(index / width);
        for (const neighbor of [index - 1, index + 1, index - width, index + width]) {
            if ((neighbor === index - 1 && x === 0) || (neighbor === index + 1 && x + 1 === width)
                || (neighbor === index - width && y === 0) || (neighbor === index + width && y + 1 === height)) continue;
            if (isBarrier(neighbor)) result[neighbor] = 1;
        }
    }
    return result;
}

export function fillEnclosedLayer(data, width, height, rgb, opacity) {
    const fillAlpha = Math.round(opacity * 255);
    if (fillAlpha <= 0) return 0;

    // Find weak pixels surrounded by the solid part of the stroke, even if those pixels already have some alpha.
    const enclosed = findStrokeEnclosedPixels(data, width, height, Math.max(1, fillAlpha - 1));
    // Keep filling transparent holes outlined with a brush fainter than the current fill opacity.
    const transparentHoles = includeEnclosureBoundary(findStrokeEnclosedPixels(data, width, height), data, width, height);
    const [red, green, blue] = rgb;
    let count = 0;
    for (let index = 0; index < enclosed.length; index++) {
        if (!enclosed[index] && !transparentHoles[index]) continue;
        const pixel = index * 4;
        if (data[pixel + 3] >= fillAlpha) continue;
        data[pixel] = red;
        data[pixel + 1] = green;
        data[pixel + 2] = blue;
        data[pixel + 3] = fillAlpha;
        count++;
    }
    return count;
}
