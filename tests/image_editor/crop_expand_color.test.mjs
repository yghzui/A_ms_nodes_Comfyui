import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';
import test from 'node:test';

const source = readFileSync(new URL('../../web/js/image_editor/image_editor.js', import.meta.url), 'utf8');
const section = (start, end) => source.slice(source.indexOf(start), source.indexOf(end, source.indexOf(start)));

function editorSettings(savedConfig = {}) {
    const values = new Map();
    const context = vm.createContext({
        savedConfig,
        console,
        localStorage: { setItem: (key, value) => values.set(key, value) },
        document: { createElement: () => ({ style: {}, appendChild() {} }) },
    });
    vm.runInContext(section('    let state = {', '    let origW ='), context);
    vm.runInContext(section('    const cropExpandColorWrap =', '    const cropActionWrap ='), context);
    return { context, values };
}

test('new and existing editor settings default to neutral gray', () => {
    for (const config of [{}, { color: '#ff0000', cropPreset: '1:1' }]) {
        const { context } = editorSettings(config);
        assert.equal(vm.runInContext('cropExpandColorInput.value', context), '#808080');
        assert.equal(vm.runInContext('cropExpandColorInput.type', context), 'color');
    }
});

test('the independent expansion color persists and restores', () => {
    const { context, values } = editorSettings({ color: '#ff0000' });
    vm.runInContext("cropExpandColorInput.oninput({target: {value: '#123456'}})", context);
    const saved = JSON.parse(values.get('myNodesImageEditorConfig'));
    assert.equal(saved.cropExpandColor, '#123456');
    assert.equal(saved.color, '#ff0000');
    const restored = editorSettings(saved);
    assert.equal(vm.runInContext('cropExpandColorInput.value', restored.context), '#123456');
});

function crop(rect, color) {
    const canvases = [];
    const context = vm.createContext({
        state: { cropExpandColor: color, maskColor: 'rgba(0, 0, 0, 1)' },
        canvasW: 100, canvasH: 80,
        roundRect: value => value,
        clampRectToCanvas: value => value,
        rasterizeBaseImage: () => 'base', paintCanvas: 'paint', maskCanvas: 'mask',
        document: {
            createElement: () => {
                const calls = [];
                const ctx = {
                    fillRect(...args) { calls.push(['fill', this.fillStyle, ...args]); },
                    clearRect(...args) { calls.push(['clear', ...args]); },
                    drawImage(...args) { calls.push(['draw', ...args]); },
                };
                const canvas = { calls, getContext: () => ctx, toDataURL: () => 'data:image/png;base64,test' };
                canvases.push(canvas);
                return canvas;
            },
        },
        setCanvasSize() {},
        pCtx: { clearRect() {}, drawImage() {} },
        mCtx: { clearRect() {}, drawImage() {} },
        tempCtx: { clearRect() {} },
        setBaseImagePreview() {}, updateInfoLabel() {}, resetView() {}, drawCropOverlay() {}, saveState() {},
    });
    vm.runInContext(section('    const getRectIntersection =', '    const rasterizeBaseImage ='), context);
    vm.runInContext(section('    const cropCurrentView =', '    applyCropBtn.onclick ='), context);
    context.rect = rect;
    vm.runInContext('cropCurrentView(rect)', context);
    return canvases.map(canvas => ({ width: canvas.width, height: canvas.height, calls: canvas.calls }));
}

test('expansion fills the base outside the image and preserves paint and mask behavior', () => {
    for (const color of ['#808080', '#123456', '#000000', '#ffffff']) {
        const [base, paint, mask] = crop({ x: -10, y: -20, width: 130, height: 120 }, color);
        assert.equal(base.width, 130);
        assert.equal(base.height, 120);
        assert.deepEqual(base.calls, [
            ['fill', color, 0, 0, 130, 120],
            ['clear', 10, 20, 100, 80],
            ['draw', 'base', 10, 20],
        ]);
        assert.deepEqual(paint.calls, [['draw', 'paint', 10, 20]]);
        assert.deepEqual(mask.calls, [
            ['fill', 'rgba(0, 0, 0, 1)', 0, 0, 130, 120],
            ['clear', 10, 20, 100, 80],
            ['draw', 'mask', 10, 20],
        ]);
    }
});

test('an interior crop clears all fill before drawing the original pixels', () => {
    const [base] = crop({ x: 10, y: 15, width: 40, height: 30 }, '#808080');
    assert.deepEqual(base.calls, [
        ['fill', '#808080', 0, 0, 40, 30],
        ['clear', 0, 0, 40, 30],
        ['draw', 'base', -10, -15],
    ]);
});
