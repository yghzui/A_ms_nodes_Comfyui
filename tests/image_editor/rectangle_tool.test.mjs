import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';
import { rectangleFromPoints, moveRectangle, createRectangleTool } from '../../web/js/image_editor/rectangle_tool.js';

test('top-left drag snaps left, top and bottom to the image', () => {
    for (const scale of [0.25, 1, 4]) {
        assert.deepEqual(rectangleFromPoints({ x: 0, y: 0 }, { x: 180, y: 600 - 5 / scale },
            800, 600, 10 / scale), { x: 0, y: 0, width: 180, height: 600 });
    }
});

test('all four image corners support reverse-direction drawing and boundary clamping', () => {
    for (const anchor of [{ x: 0, y: 0 }, { x: 800, y: 0 }, { x: 0, y: 600 }, { x: 800, y: 600 }]) {
        const rect = rectangleFromPoints(anchor, { x: 200, y: 200 }, 800, 600, 10);
        assert.equal(rect.x, Math.min(anchor.x, 200));
        assert.equal(rect.y, Math.min(anchor.y, 200));
        assert.equal(rect.width, Math.abs(anchor.x - 200));
        assert.equal(rect.height, Math.abs(anchor.y - 200));
    }
    assert.deepEqual(rectangleFromPoints({ x: 0, y: 0 }, { x: 900, y: 900 }, 800, 600, 10),
        { x: 0, y: 0, width: 800, height: 600 });
});

test('Shift preserves a square even when its edge reaches the image boundary', () => {
    for (const anchor of [{ x: 0, y: 0 }, { x: 800, y: 600 }, { x: 350, y: 250 }]) {
        const rect = rectangleFromPoints(anchor, { x: 790, y: 590 }, 800, 600, 10, true);
        assert.equal(rect.width, rect.height);
        assert.ok(rect.x >= 0 && rect.y >= 0 && rect.x + rect.width <= 800 && rect.y + rect.height <= 600);
    }
    assert.deepEqual(rectangleFromPoints({ x: 800, y: 600 }, { x: 800, y: 400 }, 800, 600, 10, true),
        { x: 600, y: 400, width: 200, height: 200 });
});

test('moving snaps without resizing or leaving the image', () => {
    const rect = { x: 50, y: 60, width: 200, height: 600 };
    assert.deepEqual(moveRectangle(rect, -45, 0, 800, 600, 10), { ...rect, x: 0, y: 0 });
    assert.deepEqual(moveRectangle(rect, 900, -100, 800, 600, 10), { ...rect, x: 600, y: 0 });
});

class Element extends EventTarget {
    style = {};
    dataset = {};
    children = [];
    append(...items) { this.children.push(...items); items.forEach(item => { item.parentElement = this; }); }
    appendChild(item) { this.append(item); }
    insertBefore(item) { this.append(item); }
    getContext() {
        this.context ||= { calls: [], drawImage(...args) { this.calls.push(['draw', ...args]); },
            clearRect() {},
            fillRect(...args) { this.calls.push(['fill', this.fillStyle, this.globalAlpha, ...args]); } };
        return this.context;
    }
    toDataURL() { return 'data:image/png;base64,test'; }
    remove() { this.removed = true; }
}

function setup(onSave) {
    const previous = { document: globalThis.document, window: globalThis.window };
    globalThis.document = { createElement: () => new Element() };
    globalThis.window = new EventTarget();
    const workArea = new Element();
    const source = new Element();
    source.width = 800;
    source.height = 600;
    source.style = { cssText: 'opacity: 0.65', visibility: 'visible' };
    const parent = new Element();
    parent.append(source);
    const calls = [];
    let saves = 0;
    const state = { tool: 'rectangle', layer: 'paint', color: '#123456', maskColor: '#000000',
        opacity: 0.5, scale: 1, panX: 0, panY: 0 };
    const ctx = {
        canvas: source,
        save() { calls.push(this.layer, 'save'); }, restore() { calls.push('restore'); },
        fillRect(...args) { calls.push({ args, alpha: this.globalAlpha, color: this.fillStyle }); },
    };
    const tool = createRectangleTool({ workArea, state, getSize: () => ({ width: 800, height: 600 }),
        getPos: event => ({ x: event.x, y: event.y }), getContext: layer => { ctx.layer = layer; return ctx; },
        saveState: () => { saves++; onSave?.(); }, createBtn: () => new Element() });
    const send = (target, type, values) => {
        const event = new Event(type, { cancelable: true });
        Object.assign(event, values);
        target.dispatchEvent(event);
        return event;
    };
    return { tool, workArea, state, calls, source, parent, saves: () => saves,
        down: (x, y, ctrlKey = false) => send(workArea, 'mousedown', { x, y, button: 0, ctrlKey }),
        move: (x, y, ctrlKey = false) => send(window, 'mousemove', { x, y, ctrlKey }),
        up: (x, y) => send(window, 'mouseup', { x, y }),
        drag(from, to, shiftKey = false, ctrlKey = false) {
            send(workArea, 'mousedown', { ...from, button: 0, ctrlKey });
            send(window, 'mousemove', { ...to, shiftKey });
            send(window, 'mouseup', { ...to, shiftKey });
        },
        key: key => send(window, 'keydown', { key }),
        cleanup() {
            tool.dispose();
            for (const key of ['document', 'window']) {
                if (previous[key] === undefined) delete globalThis[key];
                else globalThis[key] = previous[key];
            }
        },
    };
}

test('preview does not alter pixels; Enter applies selected layer, color and opacity once', () => {
    const fixture = setup();
    try {
        fixture.drag({ x: 0, y: 0 }, { x: 180, y: 598 });
        assert.deepEqual(fixture.calls, []);
        assert.equal(fixture.saves(), 1);
        assert.equal(fixture.key('Enter').defaultPrevented, true);
        assert.deepEqual(fixture.calls, ['paint', 'save', { args: [0, 0, 180, 600], alpha: 0.5, color: '#808080' }, 'restore']);
        fixture.tool.apply();
        assert.equal(fixture.saves(), 2);
    } finally { fixture.cleanup(); }
});

test('cursor distinguishes drawing, hovering, held movement and diagonal resizing', () => {
    const fixture = setup();
    try {
        fixture.move(400, 400);
        assert.equal(fixture.workArea.style.cursor, 'crosshair');
        fixture.down(100, 100);
        assert.equal(fixture.workArea.style.cursor, 'crosshair');
        fixture.move(300, 300);
        fixture.up(300, 300);
        assert.equal(fixture.workArea.style.cursor, 'nwse-resize');
        fixture.move(200, 200);
        assert.equal(fixture.workArea.style.cursor, 'grab');
        fixture.down(200, 200);
        assert.equal(fixture.workArea.style.cursor, 'grabbing');
        fixture.move(250, 250);
        assert.equal(fixture.workArea.style.cursor, 'grabbing');
        fixture.up(250, 250);
        assert.equal(fixture.workArea.style.cursor, 'grab');
        fixture.move(150, 350);
        assert.equal(fixture.workArea.style.cursor, 'nesw-resize');
        fixture.down(150, 350);
        fixture.move(130, 370);
        assert.equal(fixture.workArea.style.cursor, 'nesw-resize');
        fixture.up(130, 370);
        fixture.move(200, 200, true);
        assert.equal(fixture.workArea.style.cursor, 'crosshair');
        fixture.move(-30, -30);
        assert.equal(fixture.workArea.style.cursor, 'default');
    } finally { fixture.cleanup(); }
});

test('multiple rectangles preview their final fill, remain editable and apply in one history step', () => {
    const fixture = setup();
    try {
        fixture.drag({ x: 0, y: 0 }, { x: 180, y: 600 });
        fixture.drag({ x: 800, y: 0 }, { x: 600, y: 600 });
        const preview = fixture.parent.children.at(-1);
        assert.equal(preview.style.cssText, fixture.source.style.cssText);
        assert.equal(fixture.source.style.visibility, 'hidden');
        assert.deepEqual(preview.context.calls.slice(-2), [
            ['fill', '#808080', 0.5, 0, 0, 180, 600],
            ['fill', '#808080', 0.5, 600, 0, 200, 600],
        ]);
        assert.deepEqual(fixture.calls, []);
        fixture.drag({ x: 180, y: 600 }, { x: 200, y: 600 });
        fixture.tool.apply();
        assert.deepEqual(fixture.calls, ['paint', 'save',
            { args: [0, 0, 200, 600], alpha: 0.5, color: '#808080' },
            { args: [600, 0, 200, 600], alpha: 0.5, color: '#808080' }, 'restore']);
        assert.equal(fixture.saves(), 4);
        assert.equal(fixture.source.style.visibility, 'visible');
    } finally { fixture.cleanup(); }
});

test('Ctrl draws overlapping rectangles and Delete removes only the selected rectangle', () => {
    const fixture = setup();
    try {
        fixture.drag({ x: 100, y: 100 }, { x: 300, y: 300 });
        fixture.drag({ x: 150, y: 150 }, { x: 350, y: 350 }, false, true);
        assert.equal(fixture.key('Delete').defaultPrevented, true);
        fixture.tool.apply();
        assert.deepEqual(fixture.calls[2].args, [100, 100, 200, 200]);
        assert.equal(fixture.calls.length, 4);
        assert.equal(fixture.saves(), 4);
    } finally { fixture.cleanup(); }
});

test('preview respects layer visibility and cancellation restores that visibility', () => {
    const fixture = setup();
    try {
        fixture.drag({ x: 0, y: 0 }, { x: 100, y: 100 });
        fixture.tool.setLayerVisibility('paint', false);
        fixture.tool.cancel();
        assert.equal(fixture.source.style.visibility, 'hidden');
        fixture.tool.setLayerVisibility('paint', true);
        assert.equal(fixture.source.style.visibility, 'visible');
    } finally { fixture.cleanup(); }
});

test('interactive snapping requires a distance within three screen pixels at every zoom', () => {
    for (const scale of [0.25, 1, 4]) {
        const fixture = setup();
        try {
            fixture.state.scale = scale;
            fixture.drag({ x: 0, y: 0 }, { x: 180, y: 600 - 5 / scale });
            fixture.tool.apply();
            assert.equal(fixture.calls[2].args[3], Math.round(600 - 5 / scale));
            fixture.drag({ x: 0, y: 0 }, { x: 180, y: 600 - 2 / scale });
            fixture.tool.apply();
            assert.equal(fixture.calls[6].args[3], 600);
        } finally { fixture.cleanup(); }
    }
});

test('rectangle color defaults to gray independently of brush color and supports customization', () => {
    const fixture = setup();
    try {
        const colorInput = fixture.tool.panel.children[1].children[0];
        assert.equal(colorInput.value, '#808080');
        colorInput.oninput({ target: { value: '#4488cc' } });
        fixture.drag({ x: 0, y: 0 }, { x: 100, y: 100 });
        const preview = fixture.parent.children.at(-1);
        assert.equal(preview.context.calls.at(-1)[1], '#4488cc');
        fixture.tool.apply();
        assert.equal(fixture.calls[2].color, '#4488cc');
        assert.equal(fixture.state.color, '#123456');
    } finally { fixture.cleanup(); }
});

test('corners resize a pending rectangle; interior drags move it; Escape cancels', () => {
    const fixture = setup();
    try {
        fixture.drag({ x: 100, y: 100 }, { x: 300, y: 300 });
        fixture.drag({ x: 300, y: 300 }, { x: 400, y: 350 });
        fixture.drag({ x: 200, y: 200 }, { x: 102, y: 102 });
        fixture.tool.apply();
        assert.deepEqual(fixture.calls[2].args, [0, 0, 300, 250]);
        fixture.drag({ x: 0, y: 0 }, { x: 200, y: 200 });
        assert.equal(fixture.key('Escape').defaultPrevented, true);
        fixture.tool.apply();
        assert.equal(fixture.saves(), 6);
    } finally { fixture.cleanup(); }
});

test('mask rectangles default to neutral gray and disposal removes global interaction listeners', () => {
    const fixture = setup();
    try {
        fixture.state.layer = 'mask';
        fixture.drag({ x: 800, y: 600 }, { x: 600, y: 400 }, true);
        fixture.tool.apply();
        assert.equal(fixture.calls[0], 'mask');
        assert.equal(fixture.calls[2].color, '#808080');
        fixture.drag({ x: 0, y: 0 }, { x: 100, y: 100 });
        fixture.tool.dispose();
        assert.equal(fixture.key('Enter').defaultPrevented, false);
        assert.equal(fixture.saves(), 3);
    } finally { fixture.cleanup(); }
});

function setupEditorHistory() {
    let saveEditor;
    const fixture = setup(() => saveEditor());
    fixture.state.history = [];
    fixture.state.historyIndex = -1;
    Object.assign(fixture.state, { cropPreset: 'free', cropExpandColor: '#808080', cropRect: null,
        fillMaskedImageWithGray: false });
    const images = [];
    const undoBtn = { style: {} }, redoBtn = { style: {} };
    const source = readFileSync(new URL('../../web/js/image_editor/image_editor.js', import.meta.url), 'utf8');
    const context = vm.createContext({
        state: fixture.state, rectangleTool: fixture.tool, canvasW: 800, canvasH: 600,
        baseImg: { src: 'base' }, paintCanvas: { toDataURL: () => 'paint' }, maskCanvas: { toDataURL: () => 'mask' },
        pCtx: { clearRect() {}, drawImage() {} }, mCtx: { clearRect() {}, drawImage() {} },
        undoBtn, redoBtn, opSlider: {}, opVal: {}, cropExpandColorInput: {}, grayFillInput: {},
        cropWheelSaveTimer: null, cropScaledByWheel: false, clearTimeout,
        window: globalThis.window, workArea: fixture.workArea, document: globalThis.document,
        lastMouseE: null, cropAction: null, cropSnapEdges: null, cropDragStart: null,
        isPanning: false, isDrawing: false,
        getPos: event => ({ x: event.x, y: event.y }), updateCursorPos() {},
        cropPresetBtns: {}, fitImageCropBtn: {}, applyCropBtn: {}, resetCropBtn: {},
        tempCtx: { clearRect() {} }, tempCanvas: { style: {} },
        rasterizeBaseImage: () => fixture.source, resetView() {},
        cloneRect: rect => rect ? { ...rect } : null,
        setCropPreset: preset => { fixture.state.cropPreset = preset; },
        Image: class { constructor() { images.push(this); } },
        setCanvasSize(width, height) { context.canvasW = width; context.canvasH = height; },
        setBaseImagePreview() {}, updateInfoLabel() {}, drawCropOverlay() {}, updateTransform() {},
        setLayer: layer => { fixture.state.layer = layer; },
        setTool: tool => { fixture.state.tool = tool; },
    });
    vm.runInContext(source.slice(source.indexOf('    let historyRestoreId ='), source.indexOf('    clearBtn.onclick =')), context);
    saveEditor = () => vm.runInContext('saveState()', context);
    saveEditor();
    return { ...fixture, undo: () => undoBtn.onclick(), redo: () => redoBtn.onclick(),
        context, run: code => vm.runInContext(code, context),
        enableCrop() {
            fixture.state.tool = 'crop';
            vm.runInContext(source.slice(source.indexOf('    const clamp ='), source.indexOf('    const updateCropMeta =')), context);
            vm.runInContext(source.slice(source.indexOf('    const getRectIntersection ='), source.indexOf('    const rasterizeBaseImage =')), context);
            vm.runInContext(source.slice(source.indexOf('    const setCropPreset ='), source.indexOf('    const rectangleTool =')), context);
            vm.runInContext(source.slice(source.indexOf('    workArea.addEventListener("mousedown"'), source.indexOf('    workArea.addEventListener("wheel"')), context);
            vm.runInContext('state.cropRect = createDefaultCropRect(); saveState()', context);
        },
        flush: () => images.splice(0).forEach(image => image.onload()) };
}

test('the real editor history restores creation, movement, resizing, deletion, cancellation and application', () => {
    const fixture = setupEditorHistory();
    const snapshot = () => fixture.tool.snapshot();
    try {
        fixture.drag({ x: 100, y: 100 }, { x: 300, y: 300 });
        fixture.drag({ x: 500, y: 100 }, { x: 700, y: 300 });
        assert.equal(fixture.state.historyIndex, 2);
        fixture.undo(); fixture.flush();
        assert.equal(snapshot().rectangles.length, 1);
        fixture.redo(); fixture.flush();
        assert.equal(snapshot().rectangles.length, 2);

        fixture.down(600, 200);
        fixture.move(610, 210); fixture.move(620, 220); fixture.move(630, 230);
        assert.equal(fixture.state.historyIndex, 2);
        fixture.up(630, 230);
        assert.equal(fixture.state.historyIndex, 3);
        assert.equal(snapshot().rectangles[1].x, 530);
        fixture.undo(); fixture.flush();
        assert.equal(snapshot().rectangles[1].x, 500);
        fixture.redo(); fixture.flush();
        assert.equal(snapshot().rectangles[1].x, 530);

        fixture.drag({ x: 730, y: 330 }, { x: 750, y: 350 });
        assert.equal(snapshot().rectangles[1].width, 220);
        fixture.undo(); fixture.flush();
        assert.equal(snapshot().rectangles[1].width, 200);
        fixture.redo(); fixture.flush();
        assert.equal(snapshot().rectangles[1].width, 220);

        fixture.key('Delete');
        assert.equal(snapshot().rectangles.length, 1);
        fixture.undo(); fixture.flush();
        assert.equal(snapshot().rectangles.length, 2);
        fixture.key('Escape');
        assert.equal(snapshot().rectangles.length, 0);
        fixture.undo(); fixture.flush();
        assert.equal(snapshot().rectangles.length, 2);
        fixture.tool.apply();
        assert.equal(snapshot().rectangles.length, 0);
        fixture.undo(); fixture.flush();
        assert.equal(snapshot().rectangles.length, 2);
        fixture.redo(); fixture.flush();
        assert.equal(snapshot().rectangles.length, 0);
        assert.equal(fixture.calls.filter(item => item === 'save').length, 1);
    } finally { fixture.cleanup(); }
});

test('rapid history changes discard stale image callbacks and new edits clear the redo branch', () => {
    const fixture = setupEditorHistory();
    try {
        fixture.drag({ x: 100, y: 100 }, { x: 300, y: 300 });
        fixture.drag({ x: 500, y: 100 }, { x: 700, y: 300 });
        fixture.undo(); fixture.undo(); fixture.flush();
        assert.equal(fixture.tool.snapshot().rectangles.length, 0);
        fixture.redo(); fixture.flush();
        assert.equal(fixture.tool.snapshot().rectangles.length, 1);
        fixture.down(200, 200); fixture.up(200, 200);
        assert.equal(fixture.state.historyIndex, 1);
        fixture.drag({ x: 500, y: 400 }, { x: 700, y: 550 });
        fixture.redo(); fixture.flush();
        assert.equal(fixture.tool.snapshot().rectangles[1].y, 400);
        assert.equal(fixture.state.history.length, 3);
    } finally { fixture.cleanup(); }
});

test('crop creation, movement and resizing each record one reversible step', () => {
    const fixture = setupEditorHistory();
    try {
        fixture.enableCrop();
        const baseline = JSON.stringify(fixture.state.cropRect);
        fixture.down(10, 10);
        fixture.move(200, 200); fixture.move(250, 250);
        assert.equal(fixture.state.historyIndex, 1);
        fixture.up(250, 250);
        assert.equal(fixture.state.historyIndex, 2);
        const created = JSON.stringify(fixture.state.cropRect);
        fixture.undo(); fixture.flush();
        assert.equal(JSON.stringify(fixture.state.cropRect), baseline);
        fixture.redo(); fixture.flush();
        assert.equal(JSON.stringify(fixture.state.cropRect), created);
        fixture.drag({ x: 100, y: 100 }, { x: 150, y: 150 });
        const moved = JSON.stringify(fixture.state.cropRect);
        fixture.undo(); fixture.flush();
        assert.equal(JSON.stringify(fixture.state.cropRect), created);
        fixture.redo(); fixture.flush();
        assert.equal(JSON.stringify(fixture.state.cropRect), moved);
        const rect = fixture.state.cropRect;
        fixture.drag({ x: rect.x + rect.width, y: rect.y + rect.height }, { x: 400, y: 350 });
        const resized = JSON.stringify(fixture.state.cropRect);
        fixture.undo(); fixture.flush();
        assert.equal(JSON.stringify(fixture.state.cropRect), moved);
        fixture.redo(); fixture.flush();
        assert.equal(JSON.stringify(fixture.state.cropRect), resized);
    } finally { fixture.cleanup(); }
});

test('crop reset, fit, preset changes and applied dimensions are restored by editor history', () => {
    const fixture = setupEditorHistory();
    try {
        fixture.enableCrop();
        fixture.drag({ x: 10, y: 10 }, { x: 250, y: 250 });
        const custom = JSON.stringify(fixture.state.cropRect);
        fixture.run('resetCropSelection()');
        fixture.undo(); fixture.flush();
        assert.equal(JSON.stringify(fixture.state.cropRect), custom);
        fixture.run('fitImageCropBtn.onclick()');
        assert.equal(fixture.state.cropRect.width, 800);
        fixture.undo(); fixture.flush();
        assert.equal(JSON.stringify(fixture.state.cropRect), custom);
        fixture.run("setCropPreset('16:9'); saveState()");
        assert.equal(fixture.state.cropRect.width / fixture.state.cropRect.height, 16 / 9);
        fixture.undo(); fixture.flush();
        assert.equal(fixture.state.cropPreset, 'free');
        assert.equal(JSON.stringify(fixture.state.cropRect), custom);
        fixture.run('applyCropBtn.onclick()');
        assert.equal(fixture.state.cropRect, null);
        assert.notEqual(fixture.context.canvasW, 800);
        fixture.undo(); fixture.flush();
        assert.equal(fixture.context.canvasW, 800);
        assert.equal(JSON.stringify(fixture.state.cropRect), custom);
        fixture.redo(); fixture.flush();
        assert.equal(fixture.state.cropRect, null);
        assert.notEqual(fixture.context.canvasW, 800);
    } finally { fixture.cleanup(); }
});
