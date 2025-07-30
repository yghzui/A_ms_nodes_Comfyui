export function getResolver(timeout = 5000) {
    const resolver = {};
    resolver.id = generateId(8);
    resolver.completed = false;
    resolver.resolved = false;
    resolver.rejected = false;
    resolver.promise = new Promise((resolve, reject) => {
        resolver.reject = (e) => {
            resolver.completed = true;
            resolver.rejected = true;
            reject(e);
        };
        resolver.resolve = (data) => {
            resolver.completed = true;
            resolver.resolved = true;
            resolve(data);
        };
    });
    resolver.timeout = setTimeout(() => {
        if (!resolver.completed) {
            resolver.reject();
        }
    }, timeout);
    return resolver;
}

const DEBOUNCE_FN_TO_PROMISE = new WeakMap();
export function debounce(fn, ms = 64) {
    if (!DEBOUNCE_FN_TO_PROMISE.get(fn)) {
        DEBOUNCE_FN_TO_PROMISE.set(fn, wait(ms).then(() => {
            DEBOUNCE_FN_TO_PROMISE.delete(fn);
            fn();
        }));
    }
    return DEBOUNCE_FN_TO_PROMISE.get(fn);
}

export function wait(ms = 16) {
    if (ms === 16) {
        return new Promise((resolve) => {
            requestAnimationFrame(() => {
                resolve();
            });
        });
    }
    return new Promise((resolve) => {
        setTimeout(() => {
            resolve();
        }, ms);
    });
}

function dec2hex(dec) {
    return dec.toString(16).padStart(2, "0");
}

export function generateId(length) {
    const arr = new Uint8Array(length / 2);
    crypto.getRandomValues(arr);
    return Array.from(arr, dec2hex).join("");
}

export function getObjectValue(obj, objKey, def) {
    if (!obj || !objKey)
        return def;
    const keys = objKey.split(".");
    const key = keys.shift();
    const found = obj[key];
    if (keys.length) {
        return getObjectValue(found, keys.join("."), def);
    }
    return found;
}

export function setObjectValue(obj, objKey, value, createMissingObjects = true) {
    if (!obj || !objKey)
        return obj;
    const keys = objKey.split(".");
    const key = keys.shift();
    if (obj[key] === undefined) {
        if (!createMissingObjects) {
            return;
        }
        obj[key] = {};
    }
    if (!keys.length) {
        obj[key] = value;
    }
    else {
        if (typeof obj[key] != "object") {
            obj[key] = {};
        }
        setObjectValue(obj[key], keys.join("."), value, createMissingObjects);
    }
    return obj;
}

export function moveArrayItem(arr, itemOrFrom, to) {
    const from = typeof itemOrFrom === "number" ? itemOrFrom : arr.indexOf(itemOrFrom);
    if (from !== -1) {
        arr.splice(to, 0, arr.splice(from, 1)[0]);
    }
}

export function removeArrayItem(arr, itemOrIndex) {
    const index = typeof itemOrIndex === "number" ? itemOrIndex : arr.indexOf(itemOrIndex);
    if (index !== -1) {
        arr.splice(index, 1);
    }
}

export function injectCss(href) {
    if (document.querySelector(`link[href^="${href}"]`)) {
        return;
    }
    const link = document.createElement("link");
    link.rel = "stylesheet";
    link.href = href;
    document.head.appendChild(link);
}

export function defineProperty(instance, property, desc) {
    try {
        Object.defineProperty(instance, property, desc);
    }
    catch (e) {
        console.warn(`Failed to define property ${property}:`, e);
    }
}

export function areDataViewsEqual(a, b) {
    if (a.byteLength !== b.byteLength)
        return false;
    const viewA = new Uint8Array(a.buffer, a.byteOffset, a.byteLength);
    const viewB = new Uint8Array(b.buffer, b.byteOffset, b.byteLength);
    for (let i = 0; i < viewA.length; i++) {
        if (viewA[i] !== viewB[i])
            return false;
    }
    return true;
}

function looksLikeBase64(source) {
    return typeof source === "string" && /^[A-Za-z0-9+/]*={0,2}$/.test(source);
}

export function areArrayBuffersEqual(a, b) {
    if (a.byteLength !== b.byteLength)
        return false;
    const viewA = new Uint8Array(a);
    const viewB = new Uint8Array(b);
    for (let i = 0; i < viewA.length; i++) {
        if (viewA[i] !== viewB[i])
            return false;
    }
    return true;
}

export function getCanvasImageData(image) {
    const canvas = document.createElement("canvas");
    canvas.width = image.width;
    canvas.height = image.height;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(image, 0, 0);
    return ctx.getImageData(0, 0, canvas.width, canvas.height);
}

export async function convertToBase64(source) {
    if (typeof source === "string") {
        if (looksLikeBase64(source)) {
            return source;
        }
        if (source.startsWith("data:")) {
            return source;
        }
    }
    const arrayBuffer = await convertToArrayBuffer(source);
    const uint8Array = new Uint8Array(arrayBuffer);
    let binary = "";
    for (let i = 0; i < uint8Array.length; i++) {
        binary += String.fromCharCode(uint8Array[i]);
    }
    return btoa(binary);
}

export async function convertToArrayBuffer(source) {
    if (source instanceof ArrayBuffer) {
        return source;
    }
    if (source instanceof Uint8Array) {
        return source.buffer.slice(source.byteOffset, source.byteOffset + source.byteLength);
    }
    if (source instanceof DataView) {
        return source.buffer.slice(source.byteOffset, source.byteOffset + source.byteLength);
    }
    if (typeof source === "string") {
        if (looksLikeBase64(source)) {
            const binary = atob(source);
            const array = new Uint8Array(binary.length);
            for (let i = 0; i < binary.length; i++) {
                array[i] = binary.charCodeAt(i);
            }
            return array.buffer;
        }
        if (source.startsWith("data:")) {
            const response = await fetch(source);
            return await response.arrayBuffer();
        }
    }
    if (source instanceof Blob) {
        return await source.arrayBuffer();
    }
    if (source instanceof File) {
        return await source.arrayBuffer();
    }
    if (source instanceof ImageData) {
        return source.data.buffer;
    }
    if (source instanceof HTMLCanvasElement) {
        const ctx = source.getContext("2d");
        const imageData = ctx.getImageData(0, 0, source.width, source.height);
        return imageData.data.buffer;
    }
    if (source instanceof HTMLImageElement) {
        const imageData = getCanvasImageData(source);
        return imageData.data.buffer;
    }
    throw new Error(`Cannot convert source to ArrayBuffer: ${typeof source}`);
}

export async function loadImage(source) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.crossOrigin = "anonymous";
        img.onload = () => resolve(img);
        img.onerror = reject;
        if (typeof source === "string") {
            img.src = source;
        }
        else if (source instanceof Blob) {
            img.src = URL.createObjectURL(source);
        }
        else if (source instanceof ArrayBuffer) {
            const blob = new Blob([source]);
            img.src = URL.createObjectURL(blob);
        }
        else {
            reject(new Error(`Cannot load image from source: ${typeof source}`));
        }
    });
}

function getMimeTypeFromArrayBuffer(buffer) {
    const uint8Array = new Uint8Array(buffer);
    if (uint8Array.length < 4) {
        return null;
    }
    const signature = Array.from(uint8Array.slice(0, 4))
        .map(byte => byte.toString(16).padStart(2, "0"))
        .join("")
        .toUpperCase();
    switch (signature) {
        case "89504E47":
            return "image/png";
        case "FFD8FFDB":
        case "FFD8FFE0":
        case "FFD8FFE1":
        case "FFD8FFE2":
        case "FFD8FFE3":
        case "FFD8FFE8":
            return "image/jpeg";
        case "47494638":
            return "image/gif";
        case "52494646":
            if (uint8Array.length >= 12) {
                const webpSignature = Array.from(uint8Array.slice(8, 12))
                    .map(byte => String.fromCharCode(byte))
                    .join("");
                if (webpSignature === "WEBP") {
                    return "image/webp";
                }
            }
            return "image/unknown";
        default:
            return null;
    }
}

export class Broadcaster extends EventTarget {
    constructor(channelName) {
        super();
        this.channelName = channelName;
        this.id = generateId(8);
        this.listeners = new Map();
        this.messageId = 0;
    }
    getId() {
        return this.id;
    }
    async broadcastAndWait(action, payload, options) {
        const messageId = ++this.messageId;
        const message = {
            id: messageId,
            action,
            payload,
            from: this.id,
            timestamp: Date.now(),
        };
        const promise = new Promise((resolve, reject) => {
            const timeout = setTimeout(() => {
                this.listeners.delete(messageId);
                reject(new Error("Broadcast timeout"));
            }, options?.timeout || 5000);
            this.listeners.set(messageId, { resolve, reject, timeout });
        });
        this.broadcast(message);
        return promise;
    }
    broadcast(action, payload) {
        const message = {
            action,
            payload,
            from: this.id,
            timestamp: Date.now(),
        };
        this.broadcast(message);
    }
    reply(replyId, action, payload) {
        const message = {
            id: replyId,
            action,
            payload,
            from: this.id,
            timestamp: Date.now(),
        };
        this.broadcast(message);
    }
    openWindowAndWaitForMessage(rgthreePath, windowName) {
        const url = `${rgthreePath}?broadcast=${this.channelName}&id=${this.id}`;
        const window = open(url, windowName, "width=800,height=600");
        return window;
    }
    onMessage(e) {
        const message = e.data;
        if (message.id && this.listeners.has(message.id)) {
            const listener = this.listeners.get(message.id);
            clearTimeout(listener.timeout);
            this.listeners.delete(message.id);
            listener.resolve(message);
        }
        else {
            this.dispatchEvent(new CustomEvent("message", { detail: message }));
        }
    }
    addMessageListener(callback, options) {
        this.addEventListener("message", callback, options);
    }
}

export function broadcastOnChannel(channel, action, payload) {
    const message = {
        action,
        payload,
        timestamp: Date.now(),
    };
    if (window.parent && window.parent !== window) {
        window.parent.postMessage(message, "*");
    }
    if (window.opener) {
        window.opener.postMessage(message, "*");
    }
} 