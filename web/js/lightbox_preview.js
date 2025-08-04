// 通用全屏预览组件
console.log("Loading lightbox preview component");

import { api } from "../../../scripts/api.js";

/**
 * 创建并显示一个功能丰富的灯箱用于图片预览。
 * 支持缩放、平移、重置和图片切换。
 * @param {string[]} urls - 要显示的图片URL数组。
 * @param {number} currentIndex - 当前要显示的图片在数组中的索引。
 * @param {string} type - 媒体类型，'image' 或 'video'
 */
export function showLightbox(urls, currentIndex, type = 'image') {
    // 移除已存在的灯箱
    const existingLightbox = document.getElementById("my-nodes-lightbox");
    if (existingLightbox) {
        existingLightbox.remove();
    }

    const lightbox = document.createElement("div");
    lightbox.id = "my-nodes-lightbox";
    Object.assign(lightbox.style, {
        position: "fixed", top: "0", left: "0", width: "100%", height: "100%",
        backgroundColor: "rgba(0, 0, 0, 0.85)", display: "flex",
        flexDirection: "column", justifyContent: "center", alignItems: "center", 
        zIndex: "1001", userSelect: "none"
    });

    const container = document.createElement("div");
    container.className = "lightbox-media-container";
    Object.assign(container.style, {
        display: "flex", flexDirection: "column",
        alignItems: "center", gap: "10px"
    });

    let mediaElement;
    let sizeInfo = document.createElement("div");

    if (type === 'video') {
        // 创建视频元素
        mediaElement = document.createElement("video");
        mediaElement.controls = true;
        mediaElement.muted = true;
        mediaElement.loop = true;
        mediaElement.autoplay = true;
    } else {
        // 创建图片元素
        mediaElement = document.createElement("img");
    }

    // --- 状态变量 ---
    let scale = 1;
    let panX = 0;
    let panY = 0;
    let isPanning = false;
    let panStart = { x: 0, y: 0 };

    // --- 核心功能函数 ---
    const updateTransform = () => {
        if (type === 'image') {
            mediaElement.style.transform = `translate(${panX}px, ${panY}px) scale(${scale})`;
        }
    };

    const updateSizeInfo = () => {
        if (type === 'image' && mediaElement.naturalWidth) {
            sizeInfo.textContent = `${mediaElement.naturalWidth} × ${mediaElement.naturalHeight} | ${Math.round(scale * 100)}%`;
        } else if (type === 'video' && mediaElement.videoWidth) {
            sizeInfo.textContent = `${mediaElement.videoWidth} × ${mediaElement.videoHeight}`;
        }
    };
    
    const resetView = () => {
        if (type === 'image') {
            scale = 1; panX = 0; panY = 0;
            mediaElement.style.transition = 'transform 0.2s ease-out';
            updateTransform();
            updateSizeInfo();
            setTimeout(() => mediaElement.style.transition = 'none', 200);
        }
    };

    const loadMedia = (index) => {
        if (index < 0 || index >= urls.length) return;
        currentIndex = index;
        
        if (type === 'video') {
            mediaElement.src = urls[currentIndex];
            sizeInfo.textContent = "加载中...";
        } else {
            mediaElement.src = urls[currentIndex];
            sizeInfo.textContent = "加载中...";
            resetView();
        }
    };

    const closeLightbox = () => {
        window.removeEventListener("mousemove", handleMouseMove);
        window.removeEventListener("mouseup", handleMouseUp);
        window.removeEventListener("keydown", handleKeyDown);
        lightbox.remove();
    };

    // --- 事件处理器 ---
    const handleMouseMove = (e) => {
        if (!isPanning || type !== 'image') return;
        e.preventDefault();
        panX = e.clientX - panStart.x;
        panY = e.clientY - panStart.y;
        updateTransform();
    };
    
    const handleMouseUp = (e) => {
        if (isPanning && type === 'image') {
            isPanning = false;
            mediaElement.style.cursor = "grab";
        }
    };
    
    const handleKeyDown = (e) => {
        if (e.key === "Escape") closeLightbox();
        if (e.key === "ArrowLeft" && urls.length > 1) loadMedia((currentIndex - 1 + urls.length) % urls.length);
        if (e.key === "ArrowRight" && urls.length > 1) loadMedia((currentIndex + 1) % urls.length);
    };

    // --- 元素设置和事件绑定 ---
    Object.assign(mediaElement.style, {
        maxWidth: "95vw", maxHeight: "90vh", objectFit: "contain",
        cursor: type === 'image' ? "grab" : "default", transition: "none",
    });
    
    if (type === 'image') {
        mediaElement.onload = () => {
            resetView();
        };
        
        mediaElement.addEventListener("mousedown", (e) => {
            if (e.button !== 0) return;
            e.preventDefault();
            isPanning = true;
            panStart.x = e.clientX - panX;
            panStart.y = e.clientY - panY;
            mediaElement.style.cursor = "grabbing";
        });
        
        mediaElement.addEventListener("wheel", (e) => {
            e.preventDefault();
            const rect = mediaElement.getBoundingClientRect();
            const zoomSpeed = 0.1;
            const oldScale = scale;

            // Calculate new scale
            scale *= (1 - Math.sign(e.deltaY) * zoomSpeed);
            scale = Math.max(0.1, Math.min(scale, 15));

            const scaleRatio = scale / oldScale;

            // Get scaling origin (center of image)
            const originX = rect.left + rect.width / 2;
            const originY = rect.top + rect.height / 2;
            
            // Get mouse position
            const mouseX = e.clientX;
            const mouseY = e.clientY;

            // Calculate the pan adjustment needed to keep mouse position constant
            const panXDelta = (mouseX - originX) * (1 - scaleRatio);
            const panYDelta = (mouseY - originY) * (1 - scaleRatio);

            // Apply the adjustment to the current pan values
            panX += panXDelta;
            panY += panYDelta;
            
            updateTransform();
            updateSizeInfo();
        });
    } else {
        // 视频加载完成事件
        mediaElement.onloadeddata = () => {
            updateSizeInfo();
        };
    }

    lightbox.addEventListener("dblclick", (e) => {
        // 只有双击背景时才关闭
        if (e.target === lightbox || e.target === container) {
            closeLightbox();
        }
    });
    
    // 防止拖动时意外选中文本
    lightbox.addEventListener("mousedown", (e) => { if (e.detail > 1) e.preventDefault(); });
    
    Object.assign(sizeInfo.style, {
        position: "absolute",
        bottom: "20px",
        left: "50%",
        transform: "translateX(-50%)",
        zIndex: 1003,
        color: "white", fontSize: "16px", fontFamily: "monospace",
        padding: "5px 10px", backgroundColor: "rgba(0, 0, 0, 0.7)",
        borderRadius: "4px"
    });

    container.appendChild(mediaElement);
    lightbox.appendChild(container);
    lightbox.appendChild(sizeInfo);

    // --- 控制按钮 ---
    const createButton = (text, styles) => {
        const button = document.createElement("button");
        button.textContent = text;
        Object.assign(button.style, {
            position: "fixed", zIndex: "1002",
            backgroundColor: "rgba(0, 0, 0, 0.5)", color: "white", 
            border: "1px solid #555", cursor: "pointer",
            display: "flex", justifyContent: "center", alignItems: "center",
            transition: "background-color 0.2s ease",
            ...styles
        });
        button.addEventListener("dblclick", (e) => e.stopPropagation());
        button.addEventListener("mouseenter", () => button.style.backgroundColor = "rgba(255, 255, 255, 0.2)");
        button.addEventListener("mouseleave", () => button.style.backgroundColor = "rgba(0, 0, 0, 0.5)");
        return button;
    };
    
    if (urls.length > 1) {
        const prevButton = createButton("‹", { left: "20px", top: "50%", transform: "translateY(-50%)", borderRadius: "50%", width: "50px", height: "50px", fontSize: "24px" });
        prevButton.addEventListener("click", () => loadMedia((currentIndex - 1 + urls.length) % urls.length));
        lightbox.appendChild(prevButton);
        
        const nextButton = createButton("›", { right: "20px", top: "50%", transform: "translateY(-50%)", borderRadius: "50%", width: "50px", height: "50px", fontSize: "24px" });
        nextButton.addEventListener("click", () => loadMedia((currentIndex + 1) % urls.length));
        lightbox.appendChild(nextButton);
    }

    // 只有图片模式才显示重置按钮
    if (type === 'image') {
        const resetButton = createButton("⭯", { top: "20px", right: "70px", borderRadius: "8px", width: "40px", height: "40px", fontSize: "20px" });
        resetButton.addEventListener("click", resetView);
        lightbox.appendChild(resetButton);
    }

    const closeButton = createButton("✕", { top: "20px", right: "20px", borderRadius: "8px", width: "40px", height: "40px", fontSize: "20px" });
    closeButton.addEventListener("click", closeLightbox);
    lightbox.appendChild(closeButton);

    // --- 启动 ---
    document.body.appendChild(lightbox);
    window.addEventListener("mousemove", handleMouseMove);
    window.addEventListener("mouseup", handleMouseUp);
    window.addEventListener("keydown", handleKeyDown);
    loadMedia(currentIndex);
}

/**
 * 为图片创建全屏预览
 * @param {string[]} imagePaths - 图片路径数组
 * @param {number} currentIndex - 当前图片索引
 */
export function showImageLightbox(imagePaths, currentIndex) {
    const imageUrls = imagePaths.map(path => api.apiURL(`/view?filename=${encodeURIComponent(path)}&type=input`));
    showLightbox(imageUrls, currentIndex, 'image');
}

/**
 * 为视频创建全屏预览
 * @param {string[]} videoPaths - 视频路径数组
 * @param {number} currentIndex - 当前视频索引
 */
export function showVideoLightbox(videoPaths, currentIndex) {
    const videoUrls = videoPaths.map(path => `${window.location.origin}/static_output/${encodeURIComponent(path)}`);
    showLightbox(videoUrls, currentIndex, 'video');
} 