// 为ShowResultLast节点添加动态视频显示功能
console.log("Loading ShowResultLast.js");
import { app } from "../../../scripts/app.js";
console.log("Patching node: ShowResultLast1");
import { ComfyWidgets } from "../../../scripts/widgets.js";
console.log("Patching node: ShowResultLast2");
import { api } from "../../../scripts/api.js";

app.registerExtension({
    name: "A_my_nodes.ShowResultLast.UI",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // console.log("检查节点:", nodeData.name);
        if (nodeData.name !== "ShowResultLast") {
            return;
        }
        console.log("注册Patching node: ShowResultLast3");
        
        /**
         * 计算视频网格布局
         */
        function calculateVideoLayout(node, videoCount) {
            if (videoCount === 0) return;
            
            const containerWidth = node.size[0];
            const containerHeight = node.size[1];
            const GAP = 3;
            const PADDING = 8;
            
            // 为顶部输入控件和视频标题预留空间
            const TOP_MARGIN = 50; // 顶部控件的高度
            const TITLE_HEIGHT = 25; // 视频标题的高度
            
            const availableWidth = containerWidth - (PADDING * 2);
            const availableHeight = containerHeight - (PADDING * 2) - TOP_MARGIN - TITLE_HEIGHT;
            
            // 检查是否处于单视频模式
            if (node.singleVideoMode && node.focusedVideoIndex >= 0 && node.focusedVideoIndex < videoCount) {
                // 单视频模式：只显示一个视频，最大化显示
                const videoSize = Math.min(availableWidth, availableHeight);
                const x = PADDING + (availableWidth - videoSize) / 2;
                const y = PADDING + TOP_MARGIN + (availableHeight - videoSize) / 2;
                
                node.videoRects = [];
                for (let i = 0; i < videoCount; i++) {
                    if (i === node.focusedVideoIndex) {
                        // 显示聚焦的视频
                        node.videoRects.push({
                            x: x,
                            y: y,
                            width: videoSize,
                            height: videoSize,
                            visible: true
                        });
                    } else {
                        // 隐藏其他视频
                        node.videoRects.push({
                            x: 0,
                            y: 0,
                            width: 0,
                            height: 0,
                            visible: false
                        });
                    }
                }
                
                // 单视频模式不改变节点大小，保持当前大小
                console.log("单视频模式，保持节点大小:", node.size);
            } else {
                // 多视频模式：计算最佳网格
                let bestRows = 1;
                let bestCols = 1;
                let bestSize = 0;
                
                for (let rows = 1; rows <= videoCount; rows++) {
                    const cols = Math.ceil(videoCount / rows);
                    const sizeFromWidth = (availableWidth - (GAP * (cols - 1))) / cols;
                    const sizeFromHeight = (availableHeight - (GAP * (rows - 1))) / rows;
                    const size = Math.min(sizeFromWidth, sizeFromHeight);
                    
                    if (size > bestSize) {
                        bestSize = size;
                        bestRows = rows;
                        bestCols = cols;
                    }
                }
                
                // 计算每个视频的位置
                node.videoRects = [];
                for (let i = 0; i < videoCount; i++) {
                    const row = Math.floor(i / bestCols);
                    const col = i % bestCols;
                    const x = PADDING + col * (bestSize + GAP);
                    const y = PADDING + TOP_MARGIN + row * (bestSize + GAP);
                    
                    node.videoRects.push({
                        x: x,
                        y: y,
                        width: bestSize,
                        height: bestSize,
                        visible: true
                    });
                }
                
                // 只在初始化时调整节点大小，模式切换时不改变大小
                if (!node.sizeInitialized) {
                    const totalWidth = (bestSize * bestCols) + (GAP * (bestCols - 1)) + (PADDING * 2);
                    const totalHeight = (bestSize * bestRows) + (GAP * (bestRows - 1)) + (PADDING * 2) + TOP_MARGIN;
                    
                    const newSize = [Math.max(totalWidth, 200), Math.max(totalHeight, 100)];
                    console.log("初始化多视频模式，设置节点大小:", newSize);
                    
                    node.size[0] = newSize[0];
                    node.size[1] = newSize[1];
                    node.sizeInitialized = true;
                    node.setDirtyCanvas(true, false);
                    app.graph.setDirtyCanvas(true, false);
                } else {
                    console.log("多视频模式，保持节点大小:", node.size);
                }
            }
        }

        /**
         * 显示视频的核心实现
         */
        function showVideos(node, videoPaths) {
            console.log("开始处理新视频数据，清除旧数据...");
            
            // 清除旧的视频定时器
            if (node.videoTimer) {
                clearInterval(node.videoTimer);
                node.videoTimer = null;
            }
            
            // 暂停并清除旧的视频元素
            if (node.videos) {
                node.videos.forEach(video => {
                    if (video && !video.paused) {
                        video.pause();
                    }
                    // 清除视频源，释放内存
                    if (video.src) {
                        video.src = '';
                        video.load();
                    }
                });
            }
            
            if (!videoPaths || videoPaths.length === 0) {
                node.videos = [];
                node.videoRects = [];
                node.videoFileNames = [];
                node.videoPaths = [];
                node.fileNameRects = []; // 清除文件名区域信息
                node.deleteButtonRects = []; // 清除删除按钮区域信息
                node.singleVideoMode = false; // 清除单视频模式状态
                node.focusedVideoIndex = -1;
                node.sizeInitialized = false; // 重置大小初始化标志
                node.prevButtonRect = null; // 清除上一个按钮区域
                node.nextButtonRect = null; // 清除下一个按钮区域
                node.restoreButtonRect = null; // 清除恢复按钮区域
                console.log("没有视频数据，已清除所有旧数据");
                return;
            }
            
            const validPaths = videoPaths.filter(path => path.trim());
            console.log(`处理 ${validPaths.length} 个有效视频路径`);
            
            // 重新初始化数组
            node.videos = [];
            node.videoFileNames = [];
            node.videoPaths = validPaths; // 保存当前视频路径
            node.fileNameRects = []; // 初始化文件名区域数组
            node.deleteButtonRects = []; // 初始化删除按钮区域数组
            
            // 初始化单视频显示状态
            node.singleVideoMode = false;
            node.focusedVideoIndex = -1;
            node.sizeInitialized = false; // 标记节点大小未初始化
            
            // 为每个视频路径创建视频元素
            validPaths.forEach((path) => {
                const video = document.createElement('video');
                video.controls = true;
                video.muted = true; // 默认静音
                video.loop = true;
                video.style.maxWidth = '100%';
                video.style.maxHeight = '100%';
                video.style.objectFit = 'contain'; // 保持原始比例
                video.style.width = 'auto';
                video.style.height = 'auto';
                
                // 检测视频是否有音频轨道
                video.hasAudio = false;
                
                // 通过自定义静态文件服务获取视频URL - 使用相对路径
                const videoUrl = `${window.location.origin}/static_output/${encodeURIComponent(path)}`;
                console.log(`生成视频URL: ${videoUrl} (相对路径: ${path})`);
                video.src = videoUrl;
                
                // 添加错误处理
                video.onerror = function() {
                    console.error(`视频加载失败: ${path}, URL: ${videoUrl}`);
                };
                
                // 添加加载完成处理
                video.onloadeddata = function() {
                    console.log(`视频加载完成: ${fileName}, 原始尺寸: ${video.videoWidth}x${video.videoHeight}, 比例: ${(video.videoWidth/video.videoHeight).toFixed(2)}`);
                    
                    // 保存视频的原始比例信息
                    video.aspectRatio = video.videoWidth / video.videoHeight;
                    
                    // 检测是否有音频轨道
                    if (video.audioTracks && video.audioTracks.length > 0) {
                        video.hasAudio = true;
                        console.log(`检测到音频轨道: ${fileName}`);
                    } else {
                        // 尝试通过文件名检测音频（如果文件名包含-audio）
                        if (fileName.includes('-audio')) {
                            video.hasAudio = true;
                            console.log(`通过文件名检测到音频: ${fileName}`);
                        } else {
                            // 尝试通过其他方式检测音频
                            video.addEventListener('canplay', function() {
                                // 临时取消静音检测是否有声音
                                const wasMuted = video.muted;
                                video.muted = false;
                                
                                // 检查音频轨道
                                if (video.audioTracks && video.audioTracks.length > 0) {
                                    video.hasAudio = true;
                                    console.log(`通过音频轨道检测到音频: ${fileName}`);
                                }
                                
                                // 恢复静音状态
                                video.muted = wasMuted;
                            }, { once: true });
                        }
                    }
                    
                    // 开始播放
                    video.play().catch(e => {
                        console.warn(`自动播放失败: ${e.message}`);
                    });
                    
                    // 触发重绘以显示正确的比例
                    app.graph.setDirtyCanvas(true, false);
                };
                
                // 从相对路径中提取文件名
                const pathParts = path.split(/[\\\/]/);
                const fileName = pathParts[pathParts.length - 1];
                node.videoFileNames.push(fileName);
                
                node.videos.push(video);
            });
            
            // 计算视频布局
            calculateVideoLayout(node, validPaths.length);
            
            // 启动视频播放定时器
            if (node.videoTimer) {
                clearInterval(node.videoTimer);
            }
            
            // 每50毫秒重绘一次，确保视频流畅播放（平衡性能和流畅度）
            node.videoTimer = setInterval(() => {
                if (node.videos && node.videos.length > 0) {
                    // 检查是否有视频正在播放
                    const hasPlayingVideo = node.videos.some(video => 
                        video && !video.paused && !video.ended
                    );
                    if (hasPlayingVideo) {
                        app.graph.setDirtyCanvas(true, false);
                    }
                }
            }, 50);
            
            // 触发重绘
            app.graph.setDirtyCanvas(true, false);
        }

        /**
         * 自动调整字体大小以适应宽度
         */
        function getAdjustedFontSize(ctx, text, maxWidth, minFontSize = 8, maxFontSize = 12) {
            let fontSize = maxFontSize;
            ctx.font = `bold ${fontSize}px Arial`;
            
            while (ctx.measureText(text).width > maxWidth && fontSize > minFontSize) {
                fontSize--;
                ctx.font = `bold ${fontSize}px Arial`;
            }
            
            return fontSize;
        }

        /**
         * 绘制视频到节点
         */
        function drawNodeVideos(node, ctx) {
            if (!node.videos || !node.videoRects) return;
            
            ctx.save();
            
            for (let i = 0; i < node.videos.length && i < node.videoRects.length; i++) {
                const video = node.videos[i];
                const rect = node.videoRects[i];
                
                // 检查视频是否可见（单视频模式）
                if (rect.visible === false) {
                    continue;
                }
                
                // 绘制视频背景
                ctx.fillStyle = '#2a2a2a';
                ctx.fillRect(rect.x, rect.y, rect.width, rect.height);
                
                // 绘制视频边框
                ctx.strokeStyle = '#666';
                ctx.lineWidth = 1;
                ctx.strokeRect(rect.x, rect.y, rect.width, rect.height);
                
                // 绘制视频到Canvas - 保持原始比例，向下偏移避免被文件名遮挡
                if (video.readyState >= 2) { // HAVE_CURRENT_DATA
                    try {
                        // 为文件名预留空间
                        const titleHeight = 20;
                        const videoRect = {
                            x: rect.x,
                            y: rect.y + titleHeight, // 向下偏移
                            width: rect.width,
                            height: rect.height - titleHeight // 减去文件名高度
                        };
                        
                        // 计算视频的原始比例
                        const videoAspectRatio = video.videoWidth / video.videoHeight;
                        const rectAspectRatio = videoRect.width / videoRect.height;
                        
                        let drawWidth, drawHeight, drawX, drawY;
                        
                        if (videoAspectRatio > rectAspectRatio) {
                            // 视频更宽，以宽度为准
                            drawWidth = videoRect.width;
                            drawHeight = videoRect.width / videoAspectRatio;
                            drawX = videoRect.x;
                            drawY = videoRect.y + (videoRect.height - drawHeight) / 2;
                        } else {
                            // 视频更高，以高度为准
                            drawHeight = videoRect.height;
                            drawWidth = videoRect.height * videoAspectRatio;
                            drawX = videoRect.x + (videoRect.width - drawWidth) / 2;
                            drawY = videoRect.y;
                        }
                        
                        // 绘制视频，保持原始比例
                        ctx.drawImage(video, drawX, drawY, drawWidth, drawHeight);
                        
                        // 在视频周围绘制边框，显示实际显示区域
                        ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
                        ctx.lineWidth = 1;
                        ctx.strokeRect(drawX, drawY, drawWidth, drawHeight);
                    } catch (e) {
                        console.warn(`绘制视频失败: ${e.message}`);
                    }
                }
                
                // 绘制视频标题 - 在顶部显示文件名
                ctx.textAlign = 'center';
                
                // 使用保存的文件名
                const fileName = node.videoFileNames && node.videoFileNames[i] ? node.videoFileNames[i] : 'Unknown';
                
                // 在顶部绘制文件名背景
                ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
                ctx.fillRect(rect.x, rect.y, rect.width, 20);
                
                // 自动调整字体大小
                const maxTextWidth = rect.width - 10; // 留出边距
                const fontSize = getAdjustedFontSize(ctx, fileName, maxTextWidth);
                ctx.font = `bold ${fontSize}px Arial`;
                
                // 绘制文件名
                ctx.fillStyle = '#fff';
                ctx.fillText(fileName, rect.x + rect.width / 2, rect.y + 15);
                
                // 保存文件名区域信息，用于tooltip检测
                if (!node.fileNameRects) {
                    node.fileNameRects = [];
                }
                node.fileNameRects[i] = {
                    x: rect.x,
                    y: rect.y,
                    width: rect.width,
                    height: 20 // 文件名区域高度
                };
                
                // 绘制右上角按钮区域
                const buttonSize = 16;
                const buttonMargin = 5;
                let rightOffset = buttonMargin;
                
                // 如果视频有音频，在右上角绘制音频图标
                if (video.hasAudio) {
                    const audioIconSize = 12;
                    const audioIconX = rect.x + rect.width - rightOffset - audioIconSize;
                    const audioIconY = rect.y + buttonMargin;
                    
                    // 绘制音频图标背景
                    ctx.fillStyle = 'rgba(0, 255, 0, 0.8)';
                    ctx.beginPath();
                    ctx.arc(audioIconX + audioIconSize/2, audioIconY + audioIconSize/2, audioIconSize/2, 0, 2 * Math.PI);
                    ctx.fill();
                    
                    // 绘制音频图标
                    ctx.fillStyle = '#fff';
                    ctx.font = `${audioIconSize-2}px Arial`;
                    ctx.textAlign = 'center';
                    ctx.fillText('🔊', audioIconX + audioIconSize/2, audioIconY + audioIconSize/2 + 3);
                    
                    rightOffset += audioIconSize + buttonMargin;
                }
                
                // 绘制删除按钮 - 在音频图标左侧
                const deleteButtonX = rect.x + rect.width - rightOffset - buttonSize;
                const deleteButtonY = rect.y + buttonMargin;
                
                // 检查鼠标是否悬浮在删除按钮上
                const mouseInDeleteButton = node.mouseX !== undefined && node.mouseY !== undefined &&
                    node.mouseX >= deleteButtonX && node.mouseX <= deleteButtonX + buttonSize &&
                    node.mouseY >= deleteButtonY && node.mouseY <= deleteButtonY + buttonSize;
                
                // 绘制删除按钮背景（悬浮效果）
                ctx.fillStyle = mouseInDeleteButton ? 'rgba(255, 0, 0, 0.9)' : 'rgba(255, 0, 0, 0.7)';
                ctx.beginPath();
                ctx.arc(deleteButtonX + buttonSize/2, deleteButtonY + buttonSize/2, buttonSize/2, 0, 2 * Math.PI);
                ctx.fill();
                
                // 绘制删除按钮边框
                ctx.strokeStyle = mouseInDeleteButton ? 'rgba(255, 255, 255, 1)' : 'rgba(255, 255, 255, 0.8)';
                ctx.lineWidth = mouseInDeleteButton ? 2 : 1;
                ctx.stroke();
                
                // 绘制删除图标 (×)
                ctx.fillStyle = 'rgba(255, 255, 255, 1)';
                ctx.font = `${buttonSize - 4}px Arial`;
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText('×', deleteButtonX + buttonSize/2, deleteButtonY + buttonSize/2);
                
                // 保存删除按钮区域信息
                if (!node.deleteButtonRects) {
                    node.deleteButtonRects = [];
                }
                node.deleteButtonRects[i] = {
                    x: deleteButtonX,
                    y: deleteButtonY,
                    width: buttonSize,
                    height: buttonSize
                };
                
                // 绘制播放状态指示器 - 只在鼠标悬浮时显示
                const centerX = rect.x + rect.width / 2;
                const centerY = rect.y + rect.height / 2;
                
                // 检查鼠标是否在视频区域内 - 使用node参数而不是this
                if (node.mouseX !== undefined && node.mouseY !== undefined) {
                    const mouseInVideo = node.mouseX >= rect.x && node.mouseX <= rect.x + rect.width &&
                                       node.mouseY >= rect.y && node.mouseY <= rect.y + rect.height;
                    
                    if (mouseInVideo) {
                        // 绘制半透明背景圆形
                        ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
                        ctx.beginPath();
                        ctx.arc(centerX, centerY, 18, 0, 2 * Math.PI);
                        ctx.fill();
                        
                        // 绘制边框
                        ctx.strokeStyle = 'rgba(255, 255, 255, 0.9)';
                        ctx.lineWidth = 1.5;
                        ctx.stroke();
                        
                        if (video.paused) {
                            // 绘制播放图标
                            ctx.fillStyle = 'rgba(255, 255, 255, 1)';
                            ctx.beginPath();
                            ctx.moveTo(centerX - 6, centerY - 10);
                            ctx.lineTo(centerX - 6, centerY + 10);
                            ctx.lineTo(centerX + 10, centerY);
                            ctx.closePath();
                            ctx.fill();
                        } else {
                            // 绘制暂停图标
                            ctx.fillStyle = 'rgba(255, 255, 255, 1)';
                            ctx.fillRect(centerX - 6, centerY - 10, 3, 20);
                            ctx.fillRect(centerX + 3, centerY - 10, 3, 20);
                        }
                    }
                }
            }
            
            // 绘制控制按钮（只在单视频模式下显示）
            if (node.singleVideoMode) {
                const buttonSize = 20;
                const buttonSpacing = 5;
                
                // 检查鼠标是否悬浮在按钮上
                const mouseInRestoreButton = node.mouseX !== undefined && node.mouseY !== undefined &&
                    node.mouseX >= node.size[0] - buttonSize - 10 && node.mouseX <= node.size[0] - 10 &&
                    node.mouseY >= node.size[1] - buttonSize - 10 && node.mouseY <= node.size[1] - 10;
                
                const mouseInPrevButton = node.mouseX !== undefined && node.mouseY !== undefined &&
                    node.mouseX >= node.size[0] - buttonSize * 2 - buttonSpacing - 10 && node.mouseX <= node.size[0] - buttonSize - buttonSpacing - 10 &&
                    node.mouseY >= node.size[1] - buttonSize - 10 && node.mouseY <= node.size[1] - 10;
                
                const mouseInNextButton = node.mouseX !== undefined && node.mouseY !== undefined &&
                    node.mouseX >= node.size[0] - buttonSize * 3 - buttonSpacing * 2 - 10 && node.mouseX <= node.size[0] - buttonSize * 2 - buttonSpacing * 2 - 10 &&
                    node.mouseY >= node.size[1] - buttonSize - 10 && node.mouseY <= node.size[1] - 10;
                
                // 绘制上一个按钮 (‹)
                const prevButtonX = node.size[0] - buttonSize * 3 - buttonSpacing * 2 - 10;
                const prevButtonY = node.size[1] - buttonSize - 10;
                
                // 按钮背景（悬浮效果）
                ctx.fillStyle = mouseInPrevButton ? 'rgba(0, 0, 0, 0.9)' : 'rgba(0, 0, 0, 0.7)';
                ctx.fillRect(prevButtonX, prevButtonY, buttonSize, buttonSize);
                
                // 按钮边框
                ctx.strokeStyle = mouseInPrevButton ? 'rgba(255, 255, 255, 1)' : 'rgba(255, 255, 255, 0.8)';
                ctx.lineWidth = mouseInPrevButton ? 2 : 1;
                ctx.strokeRect(prevButtonX, prevButtonY, buttonSize, buttonSize);
                
                // 绘制‹符号
                ctx.fillStyle = 'rgba(255, 255, 255, 1)';
                ctx.font = `${buttonSize - 4}px Arial`;
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText('‹', prevButtonX + buttonSize / 2, prevButtonY + buttonSize / 2);
                
                // 绘制下一个按钮 (›)
                const nextButtonX = node.size[0] - buttonSize * 2 - buttonSpacing - 10;
                const nextButtonY = node.size[1] - buttonSize - 10;
                
                // 按钮背景（悬浮效果）
                ctx.fillStyle = mouseInNextButton ? 'rgba(0, 0, 0, 0.9)' : 'rgba(0, 0, 0, 0.7)';
                ctx.fillRect(nextButtonX, nextButtonY, buttonSize, buttonSize);
                
                // 按钮边框
                ctx.strokeStyle = mouseInNextButton ? 'rgba(255, 255, 255, 1)' : 'rgba(255, 255, 255, 0.8)';
                ctx.lineWidth = mouseInNextButton ? 2 : 1;
                ctx.strokeRect(nextButtonX, nextButtonY, buttonSize, buttonSize);
                
                // 绘制›符号
                ctx.fillStyle = 'rgba(255, 255, 255, 1)';
                ctx.fillText('›', nextButtonX + buttonSize / 2, nextButtonY + buttonSize / 2);
                
                // 绘制恢复按钮 (⭯)
                const restoreButtonX = node.size[0] - buttonSize - 10;
                const restoreButtonY = node.size[1] - buttonSize - 10;
                
                // 按钮背景（悬浮效果）
                ctx.fillStyle = mouseInRestoreButton ? 'rgba(0, 0, 0, 0.9)' : 'rgba(0, 0, 0, 0.7)';
                ctx.fillRect(restoreButtonX, restoreButtonY, buttonSize, buttonSize);
                
                // 按钮边框
                ctx.strokeStyle = mouseInRestoreButton ? 'rgba(255, 255, 255, 1)' : 'rgba(255, 255, 255, 0.8)';
                ctx.lineWidth = mouseInRestoreButton ? 2 : 1;
                ctx.strokeRect(restoreButtonX, restoreButtonY, buttonSize, buttonSize);
                
                // 绘制⭯符号
                ctx.fillStyle = 'rgba(255, 255, 255, 1)';
                ctx.fillText('⭯', restoreButtonX + buttonSize / 2, restoreButtonY + buttonSize / 2);
                
                // 保存按钮区域信息
                node.prevButtonRect = {
                    x: prevButtonX,
                    y: prevButtonY,
                    width: buttonSize,
                    height: buttonSize
                };
                node.nextButtonRect = {
                    x: nextButtonX,
                    y: nextButtonY,
                    width: buttonSize,
                    height: buttonSize
                };
                node.restoreButtonRect = {
                    x: restoreButtonX,
                    y: restoreButtonY,
                    width: buttonSize,
                    height: buttonSize
                };
            } else {
                // 清除按钮区域信息
                node.prevButtonRect = null;
                node.nextButtonRect = null;
                node.restoreButtonRect = null;
            }
            
            ctx.restore();
        }

        function populate(videoPaths) {
            console.log("收到新的视频数据，开始更新显示...");
            console.log("新视频路径:", videoPaths);
            console.log("节点当前尺寸:", this.size);
            
            // 检查是否有数据变化
            const oldPaths = this.videoPaths || [];
            const newPaths = videoPaths || [];
            
            // 比较新旧数据是否相同
            const hasChanged = oldPaths.length !== newPaths.length || 
                              oldPaths.some((oldPath, index) => oldPath !== newPaths[index]);
            
            if (!hasChanged) {
                console.log("视频数据没有变化，跳过更新");
                return;
            }
            
            console.log("检测到视频数据变化，开始清除旧数据并加载新数据");
            
            // 保存新的视频路径
            this.videoPaths = videoPaths;
            
            // 显示视频
            showVideos(this, videoPaths);
            
            // 重写节点的绘制方法
            const originalOnDrawForeground = this.onDrawForeground;
            this.onDrawForeground = function(ctx) {
                if (originalOnDrawForeground) {
                    originalOnDrawForeground.call(this, ctx);
                }
                drawNodeVideos(this, ctx);
            };
            
            // 添加鼠标事件处理
            const originalOnMouseDown = this.onMouseDown;
            const originalOnMouseMove = this.onMouseMove;
            
            console.log("设置鼠标事件处理器");
            
            // 跟踪鼠标位置并处理视频区域音频
            this.onMouseMove = function(e) {
                if (originalOnMouseMove) {
                    originalOnMouseMove.call(this, e);
                }
                
                // 保存鼠标位置用于悬浮检测
                this.mouseX = e.canvasX;
                this.mouseY = e.canvasY;
                
                // 检查鼠标是否在视频区域内，控制音频播放和显示tooltip
                if (this.videoRects && this.videos) {
                    let mouseInAnyVideo = false;
                    let currentHoveredVideo = -1;
                    
                    // 获取节点的Canvas坐标
                    const nodePos = this.pos;
                    
                    for (let i = 0; i < this.videoRects.length; i++) {
                        const rect = this.videoRects[i];
                        const video = this.videos[i];
                        
                        // 计算视频区域在Canvas中的绝对坐标
                        const absRectX = nodePos[0] + rect.x;
                        const absRectY = nodePos[1] + rect.y;
                        const absRectWidth = rect.width;
                        const absRectHeight = rect.height;
                        
                        const mouseInVideo = e.canvasX >= absRectX && e.canvasX <= absRectX + absRectWidth &&
                                           e.canvasY >= absRectY && e.canvasY <= absRectY + absRectHeight;
                        
                        if (mouseInVideo) {
                            currentHoveredVideo = i;
                            
                            if (video && video.hasAudio) {
                                mouseInAnyVideo = true;
                                // 鼠标在视频区域内，取消静音
                                if (video.muted) {
                                    console.log(`鼠标进入视频 ${i} 区域，取消静音`);
                                    video.muted = false;
                                    
                                    // 如果视频暂停了，重新播放
                                    if (video.paused) {
                                        video.play().catch(e => {
                                            console.warn(`播放音频失败: ${e.message}`);
                                        });
                                    }
                                }
                            }
                        } else {
                            // 鼠标不在视频区域内，恢复静音
                            if (video && video.hasAudio && !video.muted) {
                                console.log(`鼠标离开视频 ${i} 区域，恢复静音`);
                                video.muted = true;
                            }
                        }
                    }
                    
                    // 更新鼠标悬浮状态
                    this.mouseInVideoArea = mouseInAnyVideo;
                    
                    // 处理悬浮tooltip - 只在悬浮在文件名区域时显示
                    let tooltipShown = false;
                    if (this.fileNameRects && this.fileNameRects.length > 0) {
                        for (let i = 0; i < this.fileNameRects.length; i++) {
                            const fileNameRect = this.fileNameRects[i];
                            
                            // 计算文件名区域在Canvas中的绝对坐标
                            const absFileNameX = nodePos[0] + fileNameRect.x;
                            const absFileNameY = nodePos[1] + fileNameRect.y;
                            const absFileNameWidth = fileNameRect.width;
                            const absFileNameHeight = fileNameRect.height;
                            
                            // 检查鼠标是否在文件名区域内
                            const mouseInFileName = e.canvasX >= absFileNameX && e.canvasX <= absFileNameX + absFileNameWidth &&
                                                  e.canvasY >= absFileNameY && e.canvasY <= absFileNameY + absFileNameHeight;
                            
                            if (mouseInFileName && this.videoPaths && this.videoPaths[i]) {
                                console.log(`鼠标悬浮在文件名区域 ${i}，显示tooltip`);
                                // 显示tooltip
                                this.showTooltip(e, i);
                                tooltipShown = true;
                                break;
                            }
                        }
                    }
                    
                    // 如果没有悬浮在文件名区域，隐藏tooltip
                    if (!tooltipShown) {
                        this.hideTooltip();
                    }
                }
                
                // 触发重绘以更新悬浮状态
                app.graph.setDirtyCanvas(true, false);
            };
            
            // 鼠标离开时清除位置并恢复静音
            const originalOnMouseLeave = this.onMouseLeave;
            this.onMouseLeave = function(e) {
                if (originalOnMouseLeave) {
                    originalOnMouseLeave.call(this, e);
                }
                
                console.log("鼠标离开节点，恢复所有视频静音状态");
                
                // 清除鼠标位置
                this.mouseX = undefined;
                this.mouseY = undefined;
                this.mouseInVideoArea = false;
                
                // 隐藏tooltip
                this.hideTooltip();
                
                // 恢复所有视频的静音状态
                if (this.videos && this.videos.length > 0) {
                    this.videos.forEach((video, index) => {
                        if (video && video.hasAudio && !video.muted) {
                            console.log(`恢复视频 ${index} 的静音状态`);
                            video.muted = true;
                        }
                    });
                }
                
                // 触发重绘以隐藏指示器
                app.graph.setDirtyCanvas(true, false);
            };
            
            this.onMouseDown = function(e) {
                console.log("onMouseDown 被调用", e);
                console.log("节点信息:", this.id, this.type, this.size);
                console.log("视频区域:", this.videoRects);
                
                // 获取节点的Canvas坐标
                const nodePos = this.pos;
                
                // 检查是否点击控制按钮（单视频模式下）
                if (this.singleVideoMode) {
                    // 检查点击上一个按钮 (‹)
                    if (this.prevButtonRect) {
                        const absPrevButtonX = nodePos[0] + this.prevButtonRect.x;
                        const absPrevButtonY = nodePos[1] + this.prevButtonRect.y;
                        const absPrevButtonWidth = this.prevButtonRect.width;
                        const absPrevButtonHeight = this.prevButtonRect.height;
                        
                        if (e.canvasX >= absPrevButtonX && e.canvasX <= absPrevButtonX + absPrevButtonWidth &&
                            e.canvasY >= absPrevButtonY && e.canvasY <= absPrevButtonY + absPrevButtonHeight) {
                            
                            console.log("点击上一个按钮");
                            
                            // 阻止事件冒泡
                            e.preventDefault();
                            e.stopPropagation();
                            
                            // 切换到上一个视频
                            if (this.videoPaths && this.videoPaths.length > 0) {
                                this.focusedVideoIndex = (this.focusedVideoIndex - 1 + this.videoPaths.length) % this.videoPaths.length;
                                console.log(`切换到上一个视频，当前索引: ${this.focusedVideoIndex}`);
                                
                                // 重新计算布局
                                calculateVideoLayout(this, this.videoPaths.length);
                                
                                // 触发重绘
                                app.graph.setDirtyCanvas(true, false);
                            }
                            
                            return true;
                        }
                    }
                    
                    // 检查点击下一个按钮 (›)
                    if (this.nextButtonRect) {
                        const absNextButtonX = nodePos[0] + this.nextButtonRect.x;
                        const absNextButtonY = nodePos[1] + this.nextButtonRect.y;
                        const absNextButtonWidth = this.nextButtonRect.width;
                        const absNextButtonHeight = this.nextButtonRect.height;
                        
                        if (e.canvasX >= absNextButtonX && e.canvasX <= absNextButtonX + absNextButtonWidth &&
                            e.canvasY >= absNextButtonY && e.canvasY <= absNextButtonY + absNextButtonHeight) {
                            
                            console.log("点击下一个按钮");
                            
                            // 阻止事件冒泡
                            e.preventDefault();
                            e.stopPropagation();
                            
                            // 切换到下一个视频
                            if (this.videoPaths && this.videoPaths.length > 0) {
                                this.focusedVideoIndex = (this.focusedVideoIndex + 1) % this.videoPaths.length;
                                console.log(`切换到下一个视频，当前索引: ${this.focusedVideoIndex}`);
                                
                                // 重新计算布局
                                calculateVideoLayout(this, this.videoPaths.length);
                                
                                // 触发重绘
                                app.graph.setDirtyCanvas(true, false);
                            }
                            
                            return true;
                        }
                    }
                    
                    // 检查点击恢复按钮 (⭯)
                    if (this.restoreButtonRect) {
                        const absRestoreButtonX = nodePos[0] + this.restoreButtonRect.x;
                        const absRestoreButtonY = nodePos[1] + this.restoreButtonRect.y;
                        const absRestoreButtonWidth = this.restoreButtonRect.width;
                        const absRestoreButtonHeight = this.restoreButtonRect.height;
                        
                        if (e.canvasX >= absRestoreButtonX && e.canvasX <= absRestoreButtonX + absRestoreButtonWidth &&
                            e.canvasY >= absRestoreButtonY && e.canvasY <= absRestoreButtonY + absRestoreButtonHeight) {
                            
                            console.log("点击恢复按钮，退出单视频模式");
                            
                            // 阻止事件冒泡
                            e.preventDefault();
                            e.stopPropagation();
                            
                            // 退出单视频模式
                            this.singleVideoMode = false;
                            this.focusedVideoIndex = -1;
                            
                            // 重新计算布局
                            if (this.videoPaths && this.videoPaths.length > 0) {
                                calculateVideoLayout(this, this.videoPaths.length);
                            }
                            
                            // 触发重绘
                            app.graph.setDirtyCanvas(true, false);
                            
                            return true;
                        }
                    }
                }
                
                // 检查是否点击删除按钮
                if (this.deleteButtonRects && this.deleteButtonRects.length > 0) {
                    for (let i = 0; i < this.deleteButtonRects.length; i++) {
                        const deleteRect = this.deleteButtonRects[i];
                        
                        // 检查视频是否可见
                        if (this.videoRects && this.videoRects[i] && this.videoRects[i].visible === false) {
                            continue;
                        }
                        
                        // 计算删除按钮在Canvas中的绝对坐标
                        const absDeleteButtonX = nodePos[0] + deleteRect.x;
                        const absDeleteButtonY = nodePos[1] + deleteRect.y;
                        const absDeleteButtonWidth = deleteRect.width;
                        const absDeleteButtonHeight = deleteRect.height;
                        
                        if (e.canvasX >= absDeleteButtonX && e.canvasX <= absDeleteButtonX + absDeleteButtonWidth &&
                            e.canvasY >= absDeleteButtonY && e.canvasY <= absDeleteButtonY + absDeleteButtonHeight) {
                            
                            console.log(`点击删除按钮，视频索引: ${i}`);
                            
                            // 阻止事件冒泡
                            e.preventDefault();
                            e.stopPropagation();
                            
                            // 执行删除操作
                            this.deleteVideoWithConfirmation(i);
                            
                            return true;
                        }
                    }
                }
                
                // 检查鼠标是否在视频框内
                if (this.videoRects && this.videoRects.length > 0) {
                    console.log("检查视频区域点击", this.videoRects.length, "个视频区域");
                    
                    for (let i = 0; i < this.videoRects.length; i++) {
                        const rect = this.videoRects[i];
                        
                        // 检查视频是否可见
                        if (rect.visible === false) {
                            continue;
                        }
                        
                        // 计算视频区域在Canvas中的绝对坐标
                        const absRectX = nodePos[0] + rect.x;
                        const absRectY = nodePos[1] + rect.y;
                        const absRectWidth = rect.width;
                        const absRectHeight = rect.height;
                        
                        console.log(`检查视频 ${i}:`, {
                            rect: rect,
                            绝对坐标: {x: absRectX, y: absRectY, width: absRectWidth, height: absRectHeight},
                            鼠标位置: {x: e.canvasX, y: e.canvasY}
                        });
                        
                        // 检查鼠标是否在视频区域内
                        if (e.canvasX >= absRectX && e.canvasX <= absRectX + absRectWidth &&
                            e.canvasY >= absRectY && e.canvasY <= absRectY + absRectHeight) {
                            
                            console.log(`鼠标在视频 ${i} 区域内`);
                            
                            // 检查是否点击在播放状态指示器区域（中心区域）
                            const centerX = absRectX + absRectWidth / 2;
                            const centerY = absRectY + absRectHeight / 2;
                            const indicatorSize = 25; // 指示器的大小
                            
                            const inCenter = e.canvasX >= centerX - indicatorSize/2 && 
                                           e.canvasX <= centerX + indicatorSize/2 &&
                                           e.canvasY >= centerY - indicatorSize/2 && 
                                           e.canvasY <= centerY + indicatorSize/2;
                            
                            if (inCenter) {
                                console.log(`点击在播放状态指示器上，视频 ${i}`);
                                
                                // 阻止事件冒泡，避免触发节点选择
                                e.preventDefault();
                                e.stopPropagation();
                                
                                // 点击在播放状态指示器上，切换播放/暂停
                                if (this.videos && this.videos[i]) {
                                    const video = this.videos[i];
                                    console.log(`切换视频 ${i} 播放状态，当前状态: ${video.paused ? '暂停' : '播放'}`);
                                    
                                    // 立即执行，不使用setTimeout
                                    if (video.paused) {
                                        video.play().catch(e => {
                                            console.warn(`播放视频失败: ${e.message}`);
                                        });
                                    } else {
                                        video.pause();
                                    }
                                    
                                    // 触发重绘
                                    app.graph.setDirtyCanvas(true, false);
                                }
                                
                                // 返回true表示事件已处理
                                return true;
                            } else {
                                console.log(`点击在视频 ${i} 其他区域`);
                                
                                // 阻止事件冒泡，避免触发节点选择
                                e.preventDefault();
                                e.stopPropagation();
                                
                                // 如果不在单视频模式，进入单视频模式
                                if (!this.singleVideoMode) {
                                    console.log(`进入单视频模式，聚焦视频 ${i}`);
                                    this.singleVideoMode = true;
                                    this.focusedVideoIndex = i;
                                    
                                    // 重新计算布局
                                    if (this.videoPaths && this.videoPaths.length > 0) {
                                        calculateVideoLayout(this, this.videoPaths.length);
                                    }
                                    
                                    // 触发重绘
                                    app.graph.setDirtyCanvas(true, false);
                                }
                                
                                // 返回true表示事件已处理
                                return true;
                            }
                        }
                    }
                }
                
                // 如果没有处理视频区域点击，调用原始事件处理
                if (originalOnMouseDown) {
                    return originalOnMouseDown.call(this, e);
                }
                
                return false;
            };
            
            // 重写节点的resize方法，当大小改变时重新计算布局
            const originalOnResize = this.onResize;
            this.onResize = function(size) {
                if (originalOnResize) {
                    originalOnResize.call(this, size);
                }
                console.log("节点大小改变，重新计算布局:", size);
                
                // 重新计算视频布局，但不调整节点大小
                if (this.videoPaths && this.videoPaths.length > 0) {
                    // 临时保存当前大小
                    const currentSize = [this.size[0], this.size[1]];
                    
                    // 计算布局但不调整大小
                    calculateVideoLayout(this, this.videoPaths.length);
                    
                    // 恢复原始大小，避免递归
                    this.size[0] = currentSize[0];
                    this.size[1] = currentSize[1];
                }
            };
            
            // 添加tooltip管理方法
            this.showTooltip = function(e, videoIndex) {
                // 如果已经有tooltip，先移除
                this.hideTooltip();
                
                if (this.videoPaths && this.videoPaths[videoIndex]) {
                    const tooltip = document.createElement('div');
                    tooltip.id = 'video-tooltip-' + this.id;
                    tooltip.style.cssText = `
                        position: fixed;
                        background: rgba(0, 0, 0, 0.9);
                        color: white;
                        padding: 8px 12px;
                        border-radius: 4px;
                        font-size: 12px;
                        max-width: 400px;
                        word-wrap: break-word;
                        z-index: 10000;
                        pointer-events: none;
                        white-space: nowrap;
                    `;
                    
                    // 获取视频的原始尺寸信息
                    const video = this.videos[videoIndex];
                    let sizeInfo = '';
                    if (video && video.videoWidth && video.videoHeight) {
                        sizeInfo = ` (${video.videoWidth}x${video.videoHeight})`;
                    }
                    
                    tooltip.textContent = `相对路径: ${this.videoPaths[videoIndex]}${sizeInfo}`;
                    document.body.appendChild(tooltip);
                    
                    // 设置tooltip位置，确保不超出屏幕边界
                    const tooltipRect = tooltip.getBoundingClientRect();
                    let left = e.clientX + 10;
                    let top = e.clientY - 30;
                    
                    // 检查右边界
                    if (left + tooltipRect.width > window.innerWidth) {
                        left = e.clientX - tooltipRect.width - 10;
                    }
                    
                    // 检查下边界
                    if (top + tooltipRect.height > window.innerHeight) {
                        top = e.clientY - tooltipRect.height - 10;
                    }
                    
                    tooltip.style.left = left + 'px';
                    tooltip.style.top = top + 'px';
                }
            };
            
            this.hideTooltip = function() {
                const existingTooltip = document.getElementById('video-tooltip-' + this.id);
                if (existingTooltip) {
                    existingTooltip.remove();
                }
            };
            
            // 删除视频及其关联文件的确认对话框
            this.deleteVideoWithConfirmation = function(videoIndex) {
                if (!this.videoPaths || !this.videoPaths[videoIndex]) {
                    console.error("无效的视频索引:", videoIndex);
                    return;
                }
                
                const videoPath = this.videoPaths[videoIndex];
                const fileName = this.videoFileNames && this.videoFileNames[videoIndex] ? this.videoFileNames[videoIndex] : 'Unknown';
                
                // 生成关联文件列表
                const relatedFiles = this.generateRelatedFiles(videoPath);
                
                // 创建确认对话框
                const confirmDialog = document.createElement('div');
                confirmDialog.id = 'delete-confirm-dialog-' + this.id;
                confirmDialog.style.cssText = `
                    position: fixed;
                    top: 50%;
                    left: 50%;
                    transform: translate(-50%, -50%);
                    background: #2a2a2a;
                    border: 2px solid #666;
                    border-radius: 8px;
                    padding: 20px;
                    z-index: 10001;
                    max-width: 500px;
                    color: white;
                    font-family: Arial, sans-serif;
                `;
                
                // 构建确认消息
                let confirmMessage = `<h3 style="margin: 0 0 15px 0; color: #ff6b6b;">⚠️ 确认删除文件</h3>`;
                confirmMessage += `<p style="margin: 0 0 10px 0;"><strong>主文件:</strong> ${fileName}</p>`;
                
                if (relatedFiles.length > 1) {
                    confirmMessage += `<p style="margin: 0 0 10px 0;"><strong>关联文件:</strong></p>`;
                    confirmMessage += `<ul style="margin: 0 0 15px 0; padding-left: 20px;">`;
                    relatedFiles.forEach(file => {
                        if (file !== videoPath) {
                            const relatedFileName = file.split(/[\\\/]/).pop();
                            confirmMessage += `<li>${relatedFileName}</li>`;
                        }
                    });
                    confirmMessage += `</ul>`;
                }
                
                confirmMessage += `<p style="margin: 0 0 20px 0; color: #ff6b6b;"><strong>此操作不可撤销！</strong></p>`;
                
                // 添加按钮
                confirmMessage += `
                    <div style="display: flex; gap: 10px; justify-content: flex-end;">
                        <button id="cancel-delete-${this.id}" style="
                            padding: 8px 16px;
                            background: #666;
                            color: white;
                            border: none;
                            border-radius: 4px;
                            cursor: pointer;
                        ">取消</button>
                        <button id="confirm-delete-${this.id}" style="
                            padding: 8px 16px;
                            background: #ff6b6b;
                            color: white;
                            border: none;
                            border-radius: 4px;
                            cursor: pointer;
                        ">确认删除</button>
                    </div>
                `;
                
                confirmDialog.innerHTML = confirmMessage;
                document.body.appendChild(confirmDialog);
                
                // 添加背景遮罩
                const overlay = document.createElement('div');
                overlay.id = 'delete-overlay-' + this.id;
                overlay.style.cssText = `
                    position: fixed;
                    top: 0;
                    left: 0;
                    width: 100%;
                    height: 100%;
                    background: rgba(0, 0, 0, 0.5);
                    z-index: 10000;
                `;
                document.body.appendChild(overlay);
                
                // 绑定按钮事件
                document.getElementById(`cancel-delete-${this.id}`).onclick = () => {
                    this.removeDeleteDialog();
                };
                
                document.getElementById(`confirm-delete-${this.id}`).onclick = () => {
                    this.removeDeleteDialog();
                    this.executeDelete(videoIndex, relatedFiles);
                };
                
                // 点击遮罩关闭对话框
                overlay.onclick = () => {
                    this.removeDeleteDialog();
                };
            };
            
            // 移除删除确认对话框
            this.removeDeleteDialog = function() {
                const dialog = document.getElementById('delete-confirm-dialog-' + this.id);
                const overlay = document.getElementById('delete-overlay-' + this.id);
                if (dialog) dialog.remove();
                if (overlay) overlay.remove();
            };
            
            // 生成关联文件列表
            this.generateRelatedFiles = function(videoPath) {
                const files = [videoPath]; // 主文件
                
                // 从路径中提取文件名（不含扩展名）
                const pathParts = videoPath.split(/[\\\/]/);
                const fullFileName = pathParts[pathParts.length - 1];
                const fileNameWithoutExt = fullFileName.replace(/\.[^/.]+$/, "");
                
                // 生成可能的关联文件
                const possibleExtensions = ['.mp4'];
                const possibleSuffixes = ['-audio'];
                
                // 检查音频文件
                possibleExtensions.forEach(ext => {
                    const audioFile = videoPath.replace(/\.[^/.]+$/, "") + '-audio' + ext;
                    files.push(audioFile);
                });
                
                // 检查预览图片
                const imageExtensions = ['.png'];
                imageExtensions.forEach(ext => {
                    const imageFile = videoPath.replace(/\.[^/.]+$/, "") + ext;
                    files.push(imageFile);
                });
                
                return files;
            };
            
            // 执行删除操作
            this.executeDelete = async function(videoIndex, relatedFiles) {
                console.log(`开始删除视频 ${videoIndex} 及其关联文件`);
                console.log("要删除的文件:", relatedFiles);
                
                const results = {
                    success: [],
                    failed: [],
                    total: relatedFiles.length
                };
                
                // 显示删除进度对话框
                this.showDeleteProgress(results);
                
                // 逐个删除文件
                for (let i = 0; i < relatedFiles.length; i++) {
                    const filePath = relatedFiles[i];
                    
                    try {
                        const response = await fetch('/delete_output_file', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json',
                            },
                            body: JSON.stringify({
                                path: filePath
                            })
                        });
                        
                        const result = await response.json();
                        
                        if (response.ok && result.success) {
                            results.success.push(filePath);
                            console.log(`✅ 成功删除: ${filePath}`);
                        } else {
                            results.failed.push({
                                path: filePath,
                                error: result.error || '未知错误'
                            });
                            console.log(`❌ 删除失败: ${filePath}, 错误: ${result.error}`);
                        }
                    } catch (error) {
                        results.failed.push({
                            path: filePath,
                            error: error.message
                        });
                        console.log(`❌ 删除异常: ${filePath}, 错误: ${error.message}`);
                    }
                    
                    // 更新进度
                    this.updateDeleteProgress(results, i + 1);
                }
                
                // 显示删除结果
                this.showDeleteResults(results, videoIndex);
            };
            
            // 显示删除进度
            this.showDeleteProgress = function(results) {
                const progressDialog = document.createElement('div');
                progressDialog.id = 'delete-progress-dialog-' + this.id;
                progressDialog.style.cssText = `
                    position: fixed;
                    top: 50%;
                    left: 50%;
                    transform: translate(-50%, -50%);
                    background: #2a2a2a;
                    border: 2px solid #666;
                    border-radius: 8px;
                    padding: 20px;
                    z-index: 10001;
                    min-width: 300px;
                    color: white;
                    font-family: Arial, sans-serif;
                `;
                
                progressDialog.innerHTML = `
                    <h3 style="margin: 0 0 15px 0;">🗑️ 正在删除文件...</h3>
                    <div id="delete-progress-text-${this.id}" style="margin: 0 0 10px 0;">准备删除...</div>
                    <div style="background: #444; height: 20px; border-radius: 10px; overflow: hidden;">
                        <div id="delete-progress-bar-${this.id}" style="
                            background: #4CAF50;
                            height: 100%;
                            width: 0%;
                            transition: width 0.3s;
                        "></div>
                    </div>
                `;
                
                document.body.appendChild(progressDialog);
            };
            
            // 更新删除进度
            this.updateDeleteProgress = function(results, currentCount) {
                const progressText = document.getElementById(`delete-progress-text-${this.id}`);
                const progressBar = document.getElementById(`delete-progress-bar-${this.id}`);
                
                if (progressText && progressBar) {
                    const percentage = (currentCount / results.total) * 100;
                    progressText.textContent = `正在删除... (${currentCount}/${results.total})`;
                    progressBar.style.width = percentage + '%';
                }
            };
            
            // 显示删除结果
            this.showDeleteResults = function(results, videoIndex) {
                // 移除进度对话框
                const progressDialog = document.getElementById(`delete-progress-dialog-${this.id}`);
                if (progressDialog) progressDialog.remove();
                
                // 创建结果对话框
                const resultDialog = document.createElement('div');
                resultDialog.id = 'delete-result-dialog-' + this.id;
                resultDialog.style.cssText = `
                    position: fixed;
                    top: 50%;
                    left: 50%;
                    transform: translate(-50%, -50%);
                    background: #2a2a2a;
                    border: 2px solid #666;
                    border-radius: 8px;
                    padding: 20px;
                    z-index: 10001;
                    max-width: 500px;
                    max-height: 400px;
                    overflow-y: auto;
                    color: white;
                    font-family: Arial, sans-serif;
                `;
                
                // 构建结果消息
                let resultMessage = `<h3 style="margin: 0 0 15px 0; color: ${results.failed.length === 0 ? '#4CAF50' : '#ff6b6b'};">`;
                resultMessage += results.failed.length === 0 ? '✅ 删除完成' : '⚠️ 删除部分完成';
                resultMessage += `</h3>`;
                
                resultMessage += `<p style="margin: 0 0 10px 0;"><strong>总计:</strong> ${results.total} 个文件</p>`;
                resultMessage += `<p style="margin: 0 0 10px 0; color: #4CAF50;"><strong>成功:</strong> ${results.success.length} 个文件</p>`;
                
                if (results.failed.length > 0) {
                    resultMessage += `<p style="margin: 0 0 10px 0; color: #ff6b6b;"><strong>失败:</strong> ${results.failed.length} 个文件</p>`;
                    resultMessage += `<details style="margin: 0 0 15px 0;">`;
                    resultMessage += `<summary style="cursor: pointer; color: #ff6b6b;">查看失败详情</summary>`;
                    resultMessage += `<ul style="margin: 10px 0 0 0; padding-left: 20px;">`;
                    results.failed.forEach(fail => {
                        const fileName = fail.path.split(/[\\\/]/).pop();
                        resultMessage += `<li>${fileName}: ${fail.error}</li>`;
                    });
                    resultMessage += `</ul>`;
                    resultMessage += `</details>`;
                }
                
                // 添加关闭按钮
                resultMessage += `
                    <div style="display: flex; gap: 10px; justify-content: flex-end;">
                        <button id="close-result-${this.id}" style="
                            padding: 8px 16px;
                            background: #666;
                            color: white;
                            border: none;
                            border-radius: 4px;
                            cursor: pointer;
                        ">关闭</button>
                    </div>
                `;
                
                resultDialog.innerHTML = resultMessage;
                document.body.appendChild(resultDialog);
                
                // 绑定关闭按钮事件
                document.getElementById(`close-result-${this.id}`).onclick = () => {
                    this.removeResultDialog();
                    
                    // 如果删除成功，从列表中移除该视频
                    if (results.success.length > 0) {
                        this.removeVideoFromList(videoIndex);
                    }
                };
                
                // 3秒后自动关闭
                setTimeout(() => {
                    this.removeResultDialog();
                    if (results.success.length > 0) {
                        this.removeVideoFromList(videoIndex);
                    }
                }, 3000);
            };
            
            // 移除结果对话框
            this.removeResultDialog = function() {
                const dialog = document.getElementById('delete-result-dialog-' + this.id);
                if (dialog) dialog.remove();
            };
            
            // 从列表中移除视频
            this.removeVideoFromList = function(videoIndex) {
                if (!this.videoPaths || videoIndex < 0 || videoIndex >= this.videoPaths.length) {
                    return;
                }
                
                console.log(`从列表中移除视频 ${videoIndex}`);
                
                // 暂停并清理视频元素
                if (this.videos && this.videos[videoIndex]) {
                    const video = this.videos[videoIndex];
                    if (!video.paused) {
                        video.pause();
                    }
                    if (video.src) {
                        video.src = '';
                        video.load();
                    }
                }
                
                // 从数组中移除相关数据
                this.videoPaths.splice(videoIndex, 1);
                if (this.videos) this.videos.splice(videoIndex, 1);
                if (this.videoFileNames) this.videoFileNames.splice(videoIndex, 1);
                if (this.videoRects) this.videoRects.splice(videoIndex, 1);
                if (this.fileNameRects) this.fileNameRects.splice(videoIndex, 1);
                if (this.deleteButtonRects) this.deleteButtonRects.splice(videoIndex, 1);
                
                // 重新计算布局
                if (this.videoPaths.length > 0) {
                    calculateVideoLayout(this, this.videoPaths.length);
                } else {
                    // 如果没有视频了，清除所有数据
                    this.videos = [];
                    this.videoRects = [];
                    this.videoFileNames = [];
                    this.fileNameRects = [];
                    this.deleteButtonRects = [];
                    this.singleVideoMode = false;
                    this.focusedVideoIndex = -1;
                }
                
                // 触发重绘
                app.graph.setDirtyCanvas(true, false);
            };
            
            // 延迟触发重绘，确保布局计算完成
            setTimeout(() => {
                console.log("延迟后的节点尺寸:", this.size);
                console.log("视频区域信息:", this.videoRects);
                app.graph.setDirtyCanvas(true, false);
            }, 100);
        }

        // 监听节点输入数据变化
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated?.apply(this, arguments);
            console.log("ShowResultLast 节点创建完成");
            
            // 监听输入连接变化
            const onConnectionsChange = this.onConnectionsChange;
            this.onConnectionsChange = function (type, index, connected, link_info, output) {
                onConnectionsChange?.apply(this, arguments);
                console.log("连接变化:", type, index, connected, link_info, output);
                
                if (type === "input" && index === 0 && connected) {
                    // 当Filenames输入连接时，尝试获取数据
                    setTimeout(() => {
                        if (this.inputs && this.inputs[0] && this.inputs[0].link) {
                            console.log("检测到Filenames输入连接");
                            // 这里可以尝试获取连接的数据
                        }
                    }, 100);
                }
            };
        };

        // 添加节点销毁时的清理逻辑
        const onRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
            // 清理视频定时器
            if (this.videoTimer) {
                clearInterval(this.videoTimer);
                this.videoTimer = null;
            }
            
            // 暂停所有视频
            if (this.videos) {
                this.videos.forEach(video => {
                    if (video && !video.paused) {
                        video.pause();
                    }
                });
            }
            
            // 清理tooltip
            if (this.hideTooltip) {
                this.hideTooltip();
            }
            
            // 清理删除相关的对话框
            if (this.removeDeleteDialog) {
                this.removeDeleteDialog();
            }
            if (this.removeResultDialog) {
                this.removeResultDialog();
            }
            
            // 清理进度对话框
            const progressDialog = document.getElementById(`delete-progress-dialog-${this.id}`);
            if (progressDialog) progressDialog.remove();
            
            onRemoved?.apply(this, arguments);
            console.log("ShowResultLast 节点已清理");
        };

        // 处理节点执行完成
        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);
            console.log("ShowResultLast onExecuted 被调用，message:", message);
            
            // 处理Python返回的数据
            if (message && message.text) {
                console.log("收到Python数据:", message.text);
                
                // 检查数据是否为空或无效
                if (!message.text || (Array.isArray(message.text) && message.text.length === 0)) {
                    console.log("收到空数据，清除所有视频");
                    populate.call(this, []);
                } else {
                    populate.call(this, message.text);
                }
            } else {
                console.log("没有收到数据，显示等待状态");
                populate.call(this, ["等待数据..."]);
            }
        };
    }
}); 