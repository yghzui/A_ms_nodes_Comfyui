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
            const GAP = 5;
            const PADDING = 10;
            
            // 为顶部输入控件和视频标题预留空间
            const TOP_MARGIN = 50; // 顶部控件的高度
            const TITLE_HEIGHT = 25; // 视频标题的高度
            
            const availableWidth = containerWidth - (PADDING * 2);
            const availableHeight = containerHeight - (PADDING * 2) - TOP_MARGIN - TITLE_HEIGHT;
            
            // 计算最佳网格
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
                const y = PADDING + TOP_MARGIN + row * (bestSize + GAP + TITLE_HEIGHT); // 加上顶部边距和标题空间
                
                node.videoRects.push({
                    x: x,
                    y: y,
                    width: bestSize,
                    height: bestSize
                });
            }
            
            // 调整节点大小以适应内容
            const totalWidth = (bestSize * bestCols) + (GAP * (bestCols - 1)) + (PADDING * 2);
            const totalHeight = (bestSize * bestRows) + (GAP * (bestRows - 1)) + (PADDING * 2) + TOP_MARGIN + TITLE_HEIGHT;
            
            const newSize = [Math.max(totalWidth, 200), Math.max(totalHeight, 100)];
            console.log("计算的新尺寸:", newSize, "当前尺寸:", node.size);
            
            if (newSize[0] !== node.size[0] || newSize[1] !== node.size[1]) {
                // 使用正确的方法调整节点大小
                node.size[0] = newSize[0];
                node.size[1] = newSize[1];
                
                // 标记需要重绘，但不调用onResize避免递归
                node.setDirtyCanvas(true, false);
                app.graph.setDirtyCanvas(true, false);
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
                console.log("没有视频数据，已清除所有旧数据");
                return;
            }
            
            const validPaths = videoPaths.filter(path => path.trim());
            console.log(`处理 ${validPaths.length} 个有效视频路径`);
            
            // 重新初始化数组
            node.videos = [];
            node.videoFileNames = [];
            node.videoPaths = validPaths; // 保存当前视频路径
            
            // 为每个视频路径创建视频元素
            validPaths.forEach((path) => {
                const video = document.createElement('video');
                video.controls = true;
                video.muted = true;
                video.loop = true;
                video.style.maxWidth = '100%';
                video.style.maxHeight = '100%';
                video.style.objectFit = 'contain'; // 保持原始比例
                video.style.width = 'auto';
                video.style.height = 'auto';
                
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
                
                // 绘制播放状态指示器
                if (video.paused) {
                    // 绘制暂停图标
                    ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
                    ctx.fillRect(rect.x + rect.width / 2 - 15, rect.y + rect.height / 2 - 15, 8, 30);
                    ctx.fillRect(rect.x + rect.width / 2 + 7, rect.y + rect.height / 2 - 15, 8, 30);
                } else {
                    // 绘制播放图标
                    ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
                    ctx.beginPath();
                    ctx.moveTo(rect.x + rect.width / 2 - 10, rect.y + rect.height / 2 - 15);
                    ctx.lineTo(rect.x + rect.width / 2 - 10, rect.y + rect.height / 2 + 15);
                    ctx.lineTo(rect.x + rect.width / 2 + 15, rect.y + rect.height / 2);
                    ctx.closePath();
                    ctx.fill();
                }
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
            this.onMouseDown = function(e) {
                if (originalOnMouseDown) {
                    originalOnMouseDown.call(this, e);
                }
                
                // 检查鼠标是否在视频框内
                if (this.videoRects) {
                    for (let i = 0; i < this.videoRects.length; i++) {
                        const rect = this.videoRects[i];
                        if (e.canvasX >= rect.x && e.canvasX <= rect.x + rect.width &&
                            e.canvasY >= rect.y && e.canvasY <= rect.y + rect.height) {
                            
                            // 切换视频播放/暂停状态
                            if (this.videos && this.videos[i]) {
                                const video = this.videos[i];
                                if (video.paused) {
                                    video.play().catch(e => {
                                        console.warn(`播放视频失败: ${e.message}`);
                                    });
                                } else {
                                    video.pause();
                                }
                            }
                            
                            // 显示相对路径的tooltip
                            if (this.videoPaths && this.videoPaths[i]) {
                                const tooltip = document.createElement('div');
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
                                `;
                                // 获取视频的原始尺寸信息
                                const video = this.videos[i];
                                let sizeInfo = '';
                                if (video && video.videoWidth && video.videoHeight) {
                                    sizeInfo = ` (${video.videoWidth}x${video.videoHeight})`;
                                }
                                tooltip.textContent = `相对路径: ${this.videoPaths[i]}${sizeInfo}`;
                                document.body.appendChild(tooltip);
                                
                                // 设置tooltip位置
                                tooltip.style.left = (e.clientX + 10) + 'px';
                                tooltip.style.top = (e.clientY - 30) + 'px';
                                
                                // 3秒后移除tooltip
                                setTimeout(() => {
                                    if (tooltip.parentNode) {
                                        tooltip.parentNode.removeChild(tooltip);
                                    }
                                }, 3000);
                            }
                            break;
                        }
                    }
                }
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
            
            // 延迟触发重绘，确保布局计算完成
            setTimeout(() => {
                console.log("延迟后的节点尺寸:", this.size);
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