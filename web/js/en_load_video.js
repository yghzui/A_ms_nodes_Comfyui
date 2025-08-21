// A_my_nodes: 增强 LoadVideoUpload (VHS_LoadVideo) 节点 - 拖拽上传 + 粘贴上传（仅单文件）
// 说明：
// - 复用后端 /upload/image 接口（虽然字段名叫 image，但后端不做格式校验，纯字节保存，视频同样可用）
// - 上传目标目录设置为 input，这样文件会立即出现在 LoadVideoUpload 的下拉选项里
// - 仅处理单文件。多文件时只取第一个
// - 仅在目标节点（VHS_LoadVideo）上启用拖拽；粘贴时需要该节点被选中或鼠标悬停其上
// - 尽量避免与其他全局粘贴处理冲突（例如 VHS.core.js 中的路径粘贴创建节点逻辑）

(function() {
  // 防止重复注册
  if (window.__A_MY_NODES_ENHANCE_LOAD_VIDEO__) return;
  window.__A_MY_NODES_ENHANCE_LOAD_VIDEO__ = true;

  const VIDEO_EXTS = [
    'mp4','mov','mkv','webm','avi','m4v','wmv','mpg','mpeg','3gp','flv','m2ts','ts'
  ];

  // 尝试获取全局 api/app
  const g = (typeof globalThis !== 'undefined') ? globalThis : window;
  const app = g.app;
  const api = g.api;
  const LiteGraph = g.LiteGraph;

  if (!app || !api) {
    console.warn('[A_my_nodes][en_load_video] 未找到全局 app 或 api，扩展将不生效');
    return;
  }

  // 工具：判断是否视频文件
  function isVideoFile(file) {
    try {
      if (!file) return false;
      if (file.type && file.type.startsWith('video/')) return true;
      const name = (file.name || '').toLowerCase();
      const idx = name.lastIndexOf('.');
      if (idx !== -1) {
        const ext = name.slice(idx + 1);
        return VIDEO_EXTS.includes(ext);
      }
      return false;
    } catch (err) {
      console.warn('[A_my_nodes][en_load_video] isVideoFile 异常:', err);
      return false;
    }
  }

  // 工具：上传视频文件到 input 目录（复用 /upload/image 接口）
  async function uploadVideoToInput(file) {
    // 与 load_image_batch.js 一致：FormData 字段为 image；type 指定到 input
    const formData = new FormData();
    formData.append('image', file); // 注意：后端不校验图片格式，视频也可
    formData.append('type', 'input');
    try {
      const resp = await api.fetchApi('/upload/image', { method: 'POST', body: formData });
      if (!resp || resp.status !== 200) {
        console.warn('[A_my_nodes][en_load_video] 上传失败，状态码:', resp?.status);
        return null;
      }
      const data = await resp.json();
      // 期望返回：{ name, subfolder, type }
      if (data && data.name) return data.name;
      return null;
    } catch (err) {
      console.warn('[A_my_nodes][en_load_video] 上传异常:', err);
      return null;
    }
  }

  // 工具：查找视频选择 widget（name === 'video'），如果没有则退回 widgets[0]
  function findVideoWidget(node) {
    if (!node || !node.widgets || node.widgets.length === 0) return null;
    let w = node.widgets.find(w => (w && (w.name === 'video' || w.label === 'video')));
    if (!w) w = node.widgets[0];
    return w || null;
  }

  // 工具：更新下拉选项并选中
  function selectUploadedOnWidget(node, widget, filename) {
    try {
      if (!widget || !filename) return;
      // widget.options.values 里追加（若不存在）
      if (widget.options && Array.isArray(widget.options.values)) {
        if (!widget.options.values.includes(filename)) {
          widget.options.values.push(filename);
        }
      }
      widget.value = filename;
      if (typeof widget.callback === 'function') {
        try { widget.callback(filename); } catch (cbErr) { console.warn('[A_my_nodes][en_load_video] widget.callback 异常:', cbErr); }
      }
      // 请求重绘
      try { app.canvas?.setDirty(true, true); } catch(_) {}
    } catch (err) {
      console.warn('[A_my_nodes][en_load_video] selectUploadedOnWidget 异常:', err);
    }
  }

  // 工具：定位当前“目标” VHS_LoadVideo 节点
  function resolveTargetNodeForPaste() {
    const canvas = app.canvas;
    if (!canvas) return null;
    // 1) 优先在被选中的节点中寻找 VHS_LoadVideo
    const sel = canvas.selected_nodes;
    if (sel && typeof sel === 'object') {
      for (const k in sel) {
        const n = sel[k];
        if (n && (n.type === 'VHS_LoadVideo' || n.comfyClass === 'VHS_LoadVideo')) {
          return n;
        }
      }
      // 若选中里没有 VHS_LoadVideo，但有单个选中节点，也允许用它（只要具备 video 下拉）
      const keys = Object.keys(sel);
      if (keys.length === 1) {
        const n = sel[keys[0]];
        const w = findVideoWidget(n);
        if (w) return n;
      }
    }
    // 2) 尝试使用鼠标悬停节点（over_node）
    const over = canvas.over_node;
    if (over) {
      if (over.type === 'VHS_LoadVideo' || over.comfyClass === 'VHS_LoadVideo' || findVideoWidget(over)) {
        return over;
      }
    }
    return null;
  }

  // 安装全局粘贴监听（仅一次）
  const PASTE_GUARD_KEY = '__A_MY_NODES_ENHANCE_LOAD_VIDEO_PASTE_INSTALLED__';
  if (!window[PASTE_GUARD_KEY]) {
    window[PASTE_GUARD_KEY] = true;

    document.addEventListener('paste', async (e) => {
      try {
        // 限制范围：仅当画布区域处于焦点（避免干扰输入框粘贴）
        const target = e.target;
        if (!target) return;
        const clsList = target.classList || { contains: () => false };
        const isCanvasZone = clsList.contains('litegraph') || clsList.contains('graph-canvas-container');
        if (!isCanvasZone) return;

        // 尝试从剪贴板取到视频文件（优先 clipboardData.items）
        const data = e.clipboardData || window.clipboardData;
        let file = null;
        if (data && data.items && data.items.length) {
          for (const item of data.items) {
            if (item && typeof item.type === 'string' && item.type.startsWith('video/')) {
              file = item.getAsFile();
              break;
            }
          }
        }

        // 兜底：使用异步 Clipboard API（可能需要权限，且并非所有环境可用）
        if (!file && navigator.clipboard && navigator.clipboard.read) {
          try {
            const items = await navigator.clipboard.read();
            for (const it of items) {
              for (const t of it.types || []) {
                if (t.startsWith('video/')) {
                  const blob = await it.getType(t);
                  // 尝试从 MIME 推断扩展名
                  const ext = t.split('/')[1] || 'mp4';
                  file = new File([blob], `pasted-${Date.now()}.${ext}`, { type: t });
                  break;
                }
              }
              if (file) break;
            }
          } catch (clipErr) {
            // 权限不足或不支持，忽略
          }
        }

        if (!file || !isVideoFile(file)) return;

        const node = resolveTargetNodeForPaste();
        if (!node) return; // 没有合适的目标节点则不处理

        // 处理上传
        e.preventDefault();
        e.stopImmediatePropagation(); // 避免其它粘贴处理器误处理

        const filename = await uploadVideoToInput(file);
        if (!filename) return;
        const w = findVideoWidget(node);
        if (w) selectUploadedOnWidget(node, w, filename);
      } catch (err) {
        console.warn('[A_my_nodes][en_load_video] 粘贴处理异常:', err);
      }
    }, true);
  }

  // 注册扩展：为 VHS_LoadVideo 节点挂接拖拽上传
  app.registerExtension({
    name: 'A_My_Nodes.EnhanceLoadVideoUpload',
    // 可选：loadedGraphNode 等钩子，此处用 nodeCreated
    nodeCreated(node) {
      try {
        if (!node) return;
        // 仅针对 VHS_LoadVideo（如果希望兼容核心 LoadVideo，可放宽条件到存在 video 下拉的节点）
        const isTarget = (node.type === 'VHS_LoadVideo' || node.comfyClass === 'VHS_LoadVideo' || findVideoWidget(node));
        if (!isTarget) return;

        const origDragOver = node.onDragOver;
        const origDragDrop = node.onDragDrop;

        node.onDragOver = function(e) {
          try {
            if (!e) return origDragOver?.apply(this, arguments);
            // 仅在拖入文件时启用 copy 效果
            if (e.dataTransfer && e.dataTransfer.items && e.dataTransfer.items.length) {
              const it = e.dataTransfer.items[0];
              if (it.kind === 'file') {
                e.preventDefault();
                e.dataTransfer.dropEffect = 'copy';
              }
            }
          } catch (err) {
            console.warn('[A_my_nodes][en_load_video] onDragOver 异常:', err);
          } finally {
            if (typeof origDragOver === 'function') return origDragOver.apply(this, arguments);
          }
        };

        node.onDragDrop = async function(e) {
          try {
            if (!e || !e.dataTransfer) return origDragDrop?.apply(this, arguments);
            const files = e.dataTransfer.files;
            if (!files || files.length === 0) return origDragDrop?.apply(this, arguments);

            // 仅处理单文件（视频）
            const file = files[0];
            if (!isVideoFile(file)) return origDragDrop?.apply(this, arguments);

            e.preventDefault();

            const filename = await uploadVideoToInput(file);
            if (!filename) return;

            const w = findVideoWidget(node);
            if (w) selectUploadedOnWidget(node, w, filename);
          } catch (err) {
            console.warn('[A_my_nodes][en_load_video] onDragDrop 异常:', err);
          } finally {
            if (typeof origDragDrop === 'function') return origDragDrop.apply(this, arguments);
          }
        };
      } catch (err) {
        console.warn('[A_my_nodes][en_load_video] nodeCreated 挂载异常:', err);
      }
    },
  });
})();