import { api } from "../../../scripts/api.js";
import { showTopNotification } from "../utils/shared_utils.js";
import { showImageEditor as showCoreImageEditor } from "./core/editor_core.js";

/**
 * ComfyUI 宿主适配器：
 * 只负责把纯内核需要的图片读取、上传、保存与通知映射到 ComfyUI 接口，
 * 编辑器结构与交互全部由 ./core/editor_core.js（由主项目同步生成）提供。
 */
const buildViewUrl = (imagePath) => {
    let filename = String(imagePath || "");
    let type = "input";
    let subfolder = "";
    const typeMatch = filename.match(/^(.*)\s+\[(input|output|temp)\]$/);
    if (typeMatch) {
        filename = typeMatch[1];
        type = typeMatch[2];
    }
    const splitIndex = Math.max(filename.lastIndexOf("/"), filename.lastIndexOf("\\"));
    if (splitIndex !== -1) {
        subfolder = filename.substring(0, splitIndex);
        filename = filename.substring(splitIndex + 1);
    }
    const params = new URLSearchParams({ filename, type });
    if (subfolder) params.set("subfolder", subfolder);
    return api.apiURL(`/a_my_nodes/view_input?${params.toString()}`);
};

const getSideLayerUrls = (currentPath) => {
    if (!String(currentPath || "").includes("clipspace-painted-masked-")) return null;
    const match = String(currentPath).match(/clipspace-painted-masked-(\d+)\.png/);
    if (!match) return null;
    return {
        maskUrl: api.apiURL(`/a_my_nodes/view_input?filename=clipspace-mask-${match[1]}.png&type=input&subfolder=clipspace`),
        paintUrl: api.apiURL(`/a_my_nodes/view_input?filename=clipspace-paint-${match[1]}.png&type=input&subfolder=clipspace`)
    };
};

const uploadBaseImage = async (baseBlob, fileName) => {
    const formData = new FormData();
    formData.append("image", new File([baseBlob], fileName, { type: "image/png" }), fileName);
    formData.append("type", "input");
    formData.append("subfolder", "clipspace");
    const response = await api.fetchApi("/upload/image", { method: "POST", body: formData });
    if (!response || (response.status !== 200 && response.status !== 201)) {
        throw new Error(response ? await response.text() : "上传裁剪后的基础图失败");
    }
    const data = await response.json();
    return data.subfolder ? `${data.subfolder}/${data.name} [input]` : `${data.name} [input]`;
};

const saveEditedImage = async ({ baseImagePath, maskDataUrl, paintDataUrl, fillMaskedImageWithGray }) => {
    const response = await api.fetchApi("/a_my_nodes/upload_custom_edited_image", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            image_path: baseImagePath,
            mask_data: maskDataUrl,
            paint_data: paintDataUrl,
            fill_masked_image_with_gray: fillMaskedImageWithGray
        })
    });
    if (response.status !== 200) throw new Error(await response.text());
    const data = await response.json();
    if (!data || !data.success || !data.filepath) {
        throw new Error((data && data.message) || "保存编辑图片失败");
    }
    return { path: `${data.filepath} [input]`, filepath: data.filepath };
};

export function showImageEditor(imagePaths, currentIndex, node, onSaveCallback) {
    const host = {
        rememberOriginalPaths(paths) {
            if (!node) return;
            if (!node.properties) node.properties = {};
            if (!node.properties.original_image_paths) node.properties.original_image_paths = [];
            for (let i = 0; i < paths.length; i++) {
                if (paths[i] && !paths[i].includes("clipspace-painted-masked-")) {
                    node.properties.original_image_paths[i] = paths[i];
                }
            }
        },
        getOriginalPath(index, fallback) {
            const originalPaths = node && node.properties ? node.properties.original_image_paths : null;
            if (originalPaths && originalPaths[index]) return originalPaths[index];
            return fallback;
        },
        rememberSavedBasePath(index, path) {
            if (!node) return;
            if (!node.properties) node.properties = {};
            if (!node.properties.original_image_paths) node.properties.original_image_paths = [];
            node.properties.original_image_paths[index] = path;
        },
        resolveImageUrl(path) {
            return buildViewUrl(path);
        },
        getSideLayerUrls,
        uploadBaseImage,
        saveEditedImage,
        notify(message, level = "error") {
            showTopNotification(message, level);
        },
        onSaved(payload) {
            if (onSaveCallback) onSaveCallback(payload.imagePaths);
        }
    };
    return showCoreImageEditor({ imagePaths, currentIndex, host });
}
