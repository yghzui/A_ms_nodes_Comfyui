import os
import json

def get_resolution_presets_file_path():
    # 获取当前文件 (nodes/resolutionpreset.py) 的上级目录 (nodes) 的上级目录 (A_my_nodes)
    base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    return os.path.join(base_dir, "resolution_presets.json")

class ResolutionPresetNode:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preset": (
                    "COMBO",
                    {
                        "options": ["512x768", "1024x1440", "1280x1980"],
                        "default": "512x768",
                    },
                ),
                "mirror": ("BOOLEAN", {"default": False}),
                "custom_presets": ("STRING", {"default": "", "multiline": False}),
            }
        }

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    FUNCTION = "get_resolution"
    CATEGORY = "A_my_nodes/math"

    def _get_builtin_presets(self):
        return {
            "512x768": (512, 768),
            "1024x1440": (1024, 1440),
            "1280x1980": (1280, 1980),
        }

    def _parse_custom_presets(self, custom_presets):
        data = {}
        try:
            file_path = get_resolution_presets_file_path()
            if os.path.isfile(file_path):
                with open(file_path, "r", encoding="utf-8") as f:
                    file_data = json.load(f)
                if isinstance(file_data, dict):
                    data = file_data
        except Exception:
            data = {}
        if not data and custom_presets:
            try:
                text_data = json.loads(custom_presets)
                if isinstance(text_data, dict):
                    data = text_data
            except Exception:
                data = {}
        if not isinstance(data, dict):
            return {}
        result = {}
        for key, item in data.items():
            if not isinstance(item, dict):
                continue
            w = item.get("w")
            h = item.get("h")
            try:
                w = int(w)
                h = int(h)
            except Exception:
                continue
            if w <= 0 or h <= 0:
                continue
            result[str(key)] = (w, h)
        return result

    def get_resolution(self, preset, mirror, custom_presets):
        presets = self._get_builtin_presets()
        custom = self._parse_custom_presets(custom_presets)
        presets.update(custom)
        if preset not in presets:
            if presets:
                preset = next(iter(presets.keys()))
            else:
                return (0, 0)
        width, height = presets[preset]
        if bool(mirror):
            width, height = height, width
        return (int(width), int(height))