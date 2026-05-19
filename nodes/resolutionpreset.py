import json
import os

BUILTIN_RESOLUTION_PRESETS = {
    "512x768": {"w": 512, "h": 768},
    "1024x1440": {"w": 1024, "h": 1440},
    "1280x1980": {"w": 1280, "h": 1980},
}

RESOLUTION_PRESET_LIMITS = {
    "min": 64,
    "max": 8192,
    "default_step": 8,
    "max_name_length": 80,
}


def get_resolution_presets_file_path():
    base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    return os.path.join(base_dir, "resolution_presets.json")


def get_builtin_resolution_presets():
    return {
        name: (item["w"], item["h"])
        for name, item in BUILTIN_RESOLUTION_PRESETS.items()
    }


def serialize_builtin_resolution_presets():
    return {
        name: {"w": int(item["w"]), "h": int(item["h"])}
        for name, item in BUILTIN_RESOLUTION_PRESETS.items()
    }


def normalize_preset_name(name):
    text = str(name or "").strip()
    if not text:
        return ""
    return text[: RESOLUTION_PRESET_LIMITS["max_name_length"]]


def _parse_positive_int(value, field_name="value"):
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer")
    if isinstance(value, int):
        number = value
    elif isinstance(value, float):
        if not value.is_integer():
            raise ValueError(f"{field_name} must be an integer")
        number = int(value)
    else:
        text = str(value or "").strip()
        if not text:
            raise ValueError(f"{field_name} is required")
        if text.startswith("+"):
            text = text[1:]
        if not text.isdigit():
            raise ValueError(f"{field_name} must be an integer")
        number = int(text)
    if number <= 0:
        raise ValueError(f"{field_name} must be greater than 0")
    return number


def normalize_step_value(value):
    return _parse_positive_int(
        value if value is not None else RESOLUTION_PRESET_LIMITS["default_step"],
        field_name="step",
    )


def normalize_resolution_value(value, step=None, field_name="value"):
    limits = RESOLUTION_PRESET_LIMITS
    number = _parse_positive_int(value, field_name=field_name)
    step_value = normalize_step_value(step)
    number = max(limits["min"], min(limits["max"], number))
    if step_value > 1:
        number = ((number + (step_value // 2)) // step_value) * step_value
        number = max(limits["min"], min(limits["max"], number))
    return int(number)


def sanitize_resolution_preset_item(item, step=None):
    if not isinstance(item, dict):
        raise ValueError("preset item must be an object")
    return {
        "w": normalize_resolution_value(item.get("w"), step=step, field_name="width"),
        "h": normalize_resolution_value(item.get("h"), step=step, field_name="height"),
        "choose": bool(item.get("choose", False)),
    }


def sanitize_resolution_presets(data, step=None):
    if not isinstance(data, dict):
        return {}
    builtin_names = set(BUILTIN_RESOLUTION_PRESETS.keys())
    result = {}
    default_name = ""
    for raw_name, item in data.items():
        name = normalize_preset_name(raw_name)
        if not name or name in builtin_names:
            continue
        try:
            normalized = sanitize_resolution_preset_item(item, step=step)
        except Exception:
            continue
        if normalized["choose"] and not default_name:
            default_name = name
        normalized["choose"] = False
        result[name] = normalized
    if default_name and default_name in result:
        result[default_name]["choose"] = True
    return result


def load_resolution_presets_from_file():
    file_path = get_resolution_presets_file_path()
    if not os.path.isfile(file_path):
        return {}
    try:
        with open(file_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        return {}
    return sanitize_resolution_presets(data)


def save_resolution_presets_to_file(presets):
    file_path = get_resolution_presets_file_path()
    data = sanitize_resolution_presets(presets)
    with open(file_path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
    return data


def find_default_custom_preset_name(custom_presets):
    for name, item in custom_presets.items():
        if item.get("choose"):
            return name
    return ""


def parse_custom_presets_text(custom_presets_text):
    if not custom_presets_text:
        return {}
    try:
        data = json.loads(custom_presets_text)
    except Exception:
        return {}
    return sanitize_resolution_presets(data)


class ResolutionPresetNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preset": (
                    "COMBO",
                    {
                        "options": list(BUILTIN_RESOLUTION_PRESETS.keys()),
                        "default": "512x768",
                        "tooltip": "选择内置或自定义分辨率预设。前端节点面板负责完整的预设管理与保存。",
                    },
                ),
                "mirror": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "启用后交换宽高输出，方便在同一预设下切换横版和竖版。",
                    },
                ),
                "custom_presets": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "前端维护的自定义预设 JSON 缓存，通常无需手动编辑。",
                    },
                ),
            }
        }

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    FUNCTION = "get_resolution"
    CATEGORY = "A_my_nodes/math"
    DESCRIPTION = "通过内置或自定义分辨率预设输出宽高，支持默认预设、镜像输出和前端可视化管理。"

    def _get_all_presets(self, custom_presets_text):
        custom_from_file = load_resolution_presets_from_file()
        custom_presets = custom_from_file or parse_custom_presets_text(custom_presets_text)
        preset_map = get_builtin_resolution_presets()
        preset_map.update({
            name: (item["w"], item["h"])
            for name, item in custom_presets.items()
        })
        return preset_map, custom_presets

    def get_resolution(self, preset, mirror, custom_presets):
        preset_map, custom_items = self._get_all_presets(custom_presets)
        selected = str(preset or "")
        if selected not in preset_map:
            selected = find_default_custom_preset_name(custom_items)
        if selected not in preset_map and preset_map:
            selected = next(iter(preset_map.keys()))
        if selected not in preset_map:
            return (0, 0)
        width, height = preset_map[selected]
        if bool(mirror):
            width, height = height, width
        return (int(width), int(height))
