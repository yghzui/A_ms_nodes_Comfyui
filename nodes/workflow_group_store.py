import copy
import json
import os
import threading
import uuid


DB_FILENAME = "workflow_groups_db.json"
DB_VERSION = 1
SUPPORTED_VALUE_TYPES = (
    "INT",
    "FLOAT",
    "BOOLEAN",
    "STRING",
    "COMBO",
    "JSON_STRING",
)

DEFAULT_DB = {
    "version": DB_VERSION,
    "groups": [],
}

SCHEMA = {
    "version": DB_VERSION,
    "supported_value_types": list(SUPPORTED_VALUE_TYPES),
    "group_fields": {
        "id": "STRING",
        "name": "STRING",
        "enabled": "BOOLEAN",
        "version": "INT",
        "items": "ARRAY",
    },
    "item_fields": {
        "id": "STRING",
        "target_node_id": "STRING",
        "target_node_title": "STRING",
        "target_class_type": "STRING",
        "apply_mode": "STRING",
        "node_state": "STRING",
        "enabled": "BOOLEAN",
        "note": "STRING",
        "values": "ARRAY",
    },
    "value_fields": {
        "id": "STRING",
        "target_input_name": "STRING",
        "value_type": "STRING",
        "value": "ANY_JSON",
        "enabled": "BOOLEAN",
        "note": "STRING",
    },
}

_STORE_LOCK = threading.Lock()


def get_workflow_groups_db_path():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, DB_FILENAME)


def _deepcopy_default_db():
    return copy.deepcopy(DEFAULT_DB)


def _safe_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_bool(value, default=False):
    if isinstance(value, bool):
        return value
    if value in (1, "1", "true", "True", "yes", "on"):
        return True
    if value in (0, "0", "false", "False", "no", "off"):
        return False
    return default


def _safe_string(value):
    if value is None:
        return ""
    return str(value)


def make_group_id():
    return f"group_{uuid.uuid4().hex[:12]}"


def make_item_id():
    return f"item_{uuid.uuid4().hex[:12]}"


def make_value_id():
    return f"value_{uuid.uuid4().hex[:12]}"


def normalize_value_entry(entry, index=0):
    source = entry if isinstance(entry, dict) else {}
    value_type = _safe_string(source.get("value_type", "STRING")).upper() or "STRING"
    if value_type not in SUPPORTED_VALUE_TYPES:
        value_type = "STRING"

    return {
        "id": _safe_string(source.get("id")) or make_value_id(),
        "target_input_name": _safe_string(source.get("target_input_name")),
        "value_type": value_type,
        "value": source.get("value"),
        "enabled": _safe_bool(source.get("enabled"), True),
        "note": _safe_string(source.get("note")),
        "_order": index,
    }


def _normalize_item_values(source):
    raw_values = source.get("values", [])
    if isinstance(raw_values, list) and raw_values:
        normalized_values = [
            normalize_value_entry(value_entry, value_index)
            for value_index, value_entry in enumerate(raw_values)
        ]
    elif source.get("target_input_name"):
        normalized_values = [normalize_value_entry(source, 0)]
    else:
        normalized_values = []

    for value_entry in normalized_values:
        value_entry.pop("_order", None)
    return normalized_values


def normalize_item(item, index=0):
    source = item if isinstance(item, dict) else {}
    normalized_values = _normalize_item_values(source)
    return {
        "id": _safe_string(source.get("id")) or make_item_id(),
        "target_node_id": _safe_string(source.get("target_node_id")),
        "target_node_title": _safe_string(source.get("target_node_title")),
        "target_class_type": _safe_string(source.get("target_class_type")),
        "apply_mode": _safe_string(source.get("apply_mode")) or "set_widget_values",
        "node_state": _safe_string(source.get("node_state")) or "normal",
        "enabled": _safe_bool(source.get("enabled"), True),
        "note": _safe_string(source.get("note")),
        "values": normalized_values,
        "_order": index,
    }


def normalize_group(group, index=0):
    source = group if isinstance(group, dict) else {}
    raw_items = source.get("items", [])
    items = raw_items if isinstance(raw_items, list) else []
    normalized_items = [normalize_item(item, item_index) for item_index, item in enumerate(items)]
    for item in normalized_items:
        item.pop("_order", None)

    return {
        "id": _safe_string(source.get("id")) or make_group_id(),
        "name": _safe_string(source.get("name")) or f"未命名组_{index + 1}",
        "enabled": _safe_bool(source.get("enabled"), True),
        "version": max(1, _safe_int(source.get("version", 1), 1)),
        "items": normalized_items,
    }


def normalize_workflow_groups_db(data):
    source = data if isinstance(data, dict) else {}
    raw_groups = source.get("groups", [])
    groups = raw_groups if isinstance(raw_groups, list) else []
    normalized_groups = [normalize_group(group, index) for index, group in enumerate(groups)]
    return {
        "version": DB_VERSION,
        "groups": normalized_groups,
    }


def load_workflow_groups_from_payload(payload):
    if isinstance(payload, dict):
        return normalize_workflow_groups_db(payload)
    if isinstance(payload, str):
        text = payload.strip()
        if not text:
            return _deepcopy_default_db()
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return _deepcopy_default_db()
        return normalize_workflow_groups_db(data)
    return _deepcopy_default_db()


def ensure_workflow_groups_db_file():
    db_path = get_workflow_groups_db_path()
    if os.path.exists(db_path):
        return db_path

    with _STORE_LOCK:
        if os.path.exists(db_path):
            return db_path
        with open(db_path, "w", encoding="utf-8") as file:
            json.dump(_deepcopy_default_db(), file, ensure_ascii=False, indent=2)
    return db_path


def load_workflow_groups_db():
    db_path = ensure_workflow_groups_db_file()
    try:
        with open(db_path, "r", encoding="utf-8") as file:
            data = json.load(file)
    except (json.JSONDecodeError, OSError):
        data = _deepcopy_default_db()
    return normalize_workflow_groups_db(data)


def save_workflow_groups_db(data):
    normalized = normalize_workflow_groups_db(data)
    db_path = ensure_workflow_groups_db_file()
    with _STORE_LOCK:
        with open(db_path, "w", encoding="utf-8") as file:
            json.dump(normalized, file, ensure_ascii=False, indent=2)
    return normalized


def get_workflow_groups_schema():
    payload = copy.deepcopy(SCHEMA)
    payload["db_path"] = get_workflow_groups_db_path()
    return payload


def find_group_by_name(data, group_name):
    safe_name = _safe_string(group_name).strip()
    if not safe_name:
        return None

    normalized = normalize_workflow_groups_db(data)
    for group in normalized["groups"]:
        if group["name"] == safe_name:
            return group
    return None


def get_db_fingerprint():
    db_path = ensure_workflow_groups_db_file()
    try:
        stat_result = os.stat(db_path)
    except OSError:
        return "workflow_groups_db_missing"
    return f"{int(stat_result.st_mtime_ns)}:{stat_result.st_size}"
