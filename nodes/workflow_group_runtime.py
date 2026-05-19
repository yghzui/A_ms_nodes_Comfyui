import copy
import json

from .workflow_group_store import find_group_by_name, load_workflow_groups_from_payload


MANAGER_NODE_CLASS = "WorkflowGroupPresetManager"
LINK_VALUE_LENGTH = 2


def _is_link_value(value):
    return isinstance(value, list) and len(value) == LINK_VALUE_LENGTH


def _coerce_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"无法转换为 BOOLEAN: {value}")


def _coerce_value(item):
    value_type = str(item.get("value_type", "STRING") or "STRING").upper()
    value = item.get("value")

    if value_type == "INT":
        return int(value)
    if value_type == "FLOAT":
        return float(value)
    if value_type == "BOOLEAN":
        return _coerce_bool(value)
    if value_type in {"STRING", "COMBO"}:
        return "" if value is None else str(value)
    if value_type == "JSON_STRING":
        if isinstance(value, str):
            json.loads(value)
            return value
        return json.dumps(value, ensure_ascii=False)
    return value


def _message_by_mode(mode, text):
    if mode == "error_missing":
        raise ValueError(text)
    if mode == "warn_missing":
        return text
    return None


def _iter_value_entries(item):
    if not isinstance(item, dict):
        return []
    values = item.get("values", [])
    if isinstance(values, list):
        return [value_entry for value_entry in values if isinstance(value_entry, dict)]
    if item.get("target_input_name"):
        return [item]
    return []


def _apply_group_to_prompt(prompt, group, fallback_mode):
    report = {
        "group_name": group["name"],
        "group_id": group["id"],
        "group_version": int(group.get("version", 1)),
        "applied_count": 0,
        "missing_count": 0,
        "warnings": [],
    }

    for item in group.get("items", []):
        if not item.get("enabled", True):
            continue

        target_node_id = str(item.get("target_node_id", "")).strip()
        value_entries = _iter_value_entries(item)
        if not target_node_id or not value_entries:
            item_label = f"{group['name']}::{target_node_id or 'unknown_node'}"
            report["missing_count"] += 1
            warning = _message_by_mode(fallback_mode, f"[WorkflowGroupPreset] 节点条目不完整: {item_label}")
            if warning:
                report["warnings"].append(warning)
            continue

        target_node = prompt.get(target_node_id)
        if not isinstance(target_node, dict):
            report["missing_count"] += len(value_entries)
            warning = _message_by_mode(
                fallback_mode,
                f"[WorkflowGroupPreset] 目标节点不存在: {group['name']}::{target_node_id}"
            )
            if warning:
                report["warnings"].append(warning)
            continue

        inputs = target_node.setdefault("inputs", {})

        for value_entry in value_entries:
            if not value_entry.get("enabled", True):
                continue

            target_input_name = str(value_entry.get("target_input_name", "")).strip()
            item_label = f"{group['name']}::{target_node_id}.{target_input_name}"
            if not target_input_name:
                report["missing_count"] += 1
                warning = _message_by_mode(fallback_mode, f"[WorkflowGroupPreset] 组项目标不完整: {item_label}")
                if warning:
                    report["warnings"].append(warning)
                continue

            if target_input_name not in inputs:
                report["missing_count"] += 1
                warning = _message_by_mode(fallback_mode, f"[WorkflowGroupPreset] 目标输入不存在: {item_label}")
                if warning:
                    report["warnings"].append(warning)
                continue

            current_value = inputs.get(target_input_name)
            if _is_link_value(current_value):
                report["missing_count"] += 1
                warning = _message_by_mode(
                    fallback_mode,
                    f"[WorkflowGroupPreset] 目标输入是连线值，MVP 不覆盖连线输入: {item_label}",
                )
                if warning:
                    report["warnings"].append(warning)
                continue

            try:
                inputs[target_input_name] = _coerce_value(value_entry)
                report["applied_count"] += 1
            except Exception as exc:
                report["missing_count"] += 1
                warning = _message_by_mode(
                    fallback_mode,
                    f"[WorkflowGroupPreset] 值写入失败 {item_label}: {exc}",
                )
                if warning:
                    report["warnings"].append(warning)

    return report


def apply_workflow_groups_to_prompt_payload(json_data):
    if not isinstance(json_data, dict):
        return json_data

    prompt = json_data.get("prompt")
    if not isinstance(prompt, dict):
        return json_data

    reports = []
    prompt_changed = False

    sorted_nodes = sorted(prompt.items(), key=lambda item: str(item[0]))
    for node_id, node_data in sorted_nodes:
        if not isinstance(node_data, dict):
            continue
        if node_data.get("class_type") != MANAGER_NODE_CLASS:
            continue

        inputs = node_data.get("inputs", {})
        if not isinstance(inputs, dict):
            continue

        auto_apply = bool(inputs.get("auto_apply", True))
        group_name = str(inputs.get("group_name", "") or "").strip()
        fallback_mode = str(inputs.get("fallback_mode", "warn_missing") or "warn_missing").strip()
        apply_scope = str(inputs.get("apply_scope", "values_only") or "values_only").strip()
        groups_payload = inputs.get("groups_payload", "")
        workflow_groups_db = load_workflow_groups_from_payload(groups_payload)

        report = {
            "manager_node_id": str(node_id),
            "group_name": group_name,
            "group_id": "",
            "group_version": 0,
            "applied_count": 0,
            "missing_count": 0,
            "warnings": [],
            "auto_apply": auto_apply,
            "apply_scope": apply_scope,
        }

        if not auto_apply:
            reports.append(report)
            continue

        if apply_scope != "values_only":
            report["warnings"].append(
                "[WorkflowGroupPreset] 当前仅实现 values_only，values_and_state 会按 values_only 处理。"
            )

        if not group_name:
            report["warnings"].append("[WorkflowGroupPreset] 未选择 group_name，跳过应用。")
            reports.append(report)
            continue

        group = find_group_by_name(workflow_groups_db, group_name)
        if group is None:
            report["warnings"].append(f"[WorkflowGroupPreset] 组不存在: {group_name}")
            if fallback_mode == "error_missing":
                raise ValueError(f"[WorkflowGroupPreset] 组不存在: {group_name}")
            reports.append(report)
            continue

        applied_report = _apply_group_to_prompt(prompt, group, fallback_mode)
        report.update(applied_report)
        prompt_changed = prompt_changed or report["applied_count"] > 0
        reports.append(report)

    if reports:
        payload = copy.deepcopy(json_data)
        payload["prompt"] = prompt
        payload.setdefault("extra_data", {})
        payload["extra_data"]["a_my_nodes_workflow_groups_report"] = reports
        payload["extra_data"]["a_my_nodes_workflow_groups_changed"] = prompt_changed
        return payload

    return json_data
