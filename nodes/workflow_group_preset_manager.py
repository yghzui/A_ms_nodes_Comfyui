import json

from .workflow_group_store import (
    find_group_by_name,
    load_workflow_groups_from_payload,
)


def _is_link_value(value):
    return isinstance(value, list) and len(value) == 2


def _iter_value_entries(item):
    if not isinstance(item, dict):
        return []
    values = item.get("values", [])
    if isinstance(values, list):
        return [value_entry for value_entry in values if isinstance(value_entry, dict)]
    if item.get("target_input_name"):
        return [item]
    return []


class WorkflowGroupPresetManager:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "group_name": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "当前激活的组名。前端会把它增强为可刷新的分组选择器。",
                    },
                ),
                "auto_apply": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "开启后，后端会在 prompt 入队前根据当前组批量改写目标节点输入值。",
                    },
                ),
                "apply_scope": (
                    ["values_only", "values_and_state"],
                    {
                        "tooltip": "第 1 期仅实现 values_only。选择 values_and_state 时会自动回退为 values_only。",
                    },
                ),
                "fallback_mode": (
                    ["warn_missing", "ignore_missing", "error_missing"],
                    {
                        "tooltip": "缺失节点、缺失输入、类型不匹配等异常的处理方式。",
                    },
                ),
                "sync_ui_preview": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "是否允许前端“应用当前组到画布”同步更新 UI。该项主要供前端使用。",
                    },
                ),
                "groups_payload": (
                    "STRING",
                    {
                        "default": "{\"version\":1,\"groups\":[]}",
                        "multiline": True,
                        "tooltip": "当前管理节点私有的分组数据，前端 DOM 管理器会自动维护该 JSON。",
                    },
                ),
            },
            "hidden": {
                "prompt": "PROMPT",
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "INT", "INT", "STRING", "STRING")
    RETURN_NAMES = (
        "group_name",
        "group_id",
        "group_version",
        "item_count",
        "status_text",
        "group_payload",
    )
    OUTPUT_TOOLTIPS = (
        "当前激活组名称。",
        "当前激活组的唯一 ID。",
        "当前激活组版本号，可作为缓存与调试参考。",
        "当前激活组的启用条目数量。",
        "状态文本，包含可应用项、缺失项与第 1 期能力边界说明。",
        "当前激活组的 JSON 字符串，便于调试或传给其它节点。",
    )
    FUNCTION = "describe_group"
    CATEGORY = "A_my_nodes/workflow"
    DESCRIPTION = (
        "管理工作流切换组的第 1 期控制节点。它负责选择激活组、输出组信息，"
        "并配合后端 on_prompt 预处理在 UI/API 入队前批量改写可序列化 widget 值。"
    )

    @classmethod
    def IS_CHANGED(cls, group_name="", auto_apply=True, apply_scope="values_only", fallback_mode="warn_missing", sync_ui_preview=True, groups_payload="", prompt=None):
        return (
            str(group_name),
            bool(auto_apply),
            str(apply_scope),
            str(fallback_mode),
            bool(sync_ui_preview),
            str(groups_payload),
        )

    @staticmethod
    def _serialize_group(group):
        try:
            return json.dumps(group, ensure_ascii=False, indent=2)
        except TypeError:
            return json.dumps({}, ensure_ascii=False)

    @staticmethod
    def _build_missing_summary(group, prompt):
        if not isinstance(prompt, dict):
            return 0, 0

        applicable_count = 0
        missing_count = 0
        for item in group.get("items", []):
            if not item.get("enabled", True):
                continue

            node_id = str(item.get("target_node_id", "")).strip()
            target_node = prompt.get(node_id)
            value_entries = [entry for entry in _iter_value_entries(item) if entry.get("enabled", True)]
            if not isinstance(target_node, dict):
                missing_count += len(value_entries) or 1
                continue

            inputs = target_node.get("inputs", {})
            for value_entry in value_entries:
                input_name = str(value_entry.get("target_input_name", "")).strip()
                if input_name not in inputs or _is_link_value(inputs.get(input_name)):
                    missing_count += 1
                    continue
                applicable_count += 1

        return applicable_count, missing_count

    @staticmethod
    def _count_enabled_values(group):
        count = 0
        for item in group.get("items", []):
            if not item.get("enabled", True):
                continue
            count += len([entry for entry in _iter_value_entries(item) if entry.get("enabled", True)])
        return count

    def describe_group(self, group_name, auto_apply, apply_scope, fallback_mode, sync_ui_preview, groups_payload, prompt=None):
        workflow_groups_db = load_workflow_groups_from_payload(groups_payload)
        group = find_group_by_name(workflow_groups_db, group_name)

        if group is None:
            status_text = "未找到激活组，请先在前端创建组并选择 group_name。"
            return ("", "", 0, 0, status_text, "{}")

        enabled_items = [item for item in group.get("items", []) if item.get("enabled", True)]
        enabled_value_count = self._count_enabled_values(group)
        applicable_count, missing_count = self._build_missing_summary(group, prompt)
        group_version = int(group.get("version", 1))

        scope_text = "values_only" if apply_scope != "values_and_state" else "values_and_state(第1期按values_only处理)"
        status_text = (
            f"组 {group['name']} 已加载，启用节点条目 {len(enabled_items)} 个，参数值 {enabled_value_count} 个，可应用 {applicable_count} 个，"
            f"缺失/跳过 {missing_count} 个，auto_apply={bool(auto_apply)}，scope={scope_text}，"
            f"fallback={fallback_mode}。"
        )

        return (
            group["name"],
            group["id"],
            group_version,
            len(enabled_items),
            status_text,
            self._serialize_group(group),
        )
