# -*- coding: utf-8 -*-
# 节点功能：获取工作流中任意节点的输入选项值，返回对应字符串
import json


class GetNodeInputValue:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                # 目标节点ID（仅用于前端逻辑记录，后端不处理）
                # 注意：这里定义为 STRING 而不是 COMBO，是为了允许前端动态设置值而不触发后端验证错误
                # 前端 JS 会将其转换为下拉框
                "target_node_id": ("STRING", {"default": ""}),
                # 目标输入选项名（仅用于前端逻辑记录，后端不处理）
                "target_input_name": ("STRING", {"default": ""}),
            },
            "required": {
                # 捕获的值（由前端自动填充）
                "captured_value": ("STRING", {"default": "", "multiline": True}),
            },
            "hidden": {
                "prompt": "PROMPT",
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("value_str",)
    FUNCTION = "get_input_value"
    CATEGORY = "A_my_nodes/util"

    DESCRIPTION = "优先根据当前执行 prompt 中的 target_node_id 和 target_input_name 读取真实输入值，避免 API 模式下继续使用前端缓存值。"

    @staticmethod
    def _serialize_value(value):
        if isinstance(value, str):
            return value
        if value is None:
            return ""
        try:
            return json.dumps(value, ensure_ascii=False)
        except TypeError:
            return str(value)

    def get_input_value(self, captured_value, target_node_id="", target_input_name="", prompt=None):
        if prompt and target_node_id and target_input_name:
            target_node = prompt.get(str(target_node_id)) or prompt.get(target_node_id)
            if isinstance(target_node, dict):
                inputs = target_node.get("inputs", {})
                if target_input_name in inputs:
                    real_value = inputs[target_input_name]
                    print(f"获取到真实值: {real_value},捕获值: {captured_value}")
                    return (self._serialize_value(real_value),)
        print(f"未找到目标节点 {target_node_id} 或输入选项 {target_input_name}")
        print(f"捕获值: {captured_value}")
        return (captured_value,)


