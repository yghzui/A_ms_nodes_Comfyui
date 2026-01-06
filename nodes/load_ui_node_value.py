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
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("value_str",)
    FUNCTION = "get_input_value"
    CATEGORY = "A_my_nodes/util"

    def get_input_value(self, captured_value, target_node_id="", target_input_name=""):
        # 直接返回前端捕获的值
        return (captured_value,)


