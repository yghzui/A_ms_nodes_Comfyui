import inspect
import logging

try:
    from comfy.comfy_types.node_typing import IO
    ANY_TYPE = IO.ANY
except ImportError:
    try:
        from comfy_extras.nodes_custom_sampler import AnyType
        ANY_TYPE = AnyType("*")
    except ImportError:
        class AnyType(str):
            def __ne__(self, __value: object) -> bool:
                return False

        ANY_TYPE = AnyType("*")


class GroupSwitchAny:
    MAX_OUTPUTS = 8

    @classmethod
    def INPUT_TYPES(cls):
        dyn_inputs = {
            "input_1_1": (
                ANY_TYPE,
                {
                    "lazy": True,
                    "tooltip": "第 1 组第 1 个输入。连接后前端会继续扩展同组或下一组输入槽。",
                },
            ),
        }

        stack = inspect.stack()
        if len(stack) > 2 and stack[2].function == "get_input_info":
            # 允许前端动态添加 input_{group}_{slot} 形式的输入名。
            class AllContainer:
                def __contains__(self, item):
                    return True

                def __getitem__(self, key):
                    return ANY_TYPE, {"lazy": True}

            dyn_inputs = AllContainer()

        return {
            "required": {
                "select": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 999999,
                        "step": 1,
                        "tooltip": "选择输出第几组输入。",
                    },
                ),
                "group_size": (
                    "INT",
                    {
                        "default": 2,
                        "min": 1,
                        "max": cls.MAX_OUTPUTS,
                        "step": 1,
                        "tooltip": "每组输入的元素个数。节点会输出前 n 个结果，其余固定输出补 None。",
                    },
                ),
            },
            "optional": dyn_inputs,
        }

    RETURN_TYPES = (ANY_TYPE, ANY_TYPE, ANY_TYPE, ANY_TYPE, ANY_TYPE, ANY_TYPE, ANY_TYPE, ANY_TYPE)
    RETURN_NAMES = ("out1", "out2", "out3", "out4", "out5", "out6", "out7", "out8")
    OUTPUT_TOOLTIPS = (
        "选中组的第 1 个输出；如果该组没有该位置，则输出 None。",
        "选中组的第 2 个输出；如果该组没有该位置，则输出 None。",
        "选中组的第 3 个输出；如果该组没有该位置，则输出 None。",
        "选中组的第 4 个输出；如果该组没有该位置，则输出 None。",
        "选中组的第 5 个输出；如果该组没有该位置，则输出 None。",
        "选中组的第 6 个输出；如果该组没有该位置，则输出 None。",
        "选中组的第 7 个输出；如果该组没有该位置，则输出 None。",
        "选中组的第 8 个输出；如果该组没有该位置，则输出 None。",
    )
    FUNCTION = "switch_group"
    CATEGORY = "A_my_nodes/logic"
    DESCRIPTION = (
        "按组切换任意类型输入。前端动态扩展输入，后端固定保留 5 个输出，"
        "被选中组的前 n 个值会映射到 out1~outn，其余输出补 None。"
    )

    @classmethod
    def _get_selected_input_names(cls, selected_index, group_size):
        safe_group_size = max(1, min(int(group_size), cls.MAX_OUTPUTS))
        return [f"input_{selected_index}_{slot_index}" for slot_index in range(1, safe_group_size + 1)]

    def check_lazy_status(self, *args, **kwargs):
        selected_index = max(1, int(kwargs.get("select", 1)))
        group_size = kwargs.get("group_size", self.MAX_OUTPUTS)
        selected_inputs = self._get_selected_input_names(selected_index, group_size)

        logging.info(f"GROUP_SWITCH_SELECTED: group={selected_index}, inputs={selected_inputs}")

        return [input_name for input_name in selected_inputs if input_name in kwargs]

    def switch_group(self, *args, **kwargs):
        selected_index = max(1, int(kwargs.get("select", 1)))
        group_size = kwargs.get("group_size", self.MAX_OUTPUTS)
        selected_inputs = self._get_selected_input_names(selected_index, group_size)

        outputs = []
        for input_name in selected_inputs:
            outputs.append(kwargs.get(input_name, None))

        while len(outputs) < self.MAX_OUTPUTS:
            outputs.append(None)

        return tuple(outputs[: self.MAX_OUTPUTS])
