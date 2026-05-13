import inspect

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


class _DynamicValueInputs:
    def __contains__(self, item):
        if not isinstance(item, str):
            return False
        if not item.startswith("value_"):
            return False
        try:
            index = int(item.split("_")[1])
        except (IndexError, ValueError):
            return False
        return 1 <= index <= MultiInputStateMapper.MAX_INPUTS

    def __getitem__(self, key):
        return (
            ANY_TYPE,
            {
                "tooltip": "动态输入值，支持连接 INT、FLOAT、BOOLEAN。仅前两个输入参与状态判断。",
            },
        )


class MultiInputStateMapper:
    MAX_INPUTS = 12

    @classmethod
    def INPUT_TYPES(cls):
        dynamic_inputs = {
            "value_1": (
                ANY_TYPE,
                {
                    "tooltip": "第 1 个动态输入。默认显示该输入口，连接后前端会继续显示下一个输入口。",
                },
            ),
        }

        stack = inspect.stack()
        if len(stack) > 2 and stack[2].function == "get_input_info":
            dynamic_inputs = _DynamicValueInputs()

        return {
            "required": {
                "math_expr": (
                    "STRING",
                    {
                        "default": "+0",
                        "tooltip": "输入如 +1, -2, *100, /2，对最终映射结果(1/2/3)应用该数学运算。",
                    },
                ),
            },
            "optional": dynamic_inputs,
        }

    RETURN_TYPES = ("INT", "FLOAT")
    RETURN_NAMES = ("result_int", "result_float")
    OUTPUT_TOOLTIPS = (
        "根据前两个输入的存在状态计算结果，并应用 math_expr 后输出的整数值。",
        "根据前两个输入的存在状态计算结果，并应用 math_expr 后输出的浮点值。",
    )
    FUNCTION = "map_state"
    CATEGORY = "A_my_nodes/logic"
    DESCRIPTION = (
        "动态接收最多 12 个 INT/FLOAT/BOOLEAN 输入。默认仅显示 1 个输入口，连接后逐步显示更多输入。"
        "仅前两个输入参与状态映射：第 1 个不存在输出 1；第 1 个存在且第 2 个不存在输出 2；"
        "前两个都存在输出 3。第 3 个及以后输入不影响结果。"
    )

    @staticmethod
    def _normalize_to_exists(value):
        if isinstance(value, bool):
            return value
        if isinstance(value, int):
            return value == 1
        if isinstance(value, float):
            return value == 1.0
        return False

    @staticmethod
    def _parse_math_operand(raw_text):
        text = raw_text.strip()
        if not text:
            return None

        try:
            number = float(text)
        except ValueError:
            return None

        return number

    def _apply_math_expr(self, base_value, expr):
        if not isinstance(expr, str):
            return float(base_value)

        expr = expr.strip()
        if not expr:
            return float(base_value)

        op = expr[0]
        if op not in {"+", "-", "*", "/"}:
            return float(base_value)

        operand = self._parse_math_operand(expr[1:])
        if operand is None:
            return float(base_value)

        result = float(base_value)
        if op == "+":
            return result + operand
        if op == "-":
            return result - operand
        if op == "*":
            return result * operand
        if operand == 0:
            return result
        return result / operand

    @classmethod
    def _get_ordered_input_names(cls):
        return [f"value_{index}" for index in range(1, cls.MAX_INPUTS + 1)]

    def _collect_values(self, kwargs):
        values = []
        for input_name in self._get_ordered_input_names():
            if input_name in kwargs:
                values.append(kwargs[input_name])
        return values

    def _calculate_base_result(self, values):
        if not values:
            return 1

        first_exists = self._normalize_to_exists(values[0])
        if not first_exists:
            return 1

        if len(values) == 1:
            return 2

        second_exists = self._normalize_to_exists(values[1])
        if not second_exists:
            return 2

        return 3

    def map_state(self, math_expr="+0", **kwargs):
        values = self._collect_values(kwargs)
        base_result = self._calculate_base_result(values)
        final_float = self._apply_math_expr(base_result, math_expr)
        final_int = int(final_float)
        return (final_int, float(final_float))
