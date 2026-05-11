# -*- coding: utf-8 -*-
import random
import threading

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


class WorkflowForceRerunPassthrough:
    _state_lock = threading.Lock()
    _state = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "data": (ANY_TYPE, {"tooltip": "输入任意类型的数据，节点会原样透传输出。"}),
                "base_value": ("INT", {"default": 0, "min": -2147483648, "max": 2147483647, "step": 1, "tooltip": "变化数字的起点值。递增/递减模式会以它为初始值，随机模式会围绕它上下浮动。"}),
                "mode": (["increment", "decrement", "random"], {"tooltip": "数字变化模式：递增、递减或随机。"}),
                "step": ("INT", {"default": 1, "min": 1, "max": 2147483647, "step": 1, "tooltip": "递增或递减模式下，每次执行变化的步长。"}),
                "random_range": ("INT", {"default": 10, "min": 0, "max": 2147483647, "step": 1, "tooltip": "随机模式下，以 base_value 为中心，上下浮动的范围。"}),
                "reset_counter": ("BOOLEAN", {"default": False, "tooltip": "开启后，本次执行会重新从 base_value 开始计数或重新初始化随机状态。"}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = (ANY_TYPE, "INT")
    RETURN_NAMES = ("data", "current_value")
    FUNCTION = "passthrough"
    CATEGORY = "A_my_nodes/debug"
    DESCRIPTION = "强制节点每次执行都被视为已变化，同时透传任意输入数据，并输出一个每次运行都会更新的整数，便于测试工作流是否真的重新执行。"

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return float("NaN")

    @staticmethod
    def _make_state(base_value, mode, step, random_range):
        return {
            "initialized": False,
            "current": int(base_value),
            "last_base_value": int(base_value),
            "last_mode": mode,
            "last_step": int(step),
            "last_random_range": int(random_range),
        }

    @staticmethod
    def _config_changed(state, base_value, mode, step, random_range):
        return (
            state["last_base_value"] != int(base_value)
            or state["last_mode"] != mode
            or state["last_step"] != int(step)
            or state["last_random_range"] != int(random_range)
        )

    @staticmethod
    def _initial_value(base_value, mode, random_range):
        if mode == "random":
            low = int(base_value) - int(random_range)
            high = int(base_value) + int(random_range)
            return random.randint(low, high)
        return int(base_value)

    @staticmethod
    def _next_value(current_value, base_value, mode, step, random_range):
        if mode == "increment":
            return int(current_value) + int(step)

        if mode == "decrement":
            return int(current_value) - int(step)

        low = int(base_value) - int(random_range)
        high = int(base_value) + int(random_range)
        return random.randint(low, high)

    def passthrough(self, data, base_value, mode, step, random_range, reset_counter=False, unique_id=None):
        state_key = unique_id or f"workflow_force_rerun_{id(self)}"

        with self._state_lock:
            state = self._state.get(state_key)
            if state is None or reset_counter or self._config_changed(state, base_value, mode, step, random_range):
                state = self._make_state(base_value, mode, step, random_range)

            if not state["initialized"]:
                current_value = self._initial_value(base_value, mode, random_range)
                state["initialized"] = True
            else:
                current_value = self._next_value(state["current"], base_value, mode, step, random_range)

            state["current"] = int(current_value)
            state["last_base_value"] = int(base_value)
            state["last_mode"] = mode
            state["last_step"] = int(step)
            state["last_random_range"] = int(random_range)
            self._state[state_key] = state

        return (data, int(current_value))
