import json
import os

from .load_lora_batch import LoadLoraBatch
from .lora_common import LoraCondition


class LoraApply:
    @classmethod
    def INPUT_TYPES(cls):
        logic = LoraCondition.condition_inputs()
        return {
            "required": {"model": ("MODEL",), **logic},
            "optional": {
                "loras_info": ("STRING", {"default": "[]", "hidden": True}),
            },
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "dict_input")
    FUNCTION = "process"
    CATEGORY = "A_my_nodes/loaders"

    def process(self, model, enable_mode, dict_input, key_to_check, check_mode, loras_info="[]"):
        active, _ = LoraCondition().resolve_enable_mode(enable_mode, dict_input, key_to_check, check_mode)

        if active:
            data = json.loads(loras_info)
            loras = data.get("loras", []) if isinstance(data, dict) else data
            model, = LoadLoraBatch().load_loras(
                model, **{f"LORA_{index}": value for index, value in enumerate(loras)}
            )
        return {"ui": {"text": ["True" if active else "False"]}, "result": (model, dict_input)}


class LoraApplyAsset:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "global_enable": (["Auto", "True", "False"], {"default": "Auto"}),
                "dict_input": ("STRING", {"default": "{}"}),
                "enable_all_in_group": (["False", "True"], {"default": "False"}),
            },
            "hidden": {
                "selected_assets": ("STRING", {"default": "[]"}),
                "current_group": ("STRING", {"default": "All"}),
            },
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "dict_input")
    FUNCTION = "process"
    CATEGORY = "A_my_nodes/loaders"

    @staticmethod
    def _get_assets_db_path():
        return os.path.join(os.path.dirname(os.path.dirname(__file__)), "models_db.json")

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        try:
            stat = os.stat(cls._get_assets_db_path())
            return f"{stat.st_mtime_ns}:{stat.st_size}"
        except FileNotFoundError:
            return "missing"

    def process(self, model, global_enable, dict_input, enable_all_in_group="False",
                selected_assets="[]", current_group="All"):
        hits = []
        if global_enable == "False":
            return {"ui": {"text": [], "hit_status": hits}, "result": (model, dict_input)}

        path = self._get_assets_db_path()
        if not os.path.exists(path):
            return {"ui": {"text": [], "hit_status": hits}, "result": (model, dict_input)}
        with open(path, "r", encoding="utf-8") as source:
            groups = json.load(source).get("groups", [])
        selected = json.loads(selected_assets)
        selected = selected if isinstance(selected, list) else []
        single_items = {
            item["id"]: item for group in groups for item in group.get("items", [])
            if item.get("id") and (item.get("stream_type") == "single" or
                                     ("loras" in item and item.get("stream_type") != "double"))
        }
        selection_modes = {item["id"]: item.get("enable_mode", "Auto") for item in selected}
        group = next((group for group in groups if group.get("name") == current_group), None)
        if enable_all_in_group == "True" and current_group != "All" and group is not None:
            items = [single_items[item["id"]] for item in group.get("items", [])
                     if item.get("id") in single_items]
        else:
            items = [single_items[item["id"]] for item in selected if item.get("id") in single_items]

        processor = LoraApply()
        for index, item in enumerate(items):
            mode = "True" if global_enable == "True" else selection_modes.get(item["id"], "Auto")
            output = processor.process(
                model, mode, dict_input, item.get("keyword", ""), item.get("check_mode", "contains"),
                json.dumps(item.get("loras", [])),
            )
            model, _ = output["result"]
            hits.append({"id": item["id"], "index": index, "hit": output["ui"]["text"][0] == "True"})
        return {"ui": {"text": ["Processed"], "hit_status": hits}, "result": (model, dict_input)}
