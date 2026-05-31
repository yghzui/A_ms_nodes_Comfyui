import json
import re
import comfy.sd
import comfy.utils
import nodes
import os
import folder_paths
from .wan_video_double_stream import WanVideoDoubleStream

class WanVideoDoubleStreamAsset:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "global_enable": (["Auto", "True", "False"], {"default": "Auto"}),
                "dict_input": ("STRING", {"default": "{}", "tooltip": "输入包含标题、内容、启用状态的字典，或直接输入字符串"}),
                "enable_all_in_group": (["False", "True"], {"default": "False", "tooltip": "如果开启，将始终强制应用当前选中组内的所有条目，忽略单独的勾选"}),
                "fix_low_mem": (["Auto", "True", "False"], {"default": "Auto", "tooltip": "统一修正 High/Low 两路的 Low Mem 设置，仅影响当前节点运行时的 WAN LoRA 收集结果，不修改资产管理器中的原始设置"}),
                "fix_merge_loras": (["Auto", "True", "False"], {"default": "Auto", "tooltip": "统一修正 High/Low 两路的 Merge 设置，仅影响当前节点运行时的 WAN LoRA 收集结果，不修改资产管理器中的原始设置"}),
            },
            "optional": {
                "model_high": ("MODEL",),
                "model_low": ("MODEL",),
                "prev_lora_high": ("WANVIDLORA",),
                "prev_lora_low": ("WANVIDLORA",),
                "blocks_high": ("SELECTEDBLOCKS",),
                "blocks_low": ("SELECTEDBLOCKS",),
            },
            "hidden": {
                "selected_assets": ("STRING", {"default": "[]"}),
                "current_group": ("STRING", {"default": "All"}),
            }
        }

    RETURN_TYPES = ("MODEL", "MODEL", "WANVIDLORA", "WANVIDLORA", "STRING")
    RETURN_NAMES = ("model_high", "model_low", "prev_lora_high", "prev_lora_low", "dict_input")
    FUNCTION = "process"
    CATEGORY = "A_my_nodes/video"

    @staticmethod
    def _get_assets_db_path():
        my_nodes_dir = os.path.dirname(os.path.dirname(__file__))
        return os.path.join(my_nodes_dir, "models_db.json")

    @classmethod
    def IS_CHANGED(cls, global_enable="Auto", dict_input="{}", enable_all_in_group="False", fix_low_mem="Auto",
                   fix_merge_loras="Auto", selected_assets="[]", current_group="All", **kwargs):
        assets_path = cls._get_assets_db_path()
        assets_signature = "missing"

        try:
            stat = os.stat(assets_path)
            assets_signature = f"{stat.st_mtime_ns}:{stat.st_size}"
        except OSError:
            pass

        return (
            str(global_enable),
            str(dict_input),
            str(enable_all_in_group),
            str(fix_low_mem),
            str(fix_merge_loras),
            str(selected_assets),
            str(current_group),
            assets_signature,
        )

    def _get_assets_from_manager(self):
        try:
            import urllib.request
            # 这是一个本地请求，如果 ComfyUI 运行在非 8188 端口或者有鉴权，可能需要调整。
            # 更稳妥的做法是直接读取文件。资产管理器的数据保存在用户目录下或自定义节点下。
            # 根据 common pattern, A_my_nodes 的资产数据通常保存在 A_my_nodes/assets/models.json 或类似的地方。
            # 这里我们通过导入后端的 routes.py 或读取已知文件路径来获取最新数据。
            
            # 假设存储路径为 custom_nodes/A_my_nodes/models_db.json
            import os
            import folder_paths
            assets_path = self._get_assets_db_path()
            if os.path.exists(assets_path):
                with open(assets_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                print(f"[WanVideoDoubleStreamAsset] Models assets file not found at {assets_path}")
                return None
        except Exception as e:
            print(f"[WanVideoDoubleStreamAsset] Failed to read models data directly: {e}")
            return None

    def _normalize_stream_settings(self, settings):
        if not isinstance(settings, dict):
            settings = {}

        return {
            "value1": bool(settings.get("value1", False)),
            "value2": bool(settings.get("value2", False)),
            "value3": bool(settings.get("value3", False)),
        }

    def _apply_runtime_stream_settings_fix(self, settings, fix_low_mem="Auto", fix_merge_loras="Auto"):
        fixed_settings = self._normalize_stream_settings(settings)

        if fix_low_mem in ("True", "False"):
            fixed_settings["value2"] = fix_low_mem == "True"

        if fix_merge_loras in ("True", "False"):
            fixed_settings["value3"] = fix_merge_loras == "True"

        if not fixed_settings["value3"]:
            fixed_settings["value2"] = False

        return fixed_settings

    def _build_loras_info(self, loras, settings):
        return json.dumps({
            "loras": loras if isinstance(loras, list) else [],
            "settings": self._normalize_stream_settings(settings),
        })

    def process(self, global_enable, dict_input, enable_all_in_group="False", fix_low_mem="Auto", fix_merge_loras="Auto",
                selected_assets="[]", current_group="All",
                model_high=None, model_low=None, 
                prev_lora_high=None, prev_lora_low=None,
                blocks_high=None, blocks_low=None):
        
        # 1. 检查全局开关
        if global_enable == "False":
            return {
                "ui": {"text": [], "hit_status": []},
                "result": (model_high, model_low, prev_lora_high, prev_lora_low, dict_input)
            }

        # 获取后端最新资产数据
        latest_assets_data = self._get_assets_from_manager()
        all_items_map = {}
        group_items_map = {} # {groupName: [item, ...]}
        
        if latest_assets_data and "groups" in latest_assets_data:
            for group in latest_assets_data["groups"]:
                g_name = group.get("name", "Unknown")
                group_items_map[g_name] = []
                if "items" in group:
                    for item in group["items"]:
                        item_id = item.get("id")
                        if item_id:
                            all_items_map[item_id] = item
                            group_items_map[g_name].append(item)

        # 2. 解析前端传来的选中状态
        try:
            front_selected = json.loads(selected_assets)
            if not isinstance(front_selected, list):
                front_selected = []
        except Exception as e:
            print(f"[WanVideoDoubleStreamAsset] Failed to parse selected_assets: {e}")
            front_selected = []

        assets_to_process = []
        
        # 如果开启了"启动选中组的所有条目"
        if enable_all_in_group == "True" and current_group != "All" and current_group in group_items_map:
            # 始终应用当前选中组内的所有条目
            group_items = group_items_map[current_group]
            for item in group_items:
                # 看看前端是否为它设置了特殊的 enable_mode，如果没有则默认 Auto
                front_item = next((f for f in front_selected if f.get("id") == item["id"]), None)
                enable_mode = front_item.get("enable_mode", "Auto") if front_item else "Auto"
                
                asset_copy = item.copy()
                asset_copy["enable_mode"] = enable_mode
                assets_to_process.append(asset_copy)
        else:
            # 仅处理前端单独勾选的条目（按前端排序的顺序）
            for f_item in front_selected:
                item_id = f_item.get("id")
                if item_id and item_id in all_items_map:
                    # 使用后端最新的配置数据，但保留前端的启用状态设置
                    asset_copy = all_items_map[item_id].copy()
                    asset_copy["enable_mode"] = f_item.get("enable_mode", "Auto")
                    assets_to_process.append(asset_copy)

        if not assets_to_process:
             return {
                "ui": {"text": [], "hit_status": []},
                "result": (model_high, model_low, prev_lora_high, prev_lora_low, dict_input)
            }

        # 3. 实例化基础节点处理类
        base_processor = WanVideoDoubleStream()
        
        current_model_high = model_high
        current_model_low = model_low
        current_prev_lora_high = prev_lora_high
        current_prev_lora_low = prev_lora_low
        
        hit_status_list = []

        # 4. 按顺序遍历处理每一个资产
        for idx, asset in enumerate(assets_to_process):
            # asset 结构类似于 modelsData.groups[x].items[y]
            # 但是前端可以附加一个 enable_mode 字段 (Auto, True, False)
            item_enable_mode = asset.get("enable_mode", "Auto")
            
            # 如果全局是 True, 强制覆盖所有 Auto 为 True? 
            # 逻辑: 
            # 全局 Auto -> 尊重条目自身的 enable_mode
            # 全局 True -> 强制所有条目为 True
            if global_enable == "True":
                item_enable_mode = "True"
                
            key_to_check = asset.get("keyword", "")
            check_mode = asset.get("check_mode", "contains")
            
            # 构建 loras_info
            high_loras = asset.get("high_loras", [])
            low_loras = asset.get("low_loras", [])
            high_settings = self._apply_runtime_stream_settings_fix(
                asset.get("high_settings", {}),
                fix_low_mem=fix_low_mem,
                fix_merge_loras=fix_merge_loras,
            )
            low_settings = self._apply_runtime_stream_settings_fix(
                asset.get("low_settings", {}),
                fix_low_mem=fix_low_mem,
                fix_merge_loras=fix_merge_loras,
            )

            loras_info_high = self._build_loras_info(high_loras, high_settings)
            loras_info_low = self._build_loras_info(low_loras, low_settings)

            # 确保传入的是字典结构给 base_processor，或者是列表
            
            # 调用基础逻辑
            # 注意: base_processor 返回的是 dict，形如 {"ui": {"text": ["True"/"False"]}, "result": (m_h, m_l, p_h, p_l, d_i)}
            out = base_processor.process(
                enable_mode=item_enable_mode,
                dict_input=dict_input,
                key_to_check=key_to_check,
                check_mode=check_mode,
                model_high=current_model_high,
                model_low=current_model_low,
                prev_lora_high=current_prev_lora_high,
                prev_lora_low=current_prev_lora_low,
                blocks_high=blocks_high if blocks_high is not None else {},
                blocks_low=blocks_low if blocks_low is not None else {},
                loras_info_high=loras_info_high,
                loras_info_low=loras_info_low
            )
            
            ui_status = out["ui"]["text"][0] if "ui" in out and "text" in out["ui"] and out["ui"]["text"] else "False"
            hit_status_list.append({
                "id": asset.get("id"),
                "index": idx,
                "hit": ui_status == "True"
            })
            
            res = out["result"]
            current_model_high = res[0]
            current_model_low = res[1]
            current_prev_lora_high = res[2]
            current_prev_lora_low = res[3]
            # dict_input 不变

        # 5. 返回累加后的结果和 UI 状态
        return {
            "ui": {
                "text": ["Processed"],
                "hit_status": hit_status_list
            },
            "result": (current_model_high, current_model_low, current_prev_lora_high, current_prev_lora_low, dict_input)
        }
