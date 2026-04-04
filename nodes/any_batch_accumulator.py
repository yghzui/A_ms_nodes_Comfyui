import torch
import comfy.model_management
from .batch_utils import requeue_workflow_unchecked
try:
    from comfy_execution.graph import ExecutionBlocker
except ImportError:
    from comfy_execution.graph_utils import ExecutionBlocker
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

class AnyBatchAccumulator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "data": (ANY_TYPE,),
                "trybatch": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "batch_manager": ("MY_BATCH_MANAGER",),
            },
            "hidden": {
                "prompt": "PROMPT",
                "unique_id": "UNIQUE_ID",
            }
        }

    RETURN_TYPES = (ANY_TYPE, ANY_TYPE, ANY_TYPE, ANY_TYPE, ANY_TYPE, ANY_TYPE, ANY_TYPE, ANY_TYPE)
    RETURN_NAMES = ("data_out", "data_out_1", "data_out_2", "data_out_3", "data_out_4", "data_out_5", "data_out_6", "data_out_7")
    # OUTPUT_IS_LIST = (True, True, True, True, True) # Disable list output to prevent early execution
    INPUT_IS_LIST = True
    FUNCTION = "accumulate"
    CATEGORY = "A_my_nodes/logic"

    def _clone_value(self, value):
        if isinstance(value, torch.Tensor):
            return value.clone()
        if isinstance(value, dict) and "waveform" in value and "sample_rate" in value:
            waveform = value.get("waveform")
            if isinstance(waveform, torch.Tensor):
                return {"waveform": waveform.clone(), "sample_rate": value.get("sample_rate")}
        return value

    def _extract_mp4_files(self, data, out_list):
        if isinstance(data, dict):
            for k in data:
                self._extract_mp4_files(data[k], out_list)
        elif isinstance(data, list):
            for item in data:
                self._extract_mp4_files(item, out_list)
        elif isinstance(data, tuple):
            for item in data:
                self._extract_mp4_files(item, out_list)
        elif isinstance(data, str) and data.lower().endswith(".mp4"):
            out_list.append(data)

    def _looks_like_vhs_filenames(self, data):
        found = []
        self._extract_mp4_files(data, found)
        return len(found) > 0

    def _merge_vhs_results(self, results):
        merged = []
        for item in results:
            if isinstance(item, list):
                merged.extend(item)
            elif isinstance(item, tuple):
                merged.extend(list(item))
            else:
                merged.append(item)
        return merged

    def _try_batch_results(self, results):
        if not results:
            return results
            
        # 1. Handle VHS Filenames (Special Case: Merge nested lists)
        if all(self._looks_like_vhs_filenames(item) for item in results):
            return self._merge_vhs_results(results)

        # 2. Handle Tensor Batching
        if all(isinstance(item, torch.Tensor) for item in results):
            base_shape = results[0].shape[1:]
            if all(item.shape[1:] == base_shape for item in results):
                try:
                    return torch.cat(results, dim=0)
                except Exception:
                    return results
            return results

        # 3. Handle Audio (Waveform) Batching
        if all(isinstance(item, dict) and "waveform" in item and "sample_rate" in item for item in results):
            sample_rate = results[0].get("sample_rate")
            waveforms = []
            for item in results:
                if item.get("sample_rate") != sample_rate:
                    return results
                waveform = item.get("waveform")
                if not isinstance(waveform, torch.Tensor):
                    return results
                waveforms.append(waveform)
            base_shape = waveforms[0].shape[1:]
            if all(wf.shape[1:] == base_shape for wf in waveforms):
                try:
                    return {"waveform": torch.cat(waveforms, dim=0), "sample_rate": sample_rate}
                except Exception:
                    return results
            return results

        # 4. Handle String Joining
        if all(isinstance(item, str) for item in results):
            return "\n".join(results)
        
        # 5. Handle List of Lists (Transposed Batching)
        # If input is [[A1, A2], [B1, B2]], we try to produce [Batch(A1, B1), Batch(A2, B2)]
        if all(isinstance(x, list) for x in results):
            first_len = len(results[0])
            if all(len(x) == first_len for x in results):
                try:
                    # Transpose: zip(*results) -> [(A1, B1), (A2, B2)]
                    transposed = list(zip(*results))
                    batched_list = []
                    all_sub_success = True
                    for group in transposed:
                        # Try to batch the group items (convert tuple to list)
                        sub_res = self._try_batch_results(list(group))
                        # Simple check: if sub-result is essentially the same as input list, it failed
                        # But lists are new objects. We rely on logic:
                        # If _try_batch_results fails, it returns the input list.
                        # So if sub_res is list(group), then it likely failed.
                        # However, comparing lists is expensive. 
                        # We can assume if it returns a list of same length, it might have failed,
                        # UNLESS it was a nested list that got transposed-batched again.
                        batched_list.append(sub_res)
                    
                    return batched_list
                except Exception:
                    return results

        return results

    def _collect_inputs(self, data, kwargs):
        inputs = [None] * 8
        inputs[0] = data
        for i in range(1, 8):
            key = f"data_{i}"
            if key in kwargs:
                inputs[i] = kwargs[key]
        return inputs

    def _process_single(self, item, trybatch):
        if item is None:
            return []
        
        if isinstance(item, list):
            if trybatch:
                res = self._try_batch_results(item)
                return [res]
            return item # Return list as is

        if trybatch:
            res = self._try_batch_results([item])
            return [res]
        return [item]

    def accumulate(self, data, trybatch=None, batch_manager=None, prompt=None, unique_id=None, **kwargs):
        # Unwrap INPUT_IS_LIST parameters
        # ComfyUI passes single inputs as a list of 1 element when INPUT_IS_LIST=True
        
        # Handle batch_manager
        bm = None
        if batch_manager is not None and len(batch_manager) > 0:
            bm = batch_manager[0]
            
        # Handle trybatch (default True)
        do_batch = True
        if trybatch is not None and len(trybatch) > 0:
            do_batch = trybatch[0]
            
        inputs = self._collect_inputs(data, kwargs)
        
        # Non-batch mode (pass through)
        if bm is None:
            # When not in batch mode, we return lists (as single objects)
            # The companion node will unpack them.
            # With INPUT_IS_LIST=True, 'inputs' are lists of all items from upstream.
            # We process them as a single batch result.
            return tuple(self._process_single(inputs[i], do_batch) for i in range(8))

        # Ensure batch_manager has necessary attributes
        if not hasattr(bm, "any_batch_results") or not isinstance(bm.any_batch_results, list) or len(bm.any_batch_results) < 8:
            print(f"AnyBatchAccumulator: 重置any_batch_results")
            bm.any_batch_results = [[] for _ in range(8)]
            
        # 如果是新的运行的开始（index=0），清理旧结果
        if bm.current_index == 0:
            # 只有在确实有数据需要清理时才重置，避免不必要的对象创建
            if any(len(x) > 0 for x in bm.any_batch_results):
                print(f"AnyBatchAccumulator: 检测到新运行，清理旧结果")
                bm.any_batch_results = [[] for _ in range(8)]

        # 累积数据
        # inputs[i] is a list of items from upstream execution
        for i, item_list in enumerate(inputs):
            if item_list is not None:
                # Iterate over the list from upstream and append each item
                # ComfyUI's INPUT_IS_LIST passes the whole list of outputs from previous node
                # We treat this entire list as "one batch step's result"
                # If we want to flatten it, we extend.
                # If we want to keep structure, we append.
                # Based on previous logic, we append items one by one.
                if isinstance(item_list, list):
                    for val in item_list:
                        bm.any_batch_results[i].append(self._clone_value(val))
                else:
                    # Should not happen with INPUT_IS_LIST=True, but safe fallback
                    bm.any_batch_results[i].append(self._clone_value(item_list))
        
        # 增加批次索引
        bm.current_index += 1
        
        print(f"AnyBatchAccumulator: Processing batch {bm.current_index}/{bm.total_count}")

        # 检查是否还有更多批次
        if bm.current_index < bm.total_count:
            print(f"AnyBatchAccumulator: 批次处理完成，触发下一批次请求 ({bm.current_index}/{bm.total_count})")
            requeue_workflow_unchecked()
            return tuple(ExecutionBlocker(None) for _ in range(8))
            
        # 所有批次完成
        print("AnyBatchAccumulator: Batch complete.")
        bm.is_running = False
        raw_results = bm.any_batch_results
        # 重置结果以便下次使用
        bm.any_batch_results = [[] for _ in range(8)]

        
        # Process results with Baseline Synchronization
        processed_results = []
        
        # 1. Attempt batching for all inputs
        candidate_results = []
        success_flags = []
        
        for i in range(8):
            raw = raw_results[i]
            if not raw:
                candidate_results.append([])
                success_flags.append(True) # Empty is considered success (or irrelevant)
                continue
            
            if do_batch:
                batched = self._try_batch_results(raw)
                
                # Determine success
                is_success = False
                if batched is not raw:
                    is_success = True
                
                candidate_results.append(batched)
                success_flags.append(is_success)
            else:
                candidate_results.append(raw)
                success_flags.append(True) # Not trying batch, so valid
        
        # 2. Synchronize: If ANY input failed batching, revert ALL to raw
        use_batch_results = True
        if do_batch:
            # Check only ports that actually had data (raw is not empty)
            if not all(success_flags):
                print("AnyBatchAccumulator: One or more inputs failed batching. Reverting all to raw lists.")
                use_batch_results = False
        
        # 3. Finalize Output
        for i in range(8):
            if not raw_results[i]:
                processed_results.append([])
                continue
                
            if do_batch and use_batch_results:
                # Return the batched result directly. 
                # Since OUTPUT_IS_LIST is False, this object is passed as-is.
                processed_results.append(candidate_results[i])
            else:
                # Return raw list of lists
                processed_results.append(raw_results[i])

        return tuple(processed_results)


class AnyBatchListConverter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "data": (ANY_TYPE,),
                "data_1": (ANY_TYPE,),
                "data_2": (ANY_TYPE,),
                "data_3": (ANY_TYPE,),
                "data_4": (ANY_TYPE,),
                "data_5": (ANY_TYPE,),
                "data_6": (ANY_TYPE,),
                "data_7": (ANY_TYPE,),
            }
        }

    RETURN_TYPES = (ANY_TYPE, ANY_TYPE, ANY_TYPE, ANY_TYPE, ANY_TYPE, ANY_TYPE, ANY_TYPE, ANY_TYPE)
    RETURN_NAMES = ("data_out", "data_out_1", "data_out_2", "data_out_3", "data_out_4", "data_out_5", "data_out_6", "data_out_7")
    OUTPUT_IS_LIST = (True, True, True, True, True, True, True, True)
    FUNCTION = "convert"
    CATEGORY = "A_my_nodes/logic"

    def convert(self, data=None, data_1=None, data_2=None, data_3=None, data_4=None, data_5=None, data_6=None, data_7=None):
        inputs = [data, data_1, data_2, data_3, data_4, data_5, data_6, data_7]

        def flatten(item):
            if item is None:
                # print(f"AnyBatchListConverter: 输入为空,返回空列表")
                return []
            if isinstance(item, list):
                print(f"AnyBatchListConverter: 输入为列表,开始扁平化")
                flat = []
                for x in item:
                    # Recursively flatten or just one level?
                    # User: [[A,B], [C,D]] -> [A,B,C,D].
                    # [Batch1, Batch2] -> [Batch1, Batch2].
                    # If Batch1 is list? e.g. Transposed Batch returns [BatchA, BatchB].
                    # We want [BatchA, BatchB].
                    # If we recursively flatten, we might break structure if BatchA is a list (unlikely for Tensor/Image, but possible).
                    # But ComfyUI 'list' usually means "process these items one by one".
                    # So flattening everything is usually safe for "List Converter".
                    if isinstance(x, list):
                        flat.extend(flatten(x))
                    else:
                        flat.append(x)
                return flat
            return [item]

        return tuple(flatten(item) for item in inputs)

#
# import json
#
#
# class AnyDataAnalyzer:
#     @classmethod
#     def INPUT_TYPES(cls):
#         return {
#             "required": {
#                 "data": ("*",),
#             },
#             "hidden": {
#                 "unique_id": "UNIQUE_ID",
#                 "extra_pnginfo": "EXTRA_PNGINFO",
#             }
#         }
#
#     RETURN_TYPES = ("*",)
#     RETURN_NAMES = ("data",)
#     FUNCTION = "analyze"
#     CATEGORY = "A_my_nodes/debug"
#     OUTPUT_NODE = True
#
#     def analyze(self, data, unique_id=None, extra_pnginfo=None):
#         # ---------- 构建分析信息 ----------
#         info = []
#         info.append("=== AnyDataAnalyzer ===")
#         info.append(f"Python Type: {type(data).__name__}")
#         info.append(f"Is List: {isinstance(data, list)}")
#
#         # List 类型
#         if isinstance(data, list):
#             info.append(f"Length: {len(data)}")
#             if len(data) > 0:
#                 info.append(f"First Item Type: {type(data[0]).__name__}")
#                 try:
#                     preview = str(data[:3])
#                     if len(preview) > 120:
#                         preview = preview[:120] + "..."
#                     info.append(f"Preview: {preview}")
#                 except Exception as e:
#                     info.append(f"Preview Error: {e}")
#
#         # Tensor 类型
#         elif hasattr(data, "shape") and hasattr(data, "dtype"):
#             info.append(f"Shape: {tuple(data.shape)}")
#             info.append(f"Dtype: {data.dtype}")
#             if hasattr(data, "device"):
#                 info.append(f"Device: {data.device}")
#
#         # Dict 类型
#         elif isinstance(data, dict):
#             keys = list(data.keys())
#             info.append(f"Dict Keys: {keys[:10]}")
#
#         # 其他类型
#         else:
#             try:
#                 preview = str(data)
#                 if len(preview) > 120:
#                     preview = preview[:120] + "..."
#                 info.append(f"Content: {preview}")
#             except Exception as e:
#                 info.append(f"Preview Error: {e}")
#
#         message = "\n".join(info)
#         print(message)
#
#         # ---------- UI显示 ----------
#         values = [message]
#
#         # 强制写入前端节点控件，确保显示
#         try:
#             if extra_pnginfo and isinstance(extra_pnginfo, dict) and "workflow" in extra_pnginfo:
#                 workflow = extra_pnginfo["workflow"]
#                 node = next((x for x in workflow["nodes"] if str(x["id"]) == unique_id), None)
#                 if node:
#                     node["widgets_values"] = [values]
#         except Exception as e:
#             print(f"[AnyDataAnalyzer UI update failed] {e}")
#
#         # ---------- 返回 ----------
#         # 数据口透传原始 data，UI显示 message
#         return {"ui": {"text": values}, "result": (data,)}
#
#
