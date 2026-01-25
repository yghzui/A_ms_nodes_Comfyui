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
            }
        }

    RETURN_TYPES = (ANY_TYPE, ANY_TYPE, ANY_TYPE, ANY_TYPE, ANY_TYPE, ANY_TYPE, ANY_TYPE, ANY_TYPE)
    RETURN_NAMES = ("data_out", "data_out_1", "data_out_2", "data_out_3", "data_out_4", "data_out_5", "data_out_6", "data_out_7")
    # OUTPUT_IS_LIST = (True, True, True, True, True) # Disable list output to prevent early execution
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

    def accumulate(self, data, trybatch=True, batch_manager=None, **kwargs):
        inputs = self._collect_inputs(data, kwargs)
        
        # Non-batch mode (pass through)
        if batch_manager is None:
            # When not in batch mode, we return lists (as single objects)
            # The companion node will unpack them.
            return tuple(self._process_single(inputs[i], trybatch) for i in range(8))

        # Ensure reset on first step
        if batch_manager.current_index == 0:
            batch_manager.any_batch_results = [[] for _ in range(8)]

        if not hasattr(batch_manager, "any_batch_results") or not isinstance(batch_manager.any_batch_results, list) or len(batch_manager.any_batch_results) < 8:
            batch_manager.any_batch_results = [[] for _ in range(8)]

        for i, item in enumerate(inputs):
            if item is not None:
                batch_manager.any_batch_results[i].append(self._clone_value(item))
        batch_manager.current_index += 1
        print(f"AnyBatchAccumulator: Step {batch_manager.current_index}/{batch_manager.total_count}")

        if batch_manager.current_index < batch_manager.total_count:
            print("AnyBatchAccumulator: Triggering requeue and stopping current execution...")
            requeue_workflow_unchecked()
            # Return ExecutionBlocker directly (not wrapped in list) to stop downstream execution
            return tuple(ExecutionBlocker(None) for _ in range(8))
        
        print("AnyBatchAccumulator: Batch complete.")
        batch_manager.is_running = False
        raw_results = batch_manager.any_batch_results
        batch_manager.any_batch_results = [[] for _ in range(8)]
        
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
            
            if trybatch:
                batched = self._try_batch_results(raw)
                
                # Determine success
                is_success = False
                # If batched object is different from raw input list object, we assume some transformation happened.
                # However, _try_batch_results creates new lists for transpose, so 'is not' check is insufficient if it returns copy.
                # But our _try_batch_results returns 'results' (the input arg) on failure path.
                # So identity check 'is not' is valid, provided _try_batch_results doesn't return a copy on failure.
                # Looking at code: `return results` on failure. `results` is the argument.
                # So identity check is robust.
                if batched is not raw:
                    is_success = True
                
                candidate_results.append(batched)
                success_flags.append(is_success)
            else:
                candidate_results.append(raw)
                success_flags.append(True) # Not trying batch, so valid
        
        # 2. Synchronize: If ANY input failed batching, revert ALL to raw
        use_batch_results = True
        if trybatch:
            # Check only ports that actually had data (raw is not empty)
            # Actually success_flags handles empty ones as True.
            if not all(success_flags):
                print("AnyBatchAccumulator: One or more inputs failed batching. Reverting all to raw lists.")
                use_batch_results = False
        
        # 3. Finalize Output
        for i in range(8):
            if not raw_results[i]:
                processed_results.append([])
                continue
                
            if trybatch and use_batch_results:
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
                return []
            if isinstance(item, list):
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


class AnyDataAnalyzer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "data": (ANY_TYPE,),
            }
        }
    
    RETURN_TYPES = (ANY_TYPE,)
    RETURN_NAMES = ("data",)
    FUNCTION = "analyze"
    CATEGORY = "A_my_nodes/debug"
    OUTPUT_NODE = True

    def analyze(self, data):
        info = []
        is_list = isinstance(data, list)
        info.append(f"Is List: {is_list}")
        
        type_name = type(data).__name__
        info.append(f"Type: {type_name}")

        if is_list:
            info.append(f"Length: {len(data)}")
            if len(data) > 0:
                first_item_type = type(data[0]).__name__
                info.append(f"First Item Type: {first_item_type}")
                # Maybe string representation of first few items
                try:
                    preview = str(data[:3])
                    if len(preview) > 100:
                        preview = preview[:100] + "..."
                    info.append(f"Content Preview (first 3): {preview}")
                except:
                    pass
        else:
            try:
                preview = str(data)
                if len(preview) > 100:
                    preview = preview[:100] + "..."
                info.append(f"Content: {preview}")
            except:
                pass
        
        message = "\n".join(info)
        print(f"AnyDataAnalyzer: {message}")
        
        return {"ui": {"text": (message,)}, "result": (data,)}
