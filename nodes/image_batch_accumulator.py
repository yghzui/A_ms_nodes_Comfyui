import torch
import comfy.model_management
from .batch_utils import requeue_workflow_unchecked
try:
    from comfy_execution.graph import ExecutionBlocker
except ImportError:
    from comfy_execution.graph_utils import ExecutionBlocker

class ImageBatchAccumulator:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "batch_manager": ("MY_BATCH_MANAGER",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image_batch",)
    INPUT_IS_LIST = True
    FUNCTION = "accumulate"
    CATEGORY = "A_my_nodes/image"

    def accumulate(self, image, batch_manager=None):
        # image is a list of tensors [B, H, W, C]
        # batch_manager is a list [batch_manager_obj]
        
        bm = None
        if batch_manager is not None and len(batch_manager) > 0:
            bm = batch_manager[0]
        
        if bm is None:
            # If no batch manager is provided, just pass through the image list as a single batch
            # We need to concatenate the list into a single tensor
            try:
                if isinstance(image, list) and len(image) > 0:
                    return (torch.cat(image, dim=0),)
                return (image,)
            except:
                return (image,)

        # Ensure results is a list
        if not hasattr(bm, "results") or not isinstance(bm.results, list):
            bm.results = []
            
        # Add current image(s) to results
        if isinstance(image, list):
            for img in image:
                bm.results.append(img.clone())
        else:
            bm.results.append(image.clone())
        
        # Increment index
        bm.current_index += 1
        
        print(f"ImageBatchAccumulator: Step {bm.current_index}/{bm.total_count}")
        
        # Check if loop is finished
        if bm.current_index < bm.total_count:
             # Trigger requeue for the next step
             print("ImageBatchAccumulator: Triggering requeue and stopping current execution...")
             requeue_workflow_unchecked()
             
             # Return ExecutionBlocker to silently stop downstream nodes
             return (ExecutionBlocker(None),)
        
        # Loop finished
        print("ImageBatchAccumulator: Batch complete.")
        
        # Reset batch manager state
        bm.is_running = False
        
        # Concatenate and return final result
        try:
            # Concatenate all tensors in the list along the batch dimension (dim 0)
            final_batch = torch.cat(bm.results, dim=0)
            # Clear results to free memory
            bm.results = []
            return (final_batch,)
        except Exception as e:
            print(f"ImageBatchAccumulator Error: Failed to concatenate images. {e}")
            # Fallback to current image if concatenation fails
            return (image,)
