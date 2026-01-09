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
    FUNCTION = "accumulate"
    CATEGORY = "A_my_nodes/image"

    def accumulate(self, image, batch_manager=None):
        # image shape is [B, H, W, C]
        
        if batch_manager is None:
            # If no batch manager is provided, just pass through the image
            return (image,)

        # Ensure results is a list
        if not hasattr(batch_manager, "results") or not isinstance(batch_manager.results, list):
            batch_manager.results = []
            
        # Add current image to results
        # Clone to avoid reference issues
        batch_manager.results.append(image.clone())
        
        # Increment index
        batch_manager.current_index += 1
        
        print(f"ImageBatchAccumulator: Step {batch_manager.current_index}/{batch_manager.total_count}")
        
        # Check if loop is finished
        if batch_manager.current_index < batch_manager.total_count:
             # Trigger requeue for the next step
             print("ImageBatchAccumulator: Triggering requeue and stopping current execution...")
             requeue_workflow_unchecked()
             
             # Return ExecutionBlocker to silently stop downstream nodes
             # This prevents errors in nodes like SaveImage that don't handle empty lists
             return (ExecutionBlocker(None),)
        
        # Loop finished
        print("ImageBatchAccumulator: Batch complete.")
        
        # Reset batch manager state
        batch_manager.is_running = False
        
        # Concatenate and return final result
        try:
            # Concatenate all tensors in the list along the batch dimension (dim 0)
            final_batch = torch.cat(batch_manager.results, dim=0)
            # Clear results to free memory
            batch_manager.results = []
            return (final_batch,)
        except Exception as e:
            print(f"ImageBatchAccumulator Error: Failed to concatenate images. {e}")
            # Fallback to current image if concatenation fails (e.g. dimension mismatch)
            # But since we are finished, we must return something.
            return (image,)
