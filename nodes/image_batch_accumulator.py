import torch
from .batch_utils import requeue_workflow_unchecked

class ImageBatchAccumulator:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "batch_manager": ("MY_BATCH_MANAGER",),
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image_batch",)
    FUNCTION = "accumulate"
    CATEGORY = "A_my_nodes/image"

    def accumulate(self, batch_manager, image):
        # image shape is [B, H, W, C]
        
        # Ensure results is a list
        if not hasattr(batch_manager, "results") or not isinstance(batch_manager.results, list):
            batch_manager.results = []
            
        # Add current image to results
        # Clone to avoid reference issues if upstream modifies it (though unlikely in Comfy)
        batch_manager.results.append(image.clone())
        
        # Increment index
        batch_manager.current_index += 1
        
        print(f"ImageBatchAccumulator: Step {batch_manager.current_index}/{batch_manager.total_count}")
        
        # Check if loop is finished
        if batch_manager.current_index < batch_manager.total_count:
             # Trigger requeue
             print("ImageBatchAccumulator: Triggering requeue...")
             requeue_workflow_unchecked()
        else:
             # Loop finished
             print("ImageBatchAccumulator: Batch complete.")
             # Reset batch manager state
             batch_manager.is_running = False
             
        # Always return the accumulated batch so far
        # This ensures that even in intermediate steps, we have a valid output
        if not batch_manager.results:
            # Should not happen given we just appended, but safety check
            return (image,)
            
        try:
            # Concatenate all tensors in the list along the batch dimension (dim 0)
            final_batch = torch.cat(batch_manager.results, dim=0)
            return (final_batch,)
        except Exception as e:
            print(f"ImageBatchAccumulator Error: Failed to concatenate images. {e}")
            # Fallback to current image if concatenation fails (e.g. dimension mismatch)
            return (image,)
