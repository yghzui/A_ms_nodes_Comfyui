import torch

class IndexSelector:
    """
    A node that takes a total number 'n' and an 'index', and outputs a list of 'n' booleans.
    The boolean at the specified 'index' will be True, all others will be False.
    The frontend script will handle the dynamic creation of output slots.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "n": ("INT", {"default": 1, "min": 1, "max": 64, "step": 1}),
                "index": ("INT", {"default": 0, "min": 0, "max": 63, "step": 1}),
            }
        }

    # The node returns a single list containing the boolean values.
    # The JS frontend is responsible for creating individual output slots from this list.
    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("output_0",)
    
    FUNCTION = "select"
    CATEGORY = "A_my_nodes/utils"

    def select(self, n, index):
        """
        Generates a list of booleans with only one True value at the given index.
        """
        # 创建一个包含n个False的列表
        outputs = [False] * n
        # 如果index在有效范围内，则将对应位置的元素设置为True
        if 0 <= index < n:
            outputs[index] = True
        
        # 将列表转换为元组，因为ComfyUI的每个输出都需要一个单独的返回值
        return tuple(outputs)
