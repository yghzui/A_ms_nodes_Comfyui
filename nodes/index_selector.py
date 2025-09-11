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

    # 固定声明 64 个 BOOLEAN 输出，保证后端与校验阶段始终有足够的输出槽
    RETURN_TYPES = (
        "BOOLEAN","BOOLEAN","BOOLEAN","BOOLEAN","BOOLEAN","BOOLEAN","BOOLEAN","BOOLEAN",
        "BOOLEAN","BOOLEAN","BOOLEAN","BOOLEAN","BOOLEAN","BOOLEAN","BOOLEAN","BOOLEAN",
        "BOOLEAN","BOOLEAN","BOOLEAN","BOOLEAN","BOOLEAN","BOOLEAN","BOOLEAN","BOOLEAN",
        "BOOLEAN","BOOLEAN","BOOLEAN","BOOLEAN","BOOLEAN","BOOLEAN","BOOLEAN","BOOLEAN",
        "BOOLEAN","BOOLEAN","BOOLEAN","BOOLEAN","BOOLEAN","BOOLEAN","BOOLEAN","BOOLEAN",
        "BOOLEAN","BOOLEAN","BOOLEAN","BOOLEAN","BOOLEAN","BOOLEAN","BOOLEAN","BOOLEAN",
        "BOOLEAN","BOOLEAN","BOOLEAN","BOOLEAN","BOOLEAN","BOOLEAN","BOOLEAN","BOOLEAN",
        "BOOLEAN","BOOLEAN","BOOLEAN","BOOLEAN","BOOLEAN","BOOLEAN","BOOLEAN","BOOLEAN",
    )
    RETURN_NAMES = (
        "output_0","output_1","output_2","output_3","output_4","output_5","output_6","output_7",
        "output_8","output_9","output_10","output_11","output_12","output_13","output_14","output_15",
        "output_16","output_17","output_18","output_19","output_20","output_21","output_22","output_23",
        "output_24","output_25","output_26","output_27","output_28","output_29","output_30","output_31",
        "output_32","output_33","output_34","output_35","output_36","output_37","output_38","output_39",
        "output_40","output_41","output_42","output_43","output_44","output_45","output_46","output_47",
        "output_48","output_49","output_50","output_51","output_52","output_53","output_54","output_55",
        "output_56","output_57","output_58","output_59","output_60","output_61","output_62","output_63",
    )

    FUNCTION = "select"

    CATEGORY = "A_my_nodes/utils"

    def select(self, n, index):
        # 输出固定 64 个布尔值：只有选中的 index 为 True，其余为 False
        # 之所以固定 64，是为了与 RETURN_TYPES/RETURN_NAMES 对齐，彻底避免校验阶段“输出数量不匹配”问题
        outputs = [False] * 64

        # 安全地设置索引，index 超界时不抛错，保持全 False
        if 0 <= index < 64:
            outputs[index] = True

        # 返回与 RETURN_TYPES 声明同长度的扁平元组（BOOLEAN × 64）
        return tuple(outputs)
