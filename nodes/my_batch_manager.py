
from .batch_utils import MyBatchManagerObj

class MyBatchManager:
    def __init__(self):
        self.batch_obj = MyBatchManagerObj()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "enable": ("BOOLEAN", {"default": True}),
                "start_index": ("INT", {"default": 0, "min": 0, "step": 1}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "unique_id": "UNIQUE_ID",
            }
        }

    RETURN_TYPES = ("MY_BATCH_MANAGER",)
    RETURN_NAMES = ("batch_manager",)
    FUNCTION = "create_batch"
    CATEGORY = "A_my_nodes/logic"

    def create_batch(self, enable=True, start_index=0, prompt=None, unique_id=None):
        if not enable:
            print("BatchManager: Disabled, returning None")
            return (None,)

        # 检查是否是重入队（循环中的）运行
        requeue = 0
        if prompt and unique_id:
            inputs = prompt[unique_id].get('inputs', {})
            requeue = inputs.get('requeue', 0)

        # 如果是第一次运行 (requeue=0)，重置状态
        if requeue == 0:
            self.batch_obj.reset()
            self.batch_obj.current_index = start_index
            self.batch_obj.is_running = True
            self.batch_obj.current_requeue_count = -1  # 初始化为-1，确保第一次执行时会被识别为新的批处理请求
            print(f"BatchManager: 初始化，起始索引 {start_index}")
        else:
            print(f"BatchManager: 循环运行中，当前索引 {self.batch_obj.current_index}")

        return (self.batch_obj,)
