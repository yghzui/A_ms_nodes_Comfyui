
import server

# 简单的批处理管理器对象，用于在节点间传递状态
class MyBatchManagerObj:
    def __init__(self):
        self.current_index = 0
        self.total_count = 1
        self.results = []
        self.is_running = False
        # 跟踪当前的requeue计数，用于区分新的批处理请求和列表展开导致的多次执行
        self.current_requeue_count = -1
        self.current_list_index = 0

    def reset(self):
        self.current_index = 0
        self.results = []
        self.is_running = False
        self.current_requeue_count = -1
        self.current_list_index = 0

prompt_queue = server.PromptServer.instance.prompt_queue

def requeue_workflow_unchecked():
    """不检查直接重入队当前工作流"""
    currently_running = prompt_queue.currently_running
    if not currently_running:
        return
        
    value = next(iter(currently_running.values()))
    
    # 兼容 ComfyUI 不同版本的队列数据结构
    if len(value) == 6:
        (_, prompt_id, prompt, extra_data, outputs_to_execute, _) = value
    else:
        (_, prompt_id, prompt, extra_data, outputs_to_execute) = value
    
    # 深度复制 prompt 以便修改
    prompt = prompt.copy()
    
    # 找到 BatchManager 和 AnyBatchAccumulator 节点并增加 requeue 计数
    # 这让这些节点知道这是一次新的循环，而不是用户点击的新运行
    for uid in prompt:
        class_type = prompt[uid].get('class_type', '')
        if class_type == 'MyBatchManager' or class_type == 'AnyBatchAccumulator':
            inputs = prompt[uid].get('inputs', {})
            inputs['requeue'] = inputs.get('requeue', 0) + 1
            prompt[uid]['inputs'] = inputs

    # 生成新的任务 ID 并插入队列
    number = -server.PromptServer.instance.number
    server.PromptServer.instance.number += 1
    new_prompt_id = str(server.uuid.uuid4())
    
    # 保持原有结构重新入队
    sensitive = value[5] if len(value) > 5 else {}
    prompt_queue.put((number, new_prompt_id, prompt, extra_data, outputs_to_execute, sensitive))

def trigger_requeue():
    """触发重入队逻辑"""
    requeue_workflow_unchecked()
