import winsound
import time
import threading

class NoticeSound:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "repeat_times": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
            },
            "optional": {
                "image": ("IMAGE",),
                "latent": ("LATENT",),
                "text": ("STRING",),
                "any_type": (None,),
            }
        }

    RETURN_TYPES = tuple(["IMAGE", "LATENT", "STRING", None])
    RETURN_NAMES = tuple(["image", "latent", "text", "any_type"])
    FUNCTION = "play_notice_sound"
    CATEGORY = "My_node/通知"

    def play_sound(self, repeat_times=1):
        # 在Windows上播放系统声音，每次持续1秒
        try:
            for _ in range(repeat_times):
                # 使用系统默认提示音
                winsound.PlaySound('SystemExclamation', winsound.SND_ASYNC)
                time.sleep(1)  # 每次提示音持续1秒
            # 停止声音
            winsound.PlaySound(None, 0)
        except Exception as e:
            print(f"播放声音时出错: {str(e)}")

    def play_notice_sound(self, repeat_times=1, **kwargs):
        # 创建一个线程来播放声音，这样不会阻塞主线程
        sound_thread = threading.Thread(target=self.play_sound, args=(repeat_times,))
        sound_thread.start()
        
        # 初始化返回值
        return_values = [None] * len(self.RETURN_TYPES)
        
        # 填充返回值
        for i, key in enumerate(self.RETURN_NAMES):
            if key in kwargs and kwargs[key] is not None:
                return_values[i] = kwargs[key]
        
        return tuple(return_values)

# 这个节点可以用于导出
NODE_CLASS_MAPPINGS = {
    "NoticeSound": NoticeSound
}

# 类别描述
NODE_DISPLAY_NAME_MAPPINGS = {
    "NoticeSound": "铃声提醒"
}
