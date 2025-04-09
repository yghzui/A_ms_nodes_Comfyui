import winsound
import time
import threading
import ctypes
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

class NoticeSound:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "repeat_times": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
                "check_mute": ("BOOLEAN", {"default": False}),
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

    def get_volume_control(self):
        """获取系统音量控制接口"""
        try:
            devices = AudioUtilities.GetSpeakers()
            interface = devices.Activate(
                IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            volume = cast(interface, POINTER(IAudioEndpointVolume))
            return volume
        except Exception as e:
            print(f"获取音量控制接口失败: {str(e)}")
            return None

    def play_sound(self, repeat_times=1, check_mute=False):
        # 在Windows上播放系统声音，每次持续1秒
        volume_control = None
        original_mute_state = False
        
        try:
            # 如果需要检查静音状态
            if check_mute:
                volume_control = self.get_volume_control()
                if volume_control:
                    # 获取当前静音状态
                    original_mute_state = volume_control.GetMute()
                    # 如果是静音，则临时取消静音
                    if original_mute_state:
                        print("系统当前为静音状态，临时取消静音播放提示音")
                        volume_control.SetMute(0, None)
            
            # 播放提示音
            for _ in range(repeat_times):
                # 使用系统默认提示音
                winsound.PlaySound('SystemExclamation', winsound.SND_ASYNC)
                time.sleep(1)  # 每次提示音持续1秒
            
            # 停止声音
            winsound.PlaySound(None, 0)
            
            # 恢复原来的静音状态
            if check_mute and volume_control and original_mute_state:
                print("恢复静音状态")
                time.sleep(0.5)  # 等待一小段时间确保声音播放完毕
                volume_control.SetMute(1, None)
                
        except Exception as e:
            print(f"播放声音时出错: {str(e)}")
            # 发生错误时也要尝试恢复静音状态
            if check_mute and volume_control and original_mute_state:
                try:
                    volume_control.SetMute(1, None)
                except:
                    pass

    def play_notice_sound(self, repeat_times=1, check_mute=False, **kwargs):
        # 创建一个线程来播放声音，这样不会阻塞主线程
        sound_thread = threading.Thread(target=self.play_sound, args=(repeat_times, check_mute))
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
