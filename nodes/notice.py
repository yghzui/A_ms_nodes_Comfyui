import winsound
import time
import threading
import ctypes
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import hashlib

# Import AnyType for accepting any input type
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

class NoticeSound:
    @classmethod
    def INPUT_TYPES(cls):
         return {
            "required": {
                "repeat_times": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
                "check_mute": ("BOOLEAN", {"default": False,'tooltip':'是否检查静音状态,容易因为com释放问题导致崩溃'}),
                "max_volume": ("BOOLEAN", {"default": False,'tooltip':'是否将音量调到最大,容易因为com释放问题导致崩溃'}),
                "async_play": ("BOOLEAN", {"default": True, "tooltip": "是否异步播放(True=不阻塞, False=阻塞等待播放结束)"}),
            },
            "optional": {
                "any_input": (ANY_TYPE, {"tooltip": "输入任意类型的数据用于透传"}),
            }
        }
    OUTPUT_NODE = True
    RETURN_TYPES = (ANY_TYPE,)
    RETURN_NAMES = ("output",)
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

    def play_sound(self, repeat_times=1, check_mute=False, max_volume=False):
        # 在Windows上播放系统声音，每次持续1秒
        volume_control = None
        original_mute_state = False
        original_volume = 0.0
        
        try:
            # 获取音量控制接口（如果需要检查静音或调整音量）
            if check_mute or max_volume:
                volume_control = self.get_volume_control()
                
            if volume_control:
                # 处理静音状态
                if check_mute:
                    # 获取当前静音状态
                    original_mute_state = volume_control.GetMute()
                    # 如果是静音，则临时取消静音
                    if original_mute_state:
                        print("系统当前为静音状态，临时取消静音播放提示音")
                        volume_control.SetMute(0, None)
                
                # 处理音量调整
                if max_volume:
                    # 保存原始音量
                    original_volume = volume_control.GetMasterVolumeLevelScalar()
                    print(f"保存原始音量级别: {original_volume:.2f}")
                    
                    # 将音量调到最大
                    if original_volume < 1.0:
                        print("临时将音量调整到最大")
                        volume_control.SetMasterVolumeLevelScalar(1.0, None)
            
            # 播放提示音
            for _ in range(repeat_times):
                # 使用系统默认提示音
                winsound.PlaySound('SystemExclamation', winsound.SND_ASYNC)
                time.sleep(1)  # 每次提示音持续1秒
            
            # 停止声音
            winsound.PlaySound(None, 0)
            time.sleep(0.5)  # 等待一小段时间确保声音播放完毕
            
            # 恢复原始音量（如果之前调整过）
            if max_volume and volume_control and original_volume < 1.0:
                print(f"恢复原始音量级别: {original_volume:.2f}")
                volume_control.SetMasterVolumeLevelScalar(original_volume, None)
            
            # 恢复原来的静音状态
            if check_mute and volume_control and original_mute_state:
                print("恢复静音状态")
                volume_control.SetMute(1, None)
                
        except Exception as e:
            print(f"播放声音时出错: {str(e)}")
            
            # 发生错误时尝试恢复设置
            if volume_control:
                # 恢复音量
                if max_volume and original_volume < 1.0:
                    try:
                        volume_control.SetMasterVolumeLevelScalar(original_volume, None)
                    except:
                        pass
                
                # 恢复静音状态
                if check_mute and original_mute_state:
                    try:
                        volume_control.SetMute(1, None)
                    except:
                        pass

    def play_notice_sound(self, repeat_times=1, check_mute=False, max_volume=False, async_play=True, any_input=None, **kwargs):
        if async_play:
            # 创建一个线程来播放声音，这样不会阻塞主线程
            sound_thread = threading.Thread(target=self.play_sound, args=(repeat_times, check_mute, max_volume))
            sound_thread.start()
        else:
            # 同步播放，会阻塞主线程直到播放完成
            self.play_sound(repeat_times, check_mute, max_volume)
        
        # 直接返回输入的数据
        return (any_input,)

