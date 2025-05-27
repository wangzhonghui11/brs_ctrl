#!/usr/bin/env python3
"""
修复版 JoyCon 按键检测程序
更新内容：
1. 修正了不存在的 get_button_stick() 方法调用
2. 添加了更完善的错误处理
3. 优化了线程退出逻辑
"""

import sys
import time
from pyjoycon import JoyCon, get_R_id, get_L_id
from threading import Thread, Event
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, Dict


class Button(Enum):
    # 右JoyCon按键
    A = auto();
    B = auto();
    X = auto();
    Y = auto();
    R = auto();
    ZR = auto();
    PLUS = auto();
    HOME = auto();
    SR = auto();
    SL = auto();
    RIGHT_STICK = auto();

    # 左JoyCon按键
    UP = auto();
    DOWN = auto();
    LEFT = auto();
    RIGHT = auto();
    L = auto();
    ZL = auto();
    MINUS = auto();
    CAPTURE = auto();
    LEFT_STICK = auto();


@dataclass
class ButtonEvent:
    button: Button
    pressed: bool
    timestamp: float


class JoyConMonitor:
    def __init__(self):
        self._left_joycon = None
        self._right_joycon = None
        self._running = Event()
        self._callbacks = []
        self._connect()

    def _connect(self):
        """自动连接左右JoyCon"""
        for _ in range(3):  # 最多重试3次
            try:
                if not self._right_joycon:
                    right_id = get_R_id()
                    if right_id[0]:
                        self._right_joycon = JoyCon(*right_id)

                if not self._left_joycon:
                    left_id = get_L_id()
                    if left_id[0]:
                        self._left_joycon = JoyCon(*left_id)

                if self._right_joycon or self._left_joycon:
                    return True
            except Exception as e:
                print(f"连接错误: {e}")
            time.sleep(1)
        return False

    def register_callback(self, callback: Callable[[ButtonEvent], None]):
        """注册按键事件回调"""
        self._callbacks.append(callback)

    def _notify_callbacks(self, event: ButtonEvent):
        """通知所有回调函数"""
        for cb in self._callbacks:
            try:
                cb(event)
            except Exception as e:
                print(f"回调错误: {e}")

    def _monitor_buttons(self):
        """按钮监控线程"""
        last_states = {}

        while self._running.is_set():
            try:
                current_time = time.time()

                # 检测右JoyCon按钮
                if self._right_joycon:
                    right_buttons = {
                        Button.A: self._right_joycon.get_button_a(),
                        Button.B: self._right_joycon.get_button_b(),
                        Button.X: self._right_joycon.get_button_x(),
                        Button.Y: self._right_joycon.get_button_y(),
                        Button.R: self._right_joycon.get_button_r(),
                        Button.ZR: self._right_joycon.get_button_zr(),
                        Button.PLUS: self._right_joycon.get_button_plus(),
                        Button.HOME: self._right_joycon.get_button_home(),
                        Button.RIGHT_STICK: self._right_joycon.get_button_r_stick(),
                    }

                    for btn, state in right_buttons.items():
                        if btn not in last_states or last_states[btn] != state:
                            self._notify_callbacks(ButtonEvent(btn, state, current_time))
                        last_states[btn] = state

                # 检测左JoyCon按钮
                if self._left_joycon:
                    left_buttons = {
                        Button.UP: self._left_joycon.get_button_up(),
                        Button.DOWN: self._left_joycon.get_button_down(),
                        Button.LEFT: self._left_joycon.get_button_left(),
                        Button.RIGHT: self._left_joycon.get_button_right(),
                        Button.L: self._left_joycon.get_button_l(),
                        Button.ZL: self._left_joycon.get_button_zl(),
                        Button.MINUS: self._left_joycon.get_button_minus(),
                        Button.CAPTURE: self._left_joycon.get_button_capture(),
                        Button.LEFT_STICK: self._left_joycon.get_button_l_stick(),
                    }

                    for btn, state in left_buttons.items():
                        if btn not in last_states or last_states[btn] != state:
                            self._notify_callbacks(ButtonEvent(btn, state, current_time))
                        last_states[btn] = state

                time.sleep(0.01)  # 100Hz检测频率

            except Exception as e:
                print(f"监控错误: {e}")
                time.sleep(1)  # 出错后等待1秒再重试

    def start(self):
        """启动监控"""
        if not self._connect():
            print("无法连接JoyCon，请检查设备")
            return False

        self._running.set()
        Thread(target=self._monitor_buttons, daemon=True).start()
        return True

    def stop(self):
        """停止监控"""
        self._running.clear()
        if self._left_joycon:
            try:
                del self._left_joycon
            except:
                pass
        if self._right_joycon:
            try:
                del self._right_joycon
            except:
                pass


def print_button_event(event: ButtonEvent):
    """彩色打印按键事件"""
    color = "\033[92m" if event.pressed else "\033[0m"  # 绿色表示按下
    reset = "\033[0m"
    print(f"{color}[{event.timestamp:.3f}] {'按下' if event.pressed else '释放'} {event.button.name}{reset}")


def main():
    monitor = JoyConMonitor()
    monitor.register_callback(print_button_event)

    if not monitor.start():
        sys.exit(1)

    try:
        print("JoyCon按键检测已启动 (按Ctrl+C退出)")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        monitor.stop()
        print("\n检测已停止")


if __name__ == "__main__":
    main()