import threading
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Deque, Optional

import numpy as np

try:
    from .GripperForceController import GripperForceController
    from .Robot import Robot
    from .RG2FTSensors import RG2FTSensors
except ImportError:
    from GripperForceController import GripperForceController
    from Robot import Robot
    from RG2FTSensors import RG2FTSensors
from scipy.spatial.transform import Rotation as R


class KeyboardCommand(Enum):
    TELEMETRY = "g"
    START_GRASP = "c"
    RELEASE = "v"
    RESET_FAULT = "b"
    ZERO_FORCE_BIAS = "n"


@dataclass(frozen=True)
class TeleopScale:
    pos: float = 0.005
    rpy: float = 0.02
    gripper: float = 0.05


class KeyboardInputDevice:
    """键盘遥控设备：负责监听按键并通过逆解控制机器人。"""

    ARM_KEYS = ("w", "s", "a", "d", "q", "e", "i", "k", "j", "l", "u", "o")
    COMMAND_BY_KEY = {command.value: command for command in KeyboardCommand}
    REFERENCE_FPS = 60.0

    def __init__(
        self,
        robot: Robot,
        sensors: Optional[RG2FTSensors] = None,
        force_controller: Optional[GripperForceController] = None,
        pos_scale=0.005,
        rpy_scale=0.02,
        gripper_scale=0.05,
    ):
        self.robot = robot
        self.sensors = sensors
        self.force_controller = force_controller
        
        # Scale 表示 60 FPS 下每帧的增量，update() 会按实际 dt 做归一化。
        self.scale = TeleopScale(pos=pos_scale, rpy=rpy_scale, gripper=gripper_scale)

        self.state = {
            "w": False, "s": False, "a": False, "d": False, "q": False, "e": False, # 位置
            "i": False, "k": False, "j": False, "l": False, "u": False, "o": False, # 姿态
            "z": False, "x": False,                                                 # 夹爪
        }
        self._command_queue: Deque[KeyboardCommand] = deque()

        self.listener = None
        self.listener_thread = None
        self._pressed_command_keys = set()
        self.connected = False

    def connect(self):
        if self.connected:
            return

        from pynput import keyboard
        self.listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        
        self.listener_thread = threading.Thread(target=self._start_listener)
        self.listener_thread.daemon = True
        self.listener_thread.start()

        self.connected = True
        print(f"\n[{self.__class__.__name__}] 键盘控制器已连接！点击仿真视口使窗口获得焦点。")
        print(f"""[{self.__class__.__name__}] 操作指南:
        - W/S : X 轴前后平移
        - A/D : Y 轴左右平移
        - Q/E : Z 轴上下平移
        - I/K : 绕 Y 轴俯仰 (Pitch)
        - J/L : 绕 X 轴翻滚 (Roll)
        - U/O : 绕 Z 轴偏航 (Yaw)
        - Z/X : 夹爪闭合 / 张开
        - G   : 打印一次 RG2-FT 传感器数据
        - C   : 开始自动恒力抓取
        - V   : 释放夹爪
        - B   : 清除力控故障
        - N   : 在无接触状态校准 F/T 零偏\n""")

    def _start_listener(self):
        self.listener.start()
        self.listener.join()

    def _on_press(self, key):
        try:
            k = key.char.lower()
            if k in self.state:
                self.state[k] = True
            elif k in self.COMMAND_BY_KEY and k not in self._pressed_command_keys:
                self._pressed_command_keys.add(k)
                self._command_queue.append(self.COMMAND_BY_KEY[k])
        except AttributeError:
            pass

    def _on_release(self, key):
        try:
            k = key.char.lower()
            if k in self.state:
                self.state[k] = False
            self._pressed_command_keys.discard(k)
        except AttributeError:
            pass

    def disconnect(self):
        if self.connected:
            if self.listener:
                self.listener.stop()
            self.connected = False
            print(f"[{self.__class__.__name__}] 断开连接。")

    def update(self, dt: float):
        """每一帧调用此函数，计算增量并控制机器人"""
        if not self.connected:
            return

        self._process_command_queue()

        # 检查是否有按键被按下，如果没有则跳过计算，节省算力
        if not any(self.state.values()):
            return

        frame_scale = dt * self.REFERENCE_FPS
        if any(self.state[key] for key in self.ARM_KEYS):
            # 获取当前控制目标（若还没有目标，则以当前末端真实位姿作为起点）
            target = self.robot.get_ee_target()
            if target is None:
                target = self.robot.get_ee_pose()
            
            target_pos, target_quat_wxyz = target
            
            target_pos = target_pos + self._position_delta() * frame_scale
            target_quat_wxyz = self._apply_rotation_delta(target_quat_wxyz, frame_scale)

            # 发送底层 IK 请求
            self.robot.set_ee_pose((target_pos, target_quat_wxyz))

        # 3. 夹爪控制
        if self.force_controller is not None and not self.force_controller.manual_control_allowed:
            return

        current_gripper_angle = self.robot.get_gripper_target()
        if self.state["z"] and not self.state["x"]:
            current_gripper_angle += self.scale.gripper * frame_scale
            self.robot.set_gripper_angle(current_gripper_angle)
        elif self.state["x"] and not self.state["z"]:
            current_gripper_angle -= self.scale.gripper * frame_scale
            self.robot.set_gripper_angle(current_gripper_angle)

    def _position_delta(self) -> np.ndarray:
        delta_pos = np.zeros(3)
        if self.state["w"]: delta_pos[0] += self.scale.pos
        if self.state["s"]: delta_pos[0] -= self.scale.pos
        if self.state["a"]: delta_pos[1] += self.scale.pos
        if self.state["d"]: delta_pos[1] -= self.scale.pos
        if self.state["q"]: delta_pos[2] += self.scale.pos
        if self.state["e"]: delta_pos[2] -= self.scale.pos
        return delta_pos

    def _rotation_delta(self) -> np.ndarray:
        delta_rpy = np.zeros(3)
        if self.state["j"]: delta_rpy[0] -= self.scale.rpy
        if self.state["l"]: delta_rpy[0] += self.scale.rpy
        if self.state["i"]: delta_rpy[1] += self.scale.rpy
        if self.state["k"]: delta_rpy[1] -= self.scale.rpy
        if self.state["u"]: delta_rpy[2] += self.scale.rpy
        if self.state["o"]: delta_rpy[2] -= self.scale.rpy
        return delta_rpy

    def _apply_rotation_delta(self, target_quat_wxyz: np.ndarray, frame_scale: float) -> np.ndarray:
        delta_rpy = self._rotation_delta()
        if not np.any(delta_rpy != 0):
            return target_quat_wxyz

        r_delta = R.from_euler("xyz", delta_rpy * frame_scale)
        r_curr = R.from_quat(self._wxyz_to_xyzw(target_quat_wxyz))
        r_new = r_delta * r_curr
        return self._xyzw_to_wxyz(r_new.as_quat())

    @staticmethod
    def _wxyz_to_xyzw(quat_wxyz: np.ndarray) -> np.ndarray:
        return np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])

    @staticmethod
    def _xyzw_to_wxyz(quat_xyzw: np.ndarray) -> np.ndarray:
        return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

    def _process_command_queue(self):
        while self._command_queue:
            command = self._command_queue.popleft()
            self._process_command(command)

    def _process_command(self, command: KeyboardCommand):
        if command is KeyboardCommand.TELEMETRY:
            self._print_telemetry()
            return

        if self.force_controller is None:
            print("RG2-FT force controller is not configured.")
            return

        if command is KeyboardCommand.START_GRASP:
            self.force_controller.start_grasp()
        elif command is KeyboardCommand.RELEASE:
            self.force_controller.release()
        elif command is KeyboardCommand.RESET_FAULT:
            self.force_controller.reset_fault()
        elif command is KeyboardCommand.ZERO_FORCE_BIAS:
            self.force_controller.zero_force_bias()

    def _print_telemetry(self):
        if self.sensors is None:
            print("RG2-FT sensors are not configured.")
        else:
            print(self.sensors.format_data(self.sensors.read()))
        if self.force_controller is not None:
            print(self.force_controller.format_status())
