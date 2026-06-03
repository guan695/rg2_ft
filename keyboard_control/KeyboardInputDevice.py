import threading
from typing import Optional

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

class KeyboardInputDevice:
    """键盘遥控设备：负责监听按键并通过逆解控制机器人。"""

    ARM_KEYS = ("w", "s", "a", "d", "q", "e", "i", "k", "j", "l", "u", "o")
    COMMAND_KEYS = ("g", "c", "v", "b", "n")
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
        self.pos_scale = pos_scale
        self.rpy_scale = rpy_scale
        self.gripper_scale = gripper_scale

        self.state = {
            "w": False, "s": False, "a": False, "d": False, "q": False, "e": False, # 位置
            "i": False, "k": False, "j": False, "l": False, "u": False, "o": False, # 姿态
            "z": False, "x": False,                                                 # 夹爪
        }
        self.telemetry_requested = False
        self.grasp_requested = False
        self.release_requested = False
        self.fault_reset_requested = False
        self.zero_force_bias_requested = False

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
            elif k in self.COMMAND_KEYS and k not in self._pressed_command_keys:
                self._pressed_command_keys.add(k)
                if k == "g":
                    self.telemetry_requested = True
                elif k == "c":
                    self.grasp_requested = True
                elif k == "v":
                    self.release_requested = True
                elif k == "b":
                    self.fault_reset_requested = True
                elif k == "n":
                    self.zero_force_bias_requested = True
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

        self._process_force_control_requests()

        if self.telemetry_requested:
            self.telemetry_requested = False
            if self.sensors is None:
                print("RG2-FT sensors are not configured.")
            else:
                print(self.sensors.format_data(self.sensors.read()))
            if self.force_controller is not None:
                print(self.force_controller.format_status())

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
            
            # 1. 位置控制 (XYZ)
            delta_pos = np.zeros(3)
            if self.state["w"]: delta_pos[0] += self.pos_scale
            if self.state["s"]: delta_pos[0] -= self.pos_scale
            if self.state["a"]: delta_pos[1] += self.pos_scale
            if self.state["d"]: delta_pos[1] -= self.pos_scale
            if self.state["q"]: delta_pos[2] += self.pos_scale
            if self.state["e"]: delta_pos[2] -= self.pos_scale
            
            target_pos += delta_pos * frame_scale

            # 2. 姿态控制 (RPY)
            delta_rpy = np.zeros(3)
            if self.state["j"]: delta_rpy[0] -= self.rpy_scale
            if self.state["l"]: delta_rpy[0] += self.rpy_scale
            if self.state["i"]: delta_rpy[1] += self.rpy_scale
            if self.state["k"]: delta_rpy[1] -= self.rpy_scale
            if self.state["u"]: delta_rpy[2] += self.rpy_scale
            if self.state["o"]: delta_rpy[2] -= self.rpy_scale

            # 如果发生姿态改变
            if np.any(delta_rpy != 0):
                # 获取增量旋转
                r_delta = R.from_euler('xyz', delta_rpy * frame_scale)
                
                # Scipy 必须使用 XYZW 顺序，而 Isaac Sim 是 WXYZ
                curr_quat_xyzw = np.array([target_quat_wxyz[1], target_quat_wxyz[2], target_quat_wxyz[3], target_quat_wxyz[0]])
                r_curr = R.from_quat(curr_quat_xyzw)
                
                # 旋转叠加 (在全局坐标系下应用增量)
                r_new = r_delta * r_curr 
                
                # 转换回 WXYZ 顺序
                new_quat_xyzw = r_new.as_quat()
                target_quat_wxyz = np.array([new_quat_xyzw[3], new_quat_xyzw[0], new_quat_xyzw[1], new_quat_xyzw[2]])

            # 发送底层 IK 请求
            self.robot.set_ee_pose((target_pos, target_quat_wxyz))

        # 3. 夹爪控制
        if self.force_controller is not None and not self.force_controller.manual_control_allowed:
            return

        current_gripper_angle = self.robot.gripper_target
        if self.state["z"] and not self.state["x"]:
            current_gripper_angle += self.gripper_scale * frame_scale
            self.robot.set_gripper_angle(current_gripper_angle)
        elif self.state["x"] and not self.state["z"]:
            current_gripper_angle -= self.gripper_scale * frame_scale
            self.robot.set_gripper_angle(current_gripper_angle)

    def _process_force_control_requests(self):
        if self.force_controller is None:
            if any((
                self.grasp_requested,
                self.release_requested,
                self.fault_reset_requested,
                self.zero_force_bias_requested,
            )):
                print("RG2-FT force controller is not configured.")
            self.grasp_requested = False
            self.release_requested = False
            self.fault_reset_requested = False
            self.zero_force_bias_requested = False
            return

        if self.grasp_requested:
            self.grasp_requested = False
            self.force_controller.start_grasp()
        if self.release_requested:
            self.release_requested = False
            self.force_controller.release()
        if self.fault_reset_requested:
            self.fault_reset_requested = False
            self.force_controller.reset_fault()
        if self.zero_force_bias_requested:
            self.zero_force_bias_requested = False
            self.force_controller.zero_force_bias()
