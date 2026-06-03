from collections import deque
from dataclasses import dataclass
from enum import Enum, auto
from typing import Deque, Optional

import numpy as np

try:
    from .RG2FTSensors import RG2FTSensors
    from .Robot import Robot
except ImportError:
    from RG2FTSensors import RG2FTSensors
    from Robot import Robot


class GripperControlState(Enum):
    """夹爪力控状态。"""

    IDLE = auto()
    CONTACT_SEARCH = auto()
    FORCE_REGULATION = auto()
    HOLD = auto()
    RELEASE = auto()
    FAULT = auto()


@dataclass
class GripperForceConfig:
    """夹爪力控参数，数值均使用 SI 单位。"""

    target_force_n: float = 4.0
    contact_threshold_n: float = 0.8
    max_force_n: float = 8.0
    max_torque_nm: float = 0.25
    max_force_imbalance_n: float = 4.0

    search_speed_rad_s: float = 0.08
    proximity_search_speed_rad_s: float = 0.03
    proximity_slowdown_mm: float = 8.0
    release_speed_rad_s: float = 0.20
    max_control_speed_rad_s: float = 0.05

    kp: float = 0.012
    ki: float = 0.003
    integral_limit: float = 8.0
    force_tolerance_n: float = 0.35

    median_window_size: int = 5
    low_pass_alpha: float = 0.25
    contact_confirm_frames: int = 5
    stable_confirm_frames: int = 20
    one_sided_contact_timeout_s: float = 1.0
    search_timeout_s: float = 8.0
    regulation_timeout_s: float = 5.0


@dataclass
class GripperTelemetry:
    """控制器使用的紧凑遥测数据。"""

    left_force_n: float
    right_force_n: float
    mean_force_n: float
    force_imbalance_n: float
    max_torque_nm: float
    left_distance_mm: float
    right_distance_mm: float


class _ForceSignalFilter:
    """对左右指尖力执行中值滤波和一阶低通滤波。"""

    def __init__(self, window_size: int, alpha: float) -> None:
        self._alpha = alpha
        self._left_window: Deque[float] = deque(maxlen=window_size)
        self._right_window: Deque[float] = deque(maxlen=window_size)
        self._left_filtered: Optional[float] = None
        self._right_filtered: Optional[float] = None

    def update(self, left_force_n: float, right_force_n: float) -> tuple[float, float]:
        self._left_window.append(left_force_n)
        self._right_window.append(right_force_n)
        left_median = float(np.median(self._left_window))
        right_median = float(np.median(self._right_window))

        if self._left_filtered is None:
            self._left_filtered = left_median
            self._right_filtered = right_median
        else:
            self._left_filtered += self._alpha * (left_median - self._left_filtered)
            self._right_filtered += self._alpha * (right_median - self._right_filtered)

        return self._left_filtered, self._right_filtered

    def reset(self) -> None:
        self._left_window.clear()
        self._right_window.clear()
        self._left_filtered = None
        self._right_filtered = None


class GripperForceController:
    """在 Isaac Sim 主线程中执行 RG2-FT 接触搜索和恒力夹持。"""

    # 已通过仿真标定：指尖局部 Z 轴是主要夹持力方向，左右符号均为正。
    NORMAL_FORCE_AXIS = 2

    def __init__(
        self,
        robot: Robot,
        sensors: RG2FTSensors,
        config: Optional[GripperForceConfig] = None,
    ) -> None:
        self.robot = robot
        self.sensors = sensors
        self.config = config or GripperForceConfig()
        self._validate_config()

        self.state = GripperControlState.IDLE
        self.fault_reason: Optional[str] = None
        self.latest_telemetry: Optional[GripperTelemetry] = None

        self._force_filter = _ForceSignalFilter(
            window_size=self.config.median_window_size,
            alpha=self.config.low_pass_alpha,
        )
        self._reset_control_memory()

    @property
    def manual_control_allowed(self) -> bool:
        """仅空闲状态允许键盘直接控制夹爪角度。"""
        return self.state is GripperControlState.IDLE

    # --------------------------- 对外命令 ---------------------------

    def start_grasp(self) -> None:
        """从空闲状态开始搜索物体并建立恒力夹持。"""
        if self.state is GripperControlState.FAULT:
            print("[GripperForceController] Clear the fault before starting a new grasp.")
            return
        if self.state is not GripperControlState.IDLE:
            print(f"[GripperForceController] Grasp ignored while state is {self.state.name}.")
            return

        self._reset_control_memory()
        self._transition_to(GripperControlState.CONTACT_SEARCH)

    def release(self) -> None:
        """释放物体；若此前发生故障，释放后仍保持故障锁存。"""
        self._reset_control_memory()
        self._transition_to(GripperControlState.RELEASE)

    def reset_fault(self) -> None:
        """由操作员确认后清除故障锁存。"""
        if self.state is not GripperControlState.FAULT:
            print("[GripperForceController] No fault to clear.")
            return

        self.fault_reason = None
        self._reset_control_memory()
        self._transition_to(GripperControlState.IDLE)

    def zero_force_bias(self) -> None:
        """仅在空载空闲状态下记录 F/T 零偏。"""
        if self.state is not GripperControlState.IDLE:
            print("[GripperForceController] Zeroing is only allowed in IDLE.")
            return

        self.sensors.zero_force_bias()
        self._force_filter.reset()
        print("[GripperForceController] F/T bias zeroed. Keep fingertips unloaded when zeroing.")

    # --------------------------- 主循环 ---------------------------

    def update(self, dt: float) -> None:
        """读取传感器并执行一个力控周期。"""
        if dt <= 0.0:
            return

        telemetry = self._read_telemetry()
        self.latest_telemetry = telemetry
        self._state_elapsed_s += dt

        if self._requires_safety_check() and self._has_exceeded_limits(telemetry):
            return

        if self.state is GripperControlState.CONTACT_SEARCH:
            self._update_contact_search(telemetry, dt)
        elif self.state is GripperControlState.FORCE_REGULATION:
            self._update_force_regulation(telemetry, dt)
        elif self.state is GripperControlState.HOLD:
            self._apply_pi_control(telemetry, dt)
        elif self.state is GripperControlState.RELEASE:
            self._update_release(dt)

    def format_status(self) -> str:
        """生成适合终端输出的单行状态。"""
        telemetry = self.latest_telemetry
        if telemetry is None:
            return f"State={self.state.name} telemetry=unavailable"

        status = (
            f"State={self.state.name} mode=local_z "
            f"F=({telemetry.left_force_n:.2f}, {telemetry.right_force_n:.2f})N "
            f"mean={telemetry.mean_force_n:.2f}N "
            f"imbalance={telemetry.force_imbalance_n:.2f}N "
            f"Tmax={telemetry.max_torque_nm:.3f}Nm "
            f"D=({telemetry.left_distance_mm:.1f}, {telemetry.right_distance_mm:.1f})mm"
        )
        if self.fault_reason is not None:
            status += f" fault={self.fault_reason}"
        return status

    # --------------------------- 状态处理 ---------------------------

    def _update_contact_search(self, telemetry: GripperTelemetry, dt: float) -> None:
        """低速闭合夹爪，双侧稳定接触后进入力调节。"""
        left_contact = telemetry.left_force_n >= self.config.contact_threshold_n
        right_contact = telemetry.right_force_n >= self.config.contact_threshold_n

        if left_contact and right_contact:
            self._contact_confirm_count += 1
            self._one_sided_contact_elapsed_s = 0.0
        else:
            self._contact_confirm_count = 0
            self._one_sided_contact_elapsed_s = (
                self._one_sided_contact_elapsed_s + dt if left_contact != right_contact else 0.0
            )

        if self._contact_confirm_count >= self.config.contact_confirm_frames:
            self._integral_error = 0.0
            self._transition_to(GripperControlState.FORCE_REGULATION)
            return
        if self._one_sided_contact_elapsed_s >= self.config.one_sided_contact_timeout_s:
            self._set_fault("one-sided contact persisted too long")
            return
        if self._state_elapsed_s >= self.config.search_timeout_s:
            self._set_fault("contact search timed out")
            return

        speed = self.config.search_speed_rad_s
        if min(telemetry.left_distance_mm, telemetry.right_distance_mm) <= self.config.proximity_slowdown_mm:
            speed = self.config.proximity_search_speed_rad_s
        self._close_gripper(speed, dt)

    def _update_force_regulation(self, telemetry: GripperTelemetry, dt: float) -> None:
        """使用 PI 控制器逼近目标力，稳定后进入保持状态。"""
        self._apply_pi_control(telemetry, dt)

        if abs(self.config.target_force_n - telemetry.mean_force_n) <= self.config.force_tolerance_n:
            self._stable_confirm_count += 1
        else:
            self._stable_confirm_count = 0

        if self._stable_confirm_count >= self.config.stable_confirm_frames:
            self._transition_to(GripperControlState.HOLD)
        elif self._state_elapsed_s >= self.config.regulation_timeout_s:
            self._set_fault("force regulation timed out")

    def _update_release(self, dt: float) -> None:
        """按限速打开夹爪；故障状态只有人工确认后才会真正解除。"""
        target = self.robot.get_gripper_target() - self.config.release_speed_rad_s * dt
        if target <= self.robot.GRIPPER_MIN_ANGLE:
            self.robot.set_gripper_angle(self.robot.GRIPPER_MIN_ANGLE)
            next_state = GripperControlState.FAULT if self.fault_reason else GripperControlState.IDLE
            self._transition_to(next_state)
            return
        self.robot.set_gripper_angle(target)

    # --------------------------- 控制与保护 ---------------------------

    def _apply_pi_control(self, telemetry: GripperTelemetry, dt: float) -> None:
        """PI 输出为夹爪角速度，并执行积分与速度限幅。"""
        error = self.config.target_force_n - telemetry.mean_force_n
        self._integral_error = float(
            np.clip(
                self._integral_error + error * dt,
                -self.config.integral_limit,
                self.config.integral_limit,
            )
        )
        speed = self.config.kp * error + self.config.ki * self._integral_error
        speed = float(
            np.clip(
                speed,
                -self.config.max_control_speed_rad_s,
                self.config.max_control_speed_rad_s,
            )
        )
        self.robot.set_gripper_angle(self.robot.get_gripper_target() + speed * dt)

    def _close_gripper(self, speed_rad_s: float, dt: float) -> None:
        """按指定速度闭合夹爪，并处理到达机械限位的情况。"""
        target = self.robot.get_gripper_target() + speed_rad_s * dt
        if target >= self.robot.GRIPPER_MAX_ANGLE:
            self.robot.set_gripper_angle(self.robot.GRIPPER_MAX_ANGLE)
            self._set_fault("maximum closure reached before stable contact")
            return
        self.robot.set_gripper_angle(target)

    def _has_exceeded_limits(self, telemetry: GripperTelemetry) -> bool:
        """执行软件级过程保护；该逻辑不能替代工业安全系统。"""
        if max(telemetry.left_force_n, telemetry.right_force_n) > self.config.max_force_n:
            self._set_fault("finger force exceeded limit")
            return True
        if telemetry.max_torque_nm > self.config.max_torque_nm:
            self._set_fault("finger torque exceeded limit")
            return True
        if telemetry.force_imbalance_n > self.config.max_force_imbalance_n:
            self._set_fault("left/right force imbalance exceeded limit")
            return True
        return False

    # --------------------------- 遥测与内部状态 ---------------------------

    def _read_telemetry(self) -> GripperTelemetry:
        """读取原始数据，并提取滤波后的局部 Z 轴夹持力。"""
        data = self.sensors.read()
        left_force_n, right_force_n = self._force_filter.update(
            self._extract_normal_force(data["left"]["force_3d"]),
            self._extract_normal_force(data["right"]["force_3d"]),
        )

        left_torque = np.asarray(data["left"]["torque_3d"], dtype=np.float64)
        right_torque = np.asarray(data["right"]["torque_3d"], dtype=np.float64)
        return GripperTelemetry(
            left_force_n=left_force_n,
            right_force_n=right_force_n,
            mean_force_n=0.5 * (left_force_n + right_force_n),
            force_imbalance_n=abs(left_force_n - right_force_n),
            max_torque_nm=max(float(np.linalg.norm(left_torque)), float(np.linalg.norm(right_torque))),
            left_distance_mm=float(data["left"]["dist_mm"]),
            right_distance_mm=float(data["right"]["dist_mm"]),
        )

    def _extract_normal_force(self, force_3d: np.ndarray) -> float:
        """提取已标定的局部 Z 轴正向夹持力。"""
        return max(0.0, float(np.asarray(force_3d)[self.NORMAL_FORCE_AXIS]))

    def _requires_safety_check(self) -> bool:
        return self.state in (
            GripperControlState.CONTACT_SEARCH,
            GripperControlState.FORCE_REGULATION,
            GripperControlState.HOLD,
        )

    def _set_fault(self, reason: str) -> None:
        self.fault_reason = reason
        self._transition_to(GripperControlState.FAULT)

    def _transition_to(self, state: GripperControlState) -> None:
        previous_state = self.state
        self.state = state
        self._state_elapsed_s = 0.0
        print(f"[GripperForceController] {previous_state.name} -> {state.name}")
        if state is GripperControlState.FAULT and self.fault_reason is not None:
            print(f"[GripperForceController] Fault: {self.fault_reason}")

    def _reset_control_memory(self) -> None:
        self._integral_error = 0.0
        self._state_elapsed_s = 0.0
        self._one_sided_contact_elapsed_s = 0.0
        self._contact_confirm_count = 0
        self._stable_confirm_count = 0

    def _validate_config(self) -> None:
        if self.config.median_window_size < 1:
            raise ValueError("median_window_size must be positive.")
        if not 0.0 < self.config.low_pass_alpha <= 1.0:
            raise ValueError("low_pass_alpha must be in the range (0, 1].")
