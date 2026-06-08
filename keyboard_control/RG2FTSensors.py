from typing import Any, Optional

import numpy as np
from isaacsim.core.prims import SingleArticulation
from isaacsim.sensors.physx import _range_sensor

try:
    from .RobotConfig import DEFAULT_ROBOT_MODEL, get_robot_config
except ImportError:
    from RobotConfig import DEFAULT_ROBOT_MODEL, get_robot_config


class RG2FTSensors:
    """Read RG2-FT data through Robot's injected articulation."""

    DEFAULT_LEFT_JOINT = "l_tip"
    DEFAULT_RIGHT_JOINT = "r_tip"

    def __init__(
        self,
        articulation: SingleArticulation,
        left_joint_name: str = DEFAULT_LEFT_JOINT,
        right_joint_name: str = DEFAULT_RIGHT_JOINT,
        left_lidar_path: Optional[str] = None,
        right_lidar_path: Optional[str] = None,
        model: str = DEFAULT_ROBOT_MODEL,
        max_distance_mm: float = 100.0,
        distance_offset_mm: float = 1.0,
    ) -> None:
        config = get_robot_config(model)
        self.articulation = articulation
        self.left_joint_name = left_joint_name
        self.right_joint_name = right_joint_name
        self.left_lidar_path = left_lidar_path or config.left_lidar_path
        self.right_lidar_path = right_lidar_path or config.right_lidar_path
        self.max_distance_mm = max_distance_mm
        self.distance_offset_mm = distance_offset_mm

        self.lidar_api = _range_sensor.acquire_lidar_sensor_interface()
        self._left_force_bias = np.zeros(6)
        self._right_force_bias = np.zeros(6)
        self._initialized = False

    def initialize(self) -> None:
        """Validate the shared articulation after Robot has initialized it."""
        if not self.articulation.handles_initialized:
            raise RuntimeError(
                "The shared articulation physics handle is unavailable. Initialize "
                "Robot after starting or resetting the simulation, then initialize sensors."
            )

        self.articulation_view = self.articulation._articulation_view
        if self.articulation_view is None:
            raise RuntimeError("The shared articulation view is unavailable.")

        available_joints = set(self.articulation_view.joint_names)
        required_joints = {self.left_joint_name, self.right_joint_name}
        missing_joints = required_joints - available_joints
        if missing_joints:
            raise ValueError(
                f"RG2-FT joints not found: {sorted(missing_joints)}. "
                f"Available joints: {self.articulation_view.joint_names}"
            )

        self._initialized = True

    def read(self) -> dict[str, dict[str, Any]]:
        """Return force, torque, and distance measurements for both fingertips."""
        self._ensure_initialized()
        left_wrench, right_wrench = self._read_wrenches()

        return {
            "left": {
                "force_3d": left_wrench[:3],
                "torque_3d": left_wrench[3:],
                "dist_mm": self._read_distance_mm(self.left_lidar_path),
            },
            "right": {
                "force_3d": right_wrench[:3],
                "torque_3d": right_wrench[3:],
                "dist_mm": self._read_distance_mm(self.right_lidar_path),
            },
        }

    def zero_force_bias(self) -> None:
        """Record the unloaded fingertip wrenches as the force-torque bias."""
        self._ensure_initialized()
        left_wrench, right_wrench = self._read_raw_wrenches()
        self._left_force_bias = left_wrench
        self._right_force_bias = right_wrench

    def _read_wrenches(self) -> tuple[np.ndarray, np.ndarray]:
        left_wrench, right_wrench = self._read_raw_wrenches()
        return (
            left_wrench - self._left_force_bias,
            right_wrench - self._right_force_bias,
        )

    def _read_raw_wrenches(self) -> tuple[np.ndarray, np.ndarray]:
        forces = self.articulation_view.get_measured_joint_forces(
            joint_names=[self.left_joint_name, self.right_joint_name]
        )
        if forces is None:
            return np.zeros(6), np.zeros(6)

        forces = np.asarray(forces)
        if forces.ndim == 3:
            forces = forces[0]
        if forces.shape != (2, 6):
            raise RuntimeError(f"Unexpected RG2-FT wrench shape: {forces.shape}")

        return forces[0].copy(), forces[1].copy()

    def _read_distance_mm(self, lidar_path: str) -> float:
        depths = self.lidar_api.get_linear_depth_data(lidar_path)
        if depths is None:
            return self.max_distance_mm

        depths = np.asarray(depths, dtype=np.float64)
        valid_depths = depths[np.isfinite(depths) & (depths > 0.0)]
        if valid_depths.size == 0:
            return self.max_distance_mm

        distance_mm = float(np.min(valid_depths)) * 1000.0 - self.distance_offset_mm
        return float(np.clip(distance_mm, 0.0, self.max_distance_mm))

    def _ensure_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError("Call RG2FTSensors.initialize() before reading sensor data.")

    @staticmethod
    def format_data(data: dict[str, dict[str, Any]]) -> str:
        """Format one compact telemetry line for occasional logging."""
        left = data["left"]
        right = data["right"]
        return (
            f"L F={RG2FTSensors._format_vector(left['force_3d'], 2)}N "
            f"T={RG2FTSensors._format_vector(left['torque_3d'], 4)}Nm "
            f"D={left['dist_mm']:.1f}mm | "
            f"R F={RG2FTSensors._format_vector(right['force_3d'], 2)}N "
            f"T={RG2FTSensors._format_vector(right['torque_3d'], 4)}Nm "
            f"D={right['dist_mm']:.1f}mm"
        )

    @staticmethod
    def _format_vector(values: np.ndarray, precision: int) -> str:
        return "(" + ", ".join(f"{value:.{precision}f}" for value in values) + ")"
