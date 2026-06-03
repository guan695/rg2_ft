import numpy as np
from pathlib import Path
from typing import Optional, Tuple

from isaacsim.core.prims import SingleArticulation
from isaacsim.core.prims import RigidPrim
from isaacsim.core.utils.types import ArticulationAction

from omni.isaac.motion_generation import ArticulationKinematicsSolver, LulaKinematicsSolver

class Robot:
    ROBOT_PATH = "/World/ur5_rg2/ur5"
    EE_PATH = "/World/ur5_rg2/ur5/wrist_3_link" 
    EE_NAME = "wrist_3_link"
    ASSETS_PATH = Path(__file__).resolve().parents[1] / "assets"
    ARM_JOINT_NAMES = (
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    )
    GRIPPER_JOINT_NAME = "l_base_outer_main"
    GRIPPER_MIN_ANGLE = np.deg2rad(-2.1)
    GRIPPER_MAX_ANGLE = np.deg2rad(66.3)

    def __init__(self):
        self.robot = SingleArticulation(prim_path=self.ROBOT_PATH, name="ur5_view")
        self.original_joints_val = np.array([0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0], dtype=np.float64)
        
        self.ee = RigidPrim(prim_paths_expr=self.EE_PATH, name="ee_view")
        
        self.ee_target: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self.joints_index: Optional[np.ndarray] = None
        self.ee_index: Optional[int] = None
        self.gripper_target = 0.0
        self._initialized = False

        ur5_urdf_path = self.ASSETS_PATH / "ur5.urdf"
        ur5_yaml_path = self.ASSETS_PATH / "ur5_robot_description.yaml"

        base_lula_solver = LulaKinematicsSolver(
            robot_description_path=str(ur5_yaml_path),
            urdf_path=str(ur5_urdf_path)
        )
        self.kinematics_solver = ArticulationKinematicsSolver(
            robot_articulation=self.robot, 
            kinematics_solver=base_lula_solver,
            end_effector_frame_name=self.EE_NAME
        )

    def initialize(self):
        self.robot.initialize()
        # DOF 索引必须在 articulation 初始化后查询。
        self.joints_index = np.array(
            [self.robot.get_dof_index(name) for name in self.ARM_JOINT_NAMES],
            dtype=np.int32,
        )
        self.ee_index = self.robot.get_dof_index(self.GRIPPER_JOINT_NAME)
        
        self.robot.set_joint_positions(self.original_joints_val, joint_indices=self.joints_index)
        self._initialized = True
        self.set_gripper_angle(0.0)

    def set_original_joints(self):
        """Backward-compatible name for the robot initialization step."""
        self.initialize()

    def set_ee_pose(self, target: Tuple[np.ndarray, np.ndarray]):
        self._ensure_initialized()
        target_pos, target_quat = target
        raw_output, success = self.kinematics_solver.compute_inverse_kinematics(
            target_position=target_pos,
            target_orientation=target_quat
        )

        if success:
            self.ee_target = (target_pos.copy(), target_quat.copy())
            if isinstance(raw_output, dict):
                joint_angles = np.array(raw_output["joint_positions"], dtype=np.float64)
            else:
                joint_angles = np.array(raw_output.joint_positions, dtype=np.float64)

            action = ArticulationAction(
                joint_positions=joint_angles,
                joint_indices=self.joints_index
            )
            self.robot.apply_action(action)
            return True
        else:
            # 屏蔽过于频繁的无解打印，可自行放开
            # print("⚠️ 警告: IK 无解！目标位姿超出求解工作空间。")
            return False

    def set_gripper_angle(self, angle: float):
        self._ensure_initialized()
        self.gripper_target = np.clip(angle, self.GRIPPER_MIN_ANGLE, self.GRIPPER_MAX_ANGLE)
        action = ArticulationAction(
            joint_positions=np.array([self.gripper_target]),
            joint_indices=np.array([self.ee_index])
        )
        self.robot.apply_action(action)

    def get_gripper_target(self) -> float:
        return float(self.gripper_target)

    def get_joint_positions(self):
        all_joints = self.robot.get_joint_positions()
        return all_joints.copy() if all_joints is not None else None
    
    def get_ee_pose(self):
        ee_positions, ee_orientations = self.ee.get_world_poses()
        return ee_positions[0], ee_orientations[0]
    
    def get_ee_target(self):
        if self.ee_target is not None:
            target_pos, target_quat = self.ee_target
            return target_pos.copy(), target_quat.copy()
        return None
    
    def get_gripper_angle(self):
        self._ensure_initialized()
        current_joints = self.get_joint_positions()
        if current_joints is not None:
            return float(current_joints[self.ee_index])
        return 0.0

    def _ensure_initialized(self) -> None:
        if not self._initialized or self.joints_index is None or self.ee_index is None:
            raise RuntimeError("Call Robot.initialize() after World.reset() before commanding the robot.")
    
