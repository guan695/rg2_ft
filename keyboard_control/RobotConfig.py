from dataclasses import dataclass
from pathlib import Path


DEFAULT_ROBOT_MODEL = "ur5e"
SUPPORTED_ROBOT_MODELS = ("ur5", "ur5e")


@dataclass(frozen=True)
class RobotModelConfig:
    model: str
    scene_root_path: str
    robot_path: str
    ee_path: str
    ee_name: str
    articulation_name: str
    urdf_path: Path
    robot_description_path: Path
    left_lidar_path: str
    right_lidar_path: str
    scene_usd_path: str


def get_robot_config(model: str = DEFAULT_ROBOT_MODEL) -> RobotModelConfig:
    normalized_model = model.lower()
    if normalized_model not in SUPPORTED_ROBOT_MODELS:
        supported = ", ".join(SUPPORTED_ROBOT_MODELS)
        raise ValueError(f"Unsupported robot model '{model}'. Expected one of: {supported}.")

    assets_path = Path(__file__).resolve().parents[1] / "assets"
    scene_name = f"{normalized_model}_rg2"
    scene_root_path = f"/World/{scene_name}"
    robot_path = f"{scene_root_path}/{normalized_model}"

    return RobotModelConfig(
        model=normalized_model,
        scene_root_path=scene_root_path,
        robot_path=robot_path,
        ee_path=f"{robot_path}/wrist_3_link",
        ee_name="wrist_3_link",
        articulation_name=f"{normalized_model}_view",
        urdf_path=assets_path / f"{normalized_model}.urdf",
        robot_description_path=assets_path / f"{normalized_model}_robot_description.yaml",
        left_lidar_path=f"{scene_root_path}/rg2_ft_tip/l_finger_tip/Lidar",
        right_lidar_path=f"{scene_root_path}/rg2_ft_tip/r_finger_tip/Lidar",
        scene_usd_path=f"/home/guan/Desktop/{scene_name}/my_scene/pick.usd",
    )
