import os
import traceback

from isaacsim import SimulationApp


simulation_app = SimulationApp({"headless": False})

from isaacsim.core.api import World
from isaacsim.core.utils.stage import open_stage

from keyboard_control.Robot import Robot
from keyboard_control.RobotConfig import DEFAULT_ROBOT_MODEL, get_robot_config
from keyboard_control.RG2FTSensors import RG2FTSensors


ROBOT_MODEL = os.environ.get("RG2_ROBOT_MODEL", DEFAULT_ROBOT_MODEL)
ROBOT_CONFIG = get_robot_config(ROBOT_MODEL)
USD_PATH = os.environ.get("RG2_SCENE_USD", ROBOT_CONFIG.scene_usd_path)
PRINT_INTERVAL_FRAMES = 60


def main() -> None:
    open_stage(usd_path=USD_PATH)

    world = World(stage_units_in_meters=1.0)
    world.reset()

    robot = Robot(config=ROBOT_CONFIG)
    robot.set_original_joints()

    sensors = RG2FTSensors(
        articulation=robot.robot,
        left_lidar_path=ROBOT_CONFIG.left_lidar_path,
        right_lidar_path=ROBOT_CONFIG.right_lidar_path,
    )
    try:
        sensors.initialize()
    except Exception:
        print("Sensor initialization failed.")
        print(f"Available articulation DOFs: {robot.robot.dof_names}")
        view = robot.robot._articulation_view
        if view is not None:
            print(f"Available articulation joints: {view.joint_names}")
            print(f"Available articulation links: {view.body_names}")
        traceback.print_exc()
        raise

    print("RG2-FT sensor test started.")
    print(f"Scene: {USD_PATH}")
    print(f"Available articulation joints: {robot.robot._articulation_view.joint_names}")
    print(f"Printing telemetry every {PRINT_INTERVAL_FRAMES} physics frames.")

    frame = 0
    while simulation_app.is_running():
        world.step(render=True)
        frame += 1

        if frame % PRINT_INTERVAL_FRAMES == 0:
            data = sensors.read()
            print(f"[Frame {frame}] {sensors.format_data(data)}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Sensor test interrupted by user.")
    finally:
        simulation_app.close()
