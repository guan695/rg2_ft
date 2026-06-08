from isaacsim import SimulationApp

# 启动仿真应用
simulation_app = SimulationApp({
    "headless": False,
    "active_gpu": 0,
    "physics_gpu": 0,
})

import os

from isaacsim.core.api import World
from isaacsim.core.utils.stage import open_stage

from Robot import Robot
from RobotConfig import DEFAULT_ROBOT_MODEL, get_robot_config
from KeyboardInputDevice import KeyboardInputDevice
from GripperForceController import GripperForceConfig, GripperForceController
from RG2FTSensors import RG2FTSensors


# Change this to "ur5" or "ur5e" to switch the robot model.
ROBOT_MODEL = DEFAULT_ROBOT_MODEL
ROBOT_MODEL = os.environ.get("RG2_ROBOT_MODEL", ROBOT_MODEL)
ROBOT_CONFIG = get_robot_config(ROBOT_MODEL)
USD_PATH = os.environ.get("RG2_SCENE_USD", ROBOT_CONFIG.scene_usd_path)


def create_world():
    print(f"Robot model: {ROBOT_CONFIG.model}")
    print(f"Scene: {USD_PATH}")
    open_stage(usd_path=USD_PATH)

    # 创建仿真世界并启用 deformable body 所需的 GPU PhysX 配置
    world = World(stage_units_in_meters=1.0)
    physics_context = world.get_physics_context()
    physics_context.enable_gpu_dynamics(True)
    physics_context.set_broadphase_type("GPU")

    print("GPU dynamics before reset:", physics_context.is_gpu_dynamics_enabled())
    print("Broadphase before reset:", physics_context.get_broadphase_type())

    world.reset()

    print("GPU dynamics after reset:", physics_context.is_gpu_dynamics_enabled())
    print("Broadphase after reset:", physics_context.get_broadphase_type())
    return world


def create_control_stack():
    # 实例化机器人并复位
    robot = Robot(config=ROBOT_CONFIG)
    robot.initialize()
    sensors = RG2FTSensors(
        articulation=robot.robot,
        left_lidar_path=ROBOT_CONFIG.left_lidar_path,
        right_lidar_path=ROBOT_CONFIG.right_lidar_path,
    )
    sensors.initialize()
    force_config = GripperForceConfig(
        target_force_n=4.0,
    )
    force_controller = GripperForceController(
        robot=robot,
        sensors=sensors,
        config=force_config,
    )

    ee_pos, ee_ori = robot.get_ee_pose()
    print(f"Initial End Effector Position: {ee_pos}")
    print(f"Initial End Effector Orientation: {ee_ori}")

    # 实例化键盘控制器并连接
    keyboard_device = KeyboardInputDevice(
        robot=robot,
        sensors=sensors,
        force_controller=force_controller,
        pos_scale=0.002,
        rpy_scale=0.01,
        gripper_scale=0.01,
    )
    keyboard_device.connect()

    print("✨ 完整场景加载成功，物理句柄挂载完毕，开始仿真...")
    return keyboard_device, force_controller


def run_simulation(world, keyboard_device, force_controller):
    # 仿真主循环
    while simulation_app.is_running():
        physics_dt = world.get_physics_dt()

        # 读取键盘指令并下发 IK 指令
        keyboard_device.update(dt=physics_dt)
        force_controller.update(dt=physics_dt)
        
        # 步进物理环境
        world.step(render=True)


keyboard_device = None
try:
    my_world = create_world()
    keyboard_device, force_controller = create_control_stack()
    run_simulation(my_world, keyboard_device, force_controller)
except KeyboardInterrupt:
    print("用户强制终止。")
finally:
    # 安全退出
    if keyboard_device is not None:
        keyboard_device.disconnect()
    simulation_app.close()
