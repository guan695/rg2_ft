# Isaac Sim RG2-FT Teleoperation & Force-Controlled Grasping

[![Stars](https://img.shields.io/github/stars/guan695/rg2_ft?style=flat-square)](https://github.com/guan695/rg2_ft/stargazers)
[![Forks](https://img.shields.io/github/forks/guan695/rg2_ft?style=flat-square)](https://github.com/guan695/rg2_ft/network/members)
[![Issues](https://img.shields.io/github/issues/guan695/rg2_ft?style=flat-square)](https://github.com/guan695/rg2_ft/issues)
[![Top language](https://img.shields.io/github/languages/top/guan695/rg2_ft?style=flat-square)](https://github.com/guan695/rg2_ft)

An experimental **robotics simulation** project for **NVIDIA Isaac Sim**: control a
**UR5/UR5e** arm with an **RG2-FT gripper** using keyboard teleoperation,
Lula inverse kinematics, force/torque sensing, fingertip lidar feedback, and
constant-force grasp control.

The project is useful as a compact starting point for Isaac Sim manipulation
experiments, sensor integration, and force-aware gripper behaviors.

## Highlights

- **Interactive teleoperation**: move and rotate the end effector from the keyboard.
- **UR5 and UR5e support**: switch robot models with one environment variable.
- **Lula IK control**: convert end-effector commands into robot joint targets.
- **RG2-FT sensing**: read fingertip force, torque, and distance telemetry.
- **Force-aware grasping**: search for contact and maintain a configurable target force.
- **GPU physics setup**: start the simulation with GPU dynamics and GPU broadphase enabled.

## Requirements

- NVIDIA Isaac Sim with a Python environment where `import isaacsim` works.
- A compatible NVIDIA GPU and a graphical Isaac Sim session.
- The robot description files and USD scene assets expected by
  `keyboard_control/RobotConfig.py`; see [Runtime assets and scene paths](#runtime-assets-and-scene-paths).

## Quick start

Clone the repository and run the main keyboard-control entry point with the
Python interpreter supplied by your Isaac Sim installation:

```bash
git clone https://github.com/guan695/rg2_ft.git
cd rg2_ft

python3 keyboard_control/run.py
```

The default robot model is `ur5e`. The default force-control target is
4.0 N in the current simulation configuration.

## Choose a robot model

Use `RG2_ROBOT_MODEL` to select the robot without editing the source:

```bash
RG2_ROBOT_MODEL=ur5e python3 keyboard_control/run.py
RG2_ROBOT_MODEL=ur5  python3 keyboard_control/run.py
```

Supported values are `ur5` and `ur5e`.

## Override the USD scene

The built-in scene paths are local development defaults. Point the controller at
your own scene with `RG2_SCENE_USD`:

```bash
RG2_SCENE_USD=/path/to/your/scene.usd python3 keyboard_control/run.py
```

## Keyboard controls

| Key | Action |
| --- | --- |
| `W/S`, `A/D`, `Q/E` | Translate the end effector along X/Y/Z |
| `I/K`, `J/L`, `U/O` | Rotate pitch/roll/yaw |
| `Z/X` | Close/open the gripper manually |
| `G` | Print RG2-FT force, torque, and distance telemetry |
| `C` | Start automatic constant-force grasping |
| `V` | Release the gripper |
| `B` | Clear a force-control fault |
| `N` | Zero force/torque bias while the gripper is unloaded |

## Project structure

- [`keyboard_control/run.py`](keyboard_control/run.py): Isaac Sim startup and main loop.
- [`keyboard_control/RobotConfig.py`](keyboard_control/RobotConfig.py): UR5/UR5e model configuration.
- [`keyboard_control/Robot.py`](keyboard_control/Robot.py): robot articulation, Lula IK, and arm/gripper commands.
- [`keyboard_control/KeyboardInputDevice.py`](keyboard_control/KeyboardInputDevice.py): keyboard event handling and command mapping.
- [`keyboard_control/RG2FTSensors.py`](keyboard_control/RG2FTSensors.py): RG2-FT wrench and fingertip lidar reads.
- [`keyboard_control/GripperForceController.py`](keyboard_control/GripperForceController.py): contact search and constant-force control.
- [`keyboard_control/test/sensor_test.py`](keyboard_control/test/sensor_test.py): sensor telemetry test entry point.

## Runtime assets and scene paths

The controller resolves robot files from a repository-level `assets/` directory
and uses local USD scene paths by default. Before running on another machine:

1. Provide the UR5/UR5e URDF and Lula robot-description files expected by
   `RobotConfig.py`.
2. Provide a USD scene containing the selected robot, RG2-FT gripper, and fingertip lidar prims.
3. Set `RG2_SCENE_USD` to the scene file on your machine.

The public repository currently focuses on the Python control stack. Scene files and
machine-specific assets should be kept local unless they are ready to be shared.

## Notes and safety

- This is experimental simulation code, not an industrial safety system.
- Keep real-robot velocity, acceleration, workspace, and emergency-stop limits
  independent from Isaac Sim drive parameters.
- Warnings about unresolved USD references or missing prim paths usually indicate
  that a local scene or asset path needs to be fixed.
- Do not commit generated files such as `__pycache__/` or `*.pyc`.

## Contributing

If you try this with another Isaac Sim release, robot model, sensor setup, or scene,
please open an issue or pull request with the environment details and observed
behavior. If the project helps with your manipulation experiments, a star is a
simple way to support future documentation and reproducibility work.
