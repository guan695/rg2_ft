# rg2_ft

Isaac Sim keyboard-control project for UR5/UR5e simulation with an RG2-FT gripper.

The `keyboard_control` package loads a USD scene, creates the robot articulation,
runs Lula inverse kinematics, maps keyboard input to end-effector/gripper
commands, reads RG2-FT force/torque and lidar distance signals, and provides a
simple constant-force grasp controller.

## Modules

- `keyboard_control/run.py`: main entry point for simulation and keyboard teleop.
- `keyboard_control/RobotConfig.py`: central UR5/UR5e model configuration.
- `keyboard_control/Robot.py`: robot articulation, IK solver, arm and gripper commands.
- `keyboard_control/KeyboardInputDevice.py`: keyboard listener and command mapping.
- `keyboard_control/RG2FTSensors.py`: RG2-FT force/torque and lidar distance reads.
- `keyboard_control/GripperForceController.py`: contact search and constant-force gripper control.
- `keyboard_control/test/sensor_test.py`: small sensor telemetry test entry.
- `assets/`: local UR5/UR5e URDF and Lula YAML files.

## Select Robot Model

The current default model is `ur5e`. To switch between UR5 and UR5e, edit
`keyboard_control/run.py`:

```python
ROBOT_MODEL = "ur5"
```

or:

```python
ROBOT_MODEL = "ur5e"
```

You can also override it without editing the file:

```bash
RG2_ROBOT_MODEL=ur5 python3 keyboard_control/run.py
RG2_ROBOT_MODEL=ur5e python3 keyboard_control/run.py
```

`RobotConfig.py` derives the robot prim path, end-effector path, Lula URDF/YAML
files, RG2-FT lidar paths, and default USD scene from the selected model.

## Run

Run inside an Isaac Sim Python environment:

```bash
python3 keyboard_control/run.py
```

Default scenes:

- UR5: `/home/guan/Desktop/ur5_rg2/my_scene/pick.usd`
- UR5e: `/home/guan/Desktop/ur5e_rg2/my_scene/pick.usd`

Use `RG2_SCENE_USD` to load another scene:

```bash
RG2_SCENE_USD=/path/to/scene.usd python3 keyboard_control/run.py
```

## Keyboard Controls

- `W/S`, `A/D`, `Q/E`: translate the end effector along X/Y/Z.
- `I/K`, `J/L`, `U/O`: rotate pitch/roll/yaw.
- `Z/X`: close/open the gripper manually.
- `G`: print RG2-FT telemetry.
- `C`: start automatic constant-force grasp.
- `V`: release.
- `B`: clear force-control fault.
- `N`: zero force/torque bias while unloaded.

## Environment Variables

- `RG2_ROBOT_MODEL`: `ur5` or `ur5e`.
- `RG2_SCENE_USD`: override the default scene path.

## Notes

Isaac Sim may print warnings about unresolved USD references or missing scene
prim paths. Those warnings usually come from the USD scene assets, not from the
Python control code. Fix the referenced USD paths in the scene when they point
to missing assets.

The controller parameters are tuned for simulation experiments and should not be
treated as an industrial safety system. For real robot migration, send smooth
position/trajectory targets through the robot controller and keep real-machine
velocity, acceleration, workspace, and emergency-stop limits independent from
Isaac Sim drive parameters.

Do not commit generated caches such as `__pycache__/` or `*.pyc`.
