# rg2_ft

Isaac Sim keyboard-control prototype for a UR5 robot with an RG2-FT gripper.

This repository currently contains the `keyboard_control` package used to load an
Isaac Sim scene, teleoperate the UR5 end effector from the keyboard, read RG2-FT
finger force/torque and distance signals, and run a simple constant-force grasp
controller.

## Modules

- `keyboard_control/run.py` starts Isaac Sim, opens the USD scene, creates the
  robot, sensors, keyboard device, and force controller, then runs the simulation
  loop.
- `keyboard_control/Robot.py` wraps the UR5 articulation, Lula inverse
  kinematics, end-effector pose commands, and RG2 gripper joint command.
- `keyboard_control/KeyboardInputDevice.py` maps keyboard input to Cartesian
  motion, gripper motion, telemetry printing, grasp, release, fault reset, and
  force-bias calibration commands.
- `keyboard_control/RG2FTSensors.py` reads left/right fingertip wrench data and
  lidar distance data from Isaac Sim.
- `keyboard_control/GripperForceController.py` implements contact search,
  PI-based force regulation, hold, release, and fault protection states.
- `keyboard_control/test/sensor_test.py` is a small sensor telemetry test entry.

## Usage

Run inside an Isaac Sim Python environment:

```bash
python keyboard_control/run.py
```

By default the script opens:

```text
/home/guan/Desktop/ur5_rg2/my_scene/pick.usd
```

Use `RG2_SCENE_USD` to load another scene:

```bash
RG2_SCENE_USD=/path/to/scene.usd python keyboard_control/run.py
```

## Keyboard Controls

- `W/S`, `A/D`, `Q/E`: translate the end effector along X/Y/Z
- `I/K`, `J/L`, `U/O`: rotate pitch/roll/yaw
- `Z/X`: close/open the gripper manually
- `G`: print RG2-FT telemetry
- `C`: start automatic constant-force grasp
- `V`: release
- `B`: clear force-control fault
- `N`: zero force/torque bias while unloaded

## Notes

The controller parameters are tuned for simulation experiments and should not be
treated as an industrial safety system. The Isaac Sim scene must contain the UR5
and RG2-FT prims at the paths expected by `Robot.py` and `RG2FTSensors.py`.
