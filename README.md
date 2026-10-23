# I2ROS - Mobile Robotics Team Project

ROS 2 autonomous-driving coursework combining vehicle perception, waypoint planning, driving decisions and closed-loop control through a Unity simulator bridge.

This repository contains the source and tests for review. The Unity simulator build, its configuration and trained detector weights are supplied separately; cloning this repository alone does not provide a runnable simulation.

## Structure

| Path | Contents |
| --- | --- |
| `src/ad_perception/` | Car and traffic-light detection, depth processing, sensor transforms and occupancy mapping |
| `src/ad_planning/` | Waypoint management, trajectory generation and route configuration |
| `src/ad_decision/` | Driving state machine, traffic-light stops and obstacle responses |
| `src/dummy_controller/` | C++ vehicle controller |
| `src/simulation/` | Unity TCP/UDP bridge and vehicle-control message |
| `src/ad_interfaces/` | Shared ROS messages and waypoint service |
| `src/ad_bringup/` | Integrated launch file and runtime parameters |
| `tests/`, `src/*/test/` | Mapping, planning, decision, controller and package checks |
| `tools/` | Manual simulation controls, debugging and waypoint recording |

## Environment and build

Use **Ubuntu 24.04 with ROS 2 Jazzy**, Python 3.12, a C++17 compiler, `colcon` and `rosdep`. For the complete simulation, a Linux desktop with OpenGL support is also required. Start with the [ROS 2 Jazzy installation guide](https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html) and initialize `rosdep` if it has not been configured.

Run these commands in Bash after installing ROS 2:

```bash
git clone https://github.com/Gklsa/i2ros-mobile-robotics.git
cd i2ros-mobile-robotics
source /opt/ros/jazzy/setup.bash

sudo apt install python3-venv ros-dev-tools ros-jazzy-octomap-server
rosdep update
rosdep install --from-paths src --ignore-src -r -y

/usr/bin/python3 -m venv --system-site-packages .venv
source .venv/bin/activate
touch .venv/COLCON_IGNORE
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install ultralytics==8.3.100

python /usr/bin/colcon build --symlink-install
source install/setup.bash
```

The Python environment uses Ubuntu's interpreter so that the installed ROS Python packages remain compatible. Calling the `colcon` script through this environment also makes its installed Python packages available to the built nodes. Activate `.venv` and source both ROS and `install/setup.bash` in each new terminal used with this workspace. The CPU PyTorch packages above do not require an NVIDIA GPU.

## Tests

After building, run the package tests without starting the integrated driving stack:

```bash
python /usr/bin/colcon test
colcon test-result --verbose
python -m pytest tests/test_occupancy_grid_generator.py
```

These checks cover the available unit and package tests; they do not replace an end-to-end simulator run.

## Complete simulation

Obtain the corresponding course/project runtime assets separately and preserve these paths:

```text
src/simulation/unity_sim/Build_Ubuntu/
  AD_Sim.x86_64
  AD_Sim_Data/StreamingAssets/simulation_config.json
  ...the remaining Unity build files
src/ad_perception/trafficlight/output/runs/detect/
  car/weights/best.pt
  trafficlight_box_yolov8s/weights/best.pt
```

The simulator and weights have no download bundled with this snapshot. Use matching assets from the course project; an arbitrary Unity build or YOLO checkpoint is not a replacement.

Configure simulator connections in `simulation_config.json`, runtime parameters in [params.yaml](src/ad_bringup/config/params.yaml) and the route in [waypoints.yaml](src/ad_planning/config/waypoints.yaml). The bridge defaults use the local host with TCP port `9998` for sensor input and UDP port `6657` for control; match the simulator configuration to the machine running it.

With the assets in place, from the repository root:

```bash
bash src/setup_script.sh
make check-models
make set-spawn 0
ros2 launch ad_bringup autonomous_driving.launch.py
```

The integrated launch starts Unity, its bridge, perception, mapping, planning, decision and control. Close the run with `Ctrl+C`. Detector paths can be overridden with the launch arguments `car_model_path` and `traffic_light_model_path`.

## Attribution

Course and team project code retains its original authorship and notices, including the I2ROS Team 6 and simulator starter code. Third-party components retain their own notices. See [LICENSE](LICENSE).
