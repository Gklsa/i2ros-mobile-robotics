# Copyright 2026 i2ros Team 6
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

from ament_index_python.packages import (
    get_package_prefix,
    get_package_share_directory,
)
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    bringup_dir = get_package_share_directory('ad_bringup')
    params_file = os.path.join(bringup_dir, 'config', 'params.yaml')
    planning_dir = get_package_share_directory('ad_planning')
    waypoint_file = os.path.join(planning_dir, 'config', 'waypoints.yaml')
    simulation_dir = get_package_share_directory('simulation')
    simulation_launch = os.path.join(
        simulation_dir, 'launch', 'simulation.launch.py'
    )

    # The Unity build is intentionally kept in the source tree because it is a
    # large external binary and is not installed as part of the ROS package.
    simulation_prefix = get_package_prefix('simulation')
    # colcon can use either isolated prefixes (install/simulation) or a merged
    # prefix (install), so derive the workspace root for both layouts.
    prefix_levels = 2 if os.path.basename(simulation_prefix) == 'simulation' else 1
    workspace_dir = os.path.abspath(
        os.path.join(simulation_prefix, *(['..'] * prefix_levels))
    )
    unity_executable = os.path.join(
        workspace_dir,
        'src',
        'simulation',
        'unity_sim',
        'Build_Ubuntu',
        'AD_Sim.x86_64',
    )
    simulation_config = os.path.join(
        os.path.dirname(unity_executable),
        'AD_Sim_Data',
        'StreamingAssets',
        'simulation_config.json',
    )
    car_model_path = os.path.join(
        workspace_dir,
        'src',
        'ad_perception',
        'trafficlight',
        'output',
        'runs',
        'detect',
        'car',
        'weights',
        'best.pt',
    )
    traffic_light_model_path = os.path.join(
        workspace_dir,
        'src',
        'ad_perception',
        'trafficlight',
        'output',
        'runs',
        'detect',
        'trafficlight_box_yolov8s',
        'weights',
        'best.pt',
    )

    return LaunchDescription([
        DeclareLaunchArgument('enable_bridge', default_value='true'),
        DeclareLaunchArgument('enable_unity', default_value='true'),
        DeclareLaunchArgument('enable_decision', default_value='true'),
        DeclareLaunchArgument('enable_controller', default_value='true'),
        DeclareLaunchArgument('enable_car_detector', default_value='true'),
        # 3D voxel-grid mapping (OctoMap) built from /perception/pcl/points.
        DeclareLaunchArgument('enable_octomap', default_value='true'),
        # Superseded by octomap_server's /projected_map. Off by default; flip
        # on only if octomap_server isn't installed on a machine.
        DeclareLaunchArgument('enable_occupancy_grid_2d_fallback', default_value='false'),
        DeclareLaunchArgument('initial_waypoint_index', default_value='0'),
        # Unity spawnIndex may start the car anywhere along the recorded
        # route. Match planning to the actual first CoM pose instead of
        # forcing every spawn to waypoint 0.
        DeclareLaunchArgument('initialize_from_nearest_waypoint', default_value='true'),
        DeclareLaunchArgument('car_model_path', default_value=car_model_path),
        DeclareLaunchArgument(
            'traffic_light_model_path',
            default_value=traffic_light_model_path,
        ),
        # Historical dataset collection arguments intentionally disabled for
        # the formal workflow. Car and traffic-light models are already
        # trained; runtime detection is handled by yolo_car_detector and
        # traffic_light_detector.

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(simulation_launch),
            launch_arguments={'config_file': simulation_config}.items(),
            condition=IfCondition(LaunchConfiguration('enable_bridge')),
        ),

        # Give the TCP bridge time to bind port 9998 before Unity connects.
        TimerAction(
            period=2.0,
            actions=[
                ExecuteProcess(
                    cmd=[unity_executable],
                    cwd=os.path.dirname(unity_executable),
                    output='screen',
                    condition=IfCondition(LaunchConfiguration('enable_unity')),
                ),
            ],
        ),

        Node(
            package='ad_perception',
            executable='sensor_tf_broadcaster',
            name='sensor_tf_broadcaster',
            output='screen',
            parameters=[params_file]
        ),

        Node(
            package='ad_perception',
            executable='traffic_light_detector',
            name='traffic_light_detector',
            output='screen',
            parameters=[
                params_file,
                {
                    'traffic_light_yolo_model_path': LaunchConfiguration(
                        'traffic_light_model_path'
                    ),
                },
            ]
        ),

        Node(
            package='ad_perception',
            executable='obstacle_detector',
            name='obstacle_detector',
            output='screen',
            parameters=[params_file]
        ),

        Node(
            package='ad_perception',
            executable='yolo_car_detector',
            name='yolo_car_detector',
            output='screen',
            parameters=[
                params_file,
                {'car_model_path': LaunchConfiguration('car_model_path')},
            ],
            condition=IfCondition(LaunchConfiguration('enable_car_detector')),
        ),

        # 3D occupancy mapping (PDF 2.2.3): builds a persistent voxel octree in
        # the 'world' frame from /perception/pcl/points_preprocessed (ground/
        # sky band filtered in obstacle_detector.py, independent of Unity's
        # groundRemoval flag - see ground_filter_min/max_vertical_m). octomap_
        # server's own 2D projection of the occupied voxels is remapped to
        # /perception/local_map so the topic name/type stays the same as the
        # old 2D-only generator below.
        Node(
            package='octomap_server',
            executable='octomap_server_node',
            name='octomap_server',
            output='screen',
            parameters=[{
                'frame_id': 'world',
                # Raised from 0.2m: fewer voxels to insert/render per frame,
                # meaningfully cuts octomap_server's CPU cost under the load
                # from running 2 CPU YOLO detectors + Unity + RViz at once.
                'resolution': 0.3,
                'sensor_model.max_range': 25.0,
            }],
            remappings=[
                ('cloud_in', '/perception/pcl/points_preprocessed'),
                ('projected_map', '/perception/local_map'),
            ],
            condition=IfCondition(LaunchConfiguration('enable_octomap')),
        ),

        # Hand-rolled 2D occupancy grid, superseded by octomap_server above.
        # Kept as a fallback (enable_occupancy_grid_2d_fallback:=true) for a
        # machine where octomap_server isn't installed.
        Node(
            package='ad_perception',
            executable='occupancy_grid_generator',
            name='occupancy_grid_generator',
            output='screen',
            parameters=[params_file],
            condition=IfCondition(LaunchConfiguration('enable_occupancy_grid_2d_fallback')),
        ),

        # trafficlight_dataset_collector is a historical image-capture tool.
        # It is not launched in the formal perception stack.

        Node(
            package='ad_planning',
            executable='waypoint_manager',
            name='waypoint_manager',
            output='screen',
            parameters=[
                params_file,
                waypoint_file,
                {
                    'initial_waypoint_index': ParameterValue(
                        LaunchConfiguration('initial_waypoint_index'),
                        value_type=int,
                    ),
                    'initialize_from_nearest_waypoint': ParameterValue(
                        LaunchConfiguration('initialize_from_nearest_waypoint'),
                        value_type=bool,
                    ),
                },
            ]
        ),

        Node(
            package='ad_planning',
            executable='trajectory_planner',
            name='trajectory_planner',
            output='screen',
            parameters=[params_file, waypoint_file]
        ),

        Node(
            package='ad_decision',
            executable='behavior_state_machine',
            name='behavior_state_machine',
            output='screen',
            parameters=[params_file],
            condition=IfCondition(LaunchConfiguration('enable_decision'))
        ),

        # Start control after Unity and the upstream ROS nodes have had time to
        # publish their first pose, twist, trajectory, and decision messages.
        TimerAction(
            period=8.0,
            actions=[
                Node(
                    package='dummy_controller',
                    executable='dummy_controller_node',
                    name='dummy_controller_node',
                    output='screen',
                    condition=IfCondition(
                        LaunchConfiguration('enable_controller')
                    ),
                ),
            ],
        ),
    ])
