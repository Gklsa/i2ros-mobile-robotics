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

"""
Open the RViz point cloud, 3D octomap, and 2D local-map view.

Only visualizes - starts nothing else. Assumes the main stack (Unity +
autonomous_driving.launch.py, e.g. via `make vision`/`make drive`) is
already running and publishing /perception/pcl/points_preprocessed,
/octomap_full and /perception/local_map.
"""

from launch import LaunchDescription
from launch.actions import SetEnvironmentVariable
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """Create a standalone RViz launch description."""
    rviz_config = PathJoinSubstitution(
        [FindPackageShare('ad_perception'), 'rviz', 'perception_debug.rviz']
    )

    return LaunchDescription([
        # Works around a broken ros-jazzy-octomap-rviz-plugins .deb:
        # liboctomap_rviz_plugins.so is missing liboctomap.so from its
        # NEEDED list, so dlopen fails with "undefined symbol" for
        # OcTreeStamped's typeinfo unless liboctomap is already loaded in
        # the process first.
        SetEnvironmentVariable(
            'LD_PRELOAD', '/usr/lib/x86_64-linux-gnu/liboctomap.so'
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='perception_rviz',
            arguments=['-d', rviz_config],
            output='screen',
        ),
    ])
