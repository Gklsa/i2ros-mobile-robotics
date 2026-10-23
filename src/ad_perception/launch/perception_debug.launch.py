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

"""Open the Perception image debug view without starting detector nodes."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Create a standalone rqt_image_view launch description."""
    return LaunchDescription([
        DeclareLaunchArgument(
            'image_topic',
            default_value='/perception/debug/detection_combined',
            description='Debug image topic displayed by rqt_image_view.',
        ),
        Node(
            package='rqt_image_view',
            executable='rqt_image_view',
            name='perception_debug_view',
            arguments=[LaunchConfiguration('image_topic')],
            output='screen',
        ),
    ])
