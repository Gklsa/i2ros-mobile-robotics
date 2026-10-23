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

"""Functional checks for the static sensor transforms."""


def test_rgb_camera_lateral_positions_follow_ros_vehicle_convention():
    """Place the left camera on positive Y and the right camera on negative Y."""
    # Import ROS only when this test runs. Importing rclpy during pytest's
    # collection phase starts middleware threads before ament_flake8 forks
    # workers, which makes Python 3.12 emit a fork-in-multithreaded warning.
    from ad_perception.sensor_tf_broadcaster import SensorTfBroadcaster
    import rclpy

    rclpy.init()
    node = SensorTfBroadcaster()
    try:
        transforms = {
            transform.child_frame_id: transform
            for transform in node._build_transforms()
        }
        left = transforms['OurCar/Sensors/RGBCameraLeft'].transform.translation
        right = transforms['OurCar/Sensors/RGBCameraRight'].transform.translation

        assert left.y == 0.2
        assert right.y == -0.2
        assert left.x == right.x == 0.82
        assert left.z == right.z == 0.56
    finally:
        node.destroy_node()
        rclpy.shutdown()
