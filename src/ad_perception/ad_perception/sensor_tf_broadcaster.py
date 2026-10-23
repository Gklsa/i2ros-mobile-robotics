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

"""Static sensor transform broadcaster for the simulated ego vehicle."""

from __future__ import annotations

from geometry_msgs.msg import TransformStamped
import rclpy
from rclpy.node import Node
from tf2_ros import StaticTransformBroadcaster


class SensorTfBroadcaster(Node):
    """Publish static transforms for vehicle and sensor frames."""

    def __init__(self) -> None:
        super().__init__("sensor_tf_broadcaster")

        self.declare_parameter("ins_frame_id", "OurCar/INS")
        self.declare_parameter("vehicle_center_frame_id", "OurCar/Center")
        self.declare_parameter("depth_camera_frame_id", "OurCar/Sensors/DepthCamera")
        self.declare_parameter("rgb_left_frame_id", "OurCar/Sensors/RGBCameraLeft")
        self.declare_parameter("rgb_right_frame_id", "OurCar/Sensors/RGBCameraRight")
        self.declare_parameter("semantic_camera_frame_id", "OurCar/Sensors/SegCamera")
        self.declare_parameter(
            "semantic_camera_alias_frame_id",
            "OurCar/Sensors/SemanticCamera",
        )

        self._broadcaster = StaticTransformBroadcaster(self)
        transforms = self._build_transforms()
        self._broadcaster.sendTransform(transforms)
        self.get_logger().info(f"Published {len(transforms)} static sensor transforms")

    def _build_transforms(self) -> list[TransformStamped]:
        ins = self.get_parameter("ins_frame_id").value
        center = self.get_parameter("vehicle_center_frame_id").value
        depth = self.get_parameter("depth_camera_frame_id").value
        rgb_left = self.get_parameter("rgb_left_frame_id").value
        rgb_right = self.get_parameter("rgb_right_frame_id").value
        semantic = self.get_parameter("semantic_camera_frame_id").value
        semantic_alias = self.get_parameter("semantic_camera_alias_frame_id").value

        # Camera optical frame in the INS frame, matching the project figures:
        #   camera +Z -> INS +X  (vehicle forward)
        #   camera +X -> INS -Y
        #   camera +Y -> INS -Z
        camera_rotation = (0.5, -0.5, 0.5, -0.5)
        center_to_ins_z = -0.71
        sensor_x = 0.82
        sensor_z = 0.56
        camera_center = (sensor_x, 0.0, sensor_z)
        # INS uses the ROS vehicle convention: +Y is left and -Y is right.
        rgb_left_position = (sensor_x, 0.2, sensor_z)
        rgb_right_position = (sensor_x, -0.2, sensor_z)

        return [
            self._make_transform(
                ins,
                center,
                (0.0, 0.0, center_to_ins_z),
                (0.0, 0.0, 0.0, 1.0),
            ),
            self._make_transform(ins, semantic, camera_center, camera_rotation),
            self._make_transform(ins, depth, camera_center, camera_rotation),
            self._make_transform(ins, rgb_left, rgb_left_position, camera_rotation),
            self._make_transform(ins, rgb_right, rgb_right_position, camera_rotation),
            self._make_transform(
                ins,
                semantic_alias,
                camera_center,
                camera_rotation,
            ),
        ]

    def _make_transform(
        self,
        parent: str,
        child: str,
        translation: tuple[float, float, float],
        rotation: tuple[float, float, float, float],
    ) -> TransformStamped:
        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = parent
        transform.child_frame_id = child
        transform.transform.translation.x = translation[0]
        transform.transform.translation.y = translation[1]
        transform.transform.translation.z = translation[2]
        transform.transform.rotation.x = rotation[0]
        transform.transform.rotation.y = rotation[1]
        transform.transform.rotation.z = rotation[2]
        transform.transform.rotation.w = rotation[3]
        return transform


def main(args: list[str] | None = None) -> None:
    """Run the sensor TF broadcaster node."""
    rclpy.init(args=args)
    node = SensorTfBroadcaster()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
