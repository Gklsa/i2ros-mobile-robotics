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

"""Depth-image obstacle detector and point cloud generator."""

from __future__ import annotations

import math
from struct import pack

from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from std_msgs.msg import Float32


class ObstacleDetector(Node):
    """Convert depth images into obstacle outputs for planning and decision."""

    def __init__(self) -> None:
        super().__init__("obstacle_detector")

        self.declare_parameter(
            "depth_image_topic",
            "/OurCar/Sensors/DepthCamera/image_raw",
        )
        self.declare_parameter(
            "depth_camera_info_topic",
            "/OurCar/Sensors/DepthCamera/camera_info",
        )
        self.declare_parameter("points_topic", "/perception/pcl/points")
        self.declare_parameter(
            "preprocessed_points_topic",
            "/perception/pcl/points_preprocessed",
        )
        self.declare_parameter(
            "front_obstacle_distance_topic",
            "/perception/front_obstacle_distance",
        )
        self.declare_parameter("pixel_step", 4)
        self.declare_parameter("min_depth_m", 0.2)
        self.declare_parameter("max_depth_m", 50.0)
        self.declare_parameter("obstacle_roi_width_ratio", 0.35)
        self.declare_parameter("obstacle_roi_height_ratio", 0.45)
        self.declare_parameter("obstacle_min_valid_pixels", 20)
        self.declare_parameter("obstacle_lateral_limit", 2.0)
        self.declare_parameter("obstacle_min_forward_distance", 0.5)
        self.declare_parameter("obstacle_max_forward_distance", 25.0)
        self.declare_parameter("obstacle_min_vertical", -1.5)
        self.declare_parameter("obstacle_max_vertical", 0.8)
        self.declare_parameter("depth_camera_fallback_fov_deg", 60.0)
        # Ground/sky band filter for the *preprocessed* cloud, independent of
        # Unity's groundRemoval config flag. Points are still in the camera's
        # own optical frame here, so this is the same "vertical" quantity
        # (y = (v-cy)*z/fy) and the same -4.0/2.5 band occupancy_grid_generator
        # used for its own local map - a level-mounted camera sees a flat
        # ground point at roughly +camera_height regardless of range, so this
        # band reliably separates road/curb from real obstacles even if
        # groundRemoval is ever left off upstream.
        self.declare_parameter("ground_filter_min_vertical_m", -4.0)
        self.declare_parameter("ground_filter_max_vertical_m", 2.5)

        self._bridge = CvBridge()
        self._camera_info: CameraInfo | None = None
        self._pixel_step = max(1, int(self.get_parameter("pixel_step").value))
        self._min_depth_m = float(self.get_parameter("min_depth_m").value)
        self._max_depth_m = float(self.get_parameter("max_depth_m").value)
        self._roi_width_ratio = float(
            self.get_parameter("obstacle_roi_width_ratio").value
        )
        self._roi_height_ratio = float(
            self.get_parameter("obstacle_roi_height_ratio").value
        )
        self._min_valid_pixels = int(
            self.get_parameter("obstacle_min_valid_pixels").value
        )
        self._obstacle_lateral_limit = float(
            self.get_parameter("obstacle_lateral_limit").value
        )
        self._obstacle_min_forward_distance = float(
            self.get_parameter("obstacle_min_forward_distance").value
        )
        self._obstacle_max_forward_distance = float(
            self.get_parameter("obstacle_max_forward_distance").value
        )
        self._obstacle_min_vertical = float(
            self.get_parameter("obstacle_min_vertical").value
        )
        self._obstacle_max_vertical = float(
            self.get_parameter("obstacle_max_vertical").value
        )
        self._fallback_fov_deg = float(
            self.get_parameter("depth_camera_fallback_fov_deg").value
        )
        self._ground_filter_min_vertical_m = float(
            self.get_parameter("ground_filter_min_vertical_m").value
        )
        self._ground_filter_max_vertical_m = float(
            self.get_parameter("ground_filter_max_vertical_m").value
        )
        self._warned_fallback_info = False

        self._points_pub = self.create_publisher(
            PointCloud2,
            self.get_parameter("points_topic").value,
            10,
        )
        self._preprocessed_points_pub = self.create_publisher(
            PointCloud2,
            self.get_parameter("preprocessed_points_topic").value,
            10,
        )
        self._distance_pub = self.create_publisher(
            Float32,
            self.get_parameter("front_obstacle_distance_topic").value,
            10,
        )

        self.create_subscription(
            CameraInfo,
            self.get_parameter("depth_camera_info_topic").value,
            self._camera_info_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Image,
            self.get_parameter("depth_image_topic").value,
            self._depth_callback,
            qos_profile_sensor_data,
        )

        self.get_logger().info(
            "Obstacle detector waiting for depth image and camera_info."
        )

    def _camera_info_callback(self, msg: CameraInfo) -> None:
        self._camera_info = msg

    def _depth_callback(self, msg: Image) -> None:
        if self._camera_info is None:
            self._camera_info = self._make_fallback_camera_info(msg)
            if not self._warned_fallback_info:
                self.get_logger().warn(
                    "Depth camera_info is not available; using fallback intrinsics "
                    f"from image size and {self._fallback_fov_deg:.1f} deg FOV."
                )
                self._warned_fallback_info = True

        try:
            depth = self._bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except CvBridgeError as exc:
            self.get_logger().warn(f"Failed to convert depth image: {exc}")
            self._publish_distance(math.inf)
            return

        depth_m = self._depth_to_meters(depth, msg.encoding)
        cloud, preprocessed_cloud = self._depth_to_clouds(msg, depth_m)
        self._points_pub.publish(cloud)
        self._preprocessed_points_pub.publish(preprocessed_cloud)
        self._publish_distance(self._estimate_front_distance(depth_m))

    def _depth_to_meters(self, depth: np.ndarray, encoding: str) -> np.ndarray:
        depth_m = np.asarray(depth, dtype=np.float32)
        if encoding.upper() == "16UC1":
            return depth_m / 1000.0
        return depth_m

    def _depth_to_clouds(
        self, image_msg: Image, depth_m: np.ndarray
    ) -> tuple[PointCloud2, PointCloud2]:
        assert self._camera_info is not None
        fx = self._camera_info.k[0]
        fy = self._camera_info.k[4]
        cx = self._camera_info.k[2]
        cy = self._camera_info.k[5]

        points = []
        preprocessed_points = []
        for v in range(0, image_msg.height, self._pixel_step):
            for u in range(0, image_msg.width, self._pixel_step):
                depth = float(depth_m[v, u])
                if not math.isfinite(depth):
                    continue
                if depth < self._min_depth_m or depth > self._max_depth_m:
                    continue

                x = (u - cx) * depth / fx
                y = (v - cy) * depth / fy
                z = depth
                points.append((x, y, z))
                if (
                    self._ground_filter_min_vertical_m
                    <= y
                    <= self._ground_filter_max_vertical_m
                ):
                    preprocessed_points.append((x, y, z))

        raw_cloud = self._make_cloud(image_msg.header, points)
        preprocessed_cloud = self._make_cloud(image_msg.header, preprocessed_points)
        return raw_cloud, preprocessed_cloud

    def _estimate_front_distance(self, depth_m: np.ndarray) -> float:
        if depth_m.size == 0:
            return math.inf

        height, width = depth_m.shape[:2]
        half_w = max(1, int(width * self._roi_width_ratio / 2.0))
        roi_h = max(1, int(height * self._roi_height_ratio))
        center_x = width // 2

        x0 = max(0, center_x - half_w)
        x1 = min(width, center_x + half_w)
        y0 = max(0, int(height * 0.35))
        y1 = min(height, y0 + roi_h)
        roi = depth_m[y0:y1, x0:x1]

        valid = roi[np.isfinite(roi)]
        valid = valid[(valid > self._min_depth_m) & (valid < self._max_depth_m)]
        if valid.size < self._min_valid_pixels:
            return math.inf

        return float(np.percentile(valid, 10))

    def _publish_distance(self, distance: float) -> None:
        msg = Float32()
        msg.data = float(distance)
        self._distance_pub.publish(msg)

    def _make_cloud(self, header, points: list[tuple[float, float, float]]) -> PointCloud2:
        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        data = b"".join(pack("fff", *point) for point in points)

        cloud = PointCloud2()
        cloud.header = header
        cloud.height = 1
        cloud.width = len(points)
        cloud.fields = fields
        cloud.is_bigendian = False
        cloud.point_step = 12
        cloud.row_step = cloud.point_step * len(points)
        cloud.data = data
        cloud.is_dense = True
        return cloud

    def _make_fallback_camera_info(self, image_msg: Image) -> CameraInfo:
        info = CameraInfo()
        info.header = image_msg.header
        info.width = image_msg.width
        info.height = image_msg.height
        info.distortion_model = "plumb_bob"

        cx = float(image_msg.width) * 0.5
        cy = float(image_msg.height) * 0.5
        fov_rad = math.radians(max(1.0, min(179.0, self._fallback_fov_deg)))
        fx = 0.5 * float(image_msg.height) / math.tan(fov_rad * 0.5)
        fy = fx

        info.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
        info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        info.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        info.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        return info


def main(args: list[str] | None = None) -> None:
    """Run the obstacle detector node."""
    rclpy.init(args=args)
    node = ObstacleDetector()
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
