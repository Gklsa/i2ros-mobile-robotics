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

"""Generate a local occupancy grid from the simulated depth camera."""

from __future__ import annotations

import math

from cv_bridge import CvBridge, CvBridgeError
from nav_msgs.msg import OccupancyGrid
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image


class OccupancyGridGenerator(Node):
    """Project depth pixels into a local 2D occupancy grid."""

    def __init__(self) -> None:
        super().__init__("occupancy_grid_generator")

        self.declare_parameter(
            "depth_image_topic",
            "/OurCar/Sensors/DepthCamera/image_raw",
        )
        self.declare_parameter(
            "depth_camera_info_topic",
            "/OurCar/Sensors/DepthCamera/camera_info",
        )
        self.declare_parameter("occupancy_grid_topic", "/perception/local_map")
        self.declare_parameter("local_grid_frame_id", "OurCar/Sensors/DepthCamera")
        self.declare_parameter("occupancy_resolution", 0.25)
        self.declare_parameter("occupancy_forward_range", 30.0)
        self.declare_parameter("occupancy_lateral_range", 16.0)
        self.declare_parameter("occupancy_max_depth", 40.0)
        self.declare_parameter("occupancy_pixel_stride", 4)
        self.declare_parameter("occupancy_mark_free_space", True)
        self.declare_parameter("depth_camera_fallback_fov_deg", 60.0)

        self._bridge = CvBridge()
        self._camera_info: CameraInfo | None = None

        depth_topic = self.get_parameter("depth_image_topic").value
        info_topic = self.get_parameter("depth_camera_info_topic").value
        map_topic = self.get_parameter("occupancy_grid_topic").value

        self._resolution = float(self.get_parameter("occupancy_resolution").value)
        self._forward_range = float(
            self.get_parameter("occupancy_forward_range").value
        )
        self._lateral_range = float(
            self.get_parameter("occupancy_lateral_range").value
        )
        self._max_depth = float(self.get_parameter("occupancy_max_depth").value)
        self._pixel_stride = max(
            1,
            int(self.get_parameter("occupancy_pixel_stride").value),
        )
        self._mark_free_space = bool(
            self.get_parameter("occupancy_mark_free_space").value
        )
        self._fallback_fov_deg = float(
            self.get_parameter("depth_camera_fallback_fov_deg").value
        )
        self._warned_fallback_info = False

        self._map_pub = self.create_publisher(OccupancyGrid, map_topic, 5)
        self._info_sub = self.create_subscription(
            CameraInfo,
            info_topic,
            self._camera_info_callback,
            qos_profile_sensor_data,
        )
        self._depth_sub = self.create_subscription(
            Image,
            depth_topic,
            self._depth_callback,
            qos_profile_sensor_data,
        )

        self.get_logger().info(
            f"Occupancy grid generator subscribing to {depth_topic} and "
            f"{info_topic}, publishing {map_topic}"
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
            return

        grid = self._build_grid(depth, msg.encoding)
        grid.header.stamp = msg.header.stamp
        grid.header.frame_id = self.get_parameter("local_grid_frame_id").value
        self._map_pub.publish(grid)

    def _build_grid(self, depth: np.ndarray, encoding: str) -> OccupancyGrid:
        depth_m = np.asarray(depth, dtype=np.float32)
        if encoding.upper() == "16UC1":
            depth_m = depth_m / 1000.0

        width_cells = max(1, int(self._forward_range / self._resolution))
        height_cells = max(1, int(self._lateral_range / self._resolution))
        data = np.full((height_cells, width_cells), -1, dtype=np.int8)

        fx, fy, cx, cy = self._camera_intrinsics()
        height, width = depth_m.shape[:2]

        for v in range(0, height, self._pixel_stride):
            for u in range(0, width, self._pixel_stride):
                z = float(depth_m[v, u])
                if not math.isfinite(z) or z <= 0.05 or z > self._max_depth:
                    continue

                lateral = (u - cx) * z / fx
                vertical = (v - cy) * z / fy

                # Ignore sky/ceiling-like points and points far below the road.
                if vertical < -4.0 or vertical > 2.5:
                    continue

                cell_x, cell_y = self._point_to_cell(z, lateral)
                if cell_x is None or cell_y is None:
                    continue

                if not (0 <= cell_y < data.shape[0] and 0 <= cell_x < data.shape[1]):
                    continue

                if self._mark_free_space:
                    self._mark_ray_free(data, cell_x, cell_y)

                data[cell_y, cell_x] = 100

        return self._make_grid_message(data)

    def _camera_intrinsics(self) -> tuple[float, float, float, float]:
        assert self._camera_info is not None
        k = self._camera_info.k
        fx = k[0] if k[0] > 0.0 else 1.0
        fy = k[4] if k[4] > 0.0 else fx
        cx = k[2]
        cy = k[5]
        return fx, fy, cx, cy

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

    def _point_to_cell(
        self,
        forward: float,
        lateral: float,
    ) -> tuple[int | None, int | None]:
        if forward < 0.0 or forward >= self._forward_range:
            return None, None

        half_lateral = self._lateral_range / 2.0
        if lateral < -half_lateral or lateral >= half_lateral:
            return None, None

        cell_x = int(forward / self._resolution)
        cell_y = int((lateral + half_lateral) / self._resolution)
        return cell_x, cell_y

    def _mark_ray_free(
        self,
        data: np.ndarray,
        end_x: int,
        end_y: int,
    ) -> None:
        start_y = data.shape[0] // 2
        steps = max(abs(end_x), abs(end_y - start_y), 1)

        for step in range(steps):
            t = step / steps
            x = int(round(end_x * t))
            y = int(round(start_y + (end_y - start_y) * t))
            if 0 <= y < data.shape[0] and 0 <= x < data.shape[1]:
                if data[y, x] != 100:
                    data[y, x] = 0

    def _make_grid_message(self, data: np.ndarray) -> OccupancyGrid:
        grid = OccupancyGrid()
        grid.info.resolution = self._resolution
        grid.info.width = data.shape[1]
        grid.info.height = data.shape[0]
        grid.info.origin.position.x = 0.0
        grid.info.origin.position.y = -self._lateral_range / 2.0
        grid.info.origin.position.z = 0.0
        grid.info.origin.orientation.w = 1.0
        grid.data = data.flatten(order="C").astype(int).tolist()
        return grid


def main(args: list[str] | None = None) -> None:
    """Run the occupancy grid generator node."""
    rclpy.init(args=args)
    node = OccupancyGridGenerator()
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
