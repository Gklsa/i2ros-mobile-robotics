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

"""YOLO-based NPC car detector with depth distance estimation."""

from __future__ import annotations

import math
import os
from pathlib import Path

from ad_interfaces.msg import (
    CarDetection,
    CarDetections,
    TrafficLightDetections,
    TrafficLightState,
)
import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Float32, Int32, String
import tf2_ros


def _parse_ranges(raw: str) -> list[tuple[int, int]]:
    ranges = []
    for chunk in (raw or "").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            lo_s, hi_s = chunk.split("-", 1)
            lo, hi = int(lo_s), int(hi_s)
        else:
            lo = hi = int(chunk)
        ranges.append((lo, hi))
    return ranges


class YoloCarDetector(Node):
    """Detect cars in RGB images and publish the nearest front-car distance."""

    def __init__(self) -> None:
        super().__init__("yolo_car_detector")

        self.declare_parameter(
            "rgb_image_topic",
            "/OurCar/Sensors/RGBCameraLeft/image_raw",
        )
        self.declare_parameter(
            "depth_image_topic",
            "/OurCar/Sensors/DepthCamera/image_raw",
        )
        self.declare_parameter(
            "depth_camera_info_topic",
            "/OurCar/Sensors/DepthCamera/camera_info",
        )
        self.declare_parameter(
            "front_car_distance_topic",
            "/perception/front_car_distance",
        )
        self.declare_parameter(
            "car_detections_topic",
            "/perception/car_detections",
        )
        # Same convention as traffic_light_detector.py: fixed TF frame the
        # per-detection world_x/y/z position is expressed in.
        self.declare_parameter("world_frame_id", "world")
        self.declare_parameter(
            "car_debug_image_topic",
            "/perception/debug/car_detection_image",
        )
        self.declare_parameter(
            "combined_debug_image_topic",
            "/perception/debug/detection_combined",
        )
        self.declare_parameter(
            "traffic_light_detections_topic",
            "/perception/traffic_light_detections",
        )
        self.declare_parameter("waypoint_index_topic", "/planning/waypoint_index")
        self.declare_parameter("car_model_path", "")
        self.declare_parameter("car_confidence_threshold", 0.4)
        self.declare_parameter("car_min_depth_m", 0.5)
        self.declare_parameter("car_max_depth_m", 25.0)
        self.declare_parameter("car_max_lateral_angle_ratio", 0.45)
        self.declare_parameter("car_depth_percentile", 20.0)
        self.declare_parameter("car_publish_debug_image", True)
        self.declare_parameter("publish_combined_debug_image", True)
        self.declare_parameter("combined_debug_traffic_light_max_age_sec", 5.0)
        self.declare_parameter("car_label_topic", "/perception/car_label")
        # Waypoint zones where the left side of the frame is a known source
        # of false car candidates (e.g. parked/decorative cars), so only the
        # right half of the image is considered there.
        self.declare_parameter("car_right_only_waypoint_ranges", "60-68")
        self.declare_parameter("car_right_only_min_x_ratio", 0.5)
        # Waypoint zones with no relevant NPC/front car (outside the car-
        # following and braking scenes) - skip YOLO inference entirely there
        # instead of just filtering results, since it's pure false-positive
        # risk in those zones and also saves CPU (2 CPU YOLO detectors + Unity
        # + RViz already saturate a 6-core machine).
        self.declare_parameter(
            "car_detection_disabled_waypoint_ranges", "0-39,68-101,121-130"
        )

        self._bridge = CvBridge()
        self._latest_depth_m: np.ndarray | None = None
        self._latest_depth_frame_id: str = ""
        self._depth_camera_info: CameraInfo | None = None
        self._latest_traffic_lights: TrafficLightDetections | None = None
        self._latest_waypoint_index: int | None = None
        self._model = None
        self._model_error_logged = False
        self._world_frame_id = self.get_parameter("world_frame_id").value
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        self._conf = float(self.get_parameter("car_confidence_threshold").value)
        self._min_depth_m = float(self.get_parameter("car_min_depth_m").value)
        self._max_depth_m = float(self.get_parameter("car_max_depth_m").value)
        self._max_lateral_angle_ratio = float(
            self.get_parameter("car_max_lateral_angle_ratio").value
        )
        self._depth_percentile = float(
            self.get_parameter("car_depth_percentile").value
        )
        self._publish_debug_image = bool(
            self.get_parameter("car_publish_debug_image").value
        )
        self._publish_combined_debug_image = bool(
            self.get_parameter("publish_combined_debug_image").value
        )
        self._combined_debug_traffic_light_max_age_sec = float(
            self.get_parameter("combined_debug_traffic_light_max_age_sec").value
        )
        self._right_only_ranges = _parse_ranges(
            self.get_parameter("car_right_only_waypoint_ranges").value
        )
        self._right_only_min_x_ratio = float(
            self.get_parameter("car_right_only_min_x_ratio").value
        )
        self._detection_disabled_ranges = _parse_ranges(
            self.get_parameter("car_detection_disabled_waypoint_ranges").value
        )

        self._distance_pub = self.create_publisher(
            Float32,
            self.get_parameter("front_car_distance_topic").value,
            10,
        )
        self._label_pub = self.create_publisher(
            String,
            self.get_parameter("car_label_topic").value,
            10,
        )
        self._detections_pub = self.create_publisher(
            CarDetections,
            self.get_parameter("car_detections_topic").value,
            10,
        )
        # BEST_EFFORT + shallow history (not the default RELIABLE+depth10):
        # these are real-time visualization topics, so a subscriber (e.g.
        # rqt_image_view) that can't keep up with the publish rate should
        # drop old frames and see the latest one, not fall further and
        # further behind working through a growing backlog.
        self._debug_pub = self.create_publisher(
            Image,
            self.get_parameter("car_debug_image_topic").value,
            qos_profile_sensor_data,
        )
        self._combined_debug_pub = self.create_publisher(
            Image,
            self.get_parameter("combined_debug_image_topic").value,
            qos_profile_sensor_data,
        )

        self.create_subscription(
            Image,
            self.get_parameter("depth_image_topic").value,
            self._depth_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            CameraInfo,
            self.get_parameter("depth_camera_info_topic").value,
            self._depth_info_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Image,
            self.get_parameter("rgb_image_topic").value,
            self._image_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            TrafficLightDetections,
            self.get_parameter("traffic_light_detections_topic").value,
            self._traffic_light_callback,
            10,
        )
        self.create_subscription(
            Int32,
            self.get_parameter("waypoint_index_topic").value,
            self._waypoint_index_callback,
            10,
        )

        self._load_model()
        self.get_logger().info(
            "YOLO car detector publishing nearest car distance on "
            f"{self.get_parameter('front_car_distance_topic').value}"
        )

    def _load_model(self) -> None:
        model_path = str(self.get_parameter("car_model_path").value).strip()
        if not model_path:
            self.get_logger().warn(
                "car_model_path is empty; YOLO car detector will publish inf."
            )
            return

        resolved = Path(os.path.expanduser(model_path)).resolve()
        if not resolved.exists():
            self.get_logger().warn(
                f"car_model_path does not exist: {resolved}; publishing inf."
            )
            return

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            self.get_logger().warn(
                "ultralytics is not installed in this ROS Python environment; "
                f"car detector disabled: {exc}"
            )
            return

        self._model = YOLO(str(resolved))
        self.get_logger().info(f"Loaded YOLO car model: {resolved}")

    def _depth_callback(self, msg: Image) -> None:
        try:
            depth = self._bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except CvBridgeError as exc:
            self.get_logger().warn(f"Failed to convert depth image: {exc}")
            return

        depth_m = np.asarray(depth, dtype=np.float32)
        if msg.encoding.upper() == "16UC1":
            depth_m = depth_m / 1000.0
        self._latest_depth_m = depth_m
        self._latest_depth_frame_id = msg.header.frame_id

    def _depth_info_callback(self, msg: CameraInfo) -> None:
        self._depth_camera_info = msg

    def _traffic_light_callback(self, msg: TrafficLightDetections) -> None:
        self._latest_traffic_lights = msg

    def _waypoint_index_callback(self, msg: Int32) -> None:
        self._latest_waypoint_index = int(msg.data)

    def _image_callback(self, msg: Image) -> None:
        try:
            image = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as exc:
            self.get_logger().warn(f"Failed to convert RGB image: {exc}")
            self._publish_distance(math.inf)
            return

        if self._model is None:
            self._publish_distance(math.inf)
            return

        if self._car_detection_disabled():
            self._publish_distance(math.inf)
            self._publish_car_label([], math.inf)
            self._publish_car_detections(msg.header, [])
            if self._publish_debug_image:
                self._publish_debug(msg.header, image, [])
            if self._publish_combined_debug_image:
                self._publish_combined_debug(msg.header, image, [])
            return

        try:
            results = self._model.predict(image, conf=self._conf, verbose=False)
        except Exception as exc:  # pragma: no cover - depends on runtime model
            if not self._model_error_logged:
                self.get_logger().warn(f"YOLO inference failed: {exc}")
                self._model_error_logged = True
            self._publish_distance(math.inf)
            return

        candidates = self._extract_candidates(results, image.shape)
        image_height, image_width = image.shape[:2]
        self._augment_world_positions(candidates, image_width, image_height)
        nearest = min((cand["distance"] for cand in candidates), default=math.inf)
        self._publish_distance(nearest)
        self._publish_car_label(candidates, nearest)
        self._publish_car_detections(msg.header, candidates)

        if self._publish_debug_image:
            self._publish_debug(msg.header, image, candidates)
        if self._publish_combined_debug_image:
            self._publish_combined_debug(msg.header, image, candidates)

    def _extract_candidates(self, results, image_shape) -> list[dict]:
        if self._latest_depth_m is None or not results:
            return []

        image_height, image_width = image_shape[:2]
        candidates: list[dict] = []
        boxes = getattr(results[0], "boxes", None)
        if boxes is None:
            return candidates

        for box in boxes:
            xyxy = box.xyxy[0].detach().cpu().numpy()
            confidence = float(box.conf[0].detach().cpu().item())
            x0, y0, x1, y1 = [int(round(value)) for value in xyxy]
            distance = self._estimate_bbox_distance(
                (x0, y0, x1, y1),
                image_width,
                image_height,
            )
            reject_reason = self._distance_reject_reason(
                distance,
                (x0, x1),
                image_width,
            )
            if reject_reason is not None:
                candidates.append({
                    "bbox": (x0, y0, x1, y1),
                    "confidence": confidence,
                    "distance": math.inf,
                    "raw_distance": distance,
                    "accepted": False,
                    "reject_reason": reject_reason,
                })
                continue

            candidates.append({
                "bbox": (x0, y0, x1, y1),
                "confidence": confidence,
                "distance": distance,
                "raw_distance": distance,
                "accepted": True,
            })
        return candidates

    def _estimate_bbox_distance(
        self,
        bbox: tuple[int, int, int, int],
        image_width: int,
        image_height: int,
    ) -> float:
        if self._latest_depth_m is None:
            return math.inf

        x0, y0, x1, y1 = bbox
        depth_height, depth_width = self._latest_depth_m.shape[:2]
        dx0 = int(np.clip(x0 / image_width * depth_width, 0, depth_width - 1))
        dx1 = int(np.clip(x1 / image_width * depth_width, 0, depth_width))
        dy0 = int(np.clip(y0 / image_height * depth_height, 0, depth_height - 1))
        dy1 = int(np.clip(y1 / image_height * depth_height, 0, depth_height))
        if dx1 <= dx0 or dy1 <= dy0:
            return math.inf

        patch = self._latest_depth_m[dy0:dy1, dx0:dx1]
        valid = patch[np.isfinite(patch)]
        valid = valid[(valid >= self._min_depth_m) & (valid <= self._max_depth_m)]
        if valid.size < 8:
            return math.inf
        return float(np.percentile(valid, self._depth_percentile))

    def _sample_depth_point(
        self,
        bbox: tuple[int, int, int, int],
        image_width: int,
        image_height: int,
    ) -> tuple[float, float, float] | None:
        """
        Back-project a car bbox center into the depth camera frame.

        Same approach as traffic_light_detector.py's _sample_depth_point:
        approximate, since it samples the depth camera at the RGB-left
        detection's normalized pixel location despite the ~0.2m baseline
        between the two cameras.
        """
        if self._latest_depth_m is None or self._depth_camera_info is None:
            return None
        if image_width <= 0 or image_height <= 0:
            return None

        x0, y0, x1, y1 = bbox
        center_x = 0.5 * (x0 + x1)
        center_y = 0.5 * (y0 + y1)

        depth_height, depth_width = self._latest_depth_m.shape[:2]
        du = int(np.clip(center_x / image_width * depth_width, 0, depth_width - 1))
        dv = int(np.clip(center_y / image_height * depth_height, 0, depth_height - 1))

        depth = float(self._latest_depth_m[dv, du])
        if not math.isfinite(depth) or depth < self._min_depth_m or depth > self._max_depth_m:
            return None

        k = self._depth_camera_info.k
        fx, fy, ppx, ppy = k[0], k[4], k[2], k[5]
        if fx <= 0.0 or fy <= 0.0:
            return None

        x_cam = (du - ppx) * depth / fx
        y_cam = (dv - ppy) * depth / fy
        z_cam = depth
        return (x_cam, y_cam, z_cam)

    @staticmethod
    def _transform_point(
        transform,
        point: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        t = transform.transform.translation
        q = transform.transform.rotation
        rx, ry, rz = YoloCarDetector._rotate_vector(
            point, (q.x, q.y, q.z, q.w)
        )
        return (rx + t.x, ry + t.y, rz + t.z)

    @staticmethod
    def _rotate_vector(
        v: tuple[float, float, float],
        q: tuple[float, float, float, float],
    ) -> tuple[float, float, float]:
        vx, vy, vz = v
        qx, qy, qz, qw = q
        tx = 2.0 * (qy * vz - qz * vy)
        ty = 2.0 * (qz * vx - qx * vz)
        tz = 2.0 * (qx * vy - qy * vx)
        rx = vx + qw * tx + (qy * tz - qz * ty)
        ry = vy + qw * ty + (qz * tx - qx * tz)
        rz = vz + qw * tz + (qx * ty - qy * tx)
        return (rx, ry, rz)

    def _augment_world_positions(
        self, candidates: list[dict], image_width: int, image_height: int
    ) -> None:
        if not candidates:
            return

        transform = None
        if self._latest_depth_m is not None and self._depth_camera_info is not None:
            try:
                transform = self._tf_buffer.lookup_transform(
                    self._world_frame_id,
                    self._latest_depth_frame_id,
                    Time(),
                    timeout=Duration(seconds=0.05),
                )
            except (
                tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException,
            ):
                transform = None

        for candidate in candidates:
            cam_point = self._sample_depth_point(
                candidate["bbox"], image_width, image_height
            )
            candidate["world_xyz"] = (
                self._transform_point(transform, cam_point)
                if transform is not None and cam_point is not None
                else None
            )

    def _distance_reject_reason(
        self,
        distance: float,
        bbox_x: tuple[int, int],
        image_width: int,
    ) -> str | None:
        if not math.isfinite(distance):
            return "no_depth"
        if distance < self._min_depth_m:
            return "too_close"
        if distance > self._max_depth_m:
            return "too_far"

        x0, x1 = bbox_x
        center_x = 0.5 * (x0 + x1)
        normalized_offset = abs(center_x - 0.5 * image_width) / max(1.0, image_width)
        if normalized_offset > self._max_lateral_angle_ratio:
            return "off_axis"

        if self._in_right_only_zone() and center_x < self._right_only_min_x_ratio * image_width:
            return "left_side_excluded_zone"
        return None

    def _in_right_only_zone(self) -> bool:
        if self._latest_waypoint_index is None:
            return False
        index = self._latest_waypoint_index
        return any(lo <= index <= hi for lo, hi in self._right_only_ranges)

    def _car_detection_disabled(self) -> bool:
        if self._latest_waypoint_index is None:
            return False
        index = self._latest_waypoint_index
        return any(lo <= index <= hi for lo, hi in self._detection_disabled_ranges)

    def _publish_debug(self, header, image: np.ndarray, candidates: list[dict]) -> None:
        debug = image.copy()
        for candidate in candidates:
            x0, y0, x1, y1 = candidate["bbox"]
            accepted = candidate.get("accepted", False)
            color = (0, 255, 0) if accepted else (120, 120, 120)
            label_distance = candidate.get("distance", math.inf)
            if not math.isfinite(label_distance):
                label_distance = candidate.get("raw_distance", math.inf)
            distance_text = (
                f"{label_distance:.1f}m" if math.isfinite(label_distance) else "inf"
            )
            label = f"car {candidate['confidence']:.2f} {distance_text}"
            if not accepted:
                label += f" reject:{candidate.get('reject_reason', '')}"

            cv2.rectangle(debug, (x0, y0), (x1, y1), color, 1)
            cv2.putText(
                debug,
                label,
                (x0, max(20, y0 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

        msg = self._bridge.cv2_to_imgmsg(debug, encoding="bgr8")
        msg.header = header
        self._debug_pub.publish(msg)

    def _publish_combined_debug(
        self,
        header,
        image: np.ndarray,
        car_candidates: list[dict],
    ) -> None:
        debug = image.copy()
        traffic_summary = self._draw_traffic_lights(debug, header)
        front_car = self._draw_cars(debug, car_candidates)
        self._draw_status_panel(debug, traffic_summary, front_car)
        msg = self._bridge.cv2_to_imgmsg(debug, encoding="bgr8")
        msg.header = header
        self._combined_debug_pub.publish(msg)

    def _draw_cars(self, image: np.ndarray, candidates: list[dict]) -> dict | None:
        accepted = [
            candidate
            for candidate in candidates
            if candidate.get("accepted", False)
            and math.isfinite(float(candidate.get("distance", math.inf)))
        ]
        front_car = min(
            accepted,
            key=lambda candidate: float(candidate.get("distance", math.inf)),
            default=None,
        )

        for candidate in candidates:
            x0, y0, x1, y1 = candidate["bbox"]
            accepted = candidate.get("accepted", False)
            selected = candidate is front_car
            color = (0, 255, 0) if selected else (120, 120, 120)
            thickness = 2 if selected else 1
            label_distance = candidate.get("distance", math.inf)
            if not math.isfinite(label_distance):
                label_distance = candidate.get("raw_distance", math.inf)
            distance_text = (
                f"{label_distance:.1f}m" if math.isfinite(label_distance) else "inf"
            )
            label = (
                f"FRONT car {distance_text}"
                if selected
                else f"car {candidate['confidence']:.2f} {distance_text}"
            )
            if not accepted:
                label += f" reject:{candidate.get('reject_reason', '')}"

            cv2.rectangle(image, (x0, y0), (x1, y1), color, thickness)
            self._put_text(
                image,
                label,
                (x0, max(20, y0 - 5)),
                color,
            )
        return front_car

    def _draw_traffic_lights(self, image: np.ndarray, current_header) -> dict:
        summary = {
            "state": TrafficLightState.UNKNOWN,
            "confidence": 0.0,
            "stale": False,
            "count": 0,
        }
        if self._latest_traffic_lights is None:
            return summary
        # The combined debug image is drawn on *this* frame's RGB, but the
        # traffic-light boxes come from whatever the last-received message on
        # /perception/traffic_light_detections was - a separate, slower node.
        # Do not continuously fade R/G toward black: at the live detector's
        # ~0.5 Hz rate that made one stable RED box look red, then brown, then
        # black even though its semantic state never changed. Preserve the
        # categorical color during its validity window; once expired, show the
        # box in the same blue used for any other unclassified/unsafe result.
        age_sec = self._stamp_diff_sec(
            current_header.stamp, self._latest_traffic_lights.header.stamp
        )
        stale = (
            age_sec
            > max(0.01, self._combined_debug_traffic_light_max_age_sec)
        )
        summary["stale"] = stale
        best_detection = None
        for detection in self._latest_traffic_lights.detections:
            x0 = int(detection.bbox_x)
            y0 = int(detection.bbox_y)
            x1 = x0 + int(detection.bbox_width)
            y1 = y0 + int(detection.bbox_height)
            box_only = str(detection.label) == "traffic_light"
            if not box_only:
                summary["count"] += 1
                if (
                    not stale
                    and int(detection.state) != TrafficLightState.UNKNOWN
                    and (
                        best_detection is None
                        or float(detection.confidence)
                        > float(best_detection.confidence)
                    )
                ):
                    best_detection = detection
            color = (
                (128, 128, 128)
                if box_only
                else self._traffic_light_color(int(detection.state))
            )
            if stale:
                # Purple is deliberately outside the R/G semantic palette and
                # distinct from gray box_only: this is an old overlay, not a
                # current frame whose color classification failed.
                color = (255, 0, 255)
            label = self._tl_label(
                float(detection.confidence),
                None if box_only or stale else int(detection.state),
            )
            if stale:
                label += " stale"
            cv2.rectangle(image, (x0, y0), (x1, y1), color, 1)
            self._put_text(
                image,
                label,
                (x0, max(20, y0 - 5)),
                color,
            )
            if bool(getattr(detection, "has_world_position", False)):
                xyz = (
                    f"X={float(detection.world_x):.1f} "
                    f"Y={float(detection.world_y):.1f} "
                    f"Z={float(detection.world_z):.1f}"
                )
                self._put_text(
                    image,
                    xyz,
                    (x0, min(image.shape[0] - 8, y1 + 16)),
                    color,
                    scale=0.45,
                )

        if best_detection is not None:
            summary["state"] = int(best_detection.state)
            summary["confidence"] = float(best_detection.confidence)
        return summary

    def _draw_status_panel(
        self,
        image: np.ndarray,
        traffic_summary: dict,
        front_car: dict | None,
    ) -> None:
        panel_w = 255
        panel_h = 104
        overlay = image.copy()
        cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, image, 0.45, 0.0, image)
        cv2.rectangle(
            image,
            (10, 10),
            (10 + panel_w, 10 + panel_h),
            (220, 220, 220),
            1,
        )

        waypoint = (
            "--"
            if self._latest_waypoint_index is None
            else str(self._latest_waypoint_index)
        )
        state = int(traffic_summary.get("state", TrafficLightState.UNKNOWN))
        state_name = self._STATE_SHORT_NAME.get(state, "UN")
        state_color = self._traffic_light_color(state)
        if state == TrafficLightState.UNKNOWN:
            state_color = (220, 220, 220)
        stale_text = " stale" if traffic_summary.get("stale", False) else ""
        traffic_text = (
            f"TL: {state_name} "
            f"{float(traffic_summary.get('confidence', 0.0)):.2f}{stale_text}"
        )

        if front_car is None:
            car_text = "front car: inf"
            car_color = (220, 220, 220)
        else:
            car_text = f"front car: {float(front_car.get('distance', math.inf)):.1f}m"
            car_color = (0, 255, 0)

        self._put_text(image, f"WP: {waypoint}", (22, 35), (0, 255, 255), scale=0.6)
        self._put_text(image, traffic_text, (22, 63), state_color, scale=0.55)
        self._put_text(image, car_text, (22, 91), car_color, scale=0.55)

    @staticmethod
    def _put_text(
        image: np.ndarray,
        text: str,
        origin: tuple[int, int],
        color: tuple[int, int, int],
        scale: float = 0.5,
        thickness: int = 1,
    ) -> None:
        x, y = origin
        cv2.putText(
            image,
            text,
            (x + 1, y + 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (0, 0, 0),
            thickness + 2,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            text,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )

    _STATE_SHORT_NAME = {
        TrafficLightState.RED: "R",
        TrafficLightState.YELLOW: "Y",
        TrafficLightState.GREEN: "G",
        TrafficLightState.UNKNOWN: "UN",
    }

    @classmethod
    def _tl_label(cls, confidence: float, state: int | None) -> str:
        """
        Match the traffic-light detector's debug-box label format.

        reads the same way on every debug topic: 'TL:0.87 R' for a
        classified color, 'TL:0.42' (no suffix) for a box-only detection.
        """
        label = f"TL:{confidence:.2f}"
        if state is not None:
            label += f" {cls._STATE_SHORT_NAME.get(state, 'UN')}"
        return label

    @staticmethod
    def _stamp_diff_sec(newer, older) -> float:
        newer_sec = float(newer.sec) + float(newer.nanosec) * 1e-9
        older_sec = float(older.sec) + float(older.nanosec) * 1e-9
        return newer_sec - older_sec

    @staticmethod
    def _traffic_light_color(state: int) -> tuple[int, int, int]:
        if state == TrafficLightState.RED:
            return (0, 0, 255)
        if state == TrafficLightState.YELLOW:
            return (0, 255, 255)
        if state == TrafficLightState.GREEN:
            return (0, 255, 0)
        return (0, 0, 0)

    def _publish_distance(self, distance: float) -> None:
        msg = Float32()
        msg.data = float(distance)
        self._distance_pub.publish(msg)

    def _publish_car_detections(self, header, candidates: list[dict]) -> None:
        msg = CarDetections()
        msg.header = header
        for candidate in candidates:
            detection = CarDetection()
            detection.header = header
            detection.confidence = float(candidate["confidence"])
            label_distance = candidate.get("distance", math.inf)
            if not math.isfinite(label_distance):
                label_distance = candidate.get("raw_distance", math.inf)
            detection.distance = float(label_distance) if math.isfinite(label_distance) else -1.0

            x0, y0, x1, y1 = candidate["bbox"]
            detection.bbox_x = float(x0)
            detection.bbox_y = float(y0)
            detection.bbox_width = float(x1 - x0)
            detection.bbox_height = float(y1 - y0)

            detection.accepted = bool(candidate.get("accepted", False))
            detection.reject_reason = str(candidate.get("reject_reason", ""))

            world_xyz = candidate.get("world_xyz")
            if world_xyz is not None:
                detection.has_world_position = True
                detection.world_x = float(world_xyz[0])
                detection.world_y = float(world_xyz[1])
                detection.world_z = float(world_xyz[2])
            else:
                detection.has_world_position = False

            msg.detections.append(detection)

        self._detections_pub.publish(msg)

    def _publish_car_label(self, candidates: list[dict], nearest: float) -> None:
        total = len(candidates)
        accepted = sum(1 for cand in candidates if cand.get("accepted", False))
        nearest_text = f"{nearest:.1f}m" if math.isfinite(nearest) else "inf"
        msg = String()
        msg.data = (
            f"cars={total} accepted={accepted} nearest={nearest_text}"
            f" | right_only_zone={self._in_right_only_zone()}"
            f" | waypoint={self._latest_waypoint_index}"
        )
        self._label_pub.publish(msg)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = YoloCarDetector()
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
