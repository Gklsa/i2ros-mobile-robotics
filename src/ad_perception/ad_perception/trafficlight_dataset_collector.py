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
Traffic-light YOLO dataset collector.

Dev-time only tool, not part of the graded driving stack (see
`enable_dataset_collector` in autonomous_driving.launch.py, default off).

While the car is inside a configured waypoint range (traffic-light and
NPC-event zones selected for dataset collection), this node saves
synced rgb/depth/semantic frame triples plus a semi-automatic YOLO label
file per frame.

The labels are *not* derived from the semantic camera's class colors: Unity
ships as a pre-built binary with no source in this repo, so the
semantic-image color-to-class table isn't recoverable statically. Instead
this node reuses
TrafficLightDetector's own `/perception/traffic_light_detections` topic -
the same tuned RGB housing+bulb detector already running for driving -
as the label source, since it already outputs a full-image bbox and a
RED/YELLOW/GREEN state per candidate for exactly these waypoint zones.
The semantic frame is still saved alongside every sample (for reference and
for labelling the "car" class later); this node also logs the semantic
pixel colors under each saved traffic-light box so that table can be built
empirically if it's ever needed.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import os
import time

from ad_interfaces.msg import TrafficLightDetections, TrafficLightState
import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Int32

_STATE_TO_CLASS_ID = {
    TrafficLightState.RED: 0,
    TrafficLightState.YELLOW: 1,
    TrafficLightState.GREEN: 2,
}


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


def _bbox_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


@dataclass
class _CachedFrame:
    stamp_sec: float
    image: np.ndarray


class TrafficlightDatasetCollector(Node):
    """
    Save synchronized sensor frames and automatic YOLO labels.

    Collection runs near configured traffic-light waypoint zones to build a
    YOLO training set.
    """

    def __init__(self) -> None:
        super().__init__("trafficlight_dataset_collector")

        self.declare_parameter("rgb_image_topic", "/OurCar/Sensors/RGBCameraLeft/image_raw")
        self.declare_parameter("depth_image_topic", "/OurCar/Sensors/DepthCamera/image_raw")
        self.declare_parameter("semantic_image_topic", "/OurCar/Sensors/SemanticCamera/image_raw")
        self.declare_parameter("waypoint_index_topic", "/planning/waypoint_index")
        self.declare_parameter(
            "traffic_light_detections_topic", "/perception/traffic_light_detections"
        )
        self.declare_parameter("output_dir", "src/ad_perception/datasets/traffic_light")
        # Collection is wider than the runtime traffic-light whitelist:
        # it includes confirmed light zones, uncertain light candidates,
        # and NPC-car event zones so raw RGB/depth/semantic frames are
        # available for later review and relabeling.
        self.declare_parameter(
            "capture_waypoint_ranges",
            "1-5,33-39,39-63,70-77,87-101,101-105,126-131,145-150",
        )
        # Widens each range by this many waypoint indices on both ends
        # (collection-side only, does not touch the driving-critical
        # per-waypoint confidence overrides in params.yaml) so a fast
        # transit through a zone still gets a few frames instead of
        # racing through it between waypoint_index updates. Left at 0 by
        # default since capture_waypoint_ranges above is now already tuned
        # to the exact useful windows.
        self.declare_parameter("capture_margin_waypoints", 0)
        # RGB itself only arrives at ~2-2.5Hz with real jitter (observed
        # min 0.21s / max 0.86s gaps), so there is little point polling
        # much faster than that.
        self.declare_parameter("save_interval_sec", 0.0)
        # Tight tolerance for rgb<->depth<->semantic: these are raw camera
        # streams and need close pixel-level alignment.
        self.declare_parameter("max_sync_diff_sec", 0.25)
        # Looser tolerance for rgb<->detections specifically: detections
        # are *derived* from an rgb frame by a CPU-heavy pipeline
        # (housing/bulb search, TF lookups) and lag/jitter further behind
        # - observed gaps up to 1.1s between published detections. A
        # traffic light's state does not change within ~1s, so reusing a
        # slightly-stale detection is safe; the old tight 0.25s tolerance
        # here caused every label to be silently dropped as "unsynced".
        self.declare_parameter("max_detection_sync_diff_sec", 1.5)
        self.declare_parameter("min_detection_confidence", 0.5)
        self.declare_parameter("require_housing_for_label", True)
        self.declare_parameter("label_iou_suppress", 0.4)
        self.declare_parameter("auto_label_traffic_lights", True)
        self.declare_parameter("jpg_quality", 95)
        self.declare_parameter("log_semantic_colors", True)

        self._bridge = CvBridge()
        self._ranges = _parse_ranges(self.get_parameter("capture_waypoint_ranges").value)
        self._capture_margin = int(self.get_parameter("capture_margin_waypoints").value)
        self._save_interval = float(self.get_parameter("save_interval_sec").value)
        self._max_sync_diff = float(self.get_parameter("max_sync_diff_sec").value)
        self._max_detection_sync_diff = float(
            self.get_parameter("max_detection_sync_diff_sec").value
        )
        self._min_confidence = float(self.get_parameter("min_detection_confidence").value)
        self._require_housing = bool(self.get_parameter("require_housing_for_label").value)
        self._iou_suppress = float(self.get_parameter("label_iou_suppress").value)
        self._auto_label_traffic_lights = bool(
            self.get_parameter("auto_label_traffic_lights").value
        )
        self._jpg_quality = int(self.get_parameter("jpg_quality").value)
        self._log_semantic_colors = bool(self.get_parameter("log_semantic_colors").value)

        output_dir = str(self.get_parameter("output_dir").value)
        self._images_dir = os.path.join(output_dir, "images")
        self._labels_dir = os.path.join(output_dir, "labels")
        self._depth_dir = os.path.join(output_dir, "depth")
        self._semantic_dir = os.path.join(output_dir, "semantic")
        for path in (self._images_dir, self._labels_dir, self._depth_dir, self._semantic_dir):
            os.makedirs(path, exist_ok=True)

        self._latest_waypoint_index: int | None = None
        self._latest_depth: _CachedFrame | None = None
        self._latest_semantic: _CachedFrame | None = None
        self._latest_detections: TrafficLightDetections | None = None
        self._last_save_time = 0.0
        self._saved_count = 0
        self._labeled_count = 0

        self.create_subscription(
            Int32, self.get_parameter("waypoint_index_topic").value, self._waypoint_cb, 10
        )
        self.create_subscription(
            Image,
            self.get_parameter("depth_image_topic").value,
            self._depth_cb,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Image,
            self.get_parameter("semantic_image_topic").value,
            self._semantic_cb,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            TrafficLightDetections,
            self.get_parameter("traffic_light_detections_topic").value,
            self._detections_cb,
            10,
        )
        self.create_subscription(
            Image,
            self.get_parameter("rgb_image_topic").value,
            self._rgb_cb,
            qos_profile_sensor_data,
        )

        self.get_logger().info(
            f"Traffic-light dataset collector active for waypoint ranges {self._ranges}, "
            f"saving under {os.path.abspath(output_dir)}"
        )

    # ------------------------------------------------------------------
    # Subscriptions
    # ------------------------------------------------------------------

    def _waypoint_cb(self, msg: Int32) -> None:
        self._latest_waypoint_index = int(msg.data)

    def _depth_cb(self, msg: Image) -> None:
        try:
            depth = self._bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except CvBridgeError as exc:
            self.get_logger().warn(f"Failed to convert depth image: {exc}")
            return
        self._latest_depth = _CachedFrame(self._stamp_sec(msg.header), np.asarray(depth))

    def _semantic_cb(self, msg: Image) -> None:
        try:
            image = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as exc:
            self.get_logger().warn(f"Failed to convert semantic image: {exc}")
            return
        self._latest_semantic = _CachedFrame(self._stamp_sec(msg.header), image)

    def _detections_cb(self, msg: TrafficLightDetections) -> None:
        self._latest_detections = msg

    @staticmethod
    def _stamp_sec(header) -> float:
        return header.stamp.sec + header.stamp.nanosec * 1e-9

    def _in_capture_zone(self) -> bool:
        if self._latest_waypoint_index is None:
            return False
        idx = self._latest_waypoint_index
        margin = self._capture_margin
        return any(lo - margin <= idx <= hi + margin for lo, hi in self._ranges)

    def _rgb_cb(self, msg: Image) -> None:
        if not self._in_capture_zone():
            return

        now = time.monotonic()
        if now - self._last_save_time < self._save_interval:
            return

        try:
            rgb = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as exc:
            self.get_logger().warn(f"Failed to convert RGB image: {exc}")
            return

        rgb_stamp = self._stamp_sec(msg.header)
        depth = self._synced(self._latest_depth, rgb_stamp)
        semantic = self._synced(self._latest_semantic, rgb_stamp)
        # Wait for a properly synced set rather than saving an incomplete
        # one - the next rgb frame (at most save_interval_sec later) gets
        # another chance.
        if depth is None or semantic is None:
            return

        self._last_save_time = now
        self._save_frame(rgb, depth, semantic, rgb_stamp)

    def _synced(self, cached: _CachedFrame | None, ref_stamp: float) -> np.ndarray | None:
        if cached is None or abs(cached.stamp_sec - ref_stamp) > self._max_sync_diff:
            return None
        return cached.image

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------

    def _save_frame(
        self, rgb: np.ndarray, depth: np.ndarray, semantic: np.ndarray, stamp: float
    ) -> None:
        basename = f"wp{self._latest_waypoint_index:03d}_{self._saved_count:05d}_{stamp:.3f}"
        self._saved_count += 1

        cv2.imwrite(
            os.path.join(self._images_dir, f"{basename}.jpg"),
            rgb,
            [cv2.IMWRITE_JPEG_QUALITY, self._jpg_quality],
        )
        cv2.imwrite(os.path.join(self._semantic_dir, f"{basename}.png"), semantic)
        cv2.imwrite(os.path.join(self._depth_dir, f"{basename}.png"), self._depth_to_png16(depth))

        height, width = rgb.shape[:2]
        boxes = self._build_label_boxes(width, height, stamp)
        label_path = os.path.join(self._labels_dir, f"{basename}.txt")
        with open(label_path, "w") as fh:
            for class_id, cx, cy, w, h in boxes:
                fh.write(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

        if boxes:
            self._labeled_count += 1
            if self._log_semantic_colors:
                self._log_bbox_semantic_colors(semantic, boxes, width, height)

        self.get_logger().info(
            f"[{self._saved_count}] saved {basename} "
            f"({len(boxes)} auto-labeled box(es), {self._labeled_count} frames labeled so far)"
        )

    @staticmethod
    def _depth_to_png16(depth: np.ndarray) -> np.ndarray:
        """
        Encode depth as millimetres in a 16-bit single-channel PNG.

        ROS depth convention) regardless of whether the source was 16UC1
        (already mm) or 32FC1 (metres).
        """
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        if depth.dtype != np.uint16:
            depth = np.clip(depth * 1000.0, 0, 65535).astype(np.uint16)
        return depth

    def _build_label_boxes(
        self, width: int, height: int, rgb_stamp: float
    ) -> list[tuple[int, float, float, float, float]]:
        detections = self._latest_detections
        if not self._auto_label_traffic_lights:
            return []
        if detections is None or abs(self._stamp_sec(detections.header) - rgb_stamp) > (
            self._max_detection_sync_diff
        ):
            return []

        candidates = []
        for det in detections.detections:
            class_id = _STATE_TO_CLASS_ID.get(det.state)
            if class_id is None:
                continue
            if det.confidence < self._min_confidence:
                continue
            if self._require_housing and not det.has_housing:
                continue
            if det.bbox_width <= 0.0 or det.bbox_height <= 0.0:
                continue
            candidates.append((det.confidence, class_id, det))

        # Greedy NMS: the raw detections topic can carry more than one
        # overlapping candidate for the same physical bulb (e.g. a housing
        # candidate and a bare-blob fallback both firing) - keep only the
        # highest-confidence box per cluster so labels don't double up.
        candidates.sort(key=lambda item: item[0], reverse=True)
        kept_boxes: list[tuple[float, float, float, float]] = []
        boxes: list[tuple[int, float, float, float, float]] = []
        for _confidence, class_id, det in candidates:
            full_box = (
                det.bbox_x,
                det.bbox_y,
                det.bbox_x + det.bbox_width,
                det.bbox_y + det.bbox_height,
            )
            if any(_bbox_iou(full_box, kept) > self._iou_suppress for kept in kept_boxes):
                continue
            kept_boxes.append(full_box)

            cx = (det.bbox_x + det.bbox_width / 2.0) / width
            cy = (det.bbox_y + det.bbox_height / 2.0) / height
            w = det.bbox_width / width
            h = det.bbox_height / height
            cx, cy, w, h = (float(np.clip(v, 0.0, 1.0)) for v in (cx, cy, w, h))
            boxes.append((class_id, cx, cy, w, h))

        return boxes

    def _log_bbox_semantic_colors(
        self,
        semantic: np.ndarray,
        boxes: list[tuple[int, float, float, float, float]],
        width: int,
        height: int,
    ) -> None:
        """
        Log the dominant semantic-image colors under each labeled box.

        Purely informational: helps manually build a semantic
        color-to-class table later (see module docstring) by cross
        referencing these logs against a few saved semantic/*.png frames.
        """
        for class_id, cx, cy, w, h in boxes:
            x0 = max(0, int((cx - w / 2.0) * width))
            y0 = max(0, int((cy - h / 2.0) * height))
            x1 = min(width, int((cx + w / 2.0) * width))
            y1 = min(height, int((cy + h / 2.0) * height))
            if x1 <= x0 or y1 <= y0:
                continue
            patch = semantic[y0:y1, x0:x1].reshape(-1, 3)
            counts = Counter(map(tuple, patch))
            top = counts.most_common(3)
            self.get_logger().info(
                f"  class={class_id} bbox=({x0},{y0},{x1},{y1}) "
                f"dominant semantic BGR colors={top}"
            )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TrafficlightDatasetCollector()
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
