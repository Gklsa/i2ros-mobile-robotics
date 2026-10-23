#!/usr/bin/env python3

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
Offline dataset extraction from a rosbag2 recording.

Why this exists: the live `trafficlight_dataset_collector` node requires
rgb/depth/semantic to arrive within `max_sync_diff_sec` (0.25s) of each
other in real time, and only saves a frame if all three land in that
window while `/planning/waypoint_index` is inside a configured capture
zone. On this machine, Unity's camera rendering can fall badly behind
after the sim has been running a while (observed directly: rgb/depth/
semantic arriving several *seconds* apart instead of ~0.4s), so nothing
gets saved for a zone even though the car genuinely drove through it -
confirmed separately via a live `/planning/waypoint_index` log showing
it reaching waypoint 150 while the dataset directory gained zero new
frames past waypoint 108.

A rosbag sidesteps this: `ros2 bag record` writes down every message as
it actually arrives, at whatever rate, with its real timestamp. This
script re-syncs everything afterward with a much looser tolerance than
the live node, so a drive that "should" have produced frames but didn't
(due to real-time render lag) can still be salvaged.

Record with (all topics the live collector also needs):

  ros2 bag record -o /tmp/tl_bag \
    /OurCar/Sensors/RGBCameraLeft/image_raw \
    /OurCar/Sensors/DepthCamera/image_raw \
    /OurCar/Sensors/SemanticCamera/image_raw \
    /planning/waypoint_index \
    /perception/traffic_light_detections

Then extract just the waypoint zones that failed live, with a looser
sync tolerance:

  python3 extract_dataset_from_bag.py --bag /tmp/tl_bag \
    --output-dir $(pwd)/src/ad_perception/datasets/traffic_light \
    --capture-waypoint-ranges 126-131,145-150 \
    --max-sync-diff-sec 2.0

Output layout and label format match trafficlight_dataset_collector.py
exactly (images/depth/semantic/labels under --output-dir), so the
existing relabel_from_semantic.py / split_yolo_dataset.py /
preview_yolo_labels.py tooling works unchanged on the result.
"""

from __future__ import annotations

import argparse
import bisect
from dataclasses import dataclass
import os

import cv2
from cv_bridge import CvBridge
import numpy as np
from rclpy.serialization import deserialize_message
import rosbag2_py
from rosidl_runtime_py.utilities import get_message

RGB_TOPIC = "/OurCar/Sensors/RGBCameraLeft/image_raw"
DEPTH_TOPIC = "/OurCar/Sensors/DepthCamera/image_raw"
SEMANTIC_TOPIC = "/OurCar/Sensors/SemanticCamera/image_raw"
WAYPOINT_TOPIC = "/planning/waypoint_index"
DETECTIONS_TOPIC = "/perception/traffic_light_detections"

_STATE_TO_CLASS_ID = {1: 0, 2: 1, 3: 2}  # RED, YELLOW, GREEN (ad_interfaces/TrafficLightState)


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


def _header_ns(msg) -> int:
    return int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)


@dataclass
class _Reader:
    bag_path: str

    def messages(self):
        reader = rosbag2_py.SequentialReader()
        storage_options = rosbag2_py.StorageOptions(uri=self.bag_path, storage_id="")
        converter_options = rosbag2_py.ConverterOptions("", "")
        reader.open(storage_options, converter_options)
        topic_types = {t.name: t.type for t in reader.get_all_topics_and_types()}
        while reader.has_next():
            topic, data, recv_t = reader.read_next()
            if topic not in topic_types:
                continue
            msg_type = get_message(topic_types[topic])
            msg = deserialize_message(data, msg_type)
            yield topic, recv_t, msg


def _load_bag(bag_path: str):
    rgb, depth, semantic, waypoints, detections = [], [], [], [], []
    for topic, recv_t, msg in _Reader(bag_path).messages():
        if topic == RGB_TOPIC:
            rgb.append((recv_t, msg))
        elif topic == DEPTH_TOPIC:
            depth.append((recv_t, msg))
        elif topic == SEMANTIC_TOPIC:
            semantic.append((recv_t, msg))
        elif topic == WAYPOINT_TOPIC:
            waypoints.append((recv_t, msg))
        elif topic == DETECTIONS_TOPIC:
            detections.append((recv_t, msg))
    rgb.sort(key=lambda item: _header_ns(item[1]))
    depth.sort(key=lambda item: _header_ns(item[1]))
    semantic.sort(key=lambda item: _header_ns(item[1]))
    waypoints.sort(key=lambda item: item[0])
    detections.sort(key=lambda item: _header_ns(item[1]))
    return rgb, depth, semantic, waypoints, detections


def _nearest_by_header(sorted_msgs: list[tuple[int, object]], target_ns: int, max_diff_ns: int):
    if not sorted_msgs:
        return None
    keys = [_header_ns(m) for _, m in sorted_msgs]
    idx = bisect.bisect_left(keys, target_ns)
    candidates = []
    if idx < len(sorted_msgs):
        candidates.append(sorted_msgs[idx])
    if idx > 0:
        candidates.append(sorted_msgs[idx - 1])
    if not candidates:
        return None
    best = min(candidates, key=lambda item: abs(_header_ns(item[1]) - target_ns))
    if abs(_header_ns(best[1]) - target_ns) > max_diff_ns:
        return None
    return best[1]


def _nearest_waypoint(waypoints: list[tuple[int, object]], target_recv_t: int) -> int | None:
    if not waypoints:
        return None
    keys = [t for t, _ in waypoints]
    idx = bisect.bisect_right(keys, target_recv_t) - 1
    if idx < 0:
        idx = 0
    return int(waypoints[idx][1].data)


def _in_ranges(idx: int, ranges: list[tuple[int, int]], margin: int) -> bool:
    return any(lo - margin <= idx <= hi + margin for lo, hi in ranges)


def _depth_to_png16(depth: np.ndarray) -> np.ndarray:
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    if depth.dtype != np.uint16:
        depth = np.clip(depth * 1000.0, 0, 65535).astype(np.uint16)
    return depth


def _build_label_boxes(
    detections,
    width: int,
    height: int,
    min_confidence: float,
    require_housing: bool,
    iou_suppress: float,
) -> list[tuple[int, float, float, float, float]]:
    if detections is None:
        return []
    candidates = []
    for det in detections.detections:
        class_id = _STATE_TO_CLASS_ID.get(det.state)
        if class_id is None:
            continue
        if det.confidence < min_confidence:
            continue
        if require_housing and not det.has_housing:
            continue
        if det.bbox_width <= 0.0 or det.bbox_height <= 0.0:
            continue
        candidates.append((det.confidence, class_id, det))

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
        if any(_bbox_iou(full_box, kept) > iou_suppress for kept in kept_boxes):
            continue
        kept_boxes.append(full_box)
        cx = (det.bbox_x + det.bbox_width / 2.0) / width
        cy = (det.bbox_y + det.bbox_height / 2.0) / height
        w = det.bbox_width / width
        h = det.bbox_height / height
        cx, cy, w, h = (float(np.clip(v, 0.0, 1.0)) for v in (cx, cy, w, h))
        boxes.append((class_id, cx, cy, w, h))
    return boxes


def extract(args: argparse.Namespace) -> None:
    # No ranges given means the recording's /planning/waypoint_index is not
    # trustworthy (e.g. two overlapping launches wrote interleaved values, or
    # the spawn point didn't match the route so the planner never advanced) -
    # skip the waypoint gate entirely and tag filenames "wpXXX" instead of a
    # specific (wrong) index, rather than silently mislabeling every frame.
    ranges = _parse_ranges(args.capture_waypoint_ranges) if args.capture_waypoint_ranges else None
    max_sync_diff_ns = int(args.max_sync_diff_sec * 1e9)
    max_detection_sync_diff_ns = int(args.max_detection_sync_diff_sec * 1e9)

    print(f"loading bag {args.bag} ...")
    rgb, depth, semantic, waypoints, detections = _load_bag(args.bag)
    print(
        f"loaded: rgb={len(rgb)} depth={len(depth)} semantic={len(semantic)} "
        f"waypoint_index={len(waypoints)} detections={len(detections)}"
    )

    images_dir = os.path.join(args.output_dir, "images")
    depth_dir = os.path.join(args.output_dir, "depth")
    semantic_dir = os.path.join(args.output_dir, "semantic")
    labels_dir = os.path.join(args.output_dir, "labels")
    for path in (images_dir, depth_dir, semantic_dir, labels_dir):
        os.makedirs(path, exist_ok=True)

    bridge = CvBridge()
    saved = 0
    skipped_out_of_zone = 0
    skipped_no_sync = 0
    labeled = 0

    for recv_t, rgb_msg in rgb:
        rgb_ns = _header_ns(rgb_msg)
        if ranges is None:
            wp_tag = "XXX"
        else:
            wp = _nearest_waypoint(waypoints, recv_t)
            if wp is None or not _in_ranges(wp, ranges, args.capture_margin_waypoints):
                skipped_out_of_zone += 1
                continue
            wp_tag = f"{wp:03d}"

        depth_msg = _nearest_by_header(depth, rgb_ns, max_sync_diff_ns)
        semantic_msg = _nearest_by_header(semantic, rgb_ns, max_sync_diff_ns)
        if depth_msg is None or semantic_msg is None:
            skipped_no_sync += 1
            continue

        try:
            rgb_cv = bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
            depth_cv = np.asarray(bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough"))
            semantic_cv = bridge.imgmsg_to_cv2(semantic_msg, desired_encoding="bgr8")
        except Exception as exc:  # noqa: BLE001 - just skip unreadable frames
            print(f"  warn: failed to convert frame at wp{wp_tag}: {exc}")
            continue

        stamp = rgb_ns / 1e9
        basename = f"wp{wp_tag}_{saved:05d}_{stamp:.3f}"
        cv2.imwrite(
            os.path.join(images_dir, f"{basename}.jpg"),
            rgb_cv,
            [cv2.IMWRITE_JPEG_QUALITY, args.jpg_quality],
        )
        cv2.imwrite(os.path.join(semantic_dir, f"{basename}.png"), semantic_cv)
        cv2.imwrite(os.path.join(depth_dir, f"{basename}.png"), _depth_to_png16(depth_cv))

        boxes = []
        if args.auto_label_traffic_lights:
            det_msg = _nearest_by_header(detections, rgb_ns, max_detection_sync_diff_ns)
            height, width = rgb_cv.shape[:2]
            boxes = _build_label_boxes(
                det_msg, width, height, args.min_detection_confidence,
                args.require_housing_for_label, args.label_iou_suppress,
            )
        with open(os.path.join(labels_dir, f"{basename}.txt"), "w") as fh:
            for class_id, cx, cy, w, h in boxes:
                fh.write(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
        if boxes:
            labeled += 1

        saved += 1
        if saved % 20 == 0:
            print(f"  ... {saved} frames saved so far (latest wp{wp_tag})")

    print(f"done: saved {saved} frames ({labeled} with >=1 auto label)")
    print(f"skipped (waypoint outside ranges): {skipped_out_of_zone}")
    print(f"skipped (no depth/semantic within {args.max_sync_diff_sec}s): {skipped_no_sync}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--bag",
        required=True,
        help="Path to the rosbag2 directory (the -o argument used to record it)",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--capture-waypoint-ranges",
        default=None,
        help=(
            "e.g. 126-131,145-150. Omit if /planning/waypoint_index in this "
            "recording is not trustworthy (e.g. duplicate launches, or spawn "
            "point didn't match the route) - all frames are then saved "
            "untagged as wpXXX instead of gated/labeled by a bogus index."
        ),
    )
    parser.add_argument("--capture-margin-waypoints", type=int, default=0)
    parser.add_argument(
        "--max-sync-diff-sec",
        type=float,
        default=1.0,
        help=(
            "rgb<->depth<->semantic tolerance (live node uses 0.25s; "
            "loosen here since offline extraction is not real-time constrained)"
        ),
    )
    parser.add_argument("--max-detection-sync-diff-sec", type=float, default=1.5)
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--require-housing-for-label", action="store_true", default=True)
    parser.add_argument(
        "--no-require-housing-for-label",
        dest="require_housing_for_label",
        action="store_false",
    )
    parser.add_argument("--label-iou-suppress", type=float, default=0.4)
    parser.add_argument("--auto-label-traffic-lights", action="store_true", default=True)
    parser.add_argument(
        "--no-auto-label-traffic-lights",
        dest="auto_label_traffic_lights",
        action="store_false",
    )
    parser.add_argument("--jpg-quality", type=int, default=95)
    args = parser.parse_args()
    extract(args)
