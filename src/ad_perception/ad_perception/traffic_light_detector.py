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
Traffic light detector node for the autonomous driving project.

Detection is structured in two stages instead of a single color-blob pass:

1. Housing/frame detection: find dark, elongated rectangular regions in the
   ROI (the traffic light's opaque body) that contain at least one bright
   "lens" spot. This is what separates a real traffic light from a flat
   colored sign or billboard, which has no dark housing behind it.
2. Bulb detection: inside each housing (plus a small padding margin), look
   for circular/elliptical colored blobs and classify each as RED/YELLOW/
   GREEN. Thresholds are relaxed here relative to the bare-blob fallback
   pass, since the housing already rules out most false positives.

A fallback bare-blob pass still runs over any ROI area not claimed by a
housing, so detection degrades gracefully (instead of disappearing) when a
housing can't be confidently segmented (e.g. a distant or partially
occluded light). Bare-blob candidates are penalized in confidence and held
to stricter shape/context checks, since they lack structural confirmation.
"""

from __future__ import annotations

from collections import Counter, deque
import json
import math
import os
from pathlib import Path

from ad_interfaces.msg import (
    TrafficLightDetection,
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
from std_msgs.msg import Int32, String
import tf2_ros


class TrafficLightDetector(Node):
    """Detect traffic light color from an RGB camera image."""

    # Parameters that a per-waypoint entry in traffic_light_waypoint_overrides
    # is allowed to override, mapped to (attribute name, value caster). Kept
    # to the thresholds that have actually needed per-intersection tuning
    # (housing color/shape gates, ROI crop, confidence) rather than every
    # declared parameter.
    _TUNABLE_PARAMS: dict[str, tuple[str, type]] = {
        "traffic_light_housing_min_aspect": ("_housing_min_aspect", float),
        "traffic_light_housing_max_aspect": ("_housing_max_aspect", float),
        "traffic_light_housing_dark_value": ("_housing_dark_value", float),
        "traffic_light_housing_dark_value_soft": ("_housing_dark_value_soft", float),
        "traffic_light_housing_dark_saturation_soft": (
            "_housing_dark_saturation_soft",
            float,
        ),
        "traffic_light_housing_bright_hue_min": ("_housing_bright_hue_min", int),
        "traffic_light_housing_bright_hue_max": ("_housing_bright_hue_max", int),
        "traffic_light_housing_bright_min_saturation": (
            "_housing_bright_min_saturation",
            float,
        ),
        "traffic_light_housing_bright_min_value": (
            "_housing_bright_min_value",
            float,
        ),
        "traffic_light_housing_max_context_dark_ratio": (
            "_housing_max_context_dark_ratio",
            float,
        ),
        "traffic_light_housing_max_context_bright_ratio": (
            "_housing_max_context_bright_ratio",
            float,
        ),
        "traffic_light_housing_min_fill": ("_housing_min_fill", float),
        "traffic_light_housing_max_fill": ("_housing_max_fill", float),
        "traffic_light_housing_min_width": ("_housing_min_width", int),
        "traffic_light_housing_min_height": ("_housing_min_height", int),
        "traffic_light_confidence_threshold": ("_confidence_threshold", float),
        "traffic_light_roi_top_ratio": ("_roi_top_ratio", float),
        "traffic_light_roi_bottom_ratio": ("_roi_bottom_ratio", float),
        "traffic_light_roi_left_ratio": ("_roi_left_ratio", float),
        "traffic_light_roi_right_ratio": ("_roi_right_ratio", float),
        "traffic_light_min_pixels": ("_min_pixels", int),
        "traffic_light_red_min_saturation": ("_red_min_saturation", float),
        "traffic_light_detect_yellow": ("_detect_yellow", bool),
        "traffic_light_yellow_min_saturation": ("_yellow_min_saturation", float),
        "traffic_light_green_min_saturation": ("_green_min_saturation", float),
        "traffic_light_require_depth_for_state": ("_require_depth_for_state", bool),
        "traffic_light_state_min_depth_m": ("_state_min_depth_m", float),
        "traffic_light_state_max_depth_m": ("_state_max_depth_m", float),
        "traffic_light_state_max_lateral_angle_ratio": (
            "_state_max_lateral_angle_ratio",
            float,
        ),
        "traffic_light_bulb_min_value": ("_bulb_min_value", float),
        "traffic_light_bulb_hue_min_saturation": (
            "_bulb_hue_min_saturation",
            float,
        ),
    }

    def __init__(self) -> None:
        super().__init__("traffic_light_detector")

        self.declare_parameter(
            "rgb_image_topic",
            "/OurCar/Sensors/RGBCameraLeft/image_raw",
        )
        self.declare_parameter(
            "traffic_light_state_topic",
            "/perception/traffic_light_state",
        )
        self.declare_parameter(
            "traffic_light_label_topic",
            "/perception/traffic_light_label",
        )
        self.declare_parameter(
            "traffic_light_detections_topic",
            "/perception/traffic_light_detections",
        )
        self.declare_parameter(
            "depth_image_topic",
            "/OurCar/Sensors/DepthCamera/image_raw",
        )
        self.declare_parameter(
            "depth_camera_info_topic",
            "/OurCar/Sensors/DepthCamera/camera_info",
        )
        self.declare_parameter("world_frame_id", "world")
        self.declare_parameter("waypoint_index_topic", "/planning/waypoint_index")
        # JSON string mapping a waypoint index (or "lo-hi" inclusive range) to
        # a dict of parameter overrides, e.g.
        # '{"43-46": {"traffic_light_housing_dark_value": 130.0}, "39": {...}}'.
        # Lets specific intersections get their own tuned thresholds without
        # changing the global defaults used everywhere else. Only the
        # parameter names in _TUNABLE_PARAMS below can be overridden this
        # way; anything not listed in an active override falls back to the
        # value loaded from params.yaml/CLI at startup.
        self.declare_parameter("traffic_light_waypoint_overrides", "{}")
        self.declare_parameter("traffic_light_min_depth_m", 0.5)
        self.declare_parameter("traffic_light_max_depth_m", 100.0)
        self.declare_parameter("traffic_light_require_depth_for_state", True)
        self.declare_parameter("traffic_light_state_min_depth_m", 1.5)
        self.declare_parameter("traffic_light_state_max_depth_m", 22.0)
        self.declare_parameter("traffic_light_state_max_lateral_angle_ratio", 0.32)
        self.declare_parameter("traffic_light_min_pixels", 25)
        self.declare_parameter("traffic_light_max_pixels", 900)
        self.declare_parameter("traffic_light_min_confirm_pixels", 18)
        self.declare_parameter("traffic_light_min_bbox_width", 3)
        self.declare_parameter("traffic_light_min_bbox_height", 5)
        self.declare_parameter("traffic_light_confidence_threshold", 0.6)
        self.declare_parameter("traffic_light_roi_top_ratio", 0.0)
        self.declare_parameter("traffic_light_roi_bottom_ratio", 0.65)
        self.declare_parameter("traffic_light_roi_left_ratio", 0.2)
        self.declare_parameter("traffic_light_roi_right_ratio", 0.8)
        self.declare_parameter("traffic_light_min_aspect_ratio", 0.45)
        self.declare_parameter("traffic_light_max_aspect_ratio", 2.2)
        self.declare_parameter("traffic_light_min_extent", 0.25)
        self.declare_parameter("traffic_light_max_extent", 0.95)
        self.declare_parameter("traffic_light_min_center_fill", 0.35)
        self.declare_parameter("traffic_light_min_circularity", 0.28)
        self.declare_parameter("traffic_light_min_ellipse_ratio", 0.55)
        self.declare_parameter("traffic_light_max_candidate_y_ratio", 0.72)
        self.declare_parameter("traffic_light_max_same_color_context_ratio", 0.25)
        self.declare_parameter("traffic_light_min_dark_context_ratio", 0.12)
        self.declare_parameter("traffic_light_min_housing_context_ratio", 0.10)
        self.declare_parameter("traffic_light_min_frame_context_ratio", 0.03)
        self.declare_parameter("traffic_light_min_mean_saturation", 130.0)
        self.declare_parameter("traffic_light_min_mean_value", 120.0)
        self.declare_parameter("traffic_light_morph_kernel_size", 1)
        # Caps per-frame cost on busy scenes (e.g. mid-corner with lots of
        # glass/reflections in view): each shape_seed component costs a
        # contour/circularity pass, so an unbounded component count makes
        # frame time scale with scene clutter instead of staying roughly
        # constant.
        self.declare_parameter("traffic_light_max_seed_components", 80)
        # Enabled by default: perception's job is to report RED/YELLOW/GREEN
        # for every light and let Decision pick the relevant one. With these
        # off, the classifier can only ever emit RED or UNKNOWN.
        self.declare_parameter("traffic_light_detect_green", True)
        self.declare_parameter("traffic_light_detect_yellow", True)
        # Minimum saturation for each bulb-color mask. Split out from the
        # rest of each range (hue/value bounds stay fixed) because a
        # captured frame showed real red-bulb pixels at S~146-148 sitting
        # right next to dozens of thin false "red" fragments at S~71-78 -
        # highlighted edges on the concrete overpass beams, which read as a
        # warm/reddish hue at low saturation under this renderer's hazy
        # lighting. Raising just the saturation floor cuts that noise out
        # at its source instead of relying entirely on downstream shape
        # filters to reject it.
        self.declare_parameter("traffic_light_red_min_saturation", 70.0)
        self.declare_parameter("traffic_light_yellow_min_saturation", 65.0)
        self.declare_parameter("traffic_light_green_min_saturation", 80.0)
        # HSV value (brightness) floors for each color mask. Previously these
        # were hardcoded literals in the inRange() calls while only the
        # saturation floor was tunable - inconsistent and a tuning trap.
        self.declare_parameter("traffic_light_red_min_value", 75.0)
        self.declare_parameter("traffic_light_red_high_min_value", 85.0)
        self.declare_parameter("traffic_light_yellow_min_value", 90.0)
        self.declare_parameter("traffic_light_green_min_value", 100.0)

        # --- Housing/frame detection (stage 1) ---
        self.declare_parameter("traffic_light_housing_dark_value", 100.0)
        self.declare_parameter("traffic_light_housing_dark_value_soft", 145.0)
        self.declare_parameter("traffic_light_housing_dark_saturation_soft", 95.0)
        self.declare_parameter("traffic_light_housing_close_kernel", 9)
        self.declare_parameter("traffic_light_housing_min_width", 4)
        self.declare_parameter("traffic_light_housing_min_height", 9)
        # Absolute pixel caps, not ROI ratios: a bridge pillar, overpass
        # underside, or building wall can easily be 35-65% of the ROI, but a
        # traffic light housing at driving distance never is.
        self.declare_parameter("traffic_light_housing_max_width_px", 70)
        self.declare_parameter("traffic_light_housing_max_height_px", 180)
        self.declare_parameter("traffic_light_housing_min_aspect", 1.05)
        self.declare_parameter("traffic_light_housing_max_aspect", 7.5)
        self.declare_parameter("traffic_light_housing_min_fill", 0.28)
        self.declare_parameter("traffic_light_housing_max_fill", 0.98)
        self.declare_parameter("traffic_light_housing_pad_ratio", 0.22)
        self.declare_parameter("traffic_light_housing_max_candidates", 6)
        # A real housing sits on a pole against open background; if most of
        # the area around it is *also* dark, it's more likely a fragment of
        # a much larger dark structure (bridge, tunnel, wall, pillar).
        self.declare_parameter("traffic_light_housing_max_context_dark_ratio", 0.55)
        # Precision-oriented gate for "does this dark box contain a real
        # bulb at all" (stricter than the stage-2 in-housing thresholds
        # below, which assume the housing is already trustworthy and try to
        # maximize recall of dim/far bulbs inside it).
        self.declare_parameter("traffic_light_housing_seed_min_pixels", 3)
        self.declare_parameter("traffic_light_housing_seed_min_circularity", 0.35)
        self.declare_parameter("traffic_light_housing_seed_min_ellipse_ratio", 0.55)

        # Some signal heads in this sim use a bright yellow/gold casing
        # instead of a black one (measured from a real captured frame: casing
        # H~16-27 S~50-150 V~155-170), so the dark_mask test above never
        # fires for them. This is a second, parallel candidate-region pass
        # using a bright/saturated color range instead of darkness; it feeds
        # into the same seed-confirmation and bulb-detection stages below so
        # it gets the same false-positive protection (must still contain a
        # real red/yellow/green bulb-shaped blob).
        self.declare_parameter("traffic_light_housing_bright_hue_min", 8)
        self.declare_parameter("traffic_light_housing_bright_hue_max", 40)
        self.declare_parameter("traffic_light_housing_bright_min_saturation", 60.0)
        self.declare_parameter("traffic_light_housing_bright_min_value", 140.0)
        self.declare_parameter("traffic_light_housing_max_context_bright_ratio", 0.55)

        # --- Bulb detection (stage 2) ---
        self.declare_parameter("traffic_light_housing_confidence_bonus", 0.30)
        self.declare_parameter("traffic_light_no_housing_confidence_penalty", 0.65)
        self.declare_parameter("traffic_light_housing_bulb_min_pixels", 2)
        self.declare_parameter("traffic_light_housing_bulb_min_circularity", 0.22)
        self.declare_parameter("traffic_light_housing_bulb_min_center_fill", 0.25)
        self.declare_parameter("traffic_light_housing_bulb_min_color_ratio", 0.08)
        self.declare_parameter("traffic_light_housing_bulb_min_ellipse_ratio", 0.45)
        self.declare_parameter("traffic_light_no_housing_min_color_ratio", 0.12)
        self.declare_parameter("traffic_light_require_housing_for_state", True)
        self.declare_parameter("traffic_light_history_size", 5)
        self.declare_parameter("traffic_light_history_confirm_count", 4)
        # Per-light temporal smoothing (medium change A). Instead of only
        # confirming one global scalar state across frames, associate each
        # published detection to a short-lived track by bbox IoU and majority-
        # vote its color over the track's recent frames. This stabilizes the
        # per-light output in /traffic_light_detections that Decision consumes,
        # so a single flickery frame no longer flips a light's reported color.
        self.declare_parameter("traffic_light_track_iou", 0.3)
        self.declare_parameter("traffic_light_track_history", 5)
        self.declare_parameter("traffic_light_track_min_count", 2)
        self.declare_parameter("traffic_light_track_max_misses", 3)
        self.declare_parameter("traffic_light_detection_iou_suppress", 0.35)
        self.declare_parameter("traffic_light_use_yolo", True)
        self.declare_parameter("traffic_light_yolo_model_path", "")
        self.declare_parameter("traffic_light_yolo_confidence_threshold", 0.35)
        self.declare_parameter("traffic_light_yolo_max_detections", 4)
        self.declare_parameter("traffic_light_yolo_min_color_pixels", 2)
        self.declare_parameter("traffic_light_yolo_min_color_ratio", 0.015)
        self.declare_parameter("traffic_light_yolo_min_value", 120.0)
        # 640, not the training script's default of 960: preview_yolo_predictions.py
        # (the tool actually used to visually validate this model; see its
        # own --imgsz default) runs at 640 - that is the
        # configuration this model's detection quality was verified against,
        # so live inference should match that over the training default.
        # Upscaled inference: distant lights are only a few pixels at native
        # 640x480, below YOLO's small-object confidence. Running at 960 gives
        # them more effective resolution and lifts recall on far lights; the
        # ~2Hz camera leaves compute headroom. Lower back to 640 if too slow.
        self.declare_parameter("traffic_light_yolo_imgsz", 640)
        # Small-box HSV fallback: below this bbox area (px^2) the vertical
        # thirds + center-crop logic degenerates (a 6px light gives ~1px per
        # third), so classify by dominant color over the whole box instead,
        # with a relaxed value floor since tiny lit bulbs blur darker.
        self.declare_parameter("traffic_light_yolo_small_bbox_area", 240)
        self.declare_parameter("traffic_light_yolo_small_min_value", 80.0)
        # Color classification mode inside a YOLO traffic-light box:
        #   "bulb"   - find the single brightest lit bulb, read its hue and its
        #              vertical position (top=red / mid=yellow / bottom=green).
        #              Robust to the sim's gold housing (which the rigid thirds
        #              method mis-reads as red) because it looks at the actual
        #              lit bulb, not a whole third. This is the default.
        #   "thirds" - legacy: split the box into 3 rows and score each color
        #              mask per row. Kept as a fallback.
        self.declare_parameter("traffic_light_yolo_classify_mode", "bulb")
        # Bulb mode tuning. Measured on sim data: a lit red bulb reads
        # RGB~(255,95,93) -> HSV H~0, S~160, V~255, while the gold housing reads
        # H~22, V~200. So the lit bulb is both the brightest region and clearly
        # separated in hue if the red hue band stays tight (<=13).
        self.declare_parameter("traffic_light_bulb_bright_percentile", 88.0)
        # Brightness floor for a lit bulb. Measured across 369 sim frames: red
        # bulbs have a median V of ~174 and 76% sit below 205, because most
        # lights are distant/dim - so a high floor drops the majority. For RED
        # and GREEN the tight hue band already excludes the gold housing, so the
        # floor can be low (130) to keep far lights. YELLOW overlaps the gold
        # housing's hue (~22), so it keeps a HIGH floor (215, above housing
        # V~200) to avoid reading the housing as yellow.
        self.declare_parameter("traffic_light_bulb_min_value", 130.0)
        self.declare_parameter("traffic_light_bulb_yellow_min_value", 215.0)
        self.declare_parameter("traffic_light_bulb_hue_min_saturation", 55.0)
        self.declare_parameter("traffic_light_bulb_red_hue_max", 13)
        self.declare_parameter("traffic_light_bulb_red_hue_min", 160)
        # Keep the simulator's orange/gold housing (H~16-22) out of yellow.
        # A genuinely lit yellow bulb has a tighter yellow core around H~30.
        self.declare_parameter("traffic_light_bulb_yellow_hue_min", 23)
        self.declare_parameter("traffic_light_bulb_yellow_hue_max", 38)
        self.declare_parameter("traffic_light_bulb_green_hue_min", 42)
        self.declare_parameter("traffic_light_bulb_green_hue_max", 96)
        # The housing's own gold/tan color sits in all three thirds roughly
        # evenly, so a lit bulb has to stand out from the *other two* thirds
        # by this much mean-V (0-255), not just clear an absolute threshold -
        # otherwise a uniformly bright/gold housing can score as "lit" in
        # whichever third happens to match a color mask.
        self.declare_parameter("traffic_light_yolo_min_brightness_margin", 10.0)
        # Fraction of the third's central 40%x40% region that must be
        # covered by the matched color mask - rejects thin scattered/edge
        # pixels (housing rim, bloom fringe) in favor of an actual solid lit
        # bulb, which sits centered in its third.
        self.declare_parameter("traffic_light_yolo_min_center_fill", 0.15)

        self.declare_parameter("traffic_light_debug_images", True)
        # Single unified debug view is the full-frame annotated image
        # (.../debug/detection): it already carries the ROI box, every light's
        # bbox/color/confidence/world coords, rejected boxes with reasons, and
        # the state summary - all in full-image coordinates. The ROI-crop and
        # HSV-panel topics used different sizes/coordinate frames (the source
        # of the "topics not consistent" problem in rqt) and are redundant, so
        # they default off. Turn on only when tuning masks locally.
        self.declare_parameter("traffic_light_debug_roi_images", False)
        self.declare_parameter("traffic_light_debug_hsv_images", False)
        self.declare_parameter("traffic_light_debug_stride", 2)
        # Default off: cv2.imshow + waitKey run on the image callback thread
        # and block it every frame, starving the ~2Hz pipeline and making
        # rqt/runtime fall behind. Enable only for local desktop debugging.
        self.declare_parameter("traffic_light_show_windows", False)
        self.declare_parameter("traffic_light_show_rejected_components", True)
        self.declare_parameter(
            "traffic_light_debug_roi_topic",
            "/perception/traffic_light/debug/roi",
        )
        self.declare_parameter(
            "traffic_light_debug_hsv_topic",
            "/perception/traffic_light/debug/hsv_panel",
        )
        self.declare_parameter(
            "traffic_light_debug_detection_topic",
            "/perception/traffic_light/debug/detection",
        )

        image_topic = self.get_parameter("rgb_image_topic").value
        state_topic = self.get_parameter("traffic_light_state_topic").value
        label_topic = self.get_parameter("traffic_light_label_topic").value
        detections_topic = self.get_parameter("traffic_light_detections_topic").value

        self._bridge = CvBridge()
        self._min_pixels = int(
            self.get_parameter("traffic_light_min_pixels").value
        )
        self._max_pixels = int(
            self.get_parameter("traffic_light_max_pixels").value
        )
        self._min_bbox_width = int(
            self.get_parameter("traffic_light_min_bbox_width").value
        )
        self._min_bbox_height = int(
            self.get_parameter("traffic_light_min_bbox_height").value
        )
        self._confidence_threshold = float(
            self.get_parameter("traffic_light_confidence_threshold").value
        )
        self._roi_top_ratio = float(
            self.get_parameter("traffic_light_roi_top_ratio").value
        )
        self._roi_bottom_ratio = float(
            self.get_parameter("traffic_light_roi_bottom_ratio").value
        )
        self._roi_left_ratio = float(
            self.get_parameter("traffic_light_roi_left_ratio").value
        )
        self._roi_right_ratio = float(
            self.get_parameter("traffic_light_roi_right_ratio").value
        )
        self._min_aspect_ratio = float(
            self.get_parameter("traffic_light_min_aspect_ratio").value
        )
        self._max_aspect_ratio = float(
            self.get_parameter("traffic_light_max_aspect_ratio").value
        )
        self._min_extent = float(
            self.get_parameter("traffic_light_min_extent").value
        )
        self._max_extent = float(
            self.get_parameter("traffic_light_max_extent").value
        )
        self._min_center_fill = float(
            self.get_parameter("traffic_light_min_center_fill").value
        )
        self._min_circularity = float(
            self.get_parameter("traffic_light_min_circularity").value
        )
        self._min_ellipse_ratio = float(
            self.get_parameter("traffic_light_min_ellipse_ratio").value
        )
        self._max_candidate_y_ratio = float(
            self.get_parameter("traffic_light_max_candidate_y_ratio").value
        )
        self._max_same_color_context_ratio = float(
            self.get_parameter("traffic_light_max_same_color_context_ratio").value
        )
        self._min_dark_context_ratio = float(
            self.get_parameter("traffic_light_min_dark_context_ratio").value
        )
        self._min_housing_context_ratio = float(
            self.get_parameter("traffic_light_min_housing_context_ratio").value
        )
        self._min_frame_context_ratio = float(
            self.get_parameter("traffic_light_min_frame_context_ratio").value
        )
        self._morph_kernel_size = max(
            1,
            int(self.get_parameter("traffic_light_morph_kernel_size").value),
        )
        self._max_seed_components = max(
            1,
            int(self.get_parameter("traffic_light_max_seed_components").value),
        )
        self._detect_green = bool(
            self.get_parameter("traffic_light_detect_green").value
        )
        self._detect_yellow = bool(
            self.get_parameter("traffic_light_detect_yellow").value
        )
        self._red_min_saturation = float(
            self.get_parameter("traffic_light_red_min_saturation").value
        )
        self._yellow_min_saturation = float(
            self.get_parameter("traffic_light_yellow_min_saturation").value
        )
        self._green_min_saturation = float(
            self.get_parameter("traffic_light_green_min_saturation").value
        )
        self._red_min_value = float(
            self.get_parameter("traffic_light_red_min_value").value
        )
        self._red_high_min_value = float(
            self.get_parameter("traffic_light_red_high_min_value").value
        )
        self._yellow_min_value = float(
            self.get_parameter("traffic_light_yellow_min_value").value
        )
        self._green_min_value = float(
            self.get_parameter("traffic_light_green_min_value").value
        )

        self._housing_dark_value = float(
            self.get_parameter("traffic_light_housing_dark_value").value
        )
        self._housing_dark_value_soft = float(
            self.get_parameter("traffic_light_housing_dark_value_soft").value
        )
        self._housing_dark_saturation_soft = float(
            self.get_parameter("traffic_light_housing_dark_saturation_soft").value
        )
        self._housing_close_kernel = max(
            3,
            int(self.get_parameter("traffic_light_housing_close_kernel").value),
        )
        self._housing_min_width = int(
            self.get_parameter("traffic_light_housing_min_width").value
        )
        self._housing_min_height = int(
            self.get_parameter("traffic_light_housing_min_height").value
        )
        self._housing_max_width_px = int(
            self.get_parameter("traffic_light_housing_max_width_px").value
        )
        self._housing_max_height_px = int(
            self.get_parameter("traffic_light_housing_max_height_px").value
        )
        self._housing_min_aspect = float(
            self.get_parameter("traffic_light_housing_min_aspect").value
        )
        self._housing_max_aspect = float(
            self.get_parameter("traffic_light_housing_max_aspect").value
        )
        self._housing_min_fill = float(
            self.get_parameter("traffic_light_housing_min_fill").value
        )
        self._housing_max_fill = float(
            self.get_parameter("traffic_light_housing_max_fill").value
        )
        self._housing_max_context_dark_ratio = float(
            self.get_parameter("traffic_light_housing_max_context_dark_ratio").value
        )
        self._housing_seed_min_pixels = int(
            self.get_parameter("traffic_light_housing_seed_min_pixels").value
        )
        self._housing_seed_min_circularity = float(
            self.get_parameter("traffic_light_housing_seed_min_circularity").value
        )
        self._housing_seed_min_ellipse_ratio = float(
            self.get_parameter("traffic_light_housing_seed_min_ellipse_ratio").value
        )
        self._housing_bright_hue_min = int(
            self.get_parameter("traffic_light_housing_bright_hue_min").value
        )
        self._housing_bright_hue_max = int(
            self.get_parameter("traffic_light_housing_bright_hue_max").value
        )
        self._housing_bright_min_saturation = float(
            self.get_parameter("traffic_light_housing_bright_min_saturation").value
        )
        self._housing_bright_min_value = float(
            self.get_parameter("traffic_light_housing_bright_min_value").value
        )
        self._housing_max_context_bright_ratio = float(
            self.get_parameter("traffic_light_housing_max_context_bright_ratio").value
        )
        self._housing_pad_ratio = float(
            self.get_parameter("traffic_light_housing_pad_ratio").value
        )
        self._housing_max_candidates = max(
            1,
            int(self.get_parameter("traffic_light_housing_max_candidates").value),
        )

        self._housing_confidence_bonus = float(
            self.get_parameter("traffic_light_housing_confidence_bonus").value
        )
        self._no_housing_confidence_penalty = float(
            self.get_parameter("traffic_light_no_housing_confidence_penalty").value
        )
        self._housing_bulb_min_pixels = int(
            self.get_parameter("traffic_light_housing_bulb_min_pixels").value
        )
        self._housing_bulb_min_circularity = float(
            self.get_parameter("traffic_light_housing_bulb_min_circularity").value
        )
        self._housing_bulb_min_center_fill = float(
            self.get_parameter("traffic_light_housing_bulb_min_center_fill").value
        )
        self._housing_bulb_min_color_ratio = float(
            self.get_parameter("traffic_light_housing_bulb_min_color_ratio").value
        )
        self._housing_bulb_min_ellipse_ratio = float(
            self.get_parameter("traffic_light_housing_bulb_min_ellipse_ratio").value
        )
        self._no_housing_min_color_ratio = float(
            self.get_parameter("traffic_light_no_housing_min_color_ratio").value
        )
        self._require_housing_for_state = bool(
            self.get_parameter("traffic_light_require_housing_for_state").value
        )
        history_size = max(1, int(self.get_parameter("traffic_light_history_size").value))
        self._history_confirm_count = max(
            1,
            min(
                history_size,
                int(self.get_parameter("traffic_light_history_confirm_count").value),
            ),
        )
        self._state_history = deque(maxlen=history_size)
        self._track_iou = float(self.get_parameter("traffic_light_track_iou").value)
        self._track_history = max(1, int(self.get_parameter("traffic_light_track_history").value))
        self._track_min_count = max(
            1, int(self.get_parameter("traffic_light_track_min_count").value)
        )
        self._track_max_misses = max(
            0, int(self.get_parameter("traffic_light_track_max_misses").value)
        )
        self._tracks: list[dict] = []
        self._detection_iou_suppress = float(
            self.get_parameter("traffic_light_detection_iou_suppress").value
        )
        self._use_yolo = bool(
            self.get_parameter("traffic_light_use_yolo").value
        )
        self._yolo_confidence_threshold = float(
            self.get_parameter("traffic_light_yolo_confidence_threshold").value
        )
        self._yolo_max_detections = max(
            1,
            int(self.get_parameter("traffic_light_yolo_max_detections").value),
        )
        self._yolo_min_color_pixels = max(
            1,
            int(self.get_parameter("traffic_light_yolo_min_color_pixels").value),
        )
        self._yolo_min_color_ratio = float(
            self.get_parameter("traffic_light_yolo_min_color_ratio").value
        )
        self._yolo_min_value = float(
            self.get_parameter("traffic_light_yolo_min_value").value
        )
        self._yolo_imgsz = int(
            self.get_parameter("traffic_light_yolo_imgsz").value
        )
        self._yolo_small_bbox_area = int(
            self.get_parameter("traffic_light_yolo_small_bbox_area").value
        )
        self._yolo_small_min_value = float(
            self.get_parameter("traffic_light_yolo_small_min_value").value
        )
        self._yolo_classify_mode = str(
            self.get_parameter("traffic_light_yolo_classify_mode").value
        ).strip().lower()
        self._bulb_bright_percentile = float(
            self.get_parameter("traffic_light_bulb_bright_percentile").value
        )
        self._bulb_min_value = float(
            self.get_parameter("traffic_light_bulb_min_value").value
        )
        self._bulb_yellow_min_value = float(
            self.get_parameter("traffic_light_bulb_yellow_min_value").value
        )
        self._bulb_hue_min_saturation = float(
            self.get_parameter("traffic_light_bulb_hue_min_saturation").value
        )
        self._bulb_red_hue_max = int(self.get_parameter("traffic_light_bulb_red_hue_max").value)
        self._bulb_red_hue_min = int(self.get_parameter("traffic_light_bulb_red_hue_min").value)
        self._bulb_yellow_hue_min = int(
            self.get_parameter("traffic_light_bulb_yellow_hue_min").value
        )
        self._bulb_yellow_hue_max = int(
            self.get_parameter("traffic_light_bulb_yellow_hue_max").value
        )
        self._bulb_green_hue_min = int(
            self.get_parameter("traffic_light_bulb_green_hue_min").value
        )
        self._bulb_green_hue_max = int(
            self.get_parameter("traffic_light_bulb_green_hue_max").value
        )
        self._yolo_min_brightness_margin = float(
            self.get_parameter("traffic_light_yolo_min_brightness_margin").value
        )
        self._yolo_min_center_fill = float(
            self.get_parameter("traffic_light_yolo_min_center_fill").value
        )
        self._yolo_model = None
        self._yolo_model_error_logged = False
        if self._use_yolo:
            self._load_yolo_model()

        self._debug_images = bool(
            self.get_parameter("traffic_light_debug_images").value
        )
        self._debug_hsv_images = bool(
            self.get_parameter("traffic_light_debug_hsv_images").value
        )
        self._debug_roi_images = bool(
            self.get_parameter("traffic_light_debug_roi_images").value
        )
        self._debug_stride = max(
            1,
            int(self.get_parameter("traffic_light_debug_stride").value),
        )
        self._show_windows = bool(
            self.get_parameter("traffic_light_show_windows").value
        )
        self._show_rejected_components = bool(
            self.get_parameter("traffic_light_show_rejected_components").value
        )

        self._publisher = self.create_publisher(TrafficLightState, state_topic, 10)
        self._label_publisher = self.create_publisher(String, label_topic, 10)
        self._detections_publisher = self.create_publisher(
            TrafficLightDetections,
            detections_topic,
            10,
        )
        self._debug_roi_pub = None
        self._debug_hsv_pub = None
        if self._debug_images:
            # Primary/unified debug topic - full-frame annotated image.
            self._debug_detection_pub = self.create_publisher(
                Image,
                self.get_parameter("traffic_light_debug_detection_topic").value,
                qos_profile_sensor_data,
            )
            # Redundant secondary views, only advertised when explicitly enabled.
            if self._debug_roi_images:
                self._debug_roi_pub = self.create_publisher(
                    Image,
                    self.get_parameter("traffic_light_debug_roi_topic").value,
                    qos_profile_sensor_data,
                )
            if self._debug_hsv_images:
                self._debug_hsv_pub = self.create_publisher(
                    Image,
                    self.get_parameter("traffic_light_debug_hsv_topic").value,
                    qos_profile_sensor_data,
                )
        self._frame_count = 0

        self._world_frame_id = self.get_parameter("world_frame_id").value
        self._min_depth_m = float(
            self.get_parameter("traffic_light_min_depth_m").value
        )
        self._max_depth_m = float(
            self.get_parameter("traffic_light_max_depth_m").value
        )
        self._require_depth_for_state = bool(
            self.get_parameter("traffic_light_require_depth_for_state").value
        )
        self._state_min_depth_m = float(
            self.get_parameter("traffic_light_state_min_depth_m").value
        )
        self._state_max_depth_m = float(
            self.get_parameter("traffic_light_state_max_depth_m").value
        )
        self._state_max_lateral_angle_ratio = float(
            self.get_parameter("traffic_light_state_max_lateral_angle_ratio").value
        )
        self._depth_camera_info: CameraInfo | None = None
        self._latest_depth_m: np.ndarray | None = None
        self._latest_depth_frame_id: str = ""
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        # Snapshot the globally-loaded values of every waypoint-overridable
        # parameter *after* all of them have been declared/loaded above, so
        # we can always fall back to "no override" cleanly.
        self._base_param_values = {
            param_name: getattr(self, attr_name)
            for param_name, (attr_name, _caster) in self._TUNABLE_PARAMS.items()
        }
        self._waypoint_overrides = self._parse_waypoint_overrides(
            self.get_parameter("traffic_light_waypoint_overrides").value
        )
        self._active_waypoint_range: tuple[int, int] | None = None

        self._latest_waypoint_index: int | None = None
        self.create_subscription(
            Int32,
            self.get_parameter("waypoint_index_topic").value,
            self._waypoint_index_callback,
            10,
        )

        self.create_subscription(
            CameraInfo,
            self.get_parameter("depth_camera_info_topic").value,
            self._depth_info_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Image,
            self.get_parameter("depth_image_topic").value,
            self._depth_callback,
            qos_profile_sensor_data,
        )

        self.create_subscription(
            Image,
            image_topic,
            self._image_callback,
            qos_profile_sensor_data,
        )

        self.get_logger().info(
            f"Traffic light detector subscribing to {image_topic}, "
            f"publishing {state_topic}"
        )
        if self._show_windows:
            self.get_logger().info(
                "OpenCV debug windows enabled: traffic light detection, ROI, HSV masks"
            )
            # cv2 stacks new windows at the same default screen position, so
            # without explicit placement the three windows hide each other.
            cv2.namedWindow("traffic_light_detection", cv2.WINDOW_NORMAL)
            cv2.namedWindow("traffic_light_roi", cv2.WINDOW_NORMAL)
            cv2.namedWindow("traffic_light_hsv_masks", cv2.WINDOW_NORMAL)
            cv2.moveWindow("traffic_light_detection", 0, 0)
            cv2.moveWindow("traffic_light_roi", 700, 0)
            cv2.moveWindow("traffic_light_hsv_masks", 0, 550)

    def _load_yolo_model(self) -> None:
        model_path = str(self.get_parameter("traffic_light_yolo_model_path").value).strip()
        if not model_path:
            self.get_logger().warn(
                "traffic_light_use_yolo is true but traffic_light_yolo_model_path is empty; "
                "falling back to HSV/housing detector."
            )
            return

        resolved = Path(os.path.expanduser(model_path)).resolve()
        if not resolved.exists():
            self.get_logger().warn(
                f"traffic_light_yolo_model_path does not exist: {resolved}; "
                "falling back to HSV/housing detector."
            )
            return

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            self.get_logger().warn(
                "ultralytics is not installed in this ROS Python environment; "
                f"falling back to HSV/housing detector: {exc}"
            )
            return

        self._yolo_model = YOLO(str(resolved))
        self.get_logger().info(f"Loaded traffic-light YOLO model: {resolved}")

    def _parse_waypoint_overrides(
        self, raw: str
    ) -> list[tuple[int, int, dict]]:
        raw = (raw or "").strip()
        if not raw or raw == "{}":
            return []
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            self.get_logger().warn(
                f"traffic_light_waypoint_overrides is not valid JSON, ignoring: {exc}"
            )
            return []

        ranges: list[tuple[int, int, dict]] = []
        for key, overrides in data.items():
            key = str(key).strip()
            try:
                if "-" in key:
                    lo_s, hi_s = key.split("-", 1)
                    lo, hi = int(lo_s), int(hi_s)
                else:
                    lo = hi = int(key)
            except ValueError:
                self.get_logger().warn(
                    f"Skipping unparsable waypoint override key: {key!r}"
                )
                continue
            ranges.append((lo, hi, dict(overrides)))
        return ranges

    def _apply_param_overrides(self, overrides: dict) -> None:
        applied = []
        for param_name, value in overrides.items():
            entry = self._TUNABLE_PARAMS.get(param_name)
            if entry is None:
                self.get_logger().warn(
                    f"Ignoring unknown tunable param in waypoint override: {param_name}"
                )
                continue
            attr_name, caster = entry
            setattr(self, attr_name, caster(value))
            applied.append(param_name)

        # Anything this override doesn't mention reverts to the global
        # (params.yaml/CLI) value, so overrides never "leak" between zones.
        for param_name, (attr_name, _caster) in self._TUNABLE_PARAMS.items():
            if param_name not in overrides:
                setattr(self, attr_name, self._base_param_values[param_name])

        if applied:
            self.get_logger().info(
                f"waypoint {self._latest_waypoint_index}: "
                f"applied traffic-light param overrides {applied}"
            )

    def _waypoint_index_callback(self, msg: Int32) -> None:
        index = int(msg.data)
        # Flush the temporal-confirmation history whenever the waypoint moves:
        # votes from the previous location's light must not carry over and
        # keep a stale state confirmed after we've driven past it.
        if self._latest_waypoint_index is not None and index != self._latest_waypoint_index:
            self._state_history.clear()
            self._tracks.clear()
        self._latest_waypoint_index = index

        matched = None
        for lo, hi, overrides in self._waypoint_overrides:
            if lo <= index <= hi:
                matched = (lo, hi, overrides)
                break

        new_range = (matched[0], matched[1]) if matched else None
        if new_range == self._active_waypoint_range:
            return
        self._active_waypoint_range = new_range
        self._apply_param_overrides(matched[2] if matched else {})

    def _depth_info_callback(self, msg: CameraInfo) -> None:
        self._depth_camera_info = msg

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

    def _sample_depth_point(
        self,
        full_image_bbox: tuple[float, float, float, float],
        image_width: int,
        image_height: int,
    ) -> tuple[float, float, float] | None:
        """
        Back-project a detection's bbox center into the depth camera frame.

        Approximate: samples the depth camera at the same normalized pixel
        location as the RGB-left detection, even though the two cameras sit
        ~0.2m apart (see sensor_tf_broadcaster.py). At the distances traffic
        lights are detected from (several meters and up) this parallax error
        is small; it is not a precise per-pixel stereo correspondence.
        """
        if self._latest_depth_m is None or self._depth_camera_info is None:
            return None
        if image_width <= 0 or image_height <= 0:
            return None

        bx0, by0, bx1, by1 = full_image_bbox
        center_x = 0.5 * (bx0 + bx1)
        center_y = 0.5 * (by0 + by1)

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
        rx, ry, rz = TrafficLightDetector._rotate_by_quaternion(
            point, (q.x, q.y, q.z, q.w)
        )
        return (rx + t.x, ry + t.y, rz + t.z)

    @staticmethod
    def _rotate_by_quaternion(
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

    def _augment_world_positions(self, debug: dict, image_shape) -> None:
        if not debug:
            return
        detections = debug.get("detections", [])
        if not detections:
            return

        # Look up the depth-camera -> world transform once per frame and
        # reuse it for every candidate: a per-candidate lookup_transform()
        # call (even with a short timeout) can add up when a frame has
        # several candidates, and was the main cause of the detector
        # falling behind the camera's already-slow (~2Hz) publish rate.
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

        x0, y0, _, _ = debug.get("roi_bounds", (0, 0, 0, 0))
        image_height, image_width = image_shape[:2]
        for candidate in detections:
            bbox = candidate.get("bbox")
            if bbox is None:
                candidate["camera_xyz"] = None
                candidate["world_xyz"] = None
                continue
            bx0, by0, bx1, by1 = bbox
            full_bbox = (x0 + bx0, y0 + by0, x0 + bx1, y0 + by1)
            cam_point = self._sample_depth_point(full_bbox, image_width, image_height)
            candidate["camera_xyz"] = cam_point
            if transform is None:
                candidate["world_xyz"] = None
                continue
            candidate["world_xyz"] = (
                self._transform_point(transform, cam_point)
                if cam_point is not None
                else None
            )

    def _apply_spatial_filter(self, debug: dict) -> tuple[int, float]:
        """
        Filter candidates by depth-camera geometry before publishing state.

        The image detector can still mark a rear-facing or side-street light
        as "traffic-light-looking". For driving, only candidates with a
        plausible forward depth and a reasonable lateral angle should affect
        /perception/traffic_light_state. Rejected candidates remain in
        debug/detections with a reject reason so tuning stays visible.
        """
        accepted_detections: list[dict] = []
        for candidate in debug.get("detections", []):
            reason = self._spatial_reject_reason(candidate)
            if reason is None:
                candidate["spatial_accepted"] = True
                if (
                    not self._require_housing_for_state
                    or candidate.get("has_housing", False)
                ):
                    accepted_detections.append(candidate)
            else:
                candidate["spatial_accepted"] = False
                candidate["spatial_reject_reason"] = reason

        states = (
            TrafficLightState.RED,
            TrafficLightState.YELLOW,
            TrafficLightState.GREEN,
        )
        filtered: dict[int, dict] = {}
        for state in states:
            state_candidates = [
                candidate
                for candidate in accepted_detections
                if candidate.get("state") == state
            ]
            if state_candidates:
                filtered[state] = max(
                    state_candidates,
                    key=lambda candidate: candidate.get("score", 0.0),
                )
            else:
                filtered[state] = {
                    "score": 0.0,
                    "confidence": 0.0,
                    "bbox": None,
                    "area": 0,
                    "has_housing": False,
                    "spatial_accepted": False,
                }

        confidences = {
            state: float(candidate.get("confidence", 0.0))
            for state, candidate in filtered.items()
        }
        best_state = self._select_output_state(confidences)
        best_confidence = confidences.get(best_state, 0.0)

        debug["candidates_before_spatial_filter"] = debug.get("candidates", {})
        debug["candidates"] = filtered
        debug["state_confidences"] = confidences
        debug["best_state"] = best_state
        debug["spatial_filter_enabled"] = True

        if best_confidence <= 0.0:
            return TrafficLightState.UNKNOWN, 0.0
        return best_state, float(best_confidence)

    def _spatial_reject_reason(self, candidate: dict) -> str | None:
        cam_point = candidate.get("camera_xyz")
        if cam_point is None:
            return "no_depth" if self._require_depth_for_state else None

        x_cam, _, z_cam = cam_point
        if not math.isfinite(z_cam) or z_cam <= 0.0:
            return "invalid_depth"
        if z_cam < self._state_min_depth_m:
            return "too_close"
        if z_cam > self._state_max_depth_m:
            return "too_far"

        lateral_ratio = abs(x_cam) / max(z_cam, 0.1)
        candidate["lateral_angle_ratio"] = lateral_ratio
        if lateral_ratio > self._state_max_lateral_angle_ratio:
            return "off_axis"

        return None

    def _image_callback(self, msg: Image) -> None:
        state_msg = TrafficLightState()
        state_msg.header = msg.header

        try:
            image = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as exc:
            self.get_logger().warn(f"Failed to convert RGB image: {exc}")
            state_msg.state = TrafficLightState.UNKNOWN
            state_msg.confidence = 0.0
            self._publisher.publish(state_msg)
            self._publish_label(state_msg, {})
            self._publish_detections(msg.header, image.shape if "image" in locals() else (), {})
            return

        _, _, debug = self._classify_image(image)
        self._augment_world_positions(debug, image.shape)
        raw_state, raw_confidence = self._apply_spatial_filter(debug)
        state_msg.state, state_msg.confidence = self._confirm_state(
            raw_state,
            raw_confidence,
            debug,
        )
        self._smooth_detections(debug)
        self._publisher.publish(state_msg)
        self._publish_label(state_msg, debug)
        self._publish_detections(msg.header, image.shape, debug)
        self._frame_count += 1
        if self._debug_images and self._frame_count % self._debug_stride == 0:
            self._publish_debug_images(msg.header, image, debug, state_msg)
        if self._show_windows:
            self._show_debug_windows(image, debug, state_msg)

    def _classify_image(self, bgr_image: np.ndarray) -> tuple[int, float, dict]:
        if bgr_image.size == 0:
            return TrafficLightState.UNKNOWN, 0.0, {}

        roi, roi_bounds = self._crop_roi(bgr_image)
        if roi.size == 0:
            return TrafficLightState.UNKNOWN, 0.0, {"roi_bounds": roi_bounds}

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        red_low = cv2.inRange(
            hsv,
            np.array([0, self._red_min_saturation, self._red_min_value], dtype=np.uint8),
            np.array([22, 255, 255], dtype=np.uint8),
        )
        red_high = cv2.inRange(
            hsv,
            np.array(
                [165, max(self._red_min_saturation, 85.0), self._red_high_min_value],
                dtype=np.uint8,
            ),
            np.array([179, 255, 255], dtype=np.uint8),
        )
        red_mask = cv2.bitwise_or(red_low, red_high)
        yellow_mask = cv2.inRange(
            hsv,
            np.array([23, self._yellow_min_saturation, self._yellow_min_value], dtype=np.uint8),
            np.array([38, 255, 255], dtype=np.uint8),
        )
        if not self._detect_yellow:
            yellow_mask = np.zeros_like(yellow_mask)
        green_mask = cv2.inRange(
            hsv,
            np.array([40, self._green_min_saturation, self._green_min_value], dtype=np.uint8),
            np.array([90, 255, 255], dtype=np.uint8),
        )

        if self._use_yolo and self._yolo_model is not None:
            yolo_result = self._classify_yolo_boxes(
                bgr_image,
                roi,
                roi_bounds,
                hsv,
                red_mask,
                yellow_mask,
                green_mask,
            )
            if yolo_result is not None:
                return yolo_result

        best, detections, housings, rejected = self._detect_candidates(
            hsv,
            red_mask,
            yellow_mask,
            green_mask,
        )

        output_candidates = best
        if self._require_housing_for_state:
            output_candidates = self._housing_confirmed_best(best)

        confidences = {
            state: float(candidate.get("confidence", 0.0))
            for state, candidate in output_candidates.items()
        }
        best_state = self._select_output_state(confidences)
        best_confidence = confidences[best_state]

        debug = {
            "roi": roi,
            "roi_bounds": roi_bounds,
            "hsv": hsv,
            "red_mask": red_mask,
            "yellow_mask": yellow_mask,
            "green_mask": green_mask,
            "candidates": output_candidates,
            "raw_candidates": best,
            "detections": detections,
            "housings": housings,
            "rejected": rejected,
            "best_state": best_state,
            "state_confidences": confidences,
        }

        if best_confidence <= 0.0:
            return TrafficLightState.UNKNOWN, 0.0, debug

        return best_state, float(best_confidence), debug

    def _classify_yolo_boxes(
        self,
        bgr_image: np.ndarray,
        roi: np.ndarray,
        roi_bounds: tuple[int, int, int, int],
        hsv: np.ndarray,
        red_mask: np.ndarray,
        yellow_mask: np.ndarray,
        green_mask: np.ndarray,
    ) -> tuple[int, float, dict] | None:
        try:
            # Runs on the full frame, not the ROI crop: the box detector was
            # trained on full 640x480 frames (build_box_detector_dataset.py
            # doesn't crop), so cropping here before inference was a second
            # train/inference mismatch on top of the imgsz one - it shrinks
            # small/distant lights relative to what the model learned, and a
            # different aspect ratio besides. The ROI restriction is applied
            # below as a post-filter on detection centers instead.
            results = self._yolo_model.predict(
                bgr_image,
                conf=self._yolo_confidence_threshold,
                imgsz=self._yolo_imgsz,
                verbose=False,
            )
        except Exception as exc:  # pragma: no cover - depends on runtime model
            if not self._yolo_model_error_logged:
                self.get_logger().warn(
                    f"Traffic-light YOLO inference failed; falling back to HSV/housing: {exc}"
                )
                self._yolo_model_error_logged = True
            return None

        if not results:
            return None

        boxes = getattr(results[0], "boxes", None)
        if boxes is None or len(boxes) == 0:
            return self._empty_yolo_debug(roi, roi_bounds, hsv, red_mask, yellow_mask, green_mask)

        rx0, ry0, rx1, ry1 = roi_bounds
        # Filter to traffic-light-class boxes that overlap the ROI *before*
        # applying the top-N cap. Slicing the raw box list first let unrelated
        # classes or off-ROI boxes consume the yolo_max_detections budget and
        # silently drop real lights when a frame had more than N total boxes.
        candidate_boxes = []
        for box in boxes:
            if not self._is_yolo_traffic_light_box(box):
                continue
            xyxy = box.xyxy[0].detach().cpu().numpy()
            yolo_confidence = float(box.conf[0].detach().cpu().item())
            full_bbox = self._clip_bbox(
                tuple(int(round(value)) for value in xyxy),
                bgr_image.shape[1],
                bgr_image.shape[0],
            )
            if full_bbox is None:
                continue
            # Same "only look where we expect a light" restriction the ROI
            # pre-crop used to enforce, now applied post-hoc instead of
            # hiding the rest of the frame from YOLO. Uses overlap, not the
            # box center: a close/large light's box extends well past the
            # ROI's bottom edge (roi_bottom_ratio) since it fills more
            # of the frame at close range, so its *center* can end up below
            # the ROI even though most of the light is still in it - center
            # gating was silently dropping every close-range detection.
            fx0, fy0, fx1, fy1 = full_bbox
            if fx1 <= rx0 or fx0 >= rx1 or fy1 <= ry0 or fy0 >= ry1:
                continue
            candidate_boxes.append((yolo_confidence, full_bbox))

        candidate_boxes.sort(key=lambda item: item[0], reverse=True)
        detections: list[dict] = []
        for yolo_confidence, full_bbox in candidate_boxes[: self._yolo_max_detections]:
            fx0, fy0, fx1, fy1 = full_bbox

            # hsv/red_mask/yellow_mask/green_mask below are ROI-cropped
            # arrays, so translate the full-frame box into ROI-local coords
            # to index into them.
            bbox = self._clip_bbox(
                (fx0 - rx0, fy0 - ry0, fx1 - rx0, fy1 - ry0),
                roi.shape[1],
                roi.shape[0],
            )
            if bbox is None:
                continue

            candidate = self._classify_yolo_bbox(
                bbox,
                yolo_confidence,
                hsv,
                red_mask,
                yellow_mask,
                green_mask,
            )
            detections.append(candidate)

        detections.sort(key=lambda cand: cand.get("confidence", 0.0), reverse=True)
        best: dict[int, dict] = {}
        for state in self._enabled_signal_states():
            state_candidates = [cand for cand in detections if cand.get("state") == state]
            if state_candidates:
                best[state] = max(state_candidates, key=lambda cand: cand.get("score", 0.0))
            else:
                best[state] = {"score": 0.0, "confidence": 0.0, "bbox": None, "area": 0}

        confidences = {
            state: float(candidate.get("confidence", 0.0))
            for state, candidate in best.items()
        }
        best_state = self._select_output_state(confidences)
        best_confidence = confidences.get(best_state, 0.0)

        debug = {
            "roi": roi,
            "roi_bounds": roi_bounds,
            "hsv": hsv,
            "red_mask": red_mask,
            "yellow_mask": yellow_mask,
            "green_mask": green_mask,
            "candidates": best,
            "raw_candidates": best,
            "detections": detections,
            "housings": [],
            "rejected": [],
            "best_state": best_state,
            "state_confidences": confidences,
            "detector_backend": "yolo_hsv",
        }

        if best_confidence <= 0.0:
            return TrafficLightState.UNKNOWN, 0.0, debug
        return best_state, float(best_confidence), debug

    def _is_yolo_traffic_light_box(self, box) -> bool:
        """Accept full traffic-light boxes, not red/yellow/green bulb boxes."""
        if self._yolo_model is None:
            return True
        names = getattr(self._yolo_model, "names", {}) or {}
        if len(names) <= 1:
            return True
        try:
            cls_id = int(box.cls[0].detach().cpu().item())
        except Exception:
            return True
        class_name = str(names.get(cls_id, cls_id)).lower()
        return class_name in {"traffic_light", "traffic light", "tl"}

    def _empty_yolo_debug(
        self,
        roi: np.ndarray,
        roi_bounds: tuple[int, int, int, int],
        hsv: np.ndarray,
        red_mask: np.ndarray,
        yellow_mask: np.ndarray,
        green_mask: np.ndarray,
    ) -> tuple[int, float, dict]:
        empty = {
            TrafficLightState.RED: {"score": 0.0, "confidence": 0.0, "bbox": None, "area": 0},
            TrafficLightState.YELLOW: {"score": 0.0, "confidence": 0.0, "bbox": None, "area": 0},
            TrafficLightState.GREEN: {"score": 0.0, "confidence": 0.0, "bbox": None, "area": 0},
        }
        return TrafficLightState.UNKNOWN, 0.0, {
            "roi": roi,
            "roi_bounds": roi_bounds,
            "hsv": hsv,
            "red_mask": red_mask,
            "yellow_mask": yellow_mask,
            "green_mask": green_mask,
            "candidates": empty,
            "raw_candidates": empty,
            "detections": [],
            "housings": [],
            "rejected": [],
            "best_state": TrafficLightState.UNKNOWN,
            "state_confidences": {state: 0.0 for state in empty},
            "detector_backend": "yolo_hsv",
        }

    def _classify_yolo_bbox(
        self,
        bbox: tuple[int, int, int, int],
        yolo_confidence: float,
        hsv: np.ndarray,
        red_mask: np.ndarray,
        yellow_mask: np.ndarray,
        green_mask: np.ndarray,
    ) -> dict:
        x0, y0, x1, y1 = bbox
        height = max(1, y1 - y0)
        thirds = {
            TrafficLightState.RED: (x0, y0, x1, y0 + height // 3),
            TrafficLightState.YELLOW: (x0, y0 + height // 3, x1, y0 + 2 * height // 3),
            TrafficLightState.GREEN: (x0, y0 + 2 * height // 3, x1, y1),
        }
        masks = {
            TrafficLightState.RED: red_mask,
            TrafficLightState.YELLOW: yellow_mask,
            TrafficLightState.GREEN: green_mask,
        }

        # Bulb mode (default): read the actual brightest lit bulb - its hue and
        # its vertical position - instead of the rigid thirds. Handles small
        # boxes too, so it replaces both branches below when enabled.
        if self._yolo_classify_mode == "bulb":
            return self._classify_yolo_bbox_bulb(bbox, yolo_confidence, hsv, masks)

        # Small/distant box: the vertical thirds + center-crop below needs a
        # few pixels per region to be meaningful. Below this area it degenerates
        # (~1px per third), so fall back to a whole-box dominant-color vote.
        if (x1 - x0) * (y1 - y0) <= self._yolo_small_bbox_area:
            return self._classify_yolo_bbox_small(bbox, yolo_confidence, hsv, masks)

        # Center-crop brightness per third (not masked) - a bulb socket sits
        # in the middle of its third, with dark housing frame/gaps along the
        # edges, so averaging the *whole* third dilutes a real bulb's signal
        # with those edges (and, empirically, can even score an unlit third
        # higher than a lit one if its edges catch more ambient light). The
        # gold housing color itself sits in all three thirds roughly evenly,
        # so this comparison is used below to require the lit bulb to stand
        # out *relative to the other two thirds on the same housing*, instead
        # of only clearing a fixed absolute brightness threshold that gold
        # can also clear.
        region_brightness: dict[int, float] = {}
        for state, region in thirds.items():
            rx0, ry0, rx1, ry1 = region
            patch_hsv = hsv[ry0:ry1, rx0:rx1]
            h, w = patch_hsv.shape[:2]
            ccx0, ccx1 = int(w * 0.3), max(int(w * 0.3) + 1, int(w * 0.7))
            ccy0, ccy1 = int(h * 0.3), max(int(h * 0.3) + 1, int(h * 0.7))
            center_patch = patch_hsv[ccy0:ccy1, ccx0:ccx1]
            region_brightness[state] = (
                float(np.mean(center_patch[:, :, 2])) if center_patch.size else 0.0
            )

        scores: dict[int, dict] = {}
        for state, region in thirds.items():
            if state == TrafficLightState.YELLOW and not self._detect_yellow:
                continue
            if state == TrafficLightState.GREEN and not self._detect_green:
                continue
            rx0, ry0, rx1, ry1 = region
            patch_mask = masks[state][ry0:ry1, rx0:rx1]
            patch_hsv = hsv[ry0:ry1, rx0:rx1]
            area = max(1, patch_mask.shape[0] * patch_mask.shape[1])
            lit_pixels = int(np.count_nonzero(patch_mask))
            color_ratio = lit_pixels / float(area)
            mean_value = 0.0
            if lit_pixels > 0:
                mean_value = float(np.mean(patch_hsv[:, :, 2][patch_mask > 0]))
            value_score = np.clip(mean_value / max(1.0, self._yolo_min_value), 0.0, 1.0)
            ratio_score = np.clip(color_ratio / max(1e-6, self._yolo_min_color_ratio), 0.0, 1.0)

            other_brightness = max(
                (b for s, b in region_brightness.items() if s != state),
                default=0.0,
            )
            brightness_margin = region_brightness[state] - other_brightness
            margin_score = np.clip(
                brightness_margin / max(1.0, self._yolo_min_brightness_margin),
                0.0,
                1.0,
            )

            # Reuses the same center-fill check the classic (non-YOLO) bulb
            # path already uses: is the matched color a solid blob sitting in
            # the middle of this third, not scattered/edge pixels.
            center_fill = self._center_fill(patch_mask, 0, 0, rx1 - rx0, ry1 - ry0)
            fill_score = np.clip(
                center_fill / max(1e-6, self._yolo_min_center_fill), 0.0, 1.0
            )

            scores[state] = {
                "region": region,
                "lit_pixels": lit_pixels,
                "color_ratio": color_ratio,
                "mean_value": mean_value,
                "brightness_margin": brightness_margin,
                "center_fill": center_fill,
                "score": float(ratio_score * value_score * margin_score * fill_score),
            }

        state = max(scores, key=lambda item: scores[item]["score"])
        state_score = scores[state]["score"]
        # brightness_margin and center_fill are already folded into
        # state_score multiplicatively above - they pull confidence down for
        # a weak/edge-only match without an extra hard cutoff here, which
        # would otherwise double-penalize borderline-but-real detections
        # (distant lights, dim exposure) straight to UNKNOWN.
        if (
            scores[state]["lit_pixels"] < self._yolo_min_color_pixels
            or scores[state]["color_ratio"] < self._yolo_min_color_ratio
            or scores[state]["mean_value"] < self._yolo_min_value
            or state_score <= 0.0
        ):
            lit_total = sum(item["lit_pixels"] for item in scores.values())
            yolo_status = "box_only" if lit_total == 0 else "unknown"
            if lit_total > 0:
                # A box was found and *some* color signal was picked up, but
                # it fell short of being classified - log the winning
                # candidate's actual numbers so a real run shows exactly
                # which threshold is the bottleneck instead of guessing.
                best_candidate = scores[state]
                self.get_logger().info(
                    "yolo box classified UNKNOWN: bbox=%s ratio=%.3f (min=%.3f) "
                    "mean_v=%.1f (min=%.1f) margin=%.1f (min=%.1f) fill=%.3f (min=%.3f)"
                    % (
                        bbox,
                        best_candidate["color_ratio"],
                        self._yolo_min_color_ratio,
                        best_candidate["mean_value"],
                        self._yolo_min_value,
                        best_candidate["brightness_margin"],
                        self._yolo_min_brightness_margin,
                        best_candidate["center_fill"],
                        self._yolo_min_center_fill,
                    ),
                    throttle_duration_sec=1.0,
                )
            state = TrafficLightState.UNKNOWN
            confidence = 0.0
        else:
            yolo_status = "classified"
            confidence = float(np.clip(0.35 * yolo_confidence + 0.65 * state_score, 0.0, 1.0))

        return {
            "score": confidence,
            "confidence": confidence,
            "state": state,
            "bbox": bbox,
            "circle": None,
            "area": int((x1 - x0) * (y1 - y0)),
            "extent": 1.0,
            "center_fill": (
                float(scores.get(state, {}).get("center_fill", 0.0))
                if state != TrafficLightState.UNKNOWN
                else 0.0
            ),
            "circularity": 1.0,
            "ellipse_ratio": min(x1 - x0, y1 - y0) / float(max(1, max(x1 - x0, y1 - y0))),
            "color_ratio": (
                float(scores.get(state, {}).get("color_ratio", 0.0))
                if state != TrafficLightState.UNKNOWN
                else 0.0
            ),
            "sign_ratio": 0.0,
            "housing_context": 1.0,
            "frame_context": 1.0,
            "center_y_ratio": (y0 + 0.5 * (y1 - y0)) / float(max(1, hsv.shape[0])),
            "dark_context": 1.0,
            "has_housing": True,
            "housing_bbox": bbox,
            "yolo_bbox": bbox,
            "yolo_confidence": yolo_confidence,
            "yolo_color_scores": scores,
            "yolo_status": yolo_status,
            "detector_backend": "yolo_hsv",
        }

    def _classify_yolo_bbox_small(
        self,
        bbox: tuple[int, int, int, int],
        yolo_confidence: float,
        hsv: np.ndarray,
        masks: dict[int, np.ndarray],
    ) -> dict:
        """
        Use a whole-box dominant-color vote for tiny or distant traffic lights.

        Skips the vertical thirds and center-crop (which need several pixels
        per region) and instead picks the color with the most lit pixels over
        the whole box, using a relaxed value floor. Keeps the same output
        schema as _classify_yolo_bbox so downstream code is unaffected.
        """
        x0, y0, x1, y1 = bbox
        area = max(1, (x1 - x0) * (y1 - y0))

        scores: dict[int, dict] = {}
        for state, mask in masks.items():
            if state not in self._enabled_signal_states():
                continue
            patch_mask = mask[y0:y1, x0:x1]
            patch_hsv = hsv[y0:y1, x0:x1]
            lit_pixels = int(np.count_nonzero(patch_mask))
            color_ratio = lit_pixels / float(area)
            mean_value = (
                float(np.mean(patch_hsv[:, :, 2][patch_mask > 0])) if lit_pixels > 0 else 0.0
            )
            scores[state] = {
                "lit_pixels": lit_pixels,
                "color_ratio": color_ratio,
                "mean_value": mean_value,
                # Rank by absolute lit-pixel count: on a 6px light the winning
                # color is simply whichever bulb is actually lit.
                "score": float(lit_pixels),
            }

        state = (
            max(scores, key=lambda item: scores[item]["score"])
            if scores
            else TrafficLightState.UNKNOWN
        )
        if (
            not scores
            or scores[state]["lit_pixels"] < self._yolo_min_color_pixels
            or scores[state]["mean_value"] < self._yolo_small_min_value
        ):
            state = TrafficLightState.UNKNOWN
            confidence = 0.0
            color_ratio = 0.0
            yolo_status = "unknown"
        else:
            color_ratio = scores[state]["color_ratio"]
            # Small boxes are inherently less certain: weight YOLO's own
            # confidence more and cap the ceiling so a distant read never
            # looks as trustworthy as a large, structurally-confirmed one.
            weighted_confidence = (
                0.6 * yolo_confidence + 0.4 * min(1.0, color_ratio * 4.0)
            )
            confidence = float(np.clip(weighted_confidence, 0.0, 0.75))
            yolo_status = "classified_small"

        return {
            "score": confidence,
            "confidence": confidence,
            "state": state,
            "bbox": bbox,
            "circle": None,
            "area": int(area),
            "extent": 1.0,
            "center_fill": 0.0,
            "circularity": 1.0,
            "ellipse_ratio": min(x1 - x0, y1 - y0) / float(max(1, max(x1 - x0, y1 - y0))),
            "color_ratio": float(color_ratio),
            "sign_ratio": 0.0,
            "housing_context": 1.0,
            "frame_context": 1.0,
            "center_y_ratio": (y0 + 0.5 * (y1 - y0)) / float(max(1, hsv.shape[0])),
            "dark_context": 1.0,
            "has_housing": True,
            "housing_bbox": bbox,
            "yolo_bbox": bbox,
            "yolo_confidence": yolo_confidence,
            "yolo_color_scores": scores,
            "yolo_status": yolo_status,
            "detector_backend": "yolo_hsv",
        }

    @staticmethod
    def _clip_bbox(
        bbox: tuple[int, int, int, int],
        width: int,
        height: int,
    ) -> tuple[int, int, int, int] | None:
        x0, y0, x1, y1 = bbox
        x0 = int(np.clip(x0, 0, max(0, width - 1)))
        y0 = int(np.clip(y0, 0, max(0, height - 1)))
        x1 = int(np.clip(x1, 0, width))
        y1 = int(np.clip(y1, 0, height))
        if x1 <= x0 or y1 <= y0:
            return None
        return x0, y0, x1, y1

    def _housing_confirmed_best(self, best: dict[int, dict]) -> dict[int, dict]:
        confirmed = {}
        for state, candidate in best.items():
            if candidate.get("has_housing", False):
                confirmed[state] = candidate
            else:
                confirmed[state] = {
                    "score": 0.0,
                    "confidence": 0.0,
                    "bbox": None,
                    "area": 0,
                    "has_housing": False,
                }
        return confirmed

    def _confirm_state(
        self,
        raw_state: int,
        raw_confidence: float,
        debug: dict,
    ) -> tuple[int, float]:
        if raw_confidence < self._confidence_threshold:
            raw_state = TrafficLightState.UNKNOWN
            raw_confidence = 0.0

        self._state_history.append((raw_state, float(raw_confidence)))

        debug["raw_state"] = raw_state
        debug["raw_confidence"] = raw_confidence
        debug["history"] = list(self._state_history)

        priority = (
            TrafficLightState.RED,
            TrafficLightState.YELLOW,
            TrafficLightState.GREEN,
        )
        for state in priority:
            matches = [confidence for item, confidence in self._state_history if item == state]
            if len(matches) >= self._history_confirm_count:
                return state, float(np.mean(matches))

        return TrafficLightState.UNKNOWN, 0.0

    def _select_output_state(self, confidences: dict[int, float]) -> int:
        # Convenience/debug aggregate only. The authoritative perception output
        # is /perception/traffic_light_detections (every light with bbox, color,
        # confidence and world position); Decision selects the light relevant to
        # the current waypoint from that array. This scalar deliberately does NOT
        # apply a red-priority override anymore - that was perception making the
        # stop/go decision on the whole frame, which is Decision's job. Plain
        # argmax over per-color confidence keeps the topic informative without
        # letting a side-street or cross-traffic red hijack the global state.
        if not confidences:
            return TrafficLightState.UNKNOWN
        return max(confidences, key=confidences.get)

    def _hue_to_state(self, hue: float) -> int | None:
        """
        Map an OpenCV hue (0-179) to a signal color, or None if ambiguous.

        Bands deliberately exclude orange H~14-22 because it dominates the
        simulator's gold housing. Position is an additional validity check,
        not a substitute for a real color core.
        """
        if hue <= self._bulb_red_hue_max or hue >= self._bulb_red_hue_min:
            return TrafficLightState.RED
        if self._bulb_yellow_hue_min <= hue <= self._bulb_yellow_hue_max:
            return TrafficLightState.YELLOW
        if self._bulb_green_hue_min <= hue <= self._bulb_green_hue_max:
            return TrafficLightState.GREEN
        return None

    def _bulb_box_only(
        self, bbox: tuple[int, int, int, int], yolo_confidence: float, hsv: np.ndarray
    ) -> dict:
        """Report a YOLO box with no readable lit bulb and no color."""
        x0, y0, x1, y1 = bbox
        return {
            "score": 0.0,
            "confidence": 0.0,
            "state": TrafficLightState.UNKNOWN,
            "bbox": bbox,
            "circle": None,
            "area": int((x1 - x0) * (y1 - y0)),
            "extent": 1.0,
            "center_fill": 0.0,
            "circularity": 1.0,
            "ellipse_ratio": min(x1 - x0, y1 - y0) / float(max(1, max(x1 - x0, y1 - y0))),
            "color_ratio": 0.0,
            "sign_ratio": 0.0,
            "housing_context": 1.0,
            "frame_context": 1.0,
            "center_y_ratio": (y0 + 0.5 * (y1 - y0)) / float(max(1, hsv.shape[0])),
            "dark_context": 1.0,
            "has_housing": True,
            "housing_bbox": bbox,
            "yolo_bbox": bbox,
            "yolo_confidence": yolo_confidence,
            "yolo_color_scores": {},
            "yolo_status": "box_only",
            "detector_backend": "yolo_hsv",
        }

    def _classify_yolo_bbox_bulb(
        self,
        bbox: tuple[int, int, int, int],
        yolo_confidence: float,
        hsv: np.ndarray,
        masks: dict[int, np.ndarray],
    ) -> dict:
        """
        Read bulb color using hue and vertical position in the default mode.

        For each enabled color, finds lit pixels that are bright, saturated AND
        in that color's tight hue band, takes the largest such blob, and scores
        it by size x brightness with a boost when it sits in the color's
        expected vertical band (top=red / mid=yellow / bottom=green). The
        strongest wins. Validated on sim frames: this reliably reads the lit red
        bulb across tight and loose boxes, and the brightness+tight-hue gate
        excludes the gold housing (V~200, hue~22) that made the rigid-thirds
        method paint red into every third.
        """
        x0, y0, x1, y1 = bbox
        patch = hsv[y0:y1, x0:x1]
        ph, pw = patch.shape[:2]
        if ph < 2 or pw < 2:
            return self._bulb_box_only(bbox, yolo_confidence, hsv)

        H = patch[:, :, 0].astype(np.int32)
        S = patch[:, :, 1].astype(np.int32)
        V = patch[:, :, 2].astype(np.int32)

        # Expected vertical band of each color's bulb, with tolerance for a
        # loose YOLO box.
        expected = {
            TrafficLightState.RED: (0.0, 0.42),
            TrafficLightState.YELLOW: (0.28, 0.72),
            TrafficLightState.GREEN: (0.58, 1.0),
        }

        best_state = TrafficLightState.UNKNOWN
        best_score = 0.0
        best_pos = -1.0
        best_hue = -1.0
        best_area = 0
        for signal in self._enabled_signal_states():
            band = self._bulb_hue_mask(H, signal)
            # YELLOW keeps a high brightness floor (its hue overlaps the gold
            # housing); RED/GREEN use a low floor so distant/dim bulbs survive.
            v_floor = (
                self._bulb_yellow_min_value
                if signal == TrafficLightState.YELLOW
                else self._bulb_min_value
            )
            cand = (
                band
                & (V >= v_floor)
                & (S >= self._bulb_hue_min_saturation)
            ).astype(np.uint8)
            if int(np.count_nonzero(cand)) < self._yolo_min_color_pixels:
                continue
            lo, hi = expected[signal]
            num, labels, stats, centroids = cv2.connectedComponentsWithStats(
                cand, connectivity=8
            )
            for lbl in range(1, num):
                area = int(stats[lbl, cv2.CC_STAT_AREA])
                min_component_pixels = self._yolo_min_color_pixels
                if signal == TrafficLightState.RED and ph * pw <= 400:
                    # A distant red core can genuinely be a single tight H~0
                    # pixel after resize. Its top-slot and strict-hue checks
                    # below still reject arbitrary orange housing pixels.
                    min_component_pixels = 1
                if area < min_component_pixels:
                    continue
                width = int(stats[lbl, cv2.CC_STAT_WIDTH])
                height = int(stats[lbl, cv2.CC_STAT_HEIGHT])
                shape_ratio = min(width, height) / float(max(1, max(width, height)))
                extent = area / float(max(1, width * height))
                if shape_ratio < 0.35 or extent < 0.35:
                    continue
                pos_ratio = float(centroids[lbl][1]) / float(max(1, ph))
                # A colored reflection in the wrong slot is not a bulb. The
                # old 0.6 penalty still let a large gold strip win.
                if not lo <= pos_ratio <= hi:
                    continue
                x_ratio = float(centroids[lbl][0]) / float(max(1, pw))
                # Side-facing housings expose bright yellow edge strips but no
                # readable bulb face. A visible bulb core must remain inside
                # the housing rather than hugging either lateral edge.
                x_lo, x_hi = (
                    (0.08, 0.94)
                    if signal == TrafficLightState.RED
                    else (0.18, 0.82)
                )
                if not x_lo <= x_ratio <= x_hi:
                    continue
                comp = labels == lbl
                score = area * float(V[comp].mean())
                if score > best_score:
                    best_score = score
                    best_state = signal
                    best_pos = pos_ratio
                    best_hue = float(np.median(H[comp]))
                    best_area = area

        if best_state == TrafficLightState.UNKNOWN:
            # A box was found but no bright, saturated, correctly-hued bulb -
            # report the box without a color rather than guessing. Log the box's
            # actual peak values so a live run shows whether a threshold
            # (bulb_min_value / hue_min_saturation) is the bottleneck.
            red_band = self._bulb_hue_mask(H, TrafficLightState.RED)
            self.get_logger().info(
                "bulb box_only: maxV=%d maxS=%d red-hue px=%d (need V>=%d S>=%d)"
                % (
                    int(V.max()),
                    int(S.max()),
                    int(np.count_nonzero(red_band & (S >= self._bulb_hue_min_saturation))),
                    int(self._bulb_min_value),
                    int(self._bulb_hue_min_saturation),
                ),
                throttle_duration_sec=1.0,
            )
            return self._bulb_box_only(bbox, yolo_confidence, hsv)

        lo, hi = expected[best_state]
        quality = 0.9 if lo <= best_pos <= hi else 0.6
        confidence = float(np.clip(0.4 * yolo_confidence + 0.6 * quality, 0.0, 0.9))
        state = best_state
        yolo_status = "classified_bulb"
        pos_ratio = best_pos
        mean_hue = best_hue
        color_ratio = best_area / float(max(1, ph * pw))
        return {
            "score": confidence,
            "confidence": confidence,
            "state": state,
            "bbox": bbox,
            "circle": None,
            "area": int((x1 - x0) * (y1 - y0)),
            "extent": 1.0,
            "center_fill": 0.0,
            "circularity": 1.0,
            "ellipse_ratio": min(x1 - x0, y1 - y0) / float(max(1, max(x1 - x0, y1 - y0))),
            "color_ratio": color_ratio,
            "sign_ratio": 0.0,
            "housing_context": 1.0,
            "frame_context": 1.0,
            "center_y_ratio": (y0 + 0.5 * (y1 - y0)) / float(max(1, hsv.shape[0])),
            "dark_context": 1.0,
            "has_housing": True,
            "housing_bbox": bbox,
            "yolo_bbox": bbox,
            "yolo_confidence": yolo_confidence,
            "yolo_color_scores": {
                "bulb_pos_ratio": pos_ratio,
                "bulb_hue": mean_hue,
                "bulb_area": best_area,
            },
            "yolo_status": yolo_status,
            "detector_backend": "yolo_hsv",
        }

    def _bulb_hue_mask(self, H: np.ndarray, state: int) -> np.ndarray:
        """
        Build a mask of pixels whose hue falls in the state's tight band.

        Red wraps around 0, so it is `<= red_hue_max OR >= red_hue_min`.
        Orange H~14-22 is intentionally excluded from both red and yellow
        because it is the simulator housing's dominant color.
        """
        if state == TrafficLightState.RED:
            return (H <= self._bulb_red_hue_max) | (H >= self._bulb_red_hue_min)
        if state == TrafficLightState.YELLOW:
            return (H >= self._bulb_yellow_hue_min) & (H <= self._bulb_yellow_hue_max)
        if state == TrafficLightState.GREEN:
            return (H >= self._bulb_green_hue_min) & (H <= self._bulb_green_hue_max)
        return np.zeros_like(H, dtype=bool)

    def _enabled_signal_states(self) -> tuple[int, ...]:
        states = [TrafficLightState.RED]
        if self._detect_yellow:
            states.append(TrafficLightState.YELLOW)
        if self._detect_green:
            states.append(TrafficLightState.GREEN)
        return tuple(states)

    def _crop_roi(self, bgr_image: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        height, width = bgr_image.shape[:2]
        y0 = int(np.clip(self._roi_top_ratio, 0.0, 1.0) * height)
        y1 = int(np.clip(self._roi_bottom_ratio, 0.0, 1.0) * height)
        x0 = int(np.clip(self._roi_left_ratio, 0.0, 1.0) * width)
        x1 = int(np.clip(self._roi_right_ratio, 0.0, 1.0) * width)

        if y1 <= y0 or x1 <= x0:
            return bgr_image[0:0, 0:0], (0, 0, 0, 0)
        return bgr_image[y0:y1, x0:x1], (x0, y0, x1, y1)

    # ------------------------------------------------------------------
    # Stage 1: housing/frame detection
    # ------------------------------------------------------------------

    def _detect_housing_regions(
        self,
        hsv: np.ndarray,
        colored_mask: np.ndarray,
    ) -> list[dict]:
        """
        Find dark, elongated boxes that look like a traffic light body.

        A real traffic light housing is a small, mostly-dark, opaque
        enclosure that (a) sits against open background rather than being a
        fragment of a much larger dark structure, and (b) contains a small
        round RED/YELLOW/GREEN blob, not just any bright pixel. Bridges,
        tunnel mouths, brick walls, and building shadows can satisfy a bare
        "dark box with a bright pixel inside" test, which is why each of
        those three checks exists.
        """
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]
        hue = hsv[:, :, 0]

        dark_mask = (
            (value < self._housing_dark_value)
            | (
                (value < self._housing_dark_value_soft)
                & (saturation < self._housing_dark_saturation_soft)
            )
        ).astype(np.uint8) * 255
        dark_housings = self._find_housings_in_mask(
            dark_mask,
            colored_mask,
            self._housing_max_context_dark_ratio,
        )

        # Some signal heads use a bright yellow/gold casing instead of a
        # black one, which never satisfies the "dark" test above. Run a
        # second, parallel pass over a bright/saturated color range so those
        # still get a housing candidate; it goes through the same
        # size/aspect/fill/context/seed gates below, just against a
        # differently-defined structural mask.
        bright_mask = (
            (hue >= self._housing_bright_hue_min)
            & (hue <= self._housing_bright_hue_max)
            & (saturation > self._housing_bright_min_saturation)
            & (value > self._housing_bright_min_value)
        ).astype(np.uint8) * 255
        bright_housings = self._find_housings_in_mask(
            bright_mask,
            colored_mask,
            self._housing_max_context_bright_ratio,
        )

        housings = self._dedupe_housings(dark_housings + bright_housings)
        housings.sort(
            key=lambda cand: cand["seed_circularity"] * cand["fill"],
            reverse=True,
        )
        return housings[: self._housing_max_candidates]

    def _find_housings_in_mask(
        self,
        structural_mask: np.ndarray,
        colored_mask: np.ndarray,
        max_context_ratio: float,
    ) -> list[dict]:
        height, width = structural_mask.shape[:2]

        # Closing bridges the bright bulb hole(s) so the housing forms one
        # solid connected component instead of splitting into fragments.
        close_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self._housing_close_kernel, self._housing_close_kernel),
        )
        closed = cv2.morphologyEx(
            structural_mask, cv2.MORPH_CLOSE, close_kernel, iterations=2
        )
        closed = cv2.morphologyEx(closed, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            closed,
            connectivity=8,
        )

        housings = []
        for label in range(1, num_labels):
            x = int(stats[label, cv2.CC_STAT_LEFT])
            y = int(stats[label, cv2.CC_STAT_TOP])
            w = int(stats[label, cv2.CC_STAT_WIDTH])
            h = int(stats[label, cv2.CC_STAT_HEIGHT])

            if w < self._housing_min_width or h < self._housing_min_height:
                continue
            if w > self._housing_max_width_px or h > self._housing_max_height_px:
                continue

            elongation = max(w, h) / float(max(1, min(w, h)))
            if (
                elongation < self._housing_min_aspect
                or elongation > self._housing_max_aspect
            ):
                continue

            # Fill is measured against the *pre-closing* structural mask, not
            # the closed/hole-filled component: closing intentionally bridges
            # bulb holes so the housing forms one connected blob, but that
            # would also erase the "has holes" signal that separates a real
            # housing from a solid object (pole, tree trunk, shadow, wall)
            # if fill were measured post-closing.
            raw_area = int(np.count_nonzero(structural_mask[y:y + h, x:x + w]))
            fill = raw_area / float(max(1, w * h))
            if fill < self._housing_min_fill or fill > self._housing_max_fill:
                continue

            center_y_ratio = (y + 0.5 * h) / float(max(1, height))
            if center_y_ratio > self._max_candidate_y_ratio + 0.1:
                continue

            # Reject boxes that are just a window/gap into a much larger
            # mass of the same mask (bridge underside, tunnel mouth, wall,
            # pillar, or - for the bright mask - a large sign/billboard
            # panel) rather than an isolated pole-mounted housing.
            context_ratio = self._housing_context_dark_ratio(
                structural_mask, x, y, w, h
            )
            if context_ratio > max_context_ratio:
                continue

            seed = self._find_housing_seed(colored_mask, x, y, w, h)
            if seed is None:
                continue

            housings.append({
                "bbox": (x, y, x + w, y + h),
                "width": w,
                "height": h,
                "fill": fill,
                "elongation": elongation,
                "context_dark_ratio": context_ratio,
                "seed_area": seed["area"],
                "seed_circularity": seed["circularity"],
            })

        return housings

    def _dedupe_housings(self, housings: list[dict]) -> list[dict]:
        """Drop lower-quality duplicates when both mask passes find the same box."""
        ordered = sorted(
            housings,
            key=lambda cand: cand["seed_circularity"] * cand["fill"],
            reverse=True,
        )
        kept: list[dict] = []
        for candidate in ordered:
            if any(
                self._bbox_iou(candidate["bbox"], other["bbox"]) > 0.4
                for other in kept
            ):
                continue
            kept.append(candidate)
        return kept

    def _housing_context_dark_ratio(
        self,
        dark_mask: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int,
    ) -> float:
        pad_x = max(6, int(w * 2.0))
        pad_y = max(6, int(h * 1.0))
        x0 = max(0, x - pad_x)
        y0 = max(0, y - pad_y)
        x1 = min(dark_mask.shape[1], x + w + pad_x)
        y1 = min(dark_mask.shape[0], y + h + pad_y)
        if x1 <= x0 or y1 <= y0:
            return 0.0

        context = dark_mask[y0:y1, x0:x1]
        inner_x0, inner_y0 = x - x0, y - y0
        inner_x1, inner_y1 = inner_x0 + w, inner_y0 + h

        ring = np.ones(context.shape[:2], dtype=bool)
        ring[inner_y0:inner_y1, inner_x0:inner_x1] = False
        if not np.any(ring):
            return 0.0

        return float(np.count_nonzero(context[ring] > 0)) / float(np.count_nonzero(ring))

    def _find_housing_seed(
        self,
        colored_mask: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int,
    ) -> dict | None:
        """
        Require a small round RED/YELLOW/GREEN blob inside the box.

        This gate is intentionally stricter than the stage-2 in-housing
        bulb search: its job is precision (deciding whether this dark box
        is a real housing at all), while stage 2 assumes the housing is
        already trustworthy and instead maximizes recall of dim/far bulbs.
        """
        patch = colored_mask[y:y + h, x:x + w]
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(patch, connectivity=8)

        best = None
        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            lx = int(stats[label, cv2.CC_STAT_LEFT])
            ly = int(stats[label, cv2.CC_STAT_TOP])
            lw = int(stats[label, cv2.CC_STAT_WIDTH])
            lh = int(stats[label, cv2.CC_STAT_HEIGHT])

            if area < self._housing_seed_min_pixels or area > self._max_pixels:
                continue
            if lw > w * 0.9 or lh > h * 0.9:
                continue
            ellipse_ratio = min(lw, lh) / float(max(lw, lh))
            if ellipse_ratio < self._housing_seed_min_ellipse_ratio:
                continue
            circularity = self._component_circularity(labels, label, lx, ly, lw, lh)
            if circularity < self._housing_seed_min_circularity:
                continue

            if best is None or area > best["area"]:
                best = {"area": area, "circularity": circularity}

        return best

    # ------------------------------------------------------------------
    # Stage 2: bulb detection (inside a housing, or as a bare-blob fallback)
    # ------------------------------------------------------------------

    def _bulb_thresholds(self, has_housing: bool) -> dict:
        if has_housing:
            return {
                "min_pixels": self._housing_bulb_min_pixels,
                "min_circularity": self._housing_bulb_min_circularity,
                "min_center_fill": self._housing_bulb_min_center_fill,
                "min_color_ratio": self._housing_bulb_min_color_ratio,
                "min_ellipse_ratio": self._housing_bulb_min_ellipse_ratio,
            }
        return {
            "min_pixels": self._min_pixels,
            "min_circularity": self._min_circularity,
            "min_center_fill": self._min_center_fill,
            "min_color_ratio": self._no_housing_min_color_ratio,
            "min_ellipse_ratio": self._min_ellipse_ratio,
        }

    def _find_bulb_candidates(
        self,
        hsv: np.ndarray,
        masks: dict[int, np.ndarray],
        region: tuple[int, int, int, int],
        has_housing: bool,
        housing_bbox: tuple[int, int, int, int] | None,
        rejected: list[dict],
    ) -> list[dict]:
        rx0, ry0, rx1, ry1 = region
        if rx1 <= rx0 or ry1 <= ry0:
            return []

        thresholds = self._bulb_thresholds(has_housing)

        hsv_region = hsv[ry0:ry1, rx0:rx1]
        masks_region = {state: mask[ry0:ry1, rx0:rx1] for state, mask in masks.items()}
        saturation = hsv_region[:, :, 1]
        value = hsv_region[:, :, 2]
        shape_seed = (
            ((saturation > 55) & (value > 70))
            | ((saturation > 25) & (value > 175))
        ).astype(np.uint8) * 255

        if self._morph_kernel_size > 1:
            kernel = np.ones(
                (self._morph_kernel_size, self._morph_kernel_size),
                dtype=np.uint8,
            )
            shape_seed = cv2.morphologyEx(shape_seed, cv2.MORPH_OPEN, kernel)
            shape_seed = cv2.morphologyEx(shape_seed, cv2.MORPH_CLOSE, kernel)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            shape_seed,
            connectivity=8,
        )

        label_ids = range(1, num_labels)
        if num_labels - 1 > self._max_seed_components:
            label_ids = sorted(
                label_ids,
                key=lambda label: stats[label, cv2.CC_STAT_AREA],
                reverse=True,
            )[: self._max_seed_components]

        detections = []
        for label in label_ids:
            area = int(stats[label, cv2.CC_STAT_AREA])
            x = int(stats[label, cv2.CC_STAT_LEFT])
            y = int(stats[label, cv2.CC_STAT_TOP])
            width = int(stats[label, cv2.CC_STAT_WIDTH])
            height = int(stats[label, cv2.CC_STAT_HEIGHT])
            abs_x, abs_y = x + rx0, y + ry0

            def reject(reason: str) -> None:
                rejected.append({
                    "bbox": (abs_x, abs_y, abs_x + width, abs_y + height),
                    "area": area,
                    "reason": reason,
                    "has_housing": has_housing,
                })

            if area < thresholds["min_pixels"] or area > self._max_pixels:
                reject("area")
                continue

            if width < self._min_bbox_width or height < self._min_bbox_height:
                reject("small")
                continue

            center_y_ratio = (abs_y + 0.5 * height) / float(max(1, hsv.shape[0]))
            if center_y_ratio > self._max_candidate_y_ratio:
                reject("low")
                continue

            aspect_ratio = width / float(height)
            if (
                aspect_ratio < self._min_aspect_ratio
                or aspect_ratio > self._max_aspect_ratio
            ):
                reject("shape")
                continue

            extent = area / float(width * height)
            if extent < self._min_extent or extent > self._max_extent:
                reject("extent")
                continue

            circularity = self._component_circularity(labels, label, x, y, width, height)
            ellipse_ratio = min(width, height) / float(max(width, height))
            if (
                circularity < thresholds["min_circularity"]
                or ellipse_ratio < thresholds["min_ellipse_ratio"]
            ):
                reject("circle")
                continue

            center_fill = self._center_fill(shape_seed, x, y, width, height)
            if center_fill < thresholds["min_center_fill"]:
                reject("ring")
                continue

            component_mask = labels[y:y + height, x:x + width] == label
            total_pixels = max(1, int(np.count_nonzero(component_mask)))
            color_counts = {
                state: int(
                    np.count_nonzero(
                        mask[y:y + height, x:x + width][component_mask]
                    )
                )
                for state, mask in masks_region.items()
            }
            state = max(color_counts, key=color_counts.get)
            color_ratio = color_counts[state] / float(total_pixels)
            if color_counts[state] < 1 or color_ratio < thresholds["min_color_ratio"]:
                reject("color")
                continue

            sign_ratio = self._same_color_context_ratio(
                masks_region[state],
                x,
                y,
                width,
                height,
            )
            dark_context = self._dark_context_ratio(hsv_region, x, y, width, height)
            housing_context = self._housing_context_ratio(hsv_region, x, y, width, height)
            frame_context = self._frame_context_ratio(hsv_region, x, y, width, height)

            # The context/sign checks exist to catch bare colored blobs that
            # aren't real lights (signs, billboards, tail lights). A bulb
            # already confirmed to sit inside a detected housing doesn't need
            # them re-applied, since the housing itself is the confirmation.
            if not has_housing:
                if sign_ratio > self._max_same_color_context_ratio and area > 6:
                    reject("sign")
                    continue
                if dark_context < self._min_dark_context_ratio:
                    reject("context")
                    continue
                if (
                    state == TrafficLightState.GREEN
                    and housing_context < max(self._min_housing_context_ratio, 0.18)
                ):
                    reject("housing")
                    continue
                if frame_context < self._min_frame_context_ratio:
                    reject("frame")
                    continue

            circle = self._component_circle(labels, label, x, y, width, height)
            circle = self._offset_circle(circle, rx0, ry0)

            shape_score = (
                0.50 * min(1.0, circularity)
                + 0.25 * ellipse_ratio
                + 0.25 * np.clip(1.0 - abs(extent - 0.65) / 0.45, 0.0, 1.0)
            )
            color_score = np.clip(color_ratio, 0.0, 1.0)
            area_score = np.clip(area / 40.0, 0.0, 1.0)
            confidence = (
                0.10
                + 0.45 * shape_score
                + 0.35 * color_score
                + 0.10 * np.sqrt(area_score)
            )
            confidence = float(np.clip(confidence, 0.0, 1.0))

            if has_housing:
                confidence = float(
                    np.clip(confidence + self._housing_confidence_bonus, 0.0, 1.0)
                )
            else:
                confidence = float(
                    np.clip(confidence * self._no_housing_confidence_penalty, 0.0, 1.0)
                )

            score = confidence * (1.0 + 0.15 * min(1.0, dark_context / 0.35))
            score *= 1.0 + 0.20 * min(1.0, housing_context / 0.35)
            score *= 1.0 + 0.20 * min(1.0, frame_context / 0.25)

            detections.append({
                "score": score,
                "confidence": confidence,
                "state": state,
                "bbox": (abs_x, abs_y, abs_x + width, abs_y + height),
                "circle": circle,
                "area": area,
                "extent": extent,
                "center_fill": center_fill,
                "circularity": circularity,
                "ellipse_ratio": ellipse_ratio,
                "color_ratio": color_ratio,
                "sign_ratio": sign_ratio,
                "housing_context": housing_context,
                "frame_context": frame_context,
                "center_y_ratio": center_y_ratio,
                "dark_context": dark_context,
                "has_housing": has_housing,
                "housing_bbox": housing_bbox,
            })

        return detections

    def _detect_candidates(
        self,
        hsv: np.ndarray,
        red_mask: np.ndarray,
        yellow_mask: np.ndarray,
        green_mask: np.ndarray,
    ) -> tuple[dict[int, dict], list[dict], list[dict], list[dict]]:
        masks = {
            TrafficLightState.RED: red_mask,
            TrafficLightState.YELLOW: yellow_mask,
        }
        if self._detect_green:
            masks[TrafficLightState.GREEN] = green_mask

        roi_height, roi_width = hsv.shape[:2]
        colored_mask = red_mask | yellow_mask
        if self._detect_green:
            colored_mask = colored_mask | green_mask
        housings = self._detect_housing_regions(hsv, colored_mask)

        all_detections = []
        rejected: list[dict] = []
        claimed = np.zeros((roi_height, roi_width), dtype=bool)

        for housing in housings:
            hx0, hy0, hx1, hy1 = housing["bbox"]
            pad_x = max(2, int(round(housing["width"] * self._housing_pad_ratio)))
            pad_y = max(2, int(round(housing["height"] * self._housing_pad_ratio * 0.5)))
            rx0 = max(0, hx0 - pad_x)
            ry0 = max(0, hy0 - pad_y)
            rx1 = min(roi_width, hx1 + pad_x)
            ry1 = min(roi_height, hy1 + pad_y)

            bulbs = self._find_bulb_candidates(
                hsv,
                masks,
                (rx0, ry0, rx1, ry1),
                has_housing=True,
                housing_bbox=housing["bbox"],
                rejected=rejected,
            )
            for bulb in bulbs:
                bulb["housing_fill"] = housing["fill"]
            all_detections.extend(bulbs)
            claimed[hy0:hy1, hx0:hx1] = True

        # Fallback: search the rest of the ROI for bare colored blobs that
        # didn't fall inside any detected housing (distant/occluded lights).
        # These are held to stricter checks and a confidence penalty above.
        fallback = self._find_bulb_candidates(
            hsv,
            masks,
            (0, 0, roi_width, roi_height),
            has_housing=False,
            housing_bbox=None,
            rejected=rejected,
        )
        for bulb in fallback:
            bx0, by0, bx1, by1 = bulb["bbox"]
            cx = min(roi_width - 1, max(0, (bx0 + bx1) // 2))
            cy = min(roi_height - 1, max(0, (by0 + by1) // 2))
            if claimed[cy, cx]:
                continue
            all_detections.append(bulb)

        all_detections = self._suppress_overlapping(all_detections)
        all_detections.sort(key=lambda cand: cand.get("confidence", 0.0), reverse=True)

        best: dict[int, dict] = {}
        for state in masks:
            state_candidates = [c for c in all_detections if c["state"] == state]
            if state_candidates:
                best[state] = max(state_candidates, key=lambda c: c.get("score", 0.0))
            else:
                best[state] = {"score": 0.0, "confidence": 0.0, "bbox": None, "area": 0}

        return best, all_detections, housings, rejected

    def _suppress_overlapping(self, detections: list[dict]) -> list[dict]:
        """
        Drop lower-confidence duplicates of the same physical bulb.

        Overlapping housing search regions (from the padding margin) can
        occasionally pick up the same bulb twice; keep only the strongest.
        """
        ordered = sorted(detections, key=lambda cand: cand.get("confidence", 0.0), reverse=True)
        kept: list[dict] = []
        for candidate in ordered:
            bbox = candidate.get("bbox")
            if bbox is None:
                continue
            if any(
                self._bbox_iou(bbox, other["bbox"]) > self._detection_iou_suppress
                for other in kept
            ):
                continue
            kept.append(candidate)
        return kept

    @staticmethod
    def _bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
        ax0, ay0, ax1, ay1 = a
        bx0, by0, bx1, by1 = b
        ix0, iy0 = max(ax0, bx0), max(ay0, by0)
        ix1, iy1 = min(ax1, bx1), min(ay1, by1)
        if ix1 <= ix0 or iy1 <= iy0:
            return 0.0
        intersection = (ix1 - ix0) * (iy1 - iy0)
        area_a = max(0, ax1 - ax0) * max(0, ay1 - ay0)
        area_b = max(0, bx1 - bx0) * max(0, by1 - by0)
        union = area_a + area_b - intersection
        if union <= 0:
            return 0.0
        return float(intersection) / float(union)

    @staticmethod
    def _offset_circle(
        circle: tuple[int, int, int] | None,
        dx: int,
        dy: int,
    ) -> tuple[int, int, int] | None:
        if circle is None:
            return None
        cx, cy, radius = circle
        return (cx + dx, cy + dy, radius)

    def _component_mean_sv(
        self,
        hsv: np.ndarray,
        labels: np.ndarray,
        label: int,
        x: int,
        y: int,
        width: int,
        height: int,
    ) -> tuple[float, float]:
        component = labels[y:y + height, x:x + width] == label
        if not np.any(component):
            return 0.0, 0.0
        patch = hsv[y:y + height, x:x + width]
        return (
            float(np.mean(patch[:, :, 1][component])),
            float(np.mean(patch[:, :, 2][component])),
        )

    def _dark_context_ratio(
        self,
        hsv: np.ndarray,
        x: int,
        y: int,
        width: int,
        height: int,
    ) -> float:
        pad_x = max(4, int(width * 1.8))
        pad_y = max(4, int(height * 1.8))
        x0 = max(0, x - pad_x)
        y0 = max(0, y - pad_y)
        x1 = min(hsv.shape[1], x + width + pad_x)
        y1 = min(hsv.shape[0], y + height + pad_y)
        if x1 <= x0 or y1 <= y0:
            return 0.0

        context = hsv[y0:y1, x0:x1]
        inner_x0 = x - x0
        inner_y0 = y - y0
        inner_x1 = inner_x0 + width
        inner_y1 = inner_y0 + height

        ring = np.ones(context.shape[:2], dtype=bool)
        ring[inner_y0:inner_y1, inner_x0:inner_x1] = False
        if not np.any(ring):
            return 0.0

        saturation = context[:, :, 1]
        value = context[:, :, 2]
        dark = ((value < 90) | ((value < 130) & (saturation < 90))) & ring
        return float(np.count_nonzero(dark)) / float(np.count_nonzero(ring))

    def _housing_context_ratio(
        self,
        hsv: np.ndarray,
        x: int,
        y: int,
        width: int,
        height: int,
    ) -> float:
        pad_x = max(3, int(width * 0.9))
        pad_y = max(3, int(height * 0.9))
        x0 = max(0, x - pad_x)
        y0 = max(0, y - pad_y)
        x1 = min(hsv.shape[1], x + width + pad_x)
        y1 = min(hsv.shape[0], y + height + pad_y)
        if x1 <= x0 or y1 <= y0:
            return 0.0

        context = hsv[y0:y1, x0:x1]
        inner_x0 = x - x0
        inner_y0 = y - y0
        inner_x1 = inner_x0 + width
        inner_y1 = inner_y0 + height

        ring = np.ones(context.shape[:2], dtype=bool)
        ring[inner_y0:inner_y1, inner_x0:inner_x1] = False
        if not np.any(ring):
            return 0.0

        saturation = context[:, :, 1]
        value = context[:, :, 2]
        dark_housing = ((value < 95) | ((value < 135) & (saturation < 80))) & ring
        return float(np.count_nonzero(dark_housing)) / float(np.count_nonzero(ring))

    def _frame_context_ratio(
        self,
        hsv: np.ndarray,
        x: int,
        y: int,
        width: int,
        height: int,
    ) -> float:
        pad_x = max(4, int(width * 1.2))
        pad_y = max(8, int(height * 3.0))
        x0 = max(0, x - pad_x)
        y0 = max(0, y - pad_y)
        x1 = min(hsv.shape[1], x + width + pad_x)
        y1 = min(hsv.shape[0], y + height + pad_y)
        if x1 <= x0 or y1 <= y0:
            return 0.0

        patch = hsv[y0:y1, x0:x1]
        saturation = patch[:, :, 1]
        value = patch[:, :, 2]
        dark_or_frame = (value < 95) | ((value < 145) & (saturation < 90))

        inner_x0 = x - x0
        inner_y0 = y - y0
        inner_x1 = inner_x0 + width
        inner_y1 = inner_y0 + height
        dark_or_frame[inner_y0:inner_y1, inner_x0:inner_x1] = False

        center_band_half_width = max(2, int(width * 0.9))
        center_x = inner_x0 + width // 2
        band_x0 = max(0, center_x - center_band_half_width)
        band_x1 = min(dark_or_frame.shape[1], center_x + center_band_half_width + 1)
        vertical_band = dark_or_frame[:, band_x0:band_x1]
        if vertical_band.size <= 0:
            return 0.0
        return float(np.count_nonzero(vertical_band)) / float(vertical_band.size)

    def _same_color_context_ratio(
        self,
        mask: np.ndarray,
        x: int,
        y: int,
        width: int,
        height: int,
    ) -> float:
        pad_x = max(4, int(width * 2.5))
        pad_y = max(4, int(height * 2.5))
        x0 = max(0, x - pad_x)
        y0 = max(0, y - pad_y)
        x1 = min(mask.shape[1], x + width + pad_x)
        y1 = min(mask.shape[0], y + height + pad_y)
        if x1 <= x0 or y1 <= y0:
            return 0.0

        patch = mask[y0:y1, x0:x1]
        patch_area = patch.size
        if patch_area <= 0:
            return 0.0
        return float(np.count_nonzero(patch)) / float(patch_area)

    def _publish_label(self, state_msg: TrafficLightState, debug: dict) -> None:
        confidences = debug.get("state_confidences", {})
        detections = debug.get("detections", [])
        housed_count = sum(1 for candidate in detections if candidate.get("has_housing", False))
        msg = String()
        msg.data = (
            f"{self._state_name(state_msg.state)} "
            f"confidence={state_msg.confidence:.2f}"
            f" | R={confidences.get(TrafficLightState.RED, 0.0):.2f}"
            f" Y={confidences.get(TrafficLightState.YELLOW, 0.0):.2f}"
            f" G={confidences.get(TrafficLightState.GREEN, 0.0):.2f}"
            f" | raw={self._state_name(debug.get('raw_state', TrafficLightState.UNKNOWN))}"
            f" | housed={housed_count}/{len(detections)}"
            f" | waypoint={self._latest_waypoint_index}"
        )
        self._label_publisher.publish(msg)

    def _smooth_detections(self, debug: dict) -> None:
        """
        Stabilize each light's reported color across frames.

        Associates current detections to short-lived tracks by full-frame bbox
        IoU and majority-votes the color over each track's recent history, so a
        single noisy frame can't flip a light's published state. Overrides
        ``state``/``confidence`` in place on debug["detections"]; the pre-smooth
        value is kept as ``raw_state`` for debugging.
        """
        detections = debug.get("detections", []) if debug else []
        x0, y0, _, _ = debug.get("roi_bounds", (0, 0, 0, 0)) if debug else (0, 0, 0, 0)

        frame_dets: list[tuple[dict, tuple[int, int, int, int]]] = []
        for cand in detections:
            bbox = cand.get("bbox")
            if bbox is None:
                continue
            bx0, by0, bx1, by1 = bbox
            frame_dets.append((cand, (x0 + bx0, y0 + by0, x0 + bx1, y0 + by1)))

        used: set[int] = set()
        for cand, fbox in frame_dets:
            best_i, best_iou = -1, self._track_iou
            for i, track in enumerate(self._tracks):
                if i in used:
                    continue
                iou = self._bbox_iou(fbox, track["bbox"])
                if iou >= best_iou:
                    best_iou = iou
                    best_i = i
            raw_state = int(cand.get("state", TrafficLightState.UNKNOWN))
            if best_i >= 0:
                track = self._tracks[best_i]
                track["bbox"] = fbox
                track["misses"] = 0
                track["history"].append(raw_state)
                used.add(best_i)
            else:
                track = {
                    "bbox": fbox,
                    "misses": 0,
                    "history": deque([raw_state], maxlen=self._track_history),
                }
                self._tracks.append(track)
                used.add(len(self._tracks) - 1)

            smoothed = self._smoothed_track_state(track["history"])
            cand["raw_state"] = raw_state
            cand["state"] = smoothed
            if smoothed == TrafficLightState.UNKNOWN:
                # Not yet consistent enough to trust - report UNKNOWN with no
                # confidence so Decision doesn't act on a one-frame flicker.
                cand["confidence"] = 0.0
            elif smoothed != raw_state:
                # Held over from history while this frame disagreed; keep it but
                # damp confidence to reflect the disagreement.
                cand["confidence"] = float(cand.get("confidence", 0.0)) * 0.6

        for i, track in enumerate(self._tracks):
            if i not in used:
                track["misses"] += 1
        self._tracks = [
            track for track in self._tracks if track["misses"] <= self._track_max_misses
        ]

    def _smoothed_track_state(self, history) -> int:
        counts = Counter(history)
        best_state = TrafficLightState.UNKNOWN
        best_count = 0
        for state, count in counts.items():
            if state == TrafficLightState.UNKNOWN:
                continue
            if count > best_count:
                best_state = state
                best_count = count
        if best_count < self._track_min_count:
            return TrafficLightState.UNKNOWN
        return best_state

    def _publish_detections(self, header, image_shape, debug: dict) -> None:
        msg = TrafficLightDetections()
        msg.header = header
        if not debug or len(image_shape) < 2:
            self._detections_publisher.publish(msg)
            return

        image_height, image_width = image_shape[:2]
        x0, y0, _, _ = debug.get("roi_bounds", (0, 0, 0, 0))

        for candidate in debug.get("detections", []):
            bbox = candidate.get("bbox")
            if bbox is None:
                continue

            bx0, by0, bx1, by1 = bbox
            full_x = float(x0 + bx0)
            full_y = float(y0 + by0)
            full_width = float(max(0, bx1 - bx0))
            full_height = float(max(0, by1 - by0))

            circle = candidate.get("circle")
            if circle is None:
                center_x = full_x + 0.5 * full_width
                center_y = full_y + 0.5 * full_height
                radius = 0.5 * max(full_width, full_height)
            else:
                cx, cy, detected_radius = circle
                center_x = float(x0 + cx)
                center_y = float(y0 + cy)
                radius = float(detected_radius)

            detection = TrafficLightDetection()
            detection.header = header
            detection.state = int(candidate.get("state", TrafficLightState.UNKNOWN))
            if candidate.get("yolo_status") == "box_only":
                detection.label = "traffic_light"
                detection.confidence = float(candidate.get("yolo_confidence", 0.0))
            else:
                detection.label = self._state_name(detection.state)
                detection.confidence = float(candidate.get("confidence", 0.0))
            detection.center_x = center_x
            detection.center_y = center_y
            detection.radius = radius
            detection.bbox_x = full_x
            detection.bbox_y = full_y
            detection.bbox_width = full_width
            detection.bbox_height = full_height
            detection.normalized_x = center_x / float(max(1, image_width))
            detection.normalized_y = center_y / float(max(1, image_height))
            detection.area = float(candidate.get("area", 0.0))
            detection.color_ratio = float(candidate.get("color_ratio", 0.0))
            detection.circularity = float(candidate.get("circularity", 0.0))
            detection.apparent_size = (
                full_width * full_height / float(max(1, image_width * image_height))
            )
            detection.has_housing = bool(candidate.get("has_housing", False))

            world_xyz = candidate.get("world_xyz")
            if world_xyz is not None:
                detection.has_world_position = True
                detection.world_x = float(world_xyz[0])
                detection.world_y = float(world_xyz[1])
                detection.world_z = float(world_xyz[2])
            else:
                detection.has_world_position = False

            msg.detections.append(detection)

        self._detections_publisher.publish(msg)

    def _center_fill(
        self,
        mask: np.ndarray,
        x: int,
        y: int,
        width: int,
        height: int,
    ) -> float:
        cx0 = x + max(0, int(width * 0.3))
        cx1 = x + min(width, int(width * 0.7))
        cy0 = y + max(0, int(height * 0.3))
        cy1 = y + min(height, int(height * 0.7))
        if cx1 <= cx0 or cy1 <= cy0:
            return 0.0

        center = mask[cy0:cy1, cx0:cx1]
        if center.size == 0:
            return 0.0
        return float(np.count_nonzero(center)) / float(center.size)

    def _component_circularity(
        self,
        labels: np.ndarray,
        label: int,
        x: int,
        y: int,
        width: int,
        height: int,
    ) -> float:
        component = (labels[y:y + height, x:x + width] == label).astype(np.uint8) * 255
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0

        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0.0:
            return 0.0
        return float(4.0 * np.pi * area / (perimeter * perimeter))

    def _component_circle(
        self,
        labels: np.ndarray,
        label: int,
        x: int,
        y: int,
        width: int,
        height: int,
    ) -> tuple[int, int, int] | None:
        component = (labels[y:y + height, x:x + width] == label).astype(np.uint8) * 255
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        contour = max(contours, key=cv2.contourArea)
        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        return (
            int(round(x + cx)),
            int(round(y + cy)),
            max(2, int(round(radius))),
        )

    def _publish_debug_images(
        self,
        header,
        bgr_image: np.ndarray,
        debug: dict,
        state_msg: TrafficLightState,
    ) -> None:
        if not debug:
            return

        roi = debug.get("roi")
        if self._debug_roi_pub is not None and roi is not None and roi.size > 0:
            roi_vis = roi.copy()
            self._draw_rejected_components(roi_vis, debug)
            self._draw_housings(roi_vis, debug)
            self._draw_candidates(roi_vis, debug)
            self._draw_state_summary(roi_vis, debug, state_msg)
            self._publish_bgr_image(self._debug_roi_pub, header, roi_vis)

        hsv = debug.get("hsv")
        if self._debug_hsv_pub is not None and hsv is not None:
            hsv_panel = self._make_hsv_panel(hsv, debug)
            self._publish_bgr_image(self._debug_hsv_pub, header, hsv_panel)

        detection = self._make_detection_image(bgr_image, debug, state_msg)
        self._publish_bgr_image(self._debug_detection_pub, header, detection)

    def _publish_bgr_image(self, publisher, header, bgr_image: np.ndarray) -> None:
        msg = self._bridge.cv2_to_imgmsg(bgr_image, encoding="bgr8")
        msg.header = header
        publisher.publish(msg)

    def _show_debug_windows(
        self,
        bgr_image: np.ndarray,
        debug: dict,
        state_msg: TrafficLightState,
    ) -> None:
        if not debug:
            return

        detection = self._make_detection_image(bgr_image, debug, state_msg)
        cv2.imshow("traffic_light_detection", detection)

        roi = debug.get("roi")
        if roi is not None and roi.size > 0:
            roi_vis = roi.copy()
            self._draw_rejected_components(roi_vis, debug)
            self._draw_housings(roi_vis, debug)
            self._draw_candidates(roi_vis, debug)
            self._draw_state_summary(roi_vis, debug, state_msg)
            cv2.imshow("traffic_light_roi", roi_vis)

        hsv = debug.get("hsv")
        if hsv is not None:
            hsv_panel = self._make_hsv_panel(hsv, debug)
            cv2.imshow("traffic_light_hsv_masks", hsv_panel)

        cv2.waitKey(1)

    def _make_hsv_panel(self, hsv: np.ndarray, debug: dict) -> np.ndarray:
        hsv_vis = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        red_vis = self._colorize_mask(debug["red_mask"], (0, 0, 255), "red mask")
        yellow_vis = self._colorize_mask(debug["yellow_mask"], (0, 255, 255), "yellow mask")
        green_vis = self._colorize_mask(debug["green_mask"], (0, 255, 0), "green mask")

        top = np.hstack((self._label_image(hsv_vis, "HSV view"), red_vis))
        bottom = np.hstack((yellow_vis, green_vis))
        return np.vstack((top, bottom))

    def _make_detection_image(
        self,
        bgr_image: np.ndarray,
        debug: dict,
        state_msg: TrafficLightState,
    ) -> np.ndarray:
        image = bgr_image.copy()
        x0, y0, x1, y1 = debug.get("roi_bounds", (0, 0, 0, 0))
        if x1 > x0 and y1 > y0:
            cv2.rectangle(image, (x0, y0), (x1, y1), (255, 255, 0), 2)
            cv2.putText(
                image,
                "traffic light ROI",
                (x0, max(20, y0 + 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2,
            )

        for housing in debug.get("housings", []):
            hx0, hy0, hx1, hy1 = housing["bbox"]
            cv2.rectangle(
                image,
                (x0 + hx0, y0 + hy0),
                (x0 + hx1, y0 + hy1),
                (200, 200, 200),
                1,
            )

        for candidate in debug.get("detections", []):
            if candidate.get("detector_backend") == "yolo_hsv":
                self._draw_yolo_candidate(image, candidate, x0, y0)
                continue
            if self._require_housing_for_state and not candidate.get("has_housing", False):
                continue
            if candidate.get("spatial_accepted") is False:
                bbox = candidate.get("bbox")
                if bbox is not None:
                    bx0, by0, bx1, by1 = bbox
                    cv2.rectangle(
                        image,
                        (x0 + bx0, y0 + by0),
                        (x0 + bx1, y0 + by1),
                        (120, 120, 120),
                        1,
                    )
                    cv2.putText(
                        image,
                        f"reject:{candidate.get('spatial_reject_reason', '')}",
                        (x0 + bx0, max(15, y0 + by0 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (120, 120, 120),
                        1,
                    )
                continue
            state = candidate.get("state", TrafficLightState.UNKNOWN)
            color = self._state_colors().get(state, (0, 0, 0))
            bbox = candidate.get("bbox")
            circle = candidate.get("circle")
            if circle is not None:
                cx, cy, radius = circle
                cv2.circle(
                    image,
                    (x0 + cx, y0 + cy),
                    radius,
                    color,
                    2,
                )
            if bbox is None:
                continue
            bx0, by0, bx1, by1 = bbox
            cv2.rectangle(
                image,
                (x0 + bx0, y0 + by0),
                (x0 + bx1, y0 + by1),
                color,
                1,
            )
            cv2.putText(
                image,
                self._tl_label(candidate.get("confidence", 0.0), state),
                (x0 + bx0, max(15, y0 + by0 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )
            world_xyz = candidate.get("world_xyz")
            if world_xyz is not None:
                wx, wy, wz = world_xyz
                cv2.putText(
                    image,
                    f"world X={wx:.1f} Y={wy:.1f} Z={wz:.1f}",
                    (x0 + bx0, min(image.shape[0] - 5, y0 + by1 + 15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    color,
                    1,
                )

        self._draw_state_summary(image, debug, state_msg)
        return image

    def _draw_yolo_candidate(
        self,
        image: np.ndarray,
        candidate: dict,
        roi_x: int = 0,
        roi_y: int = 0,
    ) -> None:
        bbox = candidate.get("bbox")
        if bbox is None:
            return
        bx0, by0, bx1, by1 = bbox
        state = candidate.get("state", TrafficLightState.UNKNOWN)
        status = candidate.get("yolo_status", "")
        box_only = status == "box_only"
        if box_only:
            color = (128, 128, 128)
            label = self._tl_label(candidate.get("confidence", 0.0), None)
        else:
            color = self._state_colors().get(state, (0, 0, 0))
            label = self._tl_label(candidate.get("confidence", 0.0), state)

        cv2.rectangle(
            image,
            (roi_x + bx0, roi_y + by0),
            (roi_x + bx1, roi_y + by1),
            color,
            1,
        )
        cv2.putText(
            image,
            label,
            (roi_x + bx0, max(15, roi_y + by0 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    def _draw_state_summary(
        self,
        image: np.ndarray,
        debug: dict,
        state_msg: TrafficLightState,
    ) -> None:
        """
        Draw the confirmed state, color confidence, and waypoint overlay.

        Shared between the full-frame detection image and the ROI-only
        image so both debug topics carry the same at-a-glance summary.
        """
        label = (
            f"{self._state_name(state_msg.state)} "
            f"confidence={state_msg.confidence:.2f}"
        )
        cv2.putText(
            image,
            label,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            self._state_colors().get(state_msg.state, (0, 0, 0)),
            2,
        )
        confidences = debug.get("state_confidences", {})
        score_label = (
            f"R={confidences.get(TrafficLightState.RED, 0.0):.2f} "
            f"Y={confidences.get(TrafficLightState.YELLOW, 0.0):.2f} "
            f"G={confidences.get(TrafficLightState.GREEN, 0.0):.2f}"
        )
        cv2.putText(
            image,
            score_label,
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )
        if self._latest_waypoint_index is not None:
            cv2.putText(
                image,
                f"waypoint={self._latest_waypoint_index}",
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )

    def _draw_housings(self, image: np.ndarray, debug: dict) -> None:
        for housing in debug.get("housings", []):
            x0, y0, x1, y1 = housing["bbox"]
            cv2.rectangle(image, (x0, y0), (x1, y1), (200, 200, 200), 1)
            cv2.putText(
                image,
                f"housing fill={housing.get('fill', 0.0):.2f}",
                (x0, max(15, y0 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (200, 200, 200),
                1,
            )

    def _draw_rejected_components(self, image: np.ndarray, debug: dict) -> None:
        if not self._show_rejected_components:
            return

        for component in debug.get("rejected", []):
            bbox = component.get("bbox")
            if bbox is None:
                continue
            x0, y0, x1, y1 = bbox
            faded = (90, 90, 90)
            cv2.rectangle(image, (x0, y0), (x1, y1), faded, 1)
            cv2.putText(
                image,
                (
                    f"reject:{component.get('reason', '')} "
                    f"area={component.get('area', 0)}"
                ),
                (x0, max(15, y0 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                faded,
                1,
            )

    def _draw_candidates(self, image: np.ndarray, debug: dict) -> None:
        for candidate in debug.get("detections", []):
            if candidate.get("detector_backend") == "yolo_hsv":
                self._draw_yolo_candidate(image, candidate)
                continue
            if self._require_housing_for_state and not candidate.get("has_housing", False):
                continue
            if candidate.get("spatial_accepted") is False:
                bbox = candidate.get("bbox")
                if bbox is not None:
                    x0, y0, x1, y1 = bbox
                    cv2.rectangle(image, (x0, y0), (x1, y1), (120, 120, 120), 1)
                    cv2.putText(
                        image,
                        f"reject:{candidate.get('spatial_reject_reason', '')}",
                        (x0, max(15, y0 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.35,
                        (120, 120, 120),
                        1,
                    )
                continue
            state = candidate.get("state", TrafficLightState.UNKNOWN)
            color = self._state_colors().get(state, (0, 0, 0))
            bbox = candidate.get("bbox")
            if bbox is None:
                continue
            x0, y0, x1, y1 = bbox
            circle = candidate.get("circle")
            if circle is not None:
                cx, cy, radius = circle
                cv2.circle(image, (cx, cy), radius, color, 1)
            cv2.rectangle(image, (x0, y0), (x1, y1), color, 1)
            cv2.putText(
                image,
                self._tl_label(candidate.get("confidence", 0.0), state),
                (x0, max(15, y0 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )
            world_xyz = candidate.get("world_xyz")
            if world_xyz is not None:
                wx, wy, wz = world_xyz
                cv2.putText(
                    image,
                    f"world X={wx:.1f} Y={wy:.1f} Z={wz:.1f}",
                    (x0, min(image.shape[0] - 5, y1 + 15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    color,
                    1,
                )

    def _colorize_mask(
        self,
        mask: np.ndarray,
        color: tuple[int, int, int],
        label: str,
    ) -> np.ndarray:
        output = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        output[mask > 0] = color
        return self._label_image(output, label)

    def _label_image(self, image: np.ndarray, label: str) -> np.ndarray:
        labeled = image.copy()
        cv2.putText(
            labeled,
            label,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        return labeled

    def _state_name(self, state: int) -> str:
        names = {
            TrafficLightState.UNKNOWN: "UNKNOWN",
            TrafficLightState.RED: "RED",
            TrafficLightState.YELLOW: "YELLOW",
            TrafficLightState.GREEN: "GREEN",
        }
        return names.get(state, "UNKNOWN")

    def _state_colors(self) -> dict[int, tuple[int, int, int]]:
        # UNKNOWN is black everywhere (matches yolo_car_detector.py's
        # combined debug image) so the same state always reads as the same
        # color across every debug topic - gray was too low-contrast against
        # the sim's mostly light-colored scenery to actually see.
        return {
            TrafficLightState.UNKNOWN: (0, 0, 0),
            TrafficLightState.RED: (0, 0, 255),
            TrafficLightState.YELLOW: (0, 255, 255),
            TrafficLightState.GREEN: (0, 255, 0),
        }

    _STATE_SHORT_NAME = {
        TrafficLightState.RED: "R",
        TrafficLightState.YELLOW: "Y",
        TrafficLightState.GREEN: "G",
        TrafficLightState.UNKNOWN: "UN",
    }

    @classmethod
    def _tl_label(cls, confidence: float, state: int | None) -> str:
        """
        Build a consistent label for every traffic-light debug box.

        topic: 'TL:0.87 R' when a color was classified (state given),
        'TL:0.42' for a box-only detection (state=None, no color attempted).
        """
        label = f"TL:{confidence:.2f}"
        if state is not None:
            label += f" {cls._STATE_SHORT_NAME.get(state, 'UN')}"
        return label

    def close_windows(self) -> None:
        if self._show_windows:
            cv2.destroyAllWindows()


def main(args: list[str] | None = None) -> None:
    """Run the traffic light detector node."""
    rclpy.init(args=args)
    node = TrafficLightDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close_windows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
