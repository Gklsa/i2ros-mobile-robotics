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
Regenerate traffic-light YOLO labels from the semantic camera.

This replaces labels derived from the live RGB HSV detector's
``/perception/traffic_light_detections`` topic.

Empirically discovered (2026-07-22, by sampling semantic pixels under
already-known-good detections): this simulation's semantic camera renders
traffic-light/pole structures as a near-unique color, BGR ~= (4, 235, 255)
(bright yellow-gold), against a background of essentially just (0,0,0) and
(128,128,128) for everything else. Confirmed present in ~73% of a random
sample of captured frames vs. the RGB detector's ~1.5-3% labeling hit rate,
so this is a far more reliable *location* signal.

Important: the semantic image is 320x240 while rgb/depth are 640x480 (2x) -
coordinates must be scaled between the two. This was the actual bug behind
the collector node's on-the-fly semantic-color logging being useless.

Pipeline per frame:
  1. Threshold the semantic image for the traffic-light-class color.
  2. Connected-component the mask (in semantic-image space), drop tiny
     noise blobs.
  3. Scale each surviving bbox up 2x into rgb-image space (with a little
     padding, since the low-res semantic mask is a coarse silhouette).
  4. Emit that box as class 3 ("traffic_light", the housing/frame itself -
     lets the model learn to localize a light even when the lit bulb is too
     small/dim/occluded to classify a color for).
  5. Inside the same rgb crop, classify the bulb color via simple HSV
     thresholds (red/yellow/green) and, if confident, emit a second,
     tighter box for that specific class too. No confident color -> only
     the class-3 box is kept (semantic confirms "there is a light
     structure here", not which bulb is lit; we don't want to guess wrong
     on a training label).

Known limitation: the semantic (4,235,255) class is not exclusively
"traffic light" - nearby street signs/poles can render in the same class,
so both the class-3 boxes and any color boxes derived from them inherit
that false-positive rate. Review every generated label before training.

Usage:
  python3 relabel_from_semantic.py [--dataset-dir DIR] [--dry-run]

Overwrites dataset/labels/*.txt for every frame that has a semantic/*.png
counterpart. Run from anywhere; paths resolve relative to --dataset-dir
(default: ../../datasets/traffic_light from this script).
"""

from __future__ import annotations

import argparse
import os

import cv2
import numpy as np

SEMANTIC_TARGET_BGR = np.array([4, 235, 255])
SEMANTIC_TOLERANCE = 12
MIN_COMPONENT_PIXELS = 4  # in semantic-image (320x240) space
BBOX_PAD_RATIO = 0.35  # expand the upscaled bbox to cover the coarse mask's edges

MIN_COLOR_PIXELS = 2
CLASS_NAMES = ("red", "yellow", "green", "traffic_light")
TRAFFIC_LIGHT_CLASS_ID = 3


def detect_bulb_color_box(
    bgr_crop: np.ndarray,
) -> tuple[int, tuple[int, int, int, int]] | None:
    if bgr_crop.size == 0:
        return None
    hsv = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2HSV)

    red_low = cv2.inRange(hsv, (0, 70, 80), (22, 255, 255))
    red_high = cv2.inRange(hsv, (165, 80, 90), (179, 255, 255))
    red_mask = cv2.bitwise_or(red_low, red_high)
    yellow_mask = cv2.inRange(hsv, (23, 65, 90), (38, 255, 255))
    green_mask = cv2.inRange(hsv, (40, 70, 90), (90, 255, 255))

    candidates = [
        (0, red_mask),
        (1, yellow_mask),
        (2, green_mask),
    ]

    best: tuple[float, int, tuple[int, int, int, int]] | None = None
    for class_id, mask in candidates:
        box = best_color_component(mask)
        if box is None:
            continue
        x0, y0, x1, y1 = box
        area = float((x1 - x0) * (y1 - y0))
        score = float(np.count_nonzero(mask[y0:y1, x0:x1])) / max(1.0, area)
        score *= area
        if best is None or score > best[0]:
            best = (score, class_id, box)

    if best is None:
        return None
    return best[1], best[2]


def best_color_component(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    kernel = np.ones((2, 2), dtype=np.uint8)
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(
        cleaned,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    best_score = 0.0
    best_box: tuple[int, int, int, int] | None = None
    image_area = float(mask.shape[0] * mask.shape[1])

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < MIN_COLOR_PIXELS:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        if w <= 0 or h <= 0:
            continue

        box_area = float(w * h)
        if box_area > image_area * 0.45:
            continue

        aspect = w / h
        if aspect < 0.25 or aspect > 4.0:
            continue

        perimeter = float(cv2.arcLength(contour, True))
        circularity = 1.0
        if perimeter > 0.0:
            circularity = 4.0 * np.pi * area / (perimeter * perimeter)

        fill = float(np.count_nonzero(cleaned[y:y + h, x:x + w])) / box_area
        score = area * fill * max(0.25, min(1.0, circularity))
        if score <= best_score:
            continue

        pad = 2
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(mask.shape[1], x + w + pad)
        y1 = min(mask.shape[0], y + h + pad)
        best_score = score
        best_box = (x0, y0, x1, y1)

    return best_box


def find_boxes(
    semantic_bgr: np.ndarray,
    rgb_shape: tuple[int, int],
) -> list[tuple[int, int, int, int]]:
    sh, sw = semantic_bgr.shape[:2]
    rh, rw = rgb_shape
    scale_x, scale_y = rw / sw, rh / sh

    mask = np.all(
        np.abs(semantic_bgr.astype(int) - SEMANTIC_TARGET_BGR)
        <= SEMANTIC_TOLERANCE,
        axis=-1,
    )
    mask_u8 = mask.astype(np.uint8) * 255
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)

    boxes = []
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < MIN_COMPONENT_PIXELS:
            continue
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])

        rx0, ry0 = x * scale_x, y * scale_y
        rx1, ry1 = (x + w) * scale_x, (y + h) * scale_y
        pad_x = max(2.0, (rx1 - rx0) * BBOX_PAD_RATIO)
        pad_y = max(2.0, (ry1 - ry0) * BBOX_PAD_RATIO)
        rx0 = max(0, int(rx0 - pad_x))
        ry0 = max(0, int(ry0 - pad_y))
        rx1 = min(rw, int(rx1 + pad_x))
        ry1 = min(rh, int(ry1 + pad_y))
        if rx1 > rx0 and ry1 > ry0:
            boxes.append((rx0, ry0, rx1, ry1))
    return boxes


def process_dataset(dataset_dir: str, dry_run: bool) -> None:
    images_dir = os.path.join(dataset_dir, "images")
    semantic_dir = os.path.join(dataset_dir, "semantic")
    labels_dir = os.path.join(dataset_dir, "labels")

    basenames = sorted(
        f[:-4] for f in os.listdir(images_dir) if f.endswith(".jpg")
    )

    total_frames = 0
    labeled_frames = 0
    class_counts = [0, 0, 0, 0]

    for base in basenames:
        rgb_path = os.path.join(images_dir, f"{base}.jpg")
        sem_path = os.path.join(semantic_dir, f"{base}.png")
        if not os.path.exists(sem_path):
            continue
        rgb = cv2.imread(rgb_path)
        semantic = cv2.imread(sem_path)
        if rgb is None or semantic is None:
            continue
        total_frames += 1

        height, width = rgb.shape[:2]
        boxes = find_boxes(semantic, (height, width))

        lines = []
        for x0, y0, x1, y1 in boxes:
            cx = (x0 + x1) / 2.0 / width
            cy = (y0 + y1) / 2.0 / height
            w = (x1 - x0) / width
            h = (y1 - y0) / height

            lines.append(f"{TRAFFIC_LIGHT_CLASS_ID} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            class_counts[TRAFFIC_LIGHT_CLASS_ID] += 1

            bulb = detect_bulb_color_box(rgb[y0:y1, x0:x1])
            if bulb is None:
                continue
            class_id, (bx0, by0, bx1, by1) = bulb
            bulb_x0 = x0 + bx0
            bulb_y0 = y0 + by0
            bulb_x1 = x0 + bx1
            bulb_y1 = y0 + by1
            bulb_cx = (bulb_x0 + bulb_x1) / 2.0 / width
            bulb_cy = (bulb_y0 + bulb_y1) / 2.0 / height
            bulb_w = (bulb_x1 - bulb_x0) / width
            bulb_h = (bulb_y1 - bulb_y0) / height
            lines.append(
                f"{class_id} {bulb_cx:.6f} {bulb_cy:.6f} "
                f"{bulb_w:.6f} {bulb_h:.6f}"
            )
            class_counts[class_id] += 1

        if lines:
            labeled_frames += 1

        if not dry_run:
            with open(os.path.join(labels_dir, f"{base}.txt"), "w") as fh:
                fh.write("\n".join(lines))
                if lines:
                    fh.write("\n")

    print(f"processed {total_frames} frames")
    labeled_percent = 100.0 * labeled_frames / max(1, total_frames)
    print(f"labeled (>=1 box) {labeled_frames} frames ({labeled_percent:.1f}%)")
    for name, count in zip(CLASS_NAMES, class_counts):
        print(f"  {name}: {count} boxes")
    if dry_run:
        print("(dry run - label files NOT written)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        default=os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "datasets",
            "traffic_light",
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    process_dataset(os.path.abspath(args.dataset_dir), args.dry_run)
