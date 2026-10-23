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
Regenerate YOLO labels for the car dataset from the semantic camera.

Empirically discovered (2026-07-22, by sampling semantic pixels under a
known car-following waypoint stretch and cross-checking against the RGB
frame): this simulation's semantic camera renders car bodies as pure red,
BGR == (0, 0, 255), clearly distinct from the traffic-light class's
(4, 235, 255) gold and from the (0,0,0)/(128,128,128) background. Confirmed
visually (wp062_00016: mask area grows 441 -> 4898 px across consecutive
frames as a car approaches in the RGB image, then drops back as it passes).

Same 320x240 (semantic) vs 640x480 (rgb/depth) scaling caveat as the
traffic-light relabel script - see relabel_from_semantic.py.

Pipeline per frame:
  1. Threshold the semantic image for the car-class color.
  2. Connected-component the mask (in semantic-image space), drop tiny
     noise blobs.
  3. Scale each surviving bbox up 2x into rgb-image space, with padding
     for the coarse low-res silhouette.
  4. Emit one box per component as class 0 ("car").

Known limitation: only sampled against the 39-63/101-105 car-following and
car-passing waypoint zones. Review all generated labels before training.

Usage:
  python3 relabel_car_from_semantic.py [--dataset-dir DIR] [--dry-run]

Overwrites dataset/labels/*.txt for every frame that has a semantic/*.png
counterpart. Run from anywhere; paths resolve relative to --dataset-dir
(default: ../../datasets/car from this script).
"""

from __future__ import annotations

import argparse
import os

import cv2
import numpy as np

SEMANTIC_TARGET_BGR = np.array([0, 0, 255])
SEMANTIC_TOLERANCE = 12
MIN_COMPONENT_PIXELS = 15  # in semantic-image (320x240) space
BBOX_PAD_RATIO = 0.15  # cars are large, solid blobs - less padding needed than lights

CAR_CLASS_ID = 0


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
    os.makedirs(labels_dir, exist_ok=True)

    basenames = sorted(
        f[:-4] for f in os.listdir(images_dir) if f.endswith(".jpg")
    )

    total_frames = 0
    labeled_frames = 0
    box_count = 0

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
            lines.append(f"{CAR_CLASS_ID} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            box_count += 1

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
    print(f"  car: {box_count} boxes")
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
            "car",
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    process_dataset(os.path.abspath(args.dataset_dir), args.dry_run)
