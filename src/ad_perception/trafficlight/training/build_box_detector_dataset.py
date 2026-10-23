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
Build a cleaner single-class traffic-light dataset for runtime YOLO.

The runtime detector only needs YOLO to find the whole traffic-light housing.
Bulb color is classified later with HSV inside that box. This script converts
the existing 4-class labels into a single class:

  class 3 traffic_light -> class 0 traffic_light

It also removes sky-heavy empty frames from the output dataset. Negative frames
are still useful, but sky-only negatives dominate training and do not represent
the false positives we care about.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import cv2
import numpy as np


SOURCE_TRAFFIC_LIGHT_CLASS = "3"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "datasets" / "traffic_light",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "datasets" / "traffic_light_box",
    )
    parser.add_argument(
        "--sky-empty-threshold",
        type=float,
        default=0.35,
        help="drop empty-label frames whose upper-image sky ratio is above this value",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    image_dir = source_dir / "images"
    label_dir = source_dir / "labels"

    if not image_dir.exists() or not label_dir.exists():
        raise SystemExit(f"missing images/labels under {source_dir}")

    if not args.dry_run:
        (output_dir / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / "labels").mkdir(parents=True, exist_ok=True)

    kept_positive = 0
    kept_negative = 0
    dropped_sky_empty = 0
    dropped_missing_label = 0

    for image_path in sorted(image_dir.glob("*.jpg")):
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            dropped_missing_label += 1
            continue

        converted_lines = convert_label(label_path)
        if not converted_lines and sky_ratio(image_path) >= args.sky_empty_threshold:
            dropped_sky_empty += 1
            continue

        if converted_lines:
            kept_positive += 1
        else:
            kept_negative += 1

        if args.dry_run:
            continue

        shutil.copy2(image_path, output_dir / "images" / image_path.name)
        output_label_path = output_dir / "labels" / label_path.name
        output_label_path.write_text("".join(converted_lines), encoding="utf-8")

    if not args.dry_run:
        (output_dir / "classes.txt").write_text("traffic_light\n", encoding="utf-8")
        (output_dir / "data.yaml").write_text(
            "path: .\n"
            "train: train.txt\n"
            "val: val.txt\n"
            "names:\n"
            "  0: traffic_light\n",
            encoding="utf-8",
        )

    total_kept = kept_positive + kept_negative
    print(f"source: {source_dir}")
    print(f"output: {output_dir}")
    print(f"kept: {total_kept} frames")
    print(f"  positive traffic_light: {kept_positive}")
    print(f"  negative: {kept_negative}")
    print(f"dropped sky-heavy empty frames: {dropped_sky_empty}")
    print(f"dropped missing-label frames: {dropped_missing_label}")
    if args.dry_run:
        print("dry run - no files written")


def convert_label(label_path: Path) -> list[str]:
    lines: list[str] = []
    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        parts = raw_line.strip().split()
        if len(parts) != 5 or parts[0] != SOURCE_TRAFFIC_LIGHT_CLASS:
            continue
        lines.append("0 " + " ".join(parts[1:]) + "\n")
    return lines


def sky_ratio(image_path: Path) -> float:
    image = cv2.imread(str(image_path))
    if image is None:
        return 0.0
    height = image.shape[0]
    upper = image[: int(height * 0.75)]
    hsv = cv2.cvtColor(upper, cv2.COLOR_BGR2HSV)
    sky_mask = (
        (hsv[:, :, 0] >= 85)
        & (hsv[:, :, 0] <= 125)
        & (hsv[:, :, 1] >= 25)
        & (hsv[:, :, 1] <= 170)
        & (hsv[:, :, 2] >= 120)
    )
    return float(np.count_nonzero(sky_mask)) / float(max(1, sky_mask.size))


if __name__ == "__main__":
    main()
