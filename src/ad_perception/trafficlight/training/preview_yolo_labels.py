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

"""Draw YOLO label boxes on sample traffic-light dataset images."""

from __future__ import annotations

import argparse
from pathlib import Path
import random

import cv2


COLORS = {
    0: (0, 0, 255),
    1: (0, 255, 255),
    2: (0, 255, 0),
    3: (255, 160, 0),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        default=Path(__file__).resolve().parents[2] / "datasets" / "traffic_light",
        type=Path,
    )
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image", help="preview one image basename, e.g. wp019_x.jpg")
    parser.add_argument(
        "--output-dir",
        default=Path(__file__).resolve().parent.parent / "output" / "label_preview",
        type=Path,
    )
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.resolve()
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"
    classes = read_classes(dataset_dir / "classes.txt")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.image:
        images = [images_dir / args.image]
    else:
        images = sorted(images_dir.glob("*.jpg"))
        random.Random(args.seed).shuffle(images)
        images = images[: args.count]

    written = 0
    for image_path in images:
        label_path = labels_dir / f"{image_path.stem}.txt"
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"skip unreadable image: {image_path}")
            continue

        draw_labels(image, label_path, classes)
        output_path = args.output_dir / image_path.name
        cv2.imwrite(str(output_path), image)
        written += 1

    print(f"wrote {written} preview images to {args.output_dir.resolve()}")


def read_classes(path: Path) -> dict[int, str]:
    if not path.exists():
        return {0: "red", 1: "yellow", 2: "green", 3: "traffic_light"}

    classes: dict[int, str] = {}
    for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        name = line.strip()
        if name:
            classes[idx] = name
    return classes


def draw_labels(image, label_path: Path, classes: dict[int, str]) -> None:
    height, width = image.shape[:2]
    if not label_path.exists():
        cv2.putText(
            image,
            "no label file",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return

    lines = [line.strip() for line in label_path.read_text().splitlines()]
    lines = [line for line in lines if line]
    if not lines:
        cv2.putText(
            image,
            "empty label",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return

    for line in lines:
        parts = line.split()
        if len(parts) != 5:
            continue

        cls_id = int(float(parts[0]))
        cx, cy, box_w, box_h = (float(value) for value in parts[1:])
        x1 = int((cx - box_w / 2.0) * width)
        y1 = int((cy - box_h / 2.0) * height)
        x2 = int((cx + box_w / 2.0) * width)
        y2 = int((cy + box_h / 2.0) * height)

        color = COLORS.get(cls_id, (255, 255, 255))
        label = classes.get(cls_id, str(cls_id))
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            image,
            label,
            (x1, max(16, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )


if __name__ == "__main__":
    main()
