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

"""Train a single-class YOLO detector for whole traffic-light boxes."""

from __future__ import annotations

import argparse
import os
import tempfile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default=os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "datasets",
            "traffic_light_box",
            "data.yaml",
        ),
    )
    parser.add_argument("--model", default="yolov8s.pt")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--name", default="trafficlight_box_yolov8s")
    parser.add_argument("--workers", type=int, default=0 if os.name == "nt" else 8)
    args = parser.parse_args()

    data_path = os.path.abspath(args.data)
    if not os.path.exists(data_path):
        raise SystemExit(
            f"data.yaml not found at {data_path}. "
            "Run build_box_detector_dataset.py and split_yolo_dataset.py first."
        )

    dataset_dir = os.path.dirname(data_path)
    for manifest in ("train.txt", "val.txt"):
        if not os.path.exists(os.path.join(dataset_dir, manifest)):
            raise SystemExit(f"{manifest} missing under {dataset_dir}")

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=_write_absolute_data_yaml(dataset_dir),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        project=os.path.join(os.path.dirname(__file__), "..", "output", "runs", "detect"),
        name=args.name,
        # Small distant objects benefit from larger image size. Keep hue
        # augmentation conservative; do not vertical-flip traffic scenes.
        mosaic=1.0,
        close_mosaic=15,
        hsv_h=0.005,
        hsv_s=0.45,
        hsv_v=0.35,
        degrees=0.0,
        shear=0.0,
        perspective=0.0,
        scale=0.35,
        translate=0.08,
        flipud=0.0,
        fliplr=0.0,
        mixup=0.0,
    )


def _write_absolute_data_yaml(dataset_dir: str) -> str:
    dataset_dir_posix = dataset_dir.replace("\\", "/")
    train_manifest = _write_absolute_manifest(dataset_dir, "train.txt")
    val_manifest = _write_absolute_manifest(dataset_dir, "val.txt")
    train_manifest_posix = train_manifest.replace("\\", "/")
    val_manifest_posix = val_manifest.replace("\\", "/")
    data = f"""path: "{dataset_dir_posix}"
train: "{train_manifest_posix}"
val: "{val_manifest_posix}"

names:
  0: traffic_light
"""
    path = os.path.join(tempfile.gettempdir(), "trafficlight_box_yolo_data.yaml")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(data)
    return path


def _write_absolute_manifest(dataset_dir: str, manifest_name: str) -> str:
    input_path = os.path.join(dataset_dir, manifest_name)
    output_path = os.path.join(tempfile.gettempdir(), f"trafficlight_box_yolo_{manifest_name}")
    lines: list[str] = []
    with open(input_path, "r", encoding="utf-8") as fh:
        for raw_line in fh:
            item = raw_line.strip()
            if not item:
                continue
            image_path = item if os.path.isabs(item) else os.path.join(dataset_dir, item)
            lines.append(os.path.abspath(image_path).replace("\\", "/"))

    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return output_path


if __name__ == "__main__":
    main()
