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
Train an experimental four-class detector for bulb-color boxes.

This is not the main runtime model. It trains red/yellow/green bulb boxes plus
the whole traffic_light box, and performed much worse than the single-class
box detector. Keep it only as an experiment record.

Run from the parent training directory:

  python split_yolo_dataset.py --dataset-dir ../../datasets/traffic_light
  python color_experiment/train_bulb_color_detector_experiment.py
"""

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
            "..",
            "datasets",
            "traffic_light",
            "data.yaml",
        ),
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="pretrained checkpoint to fine-tune from",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--name", default="trafficlight_color_experiment")
    parser.add_argument(
        "--workers",
        type=int,
        default=0 if os.name == "nt" else 8,
        help="dataloader workers; 0 is more reliable on Windows",
    )
    args = parser.parse_args()

    from ultralytics import YOLO

    data_path = os.path.abspath(args.data)
    if not os.path.exists(data_path):
        raise SystemExit(f"data.yaml not found at {data_path}")

    dataset_dir = os.path.dirname(data_path)
    if not os.path.exists(os.path.join(dataset_dir, "train.txt")):
        raise SystemExit(
            "dataset/train.txt missing - run split_yolo_dataset.py first "
            "from the parent training/ folder."
        )

    absolute_data_path = _write_absolute_data_yaml(dataset_dir)
    model = YOLO(args.model)
    model.train(
        data=absolute_data_path,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        project=os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "output",
            "runs",
            "detect",
        ),
        name=args.name,
        # Augmentation tuned for this dataset - see train.ipynb's markdown
        # for the per-parameter reasoning (tiny distant lights, hue is the
        # class signal so kept low, redundant near-duplicate frames so
        # mosaic is kept at full strength).
        mosaic=1.0,
        hsv_h=0.01,
        hsv_s=0.8,
        hsv_v=0.5,
        degrees=0.0,
        shear=0.0,
        perspective=0.0,
        scale=0.3,
        translate=0.1,
        flipud=0.0,
        fliplr=0.5,
        mixup=0.0,
    )


def _write_absolute_data_yaml(dataset_dir: str) -> str:
    """Write a data.yaml whose dataset path is unambiguous on Windows."""
    dataset_dir_posix = dataset_dir.replace("\\", "/")
    train_manifest = _write_absolute_manifest(dataset_dir, "train.txt")
    val_manifest = _write_absolute_manifest(dataset_dir, "val.txt")
    train_manifest_posix = train_manifest.replace("\\", "/")
    val_manifest_posix = val_manifest.replace("\\", "/")
    data = f"""path: "{dataset_dir_posix}"
train: "{train_manifest_posix}"
val: "{val_manifest_posix}"

names:
  0: red
  1: yellow
  2: green
  3: traffic_light
"""
    path = os.path.join(tempfile.gettempdir(), "trafficlight_yolo_data.yaml")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(data)
    return path


def _write_absolute_manifest(dataset_dir: str, manifest_name: str) -> str:
    """Convert images/foo.jpg entries into absolute image paths for Ultralytics."""
    input_path = os.path.join(dataset_dir, manifest_name)
    output_path = os.path.join(
        tempfile.gettempdir(),
        f"trafficlight_yolo_{manifest_name}",
    )
    lines: list[str] = []
    with open(input_path, "r", encoding="utf-8") as fh:
        for raw_line in fh:
            item = raw_line.strip()
            if not item:
                continue
            if os.path.isabs(item):
                image_path = item
            else:
                image_path = os.path.join(dataset_dir, item)
            lines.append(os.path.abspath(image_path).replace("\\", "/"))

    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return output_path


if __name__ == "__main__":
    main()
