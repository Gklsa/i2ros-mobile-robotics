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
Fine-tune a YOLOv8 car detector on the collected car dataset.

Run from this folder:

  pip install -r requirements.txt
  python train_car_detector_yolov8n.py

The dataset is expected at src/ad_perception/datasets/car with YOLO-format
images, labels, train.txt, val.txt, and data.yaml. Results go under
src/ad_perception/trafficlight/output/runs/detect/car/.
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
            "datasets",
            "car",
            "data.yaml",
        ),
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="pretrained checkpoint to fine-tune from",
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--name", default="car")
    parser.add_argument(
        "--workers",
        type=int,
        default=0 if os.name == "nt" else 8,
        help="dataloader workers; 0 is more reliable on Windows",
    )
    args = parser.parse_args()

    data_path = os.path.abspath(args.data)
    if not os.path.exists(data_path):
        raise SystemExit(f"data.yaml not found at {data_path}")

    dataset_dir = os.path.dirname(data_path)
    for manifest in ("train.txt", "val.txt"):
        if not os.path.exists(os.path.join(dataset_dir, manifest)):
            raise SystemExit(
                f"dataset/{manifest} missing at {dataset_dir}. "
                "Create train/val splits before training."
            )

    output_dir = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "output",
            "runs",
            "detect",
        )
    )
    print(f"Using dataset: {dataset_dir}")
    print(f"Using model: {args.model}")
    print(f"Output dir: {output_dir}")
    print(
        f"Training config: epochs={args.epochs}, imgsz={args.imgsz}, "
        f"batch={args.batch}, workers={args.workers}"
    )

    from ultralytics import YOLO

    absolute_data_path = _write_absolute_data_yaml(dataset_dir)
    model = YOLO(args.model)
    model.train(
        data=absolute_data_path,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        project=output_dir,
        name=args.name,
        # Cars are larger than traffic lights and class is not defined by
        # hue, so these augmentations can be less conservative on geometry.
        mosaic=1.0,
        hsv_h=0.02,
        hsv_s=0.6,
        hsv_v=0.4,
        degrees=2.0,
        shear=0.0,
        perspective=0.0,
        scale=0.5,
        translate=0.15,
        flipud=0.0,
        fliplr=0.5,
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
  0: car
"""
    path = os.path.join(tempfile.gettempdir(), "car_yolo_data.yaml")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(data)
    return path


def _write_absolute_manifest(dataset_dir: str, manifest_name: str) -> str:
    input_path = os.path.join(dataset_dir, manifest_name)
    output_path = os.path.join(tempfile.gettempdir(), f"car_yolo_{manifest_name}")
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
