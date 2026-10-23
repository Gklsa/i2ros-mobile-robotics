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
Split dataset images into train and validation manifests for Ultralytics YOLO.

Doesn't move or copy any files - just writes dataset/train.txt and
dataset/val.txt, each listing image paths (relative to the dataset dir,
one per line), which is what data.yaml's train:/val: fields point at.
Re-run any time after collecting more frames; it reshuffles from scratch
with a fixed seed so the split stays reproducible.

Usage (from anywhere):
  python3 split_yolo_dataset.py [--dataset-dir DIR] [--val-fraction 0.15] [--seed 0]
"""

from __future__ import annotations

import argparse
import os
import random


def main() -> None:
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
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    dataset_dir = os.path.abspath(args.dataset_dir)
    images_dir = os.path.join(dataset_dir, "images")

    basenames = sorted(f for f in os.listdir(images_dir) if f.endswith(".jpg"))
    if not basenames:
        raise SystemExit(f"no .jpg files found under {images_dir}")

    random.Random(args.seed).shuffle(basenames)
    n_val = max(1, int(len(basenames) * args.val_fraction))
    val_set = set(basenames[:n_val])
    train_set = [b for b in basenames if b not in val_set]

    train_path = os.path.join(dataset_dir, "train.txt")
    val_path = os.path.join(dataset_dir, "val.txt")
    with open(train_path, "w") as fh:
        fh.write("\n".join(f"images/{b}" for b in train_set) + "\n")
    with open(val_path, "w") as fh:
        fh.write("\n".join(f"images/{b}" for b in sorted(val_set)) + "\n")

    print(f"{len(basenames)} total images")
    print(f"train: {len(train_set)} -> {train_path}")
    print(f"val:   {len(val_set)} -> {val_path}")


if __name__ == "__main__":
    main()
