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

"""Extract deduplicated RGB frames from color-evaluation rosbag directories."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

import cv2
from cv_bridge import CvBridge
from rclpy.serialization import deserialize_message
import rosbag2_py
from sensor_msgs.msg import Image


RGB_TOPIC = "/OurCar/Sensors/RGBCameraLeft/image_raw"


def parse_spawn_indexes(raw: str) -> set[int]:
    return {int(value.strip()) for value in raw.split(",") if value.strip()}


def spawn_index_from_name(name: str) -> int | None:
    prefix = name.split("_", 1)[0]
    if not prefix.startswith("spawn"):
        return None
    try:
        return int(prefix.removeprefix("spawn"))
    except ValueError:
        return None


def rgb_messages(bag_dir: Path):
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag_dir), storage_id="mcap"),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )
    while reader.has_next():
        topic, data, receive_time_ns = reader.read_next()
        if topic == RGB_TOPIC:
            yield receive_time_ns, deserialize_message(data, Image)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract unique RGB training images from selected spawn bags."
    )
    parser.add_argument("--bags-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--spawn-indexes",
        default="0,4,5,8,9",
        help="Comma-separated Unity spawnIndex values to include.",
    )
    parser.add_argument("--jpg-quality", type=int, default=95)
    args = parser.parse_args()

    selected_spawns = parse_spawn_indexes(args.spawn_indexes)
    images_dir = args.output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.csv"

    bridge = CvBridge()
    seen_hashes: set[str] = set()
    rows: list[dict[str, object]] = []
    bags_used = 0
    duplicates = 0

    bag_dirs = sorted(path for path in args.bags_dir.iterdir() if path.is_dir())
    for bag_dir in bag_dirs:
        spawn_index = spawn_index_from_name(bag_dir.name)
        if spawn_index not in selected_spawns:
            continue
        if not (bag_dir / "metadata.yaml").is_file():
            print(f"skip unfinished bag without metadata: {bag_dir}")
            continue

        bags_used += 1
        bag_saved = 0
        for frame_index, (receive_time_ns, msg) in enumerate(rgb_messages(bag_dir)):
            raw_hash = hashlib.sha256(bytes(msg.data)).hexdigest()
            if raw_hash in seen_hashes:
                duplicates += 1
                continue
            seen_hashes.add(raw_hash)

            image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            filename = f"{bag_dir.name}_rgb_{frame_index:05d}.jpg"
            output_path = images_dir / filename
            if not cv2.imwrite(
                str(output_path),
                image,
                [cv2.IMWRITE_JPEG_QUALITY, args.jpg_quality],
            ):
                raise RuntimeError(f"failed to write {output_path}")

            rows.append(
                {
                    "image": f"images/{filename}",
                    "spawn_index": spawn_index,
                    "bag": bag_dir.name,
                    "frame_index": frame_index,
                    "receive_time_ns": receive_time_ns,
                    "width": msg.width,
                    "height": msg.height,
                    "encoding": msg.encoding,
                    "raw_sha256": raw_hash,
                }
            )
            bag_saved += 1
        print(f"{bag_dir.name}: saved {bag_saved} unique RGB frames")

    fieldnames = [
        "image",
        "spawn_index",
        "bag",
        "frame_index",
        "receive_time_ns",
        "width",
        "height",
        "encoding",
        "raw_sha256",
    ]
    with manifest_path.open("w", newline="", encoding="utf-8") as manifest:
        writer = csv.DictWriter(
            manifest,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"done: bags={bags_used}, images={len(rows)}, "
        f"exact_duplicates_skipped={duplicates}"
    )
    print(f"images: {images_dir}")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
