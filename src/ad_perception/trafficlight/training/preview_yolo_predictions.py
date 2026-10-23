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

"""Run a YOLO model and draw compact prediction labels for inspection."""

from __future__ import annotations

import argparse
from pathlib import Path
import random

import cv2
from ultralytics import YOLO


COLORS = {
    0: (0, 0, 255),
    1: (0, 255, 255),
    2: (0, 255, 0),
    3: (255, 160, 0),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=Path(__file__).resolve().parent.parent
        / "output"
        / "runs"
        / "detect"
        / "trafficlight_jupyter"
        / "weights"
        / "best.pt",
        type=Path,
    )
    parser.add_argument(
        "--images-dir",
        default=Path(__file__).resolve().parents[2] / "datasets" / "traffic_light" / "images",
        type=Path,
    )
    parser.add_argument(
        "--output-dir",
        default=Path(__file__).resolve().parent.parent / "output" / "prediction_preview",
        type=Path,
    )
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--image", help="preview one image basename, e.g. wp019_x.jpg")
    parser.add_argument(
        "--no-labels",
        action="store_true",
        help="draw boxes only, without text labels",
    )
    parser.add_argument(
        "--show-all",
        action="store_true",
        help="draw every raw YOLO detection instead of only the selected signal",
    )
    parser.add_argument(
        "--save-crops",
        action="store_true",
        help="also save enlarged crops around the selected traffic light",
    )
    parser.add_argument(
        "--crop-scale",
        type=int,
        default=6,
        help="scale factor for --save-crops inspection images",
    )
    args = parser.parse_args()

    model_path = args.model.resolve()
    if not model_path.exists():
        raise SystemExit(f"model not found: {model_path}")

    images_dir = args.images_dir.resolve()
    if args.image:
        images = [images_dir / args.image]
    else:
        images = sorted(images_dir.glob("*.jpg"))
        random.Random(args.seed).shuffle(images)
        images = images[: args.count]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(model_path))
    names = model.names

    written = 0
    for image_path in images:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"skip unreadable image: {image_path}")
            continue

        result = model.predict(
            source=str(image_path),
            imgsz=args.imgsz,
            conf=args.conf,
            verbose=False,
        )[0]
        selected_light = None
        selected_color = None
        if args.show_all:
            draw_predictions(image, result, names, show_labels=not args.no_labels)
        else:
            selected_light, selected_color = draw_selected_signal(
                image,
                result,
                names,
                show_labels=not args.no_labels,
            )

        output_path = args.output_dir / image_path.name
        cv2.imwrite(str(output_path), image)
        if args.save_crops:
            save_selected_crop(
                image_path,
                image,
                selected_light,
                selected_color,
                args.output_dir / "crops",
                args.crop_scale,
            )
        written += 1

    print(f"wrote {written} preview images to {args.output_dir.resolve()}")


def draw_predictions(image, result, names: dict[int, str], show_labels: bool) -> None:
    if result.boxes is None:
        return

    for box in result.boxes:
        cls_id = int(box.cls[0])
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]

        color = COLORS.get(cls_id, (255, 255, 255))
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)

        if not show_labels:
            continue

        name = str(names.get(cls_id, cls_id))
        short_name = {
            "red": "R",
            "yellow": "Y",
            "green": "G",
            "traffic_light": "TL",
        }.get(name, name)
        label = f"{short_name}{confidence:.2f}"
        cv2.putText(
            image,
            label,
            (x1, max(10, y1 - 3)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.28,
            color,
            1,
            cv2.LINE_AA,
        )


def draw_selected_signal(
    image,
    result,
    names: dict[int, str],
    show_labels: bool,
) -> tuple[dict | None, dict | None]:
    height, width = image.shape[:2]
    boxes = parse_boxes(result)
    traffic_light_class_ids = {
        cls_id
        for cls_id, name in names.items()
        if str(name).lower() in {"traffic_light", "traffic light", "tl"}
    }
    if len(names) == 1 and not traffic_light_class_ids:
        traffic_light_class_ids = {0}

    color_class_ids = {
        cls_id
        for cls_id, name in names.items()
        if str(name).lower() in {"red", "yellow", "green"}
    }
    traffic_lights = [box for box in boxes if box["cls_id"] in traffic_light_class_ids]
    color_boxes = [box for box in boxes if box["cls_id"] in color_class_ids]

    selected_light = select_relevant_light(traffic_lights, color_boxes, width, height)
    selected_color = select_color_for_light(selected_light, color_boxes, width, height)

    if selected_light is not None:
        draw_box(
            image,
            selected_light,
            names,
            show_labels=show_labels,
            override_label="TL",
            thickness=2,
        )
    if selected_color is not None:
        draw_box(
            image,
            selected_color,
            names,
            show_labels=show_labels,
            thickness=2,
        )
    return selected_light, selected_color


def save_selected_crop(
    image_path: Path,
    annotated_image,
    selected_light: dict | None,
    selected_color: dict | None,
    output_dir: Path,
    scale: int,
) -> None:
    if selected_light is None and selected_color is None:
        return

    source_box = selected_light or selected_color
    assert source_box is not None
    x1, y1, x2, y2 = [int(value) for value in source_box["bbox"]]
    height, width = annotated_image.shape[:2]

    pad_x = max(12, int((x2 - x1) * 1.2))
    pad_y = max(12, int((y2 - y1) * 1.2))
    crop_x1 = max(0, x1 - pad_x)
    crop_y1 = max(0, y1 - pad_y)
    crop_x2 = min(width, x2 + pad_x)
    crop_y2 = min(height, y2 + pad_y)
    crop = annotated_image[crop_y1:crop_y2, crop_x1:crop_x2]
    if crop.size == 0:
        return

    enlarged = cv2.resize(
        crop,
        None,
        fx=max(1, scale),
        fy=max(1, scale),
        interpolation=cv2.INTER_CUBIC,
    )
    if selected_color is not None:
        cls_id = selected_color["cls_id"]
        confidence = selected_color["confidence"]
        label = {0: "R", 1: "Y", 2: "G"}.get(cls_id, str(cls_id))
    else:
        label = "TL"
        confidence = source_box["confidence"]
    cv2.putText(
        enlarged,
        f"{label}{confidence:.2f}",
        (6, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_dir / image_path.name), enlarged)


def parse_boxes(result) -> list[dict]:
    if result.boxes is None:
        return []

    boxes = []
    for box in result.boxes:
        cls_id = int(box.cls[0])
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
        boxes.append(
            {
                "cls_id": cls_id,
                "confidence": confidence,
                "bbox": (x1, y1, x2, y2),
                "center": ((x1 + x2) / 2.0, (y1 + y2) / 2.0),
                "area": max(0.0, x2 - x1) * max(0.0, y2 - y1),
            }
        )
    return boxes


def select_relevant_light(
    traffic_lights: list[dict],
    color_boxes: list[dict],
    image_width: int,
    image_height: int,
) -> dict | None:
    if traffic_lights:
        return max(
            traffic_lights,
            key=lambda box: relevant_light_score(
                box,
                color_boxes,
                image_width,
                image_height,
            ),
        )

    if not color_boxes:
        return None
    return max(
        color_boxes,
        key=lambda box: centrality_score(box, image_width, image_height)
        * box["confidence"],
    )


def relevant_light_score(
    light: dict,
    color_boxes: list[dict],
    image_width: int,
    image_height: int,
) -> float:
    center_score = centrality_score(light, image_width, image_height)
    color_score = 0.0
    for color_box in color_boxes:
        if color_belongs_to_light(color_box, light):
            color_score = max(color_score, color_box["confidence"])

    # Prefer the center/forward-facing signal. A visible red/yellow/green bulb
    # is strong evidence that this is not the back side of a signal.
    size_score = min(1.0, light["area"] / float(image_width * image_height) * 50.0)
    return 1.5 * center_score + 1.2 * color_score + 0.3 * size_score


def centrality_score(box: dict, image_width: int, image_height: int) -> float:
    cx, cy = box["center"]
    dx = abs(cx - image_width / 2.0) / max(1.0, image_width / 2.0)
    dy = abs(cy - image_height * 0.42) / max(1.0, image_height * 0.58)
    return max(0.0, 1.0 - 0.75 * dx - 0.25 * dy)


def select_color_for_light(
    selected_light: dict | None,
    color_boxes: list[dict],
    image_width: int,
    image_height: int,
) -> dict | None:
    if not color_boxes:
        return None

    if selected_light is None:
        return max(
            color_boxes,
            key=lambda box: box["confidence"] * centrality_score(
                box,
                image_width,
                image_height,
            ),
        )

    belonging = [
        color_box
        for color_box in color_boxes
        if color_belongs_to_light(color_box, selected_light)
    ]
    if not belonging:
        return None

    # Enforce one color per physical traffic light.
    return max(belonging, key=lambda box: box["confidence"])


def color_belongs_to_light(color_box: dict, light_box: dict) -> bool:
    cx, cy = color_box["center"]
    x1, y1, x2, y2 = light_box["bbox"]
    if x1 <= cx <= x2 and y1 <= cy <= y2:
        return True
    return bbox_iou(color_box["bbox"], light_box["bbox"]) > 0.02


def bbox_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def draw_box(
    image,
    box: dict,
    names: dict[int, str],
    show_labels: bool,
    override_label: str | None = None,
    thickness: int = 1,
) -> None:
    cls_id = int(box["cls_id"])
    confidence = float(box["confidence"])
    x1, y1, x2, y2 = [int(value) for value in box["bbox"]]
    color = COLORS.get(cls_id, (255, 255, 255))
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    if not show_labels:
        return

    name = str(names.get(cls_id, cls_id))
    short_name = override_label or {
        "red": "R",
        "yellow": "Y",
        "green": "G",
        "traffic_light": "TL",
    }.get(name, name)
    label = f"{short_name}{confidence:.2f}"
    cv2.putText(
        image,
        label,
        (x1, max(10, y1 - 3)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.28,
        color,
        1,
        cv2.LINE_AA,
    )


if __name__ == "__main__":
    main()
