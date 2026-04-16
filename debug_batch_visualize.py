"""Batch bounding-box visualizer for debug subsets.

Runs a YOLO-World model on every image in a data split, saves annotated
images (with class label + confidence score on each box) to an output
directory. Intended for visual comparison of different model sizes.

Usage:
    python debug_batch_visualize.py \\
        --model-path ../models/yolov8m-worldv2.pt \\
        --data-dir tmp_data --split val \\
        --out-dir debug_subset50_visuals_yolo_medium
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import YOLO

COLORS = {1: "red", 2: "blue", 3: "orange", 4: "green"}
LABELS = {1: "person", 2: "vehicle", 3: "obstacle", 4: "safety_marker"}

PERSON_LABELS = {"person", "man", "woman", "boy", "girl", "adult", "child", "pedestrian"}
VEHICLE_LABELS = {"bicycle", "bike", "car", "motorcycle", "motorbike", "bus", "train",
                  "truck", "van", "forklift"}
OBSTACLE_LABELS = {"suitcase", "chair", "barrel", "crate", "box", "container", "handcart",
                   "ladder", "rock", "debris", "pallet"}
SAFETY_LABELS = {"cone", "traffic sign", "stop sign", "traffic light", "bollard",
                 "barrier", "warning sign"}


def _map_label(label: str) -> int | None:
    n = label.strip().lower().replace("-", " ").replace("_", " ")
    if n in PERSON_LABELS:
        return 1
    if n in VEHICLE_LABELS:
        return 2
    if n in OBSTACLE_LABELS:
        return 3
    if n in SAFETY_LABELS:
        return 4
    return None


def _resolve_image_path(data_dir: Path, split: str, file_name: str) -> Path:
    for candidate in [
        data_dir / file_name,
        data_dir / "images" / file_name,
        data_dir / split / file_name,
        data_dir / "images" / split / file_name,
    ]:
        if candidate.exists():
            return candidate
    return data_dir / "images" / file_name


def main() -> None:
    ap = argparse.ArgumentParser(description="Batch YOLO-World bbox visualizer.")
    ap.add_argument("--model-path", default="../models/yolov8m-worldv2.pt")
    ap.add_argument("--data-dir", default="tmp_data")
    ap.add_argument("--split", default="val")
    ap.add_argument("--out-dir", default="debug_subset50_visuals_yolo_medium")
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    coco = json.load(open(data_dir / "annotations" / f"{args.split}.json"))
    images = {img["id"]: img for img in coco["images"]}
    raw_cats = [c["name"] for c in coco.get("categories", [])]

    model = YOLO(args.model_path)
    model.set_classes(raw_cats)
    print(f"Model : {args.model_path}")
    print(f"Images: {len(images)}  ->  {out_dir}/")

    for i, (img_id, rec) in enumerate(sorted(images.items()), 1):
        img_path = _resolve_image_path(data_dir, args.split, rec["file_name"])

        results = model.predict(source=str(img_path), conf=args.conf, verbose=False)
        boxes = results[0].boxes if results else None
        label_names = results[0].names if results else {}

        dets = []
        if boxes is not None:
            for box in boxes:
                cls_id = int(box.cls.item())
                raw_lbl = str(label_names.get(cls_id, ""))
                cat_id = _map_label(raw_lbl)
                if cat_id is None:
                    continue
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
                dets.append((cat_id, x1, y1, x2 - x1, y2 - y1, float(box.conf.item())))

        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.imshow(Image.open(img_path))
        for cat_id, x, y, w, h, score in dets:
            color = COLORS[cat_id]
            rect = patches.Rectangle(
                (x, y), w, h, linewidth=2, edgecolor=color, facecolor="none"
            )
            ax.add_patch(rect)
            ax.text(
                x, y - 5,
                f"{LABELS[cat_id]} {score:.2f}",
                color="white", fontsize=8, fontweight="bold",
                bbox=dict(facecolor=color, alpha=0.7, edgecolor="none", pad=1),
            )
        ax.set_axis_off()
        ax.set_title(f"{rec['file_name']}  |  {len(dets)} det(s)", fontsize=9)
        plt.tight_layout()

        out_path = out_dir / f"{i:03d}_{img_id}.jpg"
        fig.savefig(str(out_path), bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"[{i:3d}/{len(images)}] {out_path.name}  —  {len(dets)} det(s)")

    print("Done.")


if __name__ == "__main__":
    main()
