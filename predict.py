"""THEKER Hackathon -- Prediction Template.

This script loads the dataset, runs a decision function on each image,
and writes a submission-ready JSON file.  A simple baseline is provided
so participants can submit immediately, then iterate.

Key design points:
    - Train/val have annotations; test has NONE (you must detect).
    - Labels are raw and heterogeneous (27 categories).  Preprocessing
      is part of the challenge.  The baseline groups them for you as a
      starting point.

Usage::

    # Run on val (has annotations -- for development)
    python predict.py --data-dir ../data --split val --output predictions.json

    # Run on test (no annotations -- for submission)
    python predict.py --data-dir ../data --split test --output submission.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

from utils import (
    bbox_area,
    bbox_height_ratio,
    build_annotation_index,
    build_image_index,
    load_annotations,
    visualize_scene,
)

# ---------------------------------------------------------------------------
# Label preprocessing -- GROUP THE 27 RAW LABELS
# ---------------------------------------------------------------------------
# The dataset has 27 messy labels from multiple sources. This baseline
# groups them into 4 decision-relevant categories. You should refine this
# mapping or build your own.

PERSON_LABELS = {"person", "hat", "helmet", "head"}
VEHICLE_LABELS = {"bicycle", "car", "motorcycle", "bus", "train", "truck", "forklift"}
OBSTACLE_LABELS = {
    "suitcase", "chair", "barrel", "crate", "box", "handcart", "ladder",
    "Box", "Barrel", "Container", "Ladder", "Suitcase",
}
SAFETY_LABELS = {"cone", "Traffic sign", "Stop sign", "Traffic light"}

DETECTION_CATEGORIES = {
    1: "person",
    2: "vehicle",
    3: "obstacle",
    4: "safety_marker",
}

DETECTOR_PERSON_LABELS = {
    "person",
    "man",
    "woman",
    "boy",
    "girl",
    "adult",
    "child",
    "pedestrian",
}
DETECTOR_VEHICLE_LABELS = {
    "bicycle",
    "bike",
    "car",
    "motorcycle",
    "motorbike",
    "bus",
    "train",
    "truck",
    "van",
    "forklift",
}
DETECTOR_OBSTACLE_LABELS = {
    "suitcase",
    "chair",
    "barrel",
    "crate",
    "box",
    "container",
    "handcart",
    "ladder",
    "rock",
    "debris",
    "pallet",
}
DETECTOR_SAFETY_LABELS = {
    "cone",
    "traffic sign",
    "stop sign",
    "traffic light",
    "bollard",
    "barrier",
    "warning sign",
}


def _build_detector_class_prompts(raw_category_names: List[str]) -> List[str]:
    """Return detector prompts directly from raw annotation categories."""
    prompts: List[str] = []
    for name in raw_category_names:
        normalized = name.strip()
        if not normalized:
            continue
        prompts.append(normalized)
    return prompts


def _normalize_label(label: str) -> str:
    return label.strip().lower().replace("-", " ").replace("_", " ")


def _map_detector_label_to_category_id(label: str) -> int | None:
    normalized = _normalize_label(label)
    if normalized in DETECTOR_PERSON_LABELS:
        return 1
    if normalized in DETECTOR_VEHICLE_LABELS:
        return 2
    if normalized in DETECTOR_OBSTACLE_LABELS:
        return 3
    if normalized in DETECTOR_SAFETY_LABELS:
        return 4
    return None


def _resolve_image_path(data_dir: Path, split: str, file_name: str) -> Path:
    """Resolve image path for dataset layouts used in THEKER starter kit."""
    candidates = [
        data_dir / file_name,
        data_dir / "images" / file_name,
        data_dir / split / file_name,
        data_dir / "images" / split / file_name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _build_detector(model_path: Path, class_prompts: List[str]) -> Any:
    """Load YOLO-World detector once for all images."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        print(
            "ERROR: Ultralytics is required for detector inference. "
            "Install dependencies from requirements.txt.",
        )
        raise SystemExit(1) from exc

    if not model_path.exists():
        print(f"ERROR: Detector model not found: {model_path}")
        raise SystemExit(1)

    detector = YOLO(str(model_path))
    if class_prompts:
        # Prompt YOLO-World to restrict open-vocabulary detection candidates.
        detector.set_classes(class_prompts)
    return detector


def _run_detector_on_image(
    detector: Any,
    image_path: Path,
    image_id: int,
    image_w: int,
    image_h: int,
    conf: float,
) -> List[Dict[str, Any]]:
    """Run detector and convert results to COCO-like detections."""
    try:
        results = detector.predict(source=str(image_path), conf=conf, verbose=False)
    except Exception as exc:
        print(f"WARNING: Detector failed for {image_path}: {exc}")
        return []
    if not results:
        return []

    result = results[0]
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []

    label_names = getattr(result, "names", {})
    detections: List[Dict[str, Any]] = []

    for box in boxes:
        cls_id = int(box.cls.item())
        label = str(label_names.get(cls_id, ""))
        category_id = _map_detector_label_to_category_id(label)
        if category_id is None:
            continue

        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
        x1 = max(0.0, min(x1, float(image_w)))
        y1 = max(0.0, min(y1, float(image_h)))
        x2 = max(0.0, min(x2, float(image_w)))
        y2 = max(0.0, min(y2, float(image_h)))
        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)
        if w <= 0.0 or h <= 0.0:
            continue

        detections.append(
            {
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [x1, y1, w, h],
                "score": float(box.conf.item()),
            }
        )

    return detections


def classify_annotation(ann: dict, categories: dict[int, str]) -> str | None:
    """Map a raw annotation to a decision group.

    Args:
        ann: Annotation dict with ``category_id``.
        categories: Mapping of category_id -> category name.

    Returns:
        One of ``"person"``, ``"vehicle"``, ``"obstacle"``,
        ``"safety_marker"``, or ``None`` if unrecognized.
    """
    name = categories.get(ann["category_id"], "")
    if name in PERSON_LABELS:
        return "person"
    if name in VEHICLE_LABELS:
        return "vehicle"
    if name in OBSTACLE_LABELS:
        return "obstacle"
    if name in SAFETY_LABELS:
        return "safety_marker"
    return None


# ---------------------------------------------------------------------------
# Claude decision helpers
# ---------------------------------------------------------------------------

def _build_scene_description(
    annotations: List[Dict[str, Any]],
    categories: Dict[int, str],
    image_w: int,
    image_h: int,
) -> str:
    """Build a concise scene description from detections for Claude."""
    if not annotations:
        return "No objects detected in scene."

    img_area = image_w * image_h
    counts: Dict[str, int] = {}
    parts = []
    for ann in annotations:
        label = categories.get(ann["category_id"], "unknown")
        counts[label] = counts.get(label, 0) + 1
        bbox = ann["bbox"]
        cx = bbox[0] + bbox[2] / 2
        area_pct = bbox_area(bbox) / img_area * 100 if img_area > 0 else 0.0
        if cx < image_w * 0.33:
            pos = "left"
        elif cx < image_w * 0.67:
            pos = "center"
        else:
            pos = "right"
        parts.append(f"{label} ({pos}, {area_pct:.1f}% area)")

    summary = ", ".join(f"{v} {k}" for k, v in sorted(counts.items()))
    return f"Detected: {', '.join(parts)}. Total: {summary}."


def _call_claude_api_decision(
    scene_description: str,
    client: Any,
    timeout_seconds: int,
) -> Tuple[str, float, str]:
    """Call Claude API for a navigation decision. Returns (action, confidence, reasoning)."""
    system_prompt = (
        "You are a Safety Controller for an industrial autonomous robot.\n"
        "Before choosing an action, reason through these steps:\n"
        "1. List each detected object with its position, area%, and confidence.\n"
        "2. Classify each as 'Immediate Path' (center position) or "
        "'Background' (left/right, OR area <2%, OR confidence <0.4).\n"
        "3. Apply rules:\n"
        "   STOP: only if an Immediate Path object has area >12% AND confidence >=0.4.\n"
        "   SLOW: any Immediate Path object is small/low-confidence, "
        "OR Background objects are present near the path.\n"
        "   CONTINUE: all objects are Background, or none detected.\n"
        "Background objects and low-confidence detections should bias toward SLOW, not STOP.\n"
        "After your reasoning, output a single JSON object on the last line:\n"
        '{"action": "STOP|SLOW|CONTINUE", "confidence": 0.0-1.0, '
        '"reasoning": "15-30 words naming objects, positions, and decisive factor"}'
    )
    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=400,
            system=system_prompt,
            messages=[{"role": "user", "content": scene_description}],
            timeout=float(timeout_seconds),
        )
        raw = response.content[0].text.strip()
    except Exception:
        return "SLOW", 0.55, "Safety default: API connection error."
    try:
        import re as _re
        # Chain-of-thought response: find the last {...} block.
        matches = _re.findall(r"\{[^{}]+\}", raw, _re.DOTALL)
        if not matches:
            raise ValueError("No JSON object found in response")
        parsed = json.loads(matches[-1])
        action = parsed["action"]
        if action not in ("STOP", "SLOW", "CONTINUE"):
            raise ValueError(f"Invalid action: {action}")
        confidence = max(0.0, min(1.0, float(parsed["confidence"])))
        reasoning = str(parsed["reasoning"])
        return action, confidence, reasoning
    except Exception:
        return "SLOW", 0.55, "Safety default: API response malformed."


# ---------------------------------------------------------------------------
# Decision function -- REPLACE THIS WITH YOUR OWN LOGIC
# ---------------------------------------------------------------------------

def make_decision(
    image_record: Dict[str, Any],
    annotations: List[Dict[str, Any]],
    categories: Dict[int, str],
    claude_config: Dict[str, Any] | None = None,
    claude_state: Dict[str, Any] | None = None,
) -> Tuple[str, float, str]:
    """Decide whether the vehicle should STOP, SLOW, or CONTINUE.

    This baseline preprocesses the 27 raw labels into 4 groups, then
    applies simple spatial rules.  Participants should replace or extend
    this with learned models, VLMs, and richer reasoning.

    Args:
        image_record: COCO image dict (``id``, ``width``, ``height``, ``file_name``).
        annotations: Annotation dicts for this image (may be empty for test).
        categories: Mapping of category_id -> category name from the annotations file.

    Returns:
        Tuple of (action, confidence, reasoning).
    """
    img_h = image_record["height"]
    img_w = image_record["width"]
    img_area = img_h * img_w

    # Group annotations
    persons, vehicles, obstacles, markers = [], [], [], []
    for ann in annotations:
        group = classify_annotation(ann, categories)
        if group == "person":
            persons.append(ann)
        elif group == "vehicle":
            vehicles.append(ann)
        elif group == "obstacle":
            obstacles.append(ann)
        elif group == "safety_marker":
            markers.append(ann)

    # -----------------------------------------------------------------------
    # --- HYBRID LOGIC: hard safety switch + optional Claude API ---
    # -----------------------------------------------------------------------

    if claude_config and claude_config["enabled"]:
        # Hard safety switch: person too close -> immediate STOP, no API call.
        for ann in persons:
            if bbox_height_ratio(ann["bbox"], img_h) > 0.35:
                return (
                    "STOP", 1.0,
                    "Emergency stop: Person detected in immediate path.",
                )
        # Soft path: ask Claude.
        if claude_state["calls_made"] < claude_config["max_calls"]:
            scene_desc = _build_scene_description(
                annotations, categories, img_w, img_h,
            )
            scene_desc = scene_desc[: claude_config["max_input_chars"]]
            action, confidence, reasoning = _call_claude_api_decision(
                scene_desc, claude_config["client"], claude_config["timeout_seconds"],
            )
            claude_state["calls_made"] += 1
            return action, confidence, reasoning
        else:
            return "SLOW", 0.55, "Safety default: Claude call budget exhausted."

    # -----------------------------------------------------------------------
    # --- YOUR LOGIC HERE ---
    # The rules below are a minimal baseline.  Replace or extend them.
    # -----------------------------------------------------------------------

    # Rule 1: Large person (close range) -> STOP
    for ann in persons:
        if bbox_height_ratio(ann["bbox"], img_h) > 0.25:
            return (
                "STOP", 0.90,
                "Person detected at close range (height ratio > 0.25).",
            )

    # Rule 2: Large vehicle -> STOP
    for ann in vehicles:
        if bbox_area(ann["bbox"]) > 0.15 * img_area:
            return (
                "STOP", 0.85,
                "Vehicle occupying >15% of image area.",
            )

    # Rule 3: Any person or vehicle present -> SLOW
    if persons or vehicles:
        return (
            "SLOW", 0.70,
            f"Detected {len(persons)} person(s) and {len(vehicles)} vehicle(s).",
        )

    # Rule 4: Safety marker present -> SLOW
    if markers:
        return (
            "SLOW", 0.60,
            f"Safety marker(s) detected ({len(markers)}).",
        )

    # Rule 5: Obstacles covering >40% of image width -> SLOW
    if obstacles:
        x_ranges = sorted((ann["bbox"][0], ann["bbox"][0] + ann["bbox"][2]) for ann in obstacles)
        merged = [x_ranges[0]]
        for start, end in x_ranges[1:]:
            if start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        total_width = sum(e - s for s, e in merged)
        if total_width > 0.40 * img_w:
            return (
                "SLOW", 0.65,
                f"Obstacles cover {total_width / img_w:.0%} of image width.",
            )

    # Rule 6: Nothing significant -> CONTINUE
    return ("CONTINUE", 0.80, "No significant hazards detected.")

    # -----------------------------------------------------------------------
    # --- END OF YOUR LOGIC ---
    # -----------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_predictions(
    data_dir: Path,
    output_path: Path,
    team_name: str,
    split: str = "test",
    model_path: Path = Path("../models/yolov8s-worldv2.pt"),
    detector_conf: float = 0.25,
    use_detector_on_val: bool = False,
    enable_claude: bool = False,
    claude_max_calls: int = 50,
    claude_max_input_chars: int = 600,
    claude_timeout_seconds: int = 20,
    max_images: int | None = None,
) -> None:
    """Load the dataset, predict on every image, and write submission JSON.

    For train/val splits, annotations are available and used by the baseline.
    For the test split, annotations are empty -- the baseline will predict
    CONTINUE for every image (since it has no detections).  Participants
    should add their own detection pipeline for the test split.

    Args:
        data_dir: Root data directory.
        output_path: Where to write the submission JSON.
        team_name: Team name for the submission.
        split: Dataset split to run on.
    """
    ann_path = data_dir / "annotations" / f"{split}.json"
    if not ann_path.exists():
        print(f"ERROR: Annotation file not found: {ann_path}")
        sys.exit(1)

    print(f"Loading annotations from {ann_path} ...")
    coco_data = load_annotations(ann_path)

    # Build category lookup from the file itself
    categories = {c["id"]: c["name"] for c in coco_data.get("categories", [])}
    print(f"Categories: {len(categories)} raw labels")

    image_index = build_image_index(coco_data)
    annotation_index = build_annotation_index(coco_data)

    has_annotations = len(coco_data.get("annotations", [])) > 0
    if not has_annotations:
        print(f"WARNING: {split} split has no annotations. "
              "The baseline will predict CONTINUE for all images. "
              "Add your own detection pipeline for meaningful predictions.")

    print(f"Found {len(image_index)} images in the {split} split.")

    use_detector = (not has_annotations) or (split == "val" and use_detector_on_val)
    detector = None
    if use_detector:
        raw_category_names = [str(c["name"]) for c in coco_data.get("categories", [])]
        class_prompts = _build_detector_class_prompts(raw_category_names)
        detector = _build_detector(model_path, class_prompts)
        print(f"Loaded detector from {model_path}")
        print(f"Detector class prompts: {len(class_prompts)}")
        print(f"Detector confidence threshold: {detector_conf}")

    claude_config: Dict[str, Any] | None = None
    claude_state: Dict[str, Any] | None = None
    if enable_claude:
        try:
            import anthropic as _anthropic
        except ImportError as exc:
            print(
                "ERROR: anthropic package is required for --enable-claude-decision. "
                "Run: pip install anthropic"
            )
            raise SystemExit(1) from exc
        claude_config = {
            "enabled": True,
            "client": _anthropic.Anthropic(),
            "max_calls": claude_max_calls,
            "max_input_chars": claude_max_input_chars,
            "timeout_seconds": claude_timeout_seconds,
        }
        claude_state = {"calls_made": 0}
        print(
            f"Claude decision enabled: max_calls={claude_max_calls}, "
            f"max_input_chars={claude_max_input_chars}, "
            f"timeout={claude_timeout_seconds}s"
        )

    predictions: List[Dict[str, Any]] = []
    all_detections: List[Dict[str, Any]] = []

    sorted_images = sorted(image_index.items())
    if max_images is not None:
        sorted_images = sorted_images[:max_images]
        print(f"Limiting run to {len(sorted_images)} images (--max-images {max_images}).")

    for image_id, image_record in sorted_images:
        if detector is not None:
            image_path = _resolve_image_path(data_dir, split, image_record["file_name"])
            if not image_path.exists():
                print(f"ERROR: Image file not found for image_id={image_id}: {image_path}")
                raise SystemExit(1)
            anns = _run_detector_on_image(
                detector=detector,
                image_path=image_path,
                image_id=image_id,
                image_w=image_record["width"],
                image_h=image_record["height"],
                conf=detector_conf,
            )
            decision_categories = DETECTION_CATEGORIES
        else:
            anns = annotation_index.get(image_id, [])
            decision_categories = categories

        action, confidence, reasoning = make_decision(
            image_record, anns, decision_categories,
            claude_config=claude_config,
            claude_state=claude_state,
        )

        predictions.append({
            "image_id": image_id,
            "action": action,
            "confidence": round(confidence, 4),
            "reasoning": reasoning,
        })

        # Collect detections for scoring (re-emit the annotations we used)
        for ann in anns:
            all_detections.append({
                "image_id": image_id,
                "category_id": ann["category_id"],
                "bbox": ann["bbox"],
                "score": ann.get("score", 1.0),  # GT annotations have implicit score 1.0
            })

    submission: Dict[str, Any] = {
        "team_name": team_name,
        "predictions": predictions,
    }

    # Include detections (for val, these are just the given annotations;
    # for test, you should populate this with your detector's output)
    if all_detections:
        submission["detections"] = all_detections
        detection_category_map = (
            DETECTION_CATEGORIES
            if detector is not None
            else categories
        )
        submission["detection_categories"] = [
            {"id": cid, "name": cname}
            for cid, cname in sorted(detection_category_map.items())
        ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(submission, fh, indent=2)

    print(f"Wrote {len(predictions)} predictions to {output_path}")
    if all_detections:
        print(f"Included {len(all_detections)} detections for scoring.")
    else:
        print("No detections included (test set has no annotations).")

    # Summary
    counts: Dict[str, int] = {}
    for p in predictions:
        counts[p["action"]] = counts.get(p["action"], 0) + 1
    print("Action distribution:")
    for act in ("STOP", "SLOW", "CONTINUE"):
        c = counts.get(act, 0)
        print(f"  {act:10s}: {c:5d}  ({c / len(predictions):.1%})")


def run_debug_one_image(
    data_dir: Path,
    split: str,
    model_path: Path,
    detector_conf: float,
    image_id: int | None = None,
) -> None:
    """Run detector on one image and open an interactive visualization window."""
    ann_path = data_dir / "annotations" / f"{split}.json"
    if not ann_path.exists():
        print(f"ERROR: Annotation file not found: {ann_path}")
        raise SystemExit(1)

    coco_data = load_annotations(ann_path)
    image_index = build_image_index(coco_data)
    if not image_index:
        print(f"ERROR: No images found in split '{split}'.")
        raise SystemExit(1)

    target_record = None
    if image_id is None:
        target_record = image_index[sorted(image_index.keys())[0]]
    else:
        target_record = image_index.get(image_id)
        if target_record is None:
            print(f"ERROR: image_id={image_id} not found in split '{split}'.")
            raise SystemExit(1)

    raw_category_names = [str(c["name"]) for c in coco_data.get("categories", [])]
    class_prompts = _build_detector_class_prompts(raw_category_names)
    detector = _build_detector(model_path, class_prompts)

    target_image_id = int(target_record["id"])
    image_path = _resolve_image_path(data_dir, split, target_record["file_name"])
    if not image_path.exists():
        print(f"ERROR: Image file not found for image_id={target_image_id}: {image_path}")
        raise SystemExit(1)

    detections = _run_detector_on_image(
        detector=detector,
        image_path=image_path,
        image_id=target_image_id,
        image_w=target_record["width"],
        image_h=target_record["height"],
        conf=detector_conf,
    )

    print(f"Debug image_id: {target_image_id}")
    print(f"Debug image_path: {image_path}")
    print(f"Detector class prompts: {len(class_prompts)}")
    print(f"Detections: {len(detections)}")
    print("Opening interactive visualization window...")

    visualize_scene(
        image_path=image_path,
        annotations=detections,
        categories=DETECTION_CATEGORIES,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="THEKER Hackathon -- Generate predictions for submission.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", type=Path, default=Path("../data"))
    parser.add_argument("--output", type=Path, default=Path("predictions.json"))
    parser.add_argument("--team-name", type=str, default="your_team")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "val", "test"])
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("../models/yolov8s-worldv2.pt"),
    )
    parser.add_argument("--detector-conf", type=float, default=0.25)
    parser.add_argument(
        "--use-detector-on-val",
        action="store_true",
        help="Run YOLO detector on val split instead of using provided annotations.",
    )
    parser.add_argument(
        "--enable-claude-decision",
        action="store_true",
        help="Use Claude API for navigation decisions (requires ANTHROPIC_API_KEY).",
    )
    parser.add_argument("--claude-max-calls", type=int, default=50,
                        help="Max Claude API calls per run.")
    parser.add_argument("--claude-max-input-chars", type=int, default=600,
                        help="Max characters in the scene description sent to Claude.")
    parser.add_argument("--claude-timeout-seconds", type=int, default=20,
                        help="Per-call timeout for Claude API requests.")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Process only the first N images (for quick subset runs).")
    parser.add_argument(
        "--debug-one-image",
        type=int,
        nargs="?",
        const=-1,
        default=None,
        metavar="IMAGE_ID",
        help=(
            "Run detector on a single image and open interactive visualization. "
            "Optionally provide IMAGE_ID; without it, uses first image in split."
        ),
    )
    args = parser.parse_args()
    if args.debug_one_image is not None:
        debug_image_id = None if args.debug_one_image == -1 else args.debug_one_image
        run_debug_one_image(
            data_dir=args.data_dir,
            split=args.split,
            model_path=args.model_path,
            detector_conf=args.detector_conf,
            image_id=debug_image_id,
        )
        raise SystemExit(0)

    run_predictions(
        args.data_dir,
        args.output,
        args.team_name,
        args.split,
        args.model_path,
        args.detector_conf,
        args.use_detector_on_val,
        args.enable_claude_decision,
        args.claude_max_calls,
        args.claude_max_input_chars,
        args.claude_timeout_seconds,
        args.max_images,
    )
