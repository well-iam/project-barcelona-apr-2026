"""THEKER Hackathon -- Local Evaluation Helper.

Generates *approximate* ground-truth decisions from the validation
annotations using the same rule-based heuristics as the baseline in
``predict.py``, then compares your predictions against them.

**WARNING**: This is NOT the official scorer.  The hidden evaluation
pipeline uses a different (and more nuanced) ground-truth derivation.
Use this script to catch obvious issues and track relative improvement
during development -- not as an exact proxy for leaderboard scores.

Usage::

    python evaluate_local.py \\
        --predictions predictions.json \\
        --annotations data/annotations/val.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

from utils import (
    bbox_area,
    bbox_height_ratio,
    build_annotation_index,
    build_image_index,
    load_annotations,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ACTIONS = ["STOP", "SLOW", "CONTINUE"]
ACTION_SET = set(ACTIONS)

# Raw label grouping (same as predict.py baseline)
PERSON_LABELS = {"person", "hat", "helmet", "head"}
VEHICLE_LABELS = {"bicycle", "car", "motorcycle", "bus", "train", "truck", "forklift"}
OBSTACLE_LABELS = {
    "suitcase", "chair", "barrel", "crate", "box", "handcart", "ladder",
    "Box", "Barrel", "Container", "Ladder", "Suitcase",
}
SAFETY_LABELS = {"cone", "Traffic sign", "Stop sign", "Traffic light"}

# Banner printed at the top and bottom of every report.
_DISCLAIMER = (
    "NOTE: This is an APPROXIMATE local evaluation. The official hidden\n"
    "scorer uses different ground-truth logic. Use these numbers for\n"
    "relative comparisons only."
)


# ---------------------------------------------------------------------------
# Approximate ground-truth generation
# ---------------------------------------------------------------------------


def _classify(ann: Dict[str, Any], categories: Dict[int, str]) -> str | None:
    """Map raw annotation to decision group."""
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


def generate_ground_truth(
    image_record: Dict[str, Any],
    annotations: List[Dict[str, Any]],
    categories: Dict[int, str] | None = None,
) -> str:
    """Derive an approximate ground-truth action from raw annotations.

    Uses the same rule-based heuristics as the baseline ``make_decision``
    in ``predict.py``.  Groups the 27 raw labels first, then applies
    rules in priority order:

    1. Person with bbox height ratio > 0.25 -> ``STOP``
    2. Vehicle with area > 15 % of image area -> ``STOP``
    3. Any person or vehicle present -> ``SLOW``
    4. Any safety marker present -> ``SLOW``
    5. Obstacles covering > 40 % of image width -> ``SLOW``
    6. Otherwise -> ``CONTINUE``

    Args:
        image_record: COCO image dict with ``"height"`` and ``"width"``.
        annotations: Annotation dicts for this image.
        categories: Mapping of category_id -> name.  If *None*, falls back
            to category_id matching (legacy behavior).

    Returns:
        One of ``"STOP"``, ``"SLOW"``, or ``"CONTINUE"``.
    """
    if categories is None:
        categories = {}

    img_h = image_record["height"]
    img_w = image_record["width"]
    img_area = img_h * img_w

    persons: List[Dict[str, Any]] = []
    vehicles: List[Dict[str, Any]] = []
    obstacles: List[Dict[str, Any]] = []
    safety_markers: List[Dict[str, Any]] = []

    for ann in annotations:
        group = _classify(ann, categories)
        if group == "person":
            persons.append(ann)
        elif group == "vehicle":
            vehicles.append(ann)
        elif group == "obstacle":
            obstacles.append(ann)
        elif group == "safety_marker":
            safety_markers.append(ann)

    # Rule 1: Large person -> STOP
    for ann in persons:
        if bbox_height_ratio(ann["bbox"], img_h) > 0.25:
            return "STOP"

    # Rule 2: Large vehicle -> STOP
    for ann in vehicles:
        if bbox_area(ann["bbox"]) > 0.15 * img_area:
            return "STOP"

    # Rule 3: Any person or vehicle -> SLOW
    if persons or vehicles:
        return "SLOW"

    # Rule 4: Safety marker -> SLOW
    if safety_markers:
        return "SLOW"

    # Rule 5: Obstacles covering >40% of image width -> SLOW
    if obstacles:
        ranges = sorted((ann["bbox"][0], ann["bbox"][0] + ann["bbox"][2]) for ann in obstacles)
        merged: List[Tuple[float, float]] = [ranges[0]]
        for start, end in ranges[1:]:
            if start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        total_width = sum(e - s for s, e in merged)
        if total_width > 0.40 * img_w:
            return "SLOW"

    return "CONTINUE"


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def compute_confusion_matrix(
    gt_actions: List[str],
    pred_actions: List[str],
) -> Dict[str, Dict[str, int]]:
    """Build a confusion matrix as a nested dict.

    Args:
        gt_actions: Ground-truth action labels.
        pred_actions: Predicted action labels (same length as
            *gt_actions*).

    Returns:
        ``matrix[gt_label][pred_label] = count``.
    """
    matrix: Dict[str, Dict[str, int]] = {
        gt: {pred: 0 for pred in ACTIONS} for gt in ACTIONS
    }
    for gt, pred in zip(gt_actions, pred_actions):
        matrix[gt][pred] += 1
    return matrix


def print_confusion_matrix(matrix: Dict[str, Dict[str, int]]) -> None:
    """Print a formatted confusion matrix to stdout.

    Args:
        matrix: Confusion matrix as returned by
            :func:`compute_confusion_matrix`.
    """
    #header = f"{'GT \\ Pred':>12s}" + "".join(f"{a:>12s}" for a in ACTIONS)
    header = "{:>12s}".format("GT \\ Pred") + "".join(f"{a:>12s}" for a in ACTIONS)
    print(header)
    print("-" * len(header))
    for gt in ACTIONS:
        row = f"{gt:>12s}" + "".join(f"{matrix[gt][pred]:>12d}" for pred in ACTIONS)
        print(row)


def compute_per_class_accuracy(
    matrix: Dict[str, Dict[str, int]],
) -> Dict[str, float]:
    """Compute accuracy (recall) for each ground-truth class.

    Args:
        matrix: Confusion matrix from :func:`compute_confusion_matrix`.

    Returns:
        Dict mapping each action to its recall (correct / total for that
        ground-truth class).  Returns ``0.0`` for classes with no samples.
    """
    acc: Dict[str, float] = {}
    for gt in ACTIONS:
        total = sum(matrix[gt].values())
        correct = matrix[gt][gt]
        acc[gt] = correct / total if total > 0 else 0.0
    return acc


def compute_overall_accuracy(
    gt_actions: List[str],
    pred_actions: List[str],
) -> float:
    """Compute simple overall accuracy.

    Args:
        gt_actions: Ground-truth action labels.
        pred_actions: Predicted action labels (same length).

    Returns:
        Fraction of predictions that exactly match the ground truth.
    """
    if not gt_actions:
        return 0.0
    correct = sum(1 for g, p in zip(gt_actions, pred_actions) if g == p)
    return correct / len(gt_actions)


# ---------------------------------------------------------------------------
# Main evaluation pipeline
# ---------------------------------------------------------------------------


def evaluate(predictions_path: Path, annotations_path: Path) -> None:
    """Run the full local evaluation and print a report.

    Args:
        predictions_path: Path to the submission JSON produced by
            ``predict.py``.
        annotations_path: Path to the COCO annotation JSON for the
            validation split.
    """
    # ------------------------------------------------------------------
    # Load predictions
    # ------------------------------------------------------------------
    if not predictions_path.exists():
        print(f"ERROR: Predictions file not found: {predictions_path}")
        sys.exit(1)

    with open(predictions_path, "r", encoding="utf-8") as fh:
        submission = json.load(fh)

    pred_list = submission.get("predictions", [])
    team_name = submission.get("team_name", "<unknown>")

    if not pred_list:
        print("ERROR: No predictions found in submission file.")
        sys.exit(1)

    pred_by_id: Dict[int, Dict[str, Any]] = {p["image_id"]: p for p in pred_list}

    # Validate actions
    invalid = [
        p["image_id"]
        for p in pred_list
        if p.get("action") not in ACTION_SET
    ]
    if invalid:
        print(
            f"ERROR: {len(invalid)} prediction(s) have invalid actions. "
            f"Valid actions are: {ACTIONS}. First invalid image_id: {invalid[0]}"
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # Load annotations and generate approximate ground truth
    # ------------------------------------------------------------------
    if not annotations_path.exists():
        print(f"ERROR: Annotations file not found: {annotations_path}")
        sys.exit(1)

    coco_data = load_annotations(annotations_path)
    image_index = build_image_index(coco_data)
    annotation_index = build_annotation_index(coco_data)

    # Build category lookup from the annotations file
    categories = {c["id"]: c["name"] for c in coco_data.get("categories", [])}

    gt_actions: List[str] = []
    pred_actions: List[str] = []
    missing_ids: List[int] = []

    for image_id in sorted(image_index.keys()):
        image_record = image_index[image_id]
        anns = annotation_index.get(image_id, [])
        gt_action = generate_ground_truth(image_record, anns, categories)

        if image_id not in pred_by_id:
            missing_ids.append(image_id)
            continue

        gt_actions.append(gt_action)
        pred_actions.append(pred_by_id[image_id]["action"])

    # ------------------------------------------------------------------
    # Print report
    # ------------------------------------------------------------------
    print()
    print("=" * 64)
    print("  THEKER Hackathon -- Local Evaluation Report")
    print("=" * 64)
    print()
    print(_DISCLAIMER)
    print()
    print(f"  Team name       : {team_name}")
    print(f"  Predictions file: {predictions_path}")
    print(f"  Annotations file: {annotations_path}")
    print(f"  Images in GT    : {len(image_index)}")
    print(f"  Predictions     : {len(pred_list)}")
    print(f"  Evaluated pairs : {len(gt_actions)}")
    if missing_ids:
        print(
            f"  MISSING preds   : {len(missing_ids)} image(s) had no prediction"
        )
    print()

    # GT distribution
    gt_counts = Counter(gt_actions)
    pred_counts = Counter(pred_actions)
    print("  Approximate GT distribution:")
    for act in ACTIONS:
        print(f"    {act:10s}: {gt_counts.get(act, 0):5d}")
    print()
    print("  Prediction distribution:")
    for act in ACTIONS:
        print(f"    {act:10s}: {pred_counts.get(act, 0):5d}")
    print()

    # Confusion matrix
    matrix = compute_confusion_matrix(gt_actions, pred_actions)
    print("  Confusion matrix (rows = approx GT, cols = predicted):")
    print()
    print_confusion_matrix(matrix)
    print()

    # Per-class accuracy
    per_class = compute_per_class_accuracy(matrix)
    print("  Per-class accuracy (recall):")
    for act in ACTIONS:
        print(f"    {act:10s}: {per_class[act]:.2%}")
    print()

    # Overall accuracy
    overall = compute_overall_accuracy(gt_actions, pred_actions)
    print(f"  Overall accuracy: {overall:.2%}")
    print()

    # Confidence stats
    confidences = [pred_by_id[img_id]["confidence"] for img_id in sorted(image_index.keys()) if img_id in pred_by_id]
    if confidences:
        avg_conf = sum(confidences) / len(confidences)
        min_conf = min(confidences)
        max_conf = max(confidences)
        print(f"  Confidence stats: mean={avg_conf:.3f}  min={min_conf:.3f}  max={max_conf:.3f}")
        print()

    print("=" * 64)
    print(f"  {_DISCLAIMER}")
    print("=" * 64)
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed namespace with ``predictions`` and ``annotations``
        attributes.
    """
    parser = argparse.ArgumentParser(
        description=(
            "THEKER Hackathon -- Local evaluation helper.  Compares your "
            "predictions against approximate ground truth derived from "
            "validation annotations.  This is NOT the official scorer."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="Path to the submission JSON (output of predict.py).",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        required=True,
        help="Path to the COCO annotation JSON for the val split.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(
        predictions_path=args.predictions,
        annotations_path=args.annotations,
    )
