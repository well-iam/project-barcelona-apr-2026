"""Utility functions for the THEKER hackathon starter kit.

Provides helpers for loading COCO-format annotations, bounding box
operations, scene visualization, and fast index construction. Participants
can import these directly into their own pipelines.

Typical usage::

    from utils import (
        load_annotations,
        get_image_annotations,
        visualize_scene,
    )

    coco = load_annotations("data/annotations/val.json")
    anns = get_image_annotations(coco, image_id=42)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CATEGORY_NAMES: Dict[int, str] = {
    1: "person",
    2: "vehicle",
    3: "obstacle",
    4: "safety_marker",
}

CATEGORY_COLORS: Dict[int, str] = {
    1: "red",
    2: "blue",
    3: "orange",
    4: "green",
}

# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------


def load_annotations(json_path: str | Path) -> Dict[str, Any]:
    """Load a COCO-format annotation JSON file and return the parsed dict.

    Args:
        json_path: Path to a COCO annotation JSON file
            (e.g. ``data/annotations/train.json``).

    Returns:
        The full parsed JSON as a Python dict with keys such as
        ``"images"``, ``"annotations"``, and ``"categories"``.

    Raises:
        FileNotFoundError: If *json_path* does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    json_path = Path(json_path)
    with open(json_path, "r", encoding="utf-8") as fh:
        data: Dict[str, Any] = json.load(fh)
    return data


# ---------------------------------------------------------------------------
# Index builders (for fast lookup)
# ---------------------------------------------------------------------------


def build_image_index(coco_data: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    """Build a mapping from image_id to the corresponding image record.

    Args:
        coco_data: Parsed COCO annotation dict (as returned by
            :func:`load_annotations`).

    Returns:
        A dict where each key is an ``image_id`` (int) and each value is
        the full image record dict from ``coco_data["images"]``.
    """
    return {img["id"]: img for img in coco_data.get("images", [])}


def build_annotation_index(
    coco_data: Dict[str, Any],
) -> Dict[int, List[Dict[str, Any]]]:
    """Build a mapping from image_id to the list of its annotations.

    Args:
        coco_data: Parsed COCO annotation dict.

    Returns:
        A dict where each key is an ``image_id`` and each value is a list
        of annotation dicts that belong to that image.
    """
    index: Dict[int, List[Dict[str, Any]]] = {}
    for ann in coco_data.get("annotations", []):
        image_id = ann["image_id"]
        index.setdefault(image_id, []).append(ann)
    return index


# ---------------------------------------------------------------------------
# Annotation retrieval
# ---------------------------------------------------------------------------


def get_image_annotations(
    coco_data: Dict[str, Any], image_id: int
) -> List[Dict[str, Any]]:
    """Return all annotations for a given image.

    This performs a linear scan.  For repeated lookups across many images,
    prefer :func:`build_annotation_index` instead.

    Args:
        coco_data: Parsed COCO annotation dict.
        image_id: The integer ID of the target image.

    Returns:
        A list of annotation dicts whose ``image_id`` matches.
    """
    return [
        ann
        for ann in coco_data.get("annotations", [])
        if ann["image_id"] == image_id
    ]


def get_annotations_by_category(
    coco_data: Dict[str, Any], image_id: int
) -> Dict[str, List[Dict[str, Any]]]:
    """Return annotations for an image grouped by category name.

    Args:
        coco_data: Parsed COCO annotation dict.
        image_id: The integer ID of the target image.

    Returns:
        A dict mapping each category name (e.g. ``"person"``) to the list
        of matching annotations.  Categories with no annotations for this
        image are omitted.
    """
    # Build a local category-id-to-name map from the file itself, falling
    # back to the module-level CATEGORY_NAMES constant.
    cat_map: Dict[int, str] = {}
    for cat in coco_data.get("categories", []):
        cat_map[cat["id"]] = cat["name"]
    if not cat_map:
        cat_map = dict(CATEGORY_NAMES)

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for ann in get_image_annotations(coco_data, image_id):
        cat_name = cat_map.get(ann["category_id"], f"unknown_{ann['category_id']}")
        grouped.setdefault(cat_name, []).append(ann)
    return grouped


# ---------------------------------------------------------------------------
# Bounding-box utilities
# ---------------------------------------------------------------------------


def bbox_to_xyxy(bbox: List[float]) -> List[float]:
    """Convert a COCO-format bbox to ``[x1, y1, x2, y2]``.

    Args:
        bbox: A bounding box in ``[x, y, w, h]`` format (COCO standard).

    Returns:
        The same box as ``[x1, y1, x2, y2]`` where ``(x2, y2)`` is the
        bottom-right corner.
    """
    x, y, w, h = bbox
    return [x, y, x + w, y + h]


def bbox_center(bbox: List[float]) -> Tuple[float, float]:
    """Return the center point of a COCO-format bbox.

    Args:
        bbox: A bounding box in ``[x, y, w, h]`` format.

    Returns:
        A tuple ``(cx, cy)`` representing the center coordinates.
    """
    x, y, w, h = bbox
    return (x + w / 2.0, y + h / 2.0)


def bbox_area(bbox: List[float]) -> float:
    """Return the area of a COCO-format bbox.

    Args:
        bbox: A bounding box in ``[x, y, w, h]`` format.

    Returns:
        The area ``w * h``.
    """
    return bbox[2] * bbox[3]


def bbox_height_ratio(bbox: List[float], image_height: int) -> float:
    """Return the ratio of bbox height to the full image height.

    A higher ratio means the object is closer to the camera or occupies
    more of the vertical frame.

    Args:
        bbox: A bounding box in ``[x, y, w, h]`` format.
        image_height: The height of the image in pixels.

    Returns:
        A float in ``[0, 1]`` representing ``bbox_h / image_height``.
    """
    if image_height <= 0:
        return 0.0
    return bbox[3] / image_height


def compute_iou(bbox1: List[float], bbox2: List[float]) -> float:
    """Compute Intersection-over-Union between two COCO-format bboxes.

    Args:
        bbox1: First bounding box in ``[x, y, w, h]`` format.
        bbox2: Second bounding box in ``[x, y, w, h]`` format.

    Returns:
        The IoU value in ``[0, 1]``.  Returns ``0.0`` if the boxes do not
        overlap or if either box has zero area.
    """
    x1_1, y1_1, x2_1, y2_1 = bbox_to_xyxy(bbox1)
    x1_2, y1_2, x2_2, y2_2 = bbox_to_xyxy(bbox2)

    inter_x1 = max(x1_1, x1_2)
    inter_y1 = max(y1_1, y1_2)
    inter_x2 = min(x2_1, x2_2)
    inter_y2 = min(y2_1, y2_2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area1 = bbox_area(bbox1)
    area2 = bbox_area(bbox2)
    union_area = area1 + area2 - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def visualize_scene(
    image_path: str | Path,
    annotations: List[Dict[str, Any]],
    categories: Optional[Dict[int, str]] = None,
    save_path: Optional[str | Path] = None,
) -> None:
    """Draw bounding boxes on an image, color-coded by class.

    Opens the image, overlays each annotation as a colored rectangle with
    a text label, and either displays the result or saves it to disk.

    Args:
        image_path: Path to the source image file (JPEG/PNG).
        annotations: A list of annotation dicts, each containing at least
            ``"bbox"`` (``[x, y, w, h]``) and ``"category_id"`` (int).
        categories: Optional mapping from category ID to display name.
            Defaults to :data:`CATEGORY_NAMES` if not provided.
        save_path: If given, the annotated image is saved to this path
            instead of being displayed interactively.
    """
    if categories is None:
        categories = CATEGORY_NAMES

    img = Image.open(image_path)
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(img)

    for ann in annotations:
        bbox = ann["bbox"]  # [x, y, w, h]
        cat_id = ann["category_id"]
        color = CATEGORY_COLORS.get(cat_id, "white")
        label = categories.get(cat_id, f"class_{cat_id}")

        rect = patches.Rectangle(
            (bbox[0], bbox[1]),
            bbox[2],
            bbox[3],
            linewidth=2,
            edgecolor=color,
            facecolor="none",
        )
        ax.add_patch(rect)
        ax.text(
            bbox[0],
            bbox[1] - 5,
            label,
            color="white",
            fontsize=9,
            fontweight="bold",
            bbox=dict(facecolor=color, alpha=0.7, edgecolor="none", pad=1),
        )

    ax.set_axis_off()
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(str(save_path), bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
