"""Label ontology for THEKER hackathon.

Maps all 27 raw annotation category names (and YOLO-World detection labels)
to four decision-relevant groups. Uses lowercase normalization so case
variants (box/Box, barrel/Barrel, etc.) resolve automatically.

Usage::

    from ontology import get_group, normalize_label

    group = get_group("Barrel")   # -> "OBSTACLE_LIKE"
    group = get_group("helmet")   # -> "PERSON_LIKE"
    group = get_group("unknown")  # -> None
"""

from __future__ import annotations


def normalize_label(raw: str) -> str:
    """Lowercase + strip; collapses case variants to a single form."""
    return raw.strip().lower()


# ---------------------------------------------------------------------------
# Annotation ontology (covers all 27 raw train/val label names)
# Lowercase normalization means Box->box, Barrel->barrel, etc. are handled.
# ---------------------------------------------------------------------------

PERSON_LABELS: frozenset[str] = frozenset({
    "person", "hat", "helmet", "head",
    # YOLO-World open-vocabulary synonyms
    "man", "woman", "boy", "girl", "adult", "child", "pedestrian", "worker",
})

VEHICLE_LABELS: frozenset[str] = frozenset({
    "bicycle", "car", "motorcycle", "bus", "train", "truck",
    "forklift", "handcart",
    # YOLO-World synonyms
    "bike", "motorbike", "van",
})

OBSTACLE_LABELS: frozenset[str] = frozenset({
    # Raw annotation labels (case-folded)
    "suitcase", "chair", "barrel", "crate", "box", "ladder", "container",
    # YOLO-World synonyms
    "rock", "debris", "pallet",
})

SAFETY_LABELS: frozenset[str] = frozenset({
    "cone", "traffic sign", "stop sign", "traffic light",
    # YOLO-World synonyms
    "bollard", "barrier", "warning sign",
})


def get_group(raw: str) -> str | None:
    """Map a raw label string to one of the four decision groups.

    Args:
        raw: Raw category name from annotations or a YOLO-World detection.

    Returns:
        One of ``"PERSON_LIKE"``, ``"VEHICLE_LIKE"``, ``"OBSTACLE_LIKE"``,
        ``"SAFETY_MARKER_LIKE"``, or ``None`` if unrecognized.
    """
    n = normalize_label(raw)
    if n in PERSON_LABELS:
        return "PERSON_LIKE"
    if n in VEHICLE_LABELS:
        return "VEHICLE_LIKE"
    if n in OBSTACLE_LABELS:
        return "OBSTACLE_LIKE"
    if n in SAFETY_LABELS:
        return "SAFETY_MARKER_LIKE"
    return None
