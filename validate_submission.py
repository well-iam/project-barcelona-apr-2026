"""Validate a submission file before uploading.

Checks the submission JSON against all rules in SUBMISSION_FORMAT.md.
Exit code 0 = valid, exit code 1 = errors found.

Usage::

    python validate_submission.py submission.json --test-json ../data/annotations/test.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

VALID_ACTIONS = {"STOP", "SLOW", "CONTINUE"}


def validate(submission_path: Path, test_json_path: Path | None = None) -> list[str]:
    """Validate a submission file. Returns a list of error strings.

    Args:
        submission_path: Path to the submission JSON.
        test_json_path: Path to test.json for image ID validation.

    Returns:
        List of error messages. Empty list means valid.
    """
    errors: list[str] = []
    warnings: list[str] = []

    # -- Load JSON --
    try:
        with open(submission_path) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return [f"Invalid JSON: {e}"]
    except FileNotFoundError:
        return [f"File not found: {submission_path}"]

    if not isinstance(data, dict):
        return ["Root must be a JSON object, got " + type(data).__name__]

    # -- team_name --
    if "team_name" not in data:
        errors.append("Missing required field: 'team_name'")
    elif not isinstance(data["team_name"], str) or not data["team_name"].strip():
        errors.append("'team_name' must be a non-empty string")

    # -- predictions --
    preds = data.get("predictions")
    if preds is None:
        errors.append("Missing required field: 'predictions'")
        return errors
    if not isinstance(preds, list):
        errors.append("'predictions' must be an array")
        return errors
    if len(preds) == 0:
        errors.append("'predictions' array is empty")

    # Load test image IDs if available
    test_ids: set[int] | None = None
    if test_json_path and test_json_path.exists():
        test_data = json.load(open(test_json_path))
        test_ids = {img["id"] for img in test_data.get("images", [])}

    # Validate each prediction
    seen_ids: set[int] = set()
    for i, pred in enumerate(preds):
        prefix = f"predictions[{i}]"

        if not isinstance(pred, dict):
            errors.append(f"{prefix}: must be an object")
            continue

        # image_id
        img_id = pred.get("image_id")
        if img_id is None:
            errors.append(f"{prefix}: missing 'image_id'")
        elif not isinstance(img_id, int):
            errors.append(f"{prefix}: 'image_id' must be an integer, got {type(img_id).__name__}")
        else:
            if img_id in seen_ids:
                errors.append(f"{prefix}: duplicate image_id {img_id}")
            seen_ids.add(img_id)
            if test_ids is not None and img_id not in test_ids:
                errors.append(f"{prefix}: image_id {img_id} not in test set")

        # action
        action = pred.get("action")
        if action is None:
            errors.append(f"{prefix}: missing 'action'")
        elif action not in VALID_ACTIONS:
            errors.append(f"{prefix}: invalid action '{action}'. Must be STOP, SLOW, or CONTINUE")

        # confidence
        conf = pred.get("confidence")
        if conf is None:
            errors.append(f"{prefix}: missing 'confidence'")
        elif not isinstance(conf, (int, float)):
            errors.append(f"{prefix}: 'confidence' must be a number")
        elif conf < 0.0 or conf > 1.0:
            errors.append(f"{prefix}: 'confidence' {conf} out of range [0.0, 1.0]")

        # reasoning
        reason = pred.get("reasoning")
        if reason is None:
            errors.append(f"{prefix}: missing 'reasoning'")
        elif not isinstance(reason, str):
            errors.append(f"{prefix}: 'reasoning' must be a string")
        elif len(reason.strip()) < 10:
            errors.append(f"{prefix}: 'reasoning' too short (min 10 chars)")

    # Coverage check
    if test_ids is not None:
        missing = test_ids - seen_ids
        extra = seen_ids - test_ids
        if missing:
            errors.append(f"Missing predictions for {len(missing)} test images (first 5: {sorted(missing)[:5]})")
        if extra:
            errors.append(f"Predictions for {len(extra)} images not in test set")

    # -- detections (optional) --
    dets = data.get("detections")
    if dets is not None:
        if not isinstance(dets, list):
            errors.append("'detections' must be an array")
        else:
            for j, det in enumerate(dets):
                prefix = f"detections[{j}]"
                if not isinstance(det, dict):
                    errors.append(f"{prefix}: must be an object")
                    continue
                if "image_id" not in det:
                    errors.append(f"{prefix}: missing 'image_id'")
                if "category_id" not in det:
                    errors.append(f"{prefix}: missing 'category_id'")
                bbox = det.get("bbox")
                if bbox is None:
                    errors.append(f"{prefix}: missing 'bbox'")
                elif not isinstance(bbox, list) or len(bbox) != 4:
                    errors.append(f"{prefix}: 'bbox' must be [x, y, w, h] (4 numbers)")
                if "score" not in det:
                    errors.append(f"{prefix}: missing 'score'")

                # Stop reporting after 10 detection errors
                if len([e for e in errors if e.startswith("detections")]) > 10:
                    errors.append("... (truncated, too many detection errors)")
                    break
    else:
        warnings.append("No 'detections' array found. You will score 0% on detection F1 (20% of total).")

    return errors, warnings


def main() -> None:
    """Parse arguments and run validation."""
    parser = argparse.ArgumentParser(
        description="Validate a submission file against THEKER challenge rules.",
    )
    parser.add_argument("submission", type=Path, help="Path to submission JSON")
    parser.add_argument(
        "--test-json", type=Path, default=None,
        help="Path to test.json for image ID validation",
    )
    args = parser.parse_args()

    result = validate(args.submission, args.test_json)

    # Handle both old (list) and new (tuple) return
    if isinstance(result, tuple):
        errors, warnings = result
    else:
        errors, warnings = result, []

    if warnings:
        for w in warnings:
            print(f"  WARNING: {w}")
        print()

    if errors:
        print(f"INVALID: {len(errors)} error(s) found:\n")
        for e in errors:
            print(f"  ERROR: {e}")
        print(f"\nFix these errors before submitting.")
        sys.exit(1)
    else:
        print("VALID: Submission file passes all checks.")
        sys.exit(0)


if __name__ == "__main__":
    main()
