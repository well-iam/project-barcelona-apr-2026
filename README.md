# THEKER 2026 — Robot Navigation Classifier

**Team Barcelona** · THEKER 2026 Hackathon

A hybrid object detection + LLM pipeline for real-time robot safety decisions in industrial environments. Given a single image, the system outputs one of three actions: **STOP**, **SLOW**, or **CONTINUE**.

---

## The Problem

A robot navigating a warehouse or construction site must make a safety call every frame. Too cautious and it grinds operations to a halt. Too aggressive and it collides with workers. The challenge is a 27-category, multi-source dataset — COCO, SHWD, OpenImages, and Hardhat — with heterogeneous annotation conventions and no test labels.

---

## Pipeline

```
Image → YOLO-World XL → Hard Geometric Rules → Claude API → Calibrated Decision
```

1. **YOLO-World XL** detects objects using open-vocabulary prompts across all 27 label categories.
2. **Hard rules** handle unambiguous cases instantly (e.g. person h > 0.25 in central lane → STOP). No API call, no latency.
3. **Claude** is called only for ambiguous scenes, receiving a structured scene description and returning a JSON decision with action, confidence, and reasoning.

The key technical challenge was an annotation scale mismatch: ground truth labels use head/helmet bounding boxes (h ≈ 0.06–0.25), while YOLO detects full bodies (h ≈ 0.35–0.63). We resolved this by separating head-type and body-type detections and calibrating thresholds independently.

---

## Results

| Metric | Score |
|---|---|
| Weighted accuracy (competition metric) | **77.4%** |
| STOP recall | 77.5% |
| SLOW recall | 67.0% |
| Target | > 75% ✓ |

> Note: the provided `evaluate_local.py` reports exact-match accuracy (61%). The competition uses partial-credit scoring where one level off = 0.3 — the weighted accuracy under that metric is 77.4%.

Inference ran on a CoreWeave B200 (183 GB VRAM) via Northflank.

---

## Files

| File | Description |
|---|---|
| `predict.py` | Main pipeline — YOLO detection, hard switches, Claude integration, confidence calibration |
| `ontology.py` | Maps all 27 raw annotation labels to four decision groups: PERSON_LIKE, VEHICLE_LIKE, OBSTACLE_LIKE, SAFETY_MARKER_LIKE |
| `submission_b200.json` | Final test set predictions (3756 images) — 42.2% STOP, 54.3% SLOW, 3.5% CONTINUE |
| `demo.html` | Animated presentation (open in browser, requires internet for reveal.js CDN) |

---

## Usage

```bash
pip install ultralytics anthropic

# Run on validation set
python predict.py --data-dir ../data --split val --output predictions.json

# Run on test set
python predict.py \
  --data-dir ../data \
  --split test \
  --output submission.json \
  --model-path /path/to/yolov8x-worldv2.pt \
  --enable-claude-decision \
  --claude-max-calls 400
```

Set `ANTHROPIC_API_KEY` in your environment before running with `--enable-claude-decision`.
