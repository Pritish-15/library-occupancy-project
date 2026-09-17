"""Demo person detection on uploaded library videos (not a live camera system)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np


GRID_ROWS = 4
GRID_COLS = 6


def occupancy_from_count(n_people: int, capacity: int) -> dict:
    capacity = max(int(capacity), 1)
    n_people = max(int(n_people), 0)
    occupied_est = min(n_people, capacity)
    empty_est = max(capacity - n_people, 0)
    util = occupied_est / capacity
    if util < 0.4:
        level = "Low"
    elif util < 0.7:
        level = "Moderate"
    elif util < 0.9:
        level = "High"
    else:
        level = "Critical"
    return {
        "people_detected": n_people,
        "capacity": capacity,
        "occupied_estimate": occupied_est,
        "empty_seats_estimate": empty_est,
        "utilization": round(util, 4),
        "occupancy_level": level,
    }


def clutter_from_boxes(boxes: list[tuple[float, float, float, float]], width: int, height: int) -> dict:
    grid = np.zeros((GRID_ROWS, GRID_COLS), dtype=int)
    for x1, y1, x2, y2 in boxes:
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        col = min(GRID_COLS - 1, max(0, int(cx / max(width, 1) * GRID_COLS)))
        row = min(GRID_ROWS - 1, max(0, int(cy / max(height, 1) * GRID_ROWS)))
        grid[row, col] += 1
    occupied_cells = int((grid > 0).sum())
    total = GRID_ROWS * GRID_COLS
    return {
        "grid": grid.tolist(),
        "occupied_cells": occupied_cells,
        "total_cells": total,
        "clutter_ratio": round(occupied_cells / total, 4),
    }


def draw_overlay(frame: np.ndarray, boxes: list, occupancy: dict, clutter: dict) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    cell_h, cell_w = h / GRID_ROWS, w / GRID_COLS
    grid = np.array(clutter["grid"])
    overlay = out.copy()
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            x1, y1 = int(c * cell_w), int(r * cell_h)
            x2, y2 = int((c + 1) * cell_w), int((r + 1) * cell_h)
            if grid[r, c] > 0:
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (40, 70, 210), -1)
            else:
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (60, 140, 70), -1)
            cv2.rectangle(out, (x1, y1), (x2, y2), (220, 220, 220), 1)
    out = cv2.addWeighted(overlay, 0.22, out, 0.78, 0)
    for x1, y1, x2, y2 in boxes:
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(out, p1, p2, (36, 180, 255), 2)
        cv2.putText(out, "person", (p1[0], max(18, p1[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 180, 255), 1, cv2.LINE_AA)
    bar = occupancy
    hud = (
        f"People {bar['people_detected']}  |  Capacity {bar['capacity']}  |  "
        f"Empty seats (est.) {bar['empty_seats_estimate']}  |  "
        f"Clutter {clutter['clutter_ratio']:.0%}  |  {bar['occupancy_level']}"
    )
    cv2.rectangle(out, (0, 0), (w, 42), (18, 28, 24), -1)
    cv2.putText(out, hud, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (244, 239, 230), 1, cv2.LINE_AA)
    return out


@lru_cache(maxsize=1)
def _yolo():
    from ultralytics import YOLO

    weights = Path(__file__).resolve().parents[2] / "models" / "yolov8n.pt"
    if weights.exists():
        return YOLO(str(weights))
    return YOLO("yolov8n.pt")


def detect_people(frame: np.ndarray, conf: float = 0.35) -> list[tuple[float, float, float, float]]:
    model = _yolo()
    result = model.predict(frame, classes=[0], conf=conf, verbose=False)[0]
    boxes = []
    if result.boxes is None:
        return boxes
    for xyxy in result.boxes.xyxy.cpu().numpy():
        boxes.append(tuple(float(v) for v in xyxy))
    return boxes


def process_video(
    source: Path,
    dest: Path,
    capacity: int,
    max_frames: int = 180,
    stride: int = 2,
    preview_dir: Path | None = None,
) -> dict:
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise ValueError("Could not open the uploaded video")
    fps = cap.get(cv2.CAP_PROP_FPS) or 12.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 360)
    dest.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(dest), cv2.VideoWriter_fourcc(*"mp4v"), max(fps / stride, 4.0), (width, height))
    counts = []
    clutter_ratios = []
    previews = []
    idx = 0
    kept = 0
    if preview_dir:
        preview_dir.mkdir(parents=True, exist_ok=True)
    while kept < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride != 0:
            idx += 1
            continue
        boxes = detect_people(frame)
        occ = occupancy_from_count(len(boxes), capacity)
        clut = clutter_from_boxes(boxes, width, height)
        annotated = draw_overlay(frame, boxes, occ, clut)
        writer.write(annotated)
        counts.append(occ["people_detected"])
        clutter_ratios.append(clut["clutter_ratio"])
        if preview_dir is not None and kept % 8 == 0:
            shot = preview_dir / f"frame_{kept:04d}.jpg"
            cv2.imwrite(str(shot), annotated)
            previews.append(shot.name)
        idx += 1
        kept += 1
    cap.release()
    writer.release()
    peak = max(counts) if counts else 0
    mean_people = float(np.mean(counts)) if counts else 0.0
    summary = occupancy_from_count(int(round(mean_people)), capacity)
    peak_occ = occupancy_from_count(peak, capacity)
    return {
        "frames_processed": kept,
        "mean_people": round(mean_people, 2),
        "peak_people": peak,
        "mean_clutter_ratio": round(float(np.mean(clutter_ratios)) if clutter_ratios else 0.0, 4),
        "peak_clutter_ratio": round(float(max(clutter_ratios)) if clutter_ratios else 0.0, 4),
        "mean_occupancy": summary,
        "peak_occupancy": peak_occ,
        "previews": previews,
        "note": "Seat estimates assume the camera roughly covers this zone. Demo only — not a calibrated seat map.",
    }
