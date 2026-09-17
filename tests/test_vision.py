import numpy as np

from src.vision.detect import clutter_from_boxes, occupancy_from_count, draw_overlay


def test_occupancy_from_count_empty_seats():
    out = occupancy_from_count(12, 40)
    assert out["people_detected"] == 12
    assert out["empty_seats_estimate"] == 28
    assert out["occupancy_level"] == "Low"


def test_occupancy_caps_at_capacity():
    out = occupancy_from_count(90, 40)
    assert out["occupied_estimate"] == 40
    assert out["empty_seats_estimate"] == 0
    assert out["occupancy_level"] == "Critical"


def test_clutter_grid_marks_centroids():
    boxes = [(10, 10, 30, 30), (200, 10, 220, 30)]
    clut = clutter_from_boxes(boxes, width=240, height=120)
    assert clut["occupied_cells"] >= 1
    assert clut["total_cells"] == 24
    assert 0 <= clut["clutter_ratio"] <= 1


def test_draw_overlay_keeps_shape():
    frame = np.zeros((80, 120, 3), dtype=np.uint8)
    boxes = [(5, 5, 20, 40)]
    occ = occupancy_from_count(1, 10)
    clut = clutter_from_boxes(boxes, 120, 80)
    out = draw_overlay(frame, boxes, occ, clut)
    assert out.shape == frame.shape
