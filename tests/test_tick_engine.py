"""Tests for NOT OK-only pen-mark detection."""
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from row_engine import detect_checklist_rows
from tick_engine import detect_not_ok_marks


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = ROOT / "sample_photos"
SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _sample_paths():
    return sorted(path for path in SAMPLE_DIR.iterdir()
                  if path.is_file() and path.suffix.lower() in SUFFIXES)


def _synthetic_checklist():
    image = np.full((560, 800, 3), 255, np.uint8)
    x_lines = [40, 110, 470, 550, 630, 760]
    y_lines = [40 + 35 * i for i in range(13)]
    for x in x_lines:
        cv2.line(image, (x, y_lines[0]), (x, y_lines[-1]), (25, 25, 25), 1)
    for y in y_lines:
        cv2.line(image, (x_lines[0], y), (x_lines[-1], y), (25, 25, 25), 1)
    return image


def _draw_mark(image, box, color):
    x1, y1, x2, y2 = box
    width, height = x2 - x1, y2 - y1
    points = np.array([
        [x1 + round(width * 0.20), y1 + round(height * 0.56)],
        [x1 + round(width * 0.39), y1 + round(height * 0.84)],
        [x1 + round(width * 0.78), y1 + round(height * 0.16)],
    ], dtype=np.int32)
    cv2.polylines(image, [points], False, color, max(2, round(height * 0.09)), cv2.LINE_AA)


def _draw_tick_debug(detections, output):
    path, image, geometry, results = detections[0]
    canvas = geometry.corrected_image.copy()
    by_index = {result.row_index: result for result in results}
    for row in geometry.rows:
        color = (30, 160, 20) if by_index[row.row_index].not_ok_marked else (30, 30, 220)
        x1, y1, x2, y2 = row.bbox
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 160, 240), 1)
        nx1, ny1, nx2, ny2 = row.not_ok_bbox
        cv2.rectangle(canvas, (nx1, ny1), (nx2, ny2), color, 2)
        for cx1, cy1, cx2, cy2 in by_index[row.row_index].evidence["component_boxes"]:
            cv2.rectangle(canvas, (cx1, cy1), (cx2, cy2), (255, 0, 180), 2)
        cv2.putText(canvas, "M" if by_index[row.row_index].not_ok_marked else "-",
                    (nx1 + 2, ny1 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    output.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(output), canvas)


@pytest.fixture(scope="module")
def real_runs():
    paths = _sample_paths()
    assert paths, f"no sample photos found in {SAMPLE_DIR}"
    runs = []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        assert image is not None and image.shape[0] > 0 and image.shape[1] > 0
        geometry = detect_checklist_rows(image)
        results = detect_not_ok_marks(image, geometry)
        assert len(results) == len(geometry.rows)
        runs.append((path, image, geometry, results))
    _draw_tick_debug(runs, ROOT / "tests" / "artifacts" / "tick_engine_real_debug.png")
    return runs


def test_real_checklists_yield_one_result_per_detected_row(real_runs):
    for path, image, geometry, results in real_runs:
        assert results
        assert [result.row_index for result in results] == [row.row_index for row in geometry.rows], path.name
        assert all(0 <= result.confidence <= 1 for result in results)


def test_not_ok_boxes_are_taken_from_row_engine(real_runs):
    for path, image, geometry, results in real_runs:
        for row, result in zip(geometry.rows, results):
            assert result.row_index == row.row_index, path.name
            assert tuple(result.evidence["cell_bbox"]) == tuple(row.not_ok_bbox)


def test_visible_real_not_ok_marks_are_detected(real_runs):
    # Each supplied photo contains a visible pen mark in at least one NOT OK
    # cell. Assertions use detected row evidence and do not reference labels.
    for path, image, geometry, results in real_runs:
        marked = [result for result in results if result.not_ok_marked]
        assert marked, f"no visible NOT OK mark detected in {path.name}"
        assert all(result.evidence["ink_pixels"] > 0 for result in marked)


def test_empty_not_ok_cells_and_printed_borders_remain_unmarked():
    image = _synthetic_checklist()
    geometry = detect_checklist_rows(image, perspective=False)
    row = geometry.rows[1]
    x1, y1, x2, y2 = row.not_ok_bbox
    # Re-emphasize the printed box edges to exercise border suppression.
    cv2.line(image, (x1, y1), (x2, y1), (15, 15, 15), 2)
    cv2.line(image, (x1, y2 - 1), (x2, y2 - 1), (15, 15, 15), 2)
    cv2.line(image, (x1, y1), (x1, y2), (15, 15, 15), 2)
    cv2.line(image, (x2 - 1, y1), (x2 - 1, y2), (15, 15, 15), 2)
    geometry = replace(geometry, corrected_image=image)
    result = detect_not_ok_marks(image, geometry)[row.row_index]
    assert result.not_ok_marked is False
    assert result.evidence["border_excluded"] is True


@pytest.mark.parametrize("pen_color", [(20, 20, 20), (170, 45, 20), (25, 25, 180)])
def test_detects_visible_pen_marks_without_colour_specific_channels(pen_color):
    image = _synthetic_checklist()
    geometry = detect_checklist_rows(image, perspective=False)
    target = geometry.rows[1]
    _draw_mark(image, target.not_ok_bbox, pen_color)
    geometry = replace(geometry, corrected_image=image)
    result = detect_not_ok_marks(image, geometry)[target.row_index]
    assert result.not_ok_marked is True
    assert result.evidence["components"] >= 1


def test_mark_in_ok_cell_does_not_mark_not_ok_cell():
    image = _synthetic_checklist()
    geometry = detect_checklist_rows(image, perspective=False)
    target = geometry.rows[1]
    nok_x1 = target.not_ok_bbox[0]
    ok_box = next(box for box in target.column_geometry.values() if box[2] == nok_x1)
    _draw_mark(image, ok_box, (30, 70, 150))
    geometry = replace(geometry, corrected_image=image)
    results = detect_not_ok_marks(image, geometry)
    assert results[target.row_index].not_ok_marked is False


def test_resized_and_jpeg_variants_preserve_real_mark_detection(real_runs):
    for path, image, geometry, original_results in real_runs:
        variants = [cv2.resize(image, None, fx=0.91, fy=0.91, interpolation=cv2.INTER_AREA)]
        encoded_ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 82])
        assert encoded_ok
        variants.append(cv2.imdecode(encoded, cv2.IMREAD_COLOR))
        for variant in variants:
            variant_geometry = detect_checklist_rows(variant)
            variant_results = detect_not_ok_marks(variant, variant_geometry)
            assert len(variant_results) == len(variant_geometry.rows)
            assert any(result.not_ok_marked for result in variant_results), path.name
            assert len(variant_results) >= int(len(original_results) * 0.55)

