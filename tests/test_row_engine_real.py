"""Validation against real checklist photos shipped with the project."""
from pathlib import Path

import cv2
import numpy as np
import pytest

from row_engine import RowDetection, detect_checklist_rows


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = ROOT / "sample_photos"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _sample_images():
    return sorted(path for path in SAMPLE_DIR.iterdir()
                  if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def _variants(image):
    h, w = image.shape[:2]
    yield "resize", cv2.resize(image, None, fx=0.91, fy=0.91, interpolation=cv2.INTER_AREA)

    rotation = cv2.getRotationMatrix2D((w / 2, h / 2), 0.5, 1.0)
    yield "rotation", cv2.warpAffine(image, rotation, (w, h), borderValue=(245, 245, 245))

    source = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    target = np.float32([[0.008 * w, 0.006 * h], [0.993 * w, 0.002 * h],
                         [0.997 * w, 0.994 * h], [0.003 * w, 0.998 * h]])
    perspective = cv2.getPerspectiveTransform(source, target)
    yield "perspective", cv2.warpPerspective(image, perspective, (w, h), borderValue=(245, 245, 245))

    margin_x, margin_y = max(1, w // 100), max(1, h // 100)
    crop = image[margin_y:h - margin_y, margin_x:w - margin_x]
    yield "crop", cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)

    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 82])
    assert ok, "OpenCV could not encode the JPEG quality variant"
    yield "jpeg_quality", cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def _assert_geometry(image, result):
    corrected_h, corrected_w = result.corrected_image.shape[:2]
    x1, y1, x2, y2 = result.table_bbox
    assert 0 <= x1 < x2 <= corrected_w
    assert 0 <= y1 < y2 <= corrected_h

    columns = result.column_boundaries
    assert len(columns) >= 6, "expected detected boundaries for checklist and response columns"
    assert all(0 <= a < b <= corrected_w for a, b in zip(columns, columns[1:]))
    assert columns[0] >= x1 - 2 and columns[-1] <= x2 + 2
    assert len(result.rows) >= 3

    previous_bottom = None
    for row in result.rows:
        rx1, ry1, rx2, ry2 = row.bbox
        nx1, ny1, nx2, ny2 = row.not_ok_bbox
        assert x1 - 2 <= rx1 < rx2 <= x2 + 2
        assert y1 - 2 <= ry1 < ry2 <= y2 + 2
        assert previous_bottom is None or ry1 >= previous_bottom
        previous_bottom = ry2
        assert nx1 < nx2 and ny1 == ry1 and ny2 == ry2
        assert x1 <= nx1 < nx2 <= x2
        assert len(row.column_geometry) >= 5
        # The NOT OK cell must be one of the cells represented by the
        # detected column boundaries, rather than a hard-coded x coordinate.
        assert any(abs(nx1 - left) <= 2 and abs(nx2 - right) <= 2
                   for left, right in zip(columns, columns[1:]))


def _debug_overlay(result: RowDetection, destination: Path):
    """Write a visual inspection image; kept out of row_engine production code."""
    canvas = result.corrected_image.copy()
    x1, y1, x2, y2 = result.table_bbox
    cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 180, 0), 3)
    for x in result.column_boundaries:
        cv2.line(canvas, (x, y1), (x, y2), (255, 120, 0), 2)
    for row in result.rows:
        rx1, ry1, rx2, ry2 = row.bbox
        cv2.rectangle(canvas, (rx1, ry1), (rx2, ry2), (0, 170, 255), 1)
        nx1, ny1, nx2, ny2 = row.not_ok_bbox
        cv2.rectangle(canvas, (nx1, ny1), (nx2, ny2), (220, 0, 220), 2)
    destination.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(destination), canvas), "could not save the geometry debug image"


@pytest.fixture(scope="module")
def real_samples():
    paths = _sample_images()
    assert paths, f"no checklist image samples found in {SAMPLE_DIR}"
    loaded = []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        assert image is not None, f"sample image could not be loaded: {path.name}"
        assert image.ndim == 3 and image.shape[0] > 0 and image.shape[1] > 0
        loaded.append((path, image))
    return loaded


@pytest.fixture(scope="module")
def real_detections(real_samples):
    detections = []
    for path, image in real_samples:
        result = detect_checklist_rows(image)
        detections.append((path, image, result))
    _debug_overlay(detections[0][2], ROOT / "tests" / "artifacts" / "row_engine_real_debug.png")
    return detections


def test_real_images_load_with_valid_dimensions(real_samples):
    assert real_samples
    for path, image in real_samples:
        assert image.shape[0] > 1 and image.shape[1] > 1, path.name


def test_real_table_columns_rows_and_not_ok_geometry(real_detections):
    for path, image, result in real_detections:
        try:
            _assert_geometry(image, result)
        except AssertionError as error:
            raise AssertionError(f"{path.name}: {error}") from error


def test_real_geometry_remains_structurally_stable_under_small_variants(real_samples, real_detections):
    by_path = {path: result for path, _, result in real_detections}
    for path, image in real_samples:
        baseline = by_path[path]
        base_width = baseline.table_bbox[2] - baseline.table_bbox[0]
        for variant_name, variant in _variants(image):
            result = detect_checklist_rows(variant)
            _assert_geometry(variant, result)
            width = result.table_bbox[2] - result.table_bbox[0]
            assert 0.65 <= width / base_width <= 1.35, f"{path.name} {variant_name}: table width changed unexpectedly"
            assert len(result.rows) >= max(3, int(len(baseline.rows) * 0.55)), (
                f"{path.name} {variant_name}: too few rows detected"
            )
            assert len(result.column_boundaries) >= len(baseline.column_boundaries) - 1, (
                f"{path.name} {variant_name}: column structure was lost"
            )
