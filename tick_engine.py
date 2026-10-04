"""Colour-independent handwritten mark detection for detected NOT OK cells.

This module consumes row geometry from :mod:`row_engine`; it does not locate
columns, read observations, or classify marks in any other checklist cell.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

import cv2
import numpy as np


@dataclass(frozen=True)
class TickResult:
    row_index: int
    not_ok_marked: bool
    confidence: float
    evidence: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _gray(image: np.ndarray) -> np.ndarray:
    if image is None or image.size == 0:
        raise ValueError("image must be a non-empty OpenCV image")
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    raise ValueError("image must be grayscale, BGR, or BGRA")


def _rows_and_image(image: np.ndarray, row_geometry: Any):
    # RowDetection keeps the perspective-corrected image with its coordinates.
    # Prefer it when present so callers cannot accidentally apply corrected
    # coordinates to the unwarped source photo.
    corrected = getattr(row_geometry, "corrected_image", None)
    rows = getattr(row_geometry, "rows", None)
    if corrected is not None and rows is not None:
        return corrected, [(row.row_index, row.not_ok_bbox) for row in rows]

    if not isinstance(row_geometry, dict) or "rows" not in row_geometry:
        raise TypeError("row_geometry must be a RowDetection or its dictionary result")
    corrected = row_geometry.get("corrected_image", image)
    rows = row_geometry["rows"]
    parsed = []
    for row in rows:
        index = row["row_index"]
        bbox = row.get("not_ok_bbox")
        if bbox is None:
            raise ValueError(f"row {index} is missing NOT OK cell geometry")
        parsed.append((index, tuple(int(v) for v in bbox)))
    return corrected, parsed


def _analyze_cell(gray: np.ndarray, bbox: tuple[int, int, int, int]) -> tuple[bool, float, dict[str, Any]]:
    x1, y1, x2, y2 = bbox
    height, width = gray.shape
    x1, x2 = max(0, x1), min(width, x2)
    y1, y2 = max(0, y1), min(height, y2)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("NOT OK cell bounding box is empty or outside the image")

    cell = gray[y1:y2, x1:x2]
    cell_h, cell_w = cell.shape
    # Keep the central cell interior; these margins scale with detected
    # geometry and suppress the printed table rules around the cell.
    pad_x = max(1, round(cell_w * 0.075))
    pad_y = max(1, round(cell_h * 0.12))
    interior = cell[pad_y:max(pad_y + 1, cell_h - pad_y),
                    pad_x:max(pad_x + 1, cell_w - pad_x)]
    if interior.size == 0:
        return False, 0.99, {"ink_pixels": 0, "components": 0, "stroke_score": 0.0,
                             "border_excluded": True, "component_boxes": []}

    # Adaptive grayscale thresholding is independent of pen channel/colour.
    block = min(31, max(9, (min(interior.shape) // 2) * 2 - 1))
    if block % 2 == 0:
        block -= 1
    if min(interior.shape) < block:
        block = min(interior.shape) if min(interior.shape) % 2 else min(interior.shape) - 1
    block = max(3, block)
    binary = cv2.adaptiveThreshold(interior, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, block, 7)

    # Remove long printed rules if any remain in the inner crop. The kernels
    # scale from this row's cell geometry, not image coordinates.
    hk = max(7, round(interior.shape[1] * 0.55))
    vk = max(7, round(interior.shape[0] * 0.58))
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN,
                                  cv2.getStructuringElement(cv2.MORPH_RECT, (min(hk, interior.shape[1]), 1)))
    vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN,
                                cv2.getStructuringElement(cv2.MORPH_RECT, (1, min(vk, interior.shape[0]))))
    rules = cv2.bitwise_or(horizontal, vertical)
    clean = cv2.bitwise_and(binary, cv2.bitwise_not(rules))

    count, labels, stats, _ = cv2.connectedComponentsWithStats(clean, 8)
    area = int(interior.shape[0] * interior.shape[1])
    min_component_area = max(5, round(area * 0.003))
    components = []
    candidate_components = []
    total_component_ink = 0
    for component_id in range(1, count):
        cx, cy, cw, ch, component_area = map(int, stats[component_id])
        if component_area < min_component_area or (cw < 2 and ch < 3):
            continue
        components.append((cx, cy, cw, ch, component_area))
        # Require a coherent stroke-sized component. Thin horizontal/vertical
        # remnants from printed rules and small glyph fragments in column
        # headings are not pen marks.
        spans_cell = (cw >= max(6, round(interior.shape[1] * 0.25)) and
                      ch >= max(6, round(interior.shape[0] * 0.20)))
        if spans_cell:
            candidate_components.append((cx, cy, cw, ch, component_area))
            total_component_ink += component_area

    raw_ink_pixels = int(cv2.countNonZero(clean))
    ink_pixels = int(total_component_ink)
    density = ink_pixels / max(1, area)
    largest_area = max((component[4] for component in candidate_components), default=0)
    # A tick is typically a multi-stroke or diagonal connected component.
    # Keep the decision permissive for different handwriting while requiring
    # both ink amount and component evidence, which rejects isolated specks.
    size_score = min(1.0, total_component_ink / max(18.0, area * 0.045))
    density_score = min(1.0, density / 0.035)
    shape_score = min(1.0, largest_area / max(10.0, area * 0.018))
    stroke_score = 0.45 * density_score + 0.35 * size_score + 0.20 * shape_score
    marked = ink_pixels >= max(14, round(area * 0.008)) and bool(candidate_components)
    confidence = float(np.clip(0.50 + 0.48 * stroke_score if marked else 0.98 - 0.35 * stroke_score,
                               0.0, 0.99))
    component_boxes = [(x1 + pad_x + cx, y1 + pad_y + cy,
                        x1 + pad_x + cx + cw, y1 + pad_y + cy + ch)
                       for cx, cy, cw, ch, _ in candidate_components]
    evidence = {
        "ink_pixels": ink_pixels,
        "raw_ink_pixels": raw_ink_pixels,
        "components": len(candidate_components),
        "all_ink_components": len(components),
        "largest_component_area": largest_area,
        "stroke_score": round(float(stroke_score), 4),
        "ink_density": round(float(density), 4),
        "border_excluded": True,
        "component_boxes": component_boxes,
    }
    return marked, confidence, evidence


def detect_not_ok_marks(image: np.ndarray, row_geometry: Any) -> list[TickResult]:
    """Return mark evidence for each row's NOT OK cell only.

    Coordinates are read from ``row_geometry`` output produced by
    ``row_engine.detect_checklist_rows``. For a ``RowDetection`` object its
    associated corrected image is used automatically.
    """
    source, rows = _rows_and_image(image, row_geometry)
    gray = _gray(source)
    results = []
    for row_index, bbox in rows:
        marked, confidence, evidence = _analyze_cell(gray, bbox)
        evidence["cell_bbox"] = tuple(bbox)
        results.append(TickResult(int(row_index), marked, confidence, evidence))
    return results


def detect_ticks(image: np.ndarray, row_geometry: Any) -> list[dict[str, Any]]:
    """Dictionary-returning convenience entry point."""
    return [result.as_dict() for result in detect_not_ok_marks(image, row_geometry)]
