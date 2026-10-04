"""Geometry-only checklist table and row detection.

Coordinates in the returned result refer to ``corrected_image``. No OCR or
mark/tick classification is performed here.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import cv2
import numpy as np


Box = tuple[int, int, int, int]  # x1, y1, x2, y2 (right/bottom exclusive)


@dataclass(frozen=True)
class ChecklistRow:
    row_index: int
    bbox: Box
    boundaries: tuple[int, int]
    column_geometry: dict[str, Box]
    not_ok_bbox: Box


@dataclass(frozen=True)
class RowDetection:
    corrected_image: np.ndarray
    table_bbox: Box
    column_boundaries: tuple[int, ...]
    rows: tuple[ChecklistRow, ...]
    perspective_corrected: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "corrected_image": self.corrected_image,
            "table_bbox": self.table_bbox,
            "column_boundaries": self.column_boundaries,
            "perspective_corrected": self.perspective_corrected,
            "rows": [asdict(row) for row in self.rows],
        }


def _gray(image: np.ndarray) -> np.ndarray:
    if image is None or image.size == 0:
        raise ValueError("image must be a non-empty OpenCV image")
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _quad_warp(image: np.ndarray) -> tuple[np.ndarray, bool]:
    """Rectify a dominant, near-page quadrilateral when one is visible."""
    h, w = image.shape[:2]
    gray = _gray(image)
    small_scale = min(1.0, 1200.0 / max(h, w))
    small = cv2.resize(gray, None, fx=small_scale, fy=small_scale) if small_scale < 1 else gray
    edges = cv2.Canny(cv2.GaussianBlur(small, (5, 5), 0), 40, 130)
    contours, _ = cv2.findContours(cv2.dilate(edges, np.ones((3, 3), np.uint8)), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Page boundary detection is intentionally conservative; table line
    # geometry remains the fallback for cropped photographs.
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:8]:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) != 4 or cv2.contourArea(approx) < small.shape[0] * small.shape[1] * 0.20:
            continue
        pts = approx.reshape(4, 2).astype(np.float32) / small_scale
        s = pts.sum(axis=1)
        d = np.diff(pts, axis=1).ravel()
        ordered = np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]], np.float32)
        tl, tr, br, bl = ordered
        out_w = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
        out_h = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
        if out_w < w * 0.3 or out_h < h * 0.3:
            continue
        target = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], np.float32)
        matrix = cv2.getPerspectiveTransform(ordered, target)
        return cv2.warpPerspective(image, matrix, (out_w, out_h)), True
    return image.copy(), False


def _line_masks(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = gray.shape
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 31, 12)
    horizontal = cv2.morphologyEx(bw, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, w // 35), 1)))
    vertical = cv2.morphologyEx(bw, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(15, h // 45))))
    return horizontal, vertical


def _runs(values: np.ndarray, threshold: float) -> list[tuple[int, int]]:
    active = values >= threshold
    edges = np.diff(np.r_[False, active, False].astype(np.int8))
    starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]
    return [(int(a), int(b)) for a, b in zip(starts, ends)]


def _merge_centers(runs: list[tuple[int, int]], gap: int) -> list[int]:
    groups: list[list[tuple[int, int]]] = []
    for run in runs:
        if not groups or run[0] - groups[-1][-1][1] > gap:
            groups.append([run])
        else:
            groups[-1].append(run)
    return [round((g[0][0] + g[-1][1]) / 2) for g in groups]


def _bounds(mask: np.ndarray, axis: int, minimum_coverage: float) -> list[int]:
    h, w = mask.shape
    projection = (mask > 0).mean(axis=axis)
    extent = w if axis == 0 else h
    runs = _runs(projection, minimum_coverage)
    return _merge_centers(runs, max(2, extent // 300))


def _find_table(gray: np.ndarray, horizontal: np.ndarray) -> Box:
    h, w = gray.shape
    y_lines = _bounds(horizontal, 1, 0.08)
    if len(y_lines) < 3:
        raise ValueError("checklist table grid was not detected")
    # Choose the densest span of horizontal rules; this excludes the page
    # metadata and section headings where possible.
    best = None
    for i, top in enumerate(y_lines):
        for bottom in y_lines[i + 2:]:
            if bottom - top < h * 0.12:
                continue
            count = sum(top <= y <= bottom for y in y_lines)
            score = count / max(1, bottom - top) ** 0.25
            if best is None or score > best[0]:
                best = (score, top, bottom)
    if best is None:
        raise ValueError("checklist table boundary was not detected")
    _, y1, y2 = best
    # Caller supplies the table's x extent from grid coverage; the fallback
    # is the union of horizontal strokes within the selected table span.
    band = horizontal[max(0, y1 - 2):min(h, y2 + 3)]
    xs = np.where((band > 0).mean(axis=0) > 0.015)[0]
    if not len(xs):
        raise ValueError("table horizontal boundaries were not detected")
    return max(0, int(xs[0])), int(y1), min(w, int(xs[-1]) + 1), int(y2)


def detect_checklist_rows(image: np.ndarray, *, perspective: bool = True,
                          min_rows: int = 1) -> RowDetection:
    """Detect checklist geometry from a BGR, BGRA, or grayscale image.

    ``NOT OK`` is located from the second of two similarly sized response
    columns that precede a wider trailing comments column. This uses observed
    grid geometry, without OCR or fixed image coordinates/ratios.
    """
    if min_rows < 1:
        raise ValueError("min_rows must be at least 1")
    corrected, did_warp = _quad_warp(image) if perspective else (image.copy(), False)
    gray = _gray(corrected)
    horizontal, vertical = _line_masks(gray)
    x1, y1, x2, y2 = _find_table(gray, horizontal)
    h, w = gray.shape
    # Detect vertical rules by accumulating their slightly drifting strokes
    # across the table height; a high full-height threshold loses slanted rules.
    vband = vertical[y1:y2 + 1, x1:x2]
    x_projection = (vband > 0).mean(axis=0)
    candidate_runs = [run for run in _runs(x_projection, 0.008) if run[1] - run[0] >= 3]
    candidates = _merge_centers(candidate_runs, max(8, w // 45))
    x_lines = [x1 + center for center in candidates]
    if len(x_lines) >= 6:
        x1, x2 = x_lines[0], x_lines[-1]
    else:
        # When the projected vertical mask misses an outer grid rule (common
        # on clean synthetic grids), retain the independently detected table
        # edges and use projection peaks for the interior dividers.
        x_lines = [x1, *[x1 + x for x in _bounds(vband, 0, 0.20)], x2]
        tolerance = max(2, w // 300)
        clustered: list[list[int]] = []
        for x in x_lines:
            if clustered and x - clustered[-1][-1] <= tolerance:
                clustered[-1].append(x)
            else:
                clustered.append([x])
        x_lines = [round(float(np.mean(group))) for group in clustered]
        x1, x2 = x_lines[0], x_lines[-1]
    if len(x_lines) < 4:
        raise ValueError("checklist columns were not detected")

    # Select the compact adjacent response columns before a wider trailing
    # comments cell, based only on this page's detected grid geometry.
    widths = np.diff(x_lines)
    if len(widths) < 4:
        raise ValueError("response columns were not detected")
    pairs = []
    for i in range(1, len(widths) - 2):
        left_w, right_w, trailing_w = widths[i:i + 3]
        similar = max(left_w, right_w) / max(1, min(left_w, right_w)) < 2.1
        if similar and trailing_w >= max(left_w, right_w) * 1.18:
            pairs.append((i, trailing_w / max(1, left_w + right_w)))
    if not pairs:
        raise ValueError("OK and NOT OK response columns were not identified")
    pair_idx = max(pairs, key=lambda item: (item[0], item[1]))[0]
    nok_idx = pair_idx + 1
    nok_left, nok_right = x_lines[nok_idx], x_lines[nok_idx + 1]

    local_h = horizontal[y1:y2 + 1, x1:x2]
    row_y = _bounds(local_h, 1, 0.08)
    row_y = sorted(set([y1, *[y1 + y for y in row_y if y > 1 and y < y2 - y1 - 1], y2]))
    # Remove section/header bands by keeping the stable short-pitch run of rows
    # that contains the greatest number of intervals.
    intervals = [(a, b) for a, b in zip(row_y, row_y[1:]) if b - a >= max(5, h // 250)]
    if not intervals:
        raise ValueError("checklist rows were not detected")
    heights = np.array([b - a for a, b in intervals])
    typical = float(np.median(heights))
    data_intervals = [(a, b) for a, b in intervals if (b - a) <= typical * 1.8]
    if len(data_intervals) < min_rows:
        raise ValueError(f"only {len(data_intervals)} checklist rows detected")

    rows = []
    for idx, (top, bottom) in enumerate(data_intervals):
        cols = {f"column_{i}": (x_lines[i], top, x_lines[i + 1], bottom)
                for i in range(len(x_lines) - 1)}
        rows.append(ChecklistRow(idx, (x1, top, x2, bottom), (top, bottom), cols,
                                 (nok_left, top, nok_right, bottom)))
    return RowDetection(corrected, (x1, y1, x2, y2), tuple(x_lines), tuple(rows), did_warp)


def detect_rows(image: np.ndarray, **kwargs: Any) -> dict[str, Any]:
    """Dictionary-returning convenience entry point."""
    return detect_checklist_rows(image, **kwargs).as_dict()
