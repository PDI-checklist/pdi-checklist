"""Image-grounded OCR for observation text inside detected checklist rows.

This module uses row/column geometry and never consults a master observation
list. Only the selected row's Observation Point cell is sent to OCR.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import re
import unicodedata
from difflib import SequenceMatcher
from typing import Any, Callable, Iterable, Mapping

import cv2
import numpy as np


Box = tuple[int, int, int, int]
OCRCallable = Callable[[np.ndarray], Mapping[str, Any]]
_WORD_RE = re.compile(r"[^\w\-/().,&%+:]+", re.UNICODE)
_CACHE_LIMIT = 256


@dataclass(frozen=True)
class ObservationResult:
    row_index: int
    text: str
    confidence: float
    status: str
    bbox: Box | None
    source: str = "image_ocr"
    image_reference: str | None = None
    row_bbox: Box | None = None
    evidence_crop: np.ndarray | None = None
    evidence: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class ObservationOCRCache:
    """Small per-image OCR result cache, suitable for one upload batch."""

    def __init__(self):
        self._results: dict[tuple[Any, ...], ObservationResult] = {}
        self._source_geometry: dict[tuple[Any, ...], tuple[np.ndarray, Any] | None] = {}

    def get(self, key: tuple[Any, ...]) -> ObservationResult | None:
        return self._results.get(key)

    def put(self, key: tuple[Any, ...], value: ObservationResult) -> None:
        if len(self._results) >= _CACHE_LIMIT:
            self._results.pop(next(iter(self._results)))
        self._results[key] = value


def _source_geometry(image: np.ndarray, corrected: np.ndarray,
                     cache: ObservationOCRCache | None = None):
    """Recover one corrected-to-source map and source grid for this image."""
    if image is corrected or image.shape[:2] == corrected.shape[:2] and np.shares_memory(image, corrected):
        return None
    key = (id(image), id(corrected), image.shape[:2], corrected.shape[:2])
    if cache is not None and key in cache._source_geometry:
        return cache._source_geometry[key]
    result = None
    try:
        from row_engine import detect_checklist_rows
        source_gray, corrected_gray = _gray(image), _gray(corrected)
        orb = cv2.ORB_create(nfeatures=4000, edgeThreshold=15, fastThreshold=8)
        source_points, source_descriptors = orb.detectAndCompute(source_gray, None)
        corrected_points, corrected_descriptors = orb.detectAndCompute(corrected_gray, None)
        if source_descriptors is not None and corrected_descriptors is not None:
            matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(
                corrected_descriptors, source_descriptors, k=2)
            good = [first for first, second in matches
                    if first.distance < 0.72 * second.distance]
            if len(good) >= 12:
                matrix, inliers = cv2.findHomography(
                    np.float32([corrected_points[item.queryIdx].pt for item in good]),
                    np.float32([source_points[item.trainIdx].pt for item in good]),
                    cv2.RANSAC, 4.0)
                if matrix is not None and inliers is not None and int(inliers.sum()) >= 12:
                    raw_geometry = detect_checklist_rows(image, perspective=False)
                    result = (matrix, raw_geometry)
    except Exception:
        result = None
    if cache is not None:
        cache._source_geometry[key] = result
    return result


def _image_and_rows(image: np.ndarray, row_geometry: Any):
    corrected = getattr(row_geometry, "corrected_image", None)
    rows = getattr(row_geometry, "rows", None)
    if corrected is not None and rows is not None:
        return corrected, list(rows)
    if isinstance(row_geometry, Mapping) and "rows" in row_geometry:
        return row_geometry.get("corrected_image", image), list(row_geometry["rows"])
    if isinstance(row_geometry, (list, tuple)):
        return image, list(row_geometry)
    raise TypeError("row_geometry must be RowDetection, its dictionary result, or a row sequence")


def _value(row: Any, key: str, default=None):
    return row.get(key, default) if isinstance(row, Mapping) else getattr(row, key, default)


def _box(value: Any) -> Box | None:
    if isinstance(value, Mapping):
        value = value.get("bbox", value.get("box"))
    elif hasattr(value, "bbox"):
        value = value.bbox
    if value is None or len(value) != 4:
        return None
    return tuple(int(v) for v in value)


def _find_observation_box(row: Any, explicit: Any = None) -> Box | None:
    if explicit is not None:
        if isinstance(explicit, Mapping) and not any(k in explicit for k in ("bbox", "box")):
            explicit = explicit.get(_value(row, "row_index"), explicit.get(str(_value(row, "row_index"))))
        return _box(explicit)

    columns = _value(row, "column_geometry", {}) or {}
    not_ok = _box(_value(row, "not_ok_bbox"))
    if not_ok is None or not columns:
        return None
    ordered = sorted((_box(box) for box in columns.values() if _box(box) is not None), key=lambda b: b[0])
    if len(ordered) < 4:
        return None
    nok_idx = min(range(len(ordered)), key=lambda i: abs(ordered[i][0] - not_ok[0]) + abs(ordered[i][2] - not_ok[2]))
    if abs(ordered[nok_idx][0] - not_ok[0]) > 3 or abs(ordered[nok_idx][2] - not_ok[2]) > 3:
        return None
    # The response pair is located from the detected NOT OK cell. The text
    # cell is the widest detected column to the left of the OK response cell.
    # This uses the page's actual grid and remains independent of wording.
    ok_idx = nok_idx - 1
    if ok_idx < 1:
        return None
    candidates = ordered[:ok_idx]
    return max(candidates, key=lambda b: b[2] - b[0])


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


def _clean_text(tokens: Iterable[str]) -> str:
    normalized = unicodedata.normalize("NFKC", " ".join(str(t) for t in tokens))
    normalized = normalized.replace("\u00a0", " ").replace("|", " ")
    normalized = _WORD_RE.sub(" ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip(" .,:;|~_")
    # Remove isolated OCR punctuation while preserving meaningful internal
    # punctuation and all readable wording from the checklist image.
    normalized = re.sub(r"(?<!\w)[\-/().,&%+:]+(?!\w)", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _odd_block(shape: tuple[int, int], desired: int = 31) -> int:
    limit = min(shape)
    block = min(desired, limit if limit % 2 else limit - 1)
    return max(3, block if block % 2 else block - 1)


def _horizontal_rule_centers(gray: np.ndarray) -> list[int]:
    """Find long horizontal table rules in a local observation-column band."""
    if gray.size == 0 or gray.shape[1] < 30:
        return []
    block = _odd_block(gray.shape)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, block, 10)
    kernel_width = min(gray.shape[1], max(15, round(gray.shape[1] * 0.25)))
    lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 1)))
    coverage = (lines > 0).mean(axis=1)
    active = coverage >= 0.16
    edges = np.diff(np.r_[False, active, False].astype(np.int8))
    starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]
    centers = []
    for start, end in zip(starts, ends):
        weights = coverage[start:end]
        centers.append(int(round(start + float(np.average(np.arange(end - start), weights=weights)))))
    return centers


def _refine_row_vertical_bounds(gray: np.ndarray, row: Any, rows: list[Any],
                                x1: int, x2: int) -> tuple[int, int]:
    """Use nearby rules in the observation column to refine a short row box.

    Refinement is accepted only when a pair of detected rules encloses the
    geometry row center, overlaps most of its original interval, and remains
    close to the row pitch estimated from neighboring detected rows.
    """
    bounds = _box(_value(row, "bbox"))
    if bounds is None:
        return 0, gray.shape[0]
    _, y1, _, y2 = bounds
    height = max(1, y2 - y1)
    idx = int(_value(row, "row_index", -1))
    neighbor_heights = []
    for other in rows:
        other_idx = int(_value(other, "row_index", -1))
        other_box = _box(_value(other, "bbox"))
        if other_box is not None and 0 < abs(other_idx - idx) <= 3:
            neighbor_heights.append(max(1, other_box[3] - other_box[1]))
    pitch = float(np.median(neighbor_heights)) if neighbor_heights else float(height)
    radius = max(height * 1.5, pitch * 2.6)
    search_top, search_bottom = max(0, int(round((y1 + y2) / 2 - radius))), min(
        gray.shape[0], int(round((y1 + y2) / 2 + radius)))
    left, right = max(0, x1), min(gray.shape[1], x2)
    centers = _horizontal_rule_centers(gray[search_top:search_bottom, left:right])
    centers = [search_top + value for value in centers]
    center = (y1 + y2) / 2
    max_span = max(height * 2.6, pitch * 2.6)
    tolerance = max(2, int(round(height * 0.08)))
    candidates = []
    for top, bottom in zip(centers, centers[1:]):
        span = bottom - top
        overlap = max(0, min(y2, bottom) - max(y1, top)) / height
        encloses_row = top <= y1 + tolerance and bottom >= y2 - tolerance
        if (top <= center <= bottom and encloses_row and overlap >= 0.70
                and span <= max_span):
            score = (overlap, -abs(span - max(height, pitch)), -abs((top + bottom) / 2 - center))
            candidates.append((score, top, bottom))
    if not candidates:
        return y1, y2
    _, top, bottom = max(candidates, key=lambda candidate: candidate[0])
    # Never shrink the engine-provided row interval; local rules may only
    # expand a short or skewed crop while retaining its detected text band.
    return max(0, min(y1, top)), min(gray.shape[0], max(y2, bottom))


def _rule_masks(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return long grid-rule masks; glyph-sized strokes are below kernels."""
    if gray.size == 0:
        return np.zeros_like(gray), np.zeros_like(gray)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, _odd_block(gray.shape), 10)
    h_kernel = min(gray.shape[1], max(15, round(gray.shape[1] * 0.25)))
    v_kernel = min(gray.shape[0], max(5, round(gray.shape[0] * 0.68)))
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel, 1)))
    vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kernel)))
    # Perspective can slant a cell edge enough to evade a straight vertical
    # opening. Suppress only narrow, tall connected components that touch the
    # crop boundary; text begins inside the observation cell.
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    edge_limit = max(5, int(round(gray.shape[1] * 0.08)))
    for component_id in range(1, component_count):
        cx, cy, cw, ch, _ = map(int, stats[component_id])
        touches_edge = cx <= 1 or cx + cw >= gray.shape[1] - 1
        if touches_edge and ch >= max(5, round(gray.shape[0] * 0.45)) and cw <= edge_limit:
            vertical[labels == component_id] = 255
    return horizontal, vertical


def _trim_trailing_blank(gray: np.ndarray) -> np.ndarray:
    """Trim only the blank right tail, retaining a scale-relative text margin."""
    if gray.size == 0 or gray.shape[1] < 2:
        return gray
    horizontal, vertical = _rule_masks(gray)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, _odd_block(gray.shape), 8)
    rules = cv2.bitwise_or(horizontal, vertical)
    content = cv2.bitwise_and(binary, cv2.bitwise_not(rules))
    counts = (content > 0).sum(axis=0)
    active = counts >= max(3, int(round(gray.shape[0] * 0.18)))
    edges = np.diff(np.r_[False, active, False].astype(np.int8))
    runs = list(zip(np.where(edges == 1)[0], np.where(edges == -1)[0]))
    if not runs:
        return gray
    # Join the short gaps between adjacent glyph strokes/words, but keep
    # isolated specks and table-border fragments out of the trailing extent.
    join_gap = max(3, int(round(gray.shape[0] * 0.55)))
    groups: list[list[int]] = []
    for start, end in runs:
        if not groups or start - groups[-1][1] > join_gap:
            groups.append([start, end])
        else:
            groups[-1][1] = end
    substantial = [group for group in groups
                   if group[1] - group[0] >= max(5, int(round(gray.shape[0] * 0.5)))]
    if not substantial:
        return gray
    text_group = max(substantial,
                     key=lambda group: int(counts[group[0]:group[1]].sum()))
    # A row-height margin preserves punctuation and final narrow glyphs while
    # dropping the large blank remainder of wide observation cells.
    margin = max(4, int(round(gray.shape[0] * 0.85)))
    end = min(gray.shape[1], text_group[1] + margin)
    return gray[:, :max(end, min(gray.shape[1], margin * 2))]


def _resize_to_text_height(gray: np.ndarray, target: int = 72) -> np.ndarray:
    if gray.shape[0] >= target:
        return gray
    scale = target / max(1, gray.shape[0])
    return cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def _preprocessing_candidates(region: np.ndarray,
                              expanded_vertical: np.ndarray | None = None
                              ) -> list[tuple[str, np.ndarray]]:
    """Return five bounded crops covering geometry and image-quality options."""
    current = _trim_trailing_blank(_gray(region))
    expanded = _trim_trailing_blank(_gray(expanded_vertical)) if expanded_vertical is not None else current
    normalized = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 2)).apply(current)
    normalized_expanded = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 2)).apply(expanded)

    adaptive = cv2.adaptiveThreshold(current, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, _odd_block(current.shape), 8)
    horizontal, vertical = _rule_masks(current)
    rules = cv2.bitwise_or(horizontal, vertical)
    rule_suppressed = cv2.bitwise_not(cv2.bitwise_and(adaptive,
                                                       cv2.bitwise_not(rules)))

    mild = cv2.resize(current, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    return [
        ("current_crop", _resize_to_text_height(current)),
        ("expanded_vertical", _resize_to_text_height(normalized_expanded)),
        ("rule_suppressed", _resize_to_text_height(rule_suppressed)),
        ("mild_upscale", mild),
        ("contrast_normalized", _resize_to_text_height(normalized)),
    ]


def _tesseract_reader(image: np.ndarray, psm: int = 7) -> Mapping[str, Any]:
    import pytesseract
    from pytesseract import Output
    return pytesseract.image_to_data(image, config=f"--psm {psm} -c preserve_interword_spaces=1",
                                     output_type=Output.DICT)


def _parse_ocr(data: Mapping[str, Any]) -> tuple[str, float, list[dict[str, Any]]]:
    words = data.get("text", [])
    confidences = data.get("conf", [])
    accepted = []
    accepted_conf = []
    details = []
    for index, word in enumerate(words):
        word = str(word).strip()
        if not word:
            continue
        try:
            confidence = float(confidences[index])
        except (IndexError, TypeError, ValueError):
            confidence = -1.0
        if confidence >= 0:
            accepted.append(word)
            accepted_conf.append(confidence)
            details.append({"word": word, "confidence": round(confidence, 1)})
    text = _clean_text(accepted)
    confidence = float(np.mean(accepted_conf) / 100.0) if accepted_conf else 0.0
    return text, confidence, details


def _candidate_score(text: str, confidence: float,
                     details: list[dict[str, Any]]) -> float:
    """Rank evidence without comparing it with a checklist/master vocabulary."""
    if not text:
        return -1.0
    count = max(1, len(details))
    low_confidence = sum(item["confidence"] < 35 for item in details) / count
    one_character = sum(len(item["word"]) == 1 for item in details) / count
    readable = sum(char.isalnum() for char in text) / max(1, len(text))
    edge_artifact = bool(re.match(r"^[^\w]+\w", text, flags=re.UNICODE))
    return (0.76 * confidence + 0.10 * readable + 0.14 * min(1.0, len(text) / 36)
            - 0.16 * low_confidence - 0.10 * one_character
            - (0.16 if edge_artifact else 0.0))


def _text_agreement(left: str, right: str) -> float:
    left = re.sub(r"\W+", "", left.casefold())
    right = re.sub(r"\W+", "", right.casefold())
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right).ratio()


def _select_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Rank by OCR evidence and peer agreement, without checklist vocabulary."""
    usable = [item for item in candidates if item.get("text")]
    if not usable:
        return None
    ranked = []
    for item in usable:
        peer_scores = [_text_agreement(item["text"], peer["text"])
                       for peer in usable if peer is not item]
        agreement = max(peer_scores, default=0.0)
        ranked.append((item["score"] + 0.08 * agreement, item))
    return max(ranked, key=lambda pair: pair[0])[1]


def _make_result(row_index: int, row_bbox: Box | None, bbox: Box | None,
                 image_reference: str | None, status: str, text: str = "",
                 confidence: float = 0.0, crop: np.ndarray | None = None,
                 evidence: dict[str, Any] | None = None) -> ObservationResult:
    return ObservationResult(row_index, text, float(np.clip(confidence, 0, 1)), status,
                             bbox, "image_ocr", image_reference, row_bbox, crop, evidence or {})


def extract_observation_text(
    image: np.ndarray,
    row_geometry: Any,
    observation_column_geometry: Any = None,
    *,
    row_index: int | None = None,
    tick_result: Any = None,
    image_reference: str | None = None,
    cache: ObservationOCRCache | None = None,
    ocr_reader: OCRCallable | None = None,
) -> ObservationResult:
    """OCR the Observation Point cell for one geometry row.

    Pass ``tick_result`` when available; an unmarked tick result returns an
    empty OCR result without calling OCR. The explicit column argument accepts
    a ``(x1, y1, x2, y2)`` box or a mapping keyed by row index.
    """
    source, rows = _image_and_rows(image, row_geometry)
    if not rows:
        return _make_result(-1, None, None, image_reference, "INVALID_REGION",
                            evidence={"reason": "no_rows"})
    if row_index is None:
        row = rows[0]
    else:
        row = next((item for item in rows if int(_value(item, "row_index", -1)) == int(row_index)), None)
        if row is None:
            return _make_result(int(row_index), None, None, image_reference, "INVALID_REGION",
                                evidence={"reason": "row_not_found"})
    idx = int(_value(row, "row_index", 0))
    row_bbox = _box(_value(row, "bbox"))
    if tick_result is not None:
        marked = _value(tick_result, "not_ok_marked", _value(tick_result, "marked", False))
        if not marked:
            return _make_result(idx, row_bbox, None, image_reference, "OCR_EMPTY",
                                evidence={"reason": "row_not_confirmed_not_ok"})

    bbox = _find_observation_box(row, observation_column_geometry)
    if bbox is None:
        return _make_result(idx, row_bbox, None, image_reference, "INVALID_REGION",
                            evidence={"reason": "observation_column_not_found"})
    gray = _gray(source)
    image_h, image_w = gray.shape
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1 or x1 < 0 or y1 < 0 or x2 > image_w or y2 > image_h:
        return _make_result(idx, row_bbox, bbox, image_reference, "INVALID_REGION",
                            evidence={"reason": "observation_region_outside_image"})

    row_height = row_bbox[3] - row_bbox[1] if row_bbox else y2 - y1
    # The row engine's horizontal projection can understate a row height where
    # rules slope across the page. Refine locally using rules within this
    # observation column, constrained by the detected row and its neighbors.
    if row_bbox is not None:
        refined_y1, refined_y2 = _refine_row_vertical_bounds(
            gray, row, rows, x1, x2)
    else:
        refined_y1, refined_y2 = y1, y2
    refined_height = max(1, refined_y2 - refined_y1)
    # A small inset keeps the detected grid rules out while preserving short
    # glyphs. Refinement remains confined to the locally detected row rules.
    pad_y = max(1, round(refined_height * 0.025))
    # Start at the detected Observation Point boundary. A scaled inset clipped
    # the first printed glyph in the perspective-distorted stuffing sample.
    # The boundary is the dynamic column edge, so this does not enter S.No.
    pad_left = 0
    pad_right = max(1, round((x2 - x1) * 0.008))
    crop_box = (x1 + pad_left, refined_y1 + pad_y,
                x2 - pad_right, refined_y2 - pad_y)
    cx1, cy1, cx2, cy2 = crop_box
    if cx2 <= cx1 or cy2 <= cy1:
        return _make_result(idx, row_bbox, bbox, image_reference, "INVALID_REGION",
                            evidence={"reason": "empty_observation_crop"})
    crop_image = gray
    source_reconstructed = False
    source_geometry = (_source_geometry(image, source, cache)
                       if bool(_value(row_geometry, "perspective_corrected", False)) else None)
    if source_geometry is not None and row_bbox is not None:
        matrix, raw_geometry = source_geometry
        raw_rows = list(raw_geometry.rows)
        raw_box = _find_observation_box(raw_rows[0]) if raw_rows else None
        if raw_box is not None:
            rx1, _, rx2, _ = raw_box
            center_x = (x1 + x2) / 2
            center_y = (row_bbox[1] + row_bbox[3]) / 2
            mapped = cv2.perspectiveTransform(
                np.float32([[[center_x, center_y]]]), matrix)[0, 0]
            row_corners = np.float32([[[row_bbox[0], row_bbox[1]],
                                       [row_bbox[2], row_bbox[1]],
                                       [row_bbox[2], row_bbox[3]],
                                       [row_bbox[0], row_bbox[3]]]])
            mapped_corners = cv2.perspectiveTransform(row_corners, matrix)[0]
            local_height = float(np.mean([
                np.linalg.norm(mapped_corners[3] - mapped_corners[0]),
                np.linalg.norm(mapped_corners[2] - mapped_corners[1]),
            ]))
            source_gray = _gray(image)
            sy1 = max(0, int(round(mapped[1] - local_height / 2)))
            sy2 = min(source_gray.shape[0], int(round(mapped[1] + local_height / 2)))
            sx1, sx2 = max(0, rx1), min(source_gray.shape[1], rx2)
            if sx2 > sx1 and sy2 > sy1:
                crop_image = source_gray
                source_reconstructed = True
                crop_box = (sx1, sy1, sx2, sy2)
                crop = source_gray[sy1:sy2, sx1:sx2].copy()
            else:
                crop = gray[cy1:cy2, cx1:cx2].copy()
        else:
            crop = gray[cy1:cy2, cx1:cx2].copy()
    else:
        crop = gray[cy1:cy2, cx1:cx2].copy()
    trimmed = _trim_trailing_blank(crop)
    crop = trimmed.copy()
    if crop_image is gray:
        crop_box = (cx1, cy1, cx1 + trimmed.shape[1], cy2)
        expanded_crop = gray[refined_y1:refined_y2,
                             cx1:min(image_w, cx1 + crop.shape[1])].copy()
    else:
        sx1, sy1, sx2, sy2 = crop_box
        crop_box = (sx1, sy1, sx1 + trimmed.shape[1], sy2)
        expanded_crop = crop_image[max(0, sy1 - max(1, sy2 - sy1) // 4):
                                   min(crop_image.shape[0], sy2 + max(1, sy2 - sy1) // 4),
                                   sx1:sx2].copy()
    cache_key = (id(source), idx, bbox, crop_box)
    if cache is not None:
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

    # A truly empty/near-white crop needs no OCR call.
    if float(np.std(crop)) < 2.0 and float(np.mean(crop)) > 245:
        result = _make_result(idx, row_bbox, crop_box, image_reference, "OCR_EMPTY", crop=crop,
                              evidence={"reason": "blank_region", "ocr_calls": 0})
        if cache is not None:
            cache.put(cache_key, result)
        return result

    try:
        candidates = _preprocessing_candidates(crop, expanded_crop)
        candidate_results = []
        if ocr_reader is not None:
            # Keep injected readers deterministic and single-call for callers
            # that supply an alternate OCR backend or test double.
            data = ocr_reader(candidates[0][1])
            text, confidence, word_evidence = _parse_ocr(data)
            candidate_results.append({
                "variant": candidates[0][0], "psm": None, "text": text,
                "confidence": confidence, "words": word_evidence,
                "score": _candidate_score(text, confidence, word_evidence),
            })
        else:
            for variant_name, processed in candidates:
                for psm in (7, 6, 11):
                    data = _tesseract_reader(processed, psm)
                    text, confidence, word_evidence = _parse_ocr(data)
                    candidate_results.append({
                        "variant": variant_name, "psm": psm, "text": text,
                        "confidence": confidence, "words": word_evidence,
                        "score": _candidate_score(text, confidence, word_evidence),
                    })
        selected = _select_candidate(candidate_results)
        if selected is None:
            status = "OCR_EMPTY"
            text, confidence, word_evidence, selected = "", 0.0, [], None
        else:
            text = selected["text"]
            confidence = selected["confidence"]
            word_evidence = selected["words"]
            agreements = [_text_agreement(selected["text"], item["text"])
                          for item in candidate_results
                          if item is not selected and item["text"]]
            corroborated = max(agreements, default=0.0) >= 0.52
            # Higher confidence is required when the OCR candidates disagree;
            # disagreement can signal a clipped or noisy crop.
            starts_with_noise = bool(re.match(r"^[^\w]+\w", text, flags=re.UNICODE))
            edge_fragment = bool(word_evidence and
                                 (len(word_evidence[0]["word"]) == 1 or
                                  len(word_evidence[-1]["word"]) == 1))
            status = ("OCR_UNCLEAR" if confidence < 0.85 or
                      (len(candidate_results) > 1 and confidence < 0.90 and
                       (not corroborated or not source_reconstructed)) or
                      starts_with_noise or edge_fragment else "OK")
        result = _make_result(idx, row_bbox, crop_box, image_reference, status, text,
                              confidence, crop, {
                                  "words": word_evidence,
                                  "ocr_calls": len(candidate_results),
                                  "selected_variant": selected["variant"] if selected else None,
                                  "selected_psm": selected["psm"] if selected else None,
                                  "candidate_results": [
                                      {"variant": item["variant"], "psm": item["psm"],
                                       "text": item["text"],
                                       "confidence": round(item["confidence"], 4),
                                       "score": round(item["score"], 4)}
                                      for item in candidate_results],
                              })
    except Exception as error:
        result = _make_result(idx, row_bbox, crop_box, image_reference, "OCR_UNCLEAR",
                              confidence=0.0, crop=crop,
                              evidence={"reason": "ocr_engine_unavailable_or_failed",
                                        "error": type(error).__name__, "ocr_calls": 1})
    if cache is not None:
        cache.put(cache_key, result)
    return result


def extract_observations_for_rows(
    image: np.ndarray,
    detected_rows: Any,
    tick_results: Iterable[Any] | None = None,
    *,
    observation_column_geometry: Any = None,
    image_reference: str | None = None,
    cache: ObservationOCRCache | None = None,
    ocr_reader: OCRCallable | None = None,
) -> list[ObservationResult]:
    """Extract observation text for confirmed NOT OK rows.

    When tick results are provided, only marked rows are OCR'd. With no tick
    list, ``detected_rows`` is treated as the already-filtered confirmed row
    sequence. Each selected row gets one crop and a bounded set of OCR candidates.
    """
    input_image = image
    source, rows = _image_and_rows(image, detected_rows)
    active_cache = cache if cache is not None else ObservationOCRCache()
    selected = []
    if tick_results is None:
        selected = rows
    else:
        ticks_by_index = {int(_value(tick, "row_index", -1)): tick for tick in tick_results}
        selected = [row for row in rows if bool(_value(ticks_by_index.get(int(_value(row, "row_index", -1))),
                                                        "not_ok_marked", False))]
    selected_geometry = {
        "corrected_image": source,
        "rows": selected,
        "perspective_corrected": bool(_value(detected_rows, "perspective_corrected", False)),
    }
    results = []
    for row in selected:
        idx = int(_value(row, "row_index", 0))
        tick = None if tick_results is None else ticks_by_index.get(idx)
        results.append(extract_observation_text(
            input_image, selected_geometry, observation_column_geometry,
            row_index=idx, tick_result=tick, image_reference=image_reference,
            cache=active_cache, ocr_reader=ocr_reader,
        ))
    return results
