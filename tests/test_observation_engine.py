"""Unit and real-photo crop tests for row-specific observation OCR."""
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from observation_engine import (
    ObservationOCRCache,
    _preprocessing_candidates,
    _candidate_score,
    _refine_row_vertical_bounds,
    _select_candidate,
    _trim_trailing_blank,
    extract_observation_text,
    extract_observations_for_rows,
)
from row_engine import detect_checklist_rows
from tick_engine import detect_not_ok_marks


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = ROOT / "sample_photos"
SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _synthetic_page(scale=1, perspective=False):
    width, height = 800 * scale, 560 * scale
    image = np.full((height, width, 3), 255, np.uint8)
    xs = [40, 110, 470, 550, 630, 760]
    ys = [40 + 35 * i for i in range(13)]
    xs, ys = [x * scale for x in xs], [y * scale for y in ys]
    for x in xs:
        cv2.line(image, (x, ys[0]), (x, ys[-1]), (20, 20, 20), max(1, scale))
    for y in ys:
        cv2.line(image, (xs[0], y), (xs[-1], y), (20, 20, 20), max(1, scale))
    if perspective:
        src = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
        dst = np.float32([[12, 8], [width - 10, 20], [width - 2, height - 8], [8, height - 15]])
        image = cv2.warpPerspective(image, cv2.getPerspectiveTransform(src, dst),
                                    (width, height), borderValue=(245, 245, 245))
    return image


def _put_observation(image, row, text):
    box = max((b for b in row.column_geometry.values()), key=lambda b: b[2] - b[0])
    x1, y1, x2, y2 = box
    baseline = y1 + max(2, (y2 - y1) * 2 // 3)
    cv2.putText(image, text, (x1 + 5, baseline), cv2.FONT_HERSHEY_SIMPLEX,
                max(0.30, (y2 - y1) / 85), (20, 20, 20), 1, cv2.LINE_AA)


def _ocr_data(words, confidences=None):
    if confidences is None:
        confidences = ["92"] * len(words)
    return {"text": words, "conf": confidences}


class QueuedReader:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, image):
        self.calls.append(image.copy())
        if not self.responses:
            raise AssertionError("OCR reader called more times than expected")
        return self.responses.pop(0)


def test_normal_photo_like_row_uses_only_its_observation_cell():
    image = _synthetic_page()
    geometry = detect_checklist_rows(image, perspective=False)
    row = geometry.rows[1]
    _put_observation(image, row, "Door bush missing")
    geometry = replace(geometry, corrected_image=image)
    reader = QueuedReader(_ocr_data(["Door", "bush", "missing"]))
    result = extract_observation_text(image, geometry, row_index=row.row_index, ocr_reader=reader)
    obs_box = max(row.column_geometry.values(), key=lambda b: b[2] - b[0])
    assert result.text == "Door bush missing"
    assert result.status == "OK"
    assert result.confidence > 0.8
    assert result.bbox[0] >= obs_box[0] and result.bbox[2] <= obs_box[2]
    assert result.bbox[1] >= row.bbox[1] and result.bbox[3] <= row.bbox[3]
    assert len(reader.calls) == 1


def test_multiple_rows_keep_distinct_current_wording_without_master_lookup():
    image = _synthetic_page()
    geometry = detect_checklist_rows(image, perspective=False)
    texts = ["Door bush not provided", "Mounting bracket condition revised"]
    for row, text in zip(geometry.rows[1:3], texts):
        _put_observation(image, row, text)
    geometry = replace(geometry, corrected_image=image)
    ticks = [{"row_index": 1, "not_ok_marked": True},
             {"row_index": 2, "not_ok_marked": True}]
    reader = QueuedReader(_ocr_data(texts[0].split()), _ocr_data(texts[1].split()))
    results = extract_observations_for_rows(image, geometry, ticks, ocr_reader=reader)
    assert [result.text for result in results] == texts
    assert all(result.status == "OK" for result in results)
    assert results[0].row_index == 1 and results[1].row_index == 2


@pytest.mark.parametrize("scale,perspective", [(1, True), (2, False)])
def test_perspective_and_resolution_keep_observation_region_dynamic(scale, perspective):
    image = _synthetic_page(scale=scale, perspective=perspective)
    geometry = detect_checklist_rows(image)
    row = geometry.rows[2]
    _put_observation(geometry.corrected_image, row, "Panel support updated")
    geometry = replace(geometry, corrected_image=geometry.corrected_image)
    reader = QueuedReader(_ocr_data(["Panel", "support", "updated"]))
    result = extract_observation_text(image, geometry, row_index=row.row_index, ocr_reader=reader)
    assert result.status == "OK"
    assert result.text == "Panel support updated"
    assert result.evidence_crop.shape[0] > 0 and result.evidence_crop.shape[1] > 0


def test_ocr_noise_is_cleaned_but_current_text_is_preserved():
    image = _synthetic_page()
    geometry = detect_checklist_rows(image, perspective=False)
    _put_observation(image, geometry.rows[1], "Door bush not provided")
    geometry = replace(geometry, corrected_image=image)
    reader = QueuedReader(_ocr_data(["|", "Door", "bush", "not", "provided", "~"],
                                    ["-1", "93", "88", "91", "86", "-1"]))
    result = extract_observation_text(image, geometry, row_index=1, ocr_reader=reader)
    assert result.text == "Door bush not provided"
    assert result.status == "OK"


def test_low_confidence_and_empty_regions_are_not_invented():
    image = _synthetic_page()
    geometry = detect_checklist_rows(image, perspective=False)
    _put_observation(image, geometry.rows[1], "Ambiguous observation text")
    geometry = replace(geometry, corrected_image=image)
    weak_reader = QueuedReader(_ocr_data(["ambiguous", "text"], ["18", "24"]))
    unclear = extract_observation_text(image, geometry, row_index=1, ocr_reader=weak_reader)
    assert unclear.status == "OCR_UNCLEAR"
    assert unclear.text == "ambiguous text"
    assert unclear.evidence_crop is not None

    moderate_reader = QueuedReader(_ocr_data(["ambiguous", "word"], ["76", "78"]))
    moderate = extract_observation_text(image, geometry, row_index=1,
                                        ocr_reader=moderate_reader)
    assert moderate.status == "OCR_UNCLEAR"
    assert moderate.text == "ambiguous word"

    fragment_reader = QueuedReader(_ocr_data(["Partial", "m"], ["96", "96"]))
    fragment = extract_observation_text(image, geometry, row_index=1,
                                        ocr_reader=fragment_reader)
    assert fragment.status == "OCR_UNCLEAR"
    assert fragment.text == "Partial m"

    blank = np.full_like(image, 255)
    blank_geometry = detect_checklist_rows(blank, perspective=False) if False else geometry
    # The row geometry still points to a valid, uniformly blank observation ROI.
    blank_geometry = replace(blank_geometry, corrected_image=blank)
    unused_reader = QueuedReader()
    empty = extract_observation_text(blank, blank_geometry, row_index=1, ocr_reader=unused_reader)
    assert empty.status == "OCR_EMPTY"
    assert empty.text == ""
    assert empty.evidence["ocr_calls"] == 0
    assert unused_reader.calls == []


def test_adjacent_row_text_is_excluded_by_row_specific_crop():
    image = _synthetic_page()
    geometry = detect_checklist_rows(image, perspective=False)
    row = geometry.rows[1]
    next_row = geometry.rows[2]
    first_box = max(row.column_geometry.values(), key=lambda b: b[2] - b[0])
    next_box = max(next_row.column_geometry.values(), key=lambda b: b[2] - b[0])
    _put_observation(image, row, "Row alpha")
    _put_observation(image, next_row, "Neighbor contamination")
    geometry = replace(geometry, corrected_image=image)
    reader = QueuedReader(_ocr_data(["Row", "alpha"]))
    result = extract_observation_text(image, geometry, row_index=row.row_index, ocr_reader=reader)
    assert result.text == "Row alpha"
    assert result.bbox[1] >= row.boundaries[0]
    assert result.bbox[3] <= row.boundaries[1]
    assert result.bbox[3] <= next_box[1] or result.bbox[1] >= first_box[3]


def test_rephrased_unknown_observation_is_returned_as_seen():
    image = _synthetic_page()
    geometry = detect_checklist_rows(image, perspective=False)
    _put_observation(image, geometry.rows[1], "Door bush condition")
    geometry = replace(geometry, corrected_image=image)
    reader = QueuedReader(_ocr_data(["Door", "bush", "condition"]))
    result = extract_observation_text(image, geometry, row_index=1, ocr_reader=reader)
    assert result.text == "Door bush condition"
    assert result.status == "OK"


def test_only_multiple_confirmed_not_ok_rows_are_batch_ocrd():
    image = _synthetic_page()
    geometry = detect_checklist_rows(image, perspective=False)
    _put_observation(image, geometry.rows[1], "First current text")
    _put_observation(image, geometry.rows[4], "Another unmapped point")
    geometry = replace(geometry, corrected_image=image)
    ticks = [{"row_index": 1, "not_ok_marked": True},
             {"row_index": 2, "not_ok_marked": False},
             {"row_index": 4, "not_ok_marked": True}]
    reader = QueuedReader(_ocr_data(["First", "current", "text"]),
                          _ocr_data(["Another", "unmapped", "point"]))
    results = extract_observations_for_rows(image, geometry, ticks, ocr_reader=reader,
                                            image_reference="sample-page-1")
    assert [result.row_index for result in results] == [1, 4]
    assert [result.text for result in results] == ["First current text", "Another unmapped point"]
    assert all(result.image_reference == "sample-page-1" for result in results)
    assert len(reader.calls) == 2


def test_cache_prevents_duplicate_ocr_for_same_row():
    image = _synthetic_page()
    geometry = detect_checklist_rows(image, perspective=False)
    _put_observation(image, geometry.rows[1], "Cached text")
    geometry = replace(geometry, corrected_image=image)
    cache, reader = ObservationOCRCache(), QueuedReader(_ocr_data(["Cached", "text"]))
    first = extract_observation_text(image, geometry, row_index=1, cache=cache, ocr_reader=reader)
    second = extract_observation_text(image, geometry, row_index=1, cache=cache, ocr_reader=reader)
    assert first.text == second.text == "Cached text"
    assert len(reader.calls) == 1


def test_local_grid_rules_expand_a_short_row_crop_without_fixed_coordinates():
    image = np.full((110, 520), 255, np.uint8)
    cv2.line(image, (45, 35), (475, 35), 20, 1)
    cv2.line(image, (45, 75), (475, 75), 20, 1)
    row = {"row_index": 4, "bbox": (45, 45, 475, 65)}

    top, bottom = _refine_row_vertical_bounds(image, row, [row], 45, 475)

    assert top <= 35
    assert bottom >= 75
    assert bottom - top > row["bbox"][3] - row["bbox"][1]


def test_short_crop_gets_mild_upscale_and_trailing_blank_is_trimmed():
    crop = np.full((12, 420), 245, np.uint8)
    cv2.putText(crop, "Current row wording", (5, 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.28, 20, 1, cv2.LINE_AA)
    cv2.circle(crop, (390, 5), 1, 80, -1)  # isolated photo speck

    trimmed = _trim_trailing_blank(crop)
    candidates = _preprocessing_candidates(trimmed, crop)

    assert trimmed.shape[1] < crop.shape[1]
    assert {name for name, _ in candidates} == {
        "current_crop", "expanded_vertical", "rule_suppressed",
        "mild_upscale", "contrast_normalized",
    }
    by_name = dict(candidates)
    assert by_name["current_crop"].shape[0] >= 72
    assert by_name["expanded_vertical"].shape[0] >= 72
    assert by_name["mild_upscale"].shape[0] > trimmed.shape[0]
    assert by_name["contrast_normalized"].shape[0] >= 72
    assert by_name["rule_suppressed"].shape == by_name["current_crop"].shape


def test_candidate_selection_uses_confidence_and_penalizes_edge_border_noise():
    details = [{"word": "uel", "confidence": 90.0},
               {"word": "tank", "confidence": 90.0},
               {"word": "sticker", "confidence": 90.0},
               {"word": "missing", "confidence": 90.0}]
    edge_details = [{"word": "-uel", "confidence": 96.0}, *details[1:]]
    candidates = [
        {"text": "-uel tank sticker missing", "confidence": 0.96,
         "score": _candidate_score("-uel tank sticker missing", 0.96, edge_details)},
        {"text": "uel tank sticker missing", "confidence": 0.90,
         "score": _candidate_score("uel tank sticker missing", 0.90, details)},
        {"text": "", "confidence": 0.0, "score": -1.0},
    ]

    selected = _select_candidate(candidates)

    assert selected["text"] == "uel tank sticker missing"


def test_real_sample_photos_use_row_and_tick_geometry():
    paths = sorted(path for path in SAMPLE_DIR.iterdir()
                   if path.is_file() and path.suffix.lower() in SUFFIXES)
    assert paths
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        assert image is not None
        geometry = detect_checklist_rows(image)
        ticks = detect_not_ok_marks(image, geometry)
        selected = [tick for tick in ticks if tick.not_ok_marked]
        assert selected, f"no confirmed NOT OK rows in sample {path.name}"
        # The local wrapper exists, but if the native Tesseract binary is
        # unavailable the engine must preserve each review crop and report an
        # explicit OCR_UNCLEAR result without claiming OCR text.
        results = extract_observations_for_rows(image, geometry, ticks,
                                                image_reference=path.name)
        assert len(results) == len(selected)
        assert all(result.row_index in {tick.row_index for tick in selected} for result in results)
        assert all(result.evidence_crop is not None for result in results)
        assert all(result.status in {"OK", "OCR_UNCLEAR", "OCR_EMPTY"} for result in results)

