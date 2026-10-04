"""Testable upload-batch orchestration for the Streamlit application."""
from __future__ import annotations

from dataclasses import dataclass, field
import re
import time
from typing import Any, Callable, Mapping, Sequence

import cv2
import numpy as np

from central_write_adapter import (
    CentralWriteAdapter, CentralWriteResult, UnconfiguredCentralWriteAdapter,
    WriteStatus,
)
from jpc_engine import (JPCExtraction, extract_jpc_candidates, is_valid_entered_jpc, normalize_jpc)
from observation_engine import extract_observations_for_rows
from row_engine import detect_checklist_rows
from tick_engine import detect_not_ok_marks


@dataclass(frozen=True)
class PhotoJPCValidation:
    filename: str
    status: str
    candidates: tuple[str, ...] = ()
    reason: str = ""


@dataclass
class BatchResult:
    status: str
    message: str
    jpc: str
    photo_jpc_results: list[PhotoJPCValidation] = field(default_factory=list)
    observations: list[dict[str, Any]] = field(default_factory=list)
    write_result: CentralWriteResult | None = None
    timings_ms: dict[str, float] = field(default_factory=dict)


def _photo_parts(photo: Any) -> tuple[str, bytes]:
    if isinstance(photo, Mapping):
        return str(photo.get("filename") or "uploaded-photo"), bytes(photo.get("data") or b"")
    if isinstance(photo, tuple) and len(photo) == 2:
        return str(photo[0]), bytes(photo[1])
    return str(getattr(photo, "name", "uploaded-photo")), bytes(photo.getvalue())


def _field(item: Any, key: str, default=None):
    return item.get(key, default) if isinstance(item, Mapping) else getattr(item, key, default)


def _normalize_observation(value: str) -> str:
    return re.sub(r"[^\w]", "", str(value).casefold(), flags=re.UNICODE)


def _deduplicate_records(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    positions: dict[str, int] = {}
    for record in records:
        key = _normalize_observation(record.get("Observation", ""))
        # OCR_UNCLEAR rows have no text to compare and must each remain reviewable.
        if not key:
            result.append(record)
            continue
        if key not in positions:
            positions[key] = len(result)
            record["Source Photos"] = [record["Source Photo"]]
            record["Source Rows"] = [record["Source Row"]]
            result.append(record)
            continue
        existing = result[positions[key]]
        if record["Source Photo"] not in existing["Source Photos"]:
            existing["Source Photos"].append(record["Source Photo"])
        existing["Source Rows"].append(record["Source Row"])
    return result


def process_upload_batch(
    inspector: str,
    entered_jpc: str,
    photos: Sequence[Any],
    *,
    central_adapter: CentralWriteAdapter | None = None,
    jpc_extractor: Callable[[np.ndarray], JPCExtraction] = extract_jpc_candidates,
    row_detector: Callable[..., Any] = detect_checklist_rows,
    tick_detector: Callable[..., Any] = detect_not_ok_marks,
    observation_extractor: Callable[..., Any] = extract_observations_for_rows,
) -> BatchResult:
    """Validate every page before running dynamic processing or one batch write."""
    start = time.perf_counter()
    timings: dict[str, float] = {}
    normalized_expected = normalize_jpc(entered_jpc)
    if not is_valid_entered_jpc(entered_jpc):
        return BatchResult("REJECTED", "Invalid JPC Number. Use 2-3 letters, a mandatory hyphen, then digits (example: VH-3009).", normalized_expected)
    if not str(inspector).strip():
        return BatchResult("REJECTED", "Inspector Name is required.", normalized_expected)
    if not normalized_expected:
        return BatchResult("REJECTED", "JPC Number is required.", normalized_expected)
    if not photos:
        return BatchResult("REJECTED", "Upload at least one checklist photo.", normalized_expected)

    decoded: list[tuple[str, np.ndarray | None]] = []
    for photo in photos:
        filename, data = _photo_parts(photo)
        array = np.frombuffer(data, dtype=np.uint8)
        image = cv2.imdecode(array, cv2.IMREAD_COLOR) if array.size else None
        decoded.append((filename, image))

    jpc_started = time.perf_counter()
    validations: list[PhotoJPCValidation] = []
    for filename, image in decoded:
        if image is None:
            validations.append(PhotoJPCValidation(filename, "UNREADABLE", (),
                                                  "Photo could not be decoded; JPC is unreadable."))
            continue
        extraction = jpc_extractor(image)
        candidates = tuple(str(candidate) for candidate in extraction.candidates)
        if extraction.status == "MULTIPLE" or len(candidates) > 1:
            validations.append(PhotoJPCValidation(filename, "MULTIPLE", candidates,
                                                  "Multiple JPC candidates were detected."))
        elif extraction.status != "READABLE" or len(candidates) != 1:
            validations.append(PhotoJPCValidation(filename, "UNREADABLE", candidates,
                                                  "Exactly one readable JPC was not detected."))
        elif normalize_jpc(candidates[0]) != normalized_expected:
            validations.append(PhotoJPCValidation(filename, "MISMATCH", candidates,
                                                  "Detected JPC does not match the entered JPC."))
        else:
            validations.append(PhotoJPCValidation(filename, "MATCH", candidates))
    timings["jpc_validation"] = (time.perf_counter() - jpc_started) * 1000

    failures = [item for item in validations if item.status != "MATCH"]
    if failures:
        timings["total"] = (time.perf_counter() - start) * 1000
        return BatchResult("JPC_REJECTED", "Batch rejected: every photo must contain exactly one matching JPC.",
                           normalized_expected, validations, timings_ms=timings)

    records: list[dict[str, Any]] = []
    for filename, image in decoded:
        assert image is not None
        row_started = time.perf_counter()
        geometry = row_detector(image)
        timings["row_detection"] = timings.get("row_detection", 0.0) + (time.perf_counter() - row_started) * 1000

        tick_started = time.perf_counter()
        ticks = tick_detector(image, geometry)
        timings["tick_detection"] = timings.get("tick_detection", 0.0) + (time.perf_counter() - tick_started) * 1000
        confirmed = [tick for tick in ticks if bool(_field(tick, "not_ok_marked", False))]
        if not confirmed:
            continue

        ocr_started = time.perf_counter()
        ocr_results = observation_extractor(image, geometry, ticks, image_reference=filename)
        timings["observation_ocr"] = timings.get("observation_ocr", 0.0) + (time.perf_counter() - ocr_started) * 1000
        by_row = {int(_field(result, "row_index", -1)): result for result in ocr_results}
        for tick in confirmed:
            row_index = int(_field(tick, "row_index", -1))
            observation = by_row.get(row_index)
            records.append({
                "JPC Number": normalized_expected,
                "Inspector Name": str(inspector).strip(),
                "Observation": str(_field(observation, "text", "") or "").strip(),
                "OCR Status": str(_field(observation, "status", "OCR_UNCLEAR")),
                "OCR Confidence": float(_field(observation, "confidence", 0.0) or 0.0),
                "Source Photo": filename,
                "Source Row": row_index,
            })

    deduplicated = _deduplicate_records(records)
    if not deduplicated:
        timings["total"] = (time.perf_counter() - start) * 1000
        return BatchResult("NO_OBSERVATIONS", "No confirmed NOT OK rows were found; no write was attempted.",
                           normalized_expected, validations, [], None, timings)

    writer = central_adapter or UnconfiguredCentralWriteAdapter()
    write_started = time.perf_counter()
    try:
        write_result = writer.write_batch(normalized_expected, str(inspector).strip(), deduplicated)
    except Exception as error:
        write_result = CentralWriteResult(WriteStatus.ERROR,
                                          f"Central write adapter failed: {type(error).__name__}")
    timings["central_write"] = (time.perf_counter() - write_started) * 1000
    timings["total"] = (time.perf_counter() - start) * 1000
    return BatchResult(write_result.status.value, write_result.message, normalized_expected,
                       validations, deduplicated, write_result, timings)
