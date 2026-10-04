"""JPC-gated app pipeline contract tests (no real central API is configured)."""
from dataclasses import dataclass
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from app_pipeline import process_upload_batch
from central_write_adapter import CentralWriteResult, UnconfiguredCentralWriteAdapter, WriteStatus
from jpc_engine import (
    JPCExtraction, _candidate_spans, is_valid_entered_jpc,
    is_valid_photo_jpc_candidate, normalize_jpc,
)


def _photo(name, shade):
    image = np.full((40, 80, 3), shade, np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    return {"filename": name, "data": encoded.tobytes()}


def _jpc(candidate="VH-3009", status="READABLE"):
    return JPCExtraction((candidate,) if candidate else (), status, 0.99)


@dataclass
class _Tick:
    row_index: int
    not_ok_marked: bool


class _RecordingUnavailableAdapter:
    def __init__(self):
        self.calls = []

    def write_batch(self, jpc, inspector, observations):
        self.calls.append((jpc, inspector, observations))
        # A test adapter may record the contract, but it never fabricates success.
        return CentralWriteResult(WriteStatus.ERROR, "test transport unavailable")


def _harness(jpcs=None, marked=None, texts=None):
    jpc_map = jpcs or {}
    mark_map = marked or {}
    text_map = texts or {}
    calls = {"row": 0, "tick": 0, "ocr": 0}

    def jpc_extractor(image):
        shade = int(round(float(image.mean())))
        return jpc_map.get(shade, _jpc(None, "UNREADABLE"))

    def row_detector(image):
        calls["row"] += 1
        return {"marker": int(round(float(image.mean())))}

    def tick_detector(image, geometry):
        calls["tick"] += 1
        shade = geometry["marker"]
        return [_Tick(7, bool(mark_map.get(shade, False)))]

    def observation_extractor(image, geometry, ticks, **kwargs):
        calls["ocr"] += 1
        shade = geometry["marker"]
        text = text_map.get(shade, "")
        return [SimpleNamespace(row_index=7, text=text, confidence=0.97,
                                status="OK" if text else "OCR_UNCLEAR")]

    return jpc_extractor, row_detector, tick_detector, observation_extractor, calls


def _run(photos, deps, adapter, entered="VH-3009"):
    jpc_extractor, row_detector, tick_detector, observation_extractor, calls = deps
    result = process_upload_batch(
        "Inspector A", entered, photos, central_adapter=adapter,
        jpc_extractor=jpc_extractor, row_detector=row_detector,
        tick_detector=tick_detector, observation_extractor=observation_extractor,
    )
    return result, calls


def test_valid_multi_photo_batch_uses_one_normalized_jpc_and_one_write_call():
    photos = [_photo("page-a.jpg", 70), _photo("page-b.jpg", 180)]
    deps = _harness(
        {70: _jpc("VH-3009"), 180: _jpc("VH3009")},
        {70: True, 180: True},
        {70: "Rephrased inspection point alpha", 180: "Different current wording beta"},
    )
    adapter = _RecordingUnavailableAdapter()
    result, calls = _run(photos, deps, adapter)

    assert result.status == "ERROR"  # The API is unavailable; this is not a fake success.
    assert result.jpc == "VH3009"
    assert [r["Observation"] for r in result.observations] == [
        "Rephrased inspection point alpha", "Different current wording beta"]
    assert calls == {"row": 2, "tick": 2, "ocr": 2}
    assert len(adapter.calls) == 1
    assert adapter.calls[0][0] == "VH3009"
    assert len(adapter.calls[0][2]) == 2
    assert normalize_jpc("VH-3009") == normalize_jpc("VH 3009") == normalize_jpc("VH3009")


@pytest.mark.parametrize("failure", ["MISMATCH", "UNREADABLE", "MULTIPLE"])
def test_any_jpc_failure_rejects_entire_batch_before_dynamic_processing(failure):
    photos = [_photo("matching.jpg", 60), _photo("bad.jpg", 170)]
    bad = {
        "MISMATCH": _jpc("VM-3009"),
        "UNREADABLE": _jpc(None, "UNREADABLE"),
        "MULTIPLE": JPCExtraction(("VH-3009", "VM-3009"), "MULTIPLE", 0.96),
    }[failure]
    deps = _harness({60: _jpc("VH-3009"), 170: bad}, {60: True}, {60: "Current row wording"})
    adapter = _RecordingUnavailableAdapter()
    result, calls = _run(photos, deps, adapter)

    assert result.status == "JPC_REJECTED"
    assert [item.status for item in result.photo_jpc_results] == ["MATCH", failure]
    assert result.observations == []
    assert calls == {"row": 0, "tick": 0, "ocr": 0}
    assert adapter.calls == []


@pytest.mark.parametrize("case", ["ok_only", "blank_not_ok"])
def test_ok_only_and_blank_not_ok_rows_do_not_create_records(case):
    photo = _photo(f"{case}.jpg", 80)
    deps = _harness({80: _jpc("VH-3009")}, {80: False}, {80: "Must not be read"})
    adapter = _RecordingUnavailableAdapter()
    result, calls = _run([photo], deps, adapter)

    assert result.status == "NO_OBSERVATIONS"
    assert result.observations == []
    assert calls == {"row": 1, "tick": 1, "ocr": 0}
    assert adapter.calls == []


def test_duplicate_observation_across_photos_is_deduplicated_before_write():
    photos = [_photo("first.jpg", 65), _photo("second.jpg", 190)]
    wording = "Current observation wording, revised"
    deps = _harness({65: _jpc("VH-3009"), 190: _jpc("VH 3009")},
                    {65: True, 190: True}, {65: wording, 190: wording.upper()})
    adapter = _RecordingUnavailableAdapter()
    result, _ = _run(photos, deps, adapter)

    assert result.status == "ERROR"
    assert len(result.observations) == 1
    assert len(result.observations[0]["Source Photos"]) == 2
    assert len(adapter.calls) == 1
    assert len(adapter.calls[0][2]) == 1


def test_default_central_adapter_fails_closed_and_does_not_claim_success():
    result = UnconfiguredCentralWriteAdapter().write_batch("VH3009", "Inspector", [{"Observation": "x"}])
    assert result.status is WriteStatus.ERROR
    assert result.records_written == 0


def test_jpc_parser_detects_only_supported_photo_formats():
    candidates = [item[0] for item in _candidate_spans(
        "JPC Number: VH-3009 and VM 3010\nOther JPC: VHO/3020")]
    assert candidates == ["VH-3009", "VM 3010"]
    assert [item[0] for item in _candidate_spans(
        "JPC Number: VH-3009 Date 12-05-2026")] == ["VH-3009"]
    assert normalize_jpc("VH-3009") == normalize_jpc("VH 3009") == normalize_jpc("VH3009")


def test_photo_jpc_candidate_allows_only_hyphen_space_or_no_separator():
    assert is_valid_photo_jpc_candidate("VH-3009")
    assert is_valid_photo_jpc_candidate("VH 3009")
    assert is_valid_photo_jpc_candidate("VH3009")
    assert not is_valid_photo_jpc_candidate("VH/3009")
    assert not is_valid_photo_jpc_candidate("VH_3009")
    assert not is_valid_photo_jpc_candidate("V-3009")
    assert not is_valid_photo_jpc_candidate("VH-ABCD")


def test_entered_jpc_requires_mandatory_hyphen_and_2_to_3_letters():
    assert is_valid_entered_jpc("VH-3009")
    assert is_valid_entered_jpc("VM-00210093009")
    assert not is_valid_entered_jpc("VH3009")
    assert not is_valid_entered_jpc("VH 3009")
    assert not is_valid_entered_jpc("VM_3009")
    assert not is_valid_entered_jpc("V-3009")
    assert not is_valid_entered_jpc("VH-ABCD")


def test_invalid_entered_jpc_rejects_before_photo_processing():
    deps = _harness({80: _jpc("VH-3009")}, {80: True}, {80: "Must not process"})
    adapter = _RecordingUnavailableAdapter()
    result, calls = _run([_photo("one.jpg", 80)], deps, adapter, entered="VH3009")
    assert result.status == "REJECTED"
    assert "mandatory hyphen" in result.message
    assert calls == {"row": 0, "tick": 0, "ocr": 0}
    assert adapter.calls == []
