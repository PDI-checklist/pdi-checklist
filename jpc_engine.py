"""JPC candidate extraction and formatting normalization for uploaded photos."""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import shutil
from typing import Any

import numpy as np


@dataclass(frozen=True)
class JPCExtraction:
    candidates: tuple[str, ...]
    status: str
    confidence: float
    raw_text: str = ""
    error: str | None = None


_JPC_LABEL = re.compile(r"\bJPC\s*(?:(?:NUMBER)|(?:NO\.?))?\s*[:#\-]?", re.IGNORECASE)
_JPC_VALUE = re.compile(
    r"(?<![A-Z0-9])([A-Z]{2,3}(?:(?:\s*-\s*|\s+)\d{2,}|\d{2,}))(?![A-Z0-9])",
    re.IGNORECASE,
)
_PHOTO_JPC = re.compile(r"^[A-Z]{2,3}(?:-\d+|\s+\d+|\d+)$", re.IGNORECASE)
_TYPED_JPC = re.compile(r"^[A-Z]{2,3}-\d+$", re.IGNORECASE)


def normalize_jpc(value: str) -> str:
    """Normalize OCR/presentation separators for comparison."""
    return re.sub(r"[^A-Z0-9]", "", str(value).upper())


def is_valid_entered_jpc(value: str) -> bool:
    """Require the operator-entered JPC to use 2-3 letters, a hyphen, then digits."""
    return bool(_TYPED_JPC.fullmatch(str(value).strip()))


def is_valid_photo_jpc_candidate(value: str) -> bool:
    """Accept only OCR forms equivalent to LETTERS-digits, LETTERS space digits, or LETTERSdigits."""
    return bool(_PHOTO_JPC.fullmatch(str(value).strip()))


def _configure_tesseract() -> None:
    import pytesseract

    configured = os.environ.get("PDI_TESSERACT_CMD")
    if configured and Path(configured).is_file():
        pytesseract.pytesseract.tesseract_cmd = configured
        return
    found = shutil.which("tesseract")
    if found:
        pytesseract.pytesseract.tesseract_cmd = found
        return
    program_files = os.environ.get("ProgramFiles", r"C:\Program Files")
    candidate = Path(program_files) / "Tesseract-OCR" / "tesseract.exe"
    if candidate.is_file():
        pytesseract.pytesseract.tesseract_cmd = str(candidate)


def _candidate_spans(text: str) -> list[tuple[str, int, int]]:
    found: list[tuple[str, int, int]] = []
    labels = list(_JPC_LABEL.finditer(text))
    for index, label in enumerate(labels):
        next_label = labels[index + 1].start() if index + 1 < len(labels) else len(text)
        line_end = text.find("\n", label.end(), next_label)
        segment_end = min(next_label, line_end if line_end >= 0 else next_label,
                          label.end() + 100)
        snippet = text[label.end():segment_end]
        match = _JPC_VALUE.search(snippet)
        if not match:
            continue
        value = re.sub(r"\s+", " ", match.group(1)).strip()
        found.append((value, label.end() + match.start(1), label.end() + match.end(1)))
        tail = snippet[match.end(1):]
        connector = re.compile(
            r"^\s*(?:(?:and|or)\b|[,/&])\s*" +
            r"([A-Z]{1,8}(?:(?:\s*[-/]\s*|\s+)\d{2,}|\d{2,}))",
            re.IGNORECASE,
        )
        tail_offset = label.end() + match.end(1)
        while (additional := connector.match(tail)) is not None:
            additional_value = re.sub(r"\s+", " ", additional.group(1)).strip()
            start = tail_offset + additional.start(1)
            end = tail_offset + additional.end(1)
            found.append((additional_value, start, end))
            consumed = additional.end()
            tail_offset += consumed
            tail = tail[consumed:]
    return found


def extract_jpc_candidates(image: np.ndarray) -> JPCExtraction:
    """OCR a single uploaded photo and return all JPC values following JPC labels."""
    if image is None or image.size == 0:
        return JPCExtraction((), "UNREADABLE", 0.0, error="empty_image")
    try:
        import pytesseract
        from pytesseract import Output

        _configure_tesseract()
        data: dict[str, Any] = pytesseract.image_to_data(
            image, config="--psm 6", output_type=Output.DICT)
        line_tokens: dict[tuple[Any, ...], list[tuple[str, float]]] = {}
        for index, (token, raw_conf) in enumerate(zip(data.get("text", []), data.get("conf", []))):
            token = str(token).strip()
            if not token:
                continue
            try:
                confidence = max(0.0, float(raw_conf)) / 100.0
            except (TypeError, ValueError):
                confidence = 0.0
            keys = (data.get("block_num", []), data.get("par_num", []), data.get("line_num", []))
            line_key = tuple(values[index] if index < len(values) else 0 for values in keys)
            line_tokens.setdefault(line_key, []).append((token, confidence))
        lines = [tokens for tokens in line_tokens.values()]
        parts: list[str] = []
        spans: list[tuple[int, int, float]] = []
        position = 0
        for line_index, tokens in enumerate(lines):
            if line_index:
                parts.append("\n")
                position += 1
            for token_index, (token, confidence) in enumerate(tokens):
                if token_index:
                    parts.append(" ")
                    position += 1
                start = position
                parts.append(token)
                position += len(token)
                spans.append((start, position, confidence))
        text = "".join(parts)
        matches = _candidate_spans(text)
        candidates_by_key: dict[str, tuple[str, float]] = {}
        for value, start, end in matches:
            if not is_valid_photo_jpc_candidate(value):
                continue
            key = normalize_jpc(value)
            confidences = [score for left, right, score in spans
                           if left < end and right > start]
            confidence = float(np.mean(confidences)) if confidences else 0.0
            if key and (key not in candidates_by_key or
                        confidence > candidates_by_key[key][1]):
                candidates_by_key[key] = (value, confidence)
        candidates = tuple(item[0] for item in candidates_by_key.values())
        confidence = min((item[1] for item in candidates_by_key.values()), default=0.0)
        if not candidates or confidence < 0.45:
            return JPCExtraction(candidates, "UNREADABLE", confidence, text)
        status = "MULTIPLE" if len(candidates) > 1 else "READABLE"
        return JPCExtraction(candidates, status, confidence, text)
    except Exception as error:
        return JPCExtraction((), "UNREADABLE", 0.0,
                             error=type(error).__name__)
