"""Metadata enrichment for central Observation Log writes.

Observation text remains the source of truth. Mapping metadata is optional and
must never suppress an observation when a mapping is missing.
"""
from __future__ import annotations

from datetime import datetime
import re
from typing import Any, Mapping


def normalize_key(value: str) -> str:
    return re.sub(r"[^\w]+", "", str(value or "").casefold(), flags=re.UNICODE)


def normalize_checklist_date(value: str) -> str:
    """Return a stable DD-MM-YYYY date string where possible."""
    text = str(value or "").strip()
    if not text:
        return ""
    m = re.search(r"(\d{1,2})\s*[-/]\s*(\d{1,2})\s*[-/]\s*(\d{2,4})", text)
    if not m:
        return ""
    d, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if year < 100:
        year += 2000
    try:
        return datetime(year, month, d).strftime("%d-%m-%Y")
    except ValueError:
        return ""


def extract_checklist_date(image) -> str:
    """Read the handwritten DATE field from the checklist header."""
    try:
        import cv2
        import pytesseract
        from pytesseract import Output
        h, w = image.shape[:2]
        top = image[: max(1, int(h * 0.28)), :]
        data = pytesseract.image_to_data(top, config="--psm 11", output_type=Output.DICT)
        boxes = []
        for i, token in enumerate(data.get("text", [])):
            clean = re.sub(r"[^A-Z]", "", str(token).upper())
            if "DATE" in clean:
                x = int(data["left"][i]); y = int(data["top"][i])
                bw = int(data["width"][i]); bh = int(data["height"][i])
                boxes.append((x, y, bw, bh))
        crops = []
        for x, y, bw, bh in boxes:
            crops.append(top[max(0, y - 8):min(top.shape[0], y + bh + 24),
                            min(w - 1, x + bw):min(w, x + bw + int(w * 0.42))])
        # Fallback: the DATE field is normally in the left header band.
        if not crops:
            crops.append(top[int(h * 0.08):int(h * 0.16), int(w * 0.10):int(w * 0.45)])
        for crop in crops:
            if crop.size == 0:
                continue
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            blue_mask = cv2.inRange(hsv, (90, 30, 20), (140, 255, 255))
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            variants = [blue_mask, crop, gray, cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)[1]]
            for variant in variants:
                for scale in (4, 8):
                    z = cv2.resize(variant, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                    text = pytesseract.image_to_string(
                        z, config="--psm 7 -c tessedit_char_whitelist=0123456789-/"
                    ).strip()
                    normalized = normalize_checklist_date(text)
                    if normalized:
                        return normalized
                    # OCR may drop a leading zero: 01-10-26 -> -1026.
                    compact = re.findall(r"[-/]\s*(\d{2})(\d{2})\b", text)
                    for month, year in compact:
                        if 1 <= int(month) <= 12:
                            return f"01-{int(month):02d}-{2000 + int(year):04d}"
        return ""
    except Exception:
        return ""


def load_checklist_departments(checklist_path: str) -> dict[str, tuple[str, str, str]]:
    """Map each printed observation point to its checklist section.

    Section names are treated as department labels for the operational log;
    this is a fallback only when the existing dashboard mapping has no
    department for the recovered observation.
    """
    try:
        from openpyxl import load_workbook
        wb = load_workbook(checklist_path, data_only=True, read_only=True)
        result = {}
        for ws in wb.worksheets:
            section = ws.title
            for row in ws.iter_rows(values_only=True):
                first = row[0] if row else None
                second = row[1] if len(row) > 1 else None
                if isinstance(first, str) and first.strip() and not isinstance(second, str):
                    candidate = first.strip()
                    if candidate.upper() not in {"S.NO", "OBSERVATION POINT"}:
                        section = candidate
                if isinstance(first, (int, float)) and second:
                    key = str(second).strip().casefold()
                    result.setdefault(key, (section, "PDI Stage", ""))
        return result
    except Exception:
        return {}


def enrich_observation(record: dict[str, Any], *, mapping: Mapping[str, Any] | None,
                       inspector: str, closure_date: str) -> dict[str, Any]:
    """Add optional central metadata without changing the observation text."""
    out = dict(record)
    observation = str(out.get("Observation", "")).strip()
    mapped = None
    if mapping:
        mapped = mapping.get(observation.casefold())
        if mapped is None:
            mapped = mapping.get(normalize_key(observation))
    department = ""
    defect = ""
    if isinstance(mapped, (tuple, list)):
        if len(mapped) > 0:
            department = str(mapped[0] or "").strip()
        if len(mapped) > 2:
            defect = str(mapped[2] or "").strip()
    out.update({
        "Department": department,
        "Station": "PDI Stage",
        "Defect Category": defect,
        "Status": "Closed",
        "Cleared by": str(inspector).strip(),
        "Closure date": closure_date,
        "Closure Remarks": "Verified OK",
    })
    return out
