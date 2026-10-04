# STEP-65 — Real NOT OK Observation Extraction

## Purpose
Validate the full local processing path against the four real `VH-147` checklist photos supplied for acceptance testing.

## Changes
- Strengthened photographed checklist row geometry using long horizontal/vertical grid components derived from the image.
- Refined OK / NOT OK / Observation / Remarks boundaries from photographed grid rules instead of fixed coordinates.
- Streamlit batch processing now prefers unwarped (`perspective=False`) geometry for mobile checklist photos, with the existing detector as fallback.
- Added a targeted row-band OCR fast path for confirmed NOT OK rows.
- Targeted OCR uses a small vertical padding, positional text-line selection, multiple Tesseract PSMs, and conservative noise filtering.
- Strong targeted OCR results skip the slower OCR ensemble to reduce latency.
- No master observation-name matching was added; the Observation Point remains image-derived from the detected row.

## Real-photo acceptance test
Four supplied photos were processed together with:
- Inspector: `S. Harish`
- Entered JPC: `VH-147`

JPC validation:
- Photo 1: `VH-147` → MATCH
- Photo 2: no JPC → continuation accepted
- Photo 3: `VH-147` → MATCH
- Photo 4: no JPC → continuation accepted
- Batch JPC gate → PASS

Confirmed NOT OK observations:
1. Integration — Door bush missing
2. Testing — Manhole plate bolt not tight
3. Testing — Cleaning process
4. Stuffing — Foam improper

No extra observation was produced from OK/N/A marks in the tested pages.

## Performance
Local end-to-end processing of the four real photos (central adapter intentionally unconfigured): approximately 14 seconds.

## Automated tests
- `71 passed`
- `2 subtests passed`
- `0 failures`
- `2 existing deprecation warnings`

## Central write boundary
The real-photo acceptance test used the fail-closed unconfigured central adapter locally, so it intentionally returned `ERROR` instead of pretending a Google Sheet write succeeded.

The next deployment test must verify the same four observations are actually committed and read back from `PDI Traceability Central` through the configured Google Apps Script endpoint.
