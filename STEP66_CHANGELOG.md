# STEP-66 — Central Write Readiness & Metadata Enrichment

## Implemented
- Streamlit pipeline now passes observation metadata into the central write layer.
- Department is populated from the existing Observation Log mapping when an exact observation mapping exists.
- Station defaults to `PDI Stage`.
- Status defaults to `Closed`.
- Cleared by defaults to the entered Inspector Name.
- Closure Remarks defaults to `Verified OK`.
- Defect Category is suggested from the existing mapping when available and remains optional; a missing mapping never drops the observation.
- Checklist date extraction is attempted from the JPC-bearing header page and normalized to `DD-MM-YYYY` when confidently recognized.
- Legacy local HTTP central API now accepts the same metadata fields for compatibility.
- Added Google Apps Script adapter contract test to verify token injection and metadata payload.

## Real-photo acceptance test
Using the supplied four real `VH-147` checklist pages with a local central-write capture:
- JPC batch gate: PASS.
- NOT OK observations: 4 expected rows recovered.
- Department mapping: Integration / Testing recovered for mapped observations.
- Station/Status/Cleared by/Closure Remarks populated as designed.
- Local central-write capture confirmed SUCCESS for 4 records.

## External boundary
The actual deployed Google Apps Script Web App + live Google Sheet write was **not executed from this environment** because outbound access to the user's Apps Script endpoint is unavailable here. Therefore STEP-66 does **not** claim a live Google Sheet end-to-end PASS. The package is deployment-ready and the next live test must verify the four rows in `Observation Log`.
