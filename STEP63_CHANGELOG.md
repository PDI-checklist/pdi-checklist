# STEP-63 — Batch JPC Validation Logic

Implemented the requested batch-level JPC rule:

- User enters one JPC number with mandatory hyphen format.
- Upload batch may contain multiple checklist photos.
- At least one photo must contain a readable JPC matching the entered JPC.
- Photos without a readable JPC are allowed as continuation pages.
- A confidently detected different JPC anywhere in the batch rejects the entire batch before checklist processing.
- Multiple JPC candidates in one photo reject the batch as ambiguous.
- If no photo contains a readable matching JPC, the batch is rejected before dynamic processing.
- After JPC batch validation passes, all uploaded photos are processed for NOT OK ticks.
- UI label changed from per-photo validation to batch JPC validation.

Validation performed locally:
- 71 tests passed
- 2 subtests passed
- 0 test failures
- Python compilation passed for application modules

Important:
The supplied real handwritten `VH-147` photos still expose a separate OCR-recognition limitation in the current Tesseract-based JPC extractor. STEP-63 changes the batch acceptance logic, but it does not claim that handwritten JPC OCR is solved. The next real-photo acceptance test must confirm `VH-147` is actually recognized before central-write testing is declared PASS.
