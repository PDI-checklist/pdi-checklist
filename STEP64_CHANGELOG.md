# STEP-64 — Handwritten JPC Recovery + Same-Batch Anchor Validation

Implemented and tested:

- Targeted JPC label detection in the upper checklist header instead of relying only on whole-page OCR.
- Handwritten JPC value recovery using focused OCR plus pen-component digit recovery.
- Formatting normalization remains compatible with the existing JPC rules.
- Once one uploaded photo confirms the entered JPC, later photos with the same JPC field can be confirmed against the same-batch visual JPC anchor.
- Photos without a JPC field remain valid continuation pages.
- A clear OCR-detected different JPC still rejects the whole batch.
- JPC validation completes before dynamic row/tick/observation processing and before central write.

Validation:
- Full automated suite: 71 passed + 2 subtests, 0 failures.
- Python compilation: PASS.
- Real uploaded VH-147 batch JPC gate test (4 photos, downstream checklist processing stubbed): PASS.
  - Photo 1: MATCH (`VH147` normalized)
  - Photo 2: continuation page accepted
  - Photo 3: matched to confirmed JPC anchor (`VH-147`)
  - Photo 4: continuation page accepted
  - Batch gate: PASS

Important limitation:
- The complete 4-photo real checklist run was also attempted. It reached the dynamic checklist stage after JPC validation, but the current row engine raised `ValueError: checklist columns were not detected` on the supplied real photo. Therefore STEP-64 does NOT claim the 4 observation rows or Google Sheet write are PASS yet. That is the next separate real-photo row/geometry issue.
