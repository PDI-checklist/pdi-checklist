# PDI Photo Upload – Phase 2

இது paper PDI checklist-ஐ mobile-ல் photo upload செய்து, NOT OK observation candidates-ஐ உருவாக்கும் prototype.

## Workflow
1. Mobile link open செய்யவும்.
2. Inspector Name select/type செய்யவும்.
3. Checklist-ன் எல்லா page photos-ஐ upload செய்யவும்.
4. **Process Checklist**.
5. System OCR + image analysis மூலம் JPC/observation/OK-NOT OK candidate detection செய்யும்.
6. `REVIEW` items இருந்தால் சரிபார்க்கவும்.
7. Traceability Report Excel download செய்யவும்.
8. Phase-1 live Observation Log-ல் copy/paste செய்யவும்.

## முக்கிய பாதுகாப்பு
- All-OK checklist → output rows இல்லை.
- NOT OK → ஒரு observationக்கு ஒரு row.
- Mapping unambiguous இல்லையெனில் blank + Review Queue; system guess செய்யாது.
- இது first pilot version. Handwritten tick detection accuracy real production samples மூலம் validate செய்ய வேண்டும்.

## Mobile public link
இந்த package ஒரு deployable web app. இங்கிருந்து நேரடியாக permanent public URL host செய்யப்படவில்லை. Streamlit Cloud / internal server-ல் deploy செய்தால் mobile link கிடைக்கும்.

### Streamlit Cloud
1. இந்த folder-ஐ GitHub repository-ஆக upload செய்யவும்.
2. `app.py`-ஐ main file ஆக select செய்யவும்.
3. `requirements.txt` dependencies install ஆகும்.
4. Deploy செய்த URL-ஐ QR code ஆக print செய்து PDI area-ல் வைக்கலாம்.

## Revision update
`master/Observation_Checklist.xlsx` மாற்றிய revision-ஐ replace செய்யலாம். Production deploymentக்கு revision-wise template validation செய்ய வேண்டும்.

## Phase 2 limitation
இந்த prototype cloud AI document model-ஐ பயன்படுத்தவில்லை. OCR + computer vision local engine பயன்படுத்துகிறது. Handwritten ticks, page angle, lighting, shadows ஆகியவற்றால் சில rows `REVIEW` ஆகலாம். Production releaseக்கு Azure AI Document Intelligence / AI Builder போன்ற managed document model integration recommended.
