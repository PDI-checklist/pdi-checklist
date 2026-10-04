# STEP-60 — Google Apps Script Central API

This Web App is the free central write layer for the PDI Traceability Central Google Sheet.

## Sheet contract

Spreadsheet: `PDI Traceability Central`
Tab: `Observation Log`

A:J must be exactly:

1. Timestamp
2. JPC Number
3. Observation
4. Department
5. Station
6. Defect Category
7. Status
8. Cleared by
9. Closure date
10. Closure Remarks

## Script Properties

Set these in Apps Script → Project Settings → Script Properties:

- `API_TOKEN` — long random secret, preferably 32+ characters
- `SPREADSHEET_ID` — Google Sheet ID (recommended; avoids ambiguity)

Do not put the token in source code or screenshots.

## Deploy

Deploy → New deployment → Web app.

- Execute as: Me
- Who has access: Anyone

Copy the Web App URL into Streamlit secrets later.

## Request contract

POST JSON:

```json
{
  "action": "write_batch",
  "token": "YOUR_SECRET",
  "jpc": "VH-3009",
  "inspector": "S. Harish",
  "observations": [
    {
      "Observation": "Door bush missing",
      "Department": "Integration",
      "Defect Category": "Missing Part",
      "Cleared by": "S. Harish",
      "Closure date": "2026-10-01"
    }
  ]
}
```

The server automatically sets:

- Timestamp = server time
- Station = `PDI Stage`
- Status = `Closed`
- Closure Remarks = `Verified OK`

The backend does **not** invent Department, Defect Category, Cleared by, or Closure date when they are not supplied.

## Duplicate protection

Business key: normalized `JPC Number + Observation`.

A short server-side `LockService` lock protects the read/check/append transaction for concurrent users. It is internal only and creates no visible spreadsheet lock.

The server reads B:C once, filters duplicates in memory, writes the new batch in one `setValues()` call, flushes, and reads back the written rows before returning `SUCCESS`.

## Important limitation at STEP-60

The current photo pipeline does not yet extract all checklist header metadata (Department/Cleared by/Closure date) from the photo. Therefore this step provides the central API contract and safe defaults, but it is not yet the final end-to-end metadata extraction step.

## Streamlit Community Cloud secrets

In the Streamlit app settings, add these two secrets (TOML):

```toml
PDI_GOOGLE_APPS_SCRIPT_URL = "https://script.google.com/macros/s/YOUR_DEPLOYMENT_ID/exec"
PDI_GOOGLE_APPS_SCRIPT_TOKEN = "YOUR_SAME_SECRET_AS_APPS_SCRIPT_API_TOKEN"
```

Keep the token private. The app reads these values through `st.secrets`; it does not display them. Local runs may use the same names as environment variables.
