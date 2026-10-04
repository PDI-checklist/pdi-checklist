/**
 * PDI Traceability Central - Google Apps Script Web App
 * Sheet: PDI Traceability Central
 * Tab:   Observation Log
 *
 * Visible columns (A:J):
 * Timestamp | JPC Number | Observation | Department | Station |
 * Defect Category | Status | Cleared by | Closure date | Closure Remarks
 */

const SHEET_NAME = 'Observation Log';
const EXPECTED_HEADERS = [
  'Timestamp', 'JPC Number', 'Observation', 'Department', 'Station',
  'Defect Category', 'Status', 'Cleared by', 'Closure date', 'Closure Remarks'
];
const STATION_DEFAULT = 'PDI Stage';
const STATUS_DEFAULT = 'Closed';
const CLOSURE_REMARKS_DEFAULT = 'Verified OK';

function doGet(e) {
  return jsonResponse_({status: 'ok', service: 'PDI Traceability Central'});
}

function doPost(e) {
  try {
    const body = parseJsonBody_(e);
    authorize_(body.token);
    if (body.action === 'health') {
      return jsonResponse_({status: 'ok', service: 'PDI Traceability Central'});
    }
    if (body.action === 'read') {
      return jsonResponse_(readJpc_(body));
    }
    if (body.action === 'write_batch' || !body.action) {
      return jsonResponse_(writeBatch_(body));
    }
    return jsonResponse_({status: 'REJECTED', message: 'Unknown action.'}, 400);
  } catch (err) {
    return jsonResponse_({status: 'ERROR', message: safeError_(err)}, 500);
  }
}

function parseJsonBody_(e) {
  if (!e || !e.postData || !e.postData.contents) {
    throw new Error('Request body is required.');
  }
  let body;
  try {
    body = JSON.parse(e.postData.contents);
  } catch (err) {
    throw new Error('Request body must be valid JSON.');
  }
  if (!body || typeof body !== 'object') {
    throw new Error('Request body must be a JSON object.');
  }
  return body;
}

function authorize_(supplied) {
  const expected = PropertiesService.getScriptProperties().getProperty('API_TOKEN');
  if (!expected || !supplied || !constantTimeEqual_(String(supplied), String(expected))) {
    throw new HttpError_(401, 'Valid API token is required.');
  }
}

function constantTimeEqual_(a, b) {
  if (a.length !== b.length) return false;
  let diff = 0;
  for (let i = 0; i < a.length; i++) diff |= a.charCodeAt(i) ^ b.charCodeAt(i);
  return diff === 0;
}

function spreadsheet_() {
  const id = PropertiesService.getScriptProperties().getProperty('SPREADSHEET_ID');
  if (id) return SpreadsheetApp.openById(id);
  return SpreadsheetApp.getActiveSpreadsheet();
}

function sheet_() {
  const sheet = spreadsheet_().getSheetByName(SHEET_NAME);
  if (!sheet) throw new Error('Sheet "' + SHEET_NAME + '" was not found.');
  validateHeaders_(sheet);
  return sheet;
}

function validateHeaders_(sheet) {
  if (sheet.getLastColumn() < EXPECTED_HEADERS.length) {
    throw new Error('Observation Log must contain the required A:J columns.');
  }
  const actual = sheet.getRange(1, 1, 1, EXPECTED_HEADERS.length).getValues()[0]
    .map(v => String(v).trim());
  for (let i = 0; i < EXPECTED_HEADERS.length; i++) {
    if (actual[i] !== EXPECTED_HEADERS[i]) {
      throw new Error('Observation Log header mismatch at column ' + (i + 1) + '.');
    }
  }
}

function writeBatch_(body) {
  const jpc = String(body.jpc || '').trim();
  const inspector = String(body.inspector || '').trim();
  const observations = body.observations;
  if (!isValidJpc_(jpc)) {
    return {status: 'REJECTED', message: 'Invalid JPC. Use 2-3 letters, a mandatory hyphen, then digits.'};
  }
  if (!inspector) return {status: 'REJECTED', message: 'Inspector Name is required.'};
  if (!Array.isArray(observations) || observations.length === 0) {
    return {status: 'REJECTED', message: 'Observations must be a non-empty array.'};
  }
  if (observations.length > 500) {
    return {status: 'REJECTED', message: 'Batch is too large.'};
  }

  const lock = LockService.getScriptLock();
  lock.waitLock(15000);
  try {
    const sheet = sheet_();
    const existing = existingKeys_(sheet);
    const seen = {};
    const rows = [];
    const now = new Date();
    const canonicalJpc = normalizeJpc_(jpc);

    observations.forEach(item => {
      if (!item || typeof item !== 'object') return;
      const observation = String(item['Observation'] || item.observation || '').trim();
      if (!observation) return;
      const key = canonicalJpc + '|' + normalizeObservation_(observation);
      if (existing[key] || seen[key]) return;
      seen[key] = true;

      rows.push([
        now,
        jpc,
        observation,
        String(item['Department'] || item.department || '').trim(),
        STATION_DEFAULT,
        String(item['Defect Category'] || item.defect_category || '').trim(),
        STATUS_DEFAULT,
        String(item['Cleared by'] || item.cleared_by || '').trim(),
        normalizeDate_(item['Closure date'] || item.closure_date || body.closure_date || ''),
        String(item['Closure Remarks'] || item.closure_remarks || CLOSURE_REMARKS_DEFAULT).trim()
      ]);
    });

    if (rows.length === 0) {
      return {status: 'DUPLICATE', message: 'All requested observations already exist for this JPC.', records_written: 0};
    }

    const startRow = Math.max(sheet.getLastRow() + 1, 2);
    sheet.getRange(startRow, 1, rows.length, EXPECTED_HEADERS.length).setValues(rows);
    SpreadsheetApp.flush();

    // Commit confirmation: read back exactly the rows written.
    const readBack = sheet.getRange(startRow, 1, rows.length, EXPECTED_HEADERS.length).getValues();
    let confirmed = 0;
    readBack.forEach(row => {
      const key = normalizeJpc_(String(row[1])) + '|' + normalizeObservation_(String(row[2]));
      if (seen[key] && String(row[1]).trim() && String(row[2]).trim()) confirmed++;
    });
    if (confirmed !== rows.length) {
      return {status: 'ERROR', message: 'Committed batch could not be fully confirmed by read-back.', records_written: 0};
    }
    return {
      status: 'SUCCESS',
      message: 'Observation batch committed and confirmed.',
      records_written: confirmed
    };
  } finally {
    lock.releaseLock();
  }
}

function readJpc_(body) {
  const jpc = String(body.jpc || '').trim();
  if (!isValidJpc_(jpc)) return {status: 'REJECTED', message: 'Invalid JPC.'};
  const sheet = sheet_();
  const lastRow = sheet.getLastRow();
  if (lastRow < 2) return {status: 'SUCCESS', message: 'No rows found.', observations: []};
  const values = sheet.getRange(2, 1, lastRow - 1, EXPECTED_HEADERS.length).getValues();
  const key = normalizeJpc_(jpc);
  const observations = values.filter(row => normalizeJpc_(String(row[1])) === key)
    .map(row => ({
      timestamp: row[0], jpc: row[1], observation: row[2], department: row[3],
      station: row[4], defect_category: row[5], status: row[6], cleared_by: row[7],
      closure_date: row[8], closure_remarks: row[9]
    }));
  return {status: 'SUCCESS', message: 'Central rows read.', records_written: observations.length, observations: observations};
}

function existingKeys_(sheet) {
  const result = {};
  const lastRow = sheet.getLastRow();
  if (lastRow < 2) return result;
  // One read for both business-key columns. This is the main quota/performance optimization.
  const values = sheet.getRange(2, 2, lastRow - 1, 2).getValues();
  values.forEach(row => {
    const jpc = normalizeJpc_(String(row[0] || ''));
    const observation = normalizeObservation_(String(row[1] || ''));
    if (jpc && observation) result[jpc + '|' + observation] = true;
  });
  return result;
}

function isValidJpc_(value) {
  return /^[A-Z]{2,3}-\d+$/.test(String(value).trim());
}

function normalizeJpc_(value) {
  return String(value || '').toUpperCase().replace(/[^A-Z0-9]/g, '');
}

function normalizeObservation_(value) {
  return String(value || '').normalize('NFKC').toLowerCase().replace(/[^\p{L}\p{N}]+/gu, '');
}

function normalizeDate_(value) {
  if (!value) return '';
  if (Object.prototype.toString.call(value) === '[object Date]' && !isNaN(value.getTime())) return value;
  const parsed = new Date(String(value));
  return isNaN(parsed.getTime()) ? String(value) : parsed;
}

function safeError_(err) {
  if (err instanceof HttpError_) return err.message;
  return err && err.message ? String(err.message) : 'Unexpected server error.';
}

function jsonResponse_(body, statusCode) {
  // Apps Script Web Apps do not expose a reliable custom HTTP status API from ContentService.
  // The JSON status field is therefore authoritative for the client.
  return ContentService.createTextOutput(JSON.stringify(body))
    .setMimeType(ContentService.MimeType.JSON);
}

function HttpError_(status, message) {
  this.status = status;
  this.message = message;
}
