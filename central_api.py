"""Persistent, authenticated HTTP central store for observation batches.

Run as one service on a central host with ``PDI_CENTRAL_DB_PATH`` and
``PDI_CENTRAL_API_TOKEN`` configured. SQLite's database transaction and
composite business key serialize writers across request threads/processes on
that host; no application-memory state is used for duplicate protection.
"""
from __future__ import annotations

from datetime import datetime, timezone
from contextlib import asynccontextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hmac
import json
import os
from pathlib import Path
import re
import sqlite3
import sys
import unicodedata
from typing import Any
from urllib.parse import parse_qs, urlparse

try:
    from fastapi import FastAPI, Header, Request
    from fastapi.responses import JSONResponse
except ImportError:  # Existing stdlib server remains usable in minimal environments.
    FastAPI = None
    Header = None
    Request = None
    JSONResponse = None


MAX_BODY_BYTES = 1_000_000
MAX_OBSERVATIONS = 250
_BUSINESS_JPC = re.compile(r"^(?=.*[A-Z])(?=.*\d)[A-Z0-9]{3,32}$")
_ALLOWED_RECORD_FIELDS = {
    "JPC Number", "Inspector Name", "Observation", "OCR Status",
    "OCR Confidence", "Source Photo", "Source Row", "Source Photos", "Source Rows",
    "Department", "Station", "Defect Category", "Status", "Cleared by", "Closure date", "Closure Remarks",
}


def _normalized_observation(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return "".join(character for character in normalized if character.isalnum())


def _response(status: str, message: str, *, records_written: int = 0,
              observations: list[dict[str, Any]] | None = None,
              duplicates: list[str] | None = None) -> dict[str, Any]:
    body: dict[str, Any] = {"status": status, "message": message,
                            "records_written": records_written}
    if observations is not None:
        body["observations"] = observations
    if duplicates is not None:
        body["duplicates"] = duplicates
    return body


def _connect(db_path: str | os.PathLike[str]) -> sqlite3.Connection:
    connection = sqlite3.connect(str(db_path), timeout=15.0, isolation_level=None)
    connection.execute("PRAGMA busy_timeout = 15000")
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize_store(db_path: str | os.PathLike[str]) -> None:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = _connect(path)
    try:
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("""
            CREATE TABLE IF NOT EXISTS observation_records (
                jpc_key TEXT NOT NULL,
                observation_key TEXT NOT NULL,
                jpc TEXT NOT NULL,
                inspector TEXT NOT NULL,
                observation TEXT NOT NULL,
                ocr_status TEXT NOT NULL,
                ocr_confidence REAL NOT NULL,
                source_photos TEXT NOT NULL,
                source_rows TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (jpc_key, observation_key)
            ) WITHOUT ROWID
        """)
    finally:
        connection.close()


def _validate_payload(payload: Any):
    if not isinstance(payload, dict) or set(payload) != {"jpc", "inspector", "observations"}:
        return None, None, None, _response("REJECTED", "Payload must contain exactly jpc, inspector, and observations.")
    jpc = payload.get("jpc")
    inspector = payload.get("inspector")
    records = payload.get("observations")
    if not isinstance(jpc, str) or not isinstance(inspector, str):
        return None, None, None, _response("REJECTED", "JPC and inspector must be text values.")
    jpc = re.sub(r"[^A-Z0-9]", "", jpc.upper())
    if not _BUSINESS_JPC.fullmatch(jpc):
        return None, None, None, _response("REJECTED", "JPC is missing or malformed.")
    inspector = inspector.strip()
    if not inspector or len(inspector) > 160:
        return None, None, None, _response("REJECTED", "Inspector is required and must be at most 160 characters.")
    if not isinstance(records, list) or not records or len(records) > MAX_OBSERVATIONS:
        return None, None, None, _response("REJECTED", "Observations must be a non-empty supported list.")

    validated = []
    for index, record in enumerate(records):
        if not isinstance(record, dict) or not set(record).issubset(_ALLOWED_RECORD_FIELDS):
            return None, None, None, _response("REJECTED", f"Observation {index} has an unsupported structure.")
        observation = record.get("Observation")
        if not isinstance(observation, str) or not observation.strip() or len(observation.strip()) > 2000:
            return None, None, None, _response("REJECTED", f"Observation {index} text is required and must be at most 2000 characters.")
        observation = observation.strip()
        observation_key = _normalized_observation(observation)
        if not observation_key:
            return None, None, None, _response("REJECTED", f"Observation {index} text contains no letters or digits.")
        row_jpc = record.get("JPC Number")
        if row_jpc is not None and (not isinstance(row_jpc, str) or
                                    re.sub(r"[^A-Z0-9]", "", row_jpc.upper()) != jpc):
            return None, None, None, _response("REJECTED", f"Observation {index} JPC conflicts with the batch JPC.")
        row_inspector = record.get("Inspector Name")
        if row_inspector is not None and row_inspector != inspector:
            return None, None, None, _response("REJECTED", f"Observation {index} inspector conflicts with the batch inspector.")
        status = record.get("OCR Status", "OK")
        confidence = record.get("OCR Confidence", 0.0)
        if not isinstance(status, str) or not status or len(status) > 32:
            return None, None, None, _response("REJECTED", f"Observation {index} OCR status is malformed.")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
            return None, None, None, _response("REJECTED", f"Observation {index} OCR confidence is malformed.")
        source_photo = record.get("Source Photo")
        source_row = record.get("Source Row")
        source_photos = record.get("Source Photos", [source_photo] if source_photo else [])
        source_rows = record.get("Source Rows", [source_row] if isinstance(source_row, int) else [])
        if (not isinstance(source_photos, list) or
                any(not isinstance(value, str) or len(value) > 512 for value in source_photos) or
                not isinstance(source_rows, list) or
                any(isinstance(value, bool) or not isinstance(value, int) for value in source_rows)):
            return None, None, None, _response("REJECTED", f"Observation {index} source metadata is malformed.")
        validated.append({
            "observation": observation,
            "observation_key": observation_key,
            "ocr_status": status,
            "ocr_confidence": float(confidence),
            "source_photos": json.dumps(source_photos, ensure_ascii=False),
            "source_rows": json.dumps(source_rows),
        })
    return jpc, inspector, validated, None


def write_batch(db_path: str | os.PathLike[str], payload: Any) -> tuple[int, dict[str, Any]]:
    """Validate then commit all new rows or none; duplicate key is JPC + observation."""
    jpc, inspector, records, invalid = _validate_payload(payload)
    if invalid is not None:
        return 422, invalid
    jpc_key = re.sub(r"[^A-Z0-9]", "", jpc)
    # Treat repeated observations inside one request as one business observation.
    unique: dict[str, dict[str, Any]] = {}
    for record in records:
        unique.setdefault(record["observation_key"], record)
    connection = _connect(db_path)
    try:
        connection.execute("BEGIN IMMEDIATE")
        existing = []
        for observation_key in unique:
            row = connection.execute(
                "SELECT observation FROM observation_records WHERE jpc_key=? AND observation_key=?",
                (jpc_key, observation_key),
            ).fetchone()
            if row is not None:
                existing.append(str(row[0]))
        if existing:
            connection.rollback()
            return 200, _response("DUPLICATE", "At least one requested observation already exists for this JPC.",
                                  duplicates=existing)
        created = datetime.now(timezone.utc).isoformat()
        for observation_key, record in unique.items():
            connection.execute("""
                INSERT INTO observation_records
                    (jpc_key, observation_key, jpc, inspector, observation,
                     ocr_status, ocr_confidence, source_photos, source_rows, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (jpc_key, observation_key, jpc, inspector, record["observation"],
                  record["ocr_status"], record["ocr_confidence"], record["source_photos"],
                  record["source_rows"], created))
        connection.commit()
        # Commit confirmation and read-back precede SUCCESS.
        written = connection.execute(
            "SELECT COUNT(*) FROM observation_records WHERE jpc_key=? AND created_at=?",
            (jpc_key, created),
        ).fetchone()[0]
        if written != len(unique):
            return 500, _response("ERROR", "Committed batch could not be confirmed by read-back.")
        return 201, _response("SUCCESS", "Observation batch committed.", records_written=written)
    except sqlite3.IntegrityError as error:
        connection.rollback()
        if "UNIQUE" in str(error).upper() or "PRIMARY KEY" in str(error).upper():
            return 200, _response("DUPLICATE", "Requested observation already exists for this JPC.")
        return 500, _response("ERROR", "Central store rejected the transaction.")
    except sqlite3.Error:
        connection.rollback()
        return 500, _response("ERROR", "Central store transaction failed.")
    finally:
        connection.close()


def read_jpc(db_path: str | os.PathLike[str], jpc: str) -> list[dict[str, Any]]:
    jpc_key = re.sub(r"[^A-Z0-9]", "", jpc.upper())
    connection = _connect(db_path)
    try:
        rows = connection.execute("""
            SELECT jpc, inspector, observation, ocr_status, ocr_confidence,
                   source_photos, source_rows, created_at
            FROM observation_records WHERE jpc_key=? ORDER BY observation_key
        """, (jpc_key,)).fetchall()
        return [{"jpc": row[0], "inspector": row[1], "observation": row[2],
                 "ocr_status": row[3], "ocr_confidence": row[4],
                 "source_photos": json.loads(row[5]), "source_rows": json.loads(row[6]),
                 "created_at": row[7]} for row in rows]
    finally:
        connection.close()


def make_handler(db_path: str | os.PathLike[str], token: str):
    initialize_store(db_path)

    class Handler(BaseHTTPRequestHandler):
        server_version = "PDI-Central-API/1.0"

        def _send(self, status: int, body: dict[str, Any]) -> None:
            encoded = json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(encoded)

        def _authorized(self) -> bool:
            supplied = self.headers.get("Authorization", "")
            expected = f"Bearer {token}"
            if not hmac.compare_digest(supplied, expected):
                self._send(401, _response("REJECTED", "Valid bearer authorization is required."))
                return False
            return True

        def do_POST(self):
            if not self._authorized():
                return
            if urlparse(self.path).path != "/v1/observations/batch":
                self._send(404, _response("REJECTED", "Unknown API route."))
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > MAX_BODY_BYTES:
                    self._send(413 if length > MAX_BODY_BYTES else 400,
                               _response("REJECTED", "Request body size is invalid."))
                    return
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
            except (ValueError, UnicodeDecodeError, json.JSONDecodeError):
                self._send(400, _response("REJECTED", "Request body must be valid JSON."))
                return
            status, body = write_batch(db_path, payload)
            self._send(status, body)

        def do_GET(self):
            if urlparse(self.path).path == "/health":
                try:
                    connection = _connect(db_path)
                    connection.execute("SELECT 1").fetchone()
                    connection.close()
                    self._send(200, {"status": "ok"})
                except sqlite3.Error:
                    self._send(503, {"status": "unavailable"})
                return
            if not self._authorized():
                return
            parsed = urlparse(self.path)
            if parsed.path != "/v1/observations":
                self._send(404, _response("REJECTED", "Unknown API route."))
                return
            jpc_values = parse_qs(parsed.query).get("jpc", [])
            if len(jpc_values) != 1 or not _BUSINESS_JPC.fullmatch(re.sub(r"[^A-Z0-9]", "", jpc_values[0].upper())):
                self._send(422, _response("REJECTED", "Exactly one valid JPC query is required."))
                return
            items = read_jpc(db_path, jpc_values[0])
            self._send(200, _response("SUCCESS", "Central rows read.",
                                      records_written=len(items), observations=items))

        def log_message(self, format, *args):
            print("central-api:", format % args, file=sys.stderr)

    return Handler


def create_app(db_path: str | os.PathLike[str] | None = None,
               token: str | None = None):
    """Create the production ASGI app, resolving Railway settings at startup."""
    if FastAPI is None:
        raise RuntimeError("FastAPI is required; install the declared requirements.txt dependencies.")

    database = str(db_path if db_path is not None else
                   os.environ.get("PDI_CENTRAL_DB_PATH", "/data/observations.sqlite3"))
    secret = token if token is not None else os.environ.get("PDI_CENTRAL_API_TOKEN", "")
    @asynccontextmanager
    async def lifespan(_api):
        if len(secret) < 32:
            raise RuntimeError("PDI_CENTRAL_API_TOKEN must be configured with at least 32 characters.")
        if os.environ.get("RAILWAY_ENVIRONMENT"):
            mount_path = os.environ.get("RAILWAY_VOLUME_MOUNT_PATH")
            if not mount_path:
                raise RuntimeError("Attach a Railway persistent volume before starting the central API.")
            try:
                Path(database).resolve().relative_to(Path(mount_path).resolve())
            except ValueError as error:
                raise RuntimeError("PDI_CENTRAL_DB_PATH must be inside the Railway volume mount.") from error
        initialize_store(database)
        yield

    api = FastAPI(title="PDI Central API", docs_url=None, redoc_url=None,
                  openapi_url=None, lifespan=lifespan)

    @api.get("/health")
    def health():
        try:
            connection = _connect(database)
            connection.execute("SELECT 1").fetchone()
            connection.close()
            return {"status": "ok"}
        except sqlite3.Error:
            return JSONResponse(status_code=503, content={"status": "unavailable"})

    def authorize(value: str | None):
        if not value or not hmac.compare_digest(value, f"Bearer {secret}"):
            return JSONResponse(status_code=401,
                                content=_response("REJECTED", "Valid bearer authorization is required."),
                                headers={"Cache-Control": "no-store"})
        return None

    @api.post("/v1/observations/batch")
    async def submit_batch(request: Request):
        denied = authorize(request.headers.get("authorization"))
        if denied is not None:
            return denied
        try:
            length = int(request.headers.get("content-length", "0"))
            if length <= 0 or length > MAX_BODY_BYTES:
                return JSONResponse(status_code=413 if length > MAX_BODY_BYTES else 400,
                                    content=_response("REJECTED", "Request body size is invalid."))
            raw = await request.body()
            if len(raw) > MAX_BODY_BYTES:
                return JSONResponse(status_code=413,
                                    content=_response("REJECTED", "Request body size is invalid."))
            payload = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError, json.JSONDecodeError):
            return JSONResponse(status_code=400,
                                content=_response("REJECTED", "Request body must be valid JSON."))
        status, body = write_batch(database, payload)
        return JSONResponse(status_code=status, content=body,
                            headers={"Cache-Control": "no-store"})

    @api.get("/v1/observations")
    def read_observations(jpc: str, authorization: str | None = Header(default=None, alias="Authorization")):
        denied = authorize(authorization)
        if denied is not None:
            return denied
        normalized = re.sub(r"[^A-Z0-9]", "", jpc.upper())
        if not _BUSINESS_JPC.fullmatch(normalized):
            return JSONResponse(status_code=422,
                                content=_response("REJECTED", "Exactly one valid JPC query is required."))
        items = read_jpc(database, jpc)
        return JSONResponse(status_code=200,
                            content=_response("SUCCESS", "Central rows read.",
                                              records_written=len(items), observations=items),
                            headers={"Cache-Control": "no-store"})

    return api


app = create_app() if FastAPI is not None else None


def serve(db_path: str, token: str, host: str = "0.0.0.0", port: int = 8765) -> None:
    if not db_path or not token:
        raise RuntimeError("PDI_CENTRAL_DB_PATH and PDI_CENTRAL_API_TOKEN are required")
    server = ThreadingHTTPServer((host, port), make_handler(db_path, token))
    print(f"PDI central API listening on {host}:{port}; persistent DB: {db_path}")
    server.serve_forever()


if __name__ == "__main__":
    serve(os.environ.get("PDI_CENTRAL_DB_PATH", ""),
          os.environ.get("PDI_CENTRAL_API_TOKEN", ""),
          os.environ.get("PDI_CENTRAL_API_HOST", "0.0.0.0"),
          int(os.environ.get("PDI_CENTRAL_API_PORT", "8765")))
