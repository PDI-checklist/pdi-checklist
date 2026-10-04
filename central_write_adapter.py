"""HTTP client for the central observation API and explicit write results."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
import os
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen
from typing import Any, Protocol, Sequence


class WriteStatus(str, Enum):
    SUCCESS = "SUCCESS"
    DUPLICATE = "DUPLICATE"
    REJECTED = "REJECTED"
    ERROR = "ERROR"


@dataclass(frozen=True)
class CentralWriteResult:
    status: WriteStatus
    message: str
    records_written: int = 0
    response: Any = None


class CentralWriteAdapter(Protocol):
    def write_batch(self, jpc: str, inspector: str,
                    observations: Sequence[dict[str, Any]]) -> CentralWriteResult: ...


class UnconfiguredCentralWriteAdapter:
    """Fail closed until a real central API endpoint/client is configured."""

    def write_batch(self, jpc: str, inspector: str,
                    observations: Sequence[dict[str, Any]]) -> CentralWriteResult:
        return CentralWriteResult(
            WriteStatus.ERROR,
            "Central API is not configured; no records were written.",
            records_written=0,
        )


class HTTPJSONCentralWriteAdapter:
    """Authenticated adapter for ``central_api.py`` or a compatible API."""

    def __init__(self, base_url: str | None = None, token: str | None = None,
                 timeout_seconds: float = 10.0):
        self.base_url = (base_url or os.getenv("PDI_GOOGLE_APPS_SCRIPT_URL", "") or os.getenv("PDI_CENTRAL_API_URL", "")).rstrip("/")
        self.token = token if token is not None else (os.getenv("PDI_GOOGLE_APPS_SCRIPT_TOKEN", "") or os.getenv("PDI_CENTRAL_API_TOKEN", ""))
        self.apps_script = bool(os.getenv("PDI_GOOGLE_APPS_SCRIPT_URL")) or "script.google.com" in self.base_url
        self.timeout_seconds = timeout_seconds

    def _request(self, path: str, *, method: str = "GET", payload: Any = None):
        if not self.base_url or not self.token:
            raise RuntimeError("PDI_CENTRAL_API_URL and PDI_CENTRAL_API_TOKEN are required")
        parsed_url = urlparse(self.base_url)
        if (parsed_url.scheme != "https" and
                parsed_url.hostname not in {"localhost", "127.0.0.1", "::1"}):
            raise RuntimeError("Central API must use HTTPS outside loopback tests")
        body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        if self.apps_script:
            if method == "POST":
                payload = dict(payload or {})
                payload["token"] = self.token
                body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                request = Request(self.base_url, data=body, method="POST",
                                  headers={"Content-Type": "application/json"})
            else:
                raise RuntimeError("Google Apps Script adapter uses POST requests")
        else:
            request = Request(self.base_url + path, data=body, method=method,
                              headers={"Authorization": f"Bearer {self.token}",
                                       "Content-Type": "application/json"})
        with urlopen(request, timeout=self.timeout_seconds) as response:
            return response.status, json.loads(response.read().decode("utf-8"))

    @staticmethod
    def _result(body: Any, http_status: int) -> CentralWriteResult:
        try:
            status = WriteStatus(str(body["status"]))
            message = str(body.get("message", ""))
            count = int(body.get("records_written", 0))
            return CentralWriteResult(status, message, count, body)
        except (KeyError, TypeError, ValueError):
            return CentralWriteResult(WriteStatus.ERROR,
                                      f"Central API returned an invalid response (HTTP {http_status}).",
                                      response=body)

    def write_batch(self, jpc: str, inspector: str,
                    observations: Sequence[dict[str, Any]]) -> CentralWriteResult:
        payload = {"jpc": jpc, "inspector": inspector,
                   "observations": list(observations)}
        try:
            if self.apps_script:
                payload["action"] = "write_batch"
                status, body = self._request("", method="POST", payload=payload)
            else:
                status, body = self._request("/v1/observations/batch", method="POST", payload=payload)
            return self._result(body, status)
        except HTTPError as error:
            try:
                body = json.loads(error.read().decode("utf-8"))
            except Exception:
                body = {"status": "ERROR", "message": f"Central API returned HTTP {error.code}."}
            return self._result(body, error.code)
        except (URLError, TimeoutError, OSError, RuntimeError, ValueError) as error:
            return CentralWriteResult(WriteStatus.ERROR,
                                      f"Central API write failed: {type(error).__name__}")

    def read_jpc(self, jpc: str) -> CentralWriteResult:
        """Read rows back for operational verification; no client-side cache."""
        try:
            path = "/v1/observations?" + urlencode({"jpc": jpc})
            if self.apps_script:
                status, body = self._request("", method="POST", payload={"action": "read", "jpc": jpc})
            else:
                status, body = self._request(path)
            if status != 200 or body.get("status") != "SUCCESS":
                return CentralWriteResult(WriteStatus.ERROR, "Central API read-back failed.", response=body)
            return CentralWriteResult(WriteStatus.SUCCESS, "Central rows read back.",
                                      len(body.get("observations", [])), body)
        except (HTTPError, URLError, TimeoutError, OSError, RuntimeError, ValueError) as error:
            return CentralWriteResult(WriteStatus.ERROR,
                                      f"Central API read-back failed: {type(error).__name__}")
