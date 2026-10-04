"""HTTP/SQLite integration tests for atomic central observation writes."""
from concurrent.futures import ThreadPoolExecutor
from http.server import ThreadingHTTPServer
import json
import sqlite3
from threading import Thread
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import cv2
import numpy as np
import pytest

from app_pipeline import process_upload_batch
from central_api import make_handler
from central_write_adapter import HTTPJSONCentralWriteAdapter, WriteStatus
from jpc_engine import JPCExtraction


@pytest.fixture
def central_service(tmp_path):
    database = tmp_path / "central.sqlite3"
    token = "integration-test-bearer-token"
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(database, token))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"
    adapter = HTTPJSONCentralWriteAdapter(base_url, token, timeout_seconds=10)
    try:
        yield database, adapter, base_url, token
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _record(text, **extra):
    return {"Observation": text, "OCR Status": "OK", "OCR Confidence": 0.96, **extra}


def _write(adapter, jpc="VH-3009", records=None, inspector="Inspector A"):
    return adapter.write_batch(jpc, inspector, records or [_record("Observation alpha")])


def test_new_jpc_and_observation_succeeds_and_reads_back(central_service):
    _, adapter, *_ = central_service
    result = _write(adapter)
    readback = adapter.read_jpc("VH3009")
    assert result.status is WriteStatus.SUCCESS
    assert result.records_written == 1
    assert readback.status is WriteStatus.SUCCESS
    assert [row["observation"] for row in readback.response["observations"]] == ["Observation alpha"]


def test_same_jpc_and_observation_twice_returns_duplicate(central_service):
    _, adapter, *_ = central_service
    first = _write(adapter)
    second = _write(adapter)
    assert first.status is WriteStatus.SUCCESS
    assert second.status is WriteStatus.DUPLICATE
    assert adapter.read_jpc("VH3009").records_written == 1


def test_concurrent_identical_submissions_commit_only_one_row(central_service):
    _, adapter, *_ = central_service
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(_write, adapter, inspector="Inspector 1"),
                   pool.submit(_write, adapter, inspector="Inspector 2")]
        results = [future.result() for future in futures]
    assert sorted(result.status.value for result in results) == ["DUPLICATE", "SUCCESS"]
    assert adapter.read_jpc("VH3009").records_written == 1


def test_same_jpc_with_different_observations_accepts_both(central_service):
    _, adapter, *_ = central_service
    first = _write(adapter, records=[_record("Observation alpha")])
    second = _write(adapter, records=[_record("Observation beta")])
    assert first.status is second.status is WriteStatus.SUCCESS
    assert adapter.read_jpc("VH3009").records_written == 2


def test_multi_observation_batch_commits_all_rows_together(central_service):
    _, adapter, *_ = central_service
    result = _write(adapter, records=[_record("Observation alpha"),
                                      _record("Observation beta"),
                                      _record("Observation gamma")])
    assert result.status is WriteStatus.SUCCESS
    assert result.records_written == 3
    assert adapter.read_jpc("VH3009").records_written == 3


def test_different_jpcs_with_same_observation_are_distinct(central_service):
    _, adapter, *_ = central_service
    first = _write(adapter, jpc="VH-3009", records=[_record("Observation alpha")])
    second = _write(adapter, jpc="VM-4001", records=[_record("Observation alpha")])
    assert first.status is second.status is WriteStatus.SUCCESS
    assert adapter.read_jpc("VH3009").records_written == 1
    assert adapter.read_jpc("VM4001").records_written == 1


@pytest.mark.parametrize("jpc", ["", "---", "12345", "VH-"])
def test_invalid_jpc_is_rejected(central_service, jpc):
    _, adapter, *_ = central_service
    assert _write(adapter, jpc=jpc).status is WriteStatus.REJECTED


def test_empty_observation_is_rejected(central_service):
    _, adapter, *_ = central_service
    result = _write(adapter, records=[_record("   ")])
    assert result.status is WriteStatus.REJECTED


def test_malformed_request_is_rejected_at_http_boundary(central_service):
    _, _, base_url, token = central_service
    body = json.dumps({"jpc": "VH-3009", "inspector": "Inspector", "observations": "not-a-list"}).encode()
    request = Request(base_url + "/v1/observations/batch", data=body, method="POST",
                      headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"})
    with pytest.raises(HTTPError) as error:
        urlopen(request, timeout=5)
    assert error.value.code == 422
    assert json.loads(error.value.read())["status"] == "REJECTED"


def test_database_failure_returns_error_and_rolls_back_complete_batch(central_service):
    database, adapter, *_ = central_service
    with sqlite3.connect(database) as connection:
        connection.execute("""
            CREATE TRIGGER simulated_store_failure BEFORE INSERT ON observation_records
            WHEN NEW.observation = 'Failure trigger'
            BEGIN SELECT RAISE(ABORT, 'simulated central failure'); END
        """)
    result = _write(adapter, records=[_record("Would otherwise insert"),
                                      _record("Failure trigger")])
    assert result.status is WriteStatus.ERROR
    assert adapter.read_jpc("VH3009").records_written == 0


def test_business_validation_failure_is_atomic(central_service):
    _, adapter, *_ = central_service
    result = _write(adapter, records=[_record("Valid observation"), _record("")])
    assert result.status is WriteStatus.REJECTED
    assert adapter.read_jpc("VH3009").records_written == 0


def test_payload_duplicates_are_collapsed_by_api_business_key(central_service):
    _, adapter, *_ = central_service
    result = _write(adapter, records=[_record("Observation Alpha"), _record("observation-alpha")])
    assert result.status is WriteStatus.SUCCESS
    assert result.records_written == 1
    assert adapter.read_jpc("VH3009").records_written == 1


def test_missing_authentication_is_rejected(central_service):
    _, _, base_url, _ = central_service
    body = json.dumps({"jpc": "VH-3009", "inspector": "Inspector",
                       "observations": [_record("Observation alpha")]}).encode()
    request = Request(base_url + "/v1/observations/batch", data=body, method="POST",
                      headers={"Content-Type": "application/json"})
    with pytest.raises(HTTPError) as error:
        urlopen(request, timeout=5)
    assert error.value.code == 401
    assert json.loads(error.value.read())["status"] == "REJECTED"


def test_missing_client_configuration_returns_error_not_success():
    adapter = HTTPJSONCentralWriteAdapter(base_url="", token="")
    result = _write(adapter)
    assert result.status is WriteStatus.ERROR
    assert result.records_written == 0


def test_upload_pipeline_confirms_success_only_after_real_http_commit(central_service):
    _, adapter, *_ = central_service
    image = np.full((40, 80, 3), 220, np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    photo = {"filename": "page.jpg", "data": encoded.tobytes()}

    result = process_upload_batch(
        "Inspector A", "VH-3009", [photo], central_adapter=adapter,
        jpc_extractor=lambda _: JPCExtraction(("VH 3009",), "READABLE", 0.99),
        row_detector=lambda _: object(),
        tick_detector=lambda *_: [SimpleNamespace(row_index=3, not_ok_marked=True)],
        observation_extractor=lambda *_args, **_kwargs: [SimpleNamespace(
            row_index=3, text="Observation recovered from source row",
            status="OK", confidence=0.97)],
    )
    assert result.status == "SUCCESS"
    assert result.write_result.records_written == 1
    readback = adapter.read_jpc("VH3009")
    assert readback.status is WriteStatus.SUCCESS
    assert readback.response["observations"][0]["observation"] == "Observation recovered from source row"
