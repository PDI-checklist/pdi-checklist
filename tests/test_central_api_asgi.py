"""Deployment contract checks for the Railway FastAPI entry point."""
import json
from pathlib import Path
import socket
from threading import Thread
from time import monotonic, sleep
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

fastapi = pytest.importorskip("fastapi")
uvicorn = pytest.importorskip("uvicorn")

from central_api import create_app


def test_railway_deployment_config_declares_asgi_health_and_variables():
    root = Path(__file__).resolve().parents[1]
    railway = (root / ".railway" / "railway.ts").read_text(encoding="utf-8")
    api_variables = (root / ".env.railway-api.example").read_text(encoding="utf-8")
    client_variables = (root / ".env.railway-client.example").read_text(encoding="utf-8")
    assert 'start: "uvicorn central_api:app --host 0.0.0.0 --port $PORT"' in railway
    assert 'healthcheck: "/health"' in railway
    assert 'PDI_CENTRAL_DB_PATH: "/data/observations.sqlite3"' in railway
    assert '"/data": centralData' in railway
    assert "replicas: 1" in railway
    assert "PDI_CENTRAL_API_TOKEN=" in api_variables
    assert "PDI_CENTRAL_API_URL=" in client_variables
    assert "PDI_CENTRAL_API_TOKEN=" in client_variables
    assert "PDI_CENTRAL_DB_PATH=/data/observations.sqlite3" in api_variables


def test_asgi_health_auth_write_and_readback(tmp_path):
    token = "a-production-shaped-test-secret-with-32-plus-characters"
    app = create_app(tmp_path / "api.sqlite3", token)
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    port = listener.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="critical", access_log=False))
    thread = Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{port}"
    try:
        deadline = monotonic() + 10
        while not server.started and monotonic() < deadline:
            sleep(0.05)
        assert server.started, "ASGI service did not start"
        with urlopen(base + "/health", timeout=3) as response:
            assert response.status == 200
            assert json.loads(response.read()) == {"status": "ok"}

        payload = json.dumps({
            "jpc": "VH-3009", "inspector": "Inspector",
            "observations": [{"Observation": "Dynamic row text", "OCR Status": "OK",
                              "OCR Confidence": 0.95}],
        }).encode()
        request = Request(base + "/v1/observations/batch", data=payload, method="POST",
                          headers={"Content-Type": "application/json"})
        with pytest.raises(HTTPError) as denied:
            urlopen(request, timeout=3)
        assert denied.value.code == 401
        denied.value.close()

        request.add_header("Authorization", f"Bearer {token}")
        with urlopen(request, timeout=3) as response:
            assert json.loads(response.read())["status"] == "SUCCESS"
        read = Request(base + "/v1/observations?jpc=VH3009",
                       headers={"Authorization": f"Bearer {token}"})
        with urlopen(read, timeout=3) as response:
            body = json.loads(response.read())
            assert body["status"] == "SUCCESS"
            assert body["observations"][0]["observation"] == "Dynamic row text"
    finally:
        server.should_exit = True
        thread.join(timeout=5)
        listener.close()


