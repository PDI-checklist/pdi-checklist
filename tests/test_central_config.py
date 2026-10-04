from central_config import build_central_adapter, resolve_config
from central_write_adapter import HTTPJSONCentralWriteAdapter, UnconfiguredCentralWriteAdapter

def test_streamlit_secrets_take_priority_over_environment():
    secrets = {
        "PDI_GOOGLE_APPS_SCRIPT_URL": "https://script.google.com/macros/s/example/exec",
        "PDI_GOOGLE_APPS_SCRIPT_TOKEN": "secret-from-streamlit",
    }
    env = {
        "PDI_GOOGLE_APPS_SCRIPT_URL": "https://old.example/exec",
        "PDI_GOOGLE_APPS_SCRIPT_TOKEN": "old-token",
    }
    assert resolve_config(secrets, env) == (secrets["PDI_GOOGLE_APPS_SCRIPT_URL"], secrets["PDI_GOOGLE_APPS_SCRIPT_TOKEN"])
    adapter = build_central_adapter(secrets, env)
    assert isinstance(adapter, HTTPJSONCentralWriteAdapter)
    assert adapter.base_url == secrets["PDI_GOOGLE_APPS_SCRIPT_URL"]
    assert adapter.token == secrets["PDI_GOOGLE_APPS_SCRIPT_TOKEN"]

def test_environment_fallback_works_for_local_run():
    env = {
        "PDI_GOOGLE_APPS_SCRIPT_URL": "https://script.google.com/macros/s/example/exec",
        "PDI_GOOGLE_APPS_SCRIPT_TOKEN": "local-secret",
    }
    adapter = build_central_adapter({}, env)
    assert isinstance(adapter, HTTPJSONCentralWriteAdapter)

def test_missing_configuration_fails_closed():
    adapter = build_central_adapter({}, {})
    assert isinstance(adapter, UnconfiguredCentralWriteAdapter)


def test_apps_script_adapter_sends_batch_contract_and_maps_success(monkeypatch):
    from central_write_adapter import WriteStatus
    adapter = build_central_adapter({
        "PDI_GOOGLE_APPS_SCRIPT_URL": "https://script.google.com/macros/s/example/exec",
        "PDI_GOOGLE_APPS_SCRIPT_TOKEN": "secret-token",
    }, {})
    captured = {}
    def fake_request(path, *, method="GET", payload=None):
        captured.update(path=path, method=method, payload=payload)
        return 200, {"status": "SUCCESS", "message": "confirmed", "records_written": 1}
    monkeypatch.setattr(adapter, "_request", fake_request)
    result = adapter.write_batch("VH-3009", "S. Harish", [{"Observation": "Door bush missing"}])
    assert result.status is WriteStatus.SUCCESS
    assert result.records_written == 1
    assert captured["method"] == "POST"
    assert captured["payload"]["action"] == "write_batch"
    assert captured["payload"]["jpc"] == "VH-3009"
