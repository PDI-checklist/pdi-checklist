"""Resolve central API configuration for local and Streamlit deployments."""
from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from central_write_adapter import HTTPJSONCentralWriteAdapter, CentralWriteAdapter, UnconfiguredCentralWriteAdapter

URL_KEY = "PDI_GOOGLE_APPS_SCRIPT_URL"
TOKEN_KEY = "PDI_GOOGLE_APPS_SCRIPT_TOKEN"

def _lookup(source: Mapping[str, Any] | None, key: str) -> str:
    if not source:
        return ""
    try:
        value = source.get(key, "")
    except AttributeError:
        return ""
    return str(value or "").strip()

def resolve_config(secrets: Mapping[str, Any] | None = None, env: Mapping[str, str] | None = None) -> tuple[str, str]:
    env = env or os.environ
    url = _lookup(secrets, URL_KEY) or str(env.get(URL_KEY, "") or "").strip()
    token = _lookup(secrets, TOKEN_KEY) or str(env.get(TOKEN_KEY, "") or "").strip()
    return url, token

def build_central_adapter(secrets: Mapping[str, Any] | None = None, env: Mapping[str, str] | None = None, timeout_seconds: float = 15.0) -> CentralWriteAdapter:
    url, token = resolve_config(secrets, env)
    if not url or not token:
        return UnconfiguredCentralWriteAdapter()
    return HTTPJSONCentralWriteAdapter(base_url=url, token=token, timeout_seconds=timeout_seconds)
