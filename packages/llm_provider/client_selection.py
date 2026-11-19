"""Versioned, public HTTP-client binding; never changes prompts or replays a request."""

import json
from pathlib import Path

from packages.grounding.models import digest

ROOT = Path(__file__).resolve().parents[2]
VERSION = "provider-client-binding/1.0.0"


def load_selection(path):
    selected = (ROOT / path).resolve()
    if not selected.is_relative_to(ROOT / "configs/agents") or selected.suffix != ".json":
        raise ValueError("client_selection_public_configuration_required")
    if selected.stat().st_size > 65536:
        raise ValueError("client_selection_configuration_size")
    from .curl_client import read_client, verify_binary

    client = read_client(json.loads(selected.read_bytes()))
    verify_binary(client)
    return dict(
        schema_version=VERSION,
        path=str(selected.relative_to(ROOT)),
        client=client.model_dump(mode="json"),
        client_checksum=digest(client),
    )


def apply_selection(config, binding):
    if binding is None:
        return config
    from .config import with_http_client

    if (
        not isinstance(binding, dict)
        or binding != load_selection(binding.get("path", ""))
        or config.schema_version != "agent-provider-config/9.1.0"
    ):
        raise ValueError("client_selection_binding_or_origin_changed")
    return with_http_client(config, binding["client"])


def check_entry_binding(spec, protocol=None):
    """An opt-in extension whose absence retains the exact historical configuration."""
    binding = spec.get("http_client_selection")
    if protocol is not None and binding != protocol.get("http_client_selection"):
        raise ValueError("client_selection_protocol_binding")
    provider = spec.get("provider", {})
    if binding is None:
        if provider.get("http_client") is not None:
            raise ValueError("client_selection_missing_entry_binding")
        return None
    if (
        not isinstance(binding, dict)
        or binding != load_selection(binding.get("path", ""))
        or spec.get("schema_version")
        not in {"multimodal-live-entry/1.29.0", "multimodal-live-entry/1.30.0"}
        or spec.get("execution_profile") != "relay-root-question-v1"
        or provider.get("schema_version") != "agent-provider-config/9.2.0"
        or provider.get("http_client") != binding["client"]
    ):
        raise ValueError("client_selection_entry_binding")
    from .config import read_provider_config

    read_provider_config(provider)
    return binding
