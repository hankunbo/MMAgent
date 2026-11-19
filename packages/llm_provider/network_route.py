"""Explicit, frozen environment HTTPS route; legacy requests remain direct."""

import json
import os
import urllib.request
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from packages.grounding.models import Checksum, Value, digest, validate
from packages.knowledge_claims.worker import NoRedirect


class NetworkRoute(Value):
    schema_version: Literal["llm-network-route/1.0.0"] = "llm-network-route/1.0.0"
    mode: Literal["environment_https_proxy"]
    proxy_env: Literal["HTTPS_PROXY"]
    endpoint_checksum: Checksum


def read_route(value):
    if value is None:
        return None
    return validate(NetworkRoute, value)


def load_route(path):
    """Only public routing policy files, never private provider/credential config."""
    root = Path(__file__).resolve().parents[2]
    target = (root / path).resolve()
    if not target.is_relative_to(root / "configs") or target.suffix != ".json":
        raise ValueError("network_route_public_configuration_required")
    try:
        if target.stat().st_size > 4096:
            raise ValueError()
        return read_route(json.loads(target.read_bytes()))
    except (ValueError, OSError):
        raise ValueError("network_route_configuration_invalid") from None


def proxy_endpoint(value):
    """Reject credentials before hashing; do not leak the environment on errors."""
    try:
        parsed = urlsplit(value)
        if (
            not isinstance(value, str)
            or parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
            or any(ord(c) <= 32 or ord(c) >= 127 for c in value)
            or (parsed.port is not None and not 1 <= parsed.port <= 65535)
        ):
            raise ValueError()
    except (ValueError, TypeError, AttributeError):
        raise ValueError("network_proxy_endpoint_invalid") from None
    return value


def prepare_route(*, environ=None):
    """Freeze only the already configured credential-free route, not any API key."""
    env = os.environ if environ is None else environ
    endpoint = proxy_endpoint(env.get("HTTPS_PROXY"))
    return NetworkRoute(
        mode="environment_https_proxy",
        proxy_env="HTTPS_PROXY",
        endpoint_checksum=digest(endpoint),
    )


def route_opener(route, base_url, *, environ=None):
    policy = read_route(route)
    if policy is None:
        return urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    env = os.environ if environ is None else environ
    endpoint = proxy_endpoint(env.get(policy.proxy_env))
    if digest(endpoint) != policy.endpoint_checksum:
        raise ValueError("network_proxy_endpoint_changed")
    target = urlsplit(base_url)
    if target.scheme != "https" or not target.hostname:
        raise ValueError("network_proxy_requires_https_target")
    # urllib honours NO_PROXY even for an explicit ProxyHandler. Never silently
    # fall back to direct access when this request was frozen for the proxy route.
    if env.get("NO_PROXY") or env.get("no_proxy") or urllib.request.proxy_bypass(target.netloc):
        raise ValueError("network_proxy_bypass_not_authorized")
    return urllib.request.build_opener(
        urllib.request.ProxyHandler({"https": endpoint}), NoRedirect()
    )
