"""Operator intent over original bounded owners; never a second run state machine."""

import json
from datetime import UTC, datetime, timedelta
from threading import Event

from apps.vectorengine_single_case import write_once
from packages.grounding.models import digest
from packages.llm_provider.provider import ProviderError

from .generic_input import directory


def start(spec):
    path = directory(spec["key"]) / "started.json"
    if path.exists():
        value = json.loads(path.read_bytes())
        if value["key"] != spec["key"] or value["input_checksum"] != digest(spec):
            raise ValueError("generic_started_input_conflict")
        return value
    now = datetime.now(UTC)
    value = dict(
        key=spec["key"],
        input_checksum=digest(spec),
        started_at=now.isoformat(),
        deadline=(now + timedelta(seconds=7200)).isoformat(),
        semantics="orchestration_start_intent_original_owners_control_execution",
    )
    write_once(path, value)
    return value


def guard(key):
    from .generic_execution_lock import current

    current(key)
    out = directory(key)
    if (out / "cancel-request.json").exists():
        value = json.loads((out / "cancel-request.json").read_bytes())
        if value["key"] != key:
            raise ValueError("generic_cancel_scope")
        raise ProviderError("cancelled")
    path = out / "started.json"
    if path.exists():
        value = json.loads(path.read_bytes())
        # Historical pre-guard failed starts remain readable, never gain a renewed deadline.
        if not value.get("deadline"):
            raise ValueError("generic_legacy_start_requires_explicit_new_run")
        if datetime.now(UTC) >= datetime.fromisoformat(value["deadline"]):
            raise ProviderError("deadline")


def cancel(key):
    out = directory(key)
    spec = json.loads((out / "input.json").read_bytes())
    if spec["key"] != key:
        raise ValueError("generic_cancel_scope")
    path = out / "cancel-request.json"
    if not path.exists():
        write_once(
            path,
            dict(key=key, input_checksum=digest(spec), requested_at=datetime.now(UTC).isoformat()),
        )
    return dict(
        status="cancel_requested",
        key=key,
        semantics="no_next_dispatch_in_flight_owner_retains_confirmed_facts",
        tools_force_killed=False,
    )


class Cancellation(Event):
    """Original executor cancellation port observes the same durable operator intent."""

    def __init__(self, key):
        super().__init__()
        self.key = key

    def is_set(self):
        if super().is_set():
            return True
        try:
            guard(self.key)
        except ProviderError:
            return True
        return False


def status(key):
    out = directory(key)
    value = dict(key=key, prepared=(out / "input.json").exists())
    for name in ("started", "cancel-request", "export-reference"):
        path = out / (name + ".json")
        if path.exists():
            value[name] = json.loads(path.read_bytes())
    value["tool_steps"] = [
        json.loads(p.read_bytes()) for p in sorted((out / "tools").glob("*.json"))
    ]
    result = out / "evidence/result.json"
    if result.exists():
        value["result"] = json.loads(result.read_bytes())
    value["status_semantics"] = "local_status_references_not_current_execution_authority"
    return value
