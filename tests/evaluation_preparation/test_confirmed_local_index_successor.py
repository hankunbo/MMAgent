"""Confirmed local failure has explicit history, not UNKNOWN replay permission."""

from copy import deepcopy
from types import SimpleNamespace as NS

import pytest

from packages.grounding.models import digest
from packages.multimodal_research import campaign_renewal as renewal
from packages.multimodal_research import interrupted_definition as recovery
from tests.evaluation_preparation.test_interrupted_definition_successor import history, window


def local_history(tmp_path, monkeypatch):
    parent, path, c, values, failure = history(tmp_path, monkeypatch)
    failure["error_code"] = "local_owner_error"
    prefix = "definition:search"
    binding = values["definition:operation:0"][0]["binding"]
    ops = [
        ("search_group", prefix + ":search"),
        ("fetch_group", prefix + ":fetch:0"),
        ("local_index_plan", prefix + ":fetch:0:index:plan"),
        ("local_index_window", prefix + ":fetch:0:index:window:0"),
    ]
    for n, (kind, key) in enumerate(ops, 3):
        values[f"definition:operation:{n}"] = (dict(key=key, kind=kind, binding=binding),)
    values[prefix + ":search:result"] = ([dict(reference={"code": "synthetic"})],)
    values[prefix + ":fetch:0:result"] = (dict(retained="synthetic-fetch"),)
    values[prefix + ":fetch:0:index:plan:result"] = (
        dict(source_checksum=digest("body"), policies=[{"version": "2.0.0"}]),
    )
    return parent, path, c, values, failure


def repaired_history(tmp_path, monkeypatch):
    parent, path, c, values, failure = local_history(tmp_path, monkeypatch)
    before = recovery.tip(parent, path, local_failure=True, allow_pending=True)
    values[recovery.REPAIR_KEY] = (
        dict(
            schema_version="confirmed-local-index-repair/1.0.0",
            history=before["interruption"]["history"],
            failure=before["result"],
            pending=before["interruption"]["pending"],
            new_requests=0,
            indexes=[dict(key=digest("repaired-index"), checksum=digest("repaired-receipt"))],
        ),
    )
    return parent, path, c, values, failure


def test_repaired_history_retains_original_failure_and_pending_operation(tmp_path, monkeypatch):
    parent, path, c, values, failure = repaired_history(tmp_path, monkeypatch)
    before = deepcopy(values)
    proof = recovery.tip(parent, path, local_failure=True)
    assert values == before and proof["result"] == digest(failure)
    assert proof["result_kind"] == recovery.LOCAL_KIND
    assert len(proof["interruption"]["indexes"]) == 2
    assert proof["previous_attempts"][-1]["requests"] == 1
    assert proof["previous_attempts"][-1]["kind"] == recovery.LOCAL_HISTORY_KIND
    successor = NS(parent=parent, protocol={"tip": proof})
    assert (
        renewal.attempted_acquisition_targets(successor, [proof["previous_attempts"][-1]]) == set()
    )


@pytest.mark.parametrize(
    "fault", ["no_repair", "unknown", "llm", "fetch", "plan", "receipt", "counter", "cancel"]
)
def test_invalid_local_history_fails_without_relabeling(tmp_path, monkeypatch, fault):
    parent, path, c, values, failure = repaired_history(tmp_path, monkeypatch)
    if fault == "no_repair":
        values.pop(recovery.REPAIR_KEY)
    elif fault == "unknown":
        values.pop("definition:wire:1:outcome")
    elif fault == "llm":
        values["journal:1"] = ({"phase": "reserved"},)
    elif fault == "fetch":
        values.pop("definition:search:fetch:0:result")
    elif fault == "plan":
        values["definition:search:fetch:0:index:plan:result"][0]["source_checksum"] = digest(
            "wrong"
        )
    elif fault == "receipt":
        values[recovery.REPAIR_KEY][0]["history"] = digest("wrong")
    elif fault == "counter":
        values[recovery.REPAIR_KEY][0]["new_requests"] = 1
    else:
        (path / "cancel").touch()
    with pytest.raises(ValueError):
        recovery.tip(parent, path, local_failure=True)


def test_legacy_interruption_port_does_not_accept_local_failure(tmp_path, monkeypatch):
    parent, path, *_ = repaired_history(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="interrupted_definition_failure_binding"):
        recovery.tip(parent, path)


def test_explicit_local_version_preserves_remaining_budget_and_no_http(tmp_path):
    plan = window(tmp_path)
    import json

    old = json.loads((tmp_path / "protocol.json").read_bytes())
    old["started_at"] = "2020-01-01T00:00:00+00:00"
    (tmp_path / "protocol.json").write_text(json.dumps(old))
    plan.update(schema_version=recovery.LOCAL_VERSION, confirmed_local_index_failure=True)
    plan["tip"]["result_kind"] = recovery.LOCAL_KIND
    plan["tip"]["interruption"]["repair"] = digest("original-runtime-repair")
    recovery.check_plan(plan)
    for key, value in [
        ("max_new_requests", 6),
        ("new_public_search_allowed", True),
        ("confirmed_local_index_failure", False),
    ]:
        invalid = deepcopy(plan)
        invalid[key] = value
        with pytest.raises(ValueError):
            recovery.check_plan(invalid)
