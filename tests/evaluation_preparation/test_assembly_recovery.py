"""A no-phase repair is not a provider retry or a new deadline."""

import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from packages.multimodal_research import assembly_recovery as repair


def put(path, value):
    path.write_text(json.dumps(value))


def candidate(tmp_path):
    route = {"mode": "test"}
    original = {"packages/multimodal_research/campaign.py": "old"}
    current = {**original, "packages/multimodal_research/assembly_recovery.py": "new"}
    put(
        tmp_path / "entry.json",
        dict(
            schema_version="multimodal-live-entry/1.29.0",
            implementation=original,
            provider=dict(network_route=route),
            network_route=route,
            input_checksum="input",
        ),
    )
    put(
        tmp_path / "active-start.json",
        dict(
            input_checksum="input", deadline=(datetime.now(UTC) + timedelta(minutes=10)).isoformat()
        ),
    )
    put(tmp_path / "active-plan.json", dict(rows=[dict(id="a"), dict(id="b"), dict(id="a")]))
    put(
        tmp_path / "active-discovery.json",
        dict(queries=[], status="deferred_to_campaign_revision"),
    )
    put(
        tmp_path / "entry-failure-original.json",
        dict(phase="owner_assembly", error_code="qualified_provider_not_installed"),
    )
    return original, current


def test_pre_campaign_repair_retains_original_entry_and_deadline(tmp_path):
    original, current = candidate(tmp_path)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    spec, start, delta, failure = repair.validate_candidate(tmp_path, current)
    calls = []
    contexts = SimpleNamespace(read_research_run=lambda run_id, missing_ok: calls.append(run_id))
    sessions = SimpleNamespace(read=lambda run_id: None)
    checked = repair.confirm_no_phases(tmp_path, contexts=contexts, sessions=sessions)
    assert len(set(checked)) == len(calls) == 4
    assert spec["implementation"] == original
    assert start == repair.read(tmp_path / "active-start.json")
    assert delta == ["packages/multimodal_research/assembly_recovery.py"]
    assert failure.name == "entry-failure-original.json"
    assert before == {p.name: p.read_bytes() for p in tmp_path.iterdir()}


@pytest.mark.parametrize(
    "marker", ["protocol.json", "cancel", "stage-failure.json", "request.json"]
)
def test_already_started_or_unknown_is_not_pre_campaign(tmp_path, marker):
    _, current = candidate(tmp_path)
    put(tmp_path / marker, {})
    with pytest.raises(ValueError, match="assembly_repair_not_unstarted"):
        repair.validate_candidate(tmp_path, current)


@pytest.mark.parametrize("kind", ["run", "control"])
def test_original_owner_activity_refuses_repair(tmp_path, kind):
    candidate(tmp_path)
    contexts = SimpleNamespace(
        read_research_run=lambda run_id, missing_ok: object() if kind == "run" else None
    )
    sessions = SimpleNamespace(read=lambda run_id: object() if kind == "control" else None)
    with pytest.raises(ValueError, match="assembly_repair_(phase_registered|control_present)"):
        repair.confirm_no_phases(tmp_path, contexts=contexts, sessions=sessions)


@pytest.mark.parametrize("kind", ["expired", "unrelated_code", "search_started", "wrong_error"])
def test_repair_is_narrow_and_never_resets_clock(tmp_path, kind):
    _, current = candidate(tmp_path)
    if kind == "expired":
        put(
            tmp_path / "active-start.json",
            dict(
                input_checksum="input",
                deadline=(datetime.now(UTC) - timedelta(seconds=1)).isoformat(),
            ),
        )
    elif kind == "unrelated_code":
        current["packages/evidence_scoring/continuous.py"] = "changed"
    elif kind == "search_started":
        put(tmp_path / "active-discovery.json", dict(queries=[{}], status="executed"))
    else:
        put(
            tmp_path / "entry-failure-original.json",
            dict(phase="owner_assembly", error_code="timeout"),
        )
    with pytest.raises(ValueError, match="assembly_repair_"):
        repair.validate_candidate(tmp_path, current)
