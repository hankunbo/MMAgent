"""Exact software handoff cannot change medical identity or renew an expired window."""

import copy
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from packages.evaluation_preparation.generic_capability_recovery import historical_protocol
from packages.object_store import checksum


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    now = datetime.now(UTC)
    files = {}
    for name in [
        "apps/vectorengine_round_materials.py",
        "apps/active_evidence_materials.py",
        "packages/evaluation_preparation/generic_capability_recovery.py",
    ]:
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("synthetic module\n")
        files[name] = checksum(p.read_bytes())
    current = dict(
        schema_version="generic-evidence-campaign/1.0.0",
        input="same",
        provider_checksum="same",
        previous={"reservations": 10},
        observations={"same": True},
        max_new_requests=12,
        max_cumulative_graph_requests=40,
        seconds=5400,
        implementation=files,
    )
    frozen = dict(
        copy.deepcopy(current),
        implementation={"old.py": "old"},
        started_at=(now - timedelta(hours=3)).isoformat(),
        deadline=(now - timedelta(hours=1)).isoformat(),
    )
    output = tmp_path / "case/evidence"
    save(output / "protocol.json", frozen)
    cmd = tmp_path / "docs/acceptance/test/commands"
    cmd.mkdir(parents=True)
    (cmd / "command.log").write_text("synthetic offline contract test evidence\n")
    (cmd / "command.exit").write_text("0\n")
    save(
        cmd / "command.json",
        dict(exit_code=0, log_checksum=checksum((cmd / "command.log").read_bytes())),
    )
    proof = dict(
        schema_version="generic-capability-history-software/1.0.0",
        scope="expired_history_read_only_not_dispatch",
        passed=True,
        commands=[str(cmd)],
        before=frozen["implementation"],
        after=files,
        files=files.copy(),
    )
    p = tmp_path / "docs/acceptance/test/proof.json"
    save(p, proof)
    reference = dict(path=str(p), checksum=checksum(p.read_bytes()))
    campaign = SimpleNamespace(output=output, protocol_fields=lambda: copy.deepcopy(current))
    return campaign, reference, proof, current, now


def test_history_transition_preserves_bytes_time_and_blocks_dispatch(tmp_path, monkeypatch):
    campaign, ref, _, _, now = fixture(tmp_path, monkeypatch)
    before = (campaign.output / "protocol.json").read_bytes()
    result = historical_protocol(campaign, ref, now=now)
    assert result["deadline"] < now.isoformat()
    assert campaign._capability_history_only is True
    assert (campaign.output / "protocol.json").read_bytes() == before


@pytest.mark.parametrize(
    "name", ["db/memory/repository.py", "docs/acceptance/public/relation.json"]
)
@pytest.mark.parametrize("registered", [True, False])
def test_exact_legacy_dependencies_require_original_freeze(tmp_path, monkeypatch, name, registered):
    campaign, ref, proof, current, now = fixture(tmp_path, monkeypatch)
    path = Path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}\n")
    expected = checksum(path.read_bytes())
    current["implementation"][name] = expected
    proof["files"][name] = expected
    frozen_path = campaign.output / "protocol.json"
    frozen = json.loads(frozen_path.read_bytes())
    if registered:
        frozen["implementation"][name] = expected
        proof["before"][name] = expected
    save(frozen_path, frozen)
    save(Path(ref["path"]), proof)
    ref["checksum"] = checksum(Path(ref["path"]).read_bytes())
    before = frozen_path.read_bytes()
    if registered:
        historical_protocol(campaign, ref, now=now)
        assert campaign._capability_history_only
    else:
        with pytest.raises(ValueError, match="history_dependency_changed"):
            historical_protocol(campaign, ref, now=now)
    assert frozen_path.read_bytes() == before


@pytest.mark.parametrize(
    "field",
    [
        "input",
        "provider_checksum",
        "previous",
        "observations",
        "max_new_requests",
        "max_cumulative_graph_requests",
        "seconds",
    ],
)
def test_software_transition_cannot_change_semantics(tmp_path, monkeypatch, field):
    campaign, ref, _, current, now = fixture(tmp_path, monkeypatch)
    current[field] = "changed"
    with pytest.raises(ValueError, match="semantic_or_implementation_changed"):
        historical_protocol(campaign, ref, now=now)
    assert not hasattr(campaign, "_capability_history_only")


@pytest.mark.parametrize(
    "change",
    [
        "proof_bytes",
        "scope",
        "uncovered",
        "source_bytes",
        "failed_command",
        "changed_log",
        "not_expired",
        "after_map",
    ],
)
def test_missing_or_changed_proof_rejects(tmp_path, monkeypatch, change):
    campaign, ref, proof, current, now = fixture(tmp_path, monkeypatch)
    from pathlib import Path

    if change == "proof_bytes":
        Path(ref["path"]).write_text("{}")
    elif change == "scope":
        proof["scope"] = "dispatch"
    elif change == "uncovered":
        proof["files"].pop("apps/active_evidence_materials.py")
    elif change == "source_bytes":
        Path("apps/active_evidence_materials.py").write_text("changed")
    elif change == "failed_command":
        (Path(proof["commands"][0]) / "command.exit").write_text("1")
    elif change == "changed_log":
        (Path(proof["commands"][0]) / "command.log").write_text("changed")
    elif change == "not_expired":
        now -= timedelta(hours=2)
    elif change == "after_map":
        current["implementation"] = {}
    if change in {"scope", "uncovered"}:
        save(Path(ref["path"]), proof)
        ref["checksum"] = checksum(Path(ref["path"]).read_bytes())
    with pytest.raises(ValueError, match="generic_capability_history_"):
        historical_protocol(campaign, ref, now=now)
    assert not hasattr(campaign, "_capability_history_only")


@pytest.mark.parametrize(
    "fault", [None, "source_bytes", "uncovered", "observation", "map", "legacy", "budget"]
)
@pytest.mark.parametrize(
    "resource_name",
    [
        "docs/sources/public.xml",
        "docs/sources/author-readme.md",
        "apps/templates/evidence_tree_reader_v3.js",
        "db/knowledge/migrations/0019_targeted_public_discovery.sql",
        "db/runtime/repository/research.py",
        ".local/private.json",
        "docs/sources/../../.local/private.json",
        "docs/sources/private.key",
    ],
)
def test_nested_capability_freeze_is_explicit_read_only_transition(
    tmp_path, monkeypatch, fault, resource_name
):
    campaign, ref, proof, current, now = fixture(tmp_path, monkeypatch)
    resource = Path(resource_name)
    resource.parent.mkdir(parents=True, exist_ok=True)
    resource.write_text("<public>synthetic source fixture</public>")
    after = {str(resource): checksum(resource.read_bytes())}
    before = {str(resource): checksum(b"previous public fixture")}
    frozen_path = campaign.output / "protocol.json"
    frozen = json.loads(frozen_path.read_bytes())
    frozen["observations"]["capability_implementation"] = before
    current["observations"]["capability_implementation"] = after
    save(frozen_path, frozen)
    original_bytes = frozen_path.read_bytes()
    proof.update(
        schema_version="generic-capability-history-software/1.1.0",
        capability_implementation=dict(before=before, after=after),
    )
    proof["files"].update(after)
    if fault == "source_bytes":
        resource.write_text("changed")
    elif fault == "uncovered":
        proof["files"].pop(str(resource))
    elif fault == "observation":
        current["observations"]["same"] = False
    elif fault == "map":
        proof["capability_implementation"] = dict(before={}, after=after)
    elif fault == "legacy":
        proof["schema_version"] = "generic-capability-history-software/1.0.0"
    elif fault == "budget":
        current["max_new_requests"] += 1
    save(Path(ref["path"]), proof)
    ref["checksum"] = checksum(Path(ref["path"]).read_bytes())
    denied = ".local" in resource.parts or resource.suffix == ".key"
    if fault is None and not denied:
        assert historical_protocol(campaign, ref, now=now) == frozen
        from packages.evaluation_preparation.generic_capability_successor import CapabilitySuccessor

        with pytest.raises(ValueError, match="history_cannot_execute"):
            CapabilitySuccessor.guard(campaign, dispatch=True)
    else:
        with pytest.raises(ValueError, match="generic_capability_history_"):
            historical_protocol(campaign, ref, now=now)
        assert not hasattr(campaign, "_capability_history_only")
    assert frozen_path.read_bytes() == original_bytes
