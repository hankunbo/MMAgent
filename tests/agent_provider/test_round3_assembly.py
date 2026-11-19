"""Explicit round-03 adapters; synthetic ports only, no Key, PG or external requests."""

import json
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from apps import vectorengine_round as rounds
from apps.vectorengine_round_materials import RoundCaseEvidence
from packages.grounding.models import digest, validate
from packages.object_store import checksum

ROUND = Path("configs/research/vectorengine_round3.json")


def test_exact_version_dispatch_and_frozen_v1_schema(tmp_path):
    old = rounds.load_round("configs/research/vectorengine_round2.json")
    new = rounds.load_round(ROUND)
    assert type(old) is rounds.LiveRound and type(new) is rounds.LiveRoundV2
    assert old.run_id != new.run_id and old.gate != new.gate and old.output != new.output
    assert old.material_catalog == new.material_catalog
    assert old.material_checksum == new.material_checksum
    for name, model in (
        ("BoundedLiveRound", rounds.LiveRound),
        ("BoundedLiveRoundV2", rounds.LiveRoundV2),
    ):
        assert (
            json.loads(Path(f"configs/research/schemas/{name}.schema.json").read_text())
            == model.model_json_schema()
        )
    with pytest.raises(ValueError):
        validate(rounds.LiveRound, new.model_dump(mode="json"))
    raw = new.model_dump(mode="json")
    raw["schema_version"] = "bounded-live-round/9.0.0"
    path = tmp_path / "unknown-version.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="round_version"):
        rounds.load_round(path)


@pytest.mark.parametrize(
    "field,value",
    [
        ("round_id", "vectorengine-single-case-round-02"),
        ("run_id", "RUN:RANDOM"),
        ("request_key", "vectorengine-single-case-round-02"),
        ("request_code", "RESEARCH:VECTOR_SINGLE_R02"),
        ("output", ".local/vectorengine-single-case-round-02/prepared-02"),
        ("gate", "docs/acceptance/12-live-round2-preparation-v1/execution-gate.json"),
        ("material_catalog", ".local/arbitrary.json"),
        ("seconds", 2701),
    ],
)
def test_round3_identity_and_paths_cannot_alias_prior_round(field, value):
    raw = rounds.load_round(ROUND).model_dump(mode="json")
    raw[field] = value
    with pytest.raises(ValueError):
        validate(rounds.LiveRoundV2, raw)


def test_preparation_reads_exact_locked_catalog_without_key_network_or_runtime_write(
    monkeypatch, tmp_path
):
    from apps.vectorengine_single_case import load_config
    from packages.llm_provider.live_probes import PublicProbeMaterials

    config = load_config()
    r = rounds.load_round(ROUND).model_copy(update={"output": str(tmp_path)})
    public = PublicProbeMaterials(r.run_id, config)
    catalog = rounds.load_catalog(r.material_catalog)
    reads = []

    def read(path):
        reads.append(path)
        assert path == r.material_catalog
        return catalog

    def evidence(original, actual, *, catalog_path):
        assert original == "synthetic_original" and actual is catalog
        assert catalog_path == r.material_catalog and digest(actual) == r.material_checksum
        return SimpleNamespace(project=lambda **kw: (public.state, dict(synthetic=True)))

    monkeypatch.setattr(rounds, "load_round", lambda _: r)
    monkeypatch.setattr(rounds, "load_catalog", read)
    monkeypatch.setattr(rounds, "RoundCaseEvidence", evidence)
    monkeypatch.setattr(rounds, "open_case", lambda: ("synthetic_original", None))
    monkeypatch.setattr(rounds, "read_deployment", lambda *a: dict(synthetic=True, actual_writes=0))
    monkeypatch.setattr("apps.vectorengine_single_case.os.environ", {})
    monkeypatch.setattr(
        "packages.llm_provider.transport.HTTPTransport.send",
        lambda *a, **kw: pytest.fail("network forbidden"),
    )
    assert rounds.main_round(ROUND) == 0
    saved = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    assert rounds.main_round(ROUND) == 0
    assert saved == {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    assert reads == [r.material_catalog, r.material_catalog]


def test_owner_revalidation_reads_same_catalog(monkeypatch):
    from apps import vectorengine_round_materials as materials

    r = rounds.load_round(ROUND)
    obj = RoundCaseEvidence.__new__(RoundCaseEvidence)
    obj.catalog = rounds.load_catalog(r.material_catalog)
    obj.catalog_path = Path(r.material_catalog)
    reads = []
    monkeypatch.setattr(materials, "load_catalog", lambda path: reads.append(path) or obj.catalog)
    monkeypatch.setattr(materials.FirstCaseEvidence, "verify", lambda *a: None)
    obj.verify(None)
    assert reads == [Path(r.material_catalog)]
    monkeypatch.setattr(materials, "load_catalog", lambda path: {})
    with pytest.raises(ValueError, match="round_definition_catalog_changed"):
        obj.verify(None)


def test_round3_gate_conflict_file_drift_and_prepared_state(monkeypatch, tmp_path):
    r = rounds.load_round(ROUND).model_copy(update={"gate": str(tmp_path / "gate.json")})
    file = tmp_path / "material.json"
    file.write_text("{}")
    gate = dict(
        schema_version="bounded-live-round-gate/1.0.0",
        synthetic_regression_passed=True,
        round_checksum=digest(r),
        implementation={str(file): checksum(file.read_bytes())},
        materials={},
        evidence={},
        initial_state_checksum=digest({}),
    )
    Path(r.gate).write_text(json.dumps(gate))
    monkeypatch.setattr(rounds, "implementation_files", lambda: {str(file)})
    assert rounds.check_round_gate(r, {}) == gate
    with pytest.raises(ValueError, match="gate_required"):
        rounds.check_round_gate(r.model_copy(update={"run_id": "RUN:FOREIGN"}))
    with pytest.raises(ValueError, match="prepared_state"):
        rounds.check_round_gate(r, dict(changed=True))
    file.write_text("changed")
    with pytest.raises(ValueError, match="gate_drift"):
        rounds.check_round_gate(r)


def test_round3_context_and_budget_reentry_via_existing_owners(monkeypatch, tmp_path):
    from tests.agent_provider import test_round_preparation as previous

    monkeypatch.setattr(previous, "ROUND", ROUND)
    previous.test_round_context_adapter_keeps_explicit_run_and_key_across_reentry(
        monkeypatch, tmp_path
    )
    previous.test_round_restart_uses_original_counters_deadline_and_unknown_gate()


def test_round3_check_config_precedes_environment_or_database(monkeypatch):
    from tests.agent_provider import test_round_preparation as previous

    monkeypatch.setattr(previous, "ROUND", ROUND)
    previous.test_public_check_routes_before_key_environment_or_database(monkeypatch)


def test_round3_execute_cannot_reuse_stopped_control(monkeypatch, tmp_path):
    from apps.vectorengine_live_assembly import execute
    from tests.agent_provider.test_live_terminal_guard import controlled, stop

    r = rounds.load_round(ROUND)
    f = controlled()
    stop(f)
    initial = f.reader.state.model_copy(
        update={
            "material": f.reader.state.material.model_copy(
                update={
                    "scope": f.reader.state.material.scope.model_copy(update={"run_id": r.run_id})
                }
            )
        }
    )
    monkeypatch.setattr(rounds, "check_round_gate", lambda *a: None)
    from tests.agent_provider.test_case_graph_target import ready_entry_state

    initial = ready_entry_state(initial)
    monkeypatch.setattr(
        "db.runtime.repository.planner.PlannerRepository", lambda _: f.docs.sessions
    )
    monkeypatch.setattr("db.runtime.repository.research.ResearchRunRepository", lambda _: object())
    before = dict(f.docs.saved)
    with pytest.raises(ValueError, match="stopped"):
        execute(
            evidence=None,
            target=None,
            initial=initial,
            config=f.config,
            output=tmp_path,
            round_config=r,
        )
    assert f.docs.saved == before and f.budget.used == 0 and not f.wire.calls


def test_saved_earlier_start_wins_over_later_origin_on_reentry(monkeypatch, tmp_path):
    from apps import vectorengine_live_assembly as live
    from packages.planner_control.models import Profile
    from tests.agent_provider.test_bounded_live import fixture as provider_fixture
    from tests.planner.support import binding
    from tests.tool_admission.support import fixture
    from tests.tool_admission.test_research import research

    r = rounds.load_round(ROUND)
    f = provider_fixture(
        b=binding(
            run_id=r.run_id,
            profile=Profile(max_steps=20, max_no_progress=2, max_retries=1, seconds=2700),
        )
    )
    from tests.agent_provider.test_case_graph_target import ready_entry_state

    f.reader.state = ready_entry_state(f.reader.state)
    req = research(fixture(tmp_path), tmp_path)
    f.docs.sessions.origin = lambda _: (
        f.binding.started_at + timedelta(seconds=30),
        f.binding.run_context_checksum,
    )
    created, starts = [], []
    monkeypatch.setattr(rounds, "check_round_gate", lambda *a: None)
    monkeypatch.setattr(
        "db.runtime.repository.planner.PlannerRepository", lambda _: f.docs.sessions
    )
    monkeypatch.setattr(
        "db.runtime.repository.research.ResearchRunRepository",
        lambda _: SimpleNamespace(
            read_research_run=lambda _: SimpleNamespace(request=req),
            create_research_run=lambda c, key: created.append((c, key)),
        ),
    )

    def observe_binding(**kwargs):
        starts.append(kwargs["started"])
        assert kwargs["started"] + timedelta(seconds=r.seconds) == f.binding.deadline
        raise LookupError("synthetic_stop_before_assembly")

    monkeypatch.setattr(live, "make_binding", observe_binding)
    evidence = SimpleNamespace(view={"case_evidence": {"fairclip": {"run_id": req.lock.run_id}}})
    for _ in range(2):
        with pytest.raises(LookupError, match="synthetic_stop_before_assembly"):
            live.execute(
                evidence=evidence,
                target=None,
                initial=f.reader.state,
                config=f.config,
                output=tmp_path,
                round_config=r,
            )
    assert created[0] == created[1] and starts == [f.binding.started_at] * 2
    assert f.budget.used == 0 and not f.wire.calls
