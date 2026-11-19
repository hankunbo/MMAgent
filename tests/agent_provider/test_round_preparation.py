"""Round configuration/projection adapters only; no actual case, PG, keys or network."""

import copy
import importlib.util
import json
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest

from apps.vectorengine_case_materials import FirstCaseEvidence, ref
from apps.vectorengine_round import LiveRound, load_round, prepare_round
from apps.vectorengine_round_materials import (
    RoundCaseEvidence,
    definitions,
    extract_passage,
    load_catalog,
)
from apps.vectorengine_single_case import load_config
from packages.grounding.models import digest, validate
from packages.llm_provider.models import Source
from packages.object_store import checksum

ROUND = Path("configs/research/vectorengine_round2.json")


def test_fixed_round_and_limits_do_not_depend_on_output_or_start_time():
    a, b = load_round(ROUND), load_round(ROUND)
    assert a == b and a.run_id != "RUN:OID_D1FD7FEAE237DF5599658BF4A56455A027DD9EF8E715E207"
    for field, value in (
        ("run_id", "RUN:RANDOM"),
        ("request_key", "changed"),
        ("output", ".local/other"),
        ("seconds", 2701),
        ("max_requests", 41),
        ("max_no_progress", 3),
        ("max_retries", 2),
    ):
        raw = a.model_dump(mode="json")
        raw[field] = value
        with pytest.raises(ValueError):
            validate(LiveRound, raw)


def test_catalog_restores_definitions_and_binds_nice_exact_sections_without_new_rules():
    catalog = load_catalog()
    projected = definitions(catalog)
    handoff = json.loads(
        Path("docs/acceptance/12-live-terminal-guard-v1/definition-handoff.json").read_text()
    )
    omitted = [r["id"] for r in handoff["items"] if r["category"].startswith("A_")]
    assert len(omitted) == 12
    assert all(projected[i]["interpretation_installed"] for i in omitted)
    assert projected["criterion:IOP"]["interpretation_installed"]
    assert not projected["criterion:AGE"]["interpretation_installed"]
    assert not projected["criterion:VCDR_GE_07"]["interpretation_installed"]
    for identifier, row in projected.items():
        if identifier.startswith("visual:"):
            assert not row["interpretation_installed"]
    nice = next(s for s in catalog["sources"] if s["id"] == "NICE-NG81-REFERRAL")
    assert nice["reference"]["kind"] == "source_document"
    assert all(s in nice["text"] for s in ("1.1.2", "1.1.4", "1.1.5", "24", "Goldmann"))
    assert all(s["text_checksum"] == checksum(s["text"].encode()) for s in catalog["sources"])


def test_exact_excerpt_read_rejects_withdrawal_or_changed_bytes():
    text = "Synthetic irrelevant prefix. Exact observation reference. suffix"
    reference = dict(
        kind="document_chunk", code="SOURCE:SYNTHETIC", version="1.0.0", checksum=digest("record")
    )
    record = SimpleNamespace(
        source_version_ref="version",
        document_ref="document",
        object_uri="synthetic",
        content_checksum=checksum(text.encode()),
        size_bytes=len(text.encode()),
    )
    parent = SimpleNamespace(
        record=SimpleNamespace(source_ref="origin"),
        withdrawn=False,
        status=SimpleNamespace(value="draft"),
    )
    snapshot = SimpleNamespace(record=record, withdrawn=False, status=parent.status)
    source = dict(
        reference=reference,
        selector=dict(kind="text", start=29, end=56),
        text=text[29:56],
        text_checksum=checksum(text[29:56].encode()),
        locator="synthetic chars 29:56",
    )
    reader = RoundCaseEvidence.__new__(RoundCaseEvidence)
    reader.original = SimpleNamespace(
        knowledge=SimpleNamespace(
            get=lambda r: (
                snapshot if r != "version" and r != "document" and r != "origin" else parent
            )
        ),
        knowledge_objects=SimpleNamespace(read_bounded=lambda *_: text.encode()),
    )
    assert reader.read_source(source).text == text[29:56]
    snapshot.withdrawn = True
    with pytest.raises(ValueError, match="source_withdrawn"):
        reader.read_source(source)
    snapshot.withdrawn = False
    reader.original.knowledge_objects.read_bounded = lambda *_: b"changed"
    with pytest.raises(ValueError, match="source_object_changed"):
        reader.read_source(source)
    assert (
        extract_passage(
            b'{"blocks":[{"text":"a"},{"text":"b"}]}', dict(kind="json_blocks", indices=[1])
        )
        == "b"
    )


def synthetic_projection():
    source = Source(
        reference=ref("source", "SOURCE:SYNTHETIC", digest("source")),
        locator="synthetic literal",
        purpose="research_material",
        state="research_only",
        text="Synthetic measurement reference",
    )
    marker = Source(
        reference=ref("source", "POLICY:ROUND_DEFINITIONS", digest("definitions")),
        locator="project policy only",
        purpose="research_material",
        state="research_only",
        text="Not disease evidence",
    )
    fixed = source.model_copy(update={"reference": ref("source", "SOURCE:FIXED", digest("fixed"))})
    rows, observations = [], []
    ids = [
        "association:ARTERIOLAR_CALIBER",
        "descriptive:OCT_REFERENCE",
        "descriptive:VESSEL_DENSITY",
        "visual:quality",
    ]
    for i, identifier in enumerate(ids):
        rows.append(
            dict(
                id=identifier,
                node="EVIDENCE:SYNTHETIC",
                kind="descriptive"
                if identifier.startswith("descriptive")
                else identifier.split(":")[0],
                role="measurement"
                if identifier.startswith("descriptive")
                else identifier.split(":")[0],
                dependency_paths=[["INPUT:SYNTHETIC"]],
                correlation_group="synthetic",
                dependence_block="synthetic",
            )
        )
        observations.append(
            dict(
                assessment_id=identifier,
                id=f"OBS:SYNTHETIC:{i}",
                title="Synthetic title",
                result_kind="visual_review" if identifier.startswith("visual") else "unavailable",
                values=[dict(name="old_answer", value="MUST_NOT_COPY", unit="1")],
            )
        )
    # Only metadata checking occurs in this test; no real image reader or patient bytes.
    png = (
        b"\x89PNG\r\n\x1a\n"
        + struct.pack(">I", 13)
        + b"IHDR"
        + struct.pack(">IIBBBBB", 512, 664, 8, 2, 0, 0, 0)
        + b"\x00" * 4
    )
    e = FirstCaseEvidence.__new__(FirstCaseEvidence)
    e.image_artifact = SimpleNamespace(asset_code="ASSET:SYNTHETIC", checksum=checksum(png))
    e.image = lambda _: png
    e.visible_value = ref("visible", "INPUT:SYNTHETIC", digest("visible"))
    e.root = SimpleNamespace(checksum=digest("root"))
    e.fixed_source = fixed
    e.sources = (source, marker, fixed)
    e.chunks = [dict(source_id="SYNTHETIC"), dict(source_id="PROJECT_DEFINITION_ONLY")]
    e.view = dict(
        nodes=[
            dict(key="ROOT", node_code="GOAL:TEST:ROOT", node_type="GOAL", parent=None),
            dict(key="EVIDENCE:SYNTHETIC", node_code="REQ:TEST:ONE", parent="ROOT"),
        ],
        current_scoring=dict(
            reviews=rows,
            resources={"INPUT:SYNTHETIC": dict(synthetic=True)},
            groups=[
                dict(
                    id="group",
                    importance=dict(slo_image_research=1.0, comprehensive_research=1.0),
                    slots=[dict(assessment_ids=ids)],
                )
            ],
        ),
        research_criteria=dict(policy=dict(criteria=[])),
        evidence_records=dict(observations=observations),
    )
    return e


def test_opt_in_projection_preserves_missing_and_legacy_default_without_answer_leaks():
    e = synthetic_projection()
    config = load_config()
    old, _ = e.project(run_id="RUN:SYNTHETIC", config=config)
    d = definitions(load_catalog())
    for row in d.values():
        if row["sources"]:
            row["sources"] = [dict(id="SYNTHETIC")]
    new, _ = e.project(run_id="RUN:SYNTHETIC", config=config, definitions=d)
    assert old.policy.version == "first-case-live-1.0.0"
    assert new.policy.version == "first-case-live-2.0.0"
    assert all(p.definition.interpretation_status == "missing" for p in old.policy.items)
    assert [p.definition.interpretation_status for p in new.policy.items] == [
        "installed",
        "installed",
        "missing",
        "missing",
    ]
    assert new.policy.items[-1].source_codes == ("POLICY:ROUND_DEFINITIONS",)
    assert all(o.value is None for o in new.material.observations)
    assert sum(o.image is not None for o in new.material.observations) == 1
    assert "MUST_NOT_COPY" not in new.model_dump_json()
    assert e.project(run_id="RUN:SYNTHETIC", config=config)[0] == old


def test_prepare_repeat_only_writes_same_local_locks_no_transport(monkeypatch, tmp_path):
    from tests.agent_provider.test_bounded_live import fixture
    from tests.agent_provider.test_case_graph_target import ready_entry_state

    f = fixture()
    f.reader.state = ready_entry_state(f.reader.state)
    round_config = load_round(ROUND).model_copy(update={"output": str(tmp_path)})
    evidence = SimpleNamespace(project=lambda **_: (f.reader.state, dict(synthetic=True)))
    monkeypatch.setattr(
        "packages.llm_provider.transport.HTTPTransport.send",
        lambda *a, **k: pytest.fail("network forbidden"),
    )
    before = copy.deepcopy(f.docs.saved)
    a = prepare_round(round_config, f.config, evidence)
    b = prepare_round(round_config, f.config, evidence)
    assert a == b and f.docs.saved == before and f.budget.used == 0
    assert len(list(tmp_path.glob("*.json"))) == 5


def test_public_check_routes_before_key_environment_or_database(monkeypatch):
    from apps.vectorengine_single_case import main

    calls = []
    monkeypatch.setattr(
        "apps.vectorengine_round.check_round_gate", lambda r: calls.append(r.run_id)
    )
    monkeypatch.setattr(
        "apps.vectorengine_round.open_case", lambda: pytest.fail("database forbidden")
    )
    monkeypatch.setattr("apps.vectorengine_single_case.os.environ", {})
    assert main(["--round-config", str(ROUND), "--check-config"]) == 0
    assert calls == [load_round(ROUND).run_id]


def test_launcher_only_forwards_explicit_round_without_loading_a_key(monkeypatch):
    path = Path(".local/vectorengine/run.py")
    spec = importlib.util.spec_from_file_location("round_launcher_source_only", path)
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    monkeypatch.setattr(launcher.os, "environ", {})
    command, env = launcher.invocation("execute", "synthetic-test-token", ROUND)
    assert command[-2:] == ["--round-config", str(ROUND)]
    assert env[launcher.KEY_ENV] == "synthetic-test-token"
    assert launcher.invocation("execute", "synthetic-test-token")[0][-1] == "--execute"
    with pytest.raises(launcher.LocalConfigError):
        launcher.invocation("test", "synthetic-test-token", ROUND)


def test_new_gate_checks_file_set_round_material_evidence_and_prepared_state(monkeypatch, tmp_path):
    from apps.vectorengine_round import check_round_gate

    artifact = tmp_path / "synthetic.json"
    artifact.write_text("{}")
    sha = checksum(artifact.read_bytes())
    round_config = load_round(ROUND).model_copy(update={"gate": str(tmp_path / "gate.json")})
    initial = dict(synthetic=True)
    gate = dict(
        schema_version="bounded-live-round-gate/1.0.0",
        synthetic_regression_passed=True,
        round_checksum=digest(round_config),
        implementation={str(artifact): sha},
        materials={},
        evidence={},
        initial_state_checksum=digest(initial),
    )
    Path(round_config.gate).write_text(json.dumps(gate))
    monkeypatch.setattr("apps.vectorengine_round.implementation_files", lambda: {str(artifact)})
    assert check_round_gate(round_config, initial) == gate
    with pytest.raises(ValueError, match="prepared_state"):
        check_round_gate(round_config, dict(synthetic=False))
    artifact.write_text('{"changed":true}')
    with pytest.raises(ValueError, match="gate_drift"):
        check_round_gate(round_config)
    monkeypatch.setattr("apps.vectorengine_round.implementation_files", lambda: set())
    with pytest.raises(ValueError, match="gate_required"):
        check_round_gate(round_config)


def test_round_context_adapter_keeps_explicit_run_and_key_across_reentry(monkeypatch, tmp_path):
    from apps.vectorengine_live_assembly import execute
    from tests.agent_provider.test_bounded_live import fixture as provider_fixture
    from tests.tool_admission.support import fixture
    from tests.tool_admission.test_research import research

    round_config = load_round(ROUND)
    req = research(fixture(tmp_path), tmp_path)
    f = provider_fixture()
    initial = f.reader.state.model_copy(
        update={
            "material": f.reader.state.material.model_copy(
                update={
                    "scope": f.reader.state.material.scope.model_copy(
                        update={"run_id": round_config.run_id}
                    )
                }
            )
        }
    )
    from tests.agent_provider.test_case_graph_target import ready_entry_state

    initial = ready_entry_state(initial)
    saved = []

    def create(context, key):
        saved.append((context, key))
        raise LookupError("synthetic_stop_before_any_runtime_write")

    monkeypatch.setattr("apps.vectorengine_round.check_round_gate", lambda *a: None)
    monkeypatch.setattr(
        "db.runtime.repository.planner.PlannerRepository",
        lambda _: SimpleNamespace(read=lambda r: None),
    )
    monkeypatch.setattr(
        "db.runtime.repository.research.ResearchRunRepository",
        lambda _: SimpleNamespace(
            read_research_run=lambda _: SimpleNamespace(request=req), create_research_run=create
        ),
    )
    evidence = SimpleNamespace(view={"case_evidence": {"fairclip": {"run_id": req.lock.run_id}}})
    for _ in range(2):
        with pytest.raises(LookupError, match="synthetic_stop"):
            execute(
                evidence=evidence,
                target=None,
                initial=initial,
                config=f.config,
                output=tmp_path,
                round_config=round_config,
            )
    assert saved[0] == saved[1]
    assert saved[0][0].run_id == round_config.run_id
    assert saved[0][1] == round_config.request_key
    assert saved[0][0].request.code == round_config.request_code
    assert f.budget.used == 0 and not f.wire.calls


def test_round_restart_uses_original_counters_deadline_and_unknown_gate():
    from packages.llm_provider.provider import ProviderError
    from packages.planner_control.models import Profile
    from tests.agent_provider.support import review_from_wire
    from tests.agent_provider.test_bounded_live import fixture, generate
    from tests.planner.support import binding

    config = load_round(ROUND)
    b = binding(
        run_id=config.run_id,
        profile=Profile(max_steps=20, max_no_progress=2, max_retries=1, seconds=2700),
    )
    f = fixture([review_from_wire], b=b)
    generate(f)
    restored = fixture(journal=f.budget.journal, b=b)
    assert restored.budget.used == 1 and restored.budget.deadline == f.budget.deadline
    restored.budget.reserve(restored.reader.state.material.scope, b.deadline, digest(f.config))
    unknown = fixture(journal=f.budget.journal, b=b)
    with pytest.raises(ProviderError, match="transport"):
        generate(unknown)
    assert unknown.budget.used == 2 and not unknown.wire.calls


def test_round_schema_and_default_fake_remain_exact():
    from packages.llm_provider.config import load_provider_config

    schema = json.loads(Path("configs/research/schemas/BoundedLiveRound.schema.json").read_text())
    assert schema == LiveRound.model_json_schema()
    assert schema["additionalProperties"] is False
    assert load_provider_config(environ={}).mode == "fake"


def test_deployment_unknown_stops_before_preparation_and_never_creates_run(monkeypatch):
    from apps.vectorengine_round import main_round

    monkeypatch.setattr("apps.vectorengine_round.open_case", lambda: (object(), object()))
    monkeypatch.setattr(
        "apps.vectorengine_round.RoundCaseEvidence", lambda original, catalog, **kwargs: original
    )

    def unavailable(*args):
        raise LookupError("synthetic_owner_unavailable")

    monkeypatch.setattr("apps.vectorengine_round.read_deployment", unavailable)
    monkeypatch.setattr(
        "apps.vectorengine_round.prepare_round",
        lambda *a: pytest.fail("must stop before preparation"),
    )
    with pytest.raises(LookupError, match="owner_unavailable"):
        main_round(ROUND)
