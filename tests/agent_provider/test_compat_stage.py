"""Derived round identity, exact legacy reads, synthetic public shape and stage caps."""

from datetime import timedelta

import pytest

from apps.vectorengine_round import LiveRoundV3, load_round
from apps.vectorengine_single_case import load_config
from packages.evidence_details.review_tasks import project_review_tasks
from packages.grounding.models import digest, validate
from packages.llm_provider.bounded_live import BoundedLiveBudget
from packages.llm_provider.live_allocation import StageAllocation
from packages.llm_provider.provider import ProviderError
from packages.llm_provider.public_shape import PublicReviewMaterials
from packages.planner_control.models import Binding
from packages.research_runtime.agent_records import RuntimeAgentJournal
from tests.agent_provider.test_http_recovery import recovery_fixture


def test_exact_legacy_and_new_identity():
    for name, version in (("round2", "1.0.0"), ("round3", "2.0.0")):
        assert load_round(
            "configs/research/vectorengine_" + name + ".json"
        ).schema_version.endswith(version)
    for stage in ("public", "case"):
        p = "configs/research/vectorengine_compat_case_" + stage + ".json"
        assert load_round(p) == load_round(p)


@pytest.mark.parametrize(
    "field,value",
    [
        ("round_id", "vectorengine-single-case-round-03"),
        ("request_key", "old"),
        ("run_id", "RUN:FOREIGN"),
        ("request_code", "RESEARCH:FOREIGN"),
        ("gate", "../gate.json"),
        ("output", ".local/../escape"),
        ("provider_config", ".local/vectorengine/config.json"),
        ("material_catalog", "/etc/passwd"),
        ("public_config", "../../escape.json"),
    ],
)
def test_conflict_and_path_escape_rejected(field, value):
    config = load_round("configs/research/vectorengine_compat_case_public.json")
    with pytest.raises(ValueError):
        validate(LiveRoundV3, config.model_copy(update={field: value}))


def test_public_shape_45_exact_states_no_old_answers():
    cfg = load_config("configs/agents/vectorengine_compat_case.json")
    m = PublicReviewMaterials("RUN:PUBLIC", cfg)
    projection = project_review_tasks(m.state)
    assert len(projection.tasks) == 45 and len(set(projection.assessment_ids)) == 45
    assert {o.status for o in m.state.material.observations} == {
        "obtained",
        "missing",
        "unknown",
        "ungradable",
    }
    assert sum(t.definition.interpretation_status == "missing" for t in projection.tasks) == 9
    assert not projection.fixed_assessment_ids


@pytest.mark.parametrize("stage,used,cap", [("public", 0, 12), ("case", 7, 33)])
def test_stage_reservation_cap_and_restart(monkeypatch, stage, used, cap):
    f = recovery_fixture(monkeypatch, [])
    # New synthetic storage before constructing this distinct allocation. No real journal reset.
    f.docs.saved = {}
    b = f.binding
    if stage == "public":
        b = validate(
            Binding,
            b.model_copy(
                update={
                    "profile": b.profile.model_copy(update={"seconds": 1200}),
                    "deadline": b.started_at + timedelta(seconds=1200),
                }
            ),
        )
        f.docs.sessions.state = f.docs.sessions.read(b.run_id).model_copy(update={"binding": b})
    allocation = StageAllocation(
        session_id="synthetic",
        stage=stage,
        public_run_id=b.run_id,
        public_used=used,
        public_journal_checksum=digest("public") if used else None,
        configuration_checksums=(digest(f.config),),
    )
    # Persistent semantics are covered by the accompanying owned PG test.
    f.docs.sessions.origin = lambda _: (b.started_at, b.run_context_checksum)
    budget = BoundedLiveBudget(
        binding=b, config=f.config, journal=RuntimeAgentJournal(f.docs), allocation=allocation
    )
    # Use completed zero-dispatch records from the Provider rather than pending reservations.
    from packages.llm_provider.models import RequestRecord, Usage

    for _n in range(cap):
        rid, num, until = budget.reserve(
            f.reader.state.material.scope, b.deadline, digest(f.config)
        )
        budget.record(
            RequestRecord(
                request_id=rid,
                invocation_id=rid,
                retry_index=0,
                role="source",
                scope=f.reader.state.material.scope,
                provider="openai_compatible",
                model="gpt-5.6-sol",
                endpoint="chat_completions",
                config_checksum=digest(f.config),
                prompt_version="synthetic",
                prompt_checksum=digest("p"),
                input_checksum=digest("i"),
                output_checksum=None,
                remote_request_checksum=None,
                attempt=num,
                simulated=False,
                api_requests=0,
                usage=Usage(),
                latency_seconds=0,
                error="configuration",
                deadline=until,
            ),
            response_confirmed=False,
        )
    again = BoundedLiveBudget(
        binding=b, config=f.config, journal=RuntimeAgentJournal(f.docs), allocation=allocation
    )
    with pytest.raises(ProviderError, match="budget"):
        again.before_reserve(digest(f.config))
    assert again.used == cap and again.deadline == b.deadline


def test_v3_check_without_key_cannot_open_owner_or_send(monkeypatch, capsys):
    from apps.vectorengine_round import main_round

    monkeypatch.delenv("MMAGENT_VECTORENGINE_API_KEY", raising=False)
    monkeypatch.setattr("apps.vectorengine_round.check_round_gate", lambda *_: None)
    monkeypatch.setattr("apps.vectorengine_round.open_case", lambda: pytest.fail("owner accessed"))
    assert main_round("configs/research/vectorengine_compat_case_public.json", check_only=True) == 0
    assert main_round("configs/research/vectorengine_compat_case_public.json", execute=True) == 2
    assert "missing_designated_key" in capsys.readouterr().out


def test_public_45_full_local_admission(monkeypatch):
    from apps.vectorengine_public_diagnostic import public_step
    from tests.agent_provider.support import review_from_wire

    def synthetic(body):
        import json

        value = review_from_wire(body)
        p = json.loads(body["messages"][1]["content"][0]["text"])
        missing = {
            t["assessment_id"]
            for t in p["review_tasks"]["tasks"]
            if t["definition"]["interpretation_status"] == "missing"
        }
        for item in value["items"]:
            if item["assessment_id"] in missing:
                item.update(
                    status="reference_missing",
                    judgment=dict(
                        value=None,
                        lower=None,
                        upper=None,
                        reason="Synthetic missing definition",
                        method="not_assessed",
                        missing_reason="No interpretation installed",
                    ),
                )
        return value

    f = recovery_fixture(monkeypatch, [synthetic])
    materials = PublicReviewMaterials(f.binding.run_id, f.config)
    result = public_step(
        name="review45",
        role="review",
        provider=f.provider,
        budget=f.budget,
        documents=f.docs,
        materials=materials,
    )
    assert result["status"] == "passed" and result["task_count"] == 45
    assert f.budget.used == 1


def test_old_and_v2_source_wire_equal():
    from packages.llm_provider.config import prompt
    from packages.llm_provider.live_probes import PublicProbeMaterials
    from packages.llm_provider.models import SourceProposal
    from packages.llm_provider.transport import wire_request

    old = load_config()
    new = load_config("configs/agents/vectorengine_compat_case.json")
    material = PublicProbeMaterials("RUN:PUBLIC", old).state.material
    payload = dict(role="source", material=material.model_dump(mode="json"))
    wires = []
    for config in (old, new):
        setting, text = prompt(config, "source")
        wires.append(
            wire_request(
                config.transport,
                setting,
                text,
                payload,
                SourceProposal.model_json_schema(),
                (),
                scope_binding=digest(material.scope),
            )
        )
    assert wires[0] == wires[1]


def test_public_begin_uses_original_qualified_snapshot(monkeypatch):
    from types import SimpleNamespace

    from agents.planner.service import Planner
    from apps.vectorengine_public_diagnostic import PublicDiagnosticSnapshot
    from tests.planner.support import MemoryRuntime

    f = recovery_fixture(monkeypatch, [])
    verified = []
    owner = PublicDiagnosticSnapshot(
        SimpleNamespace(verify=lambda s: verified.append(s)), f.reader.state
    )
    planner = Planner(runtime=MemoryRuntime(), owner=owner)
    assert planner.begin(f.binding).binding == f.binding
    assert verified == [f.reader.state]
    with pytest.raises(ValueError, match="public_control_scope"):
        owner.current(f.binding.model_copy(update={"run_id": "RUN:OTHER"}))
