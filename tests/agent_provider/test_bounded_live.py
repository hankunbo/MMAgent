"""Synthetic wire fault injection only. These tests never use patient materials or a network."""

import json
from datetime import timedelta
from types import SimpleNamespace

import pytest

from agents.expert_agent import ResearchReview
from agents.expert_agent.reporting import read_research_report, render_research_html
from agents.expert_agent.study import compare_candidates
from agents.provider_agents import StructuredRoleAgent
from apps.vectorengine_single_case import load_config
from packages.evidence_details.acceptance import ResearchAcceptance
from packages.evidence_details.acceptance_models import (
    AcceptedRecordsV21,
    read_receipt,
    read_records,
)
from packages.evidence_details.live_permission import ResearchLiveGrant, ResearchLivePermission
from packages.grounding.models import digest, validate
from packages.llm_provider.bounded_live import BoundedKnowledge, BoundedLiveBudget
from packages.llm_provider.live_probes import run_probes
from packages.llm_provider.provider import MemoryJournal, ProviderError, StructuredProvider
from packages.llm_provider.transport import Reply
from packages.llm_provider.worker import safe_http_error
from packages.planner_control.models import Profile
from tests.agent_provider.support import Documents, materials, review_from_wire, submission
from tests.expert_agent.support import protocol
from tests.planner.support import binding


class SyntheticJournal(MemoryJournal):
    durable = True  # Explicit fault injection; actual Runtime persistence is separately tested.


class SyntheticWire:
    simulated = False  # Exercises live branches without any socket; never a real-call report.

    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def send(self, config, body, **kwargs):
        self.calls.append((body, kwargs))
        value = self.replies.pop(0)
        if callable(value):
            value = value(body)
        return (
            value
            if isinstance(value, Reply)
            else Reply(content=json.dumps(value).encode(), api_requests=1, response_confirmed=True)
        )


def fixture(replies=(review_from_wire,), *, journal=None, documents=None, b=None):
    config = load_config()
    b = b or binding(profile=Profile(max_steps=20, max_no_progress=2, max_retries=1, seconds=2700))
    reader, _ = materials(b, config=config)
    journal = journal or SyntheticJournal()
    docs = documents or getattr(journal, "documents", None) or Documents()
    from tests.planner.support import MemoryRuntime

    if not hasattr(docs, "sessions"):
        docs.sessions = MemoryRuntime()
        docs.sessions.begin(b)
    journal.documents = docs
    budget = BoundedLiveBudget(binding=b, config=config, journal=journal)
    docs.run_id = b.run_id
    if not hasattr(docs, "root"):
        docs.root = SimpleNamespace(checksum=b.graph.checksum)
    grant = ResearchLiveGrant(
        enabled=True,
        case_code=b.case_code,
        run_id=b.run_id,
        input_checksum=b.input_checksum,
        visible_checksum=b.visible_checksum,
        policy_checksum=reader.state.material.scope.policy_checksum,
        config_checksum=digest(config),
        root_checksum=docs.root.checksum,
        purpose="bounded_single_case_research",
        deadline=b.deadline,
        roles=("source", "vision", "review"),
    )
    permission = ResearchLivePermission(grant=grant, config=config, documents=docs, budget=budget)
    wire = SyntheticWire(replies)
    provider = StructuredProvider(config, live_opt_in=True, transport=wire, sleep=lambda _: None)
    return SimpleNamespace(
        config=config,
        binding=b,
        reader=reader,
        budget=budget,
        docs=docs,
        permission=permission,
        wire=wire,
        provider=provider,
    )


def generate(f, round_id="A"):
    result = StructuredRoleAgent(
        "review", provider=f.provider, budget=f.budget, materials=f.reader
    ).call(f.reader.state.material.scope, deadline=f.binding.deadline, round_id=round_id)
    return submission(f.reader, result, f.budget)


def test_default_denies_live_then_explicit_grant_accepts_and_exact_old_reader_rejects():
    f = fixture()
    sub = generate(f)
    closed = ResearchAcceptance(owner=f.reader, documents=f.docs, requests=f.budget)
    assert closed.submit(sub, key="closed").status == "rejected"
    service = ResearchAcceptance(
        owner=f.reader, documents=f.docs, requests=f.budget, live_permission=f.permission
    )
    assert service.submit(sub, key="live")["status"] == "submitted_not_accepted"
    receipt = service.accept(key="live")
    assert receipt.execution == "bounded_live_research" and receipt.new_medical_review is True
    records = service.read(key="live")[0]
    assert read_records(records).status == "accepted_live_research"
    with pytest.raises(ValueError):
        validate(AcceptedRecordsV21, records)
    assert read_receipt(receipt.model_dump(mode="json")) == receipt
    assert service.accept(key="live") == receipt
    report = read_research_report(service, key="live", parent=f.reader.state)
    assert report["scores"]["schema_version"] == "case-research-scoring/6.1.0"
    assert "有界真实 LLM" in render_research_html(report)
    assert "Fake 接纳" not in render_research_html(report)
    successor = service.successor_material(key="live")
    assert successor.scope.graph.parent.checksum == f.binding.graph.checksum


@pytest.mark.parametrize(
    "field", ["run_id", "case_code", "input_checksum", "visible_checksum", "policy_checksum"]
)
def test_single_run_input_policy_grant_rejects_cross_scope(field):
    f = fixture()
    scope = f.reader.state.material.scope
    replacement = (
        "RUN:FOREIGN"
        if field == "run_id"
        else "CASE:FOREIGN"
        if field == "case_code"
        else digest("foreign")
    )
    state = f.reader.state.model_copy(
        update={
            "material": f.reader.state.material.model_copy(
                update={"scope": scope.model_copy(update={field: replacement})}
            )
        }
    )
    with pytest.raises(ValueError, match="scope_forbidden"):
        f.permission.check(state)


def test_deadline_no_restart_reset_and_unknown_no_new_key():
    f = fixture([Reply(error="timeout", api_requests=None)])
    with pytest.raises(ProviderError):
        generate(f)
    assert len(f.wire.calls) == 1
    with pytest.raises(ProviderError):
        f.budget.reserve(f.reader.state.material.scope, f.binding.deadline, digest(f.config))
    assert f.budget.used == 1
    with pytest.raises(ProviderError):
        BoundedLiveBudget(
            binding=f.binding.model_copy(
                update={"deadline": f.binding.deadline + timedelta(seconds=1)}
            ),
            config=f.config,
            journal=f.budget.journal,
        )
    f.budget.clock = lambda: f.binding.deadline
    with pytest.raises(ValueError, match="deadline"):
        f.permission.check(f.reader.state)


def test_retry_only_once_and_400_no_retry_and_secret_error_not_retained():
    f = fixture([Reply(error="rate_limit", api_requests=1), review_from_wire])
    generate(f)
    assert f.budget.used == 2
    f = fixture([Reply(error="configuration", api_requests=1)])
    with pytest.raises(ProviderError, match="configuration"):
        generate(f)
    assert f.budget.used == 1
    error = safe_http_error(
        400, b'{"error":"secret-patient-value max_completion_tokens unsupported"}'
    )
    assert error["mentioned_fields"] == ["max_completion_tokens"]
    assert "secret-patient-value" not in json.dumps(error)


def test_forty_attempts_hard_cap_shared_and_missing_completion_stops():
    f = fixture([review_from_wire] * 40)
    for _ in range(40):
        generate(f)
    with pytest.raises(ProviderError, match="budget"):
        generate(f)
    assert len(f.wire.calls) == 40 and f.budget.summary().api_requests_observed == 40
    other = fixture(journal=f.budget.journal, b=f.binding)
    assert other.budget.used == 40
    f = fixture()
    f.budget.reserve(f.reader.state.material.scope, f.binding.deadline, digest(f.config))
    with pytest.raises(ProviderError):
        generate(f)
    assert not f.wire.calls


def test_search_cache_limit_and_unknown_claim():
    f = fixture()
    calls = []

    def search(q, **kwargs):
        calls.append(q)
        return {"hits": []}

    tool = BoundedKnowledge(
        documents=f.docs, budget=f.budget, tool=search, tool_reference="installed_test_tool"
    )
    for n in range(8):
        tool.search({"q": str(n)})
    tool.search({"q": "0"})
    assert len(calls) == 8
    with pytest.raises(ProviderError, match="budget"):
        tool.search({"q": "9"})
    f = fixture()

    def lost(q, **kwargs):
        raise OSError("synthetic_confirmation_lost")

    tool = BoundedKnowledge(
        documents=f.docs, budget=f.budget, tool=lost, tool_reference="installed_test_tool"
    )
    with pytest.raises(OSError):
        tool.search({"q": "one"})
    with pytest.raises(ValueError, match="unknown_no_repeat"):
        tool.search({"q": "one"})


def test_study_uses_llm_identity_and_requires_permission():
    f = fixture([review_from_wire] * 3)
    subs = [generate(f, r) for r in ("A", "B", "C")]
    ports = SimpleNamespace(budget=f.budget)
    with pytest.raises(ValueError, match="live_domain"):
        compare_candidates(f.reader.state, subs, protocol(), ports)
    result = compare_candidates(
        f.reader.state, subs, protocol(), ports, live_permission=f.permission
    )
    assert result["rounds"][0]["reviewer_kind"] == "llm"
    assert result["rounds"][0]["generation_settings"]["mode"] == "bounded_live_research"
    assert result["real_ABC_executed"]  # Synthetic wire identity test only, not a live result.


def test_public_probes_use_original_schemas_and_actual_synthetic_image_bytes():
    def source(body):
        p = json.loads(body["messages"][1]["content"][0]["text"])
        binding = body["response_format"]["json_schema"]["schema"]["properties"]["scope_checksum"]
        assert binding["enum"] == [digest(p["material"]["scope"])]
        return dict(
            scope_checksum=binding["enum"][0],
            source_code="SOURCE:PUBLIC_ARITHMETIC",
            quote="Two plus two equals four.",
            target="GOAL:PUBLIC_SUM",
            relation="requires",
            missing=[],
            reason="Synthetic protocol probe",
        )

    def vision(body):
        import base64

        p = json.loads(body["messages"][1]["content"][0]["text"])
        b = p["material"]["observations"][1]["image"]
        actual = base64.b64decode(
            body["messages"][1]["content"][1]["image_url"]["url"].split(",")[1]
        )
        assert actual.startswith(b"\x89PNG") and len(actual) > 30
        return dict(
            scope_checksum=digest(p["material"]["scope"]),
            image_checksum=b["reference"]["checksum"],
            preprocessing_checksum=b["preprocessing"]["checksum"],
            location=b["location"],
            status="observed",
            finding="A red square",
        )

    f = fixture([dict(model_listed=True), source, vision, review_from_wire])
    result = run_probes(provider=f.provider, budget=f.budget, documents=f.docs)
    assert all(
        result[k]["status"] == "passed" for k in ("models", "text", "image", "strict_schema")
    )
    assert f.budget.used == 4
    assert f.budget.transport_counts()["remote_response_confirmed"] == 4
    assert f.budget.transport_counts()["usage_unknown"] == 4
    assert f.budget.transport_counts()["cost"] == "unknown"
    assert f.budget.records()[0].endpoint == "models_get"
    assert run_probes(provider=f.provider, budget=f.budget, documents=f.docs) == result
    for body, _ in f.wire.calls[1:]:
        assert body["stream"] is False
        assert body["response_format"]["json_schema"]["strict"] is True
        assert body["max_completion_tokens"] > 0
        assert "CASE:PUBLIC_PROBE" in body["messages"][1]["content"][0]["text"]


def test_failed_public_candidate_retained_without_retry_or_acceptance():
    rejected = dict(
        scope_checksum=digest("wrong_scope"),
        source_code="SOURCE:PUBLIC_ARITHMETIC",
        quote="Two plus two equals four.",
        target="GOAL:PUBLIC_SUM",
        relation="requires",
        missing=[],
        reason="Synthetic rejected scope",
    )
    f = fixture([dict(model_listed=True), rejected])
    f.budget.journal.documents = f.docs
    with pytest.raises(ProviderError, match="scope"):
        run_probes(provider=f.provider, budget=f.budget, documents=f.docs)
    assert f.budget.used == 2 and len(f.wire.calls) == 2
    request = f.budget.records()[-1]
    raw = f.docs.read("live-response:" + request.request_id)[0]
    assert json.loads(raw["raw_model_text"]) == rejected
    assert raw["validation_error"] == "scope"
    assert raw["untrusted_model_output"] and not raw["accepted_medical_evidence"]
    assert f.docs.read("probe:text:result")[0]["error"] == "scope"
    with pytest.raises(ProviderError):
        run_probes(provider=f.provider, budget=f.budget, documents=f.docs)
    assert f.budget.used == 2 and len(f.wire.calls) == 2


def test_cleanup_owner_failure_keeps_primary_probe_error():
    from apps.vectorengine_live_assembly import stop_failed_run

    f = fixture()

    def unavailable(_):
        raise LookupError("synthetic_missing_owner")

    result = stop_failed_run(
        planner=None,
        sessions=SimpleNamespace(read=unavailable),
        binding=f.binding,
        budget=f.budget,
        error=ProviderError("scope"),
        cancel=None,
    )
    assert result["reason"] == "scope"
    assert result["cleanup_error_type"] == "LookupError"
    assert result["planner_read_confirmed"] is False
    assert result["requests"].reservations == 0


def test_missing_runtime_owner_preflight_prevents_run_and_network(monkeypatch, tmp_path):
    from apps.vectorengine_live_assembly import execute
    from tests.agent_provider.test_case_graph_target import ready_entry_state

    f = fixture()
    f.reader.state = ready_entry_state(f.reader.state)
    monkeypatch.setattr("apps.vectorengine_live_assembly.check_regression_gate", lambda: None)
    monkeypatch.setattr("db.runtime.repository.research.ResearchRunRepository", lambda _: object())

    def unavailable(_):
        raise LookupError("synthetic_missing_owner")

    monkeypatch.setattr(
        "db.runtime.repository.planner.PlannerRepository",
        lambda _: SimpleNamespace(read=unavailable),
    )
    with pytest.raises(LookupError, match="synthetic_missing_owner"):
        execute(
            evidence=None, target=None, initial=f.reader.state, config=f.config, output=tmp_path
        )
    assert f.budget.used == 0 and not f.wire.calls


def test_review_constructor_default_closed():
    f = fixture()
    agent = SimpleNamespace(provider=f.provider, budget=f.budget)
    ports = SimpleNamespace(
        budget=f.budget, trace=agent, expert=agent, source=agent, vision=agent, review=agent
    )
    with pytest.raises(ValueError, match="live_domain"):
        ResearchReview(
            acceptance=ResearchAcceptance(owner=f.reader, documents=f.docs, requests=f.budget),
            ports=ports,
            context=None,
            protocol=protocol(),
        )


def test_actual_composition_root_with_synthetic_memory_ports(monkeypatch):
    from apps.vectorengine_live_assembly import assemble, make_binding
    from packages.planner_control.models import Command
    from tests.agent_provider.support import ref
    from tests.planner.support import MemoryRuntime

    f = fixture()
    sessions = MemoryRuntime()
    sessions.origin = lambda run: (f.binding.started_at, f.binding.run_context_checksum)
    f.docs.sessions = sessions
    binding, ranking = make_binding(
        initial=f.reader.state,
        config=f.config,
        started=f.binding.started_at,
        context_checksum=f.binding.run_context_checksum,
        catalog=ref("catalog", "CATALOG:SYNTHETIC"),
    )
    evidence = SimpleNamespace(verify=lambda state: None, image=lambda r: b"")
    monkeypatch.setattr(
        "packages.llm_provider.transport.HTTPTransport.send",
        lambda self, config, body, **kw: f.wire.send(config, body, **kw),
    )
    planner, owner, ports, review = assemble(
        evidence=evidence,
        initial=f.reader.state,
        config=f.config,
        documents=f.docs,
        binding=binding,
        ranking=ranking,
    )
    owner.role = "review"
    planner.begin(binding)
    for action in ("load", "retrieve", "instantiate", "identify"):
        state = planner.read(binding.run_id)
        planner.step(
            binding.run_id,
            Command(key="step:" + action, expected_revision=state.revision, action=action),
        )
        assert planner.read(binding.run_id).phase != "stopped"
    state = planner.read(binding.run_id)
    assert state.phase == "select"
    cmd = Command(key="review", expected_revision=state.revision, action="review")
    event = planner.begin_recovery_review(binding.run_id, cmd)
    frame = review.context.read(event)
    chosen = next(a for a in frame.actions.actions if a.branch == "research_review")
    result = review.run(event=event, proposed_action_id=chosen.id, role="review")
    assert result["status"] == "accepted"
    assert result["schema_version"] == "research-review-result/3.0.0"
    assert planner.read(binding.run_id).steps <= 20
    assert len(f.wire.calls) == 1


def test_catalog_failure_remains_failed_on_resume_no_paid_probe():
    f = fixture([dict(model_listed=False)])
    for _ in range(2):
        with pytest.raises(ProviderError, match="configuration"):
            run_probes(provider=f.provider, budget=f.budget, documents=f.docs)
    assert f.budget.used == 1 and len(f.wire.calls) == 1


def test_cancel_is_persisted_and_profile_cannot_expand():
    from threading import Event

    f = fixture()
    f.budget.cancel_signal = Event()
    f.budget.cancel_signal.set()
    with pytest.raises(ProviderError, match="cancelled"):
        generate(f)
    restored = fixture(journal=f.budget.journal, b=f.binding)
    with pytest.raises(ProviderError, match="cancelled"):
        generate(restored)
    assert not f.wire.calls and not restored.wire.calls
    for name, value in (
        ("max_steps", 21),
        ("max_no_progress", 3),
        ("max_retries", 2),
        ("seconds", 2701),
    ):
        with pytest.raises(ProviderError, match="configuration"):
            BoundedLiveBudget(
                binding=f.binding.model_copy(
                    update={"profile": f.binding.profile.model_copy(update={name: value})}
                ),
                config=f.config,
                journal=SyntheticJournal(),
            )


def test_formal_graph_and_provider_switch_are_not_covered_by_grant():
    f = fixture()
    scope = f.reader.state.material.scope
    state = f.reader.state.model_copy(
        update={
            "material": f.reader.state.material.model_copy(
                update={
                    "scope": scope.model_copy(
                        update={
                            "graph": scope.graph.model_copy(
                                update={"kind": "formal_geceg", "code": "GRAPH:OTHER"}
                            )
                        }
                    )
                }
            )
        }
    )
    with pytest.raises(ValueError, match="scope_forbidden"):
        f.permission.check(state)
    from packages.evidence_details.live_permission import validate_live_config

    with pytest.raises(ValueError):
        validate_live_config(
            f.config.model_copy(
                update={
                    "transport": f.config.transport.model_copy(update={"model": "another-model"})
                }
            )
        )
