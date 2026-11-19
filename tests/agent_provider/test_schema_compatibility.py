"""Synthetic worker replies only. Real owner contracts, no supplier or patient data."""

import json
from types import SimpleNamespace

import pytest

from agents.planner.service import Planner
from apps.vectorengine_round import load_round
from apps.vectorengine_single_case import load_config
from packages.grounding.models import digest
from packages.llm_provider.bounded_live import BoundedLiveBudget
from packages.llm_provider.live_allocation import StageAllocation
from packages.llm_provider.provider import ProviderError
from packages.llm_provider.public_compatibility import (
    check_qualification,
    construct,
    controlled_probe,
    load_plan,
    public_materials,
    run_comparisons,
)
from packages.llm_provider.wire_compatibility import WireSchemaError, normalize, preflight
from packages.llm_provider.worker import safe_http_error
from packages.planner_control.models import Command, Profile
from packages.research_runtime.agent_records import RuntimeAgentJournal
from tests.agent_provider.support import Documents, review_from_wire
from tests.agent_provider.test_http_recovery import injected_worker
from tests.planner.support import FakeOwner, MemoryRuntime, binding

CONFIG = "configs/research/vectorengine_schema_compat_public.json"


def setup():
    r = load_round(CONFIG)
    plan = load_plan(r)
    cfg = load_config(r.provider_config)
    b = binding(
        run_id=r.run_id,
        profile=Profile(max_steps=20, max_no_progress=2, max_retries=1, seconds=1200),
    )
    docs = Documents()
    docs.run_id = b.run_id
    docs.sessions = MemoryRuntime()
    docs.sessions.origin = lambda _: (b.started_at, b.run_context_checksum)
    planner = Planner(runtime=docs.sessions, owner=FakeOwner(b))
    planner.begin(b)
    allocation = StageAllocation(
        session_id=plan.session_id,
        stage="public",
        public_run_id=b.run_id,
        public_used=0,
        public_journal_checksum=None,
        configuration_checksums=tuple(c.config_checksum for c in plan.candidates),
    )
    budget = BoundedLiveBudget(
        binding=b,
        config=cfg,
        journal=RuntimeAgentJournal(docs),
        allocation=allocation,
        diagnostic_plan=plan,
    )
    return SimpleNamespace(
        round=r,
        plan=plan,
        cfg=cfg,
        b=b,
        docs=docs,
        planner=planner,
        budget=budget,
        allocation=allocation,
    )


def format_failure(body):
    error = {
        "error": {
            "message": (
                "Invalid schema for response_format 'vision': "
                "In context=('properties', 'location'), array schema missing items."
            ),
            "type": "invalid_request_error",
            "param": "response_format",
            "code": "invalid_json_schema",
        }
    }
    diagnostic = safe_http_error(
        429,
        json.dumps(error).encode(),
        schema=body["response_format"].get("json_schema", {}).get("schema"),
        body=body,
    )
    return dict(error="rate_limit", diagnostic=diagnostic)


def success(body):
    payload = json.loads(body["messages"][1]["content"][0]["text"])
    if body["response_format"]["type"] == "json_schema":
        schema = body["response_format"]["json_schema"]["schema"]
    else:
        schema = json.loads(body["messages"][1]["content"][-1]["text"].split("\n", 1)[1])
    props = schema["properties"]
    role = payload["role"]
    if role == "vision":
        return {
            **{
                k: props[k]["enum"][0]
                for k in ("scope_checksum", "image_checksum", "preprocessing_checksum", "location")
            },
            "status": "observed",
            "finding": "A red synthetic square",
        }
    if role == "source":
        return dict(
            scope_checksum=props["scope_checksum"]["enum"][0],
            target=props["target"]["enum"][0],
            source_code=props["source_code"]["enum"][0],
            quote="Two plus two equals four.",
            relation="derived_from",
            missing=[],
            reason="Synthetic source",
        )
    value = review_from_wire(body)
    missing = {
        t["assessment_id"]
        for t in payload["review_tasks"]["tasks"]
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
                    reason="Synthetic missing",
                    method="not_assessed",
                    missing_reason="Definition missing",
                ),
            )
    return value


def test_actual_build_normalizes_all_roles_and_keeps_legacy_wire():
    f = setup()
    old = json.loads(
        open("docs/acceptance/12-live-compat-case-v1/actual/public-vision-wire.json").read()
    )["response_format"]["json_schema"]["schema"]
    with pytest.raises(WireSchemaError, match="prefixItems"):
        preflight(old)
    for candidate in f.plan.candidates:
        config = load_config(candidate.config_path)
        for role in ("source", "vision", "review"):
            body, _ = construct(config, public_materials(f.b.run_id, config, role), role)
            schema = (
                body["response_format"]["json_schema"]["schema"]
                if config.wire_mode == "normalized_strict"
                else json.loads(body["messages"][1]["content"][-1]["text"].split("\n", 1)[1])
            )
            assert preflight(schema)["subset_preflight_passed"]
            if role == "vision":
                loc = schema["properties"]["location"]
                assert loc["items"] == {"type": "integer"} and "prefixItems" not in loc
                assert loc["minItems"] == loc["maxItems"] == 4 and loc["enum"] == [[0, 0, 32, 32]]
    assert "prefixItems" in old["properties"]["location"]


def test_heterogeneous_tuple_and_dangling_ref_fail_locally():
    for schema in [
        dict(
            type="object",
            properties=dict(
                x=dict(
                    type="array",
                    prefixItems=[{"type": "integer"}, {"type": "string"}],
                    minItems=2,
                    maxItems=2,
                )
            ),
            required=["x"],
            additionalProperties=False,
        ),
        dict(
            type="object",
            properties=dict(x={"$ref": "#/$defs/missing"}),
            required=["x"],
            additionalProperties=False,
        ),
    ]:
        with pytest.raises(WireSchemaError):
            normalize(schema)


@pytest.mark.parametrize("bad", [[0, 0, 32], [0, 0, 32, 32, 0], [0, 0, 64, 64]])
def test_bad_length_or_coordinate_still_rejected(monkeypatch, bad):
    f = setup()

    def wrong(body):
        reply = success(body)
        reply["location"] = bad
        return reply

    calls = injected_worker(monkeypatch, [wrong])
    c = f.plan.candidates[0]
    result = controlled_probe(
        candidate=c, request=c.requests[0], config=f.cfg, budget=f.budget, documents=f.docs
    )
    assert result["status"] == "failed" and result["error"] in {"schema", "scope"}
    assert f.budget.used == len(calls) == 1


def test_429_path_is_schema_owned_and_secrets_patient_echo_never_escape():
    f = setup()
    body, _ = construct(f.cfg, public_materials(f.b.run_id, f.cfg, "vision"), "vision")
    good = format_failure(body)["diagnostic"]
    assert (
        good["category"] == "request_format_rejected"
        and good["error_code"] == "invalid_json_schema"
    )
    assert (
        good["schema_paths"] == [["properties", "location"]]
        and "array_missing_items" in good["schema_rules"]
    )
    for message in [
        "Invalid schema response_format context=('properties','patient-secret') patient-secret",
        "Invalid schema response_format context=('properties','location') private-key-value",
    ]:
        raw = json.dumps(
            {
                "error": {
                    "message": message,
                    "type": "invalid_request_error",
                    "param": "patient-secret",
                    "code": "private-key-value",
                }
            }
        ).encode()
        result = safe_http_error(
            429,
            raw,
            key="private-key-value",
            schema=body["response_format"]["json_schema"]["schema"],
            body=body,
        )
        assert "patient-secret" not in json.dumps(result) and "private-key-value" not in json.dumps(
            result
        )
        assert result["schema_paths"] == []


def test_confirmed_format_rejection_next_frozen_candidate_success_and_restart(monkeypatch):
    f = setup()
    calls = injected_worker(monkeypatch, [format_failure, success, success, success])
    q = run_comparisons(f.plan, f.budget, f.docs, digest(f.round))
    assert q["candidate_id"] == "json_object" and len(q["roles"]) == 3
    assert all(r["remote_constraint"] == "json_object" for r in q["roles"])
    assert f.planner.read(f.b.run_id).phase != "stopped"
    assert f.budget.used == len(calls) == 4
    assert [r.retry_index for r in f.budget.records()] == [0] * 4
    check_qualification(f.plan, q, f.docs, f.plan.candidates[1].config_checksum)
    again = BoundedLiveBudget(
        binding=f.b,
        config=f.cfg,
        journal=RuntimeAgentJournal(f.docs),
        allocation=f.allocation,
        diagnostic_plan=f.plan,
    )
    assert run_comparisons(f.plan, again, f.docs, digest(f.round)) == q and again.used == 4
    with pytest.raises(ValueError, match="binding"):
        check_qualification(f.plan, q, f.docs, f.plan.candidates[0].config_checksum)


def test_failed_candidate_replay_and_exhaustion_do_not_resend(monkeypatch):
    f = setup()
    calls = injected_worker(monkeypatch, [format_failure, format_failure])
    for _ in range(2):
        with pytest.raises(ProviderError, match="configuration"):
            run_comparisons(f.plan, f.budget, f.docs, digest(f.round))
    assert len(calls) == f.budget.used == 2
    assert not f.docs.read("public-qualification")


@pytest.mark.parametrize("kind", ["unknown", "cancel", "deadline", "quota"])
def test_terminal_conditions_cannot_enter_next_candidate(monkeypatch, kind):
    f = setup()

    def response(body):
        if kind == "cancel":
            state = f.planner.read(f.b.run_id)
            f.planner.step(
                f.b.run_id, Command(key="cancel", action="cancel", expected_revision=state.revision)
            )
        elif kind == "deadline":
            f.budget.clock = lambda: f.b.deadline
        elif kind == "quota":
            return dict(
                error="rate_limit",
                diagnostic=safe_http_error(429, b'{"error":{"code":"insufficient_quota"}}'),
            )
        return format_failure(body)

    calls = injected_worker(monkeypatch, ["unknown" if kind == "unknown" else response])
    with pytest.raises((ProviderError, ValueError)):
        run_comparisons(f.plan, f.budget, f.docs, digest(f.round))
    assert f.budget.used == len(calls) == 1 and not f.docs.read("public-qualification")


def test_changed_wire_same_slot_and_unplanned_config_are_denied(monkeypatch):
    f = setup()
    c = f.plan.candidates[0]
    req = c.requests[0]
    calls = injected_worker(monkeypatch, [])
    with pytest.raises(ValueError, match="wire_lock"):
        controlled_probe(
            candidate=c,
            request=req.model_copy(update={"wire_checksum": digest("different")}),
            config=f.cfg,
            budget=f.budget,
            documents=f.docs,
        )
    f.docs.write("compat2:" + c.id + ":vision:lock", (dict(wrong=True),))
    with pytest.raises(ValueError, match="slot_conflict"):
        controlled_probe(candidate=c, request=req, config=f.cfg, budget=f.budget, documents=f.docs)
    assert not calls and f.budget.used == 0


def test_exhaustion_uses_original_stop_owner(monkeypatch):
    from threading import Event

    from apps.vectorengine_live_assembly import stop_failed_run

    f = setup()
    calls = injected_worker(monkeypatch, [format_failure, format_failure])
    with pytest.raises(ProviderError) as failed:
        run_comparisons(f.plan, f.budget, f.docs, digest(f.round))
    stopped = stop_failed_run(
        planner=f.planner,
        sessions=f.docs.sessions,
        binding=f.b,
        budget=f.budget,
        error=failed.value,
        cancel=Event(),
    )
    assert stopped["cleanup_error_type"] is None
    assert f.planner.read(f.b.run_id).phase == "stopped"
    with pytest.raises(ProviderError, match="stopped"):
        f.budget.before_reserve(f.plan.candidates[1].config_checksum)
    assert len(calls) == f.budget.used == 2


def test_forged_format_category_cannot_unlock_budget(monkeypatch):
    from packages.llm_provider.models import RequestRecord, Usage

    f = setup()
    scope = public_materials(f.b.run_id, f.cfg, "vision").state.material.scope
    rid, number, until = f.budget.reserve(scope, f.b.deadline, digest(f.cfg))
    req = f.plan.candidates[0].requests[0]
    record = RequestRecord(
        request_id=rid,
        invocation_id=rid,
        retry_index=0,
        role="vision",
        scope=scope,
        provider="openai_compatible",
        model="gpt-5.6-sol",
        endpoint="chat_completions",
        config_checksum=digest(f.cfg),
        prompt_version="vision-1.0.0",
        prompt_checksum=req.prompt_checksum,
        input_checksum=req.input_checksum,
        output_checksum=None,
        remote_request_checksum=None,
        attempt=number,
        simulated=False,
        api_requests=1,
        usage=Usage(),
        latency_seconds=0,
        error="rate_limit",
        deadline=until,
    )
    f.budget.record(record, response_confirmed=True)
    f.budget.journal.append(
        dict(
            schema_version="agent-transport-diagnostic/1.0.0",
            request_id=rid,
            value=format_failure(
                construct(f.cfg, public_materials(f.b.run_id, f.cfg, "vision"), "vision")[0]
            )["diagnostic"],
        )
    )
    with pytest.raises(ProviderError, match="rate_limit"):
        f.budget.before_reserve(f.plan.candidates[1].config_checksum)
