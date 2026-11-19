"""Confirmed Vision state consistency feedback; no invented observations or replay."""

import base64
import copy
import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from packages.grounding.models import canonical, digest, validate
from packages.llm_provider.models import ImageBinding, Material, PartialVisionProposal
from packages.llm_provider.provider import (
    DeterministicFakeProvider,
    MemoryJournal,
    ProviderError,
    RequestBudget,
    StructuredProvider,
)
from packages.llm_provider.review_correction import verify_correction
from packages.llm_provider.transport import HTTPTransport, Reply
from packages.llm_provider.vision_content import prepare
from packages.multimodal_research.campaign import configuration
from packages.object_store import checksum
from tests.agent_provider.test_http_recovery import injected_worker
from tests.agent_provider.test_vision_projection import payload
from tests.evaluation_preparation.test_partial_observation import detail


def fixture(*, feedback=True):
    config = configuration(
        resilient_vision=True,
        directional=True,
        exact_primary_status=True,
        execution_profile="relay-root-question-v1",
        review_batch_size=3,
        **({"vision_feedback": "state-consistency-once/1.0.0"} if feedback else {}),
    )
    p = payload()
    data = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Wl2nS8AAAAASUVORK5CYII="
    )
    m = p["material"]
    m["scope"]["case_code"] = "CASE:SYNTHETIC_VISION_FEEDBACK"
    image = m["observations"][0]["image"]
    image.update(width=1, height=1, location=[0, 0, 1, 1])
    image["reference"]["checksum"] = checksum(data)
    m["scope"].pop("material_checksum")
    m["scope"]["material_checksum"] = digest(m)
    scope = validate(Material, m).scope
    images = ((validate(ImageBinding, image), data),)
    _, code, _ = prepare(scope, images, PartialVisionProposal)
    good = dict(
        schema_version="visual-observation-content/1.0.0",
        binding_code=code,
        status="observed",
        finding="Synthetic visible region only",
        partial=detail(),
    )
    bad = copy.deepcopy(good)
    bad["status"] = "unknown"
    deadline = datetime.now(UTC) + timedelta(minutes=5)
    budget = RequestBudget(
        run_id=scope.run_id, deadline=deadline, max_requests=4, journal=MemoryJournal()
    )
    return SimpleNamespace(
        config=config,
        payload=p,
        scope=scope,
        images=images,
        good=good,
        bad=bad,
        budget=budget,
        deadline=deadline,
    )


def generate(f, replies, *, before_request=None, provider=None):
    p = provider or DeterministicFakeProvider(f.config.model_copy(update={"mode": "fake"}), replies)
    return p.generate_structured(
        role="vision",
        input_payload=f.payload,
        output_schema=PartialVisionProposal,
        scope=f.scope,
        budget=f.budget,
        deadline=f.deadline,
        images=f.images,
        before_request=before_request,
    )


def confirmed(value):
    return Reply(
        content=canonical(value), finish_reason="stop", response_confirmed=True, api_requests=1
    )


def role_agent(f, provider):
    """Use the production target wrapper and role gate, not only Provider."""
    from agents.provider_agents import StructuredRoleAgent
    from packages.multimodal_research.campaign import PartialTargetReader
    from tests.agent_provider.support import Reader
    from tests.evaluation_preparation.test_dynamic_evidence import dynamic_fixture

    state = (
        dynamic_fixture(1)
        .owner.current()
        .model_copy(
            update={
                "material": validate(Material, f.payload["material"]),
                "prompt_checksum": next(
                    r.prompt_checksum for r in provider.config.roles if r.role == "vision"
                ),
                "config_checksum": digest(provider.config),
            }
        )
    )
    owner = Reader(state)
    owner.image_bytes = f.images[0][1]
    return StructuredRoleAgent(
        "vision", provider=provider, materials=PartialTargetReader(owner), budget=f.budget
    )


@pytest.mark.parametrize("feedback", [False, True])
def test_production_partial_target_role_accepts_actual_config(feedback):
    f = fixture(feedback=feedback)
    provider = DeterministicFakeProvider(f.config.model_copy(update={"mode": "fake"}), [f.good])
    result = role_agent(f, provider).call(f.scope, deadline=f.deadline)
    assert result.status == "observed" and f.budget.used == 1
    assert result.scope_checksum == digest(f.scope)


def test_production_partial_target_role_http_feedback(monkeypatch):
    f = fixture()
    durable_budget(f)
    calls = injected_worker(monkeypatch, [f.bad, f.good])
    provider = StructuredProvider(f.config, live_opt_in=True, transport=HTTPTransport())
    result = role_agent(f, provider).call(f.scope, deadline=f.deadline)
    assert result.status == "observed" and len(calls) == f.budget.used == 2
    assert [r.error for r in f.budget.records()] == ["schema", None]
    verify_correction(f.budget.records()[1], f.budget)


def test_production_partial_target_role_unknown_does_not_replay(monkeypatch):
    f = fixture()
    durable_budget(f)
    calls = injected_worker(monkeypatch, ["unknown", f.good])
    provider = StructuredProvider(f.config, live_opt_in=True, transport=HTTPTransport())
    with pytest.raises(ProviderError, match="timeout"):
        role_agent(f, provider).call(f.scope, deadline=f.deadline)
    assert len(calls) == f.budget.used == 1


def test_production_partial_target_role_rechecks_material_after_response():
    f = fixture()
    agent = None

    def response(body):
        agent.materials.owner.withdrawn = True
        return f.good

    provider = DeterministicFakeProvider(f.config.model_copy(update={"mode": "fake"}), [response])
    agent = role_agent(f, provider)
    with pytest.raises(ProviderError, match="scope"):
        agent.call(f.scope, deadline=f.deadline)
    assert f.budget.used == 1


def test_production_planner_partial_vision_acceptance_and_saved_replay(monkeypatch, tmp_path):
    from threading import Event

    import tests.expert_agent.test_batch_assembly as support
    from apps.vectorengine_live_assembly import run_case_stages
    from packages.multimodal_research.campaign import PartialTargetReader

    config = fixture().config
    monkeypatch.setattr(support, "load_config", lambda path: config)
    original = support.assemble

    def install(**kwargs):
        kwargs["evidence"].wrap_owner = PartialTargetReader
        return original(**kwargs)

    monkeypatch.setattr(support, "assemble", install)

    def reply(body):
        schema = next(
            json.loads(c["text"].split("\n", 1)[1])
            for c in body["messages"][1]["content"]
            if c.get("type") == "text" and c["text"].startswith("OUTPUT_JSON_SCHEMA:")
        )
        return confirmed(
            dict(
                schema_version="visual-observation-content/1.0.0",
                binding_code=schema["properties"]["binding_code"]["enum"][0],
                status="observed",
                finding="Synthetic visible region, not a medical conclusion",
                partial=detail(),
            )
        )

    f = support.assembly(monkeypatch, count=1, replies=[reply])
    args = dict(
        planner=f.planner,
        owner=f.owner,
        ports=f.ports,
        review=f.review,
        documents=f.documents,
        binding=f.binding,
        config=f.config,
        output=tmp_path,
        signal_cancel=Event(),
        selected_roles=("vision",),
    )
    completed, roles = run_case_stages(**args)
    assert roles == ("vision",) and completed[0]["domain_status"] == "accepted"
    saved = f.review.acceptance.read(key=completed[0]["acceptance_key"])[0]
    assert saved["candidate"]["schema_version"] == "visual-observation-candidate/2.0.0"
    assert json.loads(saved["observations"][0]["value"])["signal"] == "absent_in_visible_region"
    audit = list(f.documents.audit)
    assert run_case_stages(**args)[0] == completed
    assert f.documents.audit == audit and len(f.wire.calls) == f.ports.budget.used == 1


@pytest.mark.parametrize(
    "status,signal",
    [
        ("observed", "absent_in_visible_region"),
        ("unknown", "indeterminate"),
    ],
)
def test_one_correction_keeps_original_and_does_not_force_observed(status, signal):
    f = fixture()
    good = copy.deepcopy(f.good)
    good["status"], good["partial"]["signal"] = status, signal
    original = canonical(f.bad)

    def corrected(body):
        feedback = json.loads(body["messages"][-1]["content"])
        assert feedback["errors"][0]["constraint"] == "partial_status_signal"
        assert "state" in feedback["task"].lower()
        assert f.bad["finding"] not in json.dumps(feedback)
        return good

    value = generate(f, [confirmed(f.bad), corrected])
    assert (value.status, value.partial.signal) == (status, signal)
    assert canonical(f.bad) == original and f.budget.used == 2
    records = f.budget.records()
    assert [r.error for r in records] == ["schema", None]
    assert records[0].invocation_id == records[1].invocation_id
    verify_correction(records[1], f.budget)


@pytest.mark.parametrize("feedback,count", [(False, 1), (True, 2)])
def test_old_prompt_unchanged_and_second_bad_stops(feedback, count):
    f = fixture(feedback=feedback)
    with pytest.raises(ProviderError, match="schema"):
        generate(f, [confirmed(f.bad), confirmed(f.bad), f.good])
    assert f.budget.used == count


@pytest.mark.parametrize("error", ["timeout", "transport", "rate_limit"])
def test_unknown_transport_not_replayed(error):
    f = fixture()
    with pytest.raises(ProviderError, match=error):
        generate(f, [Reply(error=error, response_confirmed=False), f.good])
    assert f.budget.used == 1


@pytest.mark.parametrize("mode", ["budget", "cancel", "deadline", "source"])
def test_original_guards_before_second_dispatch(mode):
    f = fixture()
    if mode == "budget":
        f.budget.max_requests = 1

    def guard():
        if f.budget.used and mode != "budget":
            raise ProviderError(
                {"cancel": "cancelled", "deadline": "deadline", "source": "scope"}[mode]
            )

    with pytest.raises(
        ProviderError, match={"cancel": "cancelled", "source": "scope"}.get(mode, mode)
    ):
        generate(f, [confirmed(f.bad), f.good], before_request=guard)
    assert f.budget.used == 1


def test_wrong_binding_not_repaired_and_unknown_status_consistent_is_valid():
    f = fixture()
    f.bad["binding_code"] = "V:" + "0" * 16
    with pytest.raises(ProviderError, match="schema"):
        generate(f, [confirmed(f.bad), f.good])
    assert f.budget.used == 1


def durable_budget(f):
    from packages.research_runtime.agent_records import RuntimeAgentJournal
    from tests.agent_provider.support import Documents

    docs = Documents()
    docs.run_id = f.scope.run_id
    docs.sessions = SimpleNamespace(origin=lambda run: (f.deadline - timedelta(hours=1), None))
    f.budget = RequestBudget(
        run_id=f.scope.run_id,
        deadline=f.deadline,
        max_requests=4,
        journal=RuntimeAgentJournal(docs),
    )


def test_original_http_parser_and_cold_journal_account_two_requests(monkeypatch):
    f = fixture()
    durable_budget(f)
    calls = injected_worker(monkeypatch, [f.bad, f.good])
    p = StructuredProvider(f.config, live_opt_in=True, transport=HTTPTransport())
    value = generate(f, [], provider=p)
    assert value.status == "observed" and len(calls) == f.budget.used == 2
    verify_correction(f.budget.records()[1], f.budget)
    cold = RequestBudget(
        run_id=f.scope.run_id, deadline=f.deadline, max_requests=4, journal=f.budget.journal
    )
    assert cold.used == 2 and len(calls) == 2
    verify_correction(cold.records()[1], cold)
