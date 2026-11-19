"""Explicit root question reaches wire/domain; no force-nonzero or score flipping."""

import json
from dataclasses import replace
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from packages.grounding.models import digest
from packages.multimodal_research import root_review_window as window
from packages.multimodal_research.single_proposition import (
    root_direction_spec,
    root_direction_state,
)
from tests.evaluation_preparation.test_review_shape_window import old_config


def task_for(state):
    return SimpleNamespace(
        code="TASK:SYNTHETIC", target_code=state.material.goal_code, checksum=digest("task")
    )


@pytest.mark.parametrize("old_version", ["2.4.0", "2.5.0"])
def test_only_versioned_review_prompt_changes(old_version):
    old = old_config()
    if old_version == "2.5.0":
        from packages.multimodal_research.review_shape_window import configuration

        old = configuration(old)
    new = window.configuration(old)
    a, b = old.model_dump(mode="json"), new.model_dump(mode="json")
    ar = next(r for r in a["roles"] if r["role"] == "review")
    br = next(r for r in b["roles"] if r["role"] == "review")
    assert br["prompt_version"] == window.PROMPT
    br.update(prompt_version=ar["prompt_version"], prompt_checksum=ar["prompt_checksum"])
    assert a == b
    with pytest.raises(ValueError, match="original_configuration"):
        window.configuration(new)


def test_projection_keeps_policy_observations_sources_and_old_bytes():
    from tests.evaluation_preparation.test_directional_review import fixture

    f = fixture()
    old = f.owner.current()
    before = old.model_dump_json()
    task = task_for(old)
    spec = root_direction_spec(task)
    new = root_direction_state(old, task, spec)
    a, b = new.policy.model_dump(), old.policy.model_dump()
    assert a.pop("target_summary") == new.material.goal_summary
    b.pop("target_summary")
    assert a == b
    assert new.material.observations == old.material.observations
    assert new.material.sources == old.material.sources
    assert new.material.goal_code == old.material.goal_code
    assert "e>0" in new.material.goal_summary and "e<0" in new.material.goal_summary
    assert new.material.scope.material_checksum != old.material.scope.material_checksum
    assert old.model_dump_json() == before
    assert root_direction_state(new, task, spec) == new
    for change in (
        {"desired_direction": "refute"},
        {"title": "different"},
        {"target_code": "OTHER"},
    ):
        with pytest.raises(ValueError, match="question_binding"):
            root_direction_state(old, task, {**spec, **change})


def test_actual_root_title_is_derived_from_original_task_owner():
    from packages.dataset_adapters.multimodal.tasks import load_intent

    task = load_intent("advanced_glaucoma", "configs/research/multimodal_propositions_v1.json")[0]
    spec = root_direction_spec(task)
    assert spec["title"] == "晚期青光眼" and spec["target_code"] == task.target_code
    assert spec["desired_direction"] is None and not spec["medical_relation_changed"]


@pytest.mark.parametrize("estimate", [-0.03, 0.0, 0.03])
@pytest.mark.parametrize("batch_size", [1, 3])
def test_root_question_actual_wire_acceptance_readback_and_repeat(estimate, batch_size):
    from agents.expert_agent.reporting import read_research_report, render_research_html
    from packages.llm_provider.provider import StructuredProvider
    from packages.llm_provider.transport import HTTPTransport, parse_reply
    from tests.agent_provider.batch_support import generate
    from tests.evaluation_preparation.test_directional_review import explanation, fixture, reply

    f = fixture()
    config = window.configuration(
        old_config(), citation_projection=True, batch_size=batch_size
    ).model_copy(update={"mode": "fake"})
    f.config = config
    old = f.owner.current()
    task = task_for(old)
    state = root_direction_state(old, task, root_direction_spec(task))
    f.owner.state = state.model_copy(
        update={
            "config_checksum": digest(config),
            "prompt_checksum": next(r.prompt_checksum for r in config.roles if r.role == "review"),
        }
    )

    class InjectedHTTP(HTTPTransport):
        simulated = True

        def __init__(self):
            self.calls = []

        def send(self, config, body, **kwargs):
            self.calls.append(body)
            assert "root-direction-question/1.0.0" in body["messages"][0]["content"]
            assert "e>0" in json.dumps(body, ensure_ascii=False)
            assert task.target_code in json.dumps(body)
            raw = reply(body)
            for item in raw["items"]:
                if item["status"] == "assessed":
                    item["judgment"].update(value=estimate, lower=-0.2, upper=0.3)
                    item["direction_assessment"].update(explanation(estimate))
            envelope = {
                "choices": [{"message": {"content": json.dumps(raw)}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 10},
            }
            parsed = parse_reply(
                json.dumps(envelope).encode(), "chat_completions", retain_metadata=True
            )
            return replace(parsed, api_requests=0)

    transport = InjectedHTTP()
    f.agent.provider = StructuredProvider(config, transport=transport)
    sub = generate(f)
    f.acceptance.submit(sub, key="root-question")
    assert f.acceptance.accept(key="root-question").status == "accepted"
    report = read_research_report(f.acceptance, key="root-question", parent=f.owner.current())
    assert "e&gt;0" in render_research_html(report) or "e>0" in render_research_html(report)
    assert any(i.judgment.value == estimate for i in sub.candidate.items)
    count = len(transport.calls)
    assert generate(f) == sub and len(transport.calls) == count
    assert all(r.simulated and r.api_requests == 0 for r in f.budget.records())
    assert sub.plan.policy.batch_size == batch_size
    assert all(len(b.assessment_ids) <= batch_size for b in sub.plan.batches)
    if batch_size == 3:
        assert any(len(b.assessment_ids) == 3 for b in sub.plan.batches)


def test_multi_item_root_wire_has_per_task_citation_table_and_no_implicit_upgrade():
    from packages.llm_provider.public_shape import PublicReviewMaterials
    from tests.agent_provider.test_qualified_citation_wire import body

    config = window.configuration(old_config(), citation_projection=True, batch_size=3)
    state = PublicReviewMaterials("RUN:ROOT_BATCH3", config, task_count=3).current()
    sent = body(config, state)
    text = next(
        c["text"]
        for c in sent["messages"][1]["content"]
        if c["text"].startswith("REVIEW_CITATION_CODES:")
    )
    table = json.loads(text.split("\n", 1)[1])["tasks"]
    assert len(table) == 3
    assert [r["observation_citations"] for r in table] == [
        ["OBS:PUBLIC_00"],
        ["OBS:PUBLIC_01"],
        ["OBS:PUBLIC_02"],
    ]
    old = window.configuration(old_config())
    assert old.review_batching.batch_size == 1
    for bad in (0, 4, True, 1.5):
        with pytest.raises(ValueError, match="batch_size"):
            window.configuration(old_config(), citation_projection=True, batch_size=bad)
    with pytest.raises(ValueError, match="batch_size"):
        window.configuration(old_config(), batch_size=3)


def test_old_root_wire_keeps_history_new_wire_has_exact_primary_identities():
    from packages.llm_provider.public_shape import PublicReviewMaterials
    from tests.agent_provider.test_qualified_citation_wire import body

    old = window.configuration(old_config())
    new = window.configuration(old_config(), citation_projection=True)
    state = PublicReviewMaterials("RUN:ROOT_CITATIONS", old, task_count=3).current()
    old_body = body(old, state)
    current = body(new, state)  # Actual wire_request with explicit live scope_binding.
    assert old_body == body(old, state)
    assert not any(
        c["text"].startswith("REVIEW_CITATION_CODES:") for c in old_body["messages"][1]["content"]
    )
    text = next(
        c["text"]
        for c in current["messages"][1]["content"]
        if c["text"].startswith("REVIEW_CITATION_CODES:")
    )
    table = json.loads(text.split("\n", 1)[1])["tasks"]
    assert len(table) == 1
    assert table[0]["observation_citations"] == ["OBS:PUBLIC_00"]
    assert table[0]["required_observation_available"] is True
    assert "input_missing" not in table[0]["allowed_primary_statuses"]
    assert "observation_ungradable" not in table[0]["allowed_primary_statuses"]
    assert "judgment" not in table[0]
    assert old.transport == new.transport
    assert [r for r in old.roles if r.role != "review"] == [
        r for r in new.roles if r.role != "review"
    ]


def test_bounds_and_old_deadline_not_extended():
    parent = SimpleNamespace(
        config=old_config(),
        protocol={"deadline": "2026-10-18T00:00:00+00:00"},
        base=SimpleNamespace(
            manifest=SimpleNamespace(checksum=digest("input")),
            task=SimpleNamespace(checksum=digest("task")),
        ),
        accepted_observations=lambda: {"a": "original"},
    )
    proof = {"model_review_ids": list(range(9)), "review_target_ids": list(range(9))}
    tip = {"path": "old", "previous_attempts": [{"requests": 26}]}
    p = window.plan_for(parent, {}, tip, proof, authorization="bounded", now=datetime.now(UTC))
    assert p["max_new_requests"] == 11 and p["max_model_dispatches"] == 10
    assert p["max_cumulative_requests"] == 40 and p["original_planner_phase_max_seconds"] == 2700
    assert p["new_vision_requests"] == 0 and p["acquisition_targets"] == []
    tip["previous_attempts"][0]["requests"] = 31
    with pytest.raises(ValueError, match="scope_or_budget"):
        window.plan_for(parent, {}, tip, proof, authorization="bounded", now=datetime.now(UTC))


def test_original_cli_routes_only_explicit_root_flag(monkeypatch, capsys):
    import sys

    from apps.multimodal_case import main
    from packages.multimodal_research.campaign_renewal import prepare

    calls = []
    monkeypatch.setattr(window, "prepare", lambda **kw: calls.append(kw) or {"ok": True})
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "multimodal_case",
            "renewal-prepare",
            "--facts",
            "old",
            "--previous",
            "old",
            "--output",
            "new",
            "--mode",
            "review",
            "--authorization",
            "test",
            "--root-direction-review",
            "--local-finalization",
            "local",
        ],
    )
    assert main() == 0 and calls[0]["local_finalization"] == "local"
    assert "root_direction_review" not in calls[0] and '"ok": true' in capsys.readouterr().out
    with pytest.raises(ValueError, match="requires_explicit_mode"):
        prepare(
            facts="old",
            previous="old",
            output="new",
            mode="review",
            authorization="test",
            local_finalization="local",
        )


@pytest.mark.parametrize("field,value", [("phase", "running"), ("pending", True)])
def test_nonterminal_origin_cannot_authorize_new_root_review(field, value):
    phase = dict(target=None, run_id="RUN:OLD", accepted=True, acceptance_key="accepted")
    result = dict(phases=[phase])
    saved = {"dynamic-phase-result": (phase,), "multimodal-case-result": (result,)}
    docs = SimpleNamespace(run_id="RUN:OLD", read=saved.get)
    control = SimpleNamespace(phase="stopped", pending=None, cancelled=False)
    setattr(control, field, value)
    parent = SimpleNamespace(
        documents=lambda _: docs, sessions=SimpleNamespace(read=lambda _: control)
    )
    with pytest.raises(ValueError, match="terminal_origin"):
        window.origin_proof(parent, result, {"run_id": "RUN:OLD", "path": "old"})


def test_cancelled_origin_without_verified_closure_is_not_restarted():
    phase = dict(target=None, run_id="RUN:OLD", accepted=True, acceptance_key="accepted")
    result = dict(phases=[phase])
    saved = {"dynamic-phase-result": (phase,), "multimodal-case-result": (result,)}
    docs = SimpleNamespace(run_id="RUN:OLD", read=saved.get)
    control = SimpleNamespace(phase="stopped", pending=None, cancelled=True)
    parent = SimpleNamespace(
        documents=lambda _: docs, sessions=SimpleNamespace(read=lambda _: control)
    )
    with pytest.raises(ValueError, match="cancelled_without_accepted_closure"):
        window.origin_proof(parent, result, {"run_id": "RUN:OLD", "path": "old"})
