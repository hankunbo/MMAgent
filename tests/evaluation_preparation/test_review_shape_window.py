"""Versioned shape instructions, actual wire/parser/domain, no real HTTP."""

import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from packages.grounding.models import digest
from packages.multimodal_research import review_shape_window as window
from packages.multimodal_research.campaign import configuration


def old_config():
    return configuration(
        directional=True,
        resilient_vision=True,
        exact_primary_status=True,
        execution_profile="relay-snapshot-context-v1",
    )


def test_new_prompt_changes_only_review_contract_not_provider_or_original_bytes():
    old = old_config()
    before = Path("configs/prompts/agents/review-direction-2.4.0.txt").read_bytes()
    new = window.configuration(old)
    a, b = old.model_dump(mode="json"), new.model_dump(mode="json")
    ar = next(r for r in a["roles"] if r["role"] == "review")
    br = next(r for r in b["roles"] if r["role"] == "review")
    assert br["prompt_version"] == window.PROMPT and br["prompt_checksum"] != ar["prompt_checksum"]
    br.update(prompt_version=ar["prompt_version"], prompt_checksum=ar["prompt_checksum"])
    assert (
        a == b and before == Path("configs/prompts/agents/review-direction-2.4.0.txt").read_bytes()
    )
    with pytest.raises(ValueError, match="original_configuration"):
        window.configuration(new)  # No recursive retry-until-desired-score prompt upgrade.


def test_uniform_window_budget_and_explicit_original_phase_cap():
    parent = SimpleNamespace(
        config=old_config(),
        protocol={"deadline": "2026-10-18T00:00:00+00:00"},
        base=SimpleNamespace(
            manifest=SimpleNamespace(checksum=digest("input")),
            task=SimpleNamespace(checksum=digest("task")),
        ),
        accepted_observations=lambda: {"a": "original observation"},
    )
    proof = {
        "model_review_ids": [str(i) for i in range(8)],
        "review_target_ids": [str(i) for i in range(8)],
    }
    tip = {"path": "original", "previous_attempts": [{"requests": 14}]}
    plan = window.plan_for(parent, {}, tip, proof, authorization="test", now=datetime.now(UTC))
    assert plan["max_new_requests"] == 10 and plan["max_model_dispatches"] == 9
    assert plan["seconds"] == 5400 and plan["original_planner_phase_max_seconds"] == 2700
    assert plan["new_vision_requests"] == 0 and not plan["new_public_search_allowed"]
    assert plan["provider"] == digest(window.configuration(parent.config))
    tip["previous_attempts"][0]["requests"] = 33
    with pytest.raises(ValueError, match="scope_or_budget"):
        window.plan_for(parent, {}, tip, proof, authorization="test", now=datetime.now(UTC))


def test_existing_cli_routes_explicit_shape_option_without_execution(monkeypatch, capsys):
    import sys

    from apps.multimodal_case import main

    calls = []

    def prepare(**kwargs):
        calls.append(kwargs)
        return {"test": "cli-delegation-only"}

    monkeypatch.setattr(window, "prepare", prepare)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "multimodal_case",
            "renewal-prepare",
            "--facts",
            "original",
            "--previous",
            "original",
            "--output",
            "new",
            "--mode",
            "review",
            "--authorization",
            "test",
            "--shape-contract-review",
        ],
    )
    assert main() == 0
    assert calls[0]["facts"] == calls[0]["previous"] == "original"
    assert calls[0]["mode"] == "review" and not calls[0]["same_profile_review"]
    assert "shape_contract_review" not in calls[0]
    assert "cli-delegation-only" in capsys.readouterr().out


@pytest.mark.parametrize("estimate", [0.03, -0.03, 0.0])
def test_shape_prompt_actual_wire_http_parser_acceptance_preserves_estimate(estimate):
    from agents.expert_agent.reporting import read_research_report, render_research_html
    from packages.llm_provider.provider import StructuredProvider
    from packages.llm_provider.transport import HTTPTransport, parse_reply
    from tests.agent_provider.batch_support import generate
    from tests.evaluation_preparation.test_directional_review import explanation, fixture, reply

    f = fixture()
    config = window.configuration(old_config()).model_copy(update={"mode": "fake"})
    f.config = config
    f.owner.state = f.owner.state.model_copy(
        update={
            "config_checksum": digest(config),
            "prompt_checksum": next(r.prompt_checksum for r in config.roles if r.role == "review"),
        }
    )

    class InjectedHTTP(HTTPTransport):
        simulated = True
        calls = []

        def send(self, config, body, **kwargs):
            self.calls.append(body)
            assert "review-direction-shape/1.0.0" in body["messages"][0]["content"]
            assert "allowed_primary_statuses" in json.dumps(body)
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
    assert any(i.judgment.value == estimate for i in sub.candidate.items)
    assert f.acceptance.submit(sub, key="shape-contract")["status"] == "submitted_not_accepted"
    assert f.acceptance.accept(key="shape-contract").status == "accepted"
    detail = read_research_report(f.acceptance, key="shape-contract", parent=f.owner.current())
    assert render_research_html(detail) and detail["scores"]["direction_assessments"]
    count = len(transport.calls)
    assert generate(f) == sub and len(transport.calls) == count
    assert all(r.api_requests == 0 and r.simulated for r in f.budget.records())
