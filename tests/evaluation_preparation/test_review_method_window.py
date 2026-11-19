"""Different text method: explicit bounds, no old answer reuse or sign mandate."""

import json
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from packages.grounding.models import digest
from packages.multimodal_research import review_method_window as window
from packages.multimodal_research import root_review_window as root
from packages.multimodal_research.campaign import MultimodalCampaign, configuration
from packages.multimodal_research.campaign_renewal import RenewedCampaign


def old_config():
    return configuration(
        resilient_vision=True,
        directional=True,
        exact_primary_status=True,
        execution_profile="relay-root-question-v1",
        review_batch_size=3,
    )


def fixture_plan(tmp_path=None):
    parent = SimpleNamespace(
        config=old_config(),
        protocol=dict(deadline="2026-10-18T00:00:00+00:00", max_cumulative_requests=40),
        base=SimpleNamespace(
            manifest=SimpleNamespace(checksum=digest("input")),
            task=SimpleNamespace(checksum=digest("task")),
        ),
        accepted_observations=lambda: {"a": "retained"},
    )
    proof = dict(
        model_review_ids=[str(i) for i in range(13)],
        review_target_ids=[str(i) for i in range(13)],
        old_answers_reused=False,
    )
    tip = dict(path="old", previous_attempts=[dict(requests=40)])
    result = dict(complete=True)
    paths = []
    if tmp_path:
        path = tmp_path / "diagnostic.json"
        path.write_text(
            json.dumps(
                dict(
                    schema_version="retained-direction-diagnostic-result/1.0.0",
                    old_result_checksum=digest(result),
                    response_confirmed=True,
                    api_requests=1,
                    database_writes=0,
                    new_acceptance=False,
                    images_uploaded=0,
                )
            )
        )
        paths.append(str(path))
    args = dict(
        authorization="autonomous bounded different-method test",
        profile=window.PROFILE,
        now=datetime.now(UTC),
        diagnostic_paths=paths,
    )
    plan = window.plan_for(parent, result, tip, proof, **args)
    return parent, result, proof, tip, args, plan


def test_configuration_changes_only_text_method_transport_and_batching():
    old = old_config()
    before = old.model_dump_json()
    new = window.configuration(old, window.PROFILE)
    a, b = old.model_dump(mode="json"), new.model_dump(mode="json")
    assert a["roles"] == b["roles"]
    assert a["transport"]["model"] != b["transport"]["model"]
    assert b["transport"]["timeout_seconds"] == 180 and b["transport"]["max_retries"] == 0
    b["transport"].update(
        model=a["transport"]["model"], timeout_seconds=a["transport"]["timeout_seconds"]
    )
    assert a == b and old.model_dump_json() == before
    with pytest.raises(ValueError, match="original_configuration"):
        window.configuration(new, window.PROFILE)
    with pytest.raises(ValueError, match="profile"):
        window.configuration(old, "arbitrary")


def test_explicit_new_window_preserves_old_budget_and_diagnostic_accounting(tmp_path):
    parent, result, proof, tip, args, plan = fixture_plan(tmp_path)
    assert plan["max_new_requests"] == 7 and plan["max_model_dispatches"] == 6
    assert plan["max_cumulative_requests"] == 47
    assert plan["text_review_method"]["case_related_request_ceiling"] == 48
    assert plan["prior_deadline"] == parent.protocol["deadline"]
    assert parent.protocol["max_cumulative_requests"] == 40
    assert plan["new_vision_requests"] == 0 and plan["new_public_search_allowed"] is False
    assert window.cumulative_limit(plan) == 47
    assert plan["text_review_method"]["old_scores_reused"] is False
    for change in (
        {"max_cumulative_requests": 99},
        {"mode": "acquisition"},
        {"acquisition_targets": ["x"]},
        {"max_new_requests": 20},
    ):
        with pytest.raises(ValueError, match="budget_binding"):
            window.cumulative_limit({**plan, **change})
    parent.protocol["schema_version"] = window.VERSION
    with pytest.raises(ValueError, match="already_used"):
        window.plan_for(parent, result, tip, proof, **args)


def test_wrong_diagnostic_parent_and_unknown_rejected(tmp_path):
    _, result, _, _, args, _ = fixture_plan(tmp_path)
    path = args["diagnostic_paths"][0]
    with pytest.raises(ValueError, match="binding"):
        window.diagnostics([path], {**result, "other": True})
    with pytest.raises(ValueError, match="scope"):
        window.diagnostics([path, path], result)
    from pathlib import Path

    row = json.loads(Path(path).read_text())
    row["response_confirmed"] = False
    Path(path).write_text(json.dumps(row))
    with pytest.raises(ValueError, match="binding"):
        window.diagnostics([path], result)


def test_current_committed_owner_and_ancestor_are_required(monkeypatch):
    from agents.expert_agent import committed_review
    from packages.evidence_details import review_batches

    ancestor = object()
    parent = SimpleNamespace(
        config=old_config(), base=SimpleNamespace(confirmed_review_reader=ancestor)
    )
    state = object()
    docs = SimpleNamespace(read=lambda key: ({"submission": "owner-checked-submission"},))
    seen = []

    def read(documents, **kwargs):
        seen.append((documents, kwargs))
        return {"acceptance_key": "exact-key", "report_checksum": digest("report")}

    monkeypatch.setattr(committed_review, "read_committed_review", read)
    monkeypatch.setattr(
        review_batches,
        "read_submission",
        lambda value: SimpleNamespace(
            candidate=SimpleNamespace(
                items=[SimpleNamespace(assessment_id="b"), SimpleNamespace(assessment_id="a")]
            )
        ),
    )
    proof, ids = window.accepted_origin(parent, docs, state, "exact-key")
    assert ids == ["a", "b"] and proof["acceptance_key"] == "exact-key"
    assert seen == [
        (docs, dict(parent=state, config=parent.config, confirmed_review_reader=ancestor))
    ]
    with pytest.raises(ValueError, match="committed_origin_binding"):
        window.accepted_origin(parent, docs, state, "other-key")
    monkeypatch.setattr(committed_review, "read_committed_review", lambda *a, **kw: None)
    with pytest.raises(ValueError, match="committed_origin_binding"):
        window.accepted_origin(parent, docs, state, "exact-key")


def test_check_reconstructs_proof_and_rejects_plan_mutation(monkeypatch):
    parent, result, proof, _, _, plan = fixture_plan()
    monkeypatch.setattr(root, "origin_proof", lambda *a, **kw: proof)
    assert root.check(plan, parent, result) is None
    for changes in (
        {"review_target_ids": ["only_good_answer"]},
        {"max_new_requests": 30},
        {"deadline": "2099-01-01T00:00:00+00:00"},
    ):
        with pytest.raises(ValueError, match="binding"):
            root.check({**plan, **changes}, parent, result)
    assert root.configuration_for(parent.config, plan) == window.configuration(
        parent.config, window.PROFILE
    )


def test_legacy_cap_stays_40_and_new_cap_is_narrow():
    _, _, _, _, _, plan = fixture_plan()
    old = object.__new__(MultimodalCampaign)
    old.protocol = dict(max_new_requests=99, max_cumulative_requests=99)
    assert old.cumulative_request_limit() == 40
    renewed = object.__new__(RenewedCampaign)
    renewed.protocol = plan
    assert renewed.cumulative_request_limit() == 47
    renewed.protocol = {**plan, "schema_version": "multimodal-campaign-renewal/1.34.0"}
    assert renewed.cumulative_request_limit() == 40


def test_original_cli_routes_explicit_method_and_preserves_conflict_fields(monkeypatch, capsys):
    import sys

    from apps.multimodal_case import main
    from packages.multimodal_research.campaign_renewal import prepare

    calls = []
    monkeypatch.setattr(window, "prepare", lambda **kw: calls.append(kw) or {"ok": True})
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "case",
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
            "--review-method-profile",
            window.PROFILE,
        ],
    )
    assert main() == 0
    assert calls[0]["review_method_profile"] == window.PROFILE
    assert calls[0]["method_diagnostic"] == []
    capsys.readouterr()
    with pytest.raises(ValueError, match="requires_explicit_method"):
        prepare(
            facts="old",
            previous="old",
            output="new",
            mode="review",
            authorization="test",
            method_diagnostic=["not-authority"],
        )


@pytest.mark.parametrize("estimate", [-0.03, 0.0, 0.03])
def test_new_method_actual_provider_domain_score_and_readback(monkeypatch, estimate):
    from agents.expert_agent.reporting import read_research_report
    from packages.llm_provider import DeterministicFakeProvider
    from tests.agent_provider.batch_support import generate
    from tests.evaluation_preparation.test_directional_review import explanation, fixture, reply

    f = fixture()
    config = window.configuration(old_config(), window.PROFILE).model_copy(update={"mode": "fake"})
    old = f.owner.current()
    f.owner.state = old.model_copy(
        update={
            "config_checksum": digest(config),
            "prompt_checksum": next(r.prompt_checksum for r in config.roles if r.role == "review"),
        }
    )

    def respond(body):
        assert body["model"] == config.transport.model
        raw = reply(body)
        for row in raw["items"]:
            if row["status"] == "assessed":
                row["judgment"].update(value=estimate, lower=-0.2, upper=0.2)
                row["direction_assessment"].update(explanation(estimate))
        return raw

    f.config = config
    f.agent.provider = DeterministicFakeProvider(config, [respond] * 10)
    sub = generate(f)
    f.acceptance.submit(sub, key="new-method")
    assert f.acceptance.accept(key="new-method").status == "accepted"
    report = read_research_report(f.acceptance, key="new-method", parent=f.owner.current())
    assert report["scores"]
    calls = f.budget.used
    assert generate(f) == sub and f.budget.used == calls
    assert any(i.judgment.value == estimate for i in sub.candidate.items)
