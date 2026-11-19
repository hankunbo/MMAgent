"""Versioned exact truncated-batch recovery, original campaign dispatch bounds."""

import subprocess
import sys
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from packages.multimodal_research import truncated_review as recovery
from tests.evaluation_preparation.test_http_client_entry import public_config


def test_cli_exposes_exact_request_acknowledgement():
    result = subprocess.run(
        [sys.executable, "-m", "apps.multimodal_case", "renewal-prepare", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--acknowledge-truncated-request" in result.stdout


def test_configuration_only_changes_partition():
    config = public_config()
    old = config.model_dump(mode="json")
    new = recovery.configuration(config)
    expected = config.model_dump(mode="json")
    expected["review_batching"]["batch_size"] = 1
    assert new.model_dump(mode="json") == expected
    assert config.model_dump(mode="json") == old
    with pytest.raises(ValueError, match="truncated_review_original_configuration"):
        recovery.configuration(new)


def test_plan_keeps_cumulative_accounting_and_no_observation_dispatch(tmp_path):
    now = datetime.now(UTC)
    parent = SimpleNamespace(
        config=public_config(),
        protocol=dict(deadline=(now - timedelta(seconds=10)).isoformat()),
        base=SimpleNamespace(
            manifest=SimpleNamespace(checksum="input"), task=SimpleNamespace(checksum="task")
        ),
        accepted_observations=lambda: {},
    )
    tip = dict(path=str(tmp_path), previous_attempts=[dict(requests=12)])
    proof = dict(review_target_ids=["a", "b", "c"])
    plan = recovery.plan_for(
        parent,
        {},
        tip,
        proof,
        output=tmp_path / "next",
        authorization="existing bounded recovery",
        started=now,
    )
    assert plan["prior_deadline"] == parent.protocol["deadline"]
    assert plan["max_model_dispatches"] == 3 and plan["max_new_requests"] == 4
    assert (
        plan["max_cumulative_requests"] == 40
        and plan["previous_attempts"] == tip["previous_attempts"]
    )
    assert plan["new_vision_requests"] == 0 and plan["acquisition_targets"] == []
    assert not plan["new_public_search_allowed"]
    tip["previous_attempts"][0]["requests"] = 37
    with pytest.raises(ValueError, match="truncated_review_authorization_or_budget"):
        recovery.plan_for(
            parent, {}, tip, proof, output=tmp_path / "next", authorization="bounded", started=now
        )


def test_constructor_uses_original_runner_without_mutating_parent(tmp_path, monkeypatch):
    from packages.multimodal_research.campaign import MultimodalCampaign
    from packages.multimodal_research.campaign_renewal import RenewedCampaign

    config = public_config()
    parent = SimpleNamespace(
        base=SimpleNamespace(review_target_ids=("old",)), target=None, config=config
    )
    plan = dict(
        schema_version=recovery.VERSION,
        mode="review",
        truncated_review={"proof": True},
        review_target_ids=["pending"],
    )
    monkeypatch.setattr(recovery, "check", lambda p, v: recovery.configuration(p.config))
    monkeypatch.setattr(
        MultimodalCampaign, "__init__", lambda self, base, *_a, **_k: setattr(self, "base", base)
    )
    value = RenewedCampaign(parent, tmp_path, plan)
    assert value.base is not parent.base and parent.base.review_target_ids == ("old",)
    assert value.base.review_target_ids == ("pending",)
    assert value.base.confirmed_review_reader is None
    assert value.config == recovery.configuration(config)
    assert value.targets == value.initial_targets == ()


@pytest.mark.parametrize("fault", ["cancel", "deadline", "dispatch", "source"])
def test_original_guard_still_limits_dispatch_and_preserves_local_finish(
    tmp_path, monkeypatch, fault
):
    from packages.llm_provider.provider import ProviderError
    from packages.multimodal_research.campaign import MultimodalCampaign
    from tests.evaluation_preparation.test_exact_acquisition_budget import campaign

    value, _ = campaign(tmp_path, version="1.53.0", used=3, previous=12)
    value.protocol.update(mode="review", max_new_requests=4, facts=str(tmp_path / "parent"))
    value._final_review_active = True
    if fault == "cancel":
        (tmp_path / "cancel").touch()
        with pytest.raises(ProviderError, match="cancelled"):
            value.guard()
    elif fault == "deadline":
        value.protocol["deadline"] = (datetime.now(UTC) - timedelta(seconds=1)).isoformat()
        with pytest.raises(ProviderError, match="deadline"):
            value.guard()
    elif fault == "source":

        def withdrawn():
            raise ValueError("source_revoked")

        value.base.check = withdrawn
        with pytest.raises(ValueError, match="source_revoked"):
            value.guard()
    else:
        value.guard()
        monkeypatch.setattr(
            MultimodalCampaign,
            "budget_factory",
            lambda *_a, **_k: SimpleNamespace(before_reserve=lambda _: None),
        )
        with pytest.raises(ProviderError, match="budget"):
            value.budget_factory().before_reserve("fourth")


def test_new_report_marks_increment_until_owner_aggregation():
    from tests.evaluation_preparation.test_aggregation_report_scope import apply_scope

    protocol = dict(
        schema_version=recovery.VERSION,
        tip={"run_id": "prior"},
        truncated_review=dict(
            request_id="failed", inventory={"accepted_ids": ["old"]}, review_target_ids=["new"]
        ),
    )
    report = apply_scope(protocol, ["new"], {})
    assert not report["completion"]["all_obtained_reviewed"]
    assert report["review_scope"]["original_failed_request"] == "failed"
    report = apply_scope(
        protocol,
        None,
        dict(execution_kind="local_accepted_review_aggregation_not_new_model_review", origins=[]),
    )
    assert report["completion"]["all_obtained_reviewed"]


def test_aggregation_version_dispatch_not_silent_upgrade():
    from packages.multimodal_research.accepted_aggregation import (
        ReasonedBatchAggregationPlan,
        full_scope,
        read_plan,
    )

    raw = dict(
        schema_version="accepted-review-aggregation-plan/1.4.0",
        facts="next",
        origins=[dict(campaign="old"), dict(campaign="next")],
    )
    plan = read_plan(raw)
    assert isinstance(plan, ReasonedBatchAggregationPlan)
    campaign = SimpleNamespace(protocol=dict(schema_version="wrong"))
    with pytest.raises(ValueError, match="accepted_aggregation_reasoned_partition_origin"):
        full_scope(campaign, plan)
    assert not isinstance(
        read_plan(dict(raw, schema_version="accepted-review-aggregation-plan/1.3.0")),
        ReasonedBatchAggregationPlan,
    )


@pytest.mark.parametrize("fault", [None, "wrong_request", "changed_state", "cancelled"])
def test_current_owner_partial_selection_is_exact_and_read_only(monkeypatch, tmp_path, fault):
    from packages.grounding.models import digest
    from packages.multimodal_research.campaign import Phase
    from tests.expert_agent.test_reasoned_batch_aggregation import partial

    f, sub = partial(monkeypatch)
    # This fixture supplies synthetic immutable projection, while the real
    # acceptance/request/partial-inventory owners remain installed below.
    monkeypatch.setattr(recovery, "configuration", lambda origin: origin)
    monkeypatch.setattr(
        "packages.evidence_details.pending_review.RuntimeAgentJournal", lambda _: f.budget.journal
    )
    old = f.owner.current()
    request = f.budget.records()[-1]
    slot = "review-batches:synthetic-batches:batch:001:attempt:" + request.request_id
    from packages.llm_provider.models import Usage

    cap = next(r.max_output_tokens for r in f.config.roles if r.role == "review")
    request = request.model_copy(update=dict(usage=Usage(status="observed", output_tokens=cap)))
    for event in f.budget.journal.events:
        if (
            event.get("phase") == "completed"
            and event["record"]["request_id"] == request.request_id
        ):
            event["record"] = request.model_dump(mode="json")
    f.documents.saved[slot][0]["record_checksum"] = digest(request)
    f.budget.journal.append(
        dict(phase="response_confirmation", request_id=request.request_id, confirmed=True)
    )
    phase = dict(target=None, run_id=f.documents.run_id, accepted=True, acceptance_key="partial")
    result = dict(complete=True, phases=[phase])
    f.documents.write("dynamic-phase-result", (phase,))
    f.documents.write("multimodal-case-result", (result,))
    f.documents.write("dynamic-phase-input", (old.model_dump(mode="json"),))
    parent = SimpleNamespace(
        config=f.config,
        protocol={},
        documents=lambda _: f.documents,
        sessions=f.documents.sessions,
        contexts=SimpleNamespace(read_research_run=lambda _: SimpleNamespace()),
        base=SimpleNamespace(check=lambda: None, contexts={}),
        accepted_observations=lambda: {},
    )
    monkeypatch.setattr(
        Phase,
        "project",
        lambda *_a, **_k: (
            old
            if fault != "changed_state"
            else old.model_copy(update=dict(config_checksum=digest("changed"))),
            None,
        ),
    )
    monkeypatch.setattr(Phase, "verify", lambda _self, state: None)
    if fault == "cancelled":
        (tmp_path / "cancel").touch()
    before = digest(f.documents.saved), digest(f.budget.journal.read())
    tip = dict(run_id=f.documents.run_id, path=str(tmp_path))
    if fault:
        expected = {
            "wrong_request": "exact_request",
            "changed_state": "current_material",
            "cancelled": "terminal_origin",
        }[fault]
        with pytest.raises(ValueError, match=expected):
            recovery.origin_proof(
                parent, result, tip, "wrong" if fault == "wrong_request" else request.request_id
            )
    else:
        proof = recovery.origin_proof(parent, result, tip, request.request_id)
        assert proof["review_target_ids"] == list(sub.plan.batches[-1].assessment_ids)
        assert not set(proof["review_target_ids"]) & set(proof["accepted_ids"])
    assert before == (digest(f.documents.saved), digest(f.budget.journal.read()))
