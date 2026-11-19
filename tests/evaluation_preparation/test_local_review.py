"""Original failed Review ownership and explicit successor boundaries."""

import copy
from datetime import UTC, datetime

import pytest

from packages.evidence_details.pending_review import read_local_unsent_review
from packages.grounding.models import digest
from packages.llm_provider.transport import Reply
from packages.multimodal_research import local_review_recovery as r
from packages.multimodal_research import unstarted_window as w
from tests.evaluation_preparation.test_unstarted_context_window import old_plan
from tests.expert_agent.test_pending_failed_attempt import failed


def local_failure(monkeypatch, tmp_path, **kw):
    return failed(
        monkeypatch,
        tmp_path,
        replies=[Reply(error="configuration", api_requests=0, response_confirmed=False)],
        **kw,
    )


def test_original_owner_proves_full_zero_dispatch_partition(monkeypatch, tmp_path):
    f, state = local_failure(monkeypatch, tmp_path)
    before = digest(f.documents.saved)
    proof = read_local_unsent_review(documents=f.documents, state=state, config=f.config)
    assert len(proof["selected_ids"]) == 7 and proof["charged_attempts"] == 1
    assert proof["api_requests"] == proof["unknown_requests"] == 0
    assert proof["inventory"]["quarantined_batches"]
    assert before == digest(f.documents.saved)


@pytest.mark.parametrize(
    "reply",
    [
        Reply(error="configuration", api_requests=1, response_confirmed=False),
        Reply(error="timeout", api_requests=None, response_confirmed=None),
        Reply(error="transport", api_requests=0, response_confirmed=False),
        Reply(error="configuration", api_requests=0, response_confirmed=True),
    ],
)
def test_ambiguous_or_sent_failure_not_eligible(monkeypatch, tmp_path, reply):
    f, state = failed(monkeypatch, tmp_path, replies=[reply])
    with pytest.raises(ValueError, match="local_review_"):
        read_local_unsent_review(documents=f.documents, state=state, config=f.config)


@pytest.mark.parametrize("change", ["claim", "parent", "cancel", "reservation"])
def test_original_owner_denials_kept(monkeypatch, tmp_path, change):
    f, state = local_failure(monkeypatch, tmp_path)
    if change == "claim":
        k = next(
            k for k in f.documents.saved if k.startswith("review-batches:") and k.endswith(":claim")
        )
        f.documents.saved[k][0]["input_checksum"] = digest("different")
    elif change == "parent":
        state = state.model_copy(update={"config_checksum": digest("different")})
    elif change == "cancel":
        control = f.planner.read(f.binding.run_id)
        monkeypatch.setattr(
            f.documents.sessions, "read", lambda _: control.model_copy(update={"cancelled": True})
        )
    else:
        f.ports.budget.journal.append(dict(phase="reserved", request_id="request:unclosed"))
    with pytest.raises(ValueError, match="pending_attempt"):
        read_local_unsent_review(documents=f.documents, state=state, config=f.config)


def prepared(tmp_path):
    now, original = old_plan(tmp_path / "first")
    original["previous_attempts"] = [
        dict(directory=str(tmp_path / "root"), requests=26, checksum=digest("root"))
    ]
    old = w.review_plan_for(
        original,
        previous=tmp_path / "first",
        output=tmp_path / "failed",
        authorization="test",
        started=now,
        implementation={},
    )
    proof = dict(
        proof=dict(charged_attempts=1),
        result_checksum=digest("failed"),
        phase_checksum=digest("phase"),
    )
    return old, proof


def make(old, proof, tmp_path):
    return r.plan_for(
        old,
        proof,
        previous=old["output"],
        output=tmp_path / "new",
        authorization="bounded",
        started=datetime.now(UTC),
        implementation={},
    )


def test_new_identity_preserves_old_bytes_and_charges_failure(tmp_path):
    old, proof = prepared(tmp_path)
    before = copy.deepcopy(old)
    plan = make(old, proof, tmp_path)
    assert old == before and plan["schema_version"] == r.VERSION
    assert sum(p["requests"] for p in plan["previous_attempts"]) == 27
    for field in (
        "provider",
        "review_target_ids",
        "material_context_review",
        "max_new_requests",
        "max_model_dispatches",
        "seconds",
        "metadata_profile",
        "tip",
    ):
        assert plan[field] == old[field]
    assert "unstarted_window" not in plan
    assert plan["local_review_recovery"]["previous_protocol"] == digest(old)
    with pytest.raises(ValueError, match="local_review_successor_scope"):
        make(plan, proof, tmp_path)


@pytest.mark.parametrize(
    "change",
    [
        {"max_cumulative_requests": 999},
        {"seconds": 999999},
        {"mode": "acquisition"},
        {"acquisition_targets": ["extra"]},
        {"new_vision_requests": 1},
        {"reuse_assessment_ids": ["old"]},
        {"new_public_search_allowed": True},
    ],
)
def test_scope_not_broadened(tmp_path, change):
    old, proof = prepared(tmp_path)
    old.update(change)
    with pytest.raises(ValueError, match="scope"):
        make(old, proof, tmp_path)


def test_actual_failed_attempt_keeps_cumulative_cap(tmp_path):
    old, proof = prepared(tmp_path)
    old["previous_attempts"][0]["requests"] = 39
    with pytest.raises(ValueError, match="cumulative_budget"):
        make(old, proof, tmp_path)


def test_new_version_uses_original_runtime_lease_and_bounded_review_budget(monkeypatch):
    from datetime import timedelta
    from types import SimpleNamespace

    from packages.llm_provider.provider import ProviderError
    from packages.multimodal_research import campaign, campaign_renewal, execution_lease

    c = object.__new__(campaign_renewal.RenewedCampaign)
    deadline = datetime.now(UTC) + timedelta(hours=1)
    c.protocol = dict(
        schema_version=r.VERSION,
        deadline=deadline.isoformat(),
        max_model_dispatches=2,
        execution_lease_profile=execution_lease.PROFILE,
    )
    c.base = SimpleNamespace(detailed=True)
    lease = object()
    monkeypatch.setattr(execution_lease, "session", lambda obj: lease if obj is c else None)
    assert c.execution_session() is lease
    before = c.work_deadline()
    with c.phase_context(None):
        assert c.work_deadline() == deadline - timedelta(seconds=60)
    assert c.work_deadline() == before
    calls = []

    def factory(*a, **kw):
        assert kw["bounded_phase"] is True
        return SimpleNamespace(before_reserve=lambda value: calls.append(value))

    monkeypatch.setattr(campaign.MultimodalCampaign, "budget_factory", factory)
    c.usage = lambda: 1
    budget = c.budget_factory()
    budget.before_reserve("second")
    c.usage = lambda: 2
    with pytest.raises(ProviderError, match="budget"):
        budget.before_reserve("over-budget")
    assert calls == ["second"]


def test_cli_selects_original_recovery_owner(monkeypatch, capsys):
    import sys

    from apps import multimodal_case

    selected = []
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "multimodal_case",
            "renewal-local-review-run",
            "--previous",
            "old",
            "--output",
            "new",
            "--authorization",
            "bounded",
        ],
    )
    monkeypatch.setattr(
        r, "prepare_and_run", lambda **kw: selected.append(kw) or dict(complete=False)
    )
    assert multimodal_case.main() == 2
    assert selected == [dict(previous="old", output="new", authorization="bounded")]
    assert '"complete": false' in capsys.readouterr().out
