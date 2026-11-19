"""Explicit pending-only Runtime lease; historical flock semantics stay intact."""

import errno
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from packages.multimodal_research import execution_lease as lease
from packages.multimodal_research.campaign_renewal import RenewedCampaign


def campaign(tmp_path, version):
    value = object.__new__(RenewedCampaign)
    value.output = tmp_path
    value.base = SimpleNamespace()
    value.protocol = dict(schema_version=version, execution_lease_profile=lease.PROFILE)
    return value


def unavailable(*args):
    raise OSError(errno.ENOLCK, "synthetic filesystem lock unavailable")


@pytest.mark.parametrize("version", ["1.54.0", "1.55.0"])
def test_explicit_pending_runtime_session_does_not_use_unavailable_flock(
    tmp_path, monkeypatch, version
):
    value = campaign(tmp_path, "multimodal-campaign-renewal/" + version)
    events = []

    @contextmanager
    def original_runtime(current):
        assert current is value
        events.append("entered")
        try:
            yield
        finally:
            events.append("released")

    monkeypatch.setattr(lease, "session", original_runtime)
    monkeypatch.setattr(lease.fcntl, "flock", unavailable)
    with pytest.raises(RuntimeError, match="synthetic downstream failure"):
        with value.execution_session():
            raise RuntimeError("synthetic downstream failure")
    assert events == ["entered", "released"]
    assert not (tmp_path / "lock").exists()


def test_old_pending_version_does_not_silently_change_lease(tmp_path, monkeypatch):
    value = campaign(tmp_path, "multimodal-campaign-renewal/1.23.0")
    monkeypatch.setattr(lease.fcntl, "flock", unavailable)
    monkeypatch.setattr(lease, "session", lambda _: pytest.fail("implicit Runtime fallback"))
    with pytest.raises(OSError) as error:
        with value.execution_session():
            pytest.fail("lock omitted")
    assert error.value.errno == errno.ENOLCK


def window():
    from datetime import UTC, datetime, timedelta

    from packages.grounding.models import digest
    from tests.evaluation_preparation.test_directional_review import fixture

    now = datetime(2026, 10, 27, tzinfo=UTC)
    config = fixture().config
    config = config.model_copy(
        update={"transport": config.transport.model_copy(update={"max_retries": 0})}
    )
    parent = SimpleNamespace(
        config=config,
        base=SimpleNamespace(execution_profile="relay-root-question-v1"),
        protocol=dict(deadline=(now + timedelta(seconds=3600)).isoformat()),
    )
    plan = dict(
        schema_version="multimodal-campaign-renewal/1.23.0",
        mode="review",
        review_increment_only=True,
        pending_review=dict(
            selected_ids=["new:a"],
            accepted_ids=["old:a"],
            quarantined_batches=[
                dict(assessment_ids=["unknown:a"], new_reassessment_authorized=False)
            ],
        ),
        review_target_ids=["new:a"],
        whole_case_execution_profile=parent.base.execution_profile,
        original_window_deadline=parent.protocol["deadline"],
        provider=digest(config),
        started_at=now.isoformat(),
        deadline=(now + timedelta(seconds=2700)).isoformat(),
        seconds=2700,
        max_model_dispatches=1,
        max_new_requests=2,
        max_cumulative_requests=40,
        previous_attempts=[dict(requests=10)],
        new_vision_requests=0,
        new_public_search_allowed=False,
        acquisition_targets=[],
        expansion_policy=None,
    )
    return parent, plan


def test_runtime_window_preserves_deadline_provider_and_quarantine():
    from copy import deepcopy

    from packages.multimodal_research.pending_review import (
        declare_runtime_pending_window,
        validate_runtime_pending_window,
    )

    parent, plan = window()
    original = deepcopy(plan)
    declare_runtime_pending_window(plan, parent)
    validate_runtime_pending_window(plan, parent)
    for key in original.keys() - {"schema_version"}:
        assert plan[key] == original[key]


@pytest.mark.parametrize("version", ["1.54.0", "1.55.0"])
def test_runtime_review_uses_reserved_window_and_exact_dispatch_cap(monkeypatch, tmp_path, version):
    from datetime import datetime, timedelta

    from packages.llm_provider.provider import ProviderError
    from packages.multimodal_research.campaign import MultimodalCampaign

    value = campaign(tmp_path, "multimodal-campaign-renewal/" + version)
    _, plan = window()
    value.protocol.update(deadline=plan["deadline"], max_model_dispatches=1)
    original_deadline = value.work_deadline()
    with value.phase_context(None):
        assert value.work_deadline() == datetime.fromisoformat(plan["deadline"]) - timedelta(
            seconds=60
        )
    assert value.work_deadline() == original_deadline
    calls = []

    def factory(*args, **kwargs):
        assert kwargs["bounded_phase"] is True
        return SimpleNamespace(before_reserve=lambda v: calls.append(v))

    monkeypatch.setattr(MultimodalCampaign, "budget_factory", factory)
    value.usage = lambda: 0
    budget = value.budget_factory()
    budget.before_reserve("one")
    value.usage = lambda: 1
    with pytest.raises(ProviderError, match="budget"):
        budget.before_reserve("local-close-not-a-second-request")
    assert calls == ["one"]


@pytest.mark.parametrize(
    "change",
    [
        "accepted",
        "unknown",
        "ack",
        "authorized",
        "duplicate",
        "provider",
        "deadline",
        "seconds",
        "retries",
        "dispatch",
        "budget",
        "history",
        "cumulative",
        "search",
        "vision",
        "scope",
        "lease",
        "version",
        "window",
    ],
)
def test_runtime_window_rejects_unsafe_or_rebound_plan(change):
    from packages.multimodal_research.pending_review import (
        declare_runtime_pending_window,
        validate_runtime_pending_window,
    )

    parent, plan = window()
    declare_runtime_pending_window(plan, parent)
    p = plan["pending_review"]
    if change == "accepted":
        p["accepted_ids"].append("new:a")
    elif change == "unknown":
        p["quarantined_batches"][0]["assessment_ids"].append("new:a")
    elif change == "ack":
        p["acknowledge_unknown_request"] = "old-request"
    elif change == "authorized":
        p["quarantined_batches"][0]["new_reassessment_authorized"] = True
    elif change == "duplicate":
        p["selected_ids"].append("new:a")
    elif change == "provider":
        plan["provider"] = "wrong"
    elif change == "deadline":
        plan["deadline"] = "2026-10-28T00:00:00+00:00"
    elif change == "seconds":
        plan["seconds"] = 2701
    elif change == "retries":
        parent.config = parent.config.model_copy(
            update={"transport": parent.config.transport.model_copy(update={"max_retries": 1})}
        )
    elif change == "dispatch":
        plan["max_model_dispatches"] = 2
    elif change == "budget":
        plan["max_new_requests"] = 1
    elif change == "history":
        plan["previous_attempts"] = []
    elif change == "cumulative":
        plan["previous_attempts"] = [dict(requests=39)]
    elif change == "search":
        plan["new_public_search_allowed"] = True
    elif change == "vision":
        plan["new_vision_requests"] = 1
    elif change == "scope":
        plan["mode"] = "acquisition"
    elif change == "lease":
        plan["execution_lease_profile"] = "uncontrolled"
    elif change == "version":
        plan["schema_version"] = "multimodal-campaign-renewal/1.23.0"
    elif change == "window":
        plan["pending_new_window"] = True
    with pytest.raises(ValueError):
        validate_runtime_pending_window(plan, parent)
