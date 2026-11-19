"""Narrow stage recovery fences; real revision/Provider tests are kept separately."""

from types import SimpleNamespace

import pytest

from packages.multimodal_research.stages import (
    definition_repair_delta,
    definition_repair_eligible,
)


def test_only_projection_and_existing_stage_adapter_can_change():
    path = "packages/multimodal_research/definition_expansion.py"
    assert definition_repair_delta({path: "old"}, {path: "new"}) == [path]
    for extra in ("configs/prompts/agents/review.txt", "db/knowledge/migrations/new.sql"):
        with pytest.raises(ValueError, match="change_scope"):
            definition_repair_delta({path: "old"}, {path: "new", extra: "new"})
    with pytest.raises(ValueError, match="change_scope"):
        definition_repair_delta({path: "old"}, {path: "old"})


@pytest.mark.parametrize(
    "change", [None, "unknown", "dispatch", "new_fact", "no_revision", "scheduled", "mode"]
)
def test_recovery_only_saved_interpretation_with_unstarted_observation_work(change):
    campaign = SimpleNamespace(
        protocol={"mode": "definitions", "definition_increment_only": True}, targets=()
    )
    failure = dict(
        stage="definitions",
        error_code="local_owner_error",
        definition_revision_saved=True,
        transport=dict(remote_response_unknown=0, dispatch_unknown=0),
    )
    revision = dict(added=[], replaced=[{"id": "original-check"}])
    if change == "unknown":
        failure["transport"]["remote_response_unknown"] = 1
    elif change == "dispatch":
        failure["transport"]["dispatch_unknown"] = 1
    elif change == "new_fact":
        revision["added"] = ["new-check"]
    elif change == "no_revision":
        failure["definition_revision_saved"] = False
    elif change == "scheduled":
        campaign.targets = ("new-check",)
    elif change == "mode":
        campaign.protocol["mode"] = "acquisition"
    if change is None:
        definition_repair_eligible(campaign, failure, revision)
    else:
        with pytest.raises(ValueError, match="repair_scope"):
            definition_repair_eligible(campaign, failure, revision)


@pytest.mark.parametrize(
    "fault", [None, "implicit", "budget", "many", "unrelated", "legacy", "unknown"]
)
def test_explicit_single_acquisition_repair_preserves_failed_source_and_budget(fault):
    from packages.multimodal_research.confirmed_definition_failure import VERSION

    c = SimpleNamespace(
        protocol=dict(
            schema_version=VERSION,
            mode="definitions",
            definition_increment_only=True,
            max_new_requests=4,
        ),
        targets=("check:a",),
        usage=lambda: 2,
    )
    f = dict(
        stage="definitions",
        error_code="local_owner_error",
        definition_revision_saved=True,
        transport=dict(remote_response_unknown=0, dispatch_unknown=0),
    )
    r = dict(added=[], replaced=[dict(id="check:a")])
    if fault == "budget":
        c.usage = lambda: 3
    elif fault == "many":
        c.targets = ("check:a", "check:b")
    elif fault == "unrelated":
        r["replaced"] = [dict(id="check:other")]
    elif fault == "legacy":
        c.protocol["schema_version"] = "multimodal-campaign-renewal/1.56.0"
    elif fault == "unknown":
        f["transport"]["remote_response_unknown"] = 1
    if fault is None:
        definition_repair_eligible(c, f, r, include_acquisition=True)
    else:
        with pytest.raises(ValueError, match="repair_scope"):
            definition_repair_eligible(c, f, r, include_acquisition=fault != "implicit")


def test_exact_cap_local_finalize_never_authorizes_one_more_request(monkeypatch):
    from packages.llm_provider.provider import ProviderError
    from packages.multimodal_research.campaign import MultimodalCampaign
    from packages.multimodal_research.campaign_renewal import RenewedCampaign
    from packages.multimodal_research.confirmed_definition_failure import VERSION

    calls = []
    budget = SimpleNamespace(before_reserve=lambda checksum: calls.append(checksum))
    monkeypatch.setattr(MultimodalCampaign, "budget_factory", lambda *args, **kwargs: budget)
    c = object.__new__(RenewedCampaign)
    c.protocol = dict(schema_version=VERSION, max_new_requests=4)
    c.usage = lambda: 3
    current = c.budget_factory()
    current.before_reserve("last-allowed")
    assert calls == ["last-allowed"]
    c.usage = lambda: 4
    assert c.request_finalization_headroom == 0
    with pytest.raises(ProviderError, match="budget"):
        current.before_reserve("forbidden-fifth")
    assert calls == ["last-allowed"]


def test_acquisition_repair_uses_original_runtime_lease_not_filesystem_lock(monkeypatch):
    from contextlib import contextmanager

    from packages.multimodal_research import stages
    from packages.multimodal_research.confirmed_definition_failure import VERSION

    calls = []

    @contextmanager
    def original_session():
        calls.append("acquired")
        try:
            yield
        finally:
            calls.append("released")

    def no_flock(*args):
        raise OSError(37, "No locks available")

    monkeypatch.setattr(stages.fcntl, "flock", no_flock)
    c = SimpleNamespace(protocol={"schema_version": VERSION}, execution_session=original_session)
    with pytest.raises(RuntimeError, match="cleanup"):
        with stages.definition_repair_session(c, include_acquisition=True):
            assert calls == ["acquired"]
            raise RuntimeError("cleanup")
    assert calls == ["acquired", "released"]
    c.protocol["schema_version"] = "multimodal-campaign-renewal/1.56.0"
    with pytest.raises(ValueError, match="lease_version"):
        with stages.definition_repair_session(c, include_acquisition=True):
            pytest.fail("wrong version acquired a lease")
