"""Explicit new profiles reserve actual review work, never old unknown slots."""

from types import SimpleNamespace

from packages.multimodal_research import active_continuation as budget
from packages.multimodal_research.active_requirements import load


def test_new_harvard_profile_routes_raw_definitions_and_budget():
    old = load("configs/research/multimodal_harvard_visual_priority_v1.json")
    new = load("configs/research/multimodal_harvard_visual_priority_v2.json")
    assert old["definition_expansion"]["prompt_version"] == "definition-draft-6.1.0"
    assert new["definition_expansion"]["prompt_version"] == "definition-draft-6.2.0"
    assert new["definition_expansion"]["schema_version"].endswith("/1.26.0")
    assert new["checks"] == old["checks"]
    assert new["review_reserve_policy"]["schema_version"] == "review-work-reserve/1.0.0"


def test_new_reserve_uses_obtained_ids_and_actual_transport():
    rows = {
        str(n): SimpleNamespace(status=s)
        for n, s in enumerate(["obtained"] * 6 + ["unknown", "missing", "ungradable"])
    }
    ids = budget.reviewable_ids(rows, excluded={"0"})
    assert ids == ["1", "2", "3", "4", "5"]
    p = dict(
        schema_version="review-work-reserve/1.0.0", local_seconds_per_batch=60, close_seconds=300
    )
    result = budget.reserve_current_review(
        seconds_left=4103,
        assessment_ids=ids,
        batch_size=1,
        timeout_seconds=300,
        max_retries=0,
        requests_used=6,
        request_limit=40,
        policy=p,
    )
    assert result["stop_acquisition"] is False
    assert result["review_reserve_seconds"] == 2460
    assert result["review_reserve_requests"] == 7
    assert result["pending_assessment_ids"] == ids


def test_new_reserve_stops_on_time_or_requests():
    p = dict(
        schema_version="review-work-reserve/1.0.0", local_seconds_per_batch=60, close_seconds=300
    )
    for seconds, used in [(300, 0), (5000, 38)]:
        assert budget.reserve_current_review(
            seconds_left=seconds,
            assessment_ids=["a"],
            batch_size=1,
            timeout_seconds=300,
            max_retries=0,
            requests_used=used,
            request_limit=40,
            policy=p,
        )["stop_acquisition"]


def test_old_profile_bytes_are_not_rewritten():
    assert budget.reserve_review(
        seconds_left=3803.685695, acquired=7, batch_size=1, requests_used=6
    )["stop_acquisition"]


def test_new_profile_actual_cli_prepare_and_preflight(tmp_path, monkeypatch):
    from tests.evaluation_preparation import test_qualified_harvard as original

    monkeypatch.setattr(
        original, "PROFILE", "configs/research/multimodal_harvard_visual_priority_v2.json"
    )
    original.test_cli_prepare_preflight_and_tamper_no_live_owners(tmp_path, monkeypatch)


def test_current_reserve_is_monotonic_and_excludes_unknown(tmp_path):
    from datetime import UTC, datetime, timedelta

    from packages.grounding.models import digest
    from packages.multimodal_research.campaign import MultimodalCampaign, configuration

    c = object.__new__(MultimodalCampaign)
    c.output = tmp_path
    c.config = configuration(directional=True, qualified_observations=True)
    c.base = SimpleNamespace(
        active_schedule_v19=True,
        observations={str(n): SimpleNamespace(status="obtained") for n in range(6)},
        manifest=SimpleNamespace(checksum=digest("test")),
        active_owner=SimpleNamespace(
            profile=load("configs/research/multimodal_harvard_visual_priority_v2.json")
        ),
    )
    c.accepted_observations = lambda: {"unknown": SimpleNamespace(status="unknown")}
    c.usage = lambda: 6
    c.previous_attempts = lambda: []
    c.protocol = dict(deadline=(datetime.now(UTC) + timedelta(seconds=4100)).isoformat())
    assert not c.active_review_reserved()
    c.protocol["deadline"] = (datetime.now(UTC) + timedelta(seconds=500)).isoformat()
    assert c.active_review_reserved()
    c.usage = lambda: 0
    assert c.active_review_reserved()


def test_pending_dispatch_cap_does_not_spend_local_closure_headroom(monkeypatch):
    import pytest

    from packages.llm_provider.provider import ProviderError
    from packages.multimodal_research.campaign import MultimodalCampaign
    from packages.multimodal_research.campaign_renewal import RenewedCampaign

    calls = []
    monkeypatch.setattr(
        MultimodalCampaign,
        "budget_factory",
        lambda *_a, **_k: SimpleNamespace(before_reserve=lambda value: calls.append(value)),
    )
    c = object.__new__(RenewedCampaign)
    c.protocol = dict(schema_version="multimodal-campaign-renewal/1.15.0", max_model_dispatches=2)
    c.usage = lambda: 1
    b = c.budget_factory()
    b.before_reserve("one")
    c.usage = lambda: 2
    with pytest.raises(ProviderError, match="budget"):
        b.before_reserve("two")
    assert calls == ["one"]


def test_pending_budget_accepts_only_explicit_shorter_phase_window():
    from datetime import timedelta

    import pytest

    from packages.evaluation_preparation.dynamic_campaign import bounded_phase_binding
    from packages.llm_provider.bounded_live import BoundedLiveBudget
    from packages.llm_provider.provider import ProviderError
    from tests.agent_provider.test_bounded_live import (
        Profile,
        SyntheticJournal,
        binding,
        load_config,
    )

    original = binding(
        profile=Profile(max_steps=20, max_no_progress=2, max_retries=1, seconds=2700)
    )
    short = bounded_phase_binding(original, original.started_at + timedelta(seconds=2500))
    journal = SyntheticJournal()
    config = load_config()
    with pytest.raises(ProviderError, match="configuration"):
        BoundedLiveBudget(binding=short, config=config, journal=journal)
    b = BoundedLiveBudget(binding=short, config=config, journal=journal, bounded_phase=True)
    assert b.deadline == short.deadline
    limits = [e for e in journal.read() if e.get("schema_version") == "bounded-live-limits/1.1.0"]
    assert len(limits) == 1 and limits[0]["seconds"] == 2500
    before = journal.read()
    BoundedLiveBudget(binding=short, config=config, journal=journal, bounded_phase=True)
    assert journal.read() == before
    # The original request owner rejects a changed deadline before budget limits.
    with pytest.raises(ProviderError, match="^scope$"):
        BoundedLiveBudget(binding=original, config=config, journal=journal)


def test_original_campaign_factory_passes_narrowed_phase_contract():
    from datetime import timedelta

    from packages.evaluation_preparation.dynamic_campaign import Campaign, bounded_phase_binding
    from tests.agent_provider.test_bounded_live import (
        Profile,
        SyntheticJournal,
        binding,
        load_config,
    )

    original = binding(
        profile=Profile(max_steps=20, max_no_progress=2, max_retries=1, seconds=2700)
    )
    short = bounded_phase_binding(original, original.started_at + timedelta(seconds=2500))
    c = object.__new__(Campaign)
    c.protocol = dict(deadline=short.deadline.isoformat())
    journal = SyntheticJournal()
    budget = c.budget_factory(binding=short, config=load_config(), journal=journal)
    assert budget.effective_deadline == short.deadline
    before = journal.read()
    limits = [e for e in before if e.get("schema_version") == "bounded-live-limits/1.1.0"]
    assert len(limits) == 1 and limits[0]["seconds"] == 2500
    c.budget_factory(binding=short, config=load_config(), journal=journal)
    assert journal.read() == before
    full_journal = SyntheticJournal()
    c.budget_factory(binding=original, config=load_config(), journal=full_journal)
    assert any(e.get("schema_version") == "bounded-live-limits/1.0.0" for e in full_journal.read())


def test_phase_budget_amendment_is_exact_and_rejects_changed_old_failure(tmp_path, monkeypatch):
    import json

    import pytest

    from packages.multimodal_research import implementation
    from packages.multimodal_research.context_recovery import read_budget_amendment
    from packages.object_store import checksum

    key = "packages/evaluation_preparation/dynamic_campaign.py"
    before, after = {key: "old"}, {key: "new"}
    monkeypatch.setattr(implementation, "freeze", lambda: after)
    for name in ("entry", "protocol", "stage-failure", "request", "test"):
        (tmp_path / (name + ".json")).write_text("{}")
    sha = checksum(b"{}")
    value = dict(
        schema_version="pre-wire-phase-budget-repair/1.0.0",
        campaign_id=tmp_path.name,
        implementation_before=before,
        implementation_after=after,
        delta=[key],
        entry_checksum=sha,
        protocol_checksum=sha,
        failure_checksum=sha,
        request_path=str(tmp_path / "request.json"),
        request_checksum=sha,
        tests=[dict(path=str(tmp_path / "test.json"), checksum=sha)],
    )
    (tmp_path / "phase-budget-repair.json").write_text(json.dumps(value))
    assert read_budget_amendment(tmp_path) == value
    assert implementation.checked_snapshot(tmp_path, before) == before
    (tmp_path / "stage-failure.json").write_text('{"changed":true}')
    with pytest.raises(ValueError, match="phase_budget_repair_binding"):
        read_budget_amendment(tmp_path)


def test_phase_budget_recovery_restores_committed_definition_without_source(tmp_path, monkeypatch):
    import json
    from contextlib import nullcontext

    from packages.multimodal_research import campaign, context_recovery, stage_failure

    c = object.__new__(campaign.MultimodalCampaign)
    c.output = tmp_path
    c.base = SimpleNamespace(definition_expansion_policy={})
    c.protocol = {}
    # This unit isolates saved-definition recovery, not filesystem locking.
    # Actual mutual exclusion is exercised by the original Runtime PG tests.
    c.execution_session = nullcontext
    c.targets = ("check",)
    c.config = SimpleNamespace(transport=SimpleNamespace(api_key_env="TEST_PHASE_REPAIR_KEY"))
    c.freeze = lambda: None
    checks = []
    c.guard = lambda: checks.append("campaign_guard")
    c.acquisition_exhausted = lambda: False
    c.active_review_reserved = lambda: False
    c.documents = lambda rid: SimpleNamespace(
        read=lambda key: (amendment,) if key.endswith(":r02") else None
    )
    amendment = dict(schema_version="pre-wire-phase-budget-repair/1.1.0", run_id="run")
    (tmp_path / "stage-failure.json").write_text(json.dumps({"old_failure": True}))
    (tmp_path / "phase-budget-repair.json").write_text("{}")
    monkeypatch.setattr(context_recovery, "read_budget_amendment", lambda p: amendment)
    monkeypatch.setattr(
        campaign.importlib.util,
        "spec_from_file_location",
        lambda *a: SimpleNamespace(loader=SimpleNamespace(exec_module=lambda m: None)),
    )
    monkeypatch.setattr(
        campaign.importlib.util,
        "module_from_spec",
        lambda s: SimpleNamespace(load_secret=lambda: "synthetic-only"),
    )

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "No Source dispatch or stopped-stage control during saved revision restore"
        )

    def read_revision():
        checks.append("current_sources_readback")
        return {"saved": True}

    def install(value):
        assert value == {"saved": True}
        checks.append("installed")

    c.definition_expansion = SimpleNamespace(
        advance=forbidden, current=forbidden, read_revision=read_revision, install=install
    )

    def phase(target):
        assert target == "check"
        assert c.base.definition_current() == {"saved": True}
        checks.append("original_phase_entry")
        raise ValueError("test_stop_before_dispatch")

    c.execute_phase = phase
    monkeypatch.setattr(stage_failure, "close", lambda c, e, **kw: {"error": str(e)})
    assert c.execute() == {"error": "test_stop_before_dispatch"}
    assert checks == [
        "campaign_guard",
        "current_sources_readback",
        "installed",
        "current_sources_readback",
        "original_phase_entry",
    ]
    assert json.loads((tmp_path / "stage-failure.json").read_text()) == {"old_failure": True}
