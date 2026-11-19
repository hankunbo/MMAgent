"""Explicit new windows reuse facts, not old execution authority or unknown replies."""

import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from packages.grounding.models import digest
from packages.llm_provider.provider import ProviderError
from packages.multimodal_research import campaign_renewal as m


def test_renewal_keeps_small_mode_caps():
    assert m.LIMITS == {"review": (1800, 3), "definitions": (3600, 12), "acquisition": (5400, 12)}


@pytest.mark.parametrize(
    "version,review,allowed",
    [
        ("1.0.0", True, False),
        ("1.1.0", False, False),
        ("1.1.0", True, True),
        ("1.7.0", False, False),
        ("1.7.0", True, True),
        ("1.17.0", False, False),
        ("1.17.0", True, True),
    ],
)
def test_final_review_does_not_consume_its_own_acquisition_reserve(
    tmp_path, version, review, allowed
):
    c = object.__new__(m.RenewedCampaign)
    c.base = SimpleNamespace(check=lambda: None, detailed=True)
    c.output = tmp_path
    c.protocol = dict(
        schema_version="multimodal-campaign-renewal/" + version,
        max_new_requests=6,
        previous_attempts=[],
        deadline=(datetime.now(UTC) + timedelta(seconds=180)).isoformat(),
    )
    c.usage = lambda: 2
    c.previous_attempts = lambda: []
    c._final_review_active = review
    if allowed:
        c.guard()
    else:
        with pytest.raises(ProviderError, match="deadline"):
            c.guard()


def test_new_final_review_still_respects_original_deadline_cancel_and_budget(tmp_path):
    c = object.__new__(m.RenewedCampaign)
    c.base = SimpleNamespace(check=lambda: None, detailed=True)
    c.output = tmp_path
    c.protocol = dict(
        schema_version="multimodal-campaign-renewal/1.1.0",
        max_new_requests=6,
        previous_attempts=[],
        deadline=(datetime.now(UTC) - timedelta(seconds=1)).isoformat(),
    )
    c.usage = lambda: 2
    c.previous_attempts = lambda: []
    c._final_review_active = True
    with pytest.raises(ProviderError, match="deadline"):
        c.guard()
    c.protocol["deadline"] = (datetime.now(UTC) + timedelta(seconds=30)).isoformat()
    with pytest.raises(ProviderError, match="deadline"):
        c.guard()
    c.protocol["deadline"] = (datetime.now(UTC) + timedelta(seconds=180)).isoformat()
    c.usage = lambda: 6
    with pytest.raises(ProviderError, match="budget"):
        c.guard()
    c.usage = lambda: 2
    (tmp_path / "cancel").touch()
    with pytest.raises(ProviderError, match="cancelled"):
        c.guard()


def test_final_review_reserve_applies_only_inside_owner_review_stage(monkeypatch):
    from packages.multimodal_research.campaign import MultimodalCampaign

    c = object.__new__(m.RenewedCampaign)
    c.base = SimpleNamespace()
    flags = []

    def phase(self, target):
        flags.append((target, self._final_review_active))
        if target is None:
            raise RuntimeError("local_failure")
        return target

    monkeypatch.setattr(MultimodalCampaign, "execute_phase", phase)
    assert c.execute_phase("new_check") == "new_check"
    with pytest.raises(RuntimeError, match="local_failure"):
        c.execute_phase(None)
    assert flags == [("new_check", False), (None, True)]
    assert not c._final_review_active


def test_review_does_not_call_expansion_or_select_old_images(tmp_path):
    c = object.__new__(m.RenewedCampaign)
    calls = []
    c.protocol = dict(mode="review")
    c.targets = ()
    c.base = SimpleNamespace(
        check=lambda: calls.append("current"), definition_revision={"rows": []}
    )
    assert c.expand_definitions() == {"rows": []}
    assert calls == ["current"] and c.targets == ()


def test_actual_definition_owner_increment_reaches_renewal_scheduler(tmp_path):
    from tests.evaluation_preparation.test_definition_expansion import fixture

    e = fixture(tmp_path)
    c = object.__new__(m.RenewedCampaign)
    c.__dict__.update(e.campaign.__dict__)
    c.protocol = {**c.protocol, "mode": "definitions"}
    c.definition_expansion = e
    c.guard = lambda **kw: None
    c.guard_definition_materials = lambda **kw: None
    c.previous_attempts = lambda: []
    e.campaign = c
    revision = c.expand_definitions()
    assert revision["denominator_after"] == 1
    assert c.targets == tuple(revision["added"])
    saved = json.loads((tmp_path / "increment-schedule.json").read_bytes())
    assert saved["scheduled"] == revision["added"]
    assert e.owner.read(next(iter(revision["accepted"]))) in revision["accepted"].values()
    assert c.expand_definitions() == revision and e.budget.used == 1


def test_cap_is_checked_before_owner_guard(monkeypatch, tmp_path):
    from packages.multimodal_research.campaign import MultimodalCampaign

    c = object.__new__(m.RenewedCampaign)
    c.output, c.protocol = tmp_path, dict(max_new_requests=3)
    c.usage = lambda: 3
    calls = []
    monkeypatch.setattr(MultimodalCampaign, "guard", lambda *a, **kw: calls.append(kw))
    with pytest.raises(ProviderError, match="budget"):
        c.guard()
    c.guard(dispatch=True)
    assert calls == [dict(dispatch=True)]
    (tmp_path / "cancel").touch()
    with pytest.raises(ProviderError, match="cancelled"):
        c.guard(dispatch=True)


def test_expired_history_is_preserved_not_used_as_new_deadline(tmp_path):
    old = dict(deadline=(datetime.now(UTC) - timedelta(hours=1)).isoformat())
    result = dict(complete=False, error="timeout")
    (tmp_path / "protocol.json").write_text(json.dumps(old))
    (tmp_path / "result.json").write_text(json.dumps(result))
    plan = dict(tip=dict(path=str(tmp_path), protocol=digest(old), result=digest(result)))
    m.check_local_history(plan)
    old["deadline"] = (datetime.now(UTC) + timedelta(hours=1)).isoformat()
    (tmp_path / "protocol.json").write_text(json.dumps(old))
    with pytest.raises(ValueError, match="history_changed"):
        m.check_local_history(plan)


def test_gate_policy_is_bounded_and_maps_real_definition_gaps():
    from pathlib import Path

    from packages.multimodal_research.definition_expansion import read_expansion_policy

    p = read_expansion_policy(
        json.loads(Path("configs/research/multimodal_definition_gate_v1.json").read_bytes())
    )
    assert p.max_definitions == 1 and p.max_candidates == 2 and p.max_rounds == 1
    assert set(p.public_queries) == set(p.query_requirements) == set(p.public_topics)
    assert p.search_failure_policy == "technical_backlog_no_retry"


def test_parse_repair_retains_deadline_and_remaining_external_budget():
    now = datetime.now(UTC)
    until = now + timedelta(minutes=20)
    old = dict(mode="definitions", deadline=until.isoformat(), max_new_requests=12)
    assert m.continuation_bounds(old, 6, now) == (until, 6)
    with pytest.raises(ValueError, match="remaining_budget"):
        m.continuation_bounds(old, 12, now)
    with pytest.raises(ValueError, match="remaining_budget"):
        m.continuation_bounds(old, 6, until)
    with pytest.raises(ValueError, match="one_continuation"):
        m.continuation_bounds({**old, "continuation": True}, 6, now)


def test_compact_text_config_changes_batch_not_medical_prompt_or_timeout():
    old = m.configuration(resilient_vision=True, directional=True, exact_primary_status=True)
    new = m.compact_configuration()
    assert new.review_batching.batch_size == 2
    assert new.transport == old.transport and new.direction_policy == old.direction_policy
    for before, after in zip(old.roles, new.roles, strict=True):
        assert before.prompt_version == after.prompt_version
        assert before.prompt_checksum == after.prompt_checksum


def test_zero_request_review_repair_retains_deadline_and_cannot_loop():
    now = datetime.now(UTC)
    old = dict(
        schema_version="multimodal-campaign-renewal/1.2.0",
        reuse_confirmed_review=True,
        deadline=(now + timedelta(minutes=20)).isoformat(),
        max_new_requests=2,
    )
    result = dict(
        new_requests=0,
        complete=False,
        phases=[dict(error="review_continuation_current_binding", target=None)],
    )
    assert m.review_repair_bounds(old, result, now) == (datetime.fromisoformat(old["deadline"]), 2)
    for altered in ({**result, "new_requests": 1}, {**result, "complete": True}):
        with pytest.raises(ValueError, match="zero_request"):
            m.review_repair_bounds(old, altered, now)
    with pytest.raises(ValueError, match="zero_request"):
        m.review_repair_bounds(
            {**old, "schema_version": "multimodal-campaign-renewal/1.3.0"}, result, now
        )
