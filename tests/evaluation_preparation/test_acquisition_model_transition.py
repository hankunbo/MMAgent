"""A model-only successor must preserve history and exclude any old dispatch."""

import copy
from datetime import UTC, datetime, timedelta

import pytest

from packages.grounding.models import digest
from packages.llm_provider.config import read_provider_config
from packages.multimodal_research import provider_transition as t
from packages.multimodal_research.campaign import configuration


def configs():
    current = configuration(
        resilient_vision=True,
        directional=True,
        exact_primary_status=True,
        qualified_observations=True,
        execution_profile="relay-root-question-v1",
        review_batch_size=3,
    )
    raw = current.model_dump(mode="json")
    raw["transport"]["model"] = "gpt-4o-2024-08-06"
    old = read_provider_config(raw)
    return old, t.acquisition_configuration(old, t.ACQUISITION_PROFILE)


def protocol(old_config):
    now = datetime.now(UTC)
    return now, dict(
        schema_version="multimodal-campaign-renewal/1.26.0",
        mode="acquisition",
        acquisition_increment_only=True,
        acquisition_targets=["a", "b", "c"],
        review_target_ids=["a", "b", "c"],
        acquisition_selection={"selected": ["a", "b", "c"]},
        deadline=(now - timedelta(hours=1)).isoformat(),
        seconds=5400,
        max_new_requests=8,
        max_cumulative_requests=40,
        previous_attempts=[dict(requests=15), dict(requests=4)],
        provider=digest(old_config),
        reuse_assessment_ids=["old-1", "old-2"],
        confirmed_observations_checksum=digest("observations"),
        original_unknown_not_replayed=True,
    )


def build(old, config, now):
    return t.acquisition_plan_for(
        old,
        previous="old",
        output="new",
        authorization="explicit model switch",
        config=config,
        started=now,
        implementation={},
    )


def test_only_model_changes_not_roles_or_limits():
    old, new = configs()
    raw = new.model_dump(mode="json")
    assert raw["transport"]["model"] == "gpt-6-sol"
    raw["transport"]["model"] = old.transport.model
    assert raw == old.model_dump(mode="json")


def test_preserve_history_scope_and_original_protocol():
    origin, new = configs()
    now, old = protocol(origin)
    before = copy.deepcopy(old)
    plan = build(old, new, now)
    assert old == before
    assert plan["schema_version"] == t.ACQUISITION_VERSION
    assert plan["previous_attempts"] == old["previous_attempts"]
    assert plan["review_target_ids"] == old["review_target_ids"]
    assert plan["reuse_assessment_ids"] == old["reuse_assessment_ids"]
    assert plan["max_new_requests"] == 8
    assert plan["prior_deadline"] == old["deadline"]
    assert datetime.fromisoformat(plan["deadline"]) == now + timedelta(seconds=5400)
    assert plan["provider_transition"]["previous_protocol"] == digest(old)
    assert plan["provider_transition"]["predecessor_requests"] == 0


@pytest.mark.parametrize(
    "change",
    [
        {"mode": "review"},
        {"acquisition_increment_only": False},
        {"schema_version": "multimodal-campaign-renewal/1.17.0"},
        {"previous_attempts": [dict(requests=39)]},
        {"previous_attempts": [dict(requests=True)]},
        {"previous_attempts": [dict(requests=-1)]},
        {"max_new_requests": 0},
        {"max_new_requests": 41},
        {"seconds": 100000},
        {"seconds": True},
        {"review_target_ids": ["old-1"]},
        {"acquisition_targets": []},
        {"acquisition_targets": ["a"] * 3},
        {"deadline": (datetime.now(UTC) + timedelta(days=1)).isoformat()},
    ],
)
def test_invalid_scope_and_budget_rejected(change):
    origin, new = configs()
    now, old = protocol(origin)
    old.update(change)
    with pytest.raises(ValueError, match="acquisition_model_transition_scope_or_budget"):
        build(old, new, now)


def test_unchanged_model_and_unapproved_profile_rejected():
    _, new = configs()
    with pytest.raises(ValueError, match="acquisition_model_transition_contract"):
        t.acquisition_configuration(new, t.ACQUISITION_PROFILE)
    with pytest.raises(ValueError, match="acquisition_model_transition_profile"):
        t.acquisition_configuration(new, "/tmp/untrusted.json")


def test_original_review_reserve_is_available_for_new_version():
    from packages.multimodal_research.campaign_renewal import RenewedCampaign

    campaign = RenewedCampaign.__new__(RenewedCampaign)
    deadline = datetime.now(UTC) + timedelta(hours=1)
    campaign.protocol = dict(schema_version=t.ACQUISITION_VERSION, deadline=deadline.isoformat())
    campaign._final_review_active = False
    with campaign.phase_context(None):
        assert campaign.work_deadline() == deadline - timedelta(seconds=60)
    assert campaign._final_review_active is False


def test_changed_predecessor_file_is_not_accepted_as_cached_history(tmp_path):
    import json

    (tmp_path / "protocol.json").write_text(json.dumps({"changed": True}))
    with pytest.raises(ValueError, match="predecessor_protocol_changed"):
        t.verify_unexecuted(None, tmp_path, {"changed": False})


def test_cli_explicitly_selects_model_only_acquisition(monkeypatch, capsys):
    import sys

    from apps.multimodal_case import main

    received = []

    def prepare(**kwargs):
        received.append(kwargs)
        return {"api_requests": 0}

    monkeypatch.setattr(t, "prepare", prepare)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "multimodal_case",
            "provider-transition-prepare",
            "--previous",
            "old",
            "--output",
            "new",
            "--authorization",
            "explicit",
            "--model-only-acquisition",
        ],
    )
    assert main() == 0
    assert received == [
        dict(previous="old", output="new", authorization="explicit", model_only_acquisition=True)
    ]
    assert '"api_requests": 0' in capsys.readouterr().out
