"""Versioned expiry successor: unsent work only; no reset or quarantined retry."""

from copy import deepcopy
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from packages.grounding.models import digest
from packages.multimodal_research.pending_review import (
    UNSTARTED_WINDOW_VERSION,
    validate_unstarted_window,
)


def fixture():
    now = datetime(2026, 10, 20, tzinfo=UTC)
    parent = SimpleNamespace(
        base=SimpleNamespace(execution_profile="relay-root-question-v1"),
        config={"synthetic": True},
        protocol=dict(
            schema_version="multimodal-campaign-renewal/1.23.0",
            deadline=(now - timedelta(days=1)).isoformat(),
        ),
    )
    pending = dict(
        selected_ids=["unstarted:a", "unstarted:b", "unstarted:c"],
        accepted_ids=["old:accepted"],
        quarantined_batches=[
            dict(
                assessment_ids=["old:unknown"],
                request_ids=["original-request"],
                new_reassessment_authorized=False,
            )
        ],
    )
    plan = dict(
        schema_version=UNSTARTED_WINDOW_VERSION,
        mode="review",
        pending_new_window="owner_verified_unstarted_only/1.0.0",
        review_increment_only=True,
        pending_review=pending,
        review_target_ids=list(pending["selected_ids"]),
        previous_window_deadline=parent.protocol["deadline"],
        whole_case_execution_profile=parent.base.execution_profile,
        provider=digest(parent.config),
        started_at=now.isoformat(),
        deadline=(now + timedelta(seconds=2700)).isoformat(),
        seconds=2700,
        max_model_dispatches=4,
        max_new_requests=5,
        max_cumulative_requests=40,
        previous_attempts=[dict(requests=18)],
        new_vision_requests=0,
        new_public_search_allowed=False,
        acquisition_targets=[],
        expansion_policy=None,
    )
    return parent, plan


def test_narrow_expired_successor_keeps_history_and_quarantine():
    parent, plan = fixture()
    before = deepcopy(plan), deepcopy(parent.protocol)
    validate_unstarted_window(plan, parent)
    assert (plan, parent.protocol) == before
    assert sum(v["requests"] for v in plan["previous_attempts"]) + plan["max_new_requests"] == 23


@pytest.mark.parametrize(
    "change",
    [
        "version",
        "scope",
        "targets",
        "selected_accepted",
        "selected_unknown",
        "ack_unknown",
        "ack_format",
        "quarantine_authorized",
        "provider",
        "profile",
        "deadline",
        "seconds",
        "requests",
        "cumulative",
        "vision",
        "search",
        "repeat",
        "active_parent",
    ],
)
def test_new_window_rejects_broader_or_mismatched_authority(change):
    parent, plan = fixture()
    if change == "version":
        plan["schema_version"] = "multimodal-campaign-renewal/1.23.0"
    elif change == "scope":
        plan["pending_new_window"] = "retry-anything"
    elif change == "targets":
        plan["review_target_ids"].append("not-proven")
    elif change == "selected_accepted":
        plan["pending_review"]["accepted_ids"].append("unstarted:a")
    elif change == "selected_unknown":
        plan["pending_review"]["quarantined_batches"][0]["assessment_ids"].append("unstarted:a")
    elif change == "ack_unknown":
        plan["pending_review"]["acknowledge_unknown_request"] = "original-request"
    elif change == "ack_format":
        plan["pending_review"]["acknowledged_format_requests"] = ["known-error"]
    elif change == "quarantine_authorized":
        plan["pending_review"]["quarantined_batches"][0]["new_reassessment_authorized"] = True
    elif change == "provider":
        plan["provider"] = digest("changed-provider")
    elif change == "profile":
        plan["whole_case_execution_profile"] = "unverified-profile"
    elif change == "deadline":
        plan["deadline"] = "2026-10-20T03:00:00+00:00"
        plan["seconds"] = 10800
    elif change == "seconds":
        plan["seconds"] += 1
    elif change == "requests":
        plan["max_new_requests"] = 7
    elif change == "cumulative":
        plan["previous_attempts"] = [dict(requests=39)]
    elif change == "vision":
        plan["new_vision_requests"] = 1
    elif change == "search":
        plan["new_public_search_allowed"] = True
    elif change == "repeat":
        parent.parent = SimpleNamespace(protocol=dict(schema_version=UNSTARTED_WINDOW_VERSION))
    else:
        parent.protocol["deadline"] = "2026-10-20T00:20:00+00:00"
        plan["previous_window_deadline"] = parent.protocol["deadline"]
    with pytest.raises(ValueError, match="pending_unstarted_window"):
        validate_unstarted_window(plan, parent)


def test_constructor_cannot_opt_old_version_into_new_window(tmp_path):
    from packages.multimodal_research.campaign_renewal import RenewedCampaign

    with pytest.raises(ValueError, match="pending_unstarted_window_version"):
        RenewedCampaign(
            SimpleNamespace(),
            tmp_path,
            dict(schema_version="multimodal-campaign-renewal/1.23.0", pending_new_window=True),
        )


def followup_fixture():
    parent, plan = fixture()
    first = deepcopy(plan)
    first["started_at"] = "2026-10-19T00:00:00+00:00"
    first["deadline"] = parent.protocol["deadline"]
    parent.parent = SimpleNamespace(protocol=parent.protocol)
    parent.protocol = first
    plan.update(
        schema_version="multimodal-campaign-renewal/1.42.0",
        pending_window_parent_checksum=digest(first),
        review_target_ids=["unstarted:c"],
    )
    plan["pending_review"]["selected_ids"] = ["unstarted:c"]
    plan["pending_review"]["accepted_ids"].append("unstarted:a")
    plan["pending_review"]["quarantined_batches"].append(
        dict(
            assessment_ids=["unstarted:b"],
            request_ids=["unknown-in-first-window"],
            new_reassessment_authorized=False,
        )
    )
    return parent, plan


def test_followup_only_finishes_strict_remaining_unsent_subset():
    parent, plan = followup_fixture()
    before = deepcopy(plan), deepcopy(parent.protocol)
    validate_unstarted_window(plan, parent)
    assert (plan, parent.protocol) == before


@pytest.mark.parametrize(
    "change",
    [
        "parent_checksum",
        "not_direct_first",
        "same_selection",
        "outside_selection",
        "no_confirmed_progress",
        "unknown",
        "accepted",
        "second_followup",
        "old_version",
    ],
)
def test_followup_cannot_restart_first_window_or_replay_sent_work(change):
    parent, plan = followup_fixture()
    if change == "parent_checksum":
        plan["pending_window_parent_checksum"] = digest("wrong")
    elif change == "not_direct_first":
        parent.protocol["schema_version"] = "multimodal-campaign-renewal/1.23.0"
        plan["pending_window_parent_checksum"] = digest(parent.protocol)
    elif change == "same_selection":
        parent.protocol["pending_review"]["selected_ids"] = ["unstarted:c"]
        plan["pending_window_parent_checksum"] = digest(parent.protocol)
    elif change == "outside_selection":
        plan["pending_review"]["selected_ids"] = ["outside"]
        plan["review_target_ids"] = ["outside"]
    elif change == "no_confirmed_progress":
        plan["pending_review"]["accepted_ids"] = ["old:accepted"]
    elif change == "unknown":
        plan["pending_review"]["selected_ids"] = ["unstarted:b"]
        plan["review_target_ids"] = ["unstarted:b"]
    elif change == "accepted":
        plan["pending_review"]["selected_ids"] = ["unstarted:a"]
        plan["review_target_ids"] = ["unstarted:a"]
    elif change == "second_followup":
        parent.parent.protocol["schema_version"] = "multimodal-campaign-renewal/1.42.0"
    else:
        plan["schema_version"] = UNSTARTED_WINDOW_VERSION
    with pytest.raises(ValueError, match="pending_unstarted_window"):
        validate_unstarted_window(plan, parent)


def test_actual_cli_passes_explicit_window_to_original_prepare(monkeypatch):
    import sys

    from apps import multimodal_case
    from packages.multimodal_research import campaign_renewal

    received = {}

    def prepare(**options):
        received.update(options)
        return {"synthetic_parser_only": True, "real_api_requests": 0}

    monkeypatch.setattr(campaign_renewal, "prepare", prepare)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "multimodal_case",
            "renewal-prepare",
            "--facts",
            "synthetic-parent",
            "--previous",
            "synthetic-parent",
            "--output",
            "synthetic-child",
            "--mode",
            "review",
            "--authorization",
            "synthetic",
            "--pending-review",
            "--pending-new-window",
            "--request-limit",
            "5",
        ],
    )
    assert multimodal_case.main() == 0
    assert received["pending_review"] is received["pending_new_window"] is True
    assert received["request_limit"] == 5
    assert received["acknowledge_unknown_request"] is None
    assert received["acknowledge_format_request"] == []
