"""Body anchors must not lose limited slots to title/caption keyword matches."""

import json
from pathlib import Path

import pytest

from packages.multimodal_research.definition_expansion import read_expansion_policy
from tests.evaluation_preparation.test_definition_source_balance import fixture


def owner(version="definition-passage-plan/5.0.0"):
    source, rows, queries, reads = fixture(("old-a", "old-b"), version)
    for code, row in rows.items():
        kind = {"0": "heading", "1": "caption", "2": "paragraph"}[code[-1]]
        row["position"]["kind"] = kind
        row["hit"]["position"] = dict(row["position"])
    return source, rows, queries, reads


def test_body_anchor_uses_original_owner_and_skips_higher_ranked_titles():
    source, _, _, reads = owner()
    value = source.reconcile_pool(queries=["q"], limit=2)
    assert {m["code"] for m in value["materials"]} == {"old-a:2", "old-b:2"}
    assert set(reads) == {"old-a:2", "old-b:2"}
    rejected = [
        d
        for d in value["retrieval"]["candidate_decisions"]
        if d["reason"] == "non_explanatory_anchor"
    ]
    assert rejected and all(not d["current_material_read"] for d in rejected)
    assert value["retrieval"]["medical_entailment_proven"] is False


def test_old_plan_keeps_original_ranked_selection():
    source, _, _, _ = owner("definition-passage-plan/4.0.0")
    value = source.reconcile_pool(queries=["q"], limit=2)
    assert {m["code"] for m in value["materials"]} == {"old-a:0", "old-b:0"}


def test_only_titles_or_captions_return_explicit_empty_not_invented_definition():
    source, rows, _, _ = owner()
    for row in rows.values():
        row["position"]["kind"] = row["hit"]["position"]["kind"] = "caption"
    value = source.reconcile_pool(queries=["q"], limit=2)
    assert value["materials"] == []
    assert len(value["backlog"]) == 6


def test_owner_read_error_is_not_reclassified_as_missing_material():
    source, _, _, _ = owner()

    def changed(*args, **kwargs):
        raise RuntimeError("current_source_changed")

    source.repository.read_definition_material = changed
    with pytest.raises(RuntimeError, match="current_source_changed"):
        source.reconcile_pool(queries=["q"], limit=2)


def test_current_position_rechecked_and_not_trusted_from_search_metadata():
    source, rows, _, _ = owner()
    for row in rows.values():
        row["position"]["kind"] = "caption"
    value = source.reconcile_pool(queries=["q"], limit=2)
    assert value["materials"] == []
    assert any(
        d["reason"] == "non_explanatory_anchor" and d["current_material_read"]
        for d in value["retrieval"]["candidate_decisions"]
    )


def test_expansion_uses_explicit_version_not_silent_old_plan_upgrade():
    raw = json.loads(Path("configs/research/multimodal_definition_gap_loop_v2.json").read_bytes())
    assert raw["schema_version"] == "campaign-definition-expansion/1.24.0"
    raw["schema_version"] = "campaign-definition-expansion/1.25.0"
    for plan in raw["passage_plans"].values():
        plan["schema_version"] = "definition-passage-plan/5.0.0"
    value = read_expansion_policy(raw)
    assert read_expansion_policy(value.model_dump(mode="json")) == value
    raw["schema_version"] = "campaign-definition-expansion/1.24.0"
    with pytest.raises(ValueError):
        read_expansion_policy(raw)


def test_body_and_old_plans_cannot_mix_in_one_pool():
    from packages.evidence_reuse.definition_sources import read_passage_plan

    source, _, _, _ = owner()
    old = source.passage_plans["q"].model_dump(mode="json")
    old["schema_version"] = "definition-passage-plan/4.0.0"
    source.passage_plans["old"] = read_passage_plan(old)
    with pytest.raises(ValueError, match="definition_pool_plan_version"):
        source.reconcile_pool(queries=["q", "old"], limit=2)


@pytest.mark.parametrize("new_runtime", [False, True])
def test_current_definition_window_does_not_inherit_old_review_reuse(
    monkeypatch, tmp_path, new_runtime
):
    from types import SimpleNamespace

    from packages.multimodal_research import campaign_renewal as renewal

    def initialize(self, base, target, *, campaign_id):
        self.base, self.target = base, target

    monkeypatch.setattr(renewal.MultimodalCampaign, "__init__", initialize)
    previous_reader = object()
    previous_config = object()
    base = SimpleNamespace(
        confirmed_review_reader=previous_reader,
        review_clarification={"old": True},
        review_target_ids=("old",),
    )
    parent = SimpleNamespace(base=base, target=object(), config=previous_config)
    plan = dict(
        schema_version="multimodal-campaign-renewal/1.11.0",
        mode="definitions",
        definition_increment_only=True,
        seconds=3600,
        max_new_requests=8,
        expansion_policy=dict(
            schema_version="campaign-definition-expansion/1.25.0",
            reserve_review_requests=3,
            reserve_review_seconds=1200,
            reserve_close_seconds=300,
        ),
    )
    if new_runtime:
        from packages.multimodal_research.execution_lease import PROFILE

        plan.update(
            schema_version=renewal.RUNTIME_DEFINITION_VERSION, execution_lease_profile=PROFILE
        )
        plan["expansion_policy"]["schema_version"] = renewal.DISCOVERED_CANDIDATE_POLICY
    campaign = renewal.RenewedCampaign(parent, tmp_path / "next", plan)
    assert campaign.config is previous_config
    assert campaign.base is not base
    assert campaign.base.confirmed_review_reader is None
    assert campaign.base.review_target_ids == ()
    assert base.confirmed_review_reader is previous_reader
    assert base.review_target_ids == ("old",) and base.review_clarification == {"old": True}


@pytest.mark.parametrize("requests, seconds", [(8, 1200), (3, 3300)])
def test_impossible_definition_reserve_rejected_before_new_window(requests, seconds):
    from packages.multimodal_research.campaign_renewal import validate_increment_window

    with pytest.raises(ValueError, match="renewal_increment_no_proposal_capacity"):
        validate_increment_window(
            dict(
                reserve_review_requests=requests,
                reserve_review_seconds=seconds,
                reserve_close_seconds=300,
            ),
            seconds=3600,
            cap=8,
        )
