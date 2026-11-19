"""Current projection plan semantics; original execution is tested separately."""

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from packages.grounding.models import digest
from tests.evaluation_preparation.test_capability_window import plan_fixture


def window():
    from packages.multimodal_research import capability_window

    return capability_window


def test_current_plan_keeps_budget_and_distinguishes_reassessment(tmp_path):
    owner, tip, inventory, args = plan_fixture(tmp_path)
    owner.base.visible_value = SimpleNamespace(checksum=digest("visible"))
    inventory["confirmed_review"]["original_assessment_ids"] = ["one", "two"]
    inventory["confirmed_review"]["compatibility_preview"] = {
        "assessment_ids": ["one"],
        "execution_authorized": False,
        "current_scope_required_before_dispatch": True,
    }
    before = deepcopy((tip, inventory))
    module = window()
    old = module.plan_for(owner, tip, inventory, **args)
    new = module.plan_for(owner, tip, inventory, **args, current=True)
    assert old["schema_version"] == module.VERSION
    assert new["schema_version"] == module.CURRENT_VERSION
    assert new["reuse_assessment_ids"] == ["one"]
    assert new["review_target_ids"] == ["one", "two"]
    assert new["previous_attempts"] == old["previous_attempts"]
    assert new["max_new_requests"] == old["max_new_requests"] == 7
    assert new["deadline"] == old["deadline"]
    assert new["visible"] == owner.base.visible_value.checksum
    assert new["campaign_id"] == args["output"].name
    assert "visible" not in old and "campaign_id" not in old
    assert (tip, inventory) == before


def test_current_plan_still_rejects_exhausted_budget(tmp_path):
    owner, tip, inventory, args = plan_fixture(tmp_path)
    owner.base.visible_value = SimpleNamespace(checksum=digest("visible"))
    tip["previous_attempts"][0]["requests"] = 40
    with pytest.raises(ValueError, match="capability_window_bounds"):
        window().plan_for(owner, tip, inventory, **args, current=True)


def test_profile_binds_distinct_map_and_scope_contract():
    from packages.multimodal_research.active_requirements import load
    from packages.multimodal_research.input_scope import read_policy
    from packages.multimodal_research.scope_selection import read_policy as method_policy

    profile = load(Path("configs/research/multimodal_catalog_acquisition_v8.json"))
    assert (
        read_policy(profile["input_scope_policy"]).schema_version
        == "input-scope-phase-policy/1.1.0"
    )
    policy = method_policy(profile)
    assert policy.regions_by_question["rnflt_distribution"]
    checks = {c["id"]: c for c in profile["checks"]}
    assert checks["rnflt_distribution"] != checks["oct_rnfl"]
    raw = json.loads((Path("configs/research/multimodal_catalog_acquisition_v8.json")).read_bytes())
    assert "harvard" not in raw["id"].lower()


def test_readonly_inventory_uses_original_views_before_successor_directory_exists(
    tmp_path, monkeypatch
):
    """Regression for the actual Harvard preflight failure; no fake image paths."""
    from copy import copy

    from packages.multimodal_research import (
        active_acquisition,
        campaign,
        campaign_renewal,
        input_scope,
        reuse,
    )
    from tests.evaluation_preparation.test_input_scope_phase import fixture

    base, original, seed = fixture(tmp_path)
    parent = SimpleNamespace(
        output=base.output,
        base=base,
        protocol={},
        config=original.config,
        target=object(),
        contexts=SimpleNamespace(read_research_run=lambda _: object()),
        accepted_observations=lambda: {},
    )
    image_target = next(r["id"] for r in base.ledger if r.get("component") == "rnflt:rnflt")
    expected_image = base.view(image_target)
    future = tmp_path / "not-created"
    candidate = window()
    projected = copy(base)
    projected.contexts = dict(base.contexts)
    projected.output = future
    before = digest(base.ledger)
    reviewed = dict(run_id=seed.material.scope.run_id, assessment_ids=[])
    transition = dict(changes=[], coverage={})
    monkeypatch.setattr(campaign_renewal, "stopped_tip", lambda *a, **k: {"previous_attempts": []})
    monkeypatch.setattr(campaign_renewal, "attempted_acquisition_targets", lambda *a: set())
    monkeypatch.setattr(active_acquisition, "successor_projection", lambda *a, **k: transition)
    monkeypatch.setattr(candidate, "review_origin", lambda *a, **k: (object(), reviewed))
    monkeypatch.setattr(candidate, "projected_base", lambda *a: projected)
    monkeypatch.setattr(campaign.MultimodalCampaign, "available_targets", lambda _: [])
    monkeypatch.setattr(input_scope, "project_plan", lambda *a: dict(entries=[]))
    monkeypatch.setattr(input_scope, "configuration", lambda *a: original.config)
    monkeypatch.setattr(reuse, "repository", lambda *a: None)
    monkeypatch.setattr(
        reuse, "ProcedureReuse", lambda *a: SimpleNamespace(plan={}, prepare=lambda: None)
    )
    reads = []

    class PreviewPhase:
        def __init__(self, received, **kwargs):
            self.base = received

        def review_reuse_preview(self, **kwargs):
            reads.append(self.base.output)
            assert self.base.view(image_target) == expected_image
            return SimpleNamespace(assessment_ids=()), dict(execution_authorized=False)

    monkeypatch.setattr(campaign, "Phase", PreviewPhase)
    candidate.inventory(
        parent,
        future,
        current=True,
        profile_path=str(Path("configs/research/multimodal_catalog_acquisition_v8.json")),
    )
    assert reads == [base.output]
    assert projected.output == future and not future.exists()
    assert digest(base.ledger) == before


def test_public_cli_dispatches_current_prepare_with_original_inputs(monkeypatch, capsys):
    import sys

    from packages.multimodal_research import capability_window as original_window

    seen = []

    def prepare(**kwargs):
        seen.append(kwargs)
        return dict(api_requests=0, check_only=kwargs["check_only"])

    monkeypatch.setattr(original_window, "prepare_current", prepare, raising=False)
    from apps import multimodal_case as cli

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "multimodal_case",
            "capability-current-prepare",
            "--previous",
            "old-owner",
            "--output",
            "new-output",
            "--authorization",
            "test-explicit-window",
            "--check-only",
        ],
    )
    assert cli.main() == 0
    assert seen == [
        dict(
            previous="old-owner",
            output="new-output",
            authorization="test-explicit-window",
            check_only=True,
            network_route=None,
        )
    ]
    assert json.loads(capsys.readouterr().out) == dict(api_requests=0, check_only=True)
