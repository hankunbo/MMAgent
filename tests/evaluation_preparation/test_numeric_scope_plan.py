"""Numeric inputs need a method plan, not a fabricated visual survey."""

import json

import pytest

from packages.dataset_adapters.multimodal.adapters import prepare
from packages.grounding.models import digest
from packages.multimodal_research import input_scope, scope_selection
from tests.dataset_adapters.test_multimodal import grape
from tests.evaluation_preparation import test_input_scope_phase as scope_fixture
from tests.evaluation_preparation.test_scope_selection import PROFILE


def numeric_input(tmp_path):
    prepared = prepare(
        "grape", grape(tmp_path / "data"), "1:OD", cutoff_visit=1, semantic_profile="grape-v1"
    )
    output = tmp_path / "run"
    directory = output / "views"
    directory.mkdir(parents=True)
    (directory / "views.json").write_text(
        json.dumps(dict(input_checksum=prepared.manifest.checksum, views=[]))
    )
    return prepared, output


def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(scope_fixture, "PROFILE", PROFILE)
    monkeypatch.setattr(scope_fixture, "prepared_view", numeric_input)
    return scope_fixture.fixture(tmp_path)


def test_numeric_plan_keeps_questions_and_has_no_survey_or_extra_vote(tmp_path, monkeypatch):
    base, campaign, _ = setup(tmp_path, monkeypatch)
    assert campaign.scope_targets == ()
    before = digest(base.ledger)
    plan = scope_selection.project(base, input_scope.read_results(campaign, require_all=True))
    assert plan["schema_version"] == "input-scope-method-plan/1.1.0"
    assert plan["storage_basis"] == "original_input_capabilities_without_visual_survey"
    assert plan["scope_records"] == [] and plan["added_disease_votes"] == 0
    assert any(r["action"] == "original_numeric_owner" for r in plan["rows"])
    assert any(r["state"] == "missing_input" for r in plan["rows"])
    assert not any(r["native_candidate"] for r in plan["rows"])
    assert digest(base.ledger) == before


def test_empty_survey_plan_still_requires_original_current_binding(tmp_path, monkeypatch):
    base, campaign, _ = setup(tmp_path, monkeypatch)
    path = base.output / "input-scope-plan.json"
    raw = json.loads(path.read_bytes())
    raw["capability_checksum"] = digest("unrelated-input")
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="input_scope_plan_changed"):
        input_scope.read_results(campaign, require_all=True)


def test_visual_work_cannot_be_hidden_by_empty_target_list(tmp_path, monkeypatch):
    from tests.evaluation_preparation.test_scope_selection import setup as visual_setup

    _, campaign, _ = visual_setup(tmp_path, monkeypatch)
    assert campaign.scope_targets
    campaign.scope_targets = ()
    with pytest.raises(ValueError, match="input_scope_plan_changed"):
        input_scope.read_results(campaign, require_all=True)
