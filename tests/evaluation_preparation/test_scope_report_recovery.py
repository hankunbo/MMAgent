"""Survey versions retain original root, profile, scope and method bindings."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from packages.dataset_adapters.multimodal.tasks import load_intent
from packages.grounding.models import digest
from packages.multimodal_research.active_requirements import load
from packages.multimodal_research.report_recovery import restore_execution_context
from packages.multimodal_research.single_proposition import root_direction_spec


def binding(revision):
    task = load_intent("advanced_glaucoma", "configs/research/multimodal_propositions_v1.json")[0]
    path = f"configs/research/multimodal_catalog_acquisition_v{revision}.json"
    profile = load(path)
    spec = dict(
        schema_version="multimodal-live-entry/1.26.0",
        execution_profile="relay-root-question-v1",
        root_direction_review=root_direction_spec(task),
        review_batch_size=3,
        active_acquisition=dict(path=path, checksum=digest(profile)),
    )
    protocol = dict(
        spec,
        schema_version=f"multimodal-live-campaign/1.{revision + 17}.0",
        input_scope={"plan_checksum": digest("scope")},
    )
    if revision == 4:
        protocol["scope_selection_policy"] = profile["scope_selection_policy"]
    return task, spec, protocol


@pytest.mark.parametrize("revision", [3, 4])
def test_scope_versions_preserve_task_question_and_batch_size(revision):
    task, spec, protocol = binding(revision)
    base = SimpleNamespace(task=task)
    restore_execution_context(base, spec, protocol)
    assert base.root_direction_review == root_direction_spec(task)
    assert base.review_batch_size == 3
    for tamper in (
        {"root_direction_review": {"title": "other task"}},
        {"review_batch_size": 2},
        {"execution_profile": "relay-snapshot-context-v1"},
    ):
        with pytest.raises(ValueError, match="saved_report_"):
            restore_execution_context(SimpleNamespace(task=task), spec, dict(protocol, **tamper))


@pytest.mark.parametrize("mutation", ["digest", "policy", "version", "scope", "old"])
def test_survey_version_is_not_a_blanket_legacy_bypass(mutation):
    task, spec, protocol = binding(4)
    spec, protocol = deepcopy(spec), deepcopy(protocol)
    if mutation == "digest":
        spec["active_acquisition"]["checksum"] = digest("changed")
    elif mutation == "policy":
        protocol["scope_selection_policy"] = {}
    elif mutation == "version":
        protocol["schema_version"] = "multimodal-live-campaign/1.20.0"
    elif mutation == "scope":
        del protocol["input_scope"]
    else:
        protocol["schema_version"] = "multimodal-live-campaign/1.16.0"
    with pytest.raises(ValueError, match="saved_report_scope_"):
        restore_execution_context(SimpleNamespace(task=task), spec, protocol)


def test_original_scope_projection_checked_before_reading_accepted_records(tmp_path, monkeypatch):
    from packages.multimodal_research import input_scope
    from packages.multimodal_research.report_recovery import restore_scope_context
    from tests.evaluation_preparation.test_scope_selection import setup

    base, campaign, _ = setup(tmp_path, monkeypatch)
    campaign.protocol = dict(
        input_scope=dict(
            plan_checksum=digest(base.input_scope_plan),
            provider_checksum=digest(
                input_scope.configuration(campaign.config, base.input_scope_plan)
            ),
            targets=list(campaign.scope_targets),
            budget_owner="original_campaign_and_RuntimeAgentJournal",
            disease_votes=0,
        )
    )
    result = dict(input_scope_plan_checksum=digest(base.input_scope_plan), input_scopes=[])
    calls = []
    monkeypatch.setattr(input_scope, "read_results", lambda *a, **kw: calls.append("read") or [])
    restore_scope_context(campaign, result)
    assert calls == ["read"]
    for key, value in (
        ("targets", []),
        ("disease_votes", 1),
        ("provider_checksum", digest("other")),
    ):
        previous = deepcopy(campaign.protocol)
        campaign.protocol["input_scope"][key] = value
        with pytest.raises(ValueError, match="saved_report_scope_plan_binding"):
            restore_scope_context(campaign, result)
        assert calls == ["read"]
        campaign.protocol = previous
