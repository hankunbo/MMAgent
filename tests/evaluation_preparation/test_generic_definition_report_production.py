"""Original definition projection through report production, without new scoring."""

import copy
from types import SimpleNamespace

import pytest

from packages.evaluation_preparation.evidence_tree_v3 import build, render
from packages.evaluation_preparation.generic_definition_projection import project_interpretations
from packages.evaluation_preparation.generic_definition_successor import (
    DefinitionReview,
    report_metadata,
)
from packages.evaluation_preparation.report_production import produce
from packages.grounding.models import digest
from tests.evaluation_preparation.test_generic_definition_stage import setup


def preparation(monkeypatch, tmp_path, *, adopted=True, whole=False):
    f = setup(monkeypatch, tmp_path)
    if adopted:
        factory = f.factory
    else:
        from packages.llm_provider.provider import DeterministicFakeProvider

        response = dict(
            schema_version="research-definition-proposal/6.0.0",
            definitions=[],
            no_addition_reason="No source relation applies to the measured region",
        )

        def factory(config):
            return DeterministicFakeProvider(config, [response])

    f.stage.advance(provider_factory=factory)
    keys = f.stage.review_keys()
    campaign = object.__new__(DefinitionReview)
    campaign.definition_stage = f.stage
    campaign.checked = SimpleNamespace(
        history=dict(result_checksum=digest("old-result"), report_checksum=digest("old-report"))
    )
    campaign.documents = lambda _: f.documents
    campaign.definition_keys = keys
    if adopted:
        state, manifest = project_interpretations(
            f.state, f.stage.expansion.owner, keys, selected_only=not whole
        )
        f.documents.write("generic-definition-review-projection", (manifest,))
    else:
        from packages.evaluation_preparation.dynamic_stages import rebuild

        state = rebuild(
            f.state, f.state.policy.model_copy(update=dict(items=())), f.state.material.observations
        )
    revision = report_metadata(campaign, state, dict(new_requests=1))
    if whole:
        revision = dict(
            schema_version="generic-definition-wholecase-report-scope/1.0.0",
            definition_projection=manifest,
            historical_scores_modified=False,
        )
    report = dict(
        schema_version="dynamic-case-report/1.0.0",
        result=dict(case_code=state.material.scope.case_code),
        original_review=None,
        scores=None,
        definition_revision=revision,
        requirements=[
            dict(id=r.id, sources=[s.model_dump(mode="json") for s in r.definition.source_refs])
            for r in state.policy.requirements
        ],
    )
    return state, report


@pytest.mark.parametrize("whole", [False, True])
def test_owner_projection_retained_in_six_sections_and_tree(monkeypatch, tmp_path, whole):
    state, report = preparation(monkeypatch, tmp_path, whole=whole)
    before = copy.deepcopy(report)
    produced = produce(report, state)
    manifest = report["definition_revision"]["definition_projection"]
    row = next(r for r in produced["requirements"] if r["id"] == manifest["selected_ids"][0])
    binding = row["presentation_sections"]["rationale"]["accepted_definition_revision"]
    assert (
        binding["receipt_checksum"] == manifest["interpretation_revisions"][0]["receipt_checksum"]
    )
    assert binding["current_definition"] == state.policy.requirements[0].definition.model_dump(
        mode="json"
    )
    assert binding["denominator_before"] == manifest["requirements_before"]
    assert binding["denominator_after"] == manifest["requirements_after"]
    assert binding["added_evidence_votes"] == 0
    assert report == before and produced["scores"] is None
    packet = build(produced, input_checksum=digest(produced))
    assert packet["confidence"] is None and packet["checks"]
    assert "evidence-view" in render(produced, input_checksum=digest(produced))


def test_no_adopted_definition_is_reported_without_new_review_or_score(monkeypatch, tmp_path):
    state, report = preparation(monkeypatch, tmp_path, adopted=False)
    original = copy.deepcopy(report)
    produced = produce(report, state)
    assert produced["definition_revision"]["status"] == "no_adopted_definition"
    assert produced["definition_revision"]["definition_projection"] is None
    assert all(
        "accepted_definition_revision" not in r["presentation_sections"]["rationale"]
        for r in produced["requirements"]
    )
    assert report == original and produced["scores"] is None
    assert build(produced, input_checksum=digest(produced))["confidence"] is None


@pytest.mark.parametrize(
    "fault",
    [
        "version",
        "state",
        "definition",
        "observation",
        "method",
        "duplicate",
        "selection",
        "count",
        "mode",
    ],
)
def test_changed_projection_is_not_silently_treated_as_legacy(monkeypatch, tmp_path, fault):
    state, report = preparation(monkeypatch, tmp_path)
    scope = report["definition_revision"]
    projection = scope["definition_projection"]
    revision = projection["interpretation_revisions"][0]
    if fault == "version":
        scope["schema_version"] = "generic-definition-review-report-scope/99.0.0"
    elif fault == "state":
        projection["projected_state_checksum"] = digest("another")
    elif fault == "definition":
        revision["current_definition"]["interpretation"] = "Changed interpretation"
    elif fault == "observation":
        revision["observation"]["checksum"] = digest("different observation")
    elif fault == "method":
        revision["method"] = "different method"
    elif fault == "duplicate":
        projection["interpretation_revisions"].append(copy.deepcopy(revision))
    elif fault == "selection":
        scope["interpretation_ids"] = []
    elif fault == "count":
        projection["requirements_after"] += 1
    else:
        projection["selected_only"] = False
    with pytest.raises(ValueError, match="report_definition_"):
        produce(report, state)


def test_no_adoption_cannot_hide_nonempty_current_review(monkeypatch, tmp_path):
    state, report = preparation(monkeypatch, tmp_path)
    report["definition_revision"].update(
        definition_projection=None,
        interpretation_ids=[],
        whole_case_aggregation_required=False,
        status="no_adopted_definition",
    )
    with pytest.raises(ValueError, match="report_definition_"):
        produce(report, state)
