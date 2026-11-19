"""Explicit zero-request producer revision, legacy bytes and votes preserved."""

from contextlib import nullcontext
from copy import deepcopy

import pytest

from packages.evaluation_preparation.report_production import produce
from packages.grounding.models import digest
from packages.multimodal_research.report import attach_retained_context, present_retained_context
from tests.evaluation_preparation.test_retained_context_report import fixture


def revised_fixture(tmp_path):
    campaign, report, parent, omitted = fixture(tmp_path)
    campaign.protocol["schema_version"] = "multimodal-campaign-renewal/1.45.0"
    del campaign.protocol["review_context_scope"]
    report["counts"]["obtained_observations"] = len(parent.policy.items)
    campaign.report_projection = dict(
        schema_version="retained-context-report-projection/1.0.0",
        original_report_checksum=digest(report),
    )
    return campaign, report, parent, omitted


def test_method_legacy_export_not_silently_upgraded(tmp_path):
    campaign, report, parent, _ = revised_fixture(tmp_path)
    del campaign.report_projection
    original = deepcopy(report)
    attach_retained_context(campaign, report, parent)
    assert report == original


def test_explicit_revision_counts_facts_and_binds_six_sections_without_votes(tmp_path):
    campaign, report, parent, omitted = revised_fixture(tmp_path)
    scores = deepcopy(report["scores"])
    attach_retained_context(campaign, report, parent)
    output = produce(report, parent)
    present_retained_context(output)
    row = next(r for r in output["requirements"] if r["id"] == omitted)
    assert row["presentation_sections"]["observation"] == row["unscored_observation"]
    assert row["presentation_sections"]["score_pointer"] is None
    assert output["counts"]["obtained_observations"] == len(parent.policy.items) + 1
    assert output["counts"]["review_scope_observations"] == len(parent.policy.items)
    assert output["completion"]["all_obtained_reviewed"] is False
    assert output["completion"]["selected_increment_reviewed"] is True
    assert output["completion"]["retained_context_not_newly_reviewed_ids"] == [omitted]
    assert output["scores"] == scores


@pytest.mark.parametrize("change", ["version", "scope", "fact", "missing_target"])
def test_projection_rejects_wrong_current_binding(tmp_path, change):
    campaign, report, parent, omitted = revised_fixture(tmp_path)
    if change == "version":
        campaign.protocol["schema_version"] = "multimodal-campaign-renewal/1.22.0"
    elif change == "scope":
        campaign.base.review_context_scope = "caller_claim"
    elif change == "fact":
        old = campaign.base.observations[omitted]
        campaign.base.observations[omitted] = old.model_copy(update={"value": "changed"})
    else:
        report["requirements"] = [r for r in report["requirements"] if r["id"] != omitted]
    with pytest.raises(ValueError, match="report_retained_context"):
        attach_retained_context(campaign, report, parent)


def recovery_fixture(tmp_path, monkeypatch):
    from agents.expert_agent import committed_review
    from packages.multimodal_research import campaign_renewal, execution_lease, report, stages
    from tests.evaluation_preparation.test_method_report_recovery import fixture as stored

    path, campaign, values, page = stored(tmp_path, monkeypatch)
    campaign.protocol["schema_version"] = "multimodal-campaign-renewal/1.45.0"
    campaign.config = object()
    values["dynamic-phase-input"] = ({"original_owner": True},)
    result = values["multimodal-case-result"][0]
    monkeypatch.setattr(campaign_renewal, "completed_parent", lambda _: (campaign, result))
    monkeypatch.setattr(execution_lease, "session", lambda _: nullcontext())
    monkeypatch.setattr(stages, "local_current", lambda *args: None)
    monkeypatch.setattr(
        committed_review,
        "read_committed_review",
        lambda *args, **kw: {"acceptance_key": "accepted"},
    )
    calls = []

    def produce(current, saved_result):
        calls.append(current.report_key)
        revised = dict(
            values["dynamic-case-report"][0], report_projection=current.report_projection
        )
        values[current.report_key] = (revised, dict(html=page, report_checksum=digest(revised)))
        return report.restore_export(current, saved_result)

    monkeypatch.setattr(report, "export", produce)
    return path, campaign, values, calls


def test_recovery_port_exact_repeat_no_rerender_or_old_write(tmp_path, monkeypatch):
    from packages.multimodal_research.report_recovery import restore_context_report

    path, _, values, calls = recovery_fixture(tmp_path, monkeypatch)
    original = deepcopy(values)
    first = restore_context_report(path)
    second = restore_context_report(path)
    assert first["report_revision_created"] and not second["report_revision_created"]
    assert first["report_checksum"] == second["report_checksum"]
    assert first["new_api_requests"] == first["new_scores"] == 0
    assert calls == ["retained-context-report-v1"]
    assert all(values[k] == v for k, v in original.items())


@pytest.mark.parametrize("fault", ["source", "acceptance", "result", "score", "version"])
def test_recovery_port_refuses_before_any_new_report(tmp_path, monkeypatch, fault):
    from agents.expert_agent import committed_review
    from packages.multimodal_research import report_recovery, stages

    path, campaign, values, calls = recovery_fixture(tmp_path, monkeypatch)
    if fault == "source":

        def changed(*args):
            raise ValueError("current_source_changed")

        monkeypatch.setattr(stages, "local_current", changed)
    elif fault == "acceptance":
        monkeypatch.setattr(committed_review, "read_committed_review", lambda *a, **kw: None)
    elif fault == "result":
        values["dynamic-case-report"][0]["result"] = {"wrong": True}
    elif fault == "score":
        values["dynamic-case-report"][0]["scores"] = {"wrong": True}
    else:
        campaign.protocol["schema_version"] = "multimodal-campaign-renewal/1.22.0"
    with pytest.raises(ValueError):
        report_recovery.restore_context_report(path)
    assert not calls and "retained-context-report-v1" not in values


def unscored_fixture(tmp_path):
    campaign, report, parent, omitted = revised_fixture(tmp_path)
    campaign.protocol["schema_version"] = "multimodal-campaign-renewal/1.52.0"
    campaign.protocol["reuse_assessment_ids"] = []
    campaign.protocol["parent_inventory"] = {"quarantined": [omitted]}
    campaign.report_projection.update(
        schema_version="retained-context-report-projection/1.1.0",
        acquisition_observations_checksum=digest(
            {k: v.model_dump(mode="json") for k, v in campaign.base.observations.items()}
        ),
    )
    # The unreviewed tool result need not have been visible to the new model.
    observation = campaign.base.observations[omitted]
    parent = parent.model_copy(
        update={
            "material": parent.material.model_copy(
                update={
                    "observations": tuple(
                        o
                        for o in parent.material.observations
                        if o.reference != observation.reference
                    )
                }
            )
        }
    )
    return campaign, report, parent, omitted


def test_unscored_parent_counts_quarantined_fact_without_claiming_review_visibility(tmp_path):
    campaign, report, parent, omitted = unscored_fixture(tmp_path)
    before = digest(report["scores"]), digest(parent)
    attach_retained_context(campaign, report, parent)
    output = produce(report, parent)
    present_retained_context(output)
    row = next(r for r in output["requirements"] if r["id"] == omitted)
    assert row["unscored_reason"] == "historical_unknown_review_quarantined"
    assert row["presentation_sections"]["acquisition"]["missing_reason"] is None
    assert row["presentation_sections"]["observation"] == row["unscored_observation"]
    assert row["presentation_sections"]["score_pointer"] is None
    assert output["retained_review_context"]["visible_in_current_review"][omitted] is False
    assert output["counts"]["obtained_observations"] == len(parent.policy.items) + 1
    assert output["counts"]["retained_unscored_observations"] == 1
    assert not output["completion"]["all_obtained_reviewed"]
    assert output["completion"]["selected_increment_reviewed"]
    assert output["review_scope"]["whole_case_rescored"] is False
    assert before == (digest(output["scores"]), digest(parent))


@pytest.mark.parametrize("fault", ["fact", "target", "selected", "version"])
def test_unscored_projection_rejects_changed_saved_fact(tmp_path, fault):
    campaign, report, parent, omitted = unscored_fixture(tmp_path)
    if fault == "fact":
        old = campaign.base.observations[omitted]
        campaign.base.observations[omitted] = old.model_copy(update={"value": "changed"})
    elif fault == "target":
        report["requirements"] = [r for r in report["requirements"] if r["id"] != omitted]
    elif fault == "selected":
        next(r for r in report["requirements"] if r["id"] != omitted)["observation"] = None
    else:
        campaign.report_projection["schema_version"] = "retained-context-report-projection/1.0.0"
    with pytest.raises(ValueError, match="report_retained_context"):
        attach_retained_context(campaign, report, parent)


@pytest.mark.parametrize("changed", [False, True])
def test_unscored_restore_port_exact_fact_and_cold_repeat(tmp_path, monkeypatch, changed):
    from packages.multimodal_research import report, report_recovery

    path, campaign, values, calls = recovery_fixture(tmp_path, monkeypatch)
    campaign.protocol["schema_version"] = "multimodal-campaign-renewal/1.52.0"
    campaign.base.observations = {}
    monkeypatch.setattr(report, "acquired_observations", lambda c: {})
    values["dynamic-case-report"][0]["acquisition_observations"] = (
        {"changed": True} if changed else {}
    )
    values["dynamic-case-report"][1]["report_checksum"] = digest(values["dynamic-case-report"][0])
    before = deepcopy(values)
    if changed:
        with pytest.raises(ValueError, match="acquisition_changed"):
            report_recovery.restore_context_report(path)
        assert values == before and not calls
    else:
        first = report_recovery.restore_context_report(path)
        second = report_recovery.restore_context_report(path)
        assert first["report_revision_created"] and not second["report_revision_created"]
        assert first["report_checksum"] == second["report_checksum"]
        assert first["new_api_requests"] == first["new_scores"] == 0
        assert calls == ["retained-context-report-v2"]
        assert all(values[k] == v for k, v in before.items())
