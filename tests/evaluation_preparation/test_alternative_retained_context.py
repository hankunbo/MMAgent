"""Alternative acquisition does not hide obtained facts outside its Review."""

from copy import deepcopy

import pytest

from packages.evaluation_preparation.report_production import produce
from packages.grounding.models import digest
from packages.multimodal_research.report import attach_retained_context, present_retained_context
from tests.evaluation_preparation.test_context_report_revision import recovery_fixture
from tests.evaluation_preparation.test_retained_context_report import fixture


def alternative(tmp_path, version="1.60.0"):
    c, report, parent, omitted = fixture(tmp_path)
    c.protocol["schema_version"] = "multimodal-campaign-renewal/" + version
    c.report_projection = dict(
        schema_version="retained-context-report-projection/1.2.0",
        acquisition_observations_checksum=digest(
            {k: v.model_dump(mode="json") for k, v in c.base.observations.items()}
        ),
        scores_checksum=digest(report["scores"]),
    )
    return c, report, parent, omitted


@pytest.mark.parametrize("version", ["1.59.0", "1.60.0"])
def test_explicit_projection_counts_preserved_facts_without_new_score(tmp_path, version):
    c, report, parent, omitted = alternative(tmp_path, version)
    scores = deepcopy(report["scores"])
    reviewed = {r["id"] for r in scores["reviews"]}
    c.protocol["reuse_assessment_ids"] = sorted(reviewed)
    attach_retained_context(c, report, parent)
    output = produce(report, parent)
    present_retained_context(output)
    row = next(r for r in output["requirements"] if r["id"] == omitted)
    assert output["counts"]["obtained_observations"] == len(parent.policy.items) + 1
    assert output["counts"]["retained_unscored_observations"] == 1
    assert not output["completion"]["all_obtained_reviewed"]
    assert output["scores"] == scores
    assert output["retained_review_context"]["reused_review_ids"] == sorted(reviewed)
    assert output["retained_review_context"]["new_review_ids"] == []
    assert row["presentation_sections"]["observation"] == row["unscored_observation"]
    assert row["presentation_sections"]["score_pointer"] is None


@pytest.mark.parametrize("change", ["score", "fact", "version", "missing_target"])
def test_projection_changed_binding_refused(tmp_path, change):
    c, report, parent, omitted = alternative(tmp_path)
    if change == "score":
        report["scores"]["changed"] = True
    elif change == "fact":
        old = c.base.observations[omitted]
        c.base.observations[omitted] = old.model_copy(update={"value": "changed"})
    elif change == "version":
        c.report_projection["schema_version"] = "retained-context-report-projection/1.1.0"
    else:
        report["requirements"] = [r for r in report["requirements"] if r["id"] != omitted]
    with pytest.raises(ValueError, match="report_retained_context"):
        attach_retained_context(c, report, parent)


def test_existing_projection_port_appends_separate_report_and_cold_reads_exact(
    tmp_path, monkeypatch
):
    from packages.multimodal_research import report, report_recovery

    path, c, values, calls = recovery_fixture(tmp_path, monkeypatch)
    c.protocol["schema_version"] = "multimodal-campaign-renewal/1.60.0"
    saved = values["dynamic-case-report"][0]
    saved["acquisition_observations"] = {}
    # Original committed scores and detailed Review remain intact.
    values["dynamic-case-report"][1]["report_checksum"] = digest(saved)
    monkeypatch.setattr(report, "acquired_observations", lambda _: {})
    before = deepcopy(values)
    first = report_recovery.restore_context_report(path)
    second = report_recovery.restore_context_report(path)
    assert calls == ["retained-context-report-v3"]
    assert first["new_api_requests"] == first["new_scores"] == 0
    assert first["report_revision_created"] and not second["report_revision_created"]
    assert first["report_checksum"] == second["report_checksum"]
    assert first["report_projection"]["scores_checksum"] == digest(saved["scores"])
    assert all(values[k] == v for k, v in before.items())
