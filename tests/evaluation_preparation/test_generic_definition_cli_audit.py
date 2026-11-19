"""CLI-boundary fixtures; real persistence is covered by the separate fold PG gate."""

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from packages.evaluation_preparation import generic_definition_aggregation as ga
from packages.evaluation_preparation.generic_definition_successor import DefinitionReview
from packages.multimodal_research.report_storage import pack


def test_plan_is_idempotent_exact_and_disjoint(monkeypatch, tmp_path):
    campaign = object.__new__(DefinitionReview)
    campaign.spec, campaign.review_key = dict(key="old"), "new"
    monkeypatch.setattr(ga, "restore", lambda _: (campaign, {}))
    monkeypatch.setattr(ga, "directory", lambda key: tmp_path / key)
    target = tmp_path / "fold-plan.json"
    first = ga.prepare_plan("new", target)
    assert ga.prepare_plan("new", target) == first
    raw = json.loads(target.read_bytes())
    assert [i["campaign"] for i in raw["origins"]] == [str(tmp_path / k) for k in ("old", "new")]
    assert first["new_api_requests"] == 0
    with pytest.raises(ValueError, match="overlaps_origin"):
        ga.prepare_plan("new", tmp_path / "old" / "plan.json")


@pytest.mark.parametrize("fault", [None, "html", "retained", "changed", "ids", "requests"])
def test_audit_saved_html_and_exact_review_replacement(monkeypatch, tmp_path, fault):
    from agents.expert_agent import reporting
    from packages.evaluation_preparation import cohort_recovery
    from packages.multimodal_research import accepted_aggregation

    before = [dict(id="a", judgment=dict(value=0.2)), dict(id="b", judgment=dict(value=None))]
    changed = [dict(id="b", judgment=dict(value=-0.03))]
    after = deepcopy([before[0], changed[0]])
    if fault == "retained":
        after[0]["judgment"]["value"] = 0.9
    if fault == "changed":
        after[1]["judgment"]["value"] = 0.8
    if fault == "ids":
        after.append(dict(id="extra", judgment=dict(value=0.1)))
    result = dict(
        definition_revision_policy="owner_adopted_definition_revision/1.0.0",
        new_requests=int(fault == "requests"),
        cumulative_requests=12,
    )
    projection = dict(interpretation_revisions=[dict(assessment_id="b")])
    detail = dict(scores=dict(reviews=after))
    report = dict(
        result=result,
        scores=detail["scores"],
        original_review=detail,
        review_scope=dict(kind="whole_case_definition_reaggregation"),
        definition_revision=dict(definition_projection=projection),
        requirements=[{}, {}],
    )
    saved = pack(report, "<html>saved</html>", key="accepted-aggregation-report")
    docs = SimpleNamespace(
        read=lambda key: saved if key == "accepted-aggregation-report" else (projection,)
    )
    campaign = object.__new__(DefinitionReview)
    campaign.aggregation_baseline = SimpleNamespace(documents=docs, state="state")
    campaign.acquisition = SimpleNamespace(output=tmp_path / "old")
    campaign.output = tmp_path / "new"
    for output, rows in [(campaign.acquisition.output, before), (campaign.output, changed)]:
        (output / "review").mkdir(parents=True)
        (output / "review/review-report.json").write_text(
            json.dumps(dict(scores=dict(reviews=rows)))
        )
    report_path, tree_path = tmp_path / "case.json", tmp_path / "tree.html"
    report_path.write_text(json.dumps(report))
    report_path.with_suffix(".html").write_text("bad" if fault == "html" else "<html>saved</html>")
    tree_path.write_text("tree")
    monkeypatch.setattr(accepted_aggregation, "restore_aggregation", lambda _: (campaign, result))
    monkeypatch.setattr(reporting, "read_research_report", lambda *a, **kw: detail)
    monkeypatch.setattr(cohort_recovery, "check_tree", lambda r, p: r == report and p == "tree")
    if fault:
        with pytest.raises(ValueError, match="saved_bundle_mismatch|exact_judgment_selection"):
            ga.audit_wholecase(tmp_path / "fold", report_path, tree_path)
    else:
        audited = ga.audit_wholecase(tmp_path / "fold", report_path, tree_path)
        assert audited["reviews"] == 2 and audited["revised_ids"] == ["b"]
        assert audited["retained_reviews"] == 1 and audited["new_requests"] == 0


@pytest.mark.parametrize("operation", ["definition-aggregate-plan", "definition-audit"])
def test_public_cli_dispatches_to_definition_owner(monkeypatch, capsys, operation):
    import sys

    from apps import fairvlmed_case

    calls = []
    if operation == "definition-aggregate-plan":
        argv = [operation, "--key", "new", "--output", "plan.json"]
        monkeypatch.setattr(
            ga, "prepare_plan", lambda *args: calls.append(args) or {"new_api_requests": 0}
        )
        expected = ("new", "plan.json")
    else:
        argv = [operation, "--input", "fold", "--report", "case.json", "--tree", "tree.html"]
        monkeypatch.setattr(
            ga, "audit_wholecase", lambda *args: calls.append(args) or {"new_requests": 0}
        )
        expected = ("fold", "case.json", "tree.html")
    monkeypatch.setattr(sys, "argv", ["fairvlmed_case", *argv])
    assert fairvlmed_case.main() == 0
    assert calls == [expected]
    assert 0 in json.loads(capsys.readouterr().out).values()
