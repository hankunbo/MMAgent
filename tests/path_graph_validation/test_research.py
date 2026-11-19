import copy

import pytest

from packages.grounding.models import digest, validate
from packages.path_graph_validation import render_path_graph_html
from tests.path_graph_validation.support import accepted


def test_accepted_scope_report_and_original_scores_are_exact():
    h = accepted()
    history = copy.deepcopy(h.f.documents.saved)
    report = h.service.inspect(key="summary")
    assert report["goals"][0]["requirements_closed"]
    assert report["coverage"] == report["original_report"]["scores"]["root"]["coverage"]
    assert not report["clinical_acceptance"]
    assert report["original_report"]["graph"]["accepted_validations"] == []
    assert report["evidence"][0]["clinical_effect"] == "NONE"
    assert h.service.read(key="summary") == report == h.service.inspect(key="summary")
    assert h.f.budget.used == 1
    assert all(h.f.documents.saved[k] == v for k, v in history.items())
    assert h.f.documents.audit.count("path-graph:summary") == 1
    assert report["checksum"] in render_path_graph_html(report)


@pytest.mark.parametrize("status", ["unknown", "missing", "ungradable", "reference_missing"])
def test_unknown_and_missing_reference_never_fill_interpretation(status):
    h = accepted(statuses=(status,))
    report = h.service.inspect(key="summary")
    assert not report["goals"][0]["requirements_closed"]
    assert report["review_facts"]["uncertain"]
    assert report["evidence"][0]["judgment"]["value"] is None
    assert report["unresolved"][0]["reason"] == (
        "input_missing"
        if status in {"unknown", "missing"}
        else "observation_ungradable"
        if status == "ungradable"
        else status
    )
    coverage = report["coverage"]
    assert coverage == report["original_report"]["scores"]["root"]["coverage"]
    for row in coverage.values():
        assert row["interpretation"] == 0
        assert row["acquisition"] == (1 if status == "reference_missing" else 0)
    assert report["paths"][0]["available"] and not report["paths"][0]["interpreted"]


def test_risk_dominance_cannot_close_missing_target_or_remove_age():
    h = accepted(
        statuses=("obtained", "missing"),
        roles=("risk", "visual"),
        targets=("GOAL:RISK", "GOAL:SYNTHETIC"),
    )
    r = h.reader()
    assert r.risk["background_dominant"] and not r.risk["automatic_removal"]
    assert any(not g["requirements_closed"] for g in r.goals)
    assert r.original_report["scores"]["root"]["summary"]["confidence"]["value"] > 0.5
    assert not r.clinical_acceptance and not r.review_facts["high_risk"]


def test_same_source_multiple_targets_bound_without_new_independent_weight():
    h = accepted(statuses=("obtained", "obtained"), targets=("GOAL:SYNTHETIC", "GOAL:SECOND"))
    r = h.reader()
    assert len(r.duplicate_groups) == 1
    assert {e["target"] for e in r.evidence} == {"GOAL:SYNTHETIC", "GOAL:SECOND"}
    assert r.original_report["scores"]["root"]["summary"]["combined"]["active_groups"] == 1


def test_opposed_same_target_triggers_required_expert():
    r = accepted(statuses=("obtained", "obtained"), directions=(0.6, -0.6)).reader()
    assert r.review_facts["conflict"] and r.review_facts["required"]
    assert r.relations[0]["target"] == "GOAL:SYNTHETIC"


@pytest.mark.parametrize("change", ["withdrawn", "reset", "policy", "foreign", "source"])
def test_current_owner_rejection_does_not_rewrite_history(change):
    h = accepted()
    saved = h.service.inspect(key="summary")
    state = h.f.reader.state
    if change == "withdrawn":
        h.f.reader.withdrawn = True
    elif change == "reset":
        h.f.reader.state = state.model_copy(update={"nodes": ("GOAL:SYNTHETIC", "GOAL:RESET")})
    elif change == "policy":
        h.f.reader.state = state.model_copy(update={"config_checksum": digest("changed")})
    else:
        raw = state.material.model_dump(mode="json")
        if change == "foreign":
            raw["scope"]["run_id"] += "OTHER"
        else:
            raw["sources"][0]["reference"]["version"] = "9.0.0"
        raw["scope"].pop("material_checksum")
        raw["scope"]["material_checksum"] = digest(raw)
        from packages.llm_provider.models import Material

        h.f.reader.state = state.model_copy(update={"material": validate(Material, raw)})
    with pytest.raises(ValueError):
        h.service.read(key="summary")
    assert h.service.read(key="summary", historical=True) == saved
    assert h.f.budget.used == 1


@pytest.mark.parametrize("what", ["receipt", "candidate", "score", "binding", "html"])
def test_forged_reports_and_candidates_rejected(what):
    h = accepted()
    h.service.inspect(key="summary")
    if what == "html":
        h.f.documents.saved["path-graph:summary"][1]["html"] += "forged"
        with pytest.raises(ValueError, match="html_binding"):
            h.service.read(key="summary", historical=True)
        return
    if what == "binding":
        r = h.f.documents.saved["path-graph:summary"][0]
        r["graph"]["checksum"] = digest("forged")
        with pytest.raises(ValueError, match="checksum"):
            render_path_graph_html(r)
        return
    docs = h.f.documents.saved["accepted:path-source"]
    if what == "receipt":
        docs[-1]["status"] = "rejected"
    elif what == "candidate":
        h.f.documents.saved["submission:path-source"][0]["submission"]["request_id"] = "forged"
    else:
        docs[2]["scores"]["root"]["summary"]["combined"]["support"] = 1.0
    with pytest.raises(ValueError):
        h.reader()


@pytest.mark.parametrize("failure", ["lost_reply", "transaction", "object"])
def test_report_failure_and_original_key_recovery(failure):
    h = accepted()
    h.f.documents.fail = failure
    with pytest.raises((ValueError, OSError)):
        h.service.inspect(key="summary")
    assert ("path-graph:summary" in h.f.documents.saved) == (failure == "lost_reply")
    h.f.documents.fail = None
    result = h.service.inspect(key="summary")
    assert h.service.read(key="summary") == result
    assert h.f.documents.audit.count("path-graph:summary") == 1
    assert h.f.budget.used == 1


def test_declared_coverage_slot_or_alternative_closes_research_requirement():
    h = accepted(statuses=("obtained", "unknown"), alternative_slot=True)
    r = h.reader()
    assert r.goals[0]["requirements_closed"]
    assert r.goals[0]["required_slots"][0]["interpreted"]
    assert not r.review_facts["conflict"] and not r.review_facts["uncertain"]
    assert not r.unresolved[0]["required_for_coverage"]
    assert all(v["interpretation"] == 1 for v in r.coverage.values())


def test_shared_product_actions_remain_target_bound():
    r = accepted(
        statuses=("obtained", "obtained"),
        targets=("GOAL:FIRST", "GOAL:SECOND"),
        shared_product=True,
    ).reader()
    assert r.evidence[0]["observation"] == r.evidence[1]["observation"]
    assert r.evidence[0]["target"] != r.evidence[1]["target"]
    assert len(r.duplicate_groups) == 1


def test_rehashed_projection_cannot_forge_closed_goal_or_coverage():
    h = accepted(statuses=("unknown",))
    r = h.service.inspect(key="summary")
    r["goals"][0]["requirements_closed"] = True
    r["checksum"] = digest({k: v for k, v in r.items() if k != "checksum"})
    with pytest.raises(ValueError, match="projection_binding"):
        render_path_graph_html(r)


def test_source_withdrawn_during_commit_preserves_facts_but_refuses_current_read(monkeypatch):
    h = accepted()
    write = h.f.documents.write

    def changed(key, values):
        write(key, values)
        h.f.reader.withdrawn = True

    monkeypatch.setattr(h.f.documents, "write", changed)
    with pytest.raises(ValueError, match="source_withdrawn"):
        h.service.inspect(key="summary")
    assert (
        h.service.read(key="summary", historical=True)["original_report"]["receipt"]["status"]
        == "accepted"
    )
    assert any("source_revalidation_failed" in a for a in h.f.documents.audit)
    assert h.f.budget.used == 1


def test_submitted_but_unaccepted_and_research_identity_upgrade_rejected():
    h = accepted()
    del h.f.documents.saved["accepted:path-source"]
    with pytest.raises(ValueError, match="research_report_missing"):
        h.reader()
    h = accepted()
    graph = h.parent.material.scope.graph.model_copy(update={"kind": "formal_graph"})
    scope = h.parent.material.scope.model_copy(update={"graph": graph})
    h.f.reader.state = h.parent.model_copy(
        update={"material": h.parent.material.model_copy(update={"scope": scope})}
    )
    with pytest.raises(ValueError):
        h.reader()
