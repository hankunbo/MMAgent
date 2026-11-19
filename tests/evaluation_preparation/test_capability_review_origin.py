"""Accepted origin extension cannot change answers, frozen plans or budgets."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from packages.multimodal_research import capability_review_origin as repair
from packages.multimodal_research import capability_window as window
from tests.evaluation_preparation.test_capability_window import plan_fixture


def fixture(tmp_path):
    parent, tip, inv, kwargs = plan_fixture(tmp_path)
    plan = window.plan_for(parent, tip, inv, **kwargs)
    found = deepcopy(inv)
    found["confirmed_review"] = {"assessment_ids": ["one", "two"]}
    implementation = {"packages/multimodal_research/capability_window.py": "new"}
    return plan, found, implementation


def test_extension_preserves_original_answers_and_rejects_conflicting_merge():
    one = (object(), {"run_id": "old"}, {"one": {"e": -0.08}})
    both = (object(), {"run_id": "recorded"}, {"one": {"e": -0.08}, "two": {"e": 0}})
    assert window.extend_review(one, both) is both
    assert window.extend_review(both, one) is both
    with pytest.raises(ValueError, match="accepted_review_conflict"):
        window.extend_review(one, (None, {}, {"one": {"e": 0.01}}))
    with pytest.raises(ValueError, match="explicit_merge"):
        window.extend_review(one, (None, {}, {"two": {"e": 0.01}}))


def test_recorded_ancestor_is_not_hidden_by_inherited_reader(monkeypatch, tmp_path):
    from packages.multimodal_research import campaign_renewal as renewal

    old, recorded = object(), object()
    ancestor = SimpleNamespace(
        protocol={"recorded_definition_origin": {"proof": {"confirmation": {"run_id": "r"}}}},
        parent=None,
    )
    parent = SimpleNamespace(
        protocol={}, base=SimpleNamespace(confirmed_review_reader=old), parent=ancestor
    )
    checked = []
    monkeypatch.setattr(renewal, "recorded_definition_check", lambda p, v: checked.append(v))
    monkeypatch.setattr(renewal, "accepted_reader", lambda *a, **kw: recorded)
    monkeypatch.setattr(
        window,
        "qualified_review",
        lambda p, n, r: (
            r,
            {"assessment_ids": ["one", "two"] if r is recorded else ["one"]},
            {"one": -0.08, "two": 0} if r is recorded else {"one": -0.08},
        ),
    )
    reader, proof = window.review_origin(parent)
    assert reader is recorded and proof["assessment_ids"] == ["one", "two"] and len(checked) == 1


def test_amendment_only_changes_review_origin_and_implementation(tmp_path):
    old, found, implementation = fixture(tmp_path)
    original = deepcopy(old)
    new = repair.revised(old, found, implementation)
    assert old == original
    assert new["reuse_assessment_ids"] == ["one", "two"]
    assert {k for k in old if old[k] != new[k]} == {
        "reuse_assessment_ids",
        "capability_transition",
        "implementation",
    }


@pytest.mark.parametrize("fault", ["no_extension", "outside", "fact", "scope", "budget_file"])
def test_unrelated_changes_and_unconfirmed_ids_refused(tmp_path, fault):
    old, found, implementation = fixture(tmp_path)
    if fault == "no_extension":
        found["confirmed_review"]["assessment_ids"] = ["one"]
    elif fault == "outside":
        found["confirmed_review"]["assessment_ids"].append("foreign")
    elif fault == "fact":
        found["confirmed_observations_checksum"] = "changed"
    elif fault == "scope":
        found["input_scope"]["targets"] = []
    else:
        implementation["packages/multimodal_research/phase_budget.py"] = "changed"
    with pytest.raises(ValueError, match="capability_review_amendment"):
        repair.revised(old, found, implementation)


@pytest.mark.parametrize("registered", ["scope", "review", "control", "cancel"])
def test_original_phase_or_cancel_prevents_amendment(tmp_path, registered):
    from pathlib import Path

    from db.runtime.repository.case_pipeline import identity
    from packages.grounding.models import digest

    old, _, _ = fixture(tmp_path)
    path = Path(old["output"])
    path.mkdir()
    if registered == "cancel":
        (path / "cancel").write_text("cancel")
    target = "scope:one" if registered == "scope" else "review"
    rid = identity("RUN", "task-research:mm-live:" + digest((path.name, target))[7:55])
    parent = SimpleNamespace(
        contexts=SimpleNamespace(
            read_research_run=lambda r, **kw: (
                object() if r == rid and registered in {"scope", "review"} else None
            )
        ),
        sessions=SimpleNamespace(
            read=lambda r: object() if r == rid and registered == "control" else None
        ),
    )
    with pytest.raises(ValueError, match="capability_review_amendment"):
        repair.verify_unused(parent, old)
