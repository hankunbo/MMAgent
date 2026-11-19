"""Budget exhaustion is not UNKNOWN, nor authorization to replay old batches."""

from copy import deepcopy

import pytest

from packages.multimodal_research import confirmed_review_window as window
from tests.evaluation_preparation.test_confirmed_review_window import origin


def budget_origin(tmp_path, monkeypatch):
    f, parent, result, tip, now, journal, checks = origin(tmp_path, monkeypatch)
    docs = f.old.documents
    phase = result["phases"][0]
    phase["error"] = "candidate_failed_or_context_changed"
    phase["acceptance_key"] = None
    docs.saved["dynamic-phase-result"] = (deepcopy(phase),)
    docs.saved["multimodal-case-result"] = (deepcopy(result),)
    parent.protocol["max_model_dispatches"] = 2
    key = "review:" + docs.read("live-stage-intent:review")[0]["event"]["code"]
    docs.write(
        key + ":candidate-failure",
        (
            dict(
                schema_version="research-candidate-failure/1.0.0",
                category="provider",
                code="budget",
                fields=[],
            ),
        ),
    )
    return f, parent, result, tip, now, journal, key


def test_explicit_budget_successor_preserves_confirmed_and_old_contract(tmp_path, monkeypatch):
    f, parent, result, tip, now, _, _ = budget_origin(tmp_path, monkeypatch)
    before = deepcopy(f.old.documents.saved), deepcopy(f.old.budget.journal.read())
    with pytest.raises(ValueError, match="not_expired_partial"):
        window.origin_proof(parent, result, tip, now=now)
    proof = window.origin_proof(parent, result, tip, now=now, budget_stop=True)
    assert proof["budget_stop"]["confirmed_requests"] == 2
    assert len(proof["confirmed_assessment_ids"]) == 2
    plan = window.plan_for(
        parent, result, tip, proof, output=tmp_path / "new", authorization="test", now=now
    )
    assert plan["schema_version"] == window.BUDGET_VERSION
    assert plan["max_new_requests"] == 5 and plan["max_model_dispatches"] == 4
    assert plan["reuse_confirmed_review"] and plan["new_vision_requests"] == 0
    assert before == (f.old.documents.saved, f.old.budget.journal.read())


@pytest.mark.parametrize(
    "fault", ["missing", "timeout", "category", "unused_budget", "unknown", "cancel"]
)
def test_budget_stop_requires_original_failure_and_confirmed_ledger(tmp_path, monkeypatch, fault):
    f, parent, result, tip, now, journal, key = budget_origin(tmp_path, monkeypatch)
    docs = f.old.documents
    if fault == "missing":
        docs.saved.pop(key + ":candidate-failure")
    elif fault in {"timeout", "category"}:
        docs.saved[key + ":candidate-failure"][0]["code" if fault == "timeout" else "category"] = (
            fault
        )
    elif fault == "unused_budget":
        parent.protocol["max_model_dispatches"] = 3
    elif fault == "unknown":
        journal.append(dict(phase="reserved", request_id="unknown"))
    else:
        (tmp_path / "cancel").touch()
    with pytest.raises(ValueError):
        window.origin_proof(parent, result, tip, now=now, budget_stop=True)


def test_cli_routes_budget_stop_explicitly(monkeypatch, tmp_path):
    import sys

    from apps import multimodal_case

    calls = []
    monkeypatch.setattr(window, "prepare", lambda **kwargs: calls.append(kwargs) or {})
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "multimodal_case",
            "renewal-prepare",
            "--facts",
            str(tmp_path),
            "--previous",
            str(tmp_path),
            "--output",
            str(tmp_path / "next"),
            "--mode",
            "review",
            "--authorization",
            "test",
            "--budget-stop-review",
        ],
    )
    assert multimodal_case.main() == 0
    assert len(calls) == 1 and calls[0]["budget_stop_review"] is True


def test_budget_and_deadline_modes_cannot_be_combined(tmp_path):
    from packages.multimodal_research.campaign_renewal import prepare

    with pytest.raises(ValueError, match="explicit_text_only_scope"):
        prepare(
            facts=tmp_path,
            previous=tmp_path,
            output=tmp_path / "new",
            mode="review",
            authorization="test",
            budget_stop_review=True,
            same_profile_review=True,
        )
