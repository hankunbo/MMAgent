"""Explicit entry, full projection, exact supersession and report scope contracts.

Synthetic owner integration here is not a real Source/Review or PostgreSQL run.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from packages.evaluation_preparation import generic_definition_aggregation as ga
from packages.evaluation_preparation import generic_definition_projection as projection
from packages.evaluation_preparation.dynamic_report import GenericAggregateReportView
from packages.evaluation_preparation.generic_definition_successor import DefinitionReview
from packages.evidence_details.accepted_review_reuse import compatible_local_ids
from packages.grounding.models import digest
from packages.multimodal_research import accepted_aggregation as ag
from tests.agent_provider.batch_support import generate
from tests.evaluation_preparation.test_generic_definition_projection import fixture


def test_original_compatibility_selects_only_exactly_unchanged_accepted_judgments():
    f, wire, seed, _ = fixture()
    submission = generate(wire)
    assert wire.acceptance.submit(submission, key="old")["status"] == "submitted_not_accepted"
    assert wire.acceptance.accept(key="old").status == "accepted"
    full, manifest = projection.project_interpretations(
        seed, f.domain, ["new-definition"], selected_only=False
    )
    ids = compatible_local_ids(seed, full, submission.candidate, include_unassessed=True)
    assert set(ids) == {i.id for i in seed.policy.items} - {f.target}
    old = seed.model_copy(
        update={
            "material": seed.material.model_copy(
                update={
                    "scope": seed.material.scope.model_copy(update={"run_id": f.current["run_id"]})
                }
            )
        }
    )
    selector = ga.DefinitionRevisionSelection(
        owner=f.domain, keys=["new-definition"], seed_reader=lambda _: seed
    )
    assert set(ids) == selector.retained_ids(old, full, submission.candidate)
    assert full.policy.groups == seed.policy.groups
    assert len(full.policy.items) == len(seed.policy.items)
    assert manifest["added_requirements"] == []


def test_full_phase_projection_is_explicit_increment_stays_default(monkeypatch):
    from packages.evaluation_preparation.generic_aliases import GenericPhase

    f, _, seed, _ = fixture()
    facts = f.owner
    base = facts.campaign.base
    monkeypatch.setattr(GenericPhase, "project_current", lambda _: seed)
    for selected in [True, False]:
        phase = object.__new__(projection.GenericDefinitionReviewPhase)
        phase.base, phase.target = base, None
        phase.facts, phase.definition_owner = facts, f.domain
        phase.definition_keys, phase.selected_only = ("new-definition",), selected
        got = phase.project_current()
        assert len(got.policy.items) == (1 if selected else len(seed.policy.items))
        assert phase.definition_revision_manifest["selected_only"] is selected
    assert (
        projection.phase_type(facts, f.domain, ["new-definition"]).keywords["selected_only"] is True
    )


def plan():
    return ag.read_plan(
        dict(
            schema_version="accepted-review-aggregation-plan/1.12.0",
            facts="/new",
            origins=[dict(campaign="/old"), dict(campaign="/new")],
        )
    )


def test_plan_dispatch_does_not_treat_definition_revision_as_pending(monkeypatch):
    p = plan()
    assert type(p) is ag.GenericDefinitionAggregationPlan
    calls = []
    monkeypatch.setattr(
        ga, "restore_plan", lambda value: calls.append(value) or ("campaign", "result", "phase")
    )
    assert ag.plan_campaign(p) == ("campaign", "result", "phase")
    assert calls == [p]
    assert (
        type(
            ag.read_plan(
                dict(
                    schema_version="accepted-review-aggregation-plan/1.2.0",
                    facts="/old",
                    origins=[dict(campaign="/old")],
                )
            )
        )
        is ag.GenericAggregationPlan
    )


@pytest.mark.parametrize("fault", [None, "reverse", "extra", "foreign", "local"])
def test_plan_requires_exact_original_and_new_successor(monkeypatch, fault):
    f, _, _, _ = fixture()
    c = object.__new__(DefinitionReview)
    c.spec, c.review_key = dict(key="old"), "new"
    c.definition_keys = ("new-definition",)
    c.checked = SimpleNamespace(facts=f.owner)
    c.definition_stage = SimpleNamespace(expansion=SimpleNamespace(owner=f.domain))
    monkeypatch.setattr(ga, "restore", lambda _: (c, {"complete": True}))
    monkeypatch.setattr(ga, "directory", lambda key: Path("/" + key))
    raw = plan().model_dump(mode="json")
    if fault == "reverse":
        raw["origins"].reverse()
    if fault == "extra":
        raw["origins"].append(dict(campaign="/extra"))
    if fault == "foreign":
        raw["origins"][0]["campaign"] = "/other"
    if fault == "local":
        raw["origins"][0]["local_finalization"] = "/foreign"
    p = ag.read_plan(raw)
    if fault:
        with pytest.raises(ValueError, match="exact_origins"):
            ga.restore_plan(p)
    else:
        assert ga.restore_plan(p)[2].keywords["selected_only"] is False


def test_aggregate_report_preserves_scope_and_rejects_inconsistent_projection():
    f, _, seed, _ = fixture()
    parent, manifest = projection.project_interpretations(
        seed, f.domain, ["new-definition"], selected_only=False
    )
    c = object.__new__(DefinitionReview)
    c.definition_stage = object()
    docs = SimpleNamespace(read=lambda _: (manifest,))
    c.documents = lambda _: docs
    view = GenericAggregateReportView(c, "/new-fold")
    result = dict(
        definition_revision_policy="owner_adopted_definition_revision/1.0.0",
        definition_revision_projection_checksum=digest(manifest),
        original_result_checksum=digest("old-result"),
        origins=[],
    )
    assert view.definition_stage is None and c.definition_stage is not None
    scope = view.definition_aggregation_metadata(parent, result)
    assert scope["new_model_requests"] == 0
    assert scope["definition_projection"] == manifest
    assert scope["historical_scores_modified"] is False
    with pytest.raises(ValueError, match="projection"):
        view.definition_aggregation_metadata(
            parent, dict(result, definition_revision_projection_checksum=digest("changed"))
        )


def test_wholecase_request_accounting_is_deduplicated_and_rechecked():
    p = plan()
    history = [
        dict(directory="/old", checksum=digest("old"), requests=10),
        dict(directory="/new", checksum=digest("new"), requests=2),
    ]
    readers = [SimpleNamespace(accounting=history[:1]), SimpleNamespace(accounting=history)]
    assert ag.request_history(p, None, None, readers) == history
    result = dict(previous_attempts=history, cumulative_requests=12, new_requests=0)
    ag.verify_request_history(
        p, None, None, readers, dict(request_history_checksum=digest(history)), result
    )
    with pytest.raises(ValueError, match="request_history_binding"):
        ag.verify_request_history(
            p,
            None,
            None,
            readers,
            dict(request_history_checksum=digest(history)),
            dict(result, cumulative_requests=2),
        )
    readers[1].accounting[1]["requests"] = 31
    with pytest.raises(ValueError, match="budget"):
        ag.request_history(p, None, None, readers)
