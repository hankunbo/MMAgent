"""Explicit generic composition plan; original multimodal plans retain dispatch."""

import json
from types import SimpleNamespace

import pytest

from packages.evaluation_preparation.generic_aliases import GenericPhase
from packages.evaluation_preparation.generic_review import GenericReview
from packages.multimodal_research import accepted_aggregation as a


def plan():
    return a.read_plan(
        dict(
            schema_version="accepted-review-aggregation-plan/1.2.0",
            facts="facts",
            origins=[dict(campaign="old"), dict(campaign="increment")],
        )
    )


def test_explicit_plan_uses_full_generic_phase_not_increment_selector(monkeypatch):
    from packages.evaluation_preparation import generic_pending_review as p

    campaign = object.__new__(GenericReview)
    campaign.pending_review_proof = {"selected_ids": ["new"]}
    campaign.phase_class = p.phase_type(lambda: campaign.pending_review_proof)
    result = {"saved": True}
    monkeypatch.setattr(p, "restore_campaign", lambda _: (campaign, result))
    actual, original, phase = a.plan_campaign(plan())
    assert actual is campaign and original is result
    assert phase is GenericPhase and phase is not campaign.phase_class
    assert campaign.pending_review_proof == {"selected_ids": ["new"]}


def test_generic_plan_cannot_use_unbounded_or_unselected_campaign(monkeypatch):
    from packages.evaluation_preparation import generic_pending_review as p

    monkeypatch.setattr(p, "restore_campaign", lambda _: (SimpleNamespace(), {}))
    with pytest.raises(ValueError, match="requires_pending_successor"):
        a.plan_campaign(plan())


def test_legacy_plan_still_calls_original_completed_parent(monkeypatch):
    c, result = SimpleNamespace(), {"old": True}
    monkeypatch.setattr(a, "completed_parent", lambda _: (c, result))
    old = a.read_plan(
        dict(
            schema_version="accepted-review-aggregation-plan/1.0.0",
            facts="facts",
            origins=[dict(campaign="old")],
        )
    )
    assert a.plan_campaign(old) == (c, result, a.Phase)


def test_generic_restore_rejects_foreign_path_before_reading():
    from packages.evaluation_preparation.generic_pending_review import restore_campaign

    with pytest.raises(ValueError, match="path_scope"):
        restore_campaign("/tmp/not-a-registered-generic-case")


def test_generic_origin_cannot_claim_multimodal_local_override():
    with pytest.raises(ValueError, match="foreign_local_finalization"):
        a.origin_reader(a.OriginPath(campaign="old", local_finalization="other"), generic=True)


def test_generic_cold_restore_routes_to_generic_owner_before_result_read(monkeypatch, tmp_path):
    """A dispatch contract test, not proof of full persistent recovery."""
    from packages.evaluation_preparation import generic_pending_review as p

    class ReachedGenericOwner(Exception):
        pass

    directory = tmp_path / "aggregation"
    directory.mkdir()
    expected = plan().model_dump(mode="json")
    (directory / "plan.json").write_text(json.dumps(expected))

    def wrong_owner(*args, **kwargs):
        raise AssertionError("generic aggregation routed to multimodal owner")

    def generic_owner(path):
        assert path == a.checked_path(expected["facts"])
        raise ReachedGenericOwner

    monkeypatch.setattr(a, "completed_parent", wrong_owner)
    monkeypatch.setattr(p, "restore_campaign", generic_owner)
    with pytest.raises(ReachedGenericOwner):
        a.restore_aggregation(directory)


@pytest.mark.parametrize("boundary", ["self", "depth", "cycle"])
def test_generic_cold_restore_keeps_history_bounds_before_owner(boundary, monkeypatch, tmp_path):
    from packages.evaluation_preparation import generic_pending_review as p

    directory = tmp_path / "aggregation"
    directory.mkdir()
    expected = plan().model_dump(mode="json")
    if boundary == "self":
        expected["facts"] = str(directory)
    (directory / "plan.json").write_text(json.dumps(expected))

    def forbidden(*args, **kwargs):
        raise AssertionError("history bound must precede owner reads")

    monkeypatch.setattr(a, "completed_parent", forbidden)
    monkeypatch.setattr(p, "restore_campaign", forbidden)
    options = (
        {"depth": a.MAX_HISTORY_DEPTH + 1}
        if boundary == "depth"
        else {"ancestry": (directory, tmp_path / "later")}
        if boundary == "cycle"
        else {}
    )
    with pytest.raises(ValueError, match="history_depth|history_cycle"):
        a.restore_aggregation(directory, **options)


@pytest.mark.parametrize("generic", [True, False])
def test_cold_restore_context_cache_belongs_only_to_multimodal(generic, monkeypatch, tmp_path):
    """Verify owner-specific assembly before projection; persistence tested separately."""

    class ReachedProjection(Exception):
        pass

    directory = tmp_path / "aggregation"
    directory.mkdir()
    raw = plan().model_dump(mode="json")
    if not generic:
        raw = dict(
            schema_version="accepted-review-aggregation-plan/1.0.0",
            facts="facts",
            origins=[dict(campaign="old")],
        )
    raw = a.read_plan(raw).model_dump(mode="json")
    old = {"saved": True}
    phase = {"run_id": "RUN:RESTORE"}
    result = dict(
        schema_version=("generic" if generic else "multimodal")
        + "-accepted-aggregation-result/1.0.0",
        original_result_checksum=a.digest(old),
        phases=[phase],
    )
    manifest = dict(plan=raw, original_result_checksum=a.digest(old), run_id=phase["run_id"])
    for name, value in (("plan", raw), ("manifest", manifest), ("result", result)):
        (directory / (name + ".json")).write_text(json.dumps(value))
    saved = {
        "accepted-aggregation-manifest": (manifest,),
        "multimodal-case-result": (result,),
        "dynamic-case-result": (result,),
        "dynamic-phase-result": (phase,),
    }
    docs = SimpleNamespace(run_id=phase["run_id"], read=lambda key: saved[key])
    base = SimpleNamespace() if generic else SimpleNamespace(contexts={})
    context_reads = []

    def read_context(run_id):
        assert not generic, "generic owner must not require a multimodal context cache"
        context_reads.append(run_id)
        return "original-context"

    campaign = SimpleNamespace(
        base=base,
        documents=lambda _: docs,
        config="config",
        accepted_observations="observations",
        read_model="models",
        contexts=SimpleNamespace(read_research_run=read_context),
    )

    class Projection:
        def __init__(self, actual_base, **kwargs):
            assert actual_base is base
            assert kwargs == dict(
                observation_reader="observations", **({"model_reader": "models"} if generic else {})
            )

        def project(self, **kwargs):
            assert context_reads == ([] if generic else [docs.run_id])
            if not generic:
                assert base.contexts[docs.run_id] == "original-context"
            raise ReachedProjection

    monkeypatch.setattr(a, "plan_campaign", lambda _: (campaign, old, Projection))
    monkeypatch.setattr(a, "completed_parent", lambda *args, **kwargs: (campaign, old))
    monkeypatch.setattr(a, "Phase", Projection)
    with pytest.raises(ReachedProjection):
        a.restore_aggregation(directory)
