"""Composition-only boundary; original owner and real CLI have separate checks."""

from types import SimpleNamespace

import pytest

from packages.multimodal_research.pending_review import inventory
from tests.evaluation_preparation.test_directional_review import fixture


@pytest.mark.parametrize(
    "partial,owner_error,unknown",
    [
        (False, None, False),
        (True, None, False),
        (True, None, True),
        (True, "pending_review_acceptance_binding", True),
    ],
)
def test_failed_child_can_select_work_after_ancestor_inventory_is_empty(
    monkeypatch, partial, owner_error, unknown
):
    state = fixture().owner.current()
    checks = []

    def parent(name, accepted, targets=(), prior=None):
        phase = dict(run_id=name, target=None, accepted=accepted, acceptance_key="review")

        class Docs:
            run_id = name

            def read(self, key):
                if key == "dynamic-phase-result":
                    return (phase,)
                if key == "dynamic-phase-input":
                    return (state,)
                if key == "accepted:review":
                    return ({},)
                raise AssertionError(key)

        return SimpleNamespace(
            base=SimpleNamespace(check=lambda: checks.append(name), contexts={}),
            documents=lambda _: Docs(),
            contexts=SimpleNamespace(read_research_run=lambda _: None),
            accepted_observations=lambda: {},
            config=None,
            protocol=(
                dict(
                    pending_review=dict(quarantined_batches=[]),
                    tip=dict(run_id=prior.name),
                    review_target_ids=list(targets),
                )
                if prior
                else {}
            ),
            parent=prior,
            name=name,
        )

    from packages.multimodal_research import campaign

    monkeypatch.setattr(
        campaign, "Phase", lambda *_a, **_kw: SimpleNamespace(project=lambda **_: (state, None))
    )

    def partial_owner(**kw):
        if owner_error:
            raise ValueError(owner_error)
        return dict(
            selected_ids=["unstarted"] if kw["documents"].run_id == "done" else ["done"],
            accepted_ids=["done"] if kw["documents"].run_id == "done" else ["original"],
            quarantined_batches=[
                dict(assessment_ids=["malformed"], new_reassessment_authorized=False)
            ]
            if kw["documents"].run_id == "done"
            else [],
        )

    monkeypatch.setattr(
        "packages.multimodal_research.pending_review.read_pending_review", partial_owner
    )

    def whole_stage(*_a, **_kw):
        # A valid partial acceptance can coexist with a quarantined UNKNOWN.
        # It is not a whole-stage confirmed commit.
        if unknown:
            raise ValueError("committed_review_unknown_request")
        return {"proof": True}

    monkeypatch.setattr("agents.expert_agent.committed_review.read_committed_review", whole_stage)
    monkeypatch.setattr(
        "packages.evidence_details.acceptance_models.read_records",
        lambda _: SimpleNamespace(
            candidate=SimpleNamespace(items=[SimpleNamespace(assessment_id="done")])
        ),
    )
    # digest uses serializable records, not mock objects.
    from packages.grounding.models import digest as original_digest
    from packages.multimodal_research import pending_review

    monkeypatch.setattr(
        "packages.grounding.models.digest",
        lambda x: (
            original_digest("record") if isinstance(x, SimpleNamespace) else original_digest(x)
        ),
    )
    attempts = []

    def attempted(**kwargs):
        attempts.append(kwargs["acknowledge_request"])
        return dict(
            selected_ids=["pending"] if kwargs["acknowledge_request"] else [],
            quarantined_batches=[],
        )

    monkeypatch.setattr(
        "packages.evidence_details.pending_review.read_failed_review_attempt", attempted
    )
    root = parent("root", True)
    done = parent("done", True, ["done", "malformed", "unstarted"] if partial else ["done"], root)
    if partial:
        if owner_error:
            with pytest.raises(ValueError, match=owner_error):
                inventory(done, dict(run_id="done"))
            return
        value = inventory(done, dict(run_id="done"))
        assert value["selected_ids"] == ["unstarted"]
        assert value["accepted_ids"] == ["done", "original"]
        assert not value["quarantined_batches"][0]["new_reassessment_authorized"]
        assert checks == ["done", "root"]
        if unknown:
            assert value["partial_increment"]["proof"]["whole_stage_confirmed"] is False
        else:
            assert value["partial_increment"]["proof"] == {"proof": True}
        return
    failed = parent("failed", False, ["pending"], done)
    value = pending_review.inventory(failed, dict(run_id="failed"), "request:explicit")
    assert value["selected_ids"] == ["pending"]
    assert value["accepted_ids"] == ["done", "original"]
    assert checks == ["failed", "done", "root"]
    assert attempts == ["request:explicit"]
    with pytest.raises(ValueError, match="no_safe_unstarted_work"):
        inventory(failed, dict(run_id="failed"))
