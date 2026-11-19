"""Traversal tests; current owner qualification remains a separate required proof."""

from types import SimpleNamespace

import pytest

from packages.evidence_details.accepted_review_reuse import (
    MAX_ACCEPTED_REVIEW_DEPTH,
    ReusedAcceptedSubmissionV15,
    provenance_batches,
)


def chain(depth):
    # Only exercises traversal, not permission/acceptance of these constructed objects.
    current = SimpleNamespace(batches=("root",))
    for i in range(depth):
        current = ReusedAcceptedSubmissionV15.model_construct(
            historical=SimpleNamespace(run_id="RUN:" + str(i), submission=current),
            batches=("batch:" + str(i),),
        )
    return current


@pytest.mark.parametrize("depth", [0, 4, 5, 16])
def test_finite_depth_and_oldest_first_order(depth):
    sub = chain(depth)
    assert provenance_batches(sub) == ("root", *("batch:" + str(i) for i in range(depth)))


def test_depth_over_bound_still_rejected():
    assert MAX_ACCEPTED_REVIEW_DEPTH == 16
    with pytest.raises(ValueError, match="^accepted_review_reuse_depth$"):
        provenance_batches(chain(17))


@pytest.mark.parametrize("fault", ["object_cycle", "run_cycle"])
def test_cycles_rejected_without_unbounded_recursion(fault):
    sub = chain(3)
    if fault == "object_cycle":
        sub.historical.submission = sub
    else:
        sub.historical.run_id = sub.historical.submission.historical.run_id
    with pytest.raises(ValueError, match="accepted_review_reuse_cycle_or_depth"):
        provenance_batches(sub)


@pytest.mark.parametrize("depth", [-1, 17, True, 1.5])
def test_invalid_initial_depth_rejected(depth):
    with pytest.raises(ValueError, match="^accepted_review_reuse_depth$"):
        provenance_batches(chain(0), depth=depth)
