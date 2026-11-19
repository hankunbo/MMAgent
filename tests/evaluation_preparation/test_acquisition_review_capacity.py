"""Reject an impossible complete Review plan before acquiring more observations."""

import pytest

from packages.llm_provider.config import ReviewBatchPolicy
from packages.multimodal_research.campaign_renewal import validate_review_capacity


def policy(size=1, cap=15):
    return ReviewBatchPolicy(batch_size=size, max_batches=cap)


def ids(n, prefix="old"):
    return tuple(f"{prefix}:{i:03d}" for i in range(n))


def test_actual_failure_shape_is_refused_before_any_dispatch():
    with pytest.raises(ValueError, match="renewal_review_plan_capacity"):
        validate_review_capacity(policy(), ids(15), ids(3, "new"))


def test_previous_success_shape_keeps_identical_policy():
    p = policy()
    before = p.model_dump()
    validate_review_capacity(p, ids(12), ids(3, "new"))
    assert p.model_dump() == before


def test_size_three_handles_same_inventory_without_changing_request_cap():
    validate_review_capacity(policy(3), ids(15), ids(3, "new"))


def test_reuse_partition_rounding_is_also_counted():
    # ceil(5/3)=2 fits, but reused/new segregation needs 1+2=3 batches.
    with pytest.raises(ValueError, match="renewal_review_plan_capacity"):
        validate_review_capacity(policy(3, 2), ids(1), ids(4, "new"))


def test_clarification_of_existing_item_does_not_duplicate_inventory():
    validate_review_capacity(policy(), ids(14), (*ids(1), *ids(1, "new")))


def test_no_reuse_cannot_bypass_original_finite_limit():
    with pytest.raises(ValueError, match="renewal_review_plan_capacity"):
        validate_review_capacity(policy(), (), ids(16, "new"))


def test_caller_policy_fields_are_validated():
    with pytest.raises(ValueError):
        validate_review_capacity(dict(batch_size=0, max_batches=15), ids(1), ids(1, "new"))
