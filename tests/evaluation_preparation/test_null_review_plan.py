"""Versioned CLI-plan contracts and the original accepted Review selection."""

import json

import pytest

from packages.grounding.models import digest
from packages.multimodal_research import accepted_aggregation as entry
from tests.evaluation_preparation.test_null_review_owner_chain import combined


def raw_plan(f):
    return dict(
        schema_version="accepted-review-aggregation-plan/1.11.0",
        facts="synthetic-new-context",
        origins=[dict(campaign="synthetic-old"), dict(campaign="synthetic-new")],
        null_revisions=[r.model_dump(mode="json") for r in f.source.null_revisions],
    )


@pytest.mark.parametrize("value", [0.03, 0.0, None])
def test_saved_plan_rebuilds_exact_selection_with_original_readers(value):
    f = combined(value)
    plan = entry.read_plan(raw_plan(f))
    restored = entry.read_plan(json.loads(plan.model_dump_json()))
    assert restored == plan
    before = f.source.current(f.owner.current())
    source = entry.review_set(restored, f.source.readers, f.new.config)
    after = source.current(f.owner.current())
    assert before == after and digest(before) == digest(after)
    assert sum(i.assessment_id == f.target for i in after[0].candidate.items) == 1
    assert (
        next(i for i in after[0].candidate.items if i.assessment_id == f.target).judgment.value
        == value
    )


def test_old_plan_cannot_silently_accept_new_selection_and_new_requires_binding():
    f = combined(0.03)
    raw = raw_plan(f)
    raw["schema_version"] = "accepted-review-aggregation-plan/1.5.0"
    with pytest.raises(ValueError):
        entry.read_plan(raw)
    raw["schema_version"] = "accepted-review-aggregation-plan/1.11.0"
    raw["null_revisions"] = []
    with pytest.raises(ValueError):
        entry.read_plan(raw)


def test_changed_saved_selection_rejected_by_original_domain_owner():
    f = combined(0.03)
    raw = raw_plan(f)
    raw["null_revisions"][0]["replacement_item_checksum"] = digest("different")
    source = entry.review_set(entry.read_plan(raw), f.source.readers, f.new.config)
    with pytest.raises(ValueError, match="null_revision_accepted_item_binding"):
        source.current(f.owner.current())


def test_legacy_plan_retains_duplicate_rejection():
    f = combined(0.03)
    raw = raw_plan(f)
    raw["schema_version"] = "accepted-review-aggregation-plan/1.5.0"
    del raw["null_revisions"]
    plan = entry.read_plan(raw)
    source = entry.review_set(plan, f.source.readers, f.new.config)
    with pytest.raises(ValueError, match="accepted_aggregation_duplicate_origin_or_item"):
        source.current(f.owner.current())
