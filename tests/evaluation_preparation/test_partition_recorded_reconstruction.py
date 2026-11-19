"""Original receipt/source owners reconstruct complete partitioned batches, zero wire."""

import pytest

from packages.evidence_details.recorded_finalization import reconstruct
from packages.grounding.models import digest
from tests.evaluation_preparation.test_partition_increment import partition_fixture


def test_single_item_legacy_plan_with_original_reuse_closes_without_new_requests():
    """Real 1.26 failure shape: old items are reused, but no mixed batch exists."""
    from packages.evidence_details.recorded_finalization import PARTITION_GRANT_KEY, closure_keys

    f = partition_fixture(batch_key="new:A", batch_size=1)
    assert f.submission.plan.schema_version == "review-batch-plan/1.0.0"
    f.documents.write("live-stage-intent:review", ({"event": {"code": "new"}},))
    before = digest(f.documents.saved), digest(f.budget.journal.read())
    assert closure_keys(f.documents)[0] == PARTITION_GRANT_KEY
    for _ in range(2):
        submission, records, rejected = reconstruct(
            f.owner.current(), f.documents, f.budget, owner=f.owner
        )
        assert submission == f.submission
        assert records == tuple(f.budget.records()) and rejected == ()
    assert before == (digest(f.documents.saved), digest(f.budget.journal.read()))


@pytest.mark.parametrize("aggregate_saved", [True, False])
def test_rebuild_complete_partition_without_changing_old_batch_or_requests(aggregate_saved):
    f = partition_fixture(batch_key="new:A")
    f.documents.write("live-stage-intent:review", ({"event": {"code": "new"}},))
    if not aggregate_saved:
        del f.documents.saved["review-batches:new:A:aggregate"]
    before = digest(f.documents.saved), digest(f.budget.journal.read())
    for _ in range(2):
        submission, records, rejected = reconstruct(
            f.owner.current(), f.documents, f.budget, owner=f.owner
        )
        assert submission == f.submission
        assert records == tuple(f.budget.records()) and rejected == ()
    assert before == (digest(f.documents.saved), digest(f.budget.journal.read()))


@pytest.mark.parametrize("fault", ["owner", "origin", "batch", "aggregate"])
@pytest.mark.parametrize("batch_size", [1, 3])
def test_partition_reconstruction_rejects_incomplete_or_changed_current_material(fault, batch_size):
    f = partition_fixture(batch_key="new:A", batch_size=batch_size)
    f.documents.write("live-stage-intent:review", ({"event": {"code": "new"}},))
    owner = f.owner
    if fault == "owner":
        owner = None
    elif fault == "origin":
        f.documents.saved["review-batches:new:A:historical"][0]["acceptance_checksum"] = digest(
            "different receipt"
        )
    elif fault == "batch":
        batch = f.submission.batches[-1].batch_id
        del f.documents.saved["review-batches:new:A:" + batch + ":result"]
    else:
        f.documents.saved["review-batches:new:A:aggregate"] = ({"changed": True},)
    before = digest(f.documents.saved), digest(f.budget.journal.read())
    with pytest.raises(
        ValueError,
        match="original_reader_required|partition_origin_changed|complete_batches_required|original_aggregate_required",
    ):
        reconstruct(f.owner.current(), f.documents, f.budget, owner=owner)
    assert before == (digest(f.documents.saved), digest(f.budget.journal.read()))
