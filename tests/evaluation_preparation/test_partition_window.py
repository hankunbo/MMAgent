from datetime import timedelta

import pytest

from packages.evidence_details import recorded_finalization as module
from tests.agent_provider.finalization_support import fixture
from tests.evaluation_preparation.test_partition_increment import partition_fixture


def test_old_fifteen_minute_grant_is_not_silently_extended():
    f = fixture(corrected=True)
    permission, _ = module.RecordedReviewPermission.prepare(
        owner=f.owner, documents=f.docs, requests=f.requests, config=f.config, clock=lambda: f.now
    )
    original = permission.grant.model_dump()
    assert permission.grant.deadline - permission.grant.issued_at == timedelta(minutes=15)
    newer = dict(
        original,
        schema_version="recorded-review-finalization/1.1.0",
        previous_local_grant_checksum=None,
        previous_local_receipt_checksum=None,
    )
    newer["deadline"] = permission.grant.issued_at + timedelta(minutes=45)
    assert module.RecordedPartitionGrant.model_validate(newer).new_requests == 0
    newer["deadline"] += timedelta(seconds=1)
    with pytest.raises(ValueError, match="finalization_bound"):
        module.RecordedPartitionGrant.model_validate(newer)
    original["deadline"] = permission.grant.issued_at + timedelta(minutes=45)
    with pytest.raises(ValueError, match="finalization_bound"):
        module.RecordedReviewGrant.model_validate(original)


def test_storage_keys_come_from_original_partition_plan_not_user_flags():
    f = partition_fixture(batch_key="new:A")
    f.documents.write("live-stage-intent:review", ({"event": {"code": "new"}},))
    assert module.closure_keys(f.documents) == (
        module.PARTITION_GRANT_KEY,
        module.PARTITION_SUBMISSION_KEY,
        module.PARTITION_ACCEPTANCE_KEY,
    )
    assert module.prior_local_proof(f.documents) == dict(
        previous_local_grant_checksum=None,
        previous_local_receipt_checksum=None,
    )
    f.documents.write("accepted:" + module.ACCEPTANCE_KEY, ({"status": "accepted"},))
    with pytest.raises(ValueError, match="prior_acceptance_must_be_reused"):
        module.prior_local_proof(f.documents)


def test_clock_is_rechecked_after_expensive_reconstruction(monkeypatch):
    f = fixture(corrected=True)
    permission, _ = module.RecordedReviewPermission.prepare(
        owner=f.owner, documents=f.docs, requests=f.requests, config=f.config, clock=lambda: f.now
    )
    original = module.reconstruct

    def delayed(*args, **kwargs):
        value = original(*args, **kwargs)
        f.now = permission.grant.deadline
        return value

    monkeypatch.setattr(module, "reconstruct", delayed)
    with pytest.raises(ValueError, match="recorded_finalization_deadline"):
        permission.check(f.state, role="review")
