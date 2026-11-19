"""One saved-proof operation shares an already read state, not current authorization."""

import pytest

from packages.evidence_details import recorded_finalization as recorded
from packages.evidence_details.acceptance_models import read_receipt
from packages.evidence_details.accepted_review_reuse import RecordedAcceptedReviewReader
from packages.evidence_details.review_batches import read_submission
from tests.agent_provider.test_recorded_acceptance_reuse import closed, read


def fixture():
    f, permission = closed()
    proof = read(f)
    reads = []

    def state():
        reads.append("current_input_and_source_owner")
        return f.owner.current()

    reader = RecordedAcceptedReviewReader(
        config=f.config,
        saved_proof=proof,
        documents=f.docs,
        requests=f.requests,
        state_reader=state,
        acceptance_key=permission.acceptance_key,
    )
    submission = read_submission(
        f.docs.read("submission:" + permission.acceptance_key)[0]["submission"]
    )
    receipt = read_receipt(f.docs.read("accepted:" + permission.acceptance_key)[3])
    return f, reader, submission, receipt, reads


def test_saved_proof_operation_has_fresh_before_after_not_three_full_input_reads():
    f, reader, sub, receipt, reads = fixture()
    for iteration in range(2):
        # The original public current() reads this state before _validated_requests.
        old = reader.state_reader()
        result = reader._validated_requests(old, sub, receipt, f.requests.records())
        assert result == f.requests.records()
        assert len(reads) == 2 * (iteration + 1)
    assert len(f.wire.calls) == 3


def test_source_revocation_after_saved_proof_is_checked_before_return(monkeypatch):
    f, reader, sub, receipt, reads = fixture()
    old = reader.state_reader()
    original = recorded.read_saved_acceptance

    def revoke_after_proof(**kwargs):
        result = original(**kwargs)
        f.owner.withdrawn = True
        return result

    monkeypatch.setattr(recorded, "read_saved_acceptance", revoke_after_proof)
    with pytest.raises(ValueError, match="source_withdrawn"):
        reader._validated_requests(old, sub, receipt, f.requests.records())
    assert len(reads) == 2 and len(f.wire.calls) == 3
