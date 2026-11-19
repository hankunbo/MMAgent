"""Small pure schema/Audit compatibility probes; no model or database side effects."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from db.audit.repository import AuditEvent
from db.audit.repository.postgres import _event_document
from packages.memory.models import Review


@pytest.mark.parametrize("actor", ["PLANNER", "RULE_ENGINE", "KNOWLEDGE_SEARCH", "MEMORY"])
def test_additive_audit_actor_dispatch(actor):
    event = AuditEvent(
        event_id="MESSAGE:OID_MEMORYAUDIT",
        idempotency_key="TASK:OID_MEMORYAUDIT",
        actor=actor,
        event_type="OFFLINE_REVIEW_RECORDED",
        correlation_code="SOURCE:MEMORY",
        occurred_at=datetime.now(UTC),
        payload={"reason": "engineering_only"},
    )
    document = _event_document(event)
    assert document["actor"] == actor
    assert document["contract_schema_version"] == "1.0"


def test_clinical_approval_cannot_be_supplied():
    with pytest.raises(ValidationError):
        Review(
            key="review",
            target="target",
            target_checksum="sha256:" + "a" * 64,
            decision="approve",
            basis="engineering_review",
            reason="REVIEW:SYNTHETIC",
            clinical_approval=True,
        )
