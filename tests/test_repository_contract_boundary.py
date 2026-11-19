"""Fast tests for the contracts -> repository validation boundary."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from db.audit.repository import AuditEvent, AuditEventType
from db.audit.repository.postgres import _event_document
from db.runtime.repository.postgres import _payload
from packages.contracts import (
    ActorType,
    CapabilityGap,
    CaseBundle,
    ExpertAssessment,
    ToolRequirementDocument,
)

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "packages" / "contracts" / "examples" / "v1"


def _example(model_type: type, filename: str):
    return model_type.model_validate(json.loads((EXAMPLES / filename).read_text()))


def _invalid_models():
    case = _example(CaseBundle, "CaseBundle.valid.json")
    yield case.model_copy(
        update={"provenance": case.provenance.model_copy(update={"actor": "BAD_ACTOR"})}
    )

    gap = _example(CapabilityGap, "CapabilityGap.valid.json")
    yield gap.model_copy(update={"gap_codes": ("NOT_A_GAP",)})

    assessment = _example(ExpertAssessment, "ExpertAssessment.valid.json")
    yield assessment.model_copy(update={"allowed_action_codes": ("NOT_AN_ACTION",)})

    requirement = _example(ToolRequirementDocument, "ToolRequirementDocument.valid.json")
    yield requirement.model_copy(update={"required_input_contract": ({},)})


@pytest.mark.parametrize("invalid", tuple(_invalid_models()))
def test_repository_payload_revalidates_untrusted_model_instances(invalid) -> None:
    with pytest.raises(ValidationError):
        _payload(invalid)


@pytest.mark.parametrize(
    ("model_type", "filename"),
    (
        (CaseBundle, "CaseBundle.valid.json"),
        (CapabilityGap, "CapabilityGap.valid.json"),
        (ExpertAssessment, "ExpertAssessment.valid.json"),
        (ToolRequirementDocument, "ToolRequirementDocument.valid.json"),
    ),
)
def test_repository_payload_preserves_valid_contract(model_type, filename) -> None:
    model = _example(model_type, filename)
    assert model_type.model_validate(_payload(model)) == model


def _audit_event() -> AuditEvent:
    return AuditEvent(
        event_id="MESSAGE:OID_BOUND001",
        event_type=AuditEventType.REQUEST_RECEIVED,
        actor=ActorType.PLANNER,
        correlation_code="MESSAGE:OID_CORR0001",
        idempotency_key="TASK:OID_BOUND001",
        occurred_at=datetime(2026, 8, 31, tzinfo=UTC),
        payload={"request_code": "REQUEST:SYNTHETIC", "api_key": "sk-secret-value"},
    )


def test_audit_repository_revalidates_and_redacts_before_write() -> None:
    event = _audit_event()
    document = _event_document(event)
    assert document["payload"]["api_key"] == "[REDACTED]"
    assert document["payload_checksum"].startswith("sha256:")
    event_fields = {key: document[key] for key in AuditEvent.model_fields}
    assert AuditEvent.model_validate(event_fields).event_id == event.event_id


def test_audit_repository_rejects_unvalidated_or_forbidden_payload() -> None:
    invalid = _audit_event().model_copy(update={"event_type": "NOT_AN_EVENT"})
    with pytest.raises(ValidationError):
        _event_document(invalid)
    forbidden = _audit_event().model_copy(update={"payload": {"chain_of_thought": "hidden"}})
    with pytest.raises(ValueError, match="audit payload field is forbidden"):
        _event_document(forbidden)
