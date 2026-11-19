"""04-owned versioned engineering/medical basis binding, parallel to legacy StopRule.

Pinned deployment review files confer only engineering provenance. They cannot enter
Knowledge.publish_offline or replace an existing Published StopRule in a DiseasePack.
"""

import time
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import Field, field_serializer, model_validator

from packages.grounding.models import Checksum, Code, Value, check_seal, digest, validate
from packages.object_store import checksum
from packages.tool_execution.process import read_file
from packages.tool_execution.trusted_models import ReviewedFile
from packages.tool_execution.trusted_process import verify_file


def read_reviewed_file(record):
    record = validate(ReviewedFile, record)
    verify_file(record, deadline=time.monotonic() + 10, limit=1048576)
    raw = read_file(Path(record.path), 1048576)
    if len(raw) != record.byte_size or checksum(raw) != record.checksum:
        raise ValueError("engineering_review_bytes_changed")
    return raw


class EngineeringBasis(Value):
    version: Literal["engineering-stop-basis/1.0.0"] = "engineering-stop-basis/1.0.0"
    checksum: Checksum
    owner: Code
    trigger: Literal[
        "time_budget",
        "action_or_cost_budget",
        "graph_round_budget",
        "paths_exhausted",
        "environment_error",
    ]
    reason_code: Code
    predicate: str = Field(min_length=1, max_length=512)
    source: ReviewedFile
    symbol: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_.]{0,127}$")
    engine_version: str = Field(min_length=1, max_length=64)
    configuration: ReviewedFile
    configuration_version: str = Field(min_length=1, max_length=64)
    runtime_config_checksum: Checksum
    rule_version: str = Field(pattern=r"^[0-9]+\.[0-9]+\.[0-9]+$")

    @model_validator(mode="after")
    def coherent(self):
        check_seal(self)
        return self


class EngineeringReview(Value):
    version: Literal["engineering-stop-review/1.0.0"] = "engineering-stop-review/1.0.0"
    checksum: Checksum
    subject_checksum: Checksum
    dependency_checksum: Checksum
    reviewer: Code
    owner: Code
    decision: Literal["APPROVED", "REJECTED", "WITHDRAWN"]
    reviewed_at: datetime
    synthetic: bool
    authority: Literal["engineering_only"] = "engineering_only"

    @field_serializer("reviewed_at")
    def serialize_time(self, value):
        return value.isoformat()

    @model_validator(mode="after")
    def coherent(self):
        if self.reviewed_at.utcoffset() is None:
            raise ValueError("engineering_review_timezone")
        check_seal(self)
        return self


class MedicalStopDependency(Value):
    """Exact legacy references; the existing medical reader verifies lifecycle and reviews."""

    stop_rule: dict
    policy: dict

    @model_validator(mode="after")
    def coherent(self):
        from db.knowledge.models import KnowledgeReference

        rule = validate(KnowledgeReference, self.stop_rule)
        policy = validate(KnowledgeReference, self.policy)
        if rule.kind.value != "stop_rule" or policy.kind.value != "release_policy":
            raise ValueError("medical_stop_reference_kind")
        return self


class StopBasisBinding(Value):
    schema_version: Literal["stop-basis-binding/2.0.0"] = "stop-basis-binding/2.0.0"
    checksum: Checksum
    kind: Literal["engineering", "medical", "mixed"]
    owner: Literal["04_STOP_BASIS"] = "04_STOP_BASIS"
    engineering: tuple[EngineeringBasis, ...] = Field(default=(), max_length=16)
    review_files: tuple[ReviewedFile, ...] = Field(default=(), max_length=16)
    medical: tuple[MedicalStopDependency, ...] = Field(default=(), max_length=16)
    synthetic: bool
    clinical_publication_authority: Literal[False] = False

    @model_validator(mode="after")
    def coherent(self):
        if bool(self.engineering) != (self.kind in {"engineering", "mixed"}) or bool(
            self.medical
        ) != (self.kind in {"medical", "mixed"}):
            raise ValueError("stop_basis_kind_dependencies")
        if len(self.review_files) != len(self.engineering) or len(
            {b.checksum for b in self.engineering}
        ) != len(self.engineering):
            raise ValueError("stop_basis_review_required")
        check_seal(self)
        return self


def dependencies(basis):
    return digest(
        dict(
            source=basis.source,
            configuration=basis.configuration,
            symbol=basis.symbol,
            engine_version=basis.engine_version,
            configuration_version=basis.configuration_version,
            runtime_config_checksum=basis.runtime_config_checksum,
            rule_version=basis.rule_version,
            trigger=basis.trigger,
            reason_code=basis.reason_code,
            predicate=basis.predicate,
            owner=basis.owner,
        )
    )


class EngineeringStopReader:
    """Trusted assembly pins approved files/owners; callers select an exact checksum only."""

    def __init__(
        self,
        *,
        records,
        owners,
        reviewers,
        current_versions,
        medical_reader=None,
        synthetic_test_mode=False,
    ):
        self.records = dict(records)
        self.owners, self.reviewers = frozenset(owners), frozenset(reviewers)
        self.current_versions = current_versions
        self.medical_reader = medical_reader
        self.synthetic = synthetic_test_mode

    def read(self, expected_checksum):
        record = self.records.get(expected_checksum)
        if record is None:
            raise ValueError("engineering_stop_not_registered")
        value = StopBasisBinding.model_validate_json(read_reviewed_file(record))
        if value.checksum != expected_checksum or value.synthetic != self.synthetic:
            raise ValueError("engineering_stop_binding_identity")
        for basis, review_file in zip(value.engineering, value.review_files, strict=True):
            verify_file(basis.source, deadline=time.monotonic() + 10, limit=1048576)
            verify_file(basis.configuration, deadline=time.monotonic() + 10, limit=1048576)
            review = EngineeringReview.model_validate_json(read_reviewed_file(review_file))
            current = self.current_versions(basis.owner, basis.trigger)
            expected = (
                basis.engine_version,
                basis.configuration_version,
                basis.rule_version,
                basis.source.checksum,
                basis.configuration.checksum,
                basis.symbol,
                basis.reason_code,
                basis.predicate,
                basis.runtime_config_checksum,
            )
            if current != expected:
                raise ValueError("engineering_stop_version_drift")
            if (
                basis.owner not in self.owners
                or review.reviewer not in self.reviewers
                or review.owner != basis.owner
                or review.decision != "APPROVED"
                or review.synthetic != self.synthetic
                or review.subject_checksum != basis.checksum
                or review.dependency_checksum != dependencies(basis)
            ):
                raise ValueError("engineering_stop_review_rejected")
        for medical in value.medical:
            if self.medical_reader is None:
                raise ValueError("medical_stop_reader_required")
            self.medical_reader.require_current(medical)
        return value

    def require_trigger(self, expected_checksum, *, trigger, reason_code, runtime_config_checksum):
        value = self.read(expected_checksum)
        if not any(
            b.trigger == trigger
            and b.reason_code == reason_code
            and b.runtime_config_checksum == runtime_config_checksum
            for b in value.engineering
        ):
            raise ValueError("engineering_stop_trigger_mismatch")
        return value


def read_stop_basis(raw):
    """Explicit dispatch. The legacy model and exact serialization are untouched."""
    import json

    value = json.loads(raw)
    if value.get("schema_version") == "stop-basis-binding/2.0.0":
        return StopBasisBinding.model_validate_json(raw)
    if "schema_version" in value:
        raise ValueError("stop_basis_version_unsupported")
    from db.knowledge.models import StopRule

    return StopRule.model_validate_json(raw)


class LegacyMedicalStopReader:
    def __init__(self, repository):
        self.repository = repository

    def require_current(self, dependency):
        from db.knowledge.models import KnowledgeReference, KnowledgeStatus, StopRule

        dependency = validate(MedicalStopDependency, dependency)
        ref = validate(KnowledgeReference, dependency.stop_rule)
        snapshot = self.repository.get(ref)
        bundle = self.repository.review_bundle(ref)
        if (
            type(snapshot.record) is not StopRule
            or snapshot.status != KnowledgeStatus.PUBLISHED
            or snapshot.withdrawn
            or not bundle["available"]
        ):
            raise ValueError("medical_stop_not_published_current")
        policy = validate(KnowledgeReference, dependency.policy)
        if not snapshot.record.binding_refs or policy not in snapshot.record.binding_refs:
            raise ValueError("medical_stop_policy_binding")
        return snapshot.record


def engineering_stop_receipt(
    reader, expected_checksum, *, trigger, reason_code, runtime_config_checksum
):
    """07 may attach this provenance after a saved technical stop, without changing it."""
    from apps.ophthalmology_disease_packs import read_engineering_stop_basis

    binding = read_engineering_stop_basis(
        reader,
        expected_checksum,
        trigger=trigger,
        reason_code=reason_code,
        runtime_config_checksum=runtime_config_checksum,
    )
    return dict(
        schema_version=binding.schema_version,
        checksum=binding.checksum,
        kind=binding.kind,
        owner=binding.owner,
        synthetic=binding.synthetic,
        trigger=trigger,
        reason_code=reason_code,
        grants_medical_authority=False,
    )
