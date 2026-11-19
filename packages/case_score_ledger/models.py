"""07-owned parallel result envelope. No contracts-v1 or clinical state extensions."""

import math
from typing import Annotated, Literal

from pydantic import Field, model_validator

from packages.contracts import Laterality, VersionedReference
from packages.dataset_adapters.models import ObjectReference
from packages.grounding.models import Checksum, Code, Ref, Value, check_seal

Token = Annotated[str, Field(pattern=r"^[A-Za-z0-9_.:/+-]{1,160}$")]


class ScoreScope(Value):
    case_code: Code
    run_id: Code
    encounter_code: Code
    patient_hash: Checksum
    laterality: Laterality


class GraphSnapshot(Value):
    graph_code: Code
    graph_version: int = Field(ge=1)
    graph_checksum: Checksum


class Component(Value):
    """Exact 03 EvidenceComponent, including original order and explanations."""

    contributing_nodes: tuple[Code, ...] = Field(max_length=1024)
    rule_refs: tuple[VersionedReference, ...] = Field(max_length=1024)
    required_count: int = Field(ge=0)
    resolved_count: int = Field(ge=0)
    blocked_count: int = Field(ge=0)
    explanations: tuple[Annotated[str, Field(max_length=4096)], ...] = Field(max_length=1024)


class ScoreDefinition(Value):
    version: Literal["score-definition/1.0.0"] = "score-definition/1.0.0"
    code: Token
    layer: Literal[
        "qc", "measurement", "model", "proposition", "graph", "disease", "uncertainty", "technical"
    ]
    metric: Literal[
        "quality",
        "measurement",
        "logit",
        "segmentation_confidence",
        "support",
        "refutation",
        "completeness",
        "conflict",
        "probability",
        "calibration_error",
        "uncertainty",
        "coverage",
    ]
    definition: str = Field(min_length=1, max_length=512)
    unit: Token
    direction: Literal["higher", "lower", "none", "owner_counts"]
    minimum: float | None = None
    maximum: float | None = None

    @model_validator(mode="after")
    def semantics(self):
        layers = {
            "quality": "qc",
            "measurement": "measurement",
            "logit": "model",
            "segmentation_confidence": "model",
            "support": "proposition",
            "refutation": "proposition",
            "completeness": "graph",
            "conflict": "graph",
            "probability": "disease",
            "calibration_error": "uncertainty",
            "uncertainty": "uncertainty",
            "coverage": "technical",
        }
        if self.layer != layers[self.metric]:
            raise ValueError("score_wrong_layer")
        for value in (self.minimum, self.maximum):
            if value is not None and not math.isfinite(value):
                raise ValueError("score_nonfinite")
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError("score_invalid_range")
        if self.metric in {"probability", "segmentation_confidence", "coverage"} and (
            self.minimum,
            self.maximum,
            self.unit,
        ) != (0.0, 1.0, "1"):
            raise ValueError("score_probability_or_ratio_range")
        if self.metric == "logit" and (
            self.minimum is not None or self.maximum is not None or self.unit != "logit"
        ):
            raise ValueError("score_logit_not_probability")
        if self.metric in {"support", "refutation", "completeness", "conflict"} and (
            self.unit,
            self.direction,
            self.minimum,
            self.maximum,
        ) != ("03_counts", "owner_counts", None, None):
            raise ValueError("score_owner_component_required")
        return self


class ScoreSource(Value):
    """Accepted event IDs and validation checksums are paired by tuple position."""

    scope: ScoreScope
    qualification: Literal["research_output", "accepted_graph", "technical"]
    synthetic: bool
    correlation_group: Checksum
    policy_checksum: Checksum
    graph: GraphSnapshot | None = None
    artifact_code: Code | None = None
    artifact_checksum: Checksum | None = None
    object_ref: ObjectReference | None = None
    attempt_id: Annotated[str, Field(pattern=r"^[a-f0-9]{32}$")] | None = None
    step_id: Token | None = None
    step_checksum: Checksum | None = None
    plan_checksum: Checksum | None = None
    output_schema_version: Token | None = None
    qc_policy_checksums: tuple[Checksum, ...] = Field(default=(), max_length=64)
    implementation: Ref | None = None
    model_versions: tuple[VersionedReference, ...] = Field(default=(), max_length=16)
    accepted_event_ids: tuple[Code, ...] = Field(default=(), max_length=256)
    validation_checksums: tuple[Checksum, ...] = Field(default=(), max_length=256)

    @model_validator(mode="after")
    def coherent(self):
        if (self.artifact_code is None) != (self.artifact_checksum is None):
            raise ValueError("score_artifact_identity")
        if self.qualification == "research_output" and any(
            x is None
            for x in (
                self.artifact_code,
                self.object_ref,
                self.attempt_id,
                self.step_id,
                self.step_checksum,
                self.plan_checksum,
                self.implementation,
            )
        ):
            raise ValueError("score_tool_source_required")
        if self.object_ref is not None and self.object_ref.checksum != self.artifact_checksum:
            raise ValueError("score_object_identity")
        if self.qualification == "accepted_graph" and (
            self.graph is None or not self.accepted_event_ids or not self.validation_checksums
        ):
            raise ValueError("score_accepted_source_required")
        if (
            len(self.accepted_event_ids) != len(self.validation_checksums)
            or len(set(self.accepted_event_ids)) != len(self.accepted_event_ids)
            or len(set(self.validation_checksums)) != len(self.validation_checksums)
        ):
            raise ValueError("score_validation_attribution_mismatch")
        if any(r.checksum is None for r in self.model_versions) or (
            self.implementation is not None and self.implementation.checksum is None
        ):
            raise ValueError("score_exact_versions_required")
        return self


class ScoreEntry(Value):
    schema_version: Literal["score-entry/1.0.0"] = "score-entry/1.0.0"
    definition: ScoreDefinition
    target: Code
    status: Literal["available", "unavailable", "not_applicable", "error"]
    value: float | Component | None
    missing_reason: (
        Literal[
            "not_provided",
            "not_calibrated",
            "no_reference_mask",
            "no_probability_output",
            "graph_not_qualified",
            "required_dependency_failed",
            "tool_failed",
            "unknown_job",
            "not_executed",
            "zero_denominator",
        ]
        | None
    )
    source: ScoreSource
    calibration: Literal["not_applicable", "uncalibrated", "unavailable"] = "not_applicable"
    numerator: int | None = Field(default=None, ge=0)
    denominator: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def coherent(self):
        if (self.status == "available") != (self.value is not None) or (self.value is None) != (
            self.missing_reason is not None
        ):
            raise ValueError("score_value_missing_exclusive")
        metric = self.definition.metric
        component = metric in {"support", "refutation", "completeness", "conflict"}
        if self.value is not None:
            if component != isinstance(self.value, Component):
                raise ValueError("score_wrong_value_layer")
            if component and self.source.qualification != "accepted_graph":
                raise ValueError("score_graph_source_required")
            if isinstance(self.value, float):
                if not math.isfinite(self.value):
                    raise ValueError("score_nonfinite")
                if (
                    self.definition.minimum is not None
                    and self.value < self.definition.minimum
                    or self.definition.maximum is not None
                    and self.value > self.definition.maximum
                ):
                    raise ValueError("score_out_of_range")
            # v1 intentionally has no calibrated disease-probability implementation.
            if metric in {"probability", "calibration_error"}:
                raise ValueError("score_calibrator_not_implemented")
        if metric == "logit" and self.calibration != "uncalibrated":
            raise ValueError("score_logit_calibration_required")
        if metric == "coverage":
            if (
                self.numerator is None
                or self.denominator is None
                or self.numerator > self.denominator
            ):
                raise ValueError("score_coverage_denominator")
            if self.denominator == 0:
                if (self.status, self.missing_reason) != ("not_applicable", "zero_denominator"):
                    raise ValueError("score_zero_denominator")
            elif self.value != self.numerator / self.denominator:
                raise ValueError("score_coverage_value")
        elif self.numerator is not None or self.denominator is not None:
            raise ValueError("score_counts_reserved_for_technical_coverage")
        return self


class TechnicalFact(Value):
    key: Token
    status: Literal[
        "SUCCEEDED", "FAILED", "PARTIAL_FAILED", "CANCELLED", "UNKNOWN_JOB", "NOT_EXECUTED"
    ]
    qc_statuses: tuple[Literal["PASS", "FAIL", "PENDING", "UNASSESSED", "REJECTED"], ...]
    required_dependencies: tuple[Token, ...]
    optional_dependencies: tuple[Token, ...]
    attempt_id: str | None = None


class ValidationFact(Value):
    validation_code: Code
    validation_checksum: Checksum
    binding_checksum: Checksum
    target_node_code: Code
    clinical_effect: Literal["SUPPORTS", "REFUTES", "NONE"]
    source_snapshot: GraphSnapshot
    accepted_graph: GraphSnapshot
    audit_event_id: Code
    audit_event_sequence: int = Field(ge=1)
    rule_refs: tuple[VersionedReference, ...] = Field(max_length=64)
    evidence_bindings: tuple[Code, ...] = Field(max_length=256)


class EngineeringReceipt(Value):
    schema_version: Literal["stop-basis-binding/2.0.0"]
    checksum: Checksum
    kind: Literal["engineering", "mixed"]
    owner: Literal["04_STOP_BASIS"]
    synthetic: bool
    trigger: Token
    reason_code: Code
    grants_medical_authority: Literal[False]


class ScoreEnvelope(Value):
    """Current graph qualification is distinct from retained local accepted facts.

    Cross-field policy, snapshot and event/validation joins are enforced by the
    public model. Current source authorization still belongs to the service.
    """

    schema_version: Literal["case-score-ledger/1.0.0"] = "case-score-ledger/1.0.0"
    checksum: Checksum
    scope: ScoreScope
    synthetic: bool
    policy_checksum: Checksum
    entries: tuple[ScoreEntry, ...] = Field(max_length=1024)
    technical: tuple[TechnicalFact, ...] = Field(max_length=128)
    graph: GraphSnapshot | None
    graph_status: Literal["accepted", "unavailable"]
    original_report: ObjectReference | None
    accepted_validations: tuple[ValidationFact, ...] = Field(default=(), max_length=256)
    engineering_stop: EngineeringReceipt | None = None
    run_status: Token
    clinical_authority: Literal["none"] = "none"

    @model_validator(mode="after")
    def coherent(self):
        if any(
            e.source.scope != self.scope or e.source.synthetic != self.synthetic
            for e in self.entries
        ):
            raise ValueError("score_cross_scope")
        if any(e.source.policy_checksum != self.policy_checksum for e in self.entries):
            raise ValueError("score_source_policy_mismatch")
        keys = [
            (e.definition.code, e.target, e.source.correlation_group, e.source.artifact_code)
            for e in self.entries
        ]
        if len(keys) != len(set(keys)):
            raise ValueError("score_duplicate_source")
        if self.graph_status == "accepted" and self.graph is None:
            raise ValueError("score_graph_missing")
        available_graph = [
            e
            for e in self.entries
            if e.source.qualification == "accepted_graph" and e.status == "available"
        ]
        if (
            self.graph_status == "unavailable"
            and available_graph
            or self.graph_status == "accepted"
            and (
                not self.accepted_validations
                or not any(isinstance(e.value, Component) for e in available_graph)
            )
        ):
            raise ValueError("score_graph_qualification")
        facts = {(v.audit_event_id, v.validation_checksum) for v in self.accepted_validations}
        if len({v.audit_event_id for v in self.accepted_validations}) != len(
            self.accepted_validations
        ) or len({v.validation_checksum for v in self.accepted_validations}) != len(
            self.accepted_validations
        ):
            raise ValueError("score_validation_identity")
        snapshots = {}
        if self.graph is not None:
            snapshots[self.graph.graph_version] = self.graph.graph_checksum
        for fact in self.accepted_validations:
            if self.graph is None or not (
                fact.source_snapshot.graph_code
                == fact.accepted_graph.graph_code
                == self.graph.graph_code
                and fact.source_snapshot.graph_version
                < fact.accepted_graph.graph_version
                <= self.graph.graph_version
            ):
                raise ValueError("score_validation_graph_mismatch")
            for snapshot in (fact.source_snapshot, fact.accepted_graph):
                if (
                    snapshots.setdefault(snapshot.graph_version, snapshot.graph_checksum)
                    != snapshot.graph_checksum
                ):
                    raise ValueError("score_validation_graph_mismatch")
        for entry in self.entries:
            source = entry.source
            pairs = set(zip(source.accepted_event_ids, source.validation_checksums, strict=True))
            if (source.qualification == "accepted_graph" or pairs) and (
                source.graph != self.graph or not pairs <= facts
            ):
                raise ValueError("score_validation_attribution_mismatch")
            # These are whole-graph 03 components, never counts attributed to a subset.
            if isinstance(entry.value, Component) and pairs != facts:
                raise ValueError("score_validation_attribution_mismatch")
        if self.engineering_stop is not None and self.engineering_stop.synthetic != self.synthetic:
            raise ValueError("score_engineering_scope")
        check_seal(self)
        return self
