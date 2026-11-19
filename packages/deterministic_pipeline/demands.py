"""07-owned demand snapshots. These records neither authorize tools nor enqueue 11."""

from typing import Literal

from pydantic import Field, model_validator

from db.knowledge.models import KnowledgeReference
from packages.contracts import DataAvailability, QCStatus, VersionedReference
from packages.graph_engine import Applicability, NodeResolution
from packages.grounding.locks import GroundingLock
from packages.grounding.models import (
    CatalogSnapshot,
    Checksum,
    Ref,
    Value,
    check_seal,
    digest,
    seal,
    validate,
)


def mapped_node_resolution(node_code, *, synthetic):
    """A legal Mapping cannot establish clinical applicability or input quality.

    Preserve the explicit synthetic baseline. Real evidence remains unvalidated
    until the reviewed patient/rule binding supplies a more specific resolution.
    """
    if type(synthetic) is not bool:
        raise ValueError("explicit_input_scope_required")
    return NodeResolution(
        node_code=node_code,
        applicability=Applicability.APPLICABLE if synthetic else Applicability.UNKNOWN,
        availability=DataAvailability.DERIVABLE if synthetic else DataAvailability.UNAVAILABLE,
        reason_code="RULE:INSTANTIATION:EXPLICIT"
        if synthetic
        else "RULE:INSTANTIATION:UNVALIDATED_INPUT",
    )


class Demand(Value):
    requirement_ref: Ref
    node_ref: KnowledgeReference
    acceptance_rule_ref: KnowledgeReference
    input_artifact_code: str = Field(pattern=r"^ARTIFACT:", max_length=256)
    output_artifact_code: str = Field(pattern=r"^ARTIFACT:", max_length=256)
    input_assets: tuple[str, ...] = Field(max_length=128)
    input_state: Literal["present", "missing"]
    input_qc: tuple[QCStatus, ...] = Field(max_length=128)
    applicability: Literal["APPLICABLE", "UNKNOWN"]
    # Evidence availability, not the presence of an unvalidated source image.
    availability: Literal["UNAVAILABLE"] = "UNAVAILABLE"
    producer_refs: tuple[Ref, ...] = Field(max_length=128)
    producer_qualification: Literal["discovery_only"] = "discovery_only"
    reasons: tuple[
        Literal[
            "mapping_not_configured",
            "producer_unresolved",
            "patient_input_missing",
            "input_qc_unassessed",
            "input_qc_pending",
            "input_qc_failed",
            "applicability_unknown",
            "catalog_unreadable",
            "catalog_version_conflict",
            "catalog_scope_conflict",
            "reviewed_result_interpreter_required",
        ],
        ...,
    ] = Field(min_length=1)

    @model_validator(mode="after")
    def shape(self):
        if (self.requirement_ref.kind, self.node_ref.kind, self.acceptance_rule_ref.kind) != (
            "evidence_requirement",
            "node",
            "validation_rule",
        ):
            raise ValueError("demand_reference_kind")
        if self.input_state != ("present" if self.input_assets else "missing"):
            raise ValueError("demand_input_binding")
        if len(self.input_qc) != len(self.input_assets):
            raise ValueError("demand_qc_binding")
        if any(r.kind != "implementation" for r in self.producer_refs):
            raise ValueError("demand_producer_kind")
        return self


class DemandResolution(Value):
    schema_version: Literal["demand-resolution/1.0.0"] = "demand-resolution/1.0.0"
    checksum: Checksum
    state: Literal["mapped", "unresolved"]
    run_id: str
    graph_code: str
    graph_version: Literal[1] = 1
    graph_checksum: Checksum
    input_checksum: Checksum
    config_checksum: Checksum
    catalog_ref: Ref
    release_ref: Ref
    template_ref: Ref
    rule_refs: tuple[KnowledgeReference, ...] = Field(min_length=1)
    manifest_checksum: Checksum
    resolver_ref: VersionedReference | None = None
    catalog_state: Literal[
        "not_queried", "readable", "unreadable", "version_conflict", "scope_conflict"
    ]
    catalog_manifest_checksum: Checksum | None = None
    demands: tuple[Demand, ...] = Field(default=(), max_length=128)
    grounding_lock: GroundingLock | None = None
    grants_execution: Literal[False] = False
    extension_enqueued: Literal[False] = False

    @model_validator(mode="after")
    def shape(self):
        if (
            self.catalog_ref.kind != "catalog"
            or self.release_ref.kind != "release"
            or self.template_ref.kind != "template"
        ):
            raise ValueError("demand_envelope_reference_kind")
        if self.state == "unresolved":
            if self.grounding_lock is not None or not self.demands or self.resolver_ref is None:
                raise ValueError("explicit_unresolved_demand_required")
            if self.catalog_state == "not_queried":
                raise ValueError("demand_catalog_result_required")
            if self.resolver_ref.checksum is None:
                raise ValueError("demand_exact_resolver_required")
            if any(d.acceptance_rule_ref not in self.rule_refs for d in self.demands):
                raise ValueError("demand_rule_outside_envelope")
            if self.catalog_state != "readable" and any(
                d.producer_refs or "producer_unresolved" in d.reasons for d in self.demands
            ):
                raise ValueError("environment_failure_cannot_resolve_producers")
        else:
            lock = self.grounding_lock
            if lock is None or self.demands or self.catalog_state != "not_queried":
                raise ValueError("original_grounding_lock_required")
            if (
                lock.run_id,
                lock.graph_code,
                lock.graph_checksum,
                lock.catalog_ref,
                lock.release_ref,
                lock.template_ref,
            ) != (
                self.run_id,
                self.graph_code,
                self.graph_checksum,
                self.catalog_ref,
                self.release_ref,
                self.template_ref,
            ):
                raise ValueError("demand_lock_scope_conflict")
        if (self.catalog_state == "readable") != (self.catalog_manifest_checksum is not None):
            raise ValueError("demand_catalog_manifest_binding")
        if len({d.node_ref.code for d in self.demands}) != len(self.demands):
            raise ValueError("duplicate_demand_node")
        check_seal(self)
        return self


class DemandResolver:
    """Read-only Published requirements and exact catalog. No Mapping/executor dependency.

    The conservative projection supplies no medical interpretation. Unknown patient
    applicability remains UNKNOWN; source presence/QC never makes evidence AVAILABLE.
    Actual clinical bindings require a separately reviewed, exact service version.
    """

    reference = VersionedReference(
        code="POLICY:DEMAND_RESOLUTION",
        version="1.0.0",
        checksum=digest(
            {
                "projection": "visible_sources_unvalidated",
                "unknown_scope": "retain_unavailable",
                "execution": False,
            }
        ),
    )

    def __init__(self, *, knowledge, catalogs):
        self.knowledge = knowledge
        self.catalogs = catalogs

    def resolve(self, input_value, admission, config):
        if config.mapping_refs or admission.knowledge_gap or admission.pack.status != "Published":
            raise ValueError("unresolved_requires_admitted_unmapped_request")
        catalog = None
        state = "unreadable"
        if self.catalogs is not None:
            try:
                catalog = validate(CatalogSnapshot, self.catalogs.read_snapshot(config.catalog_ref))
            except Exception:
                # Never persist raw environment/provider errors or infer catalog absence.
                catalog = None
            if catalog is not None:
                state = "readable"
                if catalog.reference() != config.catalog_ref:
                    state, catalog = "version_conflict", None
                elif catalog.synthetic != input_value.synthetic:
                    state, catalog = "scope_conflict", None
        rows = []
        resolutions = []
        for ref in admission.pack.bindings.evidence_requirement_refs:
            requirement = self.knowledge.get_published_evidence(ref, context=config.context).record
            if requirement.reference() != ref:
                raise ValueError("demand_requirement_version_conflict")
            slot = next(
                s
                for s in admission.pack.definition.evidence_slots
                if s.node_code == requirement.node_ref.code
            )
            if requirement.acceptance_rule_ref not in admission.pack.bindings.rule_refs:
                raise ValueError("demand_rule_outside_release")
            assets = tuple(
                a
                for a in input_value.case.artifacts
                if a.artifact_type_code == slot.input_artifact_code
            )
            unknown = input_value.case.laterality.value == "UNKNOWN" or not input_value.synthetic
            applicability = "UNKNOWN" if unknown else "APPLICABLE"
            producers = (
                tuple(
                    sorted(
                        {
                            t.implementation_ref
                            for t in catalog.tools
                            if any(
                                o.artifact_ref.code == requirement.artifact_type_code
                                for o in t.outputs
                            )
                        },
                        key=lambda r: r.model_dump_json(),
                    )
                )
                if catalog
                else ()
            )
            reasons = ["mapping_not_configured"]
            if not input_value.synthetic:
                reasons.append("reviewed_result_interpreter_required")
            if not assets:
                reasons.append("patient_input_missing")
            if any(a.qc_status.value == "UNASSESSED" for a in assets):
                reasons.append("input_qc_unassessed")
            if any(a.qc_status.value == "PENDING" for a in assets):
                reasons.append("input_qc_pending")
            if any(a.qc_status.value == "FAIL" for a in assets):
                reasons.append("input_qc_failed")
            if unknown:
                reasons.append("applicability_unknown")
            if catalog is None:
                reasons.append(
                    "catalog_"
                    + (
                        "version_conflict"
                        if state == "version_conflict"
                        else "scope_conflict"
                        if state == "scope_conflict"
                        else "unreadable"
                    )
                )
            elif not producers:
                reasons.append("producer_unresolved")
            demand = Demand(
                requirement_ref=Ref.model_validate_json(ref.model_dump_json()),
                node_ref=requirement.node_ref,
                acceptance_rule_ref=requirement.acceptance_rule_ref,
                input_artifact_code=slot.input_artifact_code,
                output_artifact_code=requirement.artifact_type_code,
                input_assets=tuple(a.asset_code for a in assets),
                input_state="present" if assets else "missing",
                input_qc=tuple(a.qc_status for a in assets),
                applicability=applicability,
                producer_refs=producers,
                reasons=tuple(reasons),
            )
            rows.append(demand)
            resolutions.append(
                NodeResolution(
                    slot.node_code,
                    Applicability(applicability),
                    DataAvailability.UNAVAILABLE,
                    reason_code="RULE:INSTANTIATION:UNRESOLVED_DEMAND",
                    rule_ref=requirement.acceptance_rule_ref.to_versioned_reference(),
                )
            )
        return (
            tuple(rows),
            tuple(resolutions),
            state,
            catalog.manifest_checksum if catalog else None,
        )


def envelope(*, input_value, config, admission, graph, lock=None, resolver=None, resolved=None):
    demands, _, catalog_state, catalog_manifest = resolved or ((), (), "not_queried", None)
    from packages.graph_engine import canonical_checksum

    return seal(
        DemandResolution,
        state="mapped" if lock is not None else "unresolved",
        run_id=graph.scope.run_id,
        graph_code=graph.graph_code,
        graph_checksum=canonical_checksum(graph),
        input_checksum=input_value.input_checksum,
        config_checksum=config.checksum,
        catalog_ref=config.catalog_ref,
        release_ref=Ref.model_validate_json(admission.pack.bindings.release_ref.model_dump_json()),
        template_ref=Ref.model_validate_json(
            admission.pack.bindings.template_ref.model_dump_json()
        ),
        rule_refs=admission.pack.bindings.rule_refs,
        manifest_checksum=admission.manifest_checksum,
        resolver_ref=resolver.reference if resolver else None,
        catalog_state=catalog_state,
        catalog_manifest_checksum=catalog_manifest,
        demands=demands,
        grounding_lock=lock,
    )
