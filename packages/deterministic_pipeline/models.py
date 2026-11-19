"""Versioned application inputs, separate from frozen contracts and preparation snapshots."""

from datetime import datetime
from typing import Literal

from pydantic import Field, model_validator

from packages.contracts import CaseBundle, VersionedReference
from packages.dataset_adapters.models import ObjectReference, VisibleManifest
from packages.grounding.models import Ref, Value, check_seal
from packages.grounding.query_models import QueryConfig
from packages.knowledge_index.models import IndexConfig
from packages.retrieval.knowledge.models import RetrievalContext
from packages.retrieval.knowledge.research import ResearchRetrievalContext

from .demands import DemandResolution


class PipelineError(ValueError):
    """Sanitized stable application failure; never raw source or provider error text."""


class PipelinePolicy(Value):
    version: Literal["deterministic-pipeline/1.0.0"] = "deterministic-pipeline/1.0.0"
    checksum: str
    baseline_only: Literal[True] = True
    max_actions: int = Field(ge=1, le=32)
    max_graph_rounds: int = Field(ge=1, le=64)
    max_seconds: int = Field(ge=1, le=3600)
    max_cost: float = Field(gt=0, le=10000)
    unknown_cost_reservation: float = Field(default=1.0, gt=0, le=10000)
    max_same_action: Literal[1] = 1
    selection: Literal["grounding_order_existing_first/1.0.0"] = (
        "grounding_order_existing_first/1.0.0"
    )
    network: Literal["deny"] = "deny"

    @model_validator(mode="after")
    def sealed(self):
        check_seal(self)
        return self


class PipelineConfig(Value):
    version: Literal["case-pipeline-config/1.0.0", "case-pipeline-config/2.0.0"] = (
        "case-pipeline-config/1.0.0"
    )
    checksum: str
    pack_ref: VersionedReference
    catalog_ref: Ref
    mapping_refs: tuple[Ref, ...] = ()
    context: RetrievalContext | ResearchRetrievalContext
    query: str = Field(max_length=256)
    retrieval: IndexConfig
    grounding: QueryConfig
    policy: PipelinePolicy
    # Exact deployment/QC/validation registry and compiler/execution policy identities.
    service_refs: tuple[VersionedReference, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def sealed(self):
        if (self.version == "case-pipeline-config/2.0.0") != (
            getattr(self.context, "use", None) is not None
        ):
            raise ValueError("KNO_B03_NEW_CONFIG_REQUIRED")
        if (
            getattr(self.context, "use", None) is not None
            and self.context.use.purpose != "limited_research_evidence_planning"
        ):
            raise ValueError("KNO_B03_CLINICAL_DENIED")
        if self.version == "case-pipeline-config/2.0.0" and self.mapping_refs:
            raise ValueError("KNO_B03_DEMAND_ONLY_CONFIG")
        if self.catalog_ref.kind != "catalog" or any(
            r.kind != "mapping" for r in self.mapping_refs
        ):
            raise ValueError("pipeline_reference_kind")
        if any(r.checksum is None for r in (self.pack_ref, *self.service_refs)):
            raise ValueError("pipeline_exact_reference_required")
        check_seal(self)
        return self


class CaseInput(Value):
    """Real case always retains the full VisibleManifest; synthetic input is separate."""

    synthetic: bool
    case: CaseBundle
    pack_ref: VersionedReference
    input_checksum: str
    visible: VisibleManifest | None

    @model_validator(mode="after")
    def boundary(self):
        from packages.graph_engine import canonical_checksum

        if self.synthetic:
            if self.visible is not None or self.input_checksum != canonical_checksum(self.case):
                raise ValueError("synthetic_input_binding")
            if self.case.observations or self.case.report_uri or self.case.extensions:
                raise ValueError("synthetic_visible_input_only")
        elif (
            self.visible is None
            or self.visible.case != self.case
            or self.visible.pack_ref != self.pack_ref
            or self.visible.checksum != self.input_checksum
        ):
            raise ValueError("complete_visible_manifest_required")
        return self


class RequestState(Value):
    version: Literal["case-pipeline-request/1.0.0", "case-pipeline-request/2.0.0"] = (
        "case-pipeline-request/1.0.0"
    )
    case_code: str
    input_checksum: str
    config_checksum: str
    synthetic: bool
    started_at: datetime
    run_id: str | None = None
    revision: int = Field(default=0, ge=0)
    trace: tuple[ObjectReference, ...] = Field(default=(), max_length=256)
    report: ObjectReference | None = None
    actions: tuple[str, ...] = Field(default=(), max_length=32)
    cost: float = Field(default=0, ge=0)
    demand_resolution: DemandResolution | None = None

    @model_validator(mode="after")
    def demand_binding(self):
        d = self.demand_resolution
        if d is not None and (
            self.version != "case-pipeline-request/2.0.0"
            or (d.run_id, d.input_checksum, d.config_checksum)
            != (self.run_id, self.input_checksum, self.config_checksum)
            or (d.state == "unresolved" and (self.actions or self.cost))
        ):
            raise ValueError("request_demand_binding_conflict")
        if self.version == "case-pipeline-request/2.0.0" and d is None:
            raise ValueError("request_demand_required")
        return self
