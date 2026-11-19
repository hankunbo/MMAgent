"""04C immutable declarations; dataset contents and clinical publication live elsewhere."""

from typing import Annotated, Literal

from pydantic import Field, StringConstraints, model_validator

from db.knowledge.models import (
    Checksum,
    KnowledgeModel,
    KnowledgeReference,
    SemanticVersion,
    StableCode,
    Text,
    _version_order,
)
from packages.contracts import VersionedReference
from packages.graph_engine import canonical_checksum
from packages.knowledge_review.research import ResearchScope

Name = Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_]{0,127}$")]
Words = Annotated[tuple[Text, ...], Field(max_length=64)]


class SealedDeclaration(KnowledgeModel):
    checksum: Checksum

    @model_validator(mode="after")
    def verify_checksum(self):
        payload = self.model_dump(mode="json", exclude={"checksum"})
        if canonical_checksum(payload) != self.checksum:
            raise ValueError("declaration checksum mismatch")
        return self


def seal(model, **payload):
    """JSON roundtrip enforces strict nested models and the same checksum at every ingress."""
    import json

    from pydantic import BaseModel

    if "checksum" in payload or set(payload) - set(model.model_fields):
        raise ValueError("seal requires declared fields only and computes its own checksum")

    def wire(value):
        if isinstance(value, BaseModel):
            return value.model_dump(mode="json")
        if isinstance(value, (tuple, list)):
            return [wire(v) for v in value]
        if isinstance(value, dict):
            return {k: wire(v) for k, v in value.items()}
        return value

    # Include defaults before hashing, without accepting the constructed instance as validated.
    draft = model.model_construct(**payload)
    data = wire(dict(vars(draft)))
    data.pop("checksum", None)
    data["checksum"] = canonical_checksum(data)
    return model.model_validate_json(json.dumps(data, ensure_ascii=False, allow_nan=False))


class FieldVisibility(KnowledgeModel):
    policy_version: Literal["field-visibility/1.0.0"] = "field-visibility/1.0.0"
    planner_visible: Words
    withheld: Words
    owner_only: Words
    derived_requires_review: Words = ()
    unknown_fields: Literal["reject"] = "reject"
    raw_text_redactor_owner: Literal["07A", "none"]

    @model_validator(mode="after")
    def disjoint(self):
        groups = (self.planner_visible, self.withheld, self.owner_only)
        flat = sum(groups, ())
        if len(flat) != len(set(flat)):
            raise ValueError("visibility fields must be unique and disjoint")
        if not set(self.derived_requires_review) <= set(self.planner_visible):
            raise ValueError("reviewed derived field must be declared visible")
        return self


class DatasetReference(SealedDeclaration):
    schema_version: Literal["dataset-reference/1.0.0"] = "dataset-reference/1.0.0"
    code: StableCode
    version: SemanticVersion
    name: Text
    synthetic: bool
    role: Literal["case_evidence", "coarse_triage", "independent_tool_truth"]
    owner: Literal["06E", "07A"]
    modalities: Words
    declared_fields: Words
    visibility: FieldVisibility
    description_sources: Words
    local_root: Text | None = None
    expected_golden_levels: tuple[Literal["G1", "G2", "G3", "G4"], ...]
    patient_linkage: Literal["within_dataset_only", "no_patient_linkage"]
    label_purpose: Literal["evaluation_only_not_publication"] = "evaluation_only_not_publication"
    scope_notes: Words

    @model_validator(mode="after")
    def dataset_boundary(self):
        policy = self.visibility
        declared = set(policy.planner_visible + policy.withheld + policy.owner_only)
        if declared != set(self.declared_fields) or len(declared) != len(self.declared_fields):
            raise ValueError("every declared field needs exactly one visibility decision")
        if self.role == "independent_tool_truth" and (
            self.owner != "06E"
            or self.patient_linkage != "no_patient_linkage"
            or policy.planner_visible
        ):
            raise ValueError("independent tool truth cannot become patient/Planner input")
        if self.role != "independent_tool_truth" and self.owner != "07A":
            raise ValueError("case-level adapters belong to 07A")
        if self.synthetic and self.local_root is not None:
            raise ValueError("synthetic references cannot point to real dataset roots")
        return self

    def reference(self):
        return VersionedReference(code=self.code, version=self.version, checksum=self.checksum)


class EvidenceSlot(KnowledgeModel):
    node_code: StableCode
    observation_code: StableCode
    modality: Text
    origin: Literal["image_tool", "redacted_note", "external_exam"]
    capability_code: StableCode | None = None
    input_artifact_code: StableCode
    output_artifact_code: StableCode
    input_coordinate_space: Literal["PIXEL_2D", "NOT_APPLICABLE"] = "PIXEL_2D"
    output_coordinate_space: Literal["PIXEL_2D", "NOT_APPLICABLE"] = "NOT_APPLICABLE"
    missing_behavior: Literal["capability_gap_and_requirement", "evidence_missing"]

    @model_validator(mode="after")
    def capability_boundary(self):
        if self.capability_code is not None and not self.capability_code.startswith("CAPABILITY:"):
            raise ValueError("capability code must use the public CAPABILITY namespace")
        if any(
            not code.startswith("ARTIFACT:")
            for code in (self.input_artifact_code, self.output_artifact_code)
        ):
            raise ValueError("artifact code must use the public ARTIFACT namespace")
        if (self.capability_code is not None) != (
            self.missing_behavior == "capability_gap_and_requirement"
        ):
            raise ValueError("tool requirements must declare their missing-capability behavior")
        if self.origin == "external_exam" and self.capability_code is not None:
            raise ValueError("external examination cannot be synthesized by an image tool")
        return self


class PackDefinition(SealedDeclaration):
    schema_version: Literal["disease-pack-definition/1.0.0"] = "disease-pack-definition/1.0.0"
    pack_code: Name
    version: SemanticVersion
    synthetic: bool
    target_proposition: Text
    target_node_code: StableCode
    goal_level: Literal["evidence_planning", "coarse_multilabel_triage"]
    applicable_conditions: Words = Field(min_length=1)
    excluded_conditions: Words = Field(min_length=1)
    modalities: Words = Field(min_length=1)
    evidence_slots: tuple[EvidenceSlot, ...] = Field(min_length=1, max_length=32)
    dataset_refs: tuple[VersionedReference, ...] = Field(min_length=1, max_length=16)
    allowed_conclusions: tuple[
        Literal["unsupported", "insufficient", "suspicious", "evidence_supported"], ...
    ] = Field(min_length=1)
    forbidden_conclusions: Words = Field(min_length=1)
    expected_golden_levels: tuple[Literal["G1", "G2", "G3", "G4"], ...]
    triage_labels: tuple[Literal["N", "D", "G", "C", "A", "H", "M", "O"], ...] = ()
    triage_boundaries: Words = ()
    online_review_owner: Literal["08_Expert_Agent_legal_actions_only"] = (
        "08_Expert_Agent_legal_actions_only"
    )
    offline_reference_owner: Literal["07A_expert_reviewed_hidden_reference"] = (
        "07A_expert_reviewed_hidden_reference"
    )

    @model_validator(mode="after")
    def closed_definition(self):
        if not {"definitive_poag_diagnosis", "treatment_order"} <= set(self.forbidden_conclusions):
            raise ValueError("unsafe conclusions must be explicitly prohibited")
        if set(self.allowed_conclusions) & set(self.forbidden_conclusions):
            raise ValueError("contradictory conclusion policy")
        for values in (
            tuple(s.node_code for s in self.evidence_slots),
            tuple(s.observation_code for s in self.evidence_slots),
            tuple(r.code for r in self.dataset_refs),
            self.allowed_conclusions,
            self.triage_labels,
        ):
            if len(values) != len(set(values)):
                raise ValueError("duplicate definition members")
        if self.target_node_code in {s.node_code for s in self.evidence_slots}:
            raise ValueError("goal cannot also be an evidence slot")
        if self.goal_level == "coarse_multilabel_triage" and (
            set(self.triage_labels) != set("NDGCAHMO") or not self.triage_boundaries
        ):
            raise ValueError("ODIR requires all eight labels and explicit coarse scope boundaries")
        return self


class KnowledgeBindings(KnowledgeModel):
    release_ref: KnowledgeReference
    claim_refs: tuple[KnowledgeReference, ...] = Field(min_length=1)
    evidence_requirement_refs: tuple[KnowledgeReference, ...] = Field(min_length=1)
    template_ref: KnowledgeReference
    rule_refs: tuple[KnowledgeReference, ...] = Field(min_length=1)
    stop_rule_refs: tuple[KnowledgeReference, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def reference_kinds(self):
        for name, kind in (
            ("release_ref", "release"),
            ("claim_refs", "clinical_claim"),
            ("evidence_requirement_refs", "evidence_requirement"),
            ("template_ref", "template"),
            ("rule_refs", "validation_rule"),
            ("stop_rule_refs", "stop_rule"),
        ):
            value = getattr(self, name)
            refs = value if isinstance(value, tuple) else (value,)
            if any(r.kind != kind for r in refs) or len(set(refs)) != len(refs):
                raise ValueError("binding kinds or identities mismatch")
        return self

    def artifacts(self):
        return (self.template_ref, *self.rule_refs, *self.stop_rule_refs, *self.claim_refs)


class DiseasePack(SealedDeclaration):
    schema_version: Literal["disease-pack/1.0.0"] = "disease-pack/1.0.0"
    definition: PackDefinition
    status: Literal["Draft", "Reviewed", "Published"]
    engineering_level: Literal["L3"] = "L3"
    clinical_level: Literal["pending_L4", "L4", "synthetic_only"]
    bindings: KnowledgeBindings | None = None
    knowledge_gaps: Words
    supersedes: VersionedReference | None = None

    @model_validator(mode="after")
    def publication_is_not_a_flag(self):
        if self.status == "Published" and (self.bindings is None or self.knowledge_gaps):
            raise ValueError("Published requires complete exact Knowledge bindings and no gaps")
        if (
            self.status == "Published"
            and not self.definition.synthetic
            and self.clinical_level != "L4"
        ):
            raise ValueError("non-synthetic Published declaration requires clinical L4 admission")
        if self.clinical_level == "L4" and (
            self.status != "Published" or self.definition.synthetic
        ):
            raise ValueError("L4 requires non-synthetic Published content and live admission")
        if self.definition.synthetic != (self.clinical_level == "synthetic_only"):
            raise ValueError("synthetic engineering approval is never clinical L4")
        if self.status != "Published" and not self.knowledge_gaps:
            raise ValueError("unpublished content must expose its KnowledgeGap")
        if self.supersedes is not None and (
            self.supersedes.code != self.reference().code
            or _version_order(self.supersedes.version) >= _version_order(self.definition.version)
        ):
            raise ValueError("supersedes must preserve identity and change version")
        return self

    def reference(self):
        return VersionedReference(
            code="POLICY:DISEASE_PACK:" + self.definition.pack_code.upper(),
            version=self.definition.version,
            checksum=self.checksum,
        )


class ResearchDiseasePack(SealedDeclaration):
    """New declaration: Published research never asserts clinical evaluation approval."""

    schema_version: Literal["disease-pack/2.0.0"] = "disease-pack/2.0.0"
    definition: PackDefinition
    status: Literal["Draft", "Reviewed", "Published"]
    engineering_level: Literal["L3"] = "L3"
    clinical_level: Literal["pending_L4"]
    research_scope: ResearchScope
    research_policy_ref: KnowledgeReference
    bindings: KnowledgeBindings | None = None
    knowledge_gaps: Words
    supersedes: VersionedReference | None = None

    @model_validator(mode="after")
    def research_boundary(self):
        if self.research_policy_ref.kind != "release_policy":
            raise ValueError("KNO_B03_POLICY_REQUIRED")
        if self.research_scope.definition_ref != definition_reference(self.definition):
            raise ValueError("KNO_B03_DEFINITION_BINDING")
        if self.definition.goal_level != "evidence_planning" or set(
            self.definition.allowed_conclusions
        ) != {"insufficient"}:
            raise ValueError("KNO_B03_OUTPUT_SCOPE")
        if self.status == "Published" and (self.bindings is None or self.knowledge_gaps):
            raise ValueError("Published requires complete exact Knowledge bindings and no gaps")
        if self.status != "Published" and not self.knowledge_gaps:
            raise ValueError("unpublished content must expose its KnowledgeGap")
        if self.supersedes is not None and (
            self.supersedes.code != self.reference().code
            or _version_order(self.supersedes.version) >= _version_order(self.definition.version)
        ):
            raise ValueError("supersedes must preserve identity and change version")
        return self

    def reference(self):
        return VersionedReference(
            code=definition_reference(self.definition).code,
            version=self.definition.version,
            checksum=self.checksum,
        )


def definition_reference(definition):
    return VersionedReference(
        code="POLICY:DISEASE_PACK:" + definition.pack_code.upper(),
        version=definition.version,
        checksum=definition.checksum,
    )


def parse_pack(value):
    import json

    if isinstance(value, (DiseasePack, ResearchDiseasePack)):
        value = value.model_dump_json()
    if isinstance(value, (str, bytes)):
        value = json.loads(value)
    model = {"disease-pack/1.0.0": DiseasePack, "disease-pack/2.0.0": ResearchDiseasePack}.get(
        value.get("schema_version")
    )
    if model is None:
        raise ValueError("KNO_B03_DECLARATION_VERSION")
    return model.model_validate_json(json.dumps(value))
