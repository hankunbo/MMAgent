"""Offline SYNTHETIC fixtures only. No real medgraph connection or tool execution."""

from .models import (
    ArtifactConstraint,
    CatalogSnapshot,
    CatalogTool,
    Input,
    Mapping,
    Policy,
    PortRef,
    Ref,
    Term,
    digest,
    seal,
    validate,
)
from .ports import KnowledgeProjection, ToolSafety
from .validation import GroundingError


class SyntheticCatalogReader:
    def __init__(self, snapshots):
        self._snapshots = {}
        self._disabled = set()
        self._revision = 0
        for snapshot in snapshots:
            self.add(snapshot)

    def add(self, snapshot):
        snapshot = validate(CatalogSnapshot, snapshot)
        if not snapshot.synthetic:
            raise GroundingError("synthetic_fixture_required")
        key = (snapshot.code, snapshot.version)
        if key in self._snapshots and self._snapshots[key] != snapshot:
            raise GroundingError("immutable_catalog_conflict")
        self._snapshots[key] = snapshot

    def disable(self, implementation):
        self._disabled.add(implementation)
        self._revision += 1

    def read_snapshot(self, reference):
        reference = validate(Ref, reference)
        value = self._snapshots.get((reference.code, reference.version))
        if value is None or value.reference() != reference:
            raise GroundingError("exact_catalog_not_found")
        return value

    def current_safety(self, reference):
        self.read_snapshot(reference)
        return ToolSafety(
            catalog_ref=reference,
            disabled_implementations=tuple(sorted(self._disabled, key=lambda r: r.code)),
            revision=self._revision,
        )


class SyntheticKnowledgeReader:
    def __init__(self, projection):
        self.projection = validate(KnowledgeProjection, projection)
        self.withdrawn = False
        self.published = True

    def read(self, mapping, context, *, replay):
        if not replay and (self.withdrawn or not self.published):
            raise GroundingError("knowledge_not_published_or_withdrawn")
        return validate(
            KnowledgeProjection,
            {
                **self.projection.model_dump(mode="json"),
                "mode": "replay" if replay else "published",
            },
        )


def ref(kind, code):
    return Ref(kind=kind, code=code, version="1.0.0", checksum=digest({"synthetic": code}))


def fixture(*, version="1.0.0", code="BINDING:OID_SYNTHETIC05A"):
    """Two OR implementations, each with two required AND inputs and one optional input."""
    from packages.contracts import CoordinateSpace, Laterality, QCStatus

    policy = seal(
        Policy,
        code="POLICY:GROUNDING:SYNTHETIC",
        version="1.0.0",
        baseline_only=True,
        synthetic_only=True,
        admissible_states=["Active"],
        type_compatibility="exact",
        accepted_validation=["auto_validated_research"],
        intended_use="research_only",
        execution_policy="research",
        max_bytes=24576,
        max_candidates=16,
        max_inputs=16,
        max_artifacts=16,
        max_mappings=32,
        allowed_terms=["CONCEPT:SYNTHETIC", "CONCEPT:MODALITY:SYNTHETIC", "CONCEPT:UNIT:RATIO"],
    )
    artifact_refs = tuple(
        ref("artifact", c)
        for c in (
            "ARTIFACT:SYNTHETIC_STRUCTURE",
            "retina:SyntheticRatio",
            "retina:SyntheticDiscMask",
            "retina:SyntheticCupMask",
            "retina:SyntheticImage",
        )
    )

    def constraint(a):
        return ArtifactConstraint(
            artifact_ref=a,
            modality=Term(state="known", code="CONCEPT:MODALITY:SYNTHETIC"),
            unit=Term(state="not_applicable", code=None),
            laterality=Laterality.NOT_APPLICABLE,
            coordinate_space=CoordinateSpace.NOT_APPLICABLE,
            qc_required=(QCStatus.PASS,),
            scope_terms=("CONCEPT:SYNTHETIC",),
        )

    outputs = tuple(constraint(a) for a in artifact_refs[:2])
    native_manifests = []
    tools = []
    for n in range(2):
        raw = {
            "code": f"synthetic_ratio_{n}",
            "name_zh": "合成端口契约",
            "kind": "metric",
            "description": "SYNTHETIC SOFTWARE FIXTURE ONLY",
            "version": "1.0.0",
            "runtime": {},
            "metadata": {"synthetic": True},
            "enabled": True,
            "ports": [
                {
                    "direction": "input",
                    "port_name": name,
                    "artifact_code": artifact_refs[idx].code,
                    "artifact_name_zh": "合成",
                    "required": required,
                    "cardinality_min": int(required),
                    "cardinality_max": 1,
                    "constraints": {},
                }
                for name, idx, required in [
                    ("disc_mask", 2, True),
                    ("cup_mask", 3, True),
                    ("image", 4, False),
                ]
            ]
            + [
                {
                    "direction": "output",
                    "port_name": "ratio",
                    "artifact_code": artifact_refs[n].code,
                    "artifact_name_zh": "合成",
                    "required": True,
                    "cardinality_min": 1,
                    "cardinality_max": 1,
                    "constraints": {},
                }
            ],
        }
        native_manifests.append(raw)
        tool_ref = Ref(kind="tool", code=raw["code"], version=raw["version"], checksum=digest(raw))
        inputs = tuple(
            Input(
                port=PortRef(tool_ref=tool_ref, direction="input", port_name=p["port_name"]),
                constraint=constraint(artifact_refs[idx]),
                required=p["required"],
                cardinality_min=p["cardinality_min"],
                cardinality_max=p["cardinality_max"],
            )
            for p, idx in zip(raw["ports"][:3], (2, 3, 4), strict=True)
        )
        tools.append(
            validate(
                CatalogTool,
                dict(
                    tool_ref=tool_ref,
                    implementation_ref=ref("implementation", f"synthetic_impl_{n}"),
                    capabilities=[ref("capability", "CAPABILITY:SYNTHETIC_RATIO")],
                    inputs=inputs,
                    outputs=[outputs[n]],
                    output_ports=[
                        PortRef(tool_ref=tool_ref, direction="output", port_name="ratio")
                    ],
                    workflows=[ref("workflow", f"synthetic_workflow_{n}")],
                    enabled=True,
                    catalog_status="active",
                    validation_status="auto_validated_research",
                    intended_use="research_only",
                ),
            )
        )
    catalog = seal(
        CatalogSnapshot,
        code="TOOL:CATALOG:SYNTHETIC",
        version=version,
        synthetic=True,
        manifest_checksum=digest(native_manifests),
        artifacts=artifact_refs,
        tools=tools,
        estimates=[],
    )
    fields = dict(
        requirement_ref=ref("evidence_requirement", "SOURCE:SYNTHETIC:REQUIREMENT"),
        claim_refs=[ref("clinical_claim", "SOURCE:SYNTHETIC:CLAIM")],
        release_ref=ref("release", "SOURCE:SYNTHETIC:RELEASE"),
        template_ref=ref("template", "TEMPLATE:SYNTHETIC"),
        source_refs=[ref("source_version", "SOURCE:SYNTHETIC:VERSION")],
        applicability_refs=[ref("applicability_condition", "SOURCE:SYNTHETIC:APPLICABILITY")],
    )
    mapping = seal(
        Mapping,
        code=code,
        version=version,
        synthetic=True,
        **fields,
        catalog_ref=catalog.reference(),
        policy_ref=policy.reference(),
        accepted_artifacts=outputs,
        candidates=[
            dict(
                code=f"CAPABILITY:SYNTHETIC:PATH{n}",
                capability_ref=t.capabilities[0],
                tool_ref=t.tool_ref,
                implementation_ref=t.implementation_ref,
                workflow_ref=t.workflows[0],
                output_port=t.output_ports[0],
                output=t.outputs[0],
                inputs=t.inputs,
                validation_required="auto_validated_research",
                intended_use="research_only",
                execution_policy="research",
                estimate_ref=None,
            )
            for n, t in enumerate(tools)
        ],
    )
    knowledge = SyntheticKnowledgeReader(
        validate(
            KnowledgeProjection,
            dict(
                **fields,
                graph_template_ref=fields["template_ref"],
                target_node_code="EVIDENCE:SYNTHETIC:NODE",
                artifact_type_code=artifact_refs[0].code,
                synthetic=True,
                mode="published",
            ),
        )
    )
    return mapping, policy, catalog, knowledge, native_manifests
