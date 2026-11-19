"""Anonymous, synthetic-only Knowledge and graph fixtures through existing public ports."""

import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from db.knowledge import models as m
from db.knowledge.repository.review import bind_artifact
from packages.contracts import CaseBundle, EdgeType, GECEGTemplate, NodeType, ValidationScope
from packages.disease_packs import (
    DatasetReference,
    DiseasePack,
    EvidenceSlot,
    FieldVisibility,
    KnowledgeBindings,
    PackDefinition,
    seal,
)
from packages.knowledge_index.models import IndexConfig
from packages.retrieval.knowledge.models import RetrievalContext
from tests.database.test_knowledge_release_04b6 import policy, publish, review
from tests.knowledge_retrieval.support import make_claim

ROOT = Path(__file__).resolve().parents[2]


def code(prefix):
    return prefix + ":OID_C" + uuid4().hex.upper()


def config():
    return IndexConfig(
        mode="fts_only",
        vector_unavailable_reason="synthetic offline fixture",
        vector_weight=0.0,
        top_k=16,
    )


def context():
    return RetrievalContext(
        population=("synthetic_fixture_only", "unspecified_in_source"),
        conditions=("software_test",),
        modalities=("synthetic",),
        exams=("OCT",),
    )


def declarations():
    dataset = seal(
        DatasetReference,
        code=code("SOURCE"),
        version="1.0.0",
        name="Synthetic reference",
        synthetic=True,
        role="case_evidence",
        owner="07A",
        modalities=("synthetic",),
        declared_fields=("evidence_note_redacted", "label", "raw_note"),
        visibility=FieldVisibility(
            planner_visible=("evidence_note_redacted",),
            withheld=("label", "raw_note"),
            owner_only=(),
            derived_requires_review=("evidence_note_redacted",),
            raw_text_redactor_owner="07A",
        ),
        description_sources=("anonymous software fixture",),
        expected_golden_levels=("G1",),
        patient_linkage="within_dataset_only",
        scope_notes=("no actual patient",),
    )
    slot = EvidenceSlot(
        node_code=code("EVIDENCE"),
        observation_code="CONCEPT:SYNTHETIC_STRUCTURE",
        modality="synthetic",
        origin="image_tool",
        capability_code="CAPABILITY:SLO_QC",
        input_artifact_code="ARTIFACT:SLO_IMAGE",
        output_artifact_code="ARTIFACT:QC_REPORT",
        missing_behavior="capability_gap_and_requirement",
    )
    definition = seal(
        PackDefinition,
        pack_code="synthetic_ophthalmology_evidence",
        version="1.0.0",
        synthetic=True,
        target_proposition="Synthetic evidence planning; no clinical assertion",
        target_node_code=code("GOAL"),
        goal_level="evidence_planning",
        applicable_conditions=("software_test",),
        excluded_conditions=("clinical_use",),
        modalities=("synthetic",),
        evidence_slots=(slot,),
        dataset_refs=(dataset.reference(),),
        allowed_conclusions=("unsupported", "insufficient", "suspicious", "evidence_supported"),
        forbidden_conclusions=("definitive_poag_diagnosis", "treatment_order"),
        expected_golden_levels=("G1",),
    )
    return dataset, definition


def assembled(
    db,
    store,
    *,
    version="1.0.0",
    edge_type=EdgeType.SUPPORTS,
    connected=False,
    scope=None,
    evidence_overrides=None,
    release_policy=None,
    evidence_claims=None,
    slot_count=1,
    research=False,
    simulated_real=False,
):
    dataset, definition = declarations()
    if slot_count > 1:
        slots = tuple(
            definition.evidence_slots[0].model_copy(
                update={"node_code": code("EVIDENCE"), "observation_code": code("CONCEPT")}
            )
            for _ in range(slot_count)
        )
        definition = seal(
            PackDefinition,
            **(
                definition.model_dump(mode="python", exclude={"checksum"})
                | {"evidence_slots": slots}
            ),
        )
    if version != definition.version:
        definition = seal(
            PackDefinition,
            **(definition.model_dump(mode="python", exclude={"checksum"}) | {"version": version}),
        )
    if simulated_real and not research:
        raise ValueError("simulated real content requires the explicit research fixture")
    if research:
        from packages.disease_packs import ResearchDiseasePack
        from packages.disease_packs.models import definition_reference
        from packages.knowledge_review.research import ResearchScope, context_checksum
        from tests.disease_packs.research_support import research_claim

        dataset = seal(
            DatasetReference,
            **(dataset.model_dump(exclude={"checksum"}) | {"synthetic": not simulated_real}),
        )
        definition = seal(
            PackDefinition,
            **(
                definition.model_dump(exclude={"checksum"})
                | {
                    "synthetic": not simulated_real,
                    "dataset_refs": (dataset.reference(),),
                    "allowed_conclusions": ("insufficient",),
                }
            ),
        )
        use_scope = ResearchScope(
            purpose="limited_research_evidence_planning",
            definition_ref=definition_reference(definition),
            context_checksum=context_checksum(context()),
            allowed_outputs=("evidence_requirements", "insufficiency", "limitations"),
        )
        claim, hit, build, condition = research_claim(
            db, store, config(), synthetic=not simulated_real
        )
        p = release_policy or policy(
            db,
            synthetic=not simulated_real,
            baseline_only=False,
            synthetic_only=False,
            formal_review_policy_frozen=True,
            research_scope=use_scope,
        )
    else:
        claim, hit, build, condition = make_claim(db, store, config(), scope=scope)
        p = release_policy or policy(db)
    other = None
    if connected:
        other, _, _, _ = make_claim(db, store, config(), anatomy=("other_anatomy",))
        conflicts = db.editor.detect_claim_conflicts((claim.reference(), other.reference()))
        conflict_review = review(
            db, conflicts[0].record, p, disposition="conditional_coexist", transition=False
        )
        db.editor.resolve_claim_conflict(
            conflicts[0].record.reference(), conflict_review.reference(), version="1.1.0"
        )
    cr = review(db, claim, p)
    publish(db, store, claim, cr, p)
    common = dict(
        version="1.0.0",
        synthetic=not simulated_real,
        source_refs=claim.source_refs,
        applicability_refs=claim.applicability_refs,
    )

    def bound(model, **fields):
        selected_claim, selected_review = (evidence_claims or {}).get(model, (claim, cr))
        fields = (
            common
            | fields
            | dict(
                source_refs=selected_claim.source_refs,
                applicability_refs=selected_claim.applicability_refs,
            )
        )
        fields |= (evidence_overrides or {}).get(model, {})
        obj = bind_artifact(
            db.editor,
            m.seal(model, **fields),
            ((selected_claim.reference(), selected_review.reference()),),
            p.reference(),
        )
        db.editor.put(obj)
        return obj

    rule = bound(
        m.ValidationRule,
        code=code("RULE"),
        rule_scope=ValidationScope.EVIDENCE,
        expression_code="RULE:SYNTHETIC_ASSERTION",
        claim_refs=(claim.reference(),),
    )
    rr = review(db, rule, p)
    publish(db, store, rule, rr, p)
    stop = bound(
        m.StopRule,
        code=code("RULE"),
        reason_code="RULE:SYNTHETIC_PATH_EXHAUSTED",
        claim_refs=(claim.reference(),),
    )
    sr = review(db, stop, p)
    publish(db, store, stop, sr, p)
    concept = m.seal(m.Concept, code=code("CONCEPT"), name="Synthetic evidence", **common)
    db.editor.put(concept)
    nodes = tuple(
        m.TemplateNode(node_code=n, node_type=NodeType(t), concept_code=concept.code, required=True)
        for n, t in (
            (definition.target_node_code, "GOAL"),
            *((slot.node_code, "EVIDENCE") for slot in definition.evidence_slots),
        )
    )
    edges = tuple(
        m.GraphEdge(
            source_node_code=node.node_code,
            target_node_code=nodes[0].node_code,
            edge_type=edge_type,
            rule_ref=rule.reference().to_versioned_reference().model_dump(mode="python"),
        )
        for node in nodes[1:]
    )
    tc = code("TEMPLATE")
    data = json.loads(
        (ROOT / "packages/contracts/examples/v1/GECEGTemplate.valid.json").read_text()
    )
    data.update(
        template_code=tc,
        nodes=[n.model_dump(mode="json") for n in nodes],
        edges=[e.model_dump(mode="json") for e in edges],
        goal_node_code=nodes[0].node_code,
        citations=[r.to_versioned_reference().model_dump(mode="json") for r in claim.source_refs],
        extensions={
            "disease_pack_definition": dict(
                code=definition.pack_code, version=definition.version, checksum=definition.checksum
            )
        },
    )
    data["provenance"].update(
        source_codes=[claim.source_refs[0].code],
        rule_versions=[rule.reference().to_versioned_reference().model_dump(mode="json")],
        knowledge_versions=[claim.reference().to_versioned_reference().model_dump(mode="json")],
    )
    template = bound(
        m.Template,
        code=tc,
        graph_template=GECEGTemplate.model_validate_json(json.dumps(data)),
        goal_node_code=nodes[0].node_code,
        nodes=nodes,
        edges=edges,
        rule_refs=(rule.reference(),),
    )
    node_records = tuple(
        bound(
            m.Node,
            code=n.node_code,
            template_ref=template.reference(),
            concept_ref=concept.reference(),
            definition=n,
        )
        for n in nodes
    )
    for node, edge in zip(node_records[1:], edges, strict=True):
        bound(
            m.Edge,
            code=code("RULE"),
            template_ref=template.reference(),
            source_node_ref=node.reference(),
            target_node_ref=node_records[0].reference(),
            definition=edge,
            rule_ref=rule.reference(),
        )
    requirements = tuple(
        bound(
            m.EvidenceRequirement,
            code=code("RULE"),
            node_ref=node.reference(),
            artifact_type_code="ARTIFACT:QC_REPORT",
            acceptance_rule_ref=rule.reference(),
        )
        for node in node_records[1:]
    )
    requirement = requirements[0]
    for n in node_records:
        nr = review(db, n, p)
        publish(db, store, n, nr, p)
    for item in requirements:
        er = review(db, item, p)
        publish(db, store, item, er, p)
    tr = review(db, template, p)
    publish(db, store, template, tr, p)
    release = bound(
        m.Release,
        code=code("SOURCE"),
        policy_ref=p.reference(),
        **(dict(research_scope=use_scope) if research else {}),
        member_refs=(
            claim.reference(),
            rule.reference(),
            stop.reference(),
            template.reference(),
            *(item.reference() for item in requirements),
        ),
    )
    relr = review(db, release, p)
    publish(db, store, release, relr, p)
    pack = seal(
        ResearchDiseasePack if research else DiseasePack,
        definition=definition,
        status="Published",
        clinical_level="pending_L4" if research else "synthetic_only",
        **(dict(research_scope=use_scope, research_policy_ref=p.reference()) if research else {}),
        knowledge_gaps=(),
        bindings=KnowledgeBindings(
            release_ref=release.reference(),
            claim_refs=(claim.reference(),),
            template_ref=template.reference(),
            rule_refs=(rule.reference(),),
            stop_rule_refs=(stop.reference(),),
            evidence_requirement_refs=tuple(item.reference() for item in requirements),
        ),
    )
    return dict(
        pack=pack,
        dataset=dataset,
        template=template,
        rule=rule,
        claim=claim,
        release=release,
        policy=p,
        review=relr,
        condition=condition,
        requirement=requirement,
        requirements=requirements,
        other=other,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


def anonymous_case(goal_code="GOAL:SYNTHETIC_SCREEN"):
    data = json.loads((ROOT / "packages/contracts/examples/v1/CaseBundle.valid.json").read_text())
    identity = code("CASE")
    data.update(
        case_code=identity,
        encounter_code=code("ENCOUNTER"),
        artifacts=[],
        observations=[],
        report_uri=None,
    )
    data["scope"]["case_code"] = identity
    data["goals"] = [dict(goal_code=goal_code, goal_level="evidence_planning")]
    return CaseBundle.model_validate_json(json.dumps(data))
