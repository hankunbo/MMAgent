"""05C offline report arithmetic; query semantics remain in GroundingQueryService."""

from pathlib import Path

from packages.disease_packs import DiseasePack
from packages.grounding import digest

# Display classification is versioned and independent of detailed causal attribution.
CLASSIFICATION = {
    "version": "05c-coverage/1.0.0",
    "priority": [
        "type_feasible",
        "unsafe_to_automate",
        "missing_data",
        "missing_tool",
        "expert_review",
    ],
    "type_feasible": "returned candidate; no execution or medical evidence claim",
    "unsafe_to_automate": "safety_restriction finding, including unknown required conditions",
    "missing_data": "patient_data_missing finding",
    "missing_tool": "resolved capability_unregistered or implementation_missing finding only",
    "expert_review": "recommendation, unresolved/mapping/search gap or existing product",
    "existing_artifact": "separate existing_product flag; does not count as a tool candidate",
    "counting": "unique exact template + leaf requirement; scenarios excluded",
}
CLASSIFICATION["checksum"] = digest(CLASSIFICATION)


def classify(result):
    kinds = {f.category for f in result.gaps}
    categories = set()
    if result.candidates:
        categories.add("type_feasible")
    if "safety_restriction" in kinds:
        categories.add("unsafe_to_automate")
    if "patient_data_missing" in kinds:
        categories.add("missing_data")
    if kinds & {"capability_unregistered", "implementation_missing"}:
        categories.add("missing_tool")
    if (
        result.expert_review_recommended
        or result.existing_artifacts
        or kinds & {"mapping_missing", "unresolved_producer", "search_incomplete"}
    ):
        categories.add("expert_review")
    resolutions = {d.resolution for d in result.product_demands}
    if "search_incomplete" in kinds:
        resolutions.add("search_incomplete")
    return dict(
        categories=[c for c in CLASSIFICATION["priority"] if c in categories],
        primary=next((c for c in CLASSIFICATION["priority"] if c in categories), None),
        attributions=sorted(resolutions),
        reasons=sorted({f.reason for f in result.gaps}),
        existing_product=bool(result.existing_artifacts),
        candidate_count=len(result.candidates),
        status=result.status,
        result_checksum=result.checksum,
    )


def ratio(numerator, denominator):
    return dict(
        numerator=numerator,
        denominator=denominator,
        value=numerator / denominator if denominator else None,
        status="applicable" if denominator else "not_applicable",
    )


def summarize(rows, template_count):
    identities = [(r["template_ref"]["checksum"], r["requirement_ref"]["checksum"]) for r in rows]
    if len(set(identities)) != len(rows):
        raise ValueError("duplicate leaf requirement would inflate coverage")
    classified = sum(bool(r["classification"]["categories"]) for r in rows)
    feasible = sum(r["classification"]["candidate_count"] > 0 for r in rows)
    return dict(
        template_count=template_count,
        leaf_requirement_count=len(rows),
        classified=ratio(classified, len(rows)),
        type_feasible=ratio(feasible, len(rows)),
        uncovered=[r["requirement_ref"] for r in rows if not r["classification"]["categories"]],
    )


def real_draft_inventory(root):
    rows = []
    for path in sorted((Path(root) / "configs/disease_packs").glob("*.json")):
        pack = DiseasePack.model_validate_json(path.read_bytes())
        assert not pack.definition.synthetic and pack.status == "Draft" and pack.bindings is None
        rows.append(
            dict(
                path=str(path.relative_to(root)),
                pack_ref=pack.reference().model_dump(mode="json"),
                definition_checksum=pack.definition.checksum,
                status=pack.status,
                clinical_level=pack.clinical_level,
                knowledge_bindings=None,
                slots=[s.model_dump(mode="json") for s in pack.definition.evidence_slots],
                knowledge_gaps=pack.knowledge_gaps,
            )
        )
    return dict(
        packs=rows,
        pack_count=len(rows),
        prepared_slot_count=sum(len(r["slots"]) for r in rows),
        published_template_count=0,
        published_leaf_requirement_count=0,
        classified=ratio(0, 0),
        type_feasible=ratio(0, 0),
        reason="Draft declarations have no Published Knowledge bindings; not queried or promoted",
    )
