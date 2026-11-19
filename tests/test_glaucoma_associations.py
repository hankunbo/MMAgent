"""Synthetic association projection tests: no real sources, patient inputs or GPU."""

import copy
import json
from datetime import UTC, datetime

import pytest

from apps import glaucoma_associations as associations
from apps.evidence_graph_html import validate_snapshot
from apps.glaucoma_association_sources import CITATIONS
from apps.glaucoma_case_completion import apply as complete
from apps.glaucoma_evidence_bindings import dump
from apps.glaucoma_evidence_plan import checksum
from db.knowledge import models as k
from packages.contracts import NodeType
from packages.object_store import checksum as byte_checksum
from tests.test_glaucoma_case_completion import fixture
from tests.test_glaucoma_structure_ui import browser


def source_bundle():
    sources = {}
    for key, (provider, identifier, title) in CITATIONS.items():
        source_ref = k.KnowledgeReference(
            kind=k.KnowledgeKind.SOURCE,
            code="SOURCE:SYNTHETIC:" + key.upper(),
            version="1.0.0",
            checksum="sha256:" + "a" * 64,
        )
        provider_ref = k.KnowledgeReference(
            kind=k.KnowledgeKind.SOURCE_PROVIDER,
            code="SOURCE:PROVIDER:SYNTHETIC",
            version="1.0.0",
            checksum="sha256:" + "b" * 64,
        )
        body = ("Synthetic source fixture " + key).encode()
        digest = byte_checksum(body)
        uri = "object://synthetic/" + key
        version = k.seal(
            k.SourceVersion,
            code=source_ref.code + ":CONTENT",
            version="1.0.0",
            synthetic=True,
            source_ref=source_ref,
            provider_ref=provider_ref,
            content_checksum=digest,
            object_uri=uri,
            media_type="text/plain",
            size_bytes=len(body),
            fetched_at=datetime(2026, 1, 1, tzinfo=UTC),
            allowlist_approved=True,
        )
        document = k.seal(
            k.SourceDocument,
            code=source_ref.code + ":DOCUMENT",
            version="1.0.0",
            synthetic=True,
            source_version_ref=version.reference(),
            object_uri=uri,
            media_type="text/plain",
            size_bytes=len(body),
            content_checksum=digest,
            parser_version="1.0.0",
        )
        chunk = k.seal(
            k.DocumentChunk,
            code=source_ref.code + ":CHUNK",
            version="1.0.0",
            synthetic=True,
            source_version_ref=version.reference(),
            document_ref=document.reference(),
            object_uri=uri,
            media_type="text/plain",
            size_bytes=len(body),
            content_checksum=digest,
            parser_version="1.0.0",
            chunking_policy_version="1.0.0",
            location=k.SourceLocation(section="Synthetic", span_start=0, span_end=len(body)),
        )
        sources[key] = {
            "provider": provider,
            "record_id": identifier,
            "title": title or key,
            "url": "https://example.org/" + key,
            "reference": dump(version.reference()),
            "record": dump(version),
            "representation": "synthetic_fixture",
            "chunks": [
                {"reference": dump(chunk.reference()), "record": dump(chunk), "text": body.decode()}
            ],
        }
    result = {"schema": "association-citation-set/1.0.0", "sources": sources}
    return {**result, "checksum": checksum(result)}


def template_for(value):
    nodes = tuple(
        k.TemplateNode(
            node_code=("GOAL" if n["key"] == "ROOT" else "EVIDENCE") + ":SYNTHETIC:" + n["key"],
            node_type=NodeType.GOAL if n["key"] == "ROOT" else NodeType.EVIDENCE,
            concept_code="CONCEPT:SYNTHETIC:" + n["key"],
            required=n["key"] == "ROOT",
        )
        for n in value["nodes"]
    )
    template = k.seal(
        k.Template,
        code="TEMPLATE:SYNTHETIC:ASSOCIATIONS",
        version="1.0.0",
        synthetic=True,
        goal_node_code=nodes[0].node_code,
        nodes=nodes,
    )
    value["records"] = [{"record": dump(template)}]
    return value


def example(monkeypatch):
    return template_for(complete(fixture(monkeypatch)))


def test_real_new_branches_without_changing_original_criteria(monkeypatch):
    original = example(monkeypatch)
    before = copy.deepcopy(original)
    value = associations.apply(original, source_bundle())
    validate_snapshot(value)
    assert original == before
    assert associations.apply(value, source_bundle()) == value
    validate_snapshot(json.loads(json.dumps(value, sort_keys=True)))
    assert len(value["nodes"]) == 63
    assert value["research_criteria"] == original["research_criteria"]
    assert value["case_completion"] == original["case_completion"]
    assert value["evidence_structure"]["counts"]["main"] == 30
    a = value["source_associations"]
    assert len(a["association_edges"]) == 10 and len(a["organization_edges"]) == 11
    assert all(
        e["causal"] is False and e["patient_support_score"] is None for e in a["association_edges"]
    )
    assert all(
        e["numeric_weight"] is None and e["universal_threshold"] is None
        for e in a["association_edges"]
    )
    nodes = {n["key"]: n for n in value["nodes"]}
    assert nodes["INTEREYE_COMPARISON"]["patient_evidence"]["status"] == "当前输入无法取证"
    assert nodes["VENULAR_BRANCH_ANGLE"]["patient_evidence"]["status"] == "算法待补"
    assert (
        nodes["VASCULAR_SKELETON_DENSITY"]["patient_evidence"]["results"]
        == nodes["DENSITY"]["patient_evidence"]["results"]
    )
    assert (
        nodes["DISC_VESSEL_COURSE"]["patient_evidence"]["visual_observation"][
            "deployed_vlm_executed"
        ]
        is False
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "score",
        "threshold",
        "direction",
        "source",
        "endpoint",
        "missing",
        "node_result",
        "scope",
        "chunk",
    ],
)
def test_rehash_does_not_turn_association_into_fake_fact(monkeypatch, mutation):
    value = associations.apply(example(monkeypatch), source_bundle())
    a = value["source_associations"]
    if mutation == "score":
        a["association_edges"][0]["patient_support_score"] = 0.9
    elif mutation == "threshold":
        a["association_edges"][0]["universal_threshold"] = 6
    elif mutation == "direction":
        a["association_edges"][0]["direction"] = "diagnostic"
    elif mutation == "source":
        a["nodes"][1]["knowledge"] = []
    elif mutation == "endpoint":
        value["relationships"][-1]["child"] = "ROOT"
    elif mutation == "missing":
        value["nodes"].pop()
    elif mutation == "node_result":
        value["nodes"][-1]["patient_evidence"]["summary"] = "visual field normal"
    elif mutation == "scope":
        a["case_code"] = "CASE:ANON_OTHER"
    else:
        a["sources"]["sources"]["caliber"]["chunks"][0]["text"] = "fake"
    a["checksum"] = checksum({k: v for k, v in a.items() if k != "checksum"})
    value["checksum"] = checksum({k: v for k, v in value.items() if k != "checksum"})
    with pytest.raises(ValueError):
        validate_snapshot(value)


def test_html_shows_associations_in_tree_and_links_to_existing_results(monkeypatch):
    value = associations.apply(example(monkeypatch), source_bundle())
    browser(
        value,
        r"""
check(treeKeys().length===30,'new branches absent');
check(treeKeys().includes('VASCULAR_ASSOC'),'vascular root not visible');
check(treeKeys().includes('DISC_VESSEL_COURSE'),'morphology branch absent');
check(text(elements.detail).includes('本次新增的文献关联分支'),'overview absent');
run("show('ARTERIOLAR_CALIBER')");
check(text(elements.detail).includes('1.77'),'source effect not explained');
check(text(elements.detail).includes('未检出显著差异'),'counter evidence absent');
check(text(elements.detail).includes('全局像素宽度不是CRAE'),'measurement mismatch hidden');
run("show('STRUCTURE_FUNCTION_MATCH')");
check(text(elements.detail).includes('没有同眼视野'),'missing input misrepresented');
run('for(const n of D.nodes)show(n.key)');
""",
    )


def test_knowledge_extension_contains_no_patient_measurements(monkeypatch):
    value = associations.apply(example(monkeypatch), source_bundle())
    records = value["source_associations"]["knowledge_records"]
    assert {r["kind"] for r in records} == {"concept", "template", "node", "evidence_requirement"}
    assert value["case_evidence"]["case_code"] not in json.dumps(records)
    assert all(
        r.get("acceptance_rule_ref") is None for r in records if r["kind"] == "evidence_requirement"
    )
    assert len(next(r for r in records if r["kind"] == "template")["nodes"]) == 63
