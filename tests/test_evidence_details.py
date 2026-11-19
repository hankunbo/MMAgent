import copy
import json

import pytest

from apps.evidence_graph_details import attach
from apps.evidence_graph_html import validate_snapshot
from apps.glaucoma_evidence_plan import checksum
from apps.glaucoma_tool_evidence import upgrade
from packages.evidence_details.models import FACETS, Assessment, Ledger, digest
from packages.evidence_details.projection import build, validate
from tests.test_glaucoma_structure_ui import browser
from tests.test_glaucoma_tool_evidence import fixture


def profiles():
    return {
        "schema": "evidence-source-tool-profiles/1.0.0",
        "assembly": {
            "builder": "judgment-binding/1.0.0",
            "actor": "synthetic_test_fixture",
            "recorded_at": "2026-09-19T00:00:00Z",
            "medical_review": False,
        },
        "tools": [],
        "evaluations": [],
        "sources": [],
    }


def test_knowledge_provider_lookup_parses_strict_json_refs_and_deduplicates(monkeypatch):
    from datetime import UTC, datetime
    from types import SimpleNamespace

    from apps.evidence_detail_profiles import knowledge_sources
    from db.knowledge.models import KnowledgeKind, KnowledgeReference

    source = KnowledgeReference(
        kind=KnowledgeKind.SOURCE_VERSION,
        code="SOURCE:DETAIL_TEST",
        version="1.0.0",
        checksum="sha256:" + "a" * 64,
    )
    provider = KnowledgeReference(
        kind=KnowledgeKind.SOURCE_PROVIDER,
        code="SOURCE:PROVIDER:TEST",
        version="1.0.0",
        checksum="sha256:" + "b" * 64,
    )
    raw = source.model_dump(mode="json")
    record = SimpleNamespace(
        provider_ref=provider,
        source_ref=source,
        content_checksum=source.checksum,
        object_uri="object://fixture",
        media_type="text/html",
        fetched_at=datetime(2026, 9, 19, tzinfo=UTC),
        size_bytes=100,
        discovery_refs=[],
    )
    calls = []

    class Reader:
        def get(self, ref):
            assert isinstance(ref.kind, KnowledgeKind)
            calls.append(ref)
            return SimpleNamespace(
                record=record, status=SimpleNamespace(value="Draft"), withdrawn=False
            )

    monkeypatch.setattr("apps.knowledge_source_tools.repository", lambda role: Reader())
    result = knowledge_sources({"a": raw, "b": [raw]})
    assert len(result) == 1 and calls == [source, provider]
    assert result[0]["provider"]["code"] == "SOURCE:PROVIDER:TEST"
    assert result[0]["current_read_is_publication_authorization"] is False


def example(monkeypatch):
    _, raw = fixture(monkeypatch)
    base = upgrade(raw)
    p = profiles()
    return base, p, build(base, p)


def test_all_nodes_judgments_and_dependencies_are_bound(monkeypatch):
    base, p, ledger = example(monkeypatch)
    validate(base, p, ledger)
    assert len(ledger["node_index"]) == len(base["nodes"]) == 63
    assert len(ledger["assessments"]) == 106
    assert all(set(a["facets"]) == set(FACETS) for a in ledger["assessments"])
    assert all(a["confidence_score"] is None for a in ledger["assessments"])
    a = next(a for a in ledger["assessments"] if a["id"] == "criterion:VCDR_GE_07")
    assert a["comparison"]["threshold"] == 0.7
    for node in ["ROOT", "STRUCTURE", "CUPDISC", "CUP_RIM_MEASURE"]:
        assert ledger["node_index"][node].count(a["id"]) == 1
    assert ledger["coverage"]["clinical_evidence_coverage"] is None
    assert not base.get("judgment_details")
    current = attach(base, p, ledger)
    validate_snapshot(current)
    for key in ("research_criteria", "branch_geometry", "source_associations", "case_completion"):
        assert current[key] == base[key]
    assert build(base, p) == json.loads(json.dumps(ledger))


@pytest.mark.parametrize(
    "change",
    [
        "scope",
        "unit",
        "threshold",
        "score",
        "definition",
        "duplicate",
        "missing_reason",
        "profile",
        "pointer",
        "role",
    ],
)
def test_rehash_cannot_forge_details(monkeypatch, change):
    base, p, ledger = example(monkeypatch)
    value = copy.deepcopy(ledger)
    a = value["assessments"][0]
    if change == "scope":
        value["case_code"] = "CASE:ANON_OTHER"
    elif change == "unit":
        a["comparison"]["unit"] = "mm"
    elif change == "threshold":
        a["comparison"]["threshold"] = 0.2
    elif change == "score":
        a["confidence_score"] = 0.9
    elif change == "definition":
        a["facets"]["reference"]["explanation"] = "caller medical formula"
    elif change == "duplicate":
        value["node_index"][a["node"]].append(a["id"])
    elif change == "missing_reason":
        a["facets"]["uncertainty"]["missing"] = []
    elif change == "profile":
        p["assembly"]["actor"] = "forged_review"
    elif change == "pointer":
        a["facets"]["target"]["facts"][0]["pointer"] = "/case_evidence/input"
    else:
        a["comparison"]["medical_effect"] = "SUPPORTS"
    value["checksum"] = digest({k: v for k, v in value.items() if k != "checksum"})
    with pytest.raises(ValueError):
        validate(base, p, value)


def test_constructed_model_requires_revalidation_and_no_nonfinite(monkeypatch):
    base, p, ledger = example(monkeypatch)
    a = Assessment.model_validate(ledger["assessments"][0])
    mutated = a.model_copy(update={"confidence_score": 0.8})
    with pytest.raises(ValueError):
        Assessment.model_validate(mutated.model_dump(warnings=False))
    ledger["assessments"][0]["comparison"]["observed_value"] = float("nan")
    with pytest.raises(ValueError):
        Ledger.model_validate(ledger)


def test_missing_inputs_visual_failure_and_correlated_masks_remain_distinct(monkeypatch):
    base, p, ledger = example(monkeypatch)
    visual = next(a for a in ledger["assessments"] if a["id"] == "visual:fovea")
    assert visual["comparison"]["finding"] == "not_localizable"
    assert "HISTORICAL_OBSERVER_CONFIGURATION_INCOMPLETE" in visual["gap_codes"]
    assert "未执行部署VLM" in visual["facets"]["execution"]["explanation"]
    branch = next(a for a in ledger["assessments"] if a["id"] == "association:VENULAR_BRANCH_ANGLE")
    assert branch["comparison"]["threshold"] is None
    assert "NO_GENERAL_MEDICAL_CUTOFF" in branch["gap_codes"]
    # Existing formal results, optional failures and scale checks are not overwritten.
    assert attach(base, p, ledger)["branch_geometry"] == base["branch_geometry"]
    assert len({a["same_image_group"] for a in ledger["assessments"]}) == 1
    assert len({tuple(a["artifact_groups"]) for a in ledger["assessments"]}) > 1


def test_ui_keeps_six_sections_for_every_node(monkeypatch):
    base, p, ledger = example(monkeypatch)
    view = attach(base, p, ledger)
    browser(
        view,
        r"""
for(const key of run('D.nodes.map(n=>n.key)')){
 run('show('+JSON.stringify(key)+')');
 const sections=all(elements.detail).filter(e=>e.tag==='h3'&&/^\d ·/.test(e.textContent));
 check(sections.length===6,'six sections');
 check(text(elements.detail).includes('判读链与后续评分依据'),'trace missing');
 check(text(elements.detail).includes('输入匹配'),'input details missing');
}
""",
    )
    # A rehashed outer export still cannot override the controlled inner projection.
    view["judgment_details"]["ledger"]["assessments"][0]["comparison"]["threshold"] = 0.1
    view["checksum"] = checksum({k: v for k, v in view.items() if k != "checksum"})
    with pytest.raises(ValueError):
        validate_snapshot(view)
