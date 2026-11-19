"""Source facts and visual hypotheses remain distinct through CLI-owned outputs."""

import copy
import json
from pathlib import Path

import pytest

from packages.llm_provider.transport import Reply
from packages.qualitative_analysis import runner
from packages.qualitative_analysis.comparison import compare, integrate_saved, verified_result
from packages.qualitative_analysis.integration import (
    canonical,
    field_contexts,
    integrate,
    unit_result,
)
from packages.qualitative_analysis.models import VisualResultV2
from packages.qualitative_analysis.storage import read
from tests.qualitative_analysis.test_analyzer import Transport, candidate
from tests.qualitative_analysis.test_analyzer import prepared as existing_prepared


@pytest.fixture
def prepared(tmp_path):
    return existing_prepared.__wrapped__(tmp_path)


CONFIG = Path("configs/research/qualitative_analysis_v2.json").absolute()


def v2():
    v = json.loads(candidate())
    v.update(
        schema_version="qualitative-visual-result/1.1.0",
        anatomy=[
            dict(
                name="视盘",
                identification="candidate",
                view_id="V1",
                location="中心",
                evidence="圆盘形边界与血管汇聚",
                limitation="边界局部模糊",
            )
        ],
        visual_laterality=dict(
            status="not_visually_confirmable",
            value="UNKNOWN",
            evidence="没有图中标记",
            view_id="V1",
        ),
        uncertainties=[dict(field="laterality", reason="图中无标记", view_id="V1")],
    )
    v["annotations"][0].update(attribute="modality", value="彩色眼底图像")
    return v


def new_plan(prepared, tmp_path):
    packet, _ = prepared
    p = read(packet / "case-input.json")
    out = tmp_path / "new"
    runner.plan(
        packet,
        out,
        CONFIG,
        dict(
            mode="archive_description",
            component_ids=[c["id"] for c in p["components"]],
            background_ids=[],
        ),
    )
    return out


def test_v2_real_packet_request_response_export_resume(prepared, tmp_path):
    out = new_plan(prepared, tmp_path)
    t = Transport(Reply(content=json.dumps(v2()).encode(), api_requests=0, response_confirmed=True))
    saved = runner.run(out, live=True, transport=t)
    assert saved["all_planned_jobs_completed"]
    body = t.calls[0]["messages"][1]["content"][0]["text"]
    assert "laterality_basis" in body and "qualitative-visual-result/1.1.0" in body
    assert "SECRET_DIAGNOSIS" not in body
    value = verified_result(saved["json"])
    assert value["schema_version"] == "qualitative-case-description/1.1.0"
    assert value["components"][0]["integrated"]["anatomy"][0]["name"] == "视盘"
    assert "患者资料汇总" in Path(saved["html"]).read_text()
    assert value["disease_score"] is None
    runner.run(out, live=True, transport=t)
    assert len(t.calls) == 2
    runner.verify(out)


def test_known_eye_unknown_vision_and_synonyms(prepared):
    _, out = prepared
    saved = runner.run(out, live=True, transport=Transport())
    value = read(saved["json"])
    row = value["components"][0]
    row["source_declared"]["laterality"] = "RIGHT"
    row["source_declared"]["modality"] = "color_fundus"
    row["visual_laterality"] = v2()["visual_laterality"]
    row["uncertainties"] = v2()["uncertainties"]
    row["model_annotations"] = v2()["annotations"]
    before = copy.deepcopy(row["source_declared"])
    integrate(value)
    assert row["integrated"]["laterality"]["value"] == "RIGHT"
    assert (
        row["integrated"]["uncertainties"][0]["scope"] == "visual_confirmation_only_source_retained"
    )
    assert row["integrated"]["terminology_comparisons"][0]["status"] == "equivalent"
    assert not row["conflicts"] and row["source_declared"] == before
    assert row["integrated"]["anatomy"] == []  # No code-imagined anatomy.
    row["visual_laterality"] = dict(
        status="marker_readable", value="LEFT", evidence="OS", view_id="V1"
    )
    integrate(value)
    assert row["conflicts"] and row["integrated"]["laterality"]["value"] == "RIGHT"


@pytest.mark.parametrize(
    "field,value,expected",
    [
        ("laterality", "OD", "RIGHT"),
        ("laterality", "左眼", "LEFT"),
        ("modality", "彩色眼底图像", "color_fundus"),
        ("unit", "μm", "µm"),
        ("modality", "看起来像某种眼底图像", None),
    ],
)
def test_exact_terms(field, value, expected):
    assert canonical(field, value) == expected


def test_units_no_numeric_guess_and_original_owner_evidence():
    from packages.dataset_adapters.multimodal import grape_semantics as g

    source = dict(
        kind="scalar",
        concept="iop",
        unit=None,
        value=14.7,
        provenance_ref="R2",
        semantic_limits=list(g.fields("iop")["semantic_limits"]),
    )
    row = dict(component_id="C2", component_checksum="binding", source_declared=source)
    ctx = field_contexts([row])["C2"]
    assert ctx["status"] == "verified_existing_field_context"
    result = unit_result(source, {}, ctx)
    assert result["status"] == "documented_method_inference" and result["proposed_unit"] == "mmHg"
    assert source["unit"] is None and source["value"] == 14.7
    source["semantic_limits"] = []
    assert field_contexts([row]) == {}
    assert unit_result(source, {}, {})["status"] == "unknown"
    assert (
        unit_result(source, {"unit_state": "source_declared_code_mapping_not_physical_unit"}, {})[
            "status"
        ]
        == "categorical_code_not_physical_unit"
    )
    source["kind"] = "image"
    assert unit_result(source, {}, {})["physical_scale"] == "not_bound"


def test_source_change_no_unit_grant(monkeypatch):
    from packages.dataset_adapters.multimodal import grape_semantics as g

    source = dict(
        kind="scalar",
        concept="iop",
        unit=None,
        value=14.7,
        provenance_ref="R2",
        semantic_limits=list(g.fields("iop")["semantic_limits"]),
    )

    def fail(*args, **kw):
        raise ValueError("changed_source")

    monkeypatch.setattr(g, "method_context", fail)
    ctx = field_contexts(
        [dict(component_id="C2", component_checksum="binding", source_declared=source)]
    )["C2"]
    assert ctx["status"] == "source_context_unavailable"
    assert unit_result(source, {}, ctx)["status"] == "unknown"


def test_saved_integration_no_request_no_old_write(prepared, tmp_path):
    _, out = prepared
    t = Transport()
    saved = runner.run(out, live=True, transport=t)
    original = {str(p): p.read_bytes() for p in out.rglob("*") if p.is_file()}
    result = integrate_saved(saved["json"], tmp_path / "derived")
    new = verified_result(result["json"])
    assert new["derivation"]["new_api_requests"] == 0
    assert all(Path(p).read_bytes() == b for p, b in original.items())
    report = compare(saved["json"], result["json"], tmp_path / "comparison")
    assert report["summary"]["new_visual_component_count"] == 0
    assert report["summary"]["source_facts_preserved"]
    assert len(t.calls) == 2


def test_v2_view_and_status_binding(prepared, tmp_path):
    out = new_plan(prepared, tmp_path)
    bad = v2()
    bad["anatomy"][0]["view_id"] = "V999"
    saved = runner.run(
        out,
        live=True,
        transport=Transport(Reply(content=json.dumps(bad).encode(), response_confirmed=True)),
    )
    assert set(saved["jobs"].values()) == {"contract_rejected"}
    bad = v2()
    bad["visual_laterality"]["value"] = "RIGHT"
    with pytest.raises(ValueError):
        VisualResultV2.model_validate(bad)


def test_patient_visit_and_missing_preserved(prepared):
    _, out = prepared
    saved = runner.run(out, live=True, transport=Transport())
    value = read(saved["json"])
    before = copy.deepcopy(value["relations"])
    integrate(value)
    assert value["patient_summary"]["eye_visit_groups"] == before["eye_visit_groups"]
    assert (
        value["patient_summary"]["identical_payload_groups"] == before["identical_payload_groups"]
    )


def test_confirmed_fallback_keeps_unknown_and_old_contract(prepared, tmp_path):
    from packages.qualitative_analysis.comparison import combine_confirmed

    _, old_out = prepared
    previous = runner.run(old_out, live=True, transport=Transport())
    new_out = new_plan(prepared, tmp_path)
    transport = Transport(Reply(error="timeout", api_requests=1, response_confirmed=None))
    current = runner.run(new_out, live=True, transport=transport)
    combined = combine_confirmed(current["json"], previous["json"], tmp_path / "combined")
    value = verified_result(combined["json"])
    assert not value["coverage"]["all_planned_jobs_completed"]
    assert value["execution"]["unknown_request_count"] == 2
    assert len(combined["historical_fallback_components"]) == 2
    assert all(
        not r["execution"]["historical_fallback"]["fulfills_new_visual_contract"]
        for r in value["components"]
    )
    assert all(not r["integrated"]["anatomy"] for r in value["components"])
    assert "复用旧版视觉记录（新版未完成）" in Path(combined["html"]).read_text()
    runner.run(new_out, live=True, transport=transport)
    assert len(transport.calls) == 2
