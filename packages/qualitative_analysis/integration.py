"""Versioned source/observation integration. Never invent anatomy, units or diagnoses."""

from collections import Counter
from pathlib import Path

from packages.dataset_adapters.multimodal import grape_semantics as grape
from packages.object_store import checksum

ROOT = Path(__file__).resolve().parents[2]
VERSION = "qualitative-integration/1.0.0"
# Exact synonyms only. Unknown free text stays unresolved, not a medical conflict.
TERMS = {
    "modality": {
        "color_fundus": (
            "color_fundus",
            "彩色眼底图像",
            "彩色眼底照片",
            "彩色眼底摄影",
            "color fundus photograph",
            "color fundus image",
        ),
        "slo": ("slo", "扫描激光眼底图像", "scanning laser ophthalmoscopy"),
        "oct": ("oct", "光学相干断层扫描"),
    },
    "laterality": {
        "RIGHT": ("right", "od", "右眼"),
        "LEFT": ("left", "os", "左眼"),
        "UNKNOWN": ("unknown", "未知", "眼别未知"),
    },
    "unit": {
        "µm": ("um", "μm", "µm", "微米"),
        "year": ("year", "years", "年"),
        "mmHg": ("mmhg", "毫米汞柱"),
        "dB": ("db", "分贝"),
    },
}
NAMES = {
    "fundus": "彩色眼底图像",
    "color_fundus": "彩色眼底图像",
    "iop": "眼压",
    "cct": "中央角膜厚度",
    "age": "年龄",
    "vf": "视野光敏感度向量",
    "rnfl_mean": "RNFL平均厚度",
    "rnfl_superior": "RNFL上方厚度",
    "rnfl_nasal": "RNFL鼻侧厚度",
    "rnfl_inferior": "RNFL下方厚度",
    "rnfl_temporal": "RNFL颞侧厚度",
}
EYES = {"RIGHT": "右眼", "LEFT": "左眼", "UNKNOWN": "来源眼别未知"}


def canonical(field, value):
    if not isinstance(value, str):
        return None
    for code, aliases in TERMS.get(field, {}).items():
        if value.strip().casefold() in {a.casefold() for a in (*aliases, code)}:
            return code
    return None


def field_contexts(components):
    """Use original release semantic binding, never a dataset-name routing rule.

    Existing owner validates exact local source and excerpt hashes. Only verified
    field-method bridges are snapshotted; future export reads the frozen context.
    """
    contexts = {}
    for row in components:
        source = row["source_declared"]
        limits = source.get("semantic_limits", [])
        required = {
            "semantic_profile:" + grape.PROFILE,
            "source_document:" + grape.PAPER,
            "source_document_checksum:" + grape.SOURCE_CHECKSUM,
        }
        if not required <= set(limits):
            continue
        concept = source["concept"]
        if concept not in {"iop", "cct", "age", "vf"} and not concept.startswith("rnfl_"):
            continue
        context = dict(
            component_checksum=row["component_checksum"], provenance_ref=source["provenance_ref"]
        )
        try:
            raw = (ROOT / "docs/sources/grape-author-2023-PMC10404253.xml").read_bytes()
            if checksum(raw) != grape.SOURCE_CHECKSUM:
                raise ValueError("source_binding")
            context.update(
                field_meaning=grape.context(concept),
                source_checksum=grape.SOURCE_CHECKSUM,
                source_url=grape.PAPER,
                status="verified_existing_field_context",
            )
            if concept.startswith("rnfl_") or concept in {"iop", "cct"}:
                # Original owner paths are repository-relative, as are all CLI commands.
                context["method"] = grape.method_context(
                    concept,
                    profile=grape.RNFL_UNIT_PROFILE
                    if concept.startswith("rnfl_")
                    else grape.TONOPACHY_UNIT_PROFILE,
                )
        except (ValueError, OSError) as error:
            context = dict(
                component_checksum=row["component_checksum"],
                status="source_context_unavailable",
                error_type=type(error).__name__,
                unit_not_granted=True,
            )
        contexts[row["component_id"]] = context
    return contexts


def unit_result(source, semantics, context):
    result = dict(
        source_unit=source.get("unit"),
        source_value=source.get("value"),
        normalized_unit=semantics.get("normalized_unit"),
        numeric_conversion="none",
    )
    if semantics.get("unit_state") == "source_declared_code_mapping_not_physical_unit":
        result.update(status="categorical_code_not_physical_unit")
    elif source.get("unit"):
        result.update(status="source_declared", basis=source.get("unit_basis"))
    elif source["kind"] in {"image", "image_series", "volume", "map"}:
        result.update(
            status="intensity_unit_unspecified",
            physical_scale="not_bound",
            note="图像本身不要求长度单位；像素强度单位及物理比例尺须分别由来源提供。",
        )
    else:
        bridge = context.get("method", {}).get("unit_bridge")
        if bridge:
            result.update(
                status="documented_method_inference",
                proposed_unit=canonical("unit", bridge["proposed_unit"]) or bridge["proposed_unit"],
                basis=bridge,
                source_unit_confirmed=False,
            )
        else:
            result.update(status="unknown", reason="no_bound_field_unit_evidence")
    return result


def integrate(result):
    """Deterministic integration over saved facts and annotations only."""
    common = {}
    contexts = result.get("field_contexts", {})
    for row in result["components"]:
        source = row["source_declared"]
        context = contexts.get(row["component_id"], {})
        if context and context["component_checksum"] != row["component_checksum"]:
            raise ValueError("qualitative_field_context_binding")
        eye = source.get("laterality", "UNKNOWN")
        visual = row.get(
            "visual_laterality", dict(status="not_structured_in_original_response", value="UNKNOWN")
        )
        conflict = (
            visual["status"] == "marker_readable" and eye != "UNKNOWN" and visual["value"] != eye
        )
        laterality = dict(
            source_value=eye,
            source_ref=source.get("provenance_ref"),
            visual=visual,
            value=eye,
            basis="source_declared" if eye != "UNKNOWN" else "unknown",
            status="conflict_requires_resolution"
            if conflict
            else "source_retained"
            if eye != "UNKNOWN"
            else "unknown",
        )
        # A readable image marker is still a visual candidate, not a replacement source fact.
        if eye == "UNKNOWN" and visual["status"] == "marker_readable":
            laterality.update(status="visual_candidate_only", candidate_value=visual["value"])
        comparisons = []
        for a in row.get("model_annotations", []):
            field = a.get("attribute")
            if field not in TERMS:
                continue
            raw_source, raw_candidate = source.get(field), a["value"]
            left, right = canonical(field, raw_source), canonical(field, raw_candidate)
            status = (
                "equivalent"
                if raw_source == raw_candidate or (left is not None and left == right)
                else "unresolved_text"
            )
            if field == "laterality" and right == "UNKNOWN" and eye != "UNKNOWN":
                status = "visual_not_confirmed_source_retained"
            elif left is not None and right is not None and left != right:
                status = "different_canonical_candidates_not_automatic_fact_override"
            comparisons.append(
                dict(
                    field=field,
                    source_value=raw_source,
                    candidate_value=raw_candidate,
                    source_canonical=left,
                    candidate_canonical=right,
                    status=status,
                    view_id=a["view_id"],
                )
            )
        row["integrated"] = dict(
            version=VERSION,
            terminology_version="qualitative-terms/1.0.0",
            laterality=laterality,
            modality=dict(
                source_value=source["modality"],
                canonical=canonical("modality", source["modality"]),
                basis="source_declared",
            ),
            terminology_comparisons=comparisons,
            units=unit_result(source, row["semantics"], context),
            field_context=context,
            anatomy=row.get("anatomy", []),
            anatomy_state="new_visual_annotation"
            if "anatomy" in row
            else "original_descriptions_only_no_code_inferred_anatomy",
            uncertainties=[
                dict(
                    u,
                    scope="visual_confirmation_only_source_retained"
                    if u["field"] == "laterality" and eye != "UNKNOWN"
                    else "supplied_views",
                )
                for u in row.get("uncertainties", [])
            ],
            original_unconfirmed_scope="raw_model_statements_not_patient_fact_overrides",
        )
        row["conflicts"] = (
            [dict(field="laterality", source_value=eye, visual_marker=visual)] if conflict else []
        )
        # Scope-aware repeated exact text is indexed once in the human report.
        refs = []
        for a in row.get("model_annotations", []):
            text = a["limitation"]
            entry = common.setdefault(
                text, dict(id="L" + str(len(common) + 1), text=text, component_ids=[])
            )
            if row["component_id"] not in entry["component_ids"]:
                entry["component_ids"].append(row["component_id"])
            refs.append(entry["id"])
        row["integrated"]["limitation_refs"] = sorted(set(refs))
    result["limitation_registry"] = list(common.values())
    result["patient_summary"] = dict(
        purpose="data_inventory_not_diagnosis_or_temporal_task_selection",
        component_count=len(result["components"]),
        background_count=len(result["background"]),
        component_kinds=dict(Counter(r["source_declared"]["kind"] for r in result["components"])),
        source_laterality_counts=dict(
            Counter(r["source_declared"]["laterality"] for r in result["components"])
        ),
        unit_states=dict(Counter(r["integrated"]["units"]["status"] for r in result["components"])),
        unavailable_components=[
            r["component_id"]
            for r in result["components"]
            if r["source_declared"]["status"] in {"missing", "withheld"}
        ],
        visual_completed_components=[
            r["component_id"]
            for r in result["components"]
            if r["execution"]["visual"] == "completed"
        ],
        eye_visit_groups=result["relations"]["eye_visit_groups"],
        identical_payload_groups=result["relations"]["identical_payload_groups"],
        background=[
            dict(
                id=b["source_declared"]["id"],
                concept=b["source_declared"]["concept"],
                value=b["normalized_value"],
                status=b["source_declared"]["status"],
                laterality=b["source_declared"]["laterality"],
                visit=b["source_declared"]["visit"],
                meaning_status=b["semantic_state"],
                source_ref=b["source_declared"]["provenance_ref"],
            )
            for b in result["background"]
        ],
        shared_limits=[
            "仅解释所提供的图像视图，不代表完整体积或临床诊断。",
            "来源眼别与图像是否能独立辨眼分开记录。",
            "同编号访视不保证同时间；同内容不证明同一次采集。",
            "设备方法支持的单位推断不改写原值或升级为源表单位确认。",
        ],
    )
    result["schema_version"] = "qualitative-case-description/1.1.0"
    return result
