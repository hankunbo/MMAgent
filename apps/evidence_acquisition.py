"""Read-only first-case acquisition preparation. Never starts a run or loads a key."""

import argparse
import json
from collections import Counter
from html import escape
from pathlib import Path

from apps.vectorengine_round_materials import CATALOG, load_catalog
from apps.vectorengine_single_case import write_once
from packages.evidence_acquisition.materials import observation_alias_candidate, prepare_visual_task
from packages.evidence_acquisition.models import Requirement, SearchProgress, SearchSuggestion
from packages.evidence_acquisition.planning import plan_acquisition, search_plan
from packages.evidence_details.acceptance_models import DomainState
from packages.evidence_details.review_tasks import project_review_tasks
from packages.grounding.models import digest, validate

STATE = Path(
    ".local/vectorengine-batch-integrated-r01-case-runtime-r01/prepared-01/domain-state.json"
)
REPORT = Path("docs/exports/青光眼证据图_真实LLM完整评阅版_20261001.json")

# Explicit first-case modality manifest. These declarations do not apply to other cases.
EXTERNAL = {
    "association:INTEREYE_COMPARISON": ("fellow_eye", "paired_laterality"),
    "association:STRUCTURE_FUNCTION_MATCH": ("visual_field", "field_reliability"),
    "criterion:FAMILY": ("family_history",),
    "criterion:FIELD": ("visual_field",),
    "criterion:IOP": ("measured_iop",),
    "criterion:SAME_EYE_CHANGE": ("same_eye_followup",),
    "descriptive:ANGLE": ("gonioscopy",),
    "descriptive:CORNEA": ("central_corneal_thickness",),
    "descriptive:DISC_SIZE_CONTEXT": ("pixel_to_physical_scale",),
    "descriptive:FIELD_RELIABILITY": ("field_reliability",),
    "descriptive:OCT_REFERENCE": ("oct_thickness_and_reference",),
}
VISUAL = {
    "association:DISC_VESSEL_COURSE",
    "association:PERIPAPILLARY_APPEARANCE",
    "criterion:DISC_HEMORRHAGE",
    "criterion:RIM_NOTCH",
    "criterion:RNFL_DEFECT",
    "descriptive:FOVEA",
    "descriptive:IMAGE_QUALITY",
}
METHOD = {
    "association:ARTERIOLAR_CALIBER",
    "association:AV_TORTUOSITY",
    "association:VASCULAR_FRACTAL",
    "association:VASCULAR_SKELETON_DENSITY",
    "association:VENULAR_BRANCH_ANGLE",
}
TOPICS = {
    "criterion:VCDR_GE_07": ("cup_disc_ratio", "disc_size", "slo"),
    "criterion:RIM_RADIAL_LT_01": ("rim", "disc_size", "slo"),
    "criterion:RIM_NOTCH": ("rim", "slo"),
    "criterion:RNFL_DEFECT": ("rnfl", "slo"),
    "criterion:AGE": ("age",),
    "association:ARTERIOLAR_CALIBER": ("vessels", "slo"),
    "association:AV_TORTUOSITY": ("tortuosity", "slo"),
    "association:VASCULAR_FRACTAL": ("fractal", "slo"),
    "association:VASCULAR_SKELETON_DENSITY": ("vessels", "slo"),
    "association:VENULAR_BRANCH_ANGLE": ("vessels", "slo"),
}
PRIORITY = {
    "criterion:VCDR_GE_07": 100,
    "criterion:RIM_RADIAL_LT_01": 95,
    "criterion:RIM_NOTCH": 90,
}
LABELS = {
    "disease_relation": "疾病关联",
    "measurement_dependency": "测量依赖",
    "quality": "图像质量",
    "context": "背景",
    "obtained": "已取得",
    "missing": "缺失",
    "ungradable": "无法评估",
    "unknown": "未取得有效观察",
    "installed": "已绑定",
    "not_independent_disease_rule": "辅助用途，无需独立疾病阈值",
    "ready_for_review": "可进入判读",
    "needs_acquisition": "需要补取证或接入",
    "current_input_unavailable": "当前输入无法取证",
    "auxiliary_observed": "已有辅助结果",
    "quality_limited": "质量受限",
    "bind_existing_reference": "核对并绑定已知来源",
    "search_knowledge": "有界补检索",
    "read_method_provenance": "补测量方法记录",
    "reuse_observation": "核对已有观察",
    "targeted_vision": "针对本项观察图像",
    "review": "按新材料判读",
    "request_tool_approval": "提交新增工具需求",
}


def load_case(state_path=STATE, report_path=REPORT):
    """Historical exports are preparation input, never current DB authorization."""
    initial = validate(DomainState, json.loads(Path(state_path).read_text()))
    report = json.loads(Path(report_path).read_text())
    old = initial.material.scope
    current = report["material"]["scope"]
    if any(
        current[k] != getattr(old, k)
        for k in ("case_code", "run_id", "input_checksum", "visible_checksum", "policy_checksum")
    ):
        raise ValueError("acquisition_export_case_binding")
    state = validate(DomainState, initial.model_copy(update={"material": report["material"]}))
    if project_review_tasks(state).model_dump(mode="json") != report["records"]["review_tasks"]:
        raise ValueError("acquisition_export_tasks_changed")
    return state


def requirements_for(state, catalog):
    tasks = project_review_tasks(state).tasks
    if {t.assessment_id for t in tasks} != set(catalog["tasks"]):
        raise ValueError("acquisition_first_case_catalog_mismatch")
    requirements = []
    for task in tasks:
        identifier = task.assessment_id
        definition = catalog["tasks"][identifier]
        purpose = (
            "disease_relation"
            if identifier.startswith(("criterion:", "association:"))
            else "measurement_dependency"
        )
        if identifier in {"visual:quality", "descriptive:IMAGE_QUALITY"}:
            purpose = "quality"
        if identifier in {"descriptive:DEMOGRAPHICS", "descriptive:CUP_AREA_CONTEXT"}:
            purpose = "context"
        acquisition = (
            "image" if identifier.startswith("visual:") or identifier in VISUAL else "existing"
        )
        unavailable = EXTERNAL.get(identifier, ())
        requirements.append(
            Requirement(
                assessment_id=identifier,
                purpose=purpose,
                acquisition="external_exam" if unavailable else acquisition,
                required_inputs=unavailable
                or (("slo_image",) if acquisition == "image" else ("registered_measurement",)),
                unavailable_inputs=unavailable,
                missing_source_ids=tuple(definition["missing_exact_source_ids"]),
                public_topics=TOPICS.get(identifier, ()),
                method_details_required=identifier in METHOD,
                priority=PRIORITY.get(identifier, 40),
            )
        )
    return tuple(requirements)


def audit(state, catalog):
    plan = plan_acquisition(state, requirements_for(state, catalog))
    source_gaps = {}
    for card in plan.cards:
        if "interpretation_reference" in card.blockers:
            for source in card.missing_source_ids:
                source_gaps.setdefault(source, []).append(card.assessment_id)
    locators = {}
    for task in catalog["tasks"].values():
        for source in (task.get("definition") or {}).get("sources", []):
            if source["id"] in source_gaps:
                entry = dict(
                    source_id=source["id"],
                    title=source["title"],
                    url=source["url"],
                    affected_tasks=source_gaps[source["id"]],
                    current_archive_verified=False,
                    interpretation_installed=False,
                    next_step="check_existing_knowledge_archive_then_fetch_exact_public_source",
                )
                if source["id"] in locators and locators[source["id"]] != entry:
                    raise ValueError("acquisition_source_locator_conflict")
                locators[source["id"]] = entry
    candidates = [
        observation_alias_candidate(state, "visual:quality", "descriptive:IMAGE_QUALITY"),
        observation_alias_candidate(state, "visual:fovea", "descriptive:FOVEA"),
    ]
    return dict(
        schema_version="first-case-acquisition-audit/1.0.0",
        plan=plan.model_dump(mode="json"),
        counts=dict(Counter(c.disposition for c in plan.cards)),
        source_gaps=source_gaps,
        reference_recovery_queue=list(locators.values()),
        task_labels={
            key: (value.get("definition") or {}).get("title", key)
            for key, value in catalog["tasks"].items()
        },
        observation_reuse_candidates=candidates,
        pilot_tasks=list(PRIORITY),
        source_catalog_checksum=digest(catalog),
        persistence="export_only_not_database_or_current_authorization",
        real_requests=0,
        new_scores=False,
        old_scores_preserved=True,
    )


def render_report(result):
    """Small escaped gap view; not another medical graph or score report."""
    rows = []
    for c in result["plan"]["cards"]:
        values = (
            result.get("task_labels", {}).get(c["assessment_id"], c["assessment_id"])
            + " · "
            + c["assessment_id"],
            LABELS.get(c["purpose"], c["purpose"]),
            LABELS.get(c["observation_status"], c["observation_status"]),
            LABELS.get(c["interpretation_status"], c["interpretation_status"]),
            LABELS.get(c["disposition"], c["disposition"]),
            ", ".join(c["missing_inputs"] or c["missing_source_ids"] or c["blockers"]),
            " → ".join(LABELS.get(a, a) for a in c["next_actions"]),
        )
        rows.append("<tr>" + "".join("<td>" + escape(v) + "</td>" for v in values) + "</tr>")
    return (
        '<!doctype html><html lang="zh"><meta charset="utf-8">'
        "<title>病例主动取证缺口清单</title><style>"
        "body{font:15px sans-serif;margin:24px;color:#223}"
        "table{border-collapse:collapse;width:100%}"
        "th,td{border:1px solid #ccd;padding:10px;text-align:left;vertical-align:top}"
        "th{background:#edf2f7}tr:nth-child(even){background:#fafafa}</style>"
        "<h1>病例主动取证缺口清单</h1><p>基于已保存病例材料的准备视图；"
        "未执行新取证、未修改原评分。辅助项目不要求独立疾病阈值。</p>"
        "<p>优先试点：垂直杯盘比、最薄径向盘沿比例、局灶盘沿观察。"
        "同一观察重复出现时按来源复用，不算两份独立证据。</p><table><thead><tr>"
        + "".join(
            "<th>" + t + "</th>"
            for t in ("任务", "用途", "观察", "判读定义", "当前处置", "具体缺项", "处理路径")
        )
        + "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></html>"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("audit", "prepare-vision", "search-plan"))
    parser.add_argument("--state", type=Path, default=STATE)
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--catalog", type=Path, default=CATALOG)
    parser.add_argument("--task", default="criterion:RIM_NOTCH")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--html", type=Path)
    args = parser.parse_args(argv)
    paths = [p for p in (args.output, args.html) if p is not None]
    if len(set(paths)) != len(paths) or any(p.exists() or not p.parent.is_dir() for p in paths):
        parser.error("outputs must be distinct new files in existing directories")
    if args.html and args.command != "audit":
        parser.error("HTML is only available for audit")
    state, catalog = load_case(args.state, args.report), load_catalog(args.catalog)
    result = audit(state, catalog)
    if args.command == "prepare-vision":
        requirement = next(
            r for r in requirements_for(state, catalog) if r.assessment_id == args.task
        )
        if requirement.acquisition != "image":
            raise ValueError("acquisition_task_not_visual")
        image = next(o.image for o in state.material.observations if o.image)
        task = next(t for t in project_review_tasks(state).tasks if t.assessment_id == args.task)
        instruction = (
            "只检查此项并说明可见部位与遮挡；不推测其他检查。" + task.definition.examination
        )
        prepared = prepare_visual_task(state, args.task, image=image, instruction=instruction)
        result = dict(
            schema_version="targeted-vision-preparation/1.0.0",
            parent_checksum=digest(state),
            state=prepared.model_dump(mode="json"),
            requires_new_owner_installation_and_gate=True,
            execution_authorized=False,
            real_requests=0,
        )
    elif args.command == "search-plan":
        plan = plan_acquisition(state, requirements_for(state, catalog))
        card = next(c for c in plan.cards if c.assessment_id == args.task)
        suggestion = SearchSuggestion(
            assessment_id=card.assessment_id, topics=card.public_topics[:1], purpose="association"
        )
        query = search_plan(
            card,
            suggestion,
            SearchProgress(),
            ("SOURCE:PROVIDER:EUROPEPMC", "SOURCE:PROVIDER:PUBMED"),
        )
        result = dict(
            schema_version="acquisition-search-preparation/1.0.0",
            query=query.model_dump(mode="json"),
            execution_authorized=False,
            real_requests=0,
        )
    if args.output:
        write_once(args.output, result)
    if args.html:
        with args.html.open("x", encoding="utf-8") as target:
            target.write(render_report(result))
    print(
        json.dumps(
            result
            if not args.output
            else dict(
                output=str(args.output), html=str(args.html) if args.html else None, real_requests=0
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
