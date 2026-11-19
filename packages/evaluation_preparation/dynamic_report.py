"""Readable case export, derived from exact original Runtime records."""

import html
import json
from pathlib import Path

from agents.expert_agent.reporting import render_research_html
from apps.vectorengine_single_case import write_once
from packages.evidence_details.acceptance_models import DomainState
from packages.grounding.models import canonical, digest, validate


class GenericAggregateReportView:
    """Read original observation paths and a separately saved aggregate Review.

    This view is only a report producer input. It never changes the campaign's
    dispatch paths, selected targets, original output, scores or permissions.
    The caller must first restore the aggregate through its original owner.
    """

    def __init__(self, campaign, output):
        self._source = campaign
        self.definition_stage = None  # Whole-case production must not inherit increment labeling.
        self.output = Path(output)
        self.report_name = self.output.name
        self.report_key = "accepted-aggregation-report"
        self.review_report_path = self.output / "review/review-report.json"
        self.export_suffix = "_原接纳组合_" + self.output.name

    def __getattr__(self, name):
        return getattr(self._source, name)

    def definition_aggregation_metadata(self, parent, result):
        if result.get("definition_revision_policy") is None:
            return None
        docs = self.documents(parent.material.scope.run_id)
        saved = docs.read("generic-definition-wholecase-projection")
        if (
            not saved
            or saved[0]["selected_only"]
            or saved[0]["projected_state_checksum"] != digest(parent)
            or digest(saved[0]) != result["definition_revision_projection_checksum"]
        ):
            raise ValueError("generic_definition_aggregate_report_projection")
        return dict(
            schema_version="generic-definition-wholecase-report-scope/1.0.0",
            definition_projection=saved[0],
            policy=result["definition_revision_policy"],
            original_result_checksum=result["original_result_checksum"],
            origins=result["origins"],
            new_model_requests=0,
            old_judgments_superseded_only_for_revised_items=True,
            historical_scores_modified=False,
        )

    def observation_phase_path(self, target):
        from .generic_acquisition_successor import AcquisitionSuccessor
        from .generic_capability_successor import CapabilitySuccessor
        from .generic_review import GenericReview

        current, seen = self._source, set()
        for _ in range(17):
            if id(current) in seen:
                raise ValueError("aggregate_report_observation_cycle")
            seen.add(id(current))
            if type(current) is CapabilitySuccessor:
                if target in current.visual_targets:
                    return current.phase_path(target)
            elif isinstance(current, AcquisitionSuccessor):
                if target == current.selected_target:
                    return current.phase_path(target)
            elif not isinstance(current, GenericReview):
                return current.phase_path(target)
            current = current.acquisition
        raise ValueError("aggregate_report_observation_depth")

    def capability_aggregation_metadata(self, parent, result):
        if result.get("capability_revision_policy") is None:
            return None
        docs = self.documents(parent.material.scope.run_id)
        saved = docs.read("generic-capability-wholecase-projection")
        if (
            not saved
            or len(saved) != 1
            or saved[0]["phase_input_checksum"] != digest(parent)
            or digest(saved[0]) != result["capability_revision_projection_checksum"]
        ):
            raise ValueError("generic_capability_aggregate_report_projection")
        return dict(
            schema_version="generic-capability-wholecase-report-scope/1.0.0",
            projection=saved[0],
            policy=result["capability_revision_policy"],
            original_result_checksum=result["original_result_checksum"],
            origins=result["origins"],
            new_model_requests=0,
            historical_scores_modified=False,
            whole_case_aggregation_required=False,
        )

    def process_evaluations(self):
        from copy import copy

        from .generic_campaign import GenericCampaign

        # The existing producer reads this output for its aggregate score only;
        # facts and execution provenance remain bound to the original owners.
        view = copy(self._source)
        view.output = self.output
        view.accepted_observations = self._source.accepted_observations
        view.read_model = self._source.read_model
        view.phase_path = self.observation_phase_path
        return GenericCampaign.process_evaluations(view)


def export(campaign, result, *, directory=Path("docs/exports")):
    final = result["phases"][-1]
    docs = campaign.documents(final["run_id"])
    parent = validate(DomainState, docs.read("dynamic-phase-input")[0])
    detail = None
    if final["accepted"]:
        detail = json.loads((campaign.output / "review/review-report.json").read_bytes())
        saved = docs.read("accepted:" + final["acceptance_key"])
        if (
            not saved
            or tuple(detail[k] for k in ("records", "graph", "receipt"))
            != (saved[0], saved[1], saved[3])
            or detail["scores"] != saved[2]["scores"]
            or detail["material"] != parent.material.model_dump(mode="json")
        ):
            raise ValueError("dynamic_case_export_owner_binding")
        rendered = render_research_html(detail)
        scores = detail["scores"]
    else:
        zero = docs.read("dynamic-zero-evidence")
        scores = zero[0]["scores"] if zero else None
        rendered = "<!doctype html><html lang='zh'><meta charset='utf-8'><body></body></html>"
    items = {p.id for p in parent.policy.items}
    attempts = (
        campaign.accepted_observations() if hasattr(campaign, "accepted_observations") else {}
    )
    rows = []
    for requirement in parent.policy.requirements:
        status = (
            "已有结果"
            if requirement.id in items
            else {
                "patient_modality_not_provided": "当前输入无法取证",
                "observation_ungradable": "已完成视觉检查，本项图像不足以判读",
            }.get(requirement.missing_reason, "本轮未取得：保留需求，不建立空评分项")
        )
        rows.append(
            dict(
                id=requirement.id,
                examination=requirement.definition.examination,
                modality=list(requirement.input_modalities),
                status=status,
                observation=attempts[requirement.id].model_dump(mode="json")
                if requirement.id in attempts
                else None,
                sources=[r.model_dump(mode="json") for r in requirement.definition.source_refs],
            )
        )
    from .dynamic_model import read_model

    model = (
        campaign.read_model()
        if hasattr(campaign, "read_model")
        else read_model(campaign.base, campaign.target)
    )
    report = dict(
        schema_version="dynamic-case-report/1.0.0",
        result=result,
        requirements=rows,
        acquired_count=len(items),
        required_count=len(rows),
        scores=scores,
        original_review=detail,
        model_output=model,
        segmentation_qc=campaign.base.quality,
        reasoning_scope="source-first; four selected SLO checks and optional existing classifier",
        missing_is_negative=False,
        calibrated_probability=False,
        clinical_diagnosis=False,
    )
    if hasattr(campaign, "process_evaluations"):
        report["process_evaluations"] = campaign.process_evaluations()
        report["reasoning_scope"] = (
            "all_source_backed_requirements; task_specific_partial_visibility"
        )
        report["display_transform"] = campaign.base.view_policy
        report["native_tools"] = campaign.base.tools
        report["counts"] = dict(
            requirements=len(rows),
            acquired=len(items),
            reviewed=len(scores.get("reviews", [])) if scores else 0,
            directed=sum(r["judgment"]["value"] is not None for r in scores.get("reviews", []))
            if scores
            else 0,
        )
    if hasattr(campaign, "recovery"):
        report["recovery"] = campaign.recovery
    score = scores["root"]["summary"]["confidence"]["value"] if scores else None
    if getattr(campaign, "structured_completion", False):
        pending = (
            scores.get("review_completion", {}).get("pending_ids", []) if scores else sorted(items)
        )
        report["completion"] = dict(
            report_saved=True,
            domain_accepted=final["accepted"],
            all_obtained_reviewed=bool(scores is not None and not pending),
            pending_review_ids=pending,
            evidence_limited=final.get("evidence_limited", False),
            semantics="saved_report_is_not_full_review_or_clinical_validation",
        )

    if getattr(campaign, "definition_stage", None) is not None:
        from .generic_definition_successor import report_metadata

        scope = report_metadata(campaign, parent, result)
        report["definition_revision"] = scope
        report["review_scope"] = dict(
            kind="definition_increment_only",
            explanation="本页为新定义评阅增量；原完整病例与评分保留，本页研究分不是整例组合结果。",
        )
        report["reasoning_scope"] = "new_definition_review_increment_not_whole_case"
        report["completion"].update(
            all_obtained_reviewed=False,
            whole_case_aggregation_required=scope["whole_case_aggregation_required"],
            semantics="increment_saved_original_complete_case_retained",
            scope_notice="新定义评阅增量，原完整病例评分保留；本页不是整例组合结果",
            definition_revision=scope,
        )

    from .generic_capability_successor import CapabilitySuccessor

    if type(campaign) is CapabilitySuccessor:
        from .generic_capability_execution import report_metadata as capability_metadata

        scope = capability_metadata(campaign, parent)
        report["capability_revision"] = scope
        report["review_scope"] = dict(
            kind="capability_increment_only",
            explanation="本页仅为能力匹配产生的新增需求及其评阅，原病例评分保持；尚需原接纳入口整例组合。",
        )
        report["reasoning_scope"] = "capability_demand_increment_not_whole_case"
        report["completion"].update(
            all_obtained_reviewed=False,
            whole_case_aggregation_required=True,
            semantics="increment_saved_original_complete_case_retained",
            scope_notice=report["review_scope"]["explanation"],
            capability_revision=scope,
        )

    if isinstance(campaign, GenericAggregateReportView):
        capability = campaign.capability_aggregation_metadata(parent, result)
        if capability is not None:
            report["capability_revision"] = capability
            report["review_scope"] = dict(
                kind="whole_case_capability_aggregation",
                explanation="本例已接纳的原判读与新增需求判读，由原评分内核组合；历史评分保留。",
            )
            report["reasoning_scope"] = "whole_case_original_owner_capability_aggregation"
            report["completion"].update(
                whole_case_aggregation_required=False,
                capability_revision=capability,
                semantics="original_owner_whole_case_aggregation_not_new_joint_model_review",
            )
        scope = campaign.definition_aggregation_metadata(parent, result)
        if scope is not None:
            report["definition_revision"] = scope
            report["review_scope"] = dict(
                kind="whole_case_definition_reaggregation",
                explanation="新定义的已接纳判读替换对应旧判读，其余有效判读保留；原评分内核重新聚合，旧报告保持。",
            )
            report["reasoning_scope"] = "whole_case_original_owner_definition_reaggregation"
            report["completion"].update(
                whole_case_aggregation_required=False,
                definition_revision=scope,
                semantics="original_owner_whole_case_reaggregation_not_new_joint_model_review",
            )

    def esc(x):
        return html.escape(str(x), quote=True)

    intro = "<section style='margin:24px;padding:24px;background:#eef5ff;border-radius:12px'>"
    intro += f"<h1>{esc(campaign.sample)} · 动态证据图</h1>"
    if report.get("review_scope", {}).get("kind") == "capability_increment_only":
        intro += f"<p>{esc(report['review_scope']['explanation'])}</p>"
    if report.get("review_scope", {}).get("kind") == "definition_increment_only":
        intro += (
            "<p>本页为新定义评阅增量，保留原完整病例及其评分；"
            "本页研究分仅表示本轮增量，不是整例组合结果。"
            "无有效定义增量时不发送空评阅，不生成替代旧评分。</p>"
        )
    if "completion" in report:
        completion = report["completion"]
        intro += (
            "<p>报告已保存；已取得证据的评阅："
            + ("全部完成" if completion["all_obtained_reviewed"] else "部分完成")
            + "；待评阅 "
            + str(len(completion["pending_review_ids"]))
            + " 项。</p>"
        )
    if "recovery" in report:
        intro += (
            "<p>本报告为已确认评阅的本地收尾：0次新API，原答案未改写；"
            "旧运行的deadline失败和停止状态保留，不授权继续执行。</p>"
        )
    intro += f"<p>实际证据 {len(items)} 项 / 文献及工具依据需求 {len(rows)} 项；"
    intro += (
        f"本轮请求 {result['new_requests']} 次，旧图请求 {result['prior_graph_requests']} 次。</p>"
    )
    intro += f"<p>综合研究分：{esc(score)}。不是患病概率，不以50分作为临床诊断阈值。</p>"
    intro += (
        "<p>可计算测量保留，质量 Q 作为权重累乘因子；空掩膜记录缺项，不填假数值。"
        if parent.policy.schema_version == "research-task-policy/4.0.0"
        else "<p>两例采用相同四项视觉取证流程；分割未通过本例QC的测量不进入判读。"
    )
    intro += "缺项保留在下表，不填负分、不复制旧病例答案。</p><h2>1 · 取证清单</h2>"
    intro += "<table><tr><th>检查</th><th>模态</th><th>本轮状态与观察</th></tr>"
    intro += (
        "".join(
            f"<tr><td>{esc(r['examination'])}</td><td>{esc(', '.join(r['modality']))}</td>"
            f"<td>{esc(r['status'])}<br>"
            f"{esc(r['observation']['value'] if r['observation'] else '')}</td></tr>"
            for r in rows
        )
        + "</table>"
    )
    intro += (
        "<h2>2 · 已有工具与输入质量</h2><pre>"
        + esc(
            json.dumps(
                dict(segmentation_qc=campaign.base.quality, fairclip=model),
                ensure_ascii=False,
                indent=2,
            )
        )
        + "</pre>"
    )
    intro += "<h2>3 · 已接纳的观察、判读、评分与溯源</h2></section>"
    if "process_evaluations" in report:
        intro += "<section style='margin:24px'><h2>逐项过程评价与部分观察</h2><table><tr>"
        intro += (
            "<th>检查</th><th>取得</th><th>覆盖比例</th>"
            "<th>有效质量</th><th>缺口</th><th>处理</th></tr>"
        )
        for r in report["process_evaluations"]:
            intro += (
                "<tr>"
                + "".join(
                    "<td>" + esc(r.get(k)) + "</td>"
                    for k in (
                        "id",
                        "availability_score",
                        "inspected_fraction",
                        "effective_quality",
                        "gap_score",
                        "action",
                    )
                )
                + "</tr>"
            )
        intro += (
            "</table><p>覆盖比例为视觉模型的任务级估计；未知区域未虚构病变概率。"
            "测量质量见对应Q/O。</p></section>"
        )
    if getattr(campaign.base, "unified", False):
        from .evidence_view import normalize, render

        report["requirement_inventory"] = campaign.base.structure["inventory"]
        report["input_context"] = campaign.base.input_context
        if getattr(campaign, "spec", {}).get("upstream_profile"):
            from packages.evidence_reuse.integration import saved_report_context

            report["upstream_reuse"] = saved_report_context(campaign)
        if getattr(campaign, "spec", {}).get("report_profile"):
            from .report_production import campaign_report

            report = campaign_report(report, campaign, parent)
        rendered = render(normalize(report, input_checksum=digest(report)))
    else:
        # Legacy exports remain exact; current read-only conversion fixes layout
        # without mutating any previously saved HTML or recovery payload.
        body_start = rendered.find(">", rendered.find("<body")) + 1
        rendered = rendered[:body_start] + intro + rendered[body_start:]
    report = json.loads(canonical(report))
    with docs.session():
        docs.write(
            getattr(campaign, "report_key", "dynamic-case-report"),
            (report, dict(html=rendered, report_checksum=digest(report))),
        )
    paths = [
        directory
        / (
            "青光眼证据图_"
            + Path(campaign.sample).stem
            + getattr(campaign, "export_suffix", "_动态取证版")
            + ext
        )
        for ext in (".json", ".html")
    ]
    directory.mkdir(parents=True, exist_ok=True)
    write_once(paths[0], report)
    if paths[1].exists() and paths[1].read_text() != rendered:
        raise ValueError("dynamic_html_export_conflict")
    if not paths[1].exists():
        paths[1].write_text(rendered)
    return dict(json=str(paths[0]), html=str(paths[1]), report_checksum=digest(report))
