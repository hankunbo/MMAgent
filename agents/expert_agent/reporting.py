"""Exact version dispatch for research successor reports; legacy exporters untouched."""

import html
import json

from packages.evidence_details.acceptance_models import (
    ResearchGraphValue,
    read_receipt,
    read_records,
)
from packages.evidence_scoring.api import validate as validate_scores
from packages.grounding.models import digest, validate
from packages.llm_provider.models import Material


def read_research_report(acceptance, *, key, parent):
    saved = acceptance.read(key=key)
    if not saved:
        raise ValueError("research_report_missing")
    receipt = read_receipt(saved[-1])
    if receipt.status != "accepted":
        return dict(receipt=receipt.model_dump(mode="json"))
    if len(saved) != 4:
        raise ValueError("research_report_version")
    records, graph = read_records(saved[0]), validate(ResearchGraphValue, saved[1])
    if (records.status == "accepted_live_research") != (
        receipt.execution == "bounded_live_research"
    ):
        raise ValueError("research_report_execution_identity")
    scores = saved[2]["scores"]
    if (
        records.scope != parent.material.scope
        or graph.scope != records.scope
        or digest(records) != receipt.records_checksum
        or graph.records_checksum != digest(records)
        or graph.scores_checksum != receipt.scores_checksum
        or (digest(scores) if scores is not None else None) != receipt.scores_checksum
        or receipt.graph
        != acceptance.documents.reference(
            "accepted:" + key, 1, graph, role="graph", version=graph.schema_version
        )
    ):
        raise ValueError("research_report_binding")
    if scores is not None:
        inputs = dict(state=parent, candidate=records.candidate)
        if records.schema_version in {
            "case-observation-interpretation/2.4.0",
            "case-observation-interpretation/2.6.0",
        }:
            inputs["reviewed_ids"] = tuple(i.assessment_id for i in records.candidate.items)
        validate_scores(inputs, scores)
    report = dict(
        schema_version="research-successor-report/1.0.0",
        clinical_acceptance=False,
        material=parent.material.model_dump(mode="json"),
        receipt=receipt.model_dump(mode="json"),
        records=records.model_dump(mode="json"),
        graph=graph.model_dump(mode="json"),
        scores=scores,
        source_uses=[
            dict(
                reference=s.reference.model_dump(mode="json"), purpose=s.purpose, locator=s.locator
            )
            for s in parent.material.sources
        ],
    )
    submission = acceptance.documents.read("submission:" + key)
    if submission and submission[0].get("schema_version") == "research-submission-receipt/1.1.0":
        report.update(
            schema_version="research-successor-report/1.1.0",
            source_submission=submission[0]["submission"],
            source_quote_validation=submission[0]["source_quote_validation"],
        )
        _check_source_quote_report(report)
    return report


def _check_source_quote_report(report):
    from packages.evidence_details.review_batches import read_submission
    from packages.evidence_details.source_quotes import source_quote_proof
    from packages.knowledge_claims.research_quote import LAYOUT_POLICY

    submission = read_submission(report["source_submission"])
    material = validate(Material, report["material"])
    if (
        digest(submission) != report["receipt"]["submission_checksum"]
        or submission.candidate.model_dump(mode="json") != report["records"]["candidate"]
        or submission.scope != material.scope
        or submission.request_id != report["records"]["request_id"]
        or source_quote_proof(submission, material, policy=LAYOUT_POLICY)
        != report["source_quote_validation"]
    ):
        raise ValueError("research_report_quote_binding")


def render_research_html(report):
    if report.get("schema_version") not in {
        "research-successor-report/1.0.0",
        "research-successor-report/1.1.0",
    }:
        raise ValueError("research_html_version")
    if report["schema_version"] == "research-successor-report/1.1.0":
        _check_source_quote_report(report)
    elif "source_quote_validation" in report or "source_submission" in report:
        raise ValueError("research_html_version")
    records, graph, scores = report["records"], report["graph"], report["scores"]
    receipt = report["receipt"]
    read_receipt(receipt)
    parsed = read_records(records)
    if (parsed.status == "accepted_live_research") != (
        receipt.get("execution") == "bounded_live_research"
    ):
        raise ValueError("research_html_execution_identity")
    material = validate(Material, report["material"])
    sources = [
        dict(reference=s.reference.model_dump(mode="json"), purpose=s.purpose, locator=s.locator)
        for s in material.sources
    ]
    if (
        digest(records) != receipt["records_checksum"]
        or graph["records_checksum"] != digest(records)
        or (digest(scores) if scores is not None else None) != receipt["scores_checksum"]
        or report["source_uses"] != sources
        or records["scope"] != material.scope.model_dump(mode="json")
        or graph["scope"] != records["scope"]
        or graph["scores_checksum"] != receipt["scores_checksum"]
    ):
        raise ValueError("research_html_binding")
    # Entire visible provenance is escaped data, including hostile source text/codes.
    body = html.escape(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    heading = (
        "已接纳研究判读的本地汇总（零新模型请求）"
        if receipt.get("execution") == "local_reaggregation_of_accepted_reviews"
        else "有界真实 LLM 研究接纳记录"
        if receipt.get("execution") == "bounded_live_research"
        else "研究离线 / Fake 接纳记录"
    )
    if parsed.schema_version in {
        "case-observation-interpretation/2.3.0",
        "case-observation-interpretation/2.4.0",
        "case-observation-interpretation/2.5.0",
        "case-observation-interpretation/2.6.0",
        "case-observation-interpretation/2.7.0",
        "case-observation-interpretation/2.8.0",
        "case-observation-interpretation/2.9.0",
        "case-observation-interpretation/2.10.0",
        "case-observation-interpretation/2.11.0",
        "case-observation-interpretation/2.12.0",
        "case-observation-interpretation/2.14.0",
        "case-observation-interpretation/2.19.0",
    }:
        return _batched_html(report, heading, body)
    return (
        '<!doctype html><html lang="zh"><meta charset="utf-8"><title>研究候选接纳记录</title>'
        f"<h1>{heading}</h1><p>非正式医学图；非临床结论。未知不等于阴性。</p>"
        f"<p>报告摘要：{digest(report)}</p><pre>{body}</pre></html>"
    )


def _batched_html(report, heading, trace):
    """Six owner-derived sections; numerical values come solely from original v6.1."""

    def data(value):
        return html.escape(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2))

    records, material, scores = report["records"], report["material"], report["scores"]
    observations = {o["reference"]["code"]: o for o in records["observation_facts"]}
    tasks = {t["assessment_id"]: t for t in records["review_tasks"]["tasks"]}
    interpretations = {i["review"]["assessment_id"]: i for i in records["interpretations"]}
    sections = []
    for row in scores["reviews"]:
        task, interpretation = tasks.get(row["id"]), interpretations.get(row["id"])
        if task:
            source_codes = {s["code"] for s in task["allowed_sources"]}
            values = [
                [s for s in material["sources"] if s["reference"]["code"] in source_codes],
                task["definition"]["applicability_conditions"],
                {k: task[k] for k in ("observation", "method", "unit", "laterality")},
                observations[task["observation"]["code"]],
                dict(reference=task["definition"], review=interpretation["review"]),
                dict(
                    dependency_paths=interpretation["dependency_paths"],
                    correlation_group=interpretation["correlation_group"],
                    dependence_block=interpretation["dependence_block"],
                ),
            ]
        else:
            values = [
                dict(source_binding_checksum=row["source_binding_checksum"]),
                "原 owner 固定模型映射；本轮未请求 LLM 重评。",
                row["dependency_paths"],
                dict(assessment_checksum=row["assessment_checksum"]),
                {k: row[k] for k in ("observation", "relation", "applicability", "judgment")},
                {k: row[k] for k in ("correlation_group", "dependence_block", "dependency_paths")},
            ]
        labels = (
            "来源事实",
            "我们的提炼理由",
            "本节点取证需求",
            "本例工具与观察结果",
            "目前如何判读",
            "计算依赖与判读背景",
        )
        sections.append(
            "<article><h2>"
            + html.escape(row["id"])
            + "</h2>"
            + "".join(
                f"<section><h3>{n}. {label}</h3><pre>{data(value)}</pre></section>"
                for n, (label, value) in enumerate(zip(labels, values, strict=True), 1)
            )
            + "</article>"
        )
    return (
        '<!doctype html><html lang="zh"><meta charset="utf-8"><title>分批研究评阅报告</title>'
        f"<h1>{heading}</h1><p>非正式医学图；非临床结论。未知不等于阴性。"
        "研究评分不是校准概率，50 不是诊断阈值。</p>"
        f"<p>Run：{html.escape(material['scope']['run_id'])}；报告摘要：{digest(report)}</p>"
        "<h2>技术结果与研究评分</h2><p>"
        + (
            "部分批次已接纳；未完成项不计为阴性。"
            if records.get("completion") == "partial"
            else "已接纳逐项判读已本地汇总，未发起新评阅。"
            if records.get("execution") == "local_reaggregation_of_accepted_reviews"
            else "完整批次已接纳。"
        )
        + "本页展示保存的观察与评阅，工具执行次数以原执行记录为准。</p>"
        "<pre>"
        + data(
            dict(
                root=scores["root"],
                medical_completeness="按各项缺项保留",
                sensitivity={k: v for k, v in scores.items() if k not in {"reviews", "root"}},
            )
        )
        + "</pre>"
        + (
            "<p>来源显示编号经过原 owner 精确映射；医学数值和原解释未改写。"
            "原解释文字中的编号按对应原评阅档案读取，结构化引用使用本报告编号。</p><pre>"
            + data(records["origins"])
            + "</pre>"
            if records.get("schema_version") == "case-observation-interpretation/2.19.0"
            else ""
        )
        + "".join(sections)
        + "<details><summary>批次、原回执与完整追溯</summary><pre>"
        + trace
        + "</pre></details></html>"
    )
