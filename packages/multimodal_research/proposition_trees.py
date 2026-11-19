"""Independent case/proposition display snapshots, never new medical acceptance.

Historical joint reviews retain their exact source and joint-review provenance.
The root is a single proposition; comparison lives outside all evidence trees.
"""

import copy
import html
import json
from pathlib import Path

from apps.vectorengine_single_case import write_once
from packages.evaluation_preparation.evidence_presentation import validate_packet
from packages.evaluation_preparation.evidence_tree_v3 import bundle, render_packet
from packages.grounding.models import digest

VERSION = "case-proposition-tree/1.1.0"


def split_packet(report, packet):
    """Project original accepted per-proposition summaries without re-scoring."""
    if packet["report_content_checksum"] != digest(report):
        raise ValueError("proposition_source_binding")
    scores = report["scores"]
    hypotheses = scores.get("root", {}).get("summary", {}).get("hypotheses", [])
    if not hypotheses or len({h["hypothesis"]["code"] for h in hypotheses}) != len(hypotheses):
        raise ValueError("proposition_independent_summaries_required")
    result = []
    for hypothesis in hypotheses:
        goal, summary = hypothesis["hypothesis"], hypothesis["summary"]
        code = goal["code"]
        selected = {r["id"] for r in scores["reviews"] if r["target"] == code}
        if selected != set(summary["assessment_ids"]):
            raise ValueError("proposition_summary_binding")
        requirements = {
            r["id"] for r in report["requirements"] if r["id"].endswith(":hyp:" + goal["label"])
        }
        # Validate against the owner's explicit bindings, not the suffix alone.
        policy = report["production_inputs"]["policy"]
        if requirements != {k for k, v in policy["hypothesis_bindings"].items() if v == code}:
            raise ValueError("proposition_requirement_binding")
        value = copy.deepcopy(packet)
        chosen = {k: c for k, c in value["checks"].items() if k in requirements}
        keep = {"root"}
        for check in chosen.values():
            keep.add(check["node_id"])
        # Each check's six section nodes and resource edges remain attached.
        changed = True
        while changed:
            before = len(keep)
            for edge in value["edges"]:
                if edge["source"] in keep and edge["source"] != "root":
                    keep.add(edge["target"])
            changed = len(keep) != before
        value["nodes"] = [n for n in value["nodes"] if n["id"] in keep]
        value["edges"] = [e for e in value["edges"] if e["source"] in keep and e["target"] in keep]
        for check in chosen.values():
            value["edges"].append(
                dict(
                    source="root",
                    target=check["node_id"],
                    relation="contains",
                    origin=dict(
                        kind="single_proposition_projection",
                        pointer="/production_inputs/policy/hypothesis_bindings",
                        checksum=digest(policy["hypothesis_bindings"]),
                    ),
                )
            )
        for node in value["nodes"]:
            if node["id"] == "root":
                node["title"] = goal["title"] + "：本病例是否支持此命题"
                node["goal_summary"] = goal["meaning"]
            elif node["id"] in {c["node_id"] for c in chosen.values()}:
                node["parent"] = "root"
        value["navigation"] = [
            n
            for n in value["nodes"]
            if n["id"] == "root" or n["id"] in {c["node_id"] for c in chosen.values()}
        ]
        value["checks"] = chosen
        projected_scores = dict(
            schema_version="proposition-score-projection/1.0.0",
            source_scores_checksum=digest(scores),
            target=code,
            reviews=[r for r in scores["reviews"] if r["id"] in selected],
            root=dict(summary=summary, coverage=scores["root"]["coverage"]),
            clinical_acceptance=False,
        )
        value["scores"] = projected_scores
        value["scores_checksum"] = digest(projected_scores)
        value["confidence"] = summary["confidence"]["value"]
        rows = projected_scores["reviews"]
        value["counts"] = dict(
            requirements=len(requirements),
            reviewed=len(rows),
            numeric=sum(r["judgment"]["value"] is not None for r in rows),
            nonzero_contributors=sum(bool(r["support"] or r["refutation"]) for r in rows),
            neutral=sum(r["judgment"]["value"] == 0 for r in rows),
            unassessed=sum(r["judgment"]["value"] is None for r in rows),
        )
        value["assets"]["links"] = [x for x in value["assets"]["links"] if x["node_id"] in keep]
        linked = {x["asset_id"] for x in value["assets"]["links"]}
        value["assets"]["items"] = [x for x in value["assets"]["items"] if x["id"] in linked]
        value["coverage"]["entries"] = [
            x
            for x in value["coverage"]["entries"]
            if x["destination"] in keep or x["destination"].startswith("appendix:")
        ]
        value["removals"] = []  # Joint intervention reports are retained only in source appendix.
        identity = dict(
            case_code=report["result"]["case_code"],
            target=code,
            source_report_checksum=digest(report),
            projection_version=VERSION,
        )
        value["proposition_tree"] = dict(
            schema_version=VERSION,
            tree_id="TREE:" + digest(identity)[7:39],
            **identity,
            acquisition_shared=True,
            review_origin="historical_joint_review",
            new_review=False,
            new_acceptance=False,
            rescored=False,
            comparison_is_external=True,
            clinical_diagnosis=False,
        )
        value["diagnostics"].append(dict(code="historical_joint_review_derived_single_proposition"))
        validate_packet(value)
        result.append((goal, value))
    return result


def export_saved(source, output):
    """Read authoritative saved report/HTML, append separate immutable Runtime objects."""
    from db.runtime.repository.planner import PlannerRepository
    from db.runtime.repository.research import ResearchRunRepository
    from packages.research_runtime.agent_records import RuntimeDocuments

    from .entry import owners
    from .report_storage import save, unpack

    source, out = Path(source), Path(output)
    if out.exists():
        raise ValueError("proposition_fresh_output_required")
    report = json.loads(source.read_bytes())
    final = report["result"]["phases"][-1]
    runtime, objects, inputs, target = owners()
    visible = inputs.read_visible(report["result"]["case_code"])
    documents = RuntimeDocuments(
        repo=runtime,
        contexts=ResearchRunRepository(target),
        sessions=PlannerRepository(target),
        store=objects,
        run_id=final["run_id"],
        root=visible.case.artifacts[0],
    )
    key = report.get("recorded_finalization", {}).get("report_key")
    # The original owner declares the historical report key in its result.
    keys = (
        [key]
        if key
        else [
            "multimodal-recorded-finalization-report",
            "multimodal-partial-finalization-report",
            "dynamic-case-report",
        ]
    )
    saved = None
    for candidate in keys:
        raw = documents.read(candidate)
        if raw:
            found, page = unpack(raw, key=candidate)
            if found == report:
                saved = page["html"]
                break
    if saved is None:
        raise ValueError("proposition_saved_report_not_found")
    marker = '<script id="evidence-view" type="application/json">'
    packet = json.loads(saved.split(marker, 1)[1].split("</script>", 1)[0])
    trees = split_packet(report, packet)
    out.mkdir(parents=True, mode=0o700)
    entries = []
    for goal, value in trees:
        body = bundle(value)
        page = render_packet(value)
        identity = value["proposition_tree"]
        save(documents, body, page, key="proposition-tree:" + identity["tree_id"])
        name = goal["label"]
        write_once(out / (name + ".json"), body)
        with (out / (name + ".html")).open("x") as stream:
            stream.write(page)
        entries.append(
            dict(
                **identity,
                title=goal["title"],
                html=name + ".html",
                json=name + ".json",
                score=value["confidence"],
                counts=value["counts"],
                images=len(value["assets"]["items"]),
            )
        )
    index = dict(
        schema_version="case-proposition-comparison/1.0.0",
        trees=entries,
        source_report_checksum=digest(report),
        new_api_requests=0,
        class_label=None,
        probabilities=None,
        comparison=report["scores"]["hypothesis_comparison"],
    )
    write_once(out / "comparison.json", index)
    links = "".join(
        '<li><a href="'
        + e["html"]
        + '">'
        + html.escape(e["title"])
        + "</a> — 研究分 "
        + ("未知" if e["score"] is None else f"{e['score'] * 100:.4f}/100")
        + "</li>"
        for e in entries
    )
    page = (
        '<!doctype html><meta charset="utf-8"><h1>同一病例的独立命题树</h1>'
        "<p>每棵树仅回答一个命题。本页只比较，不是总根节点。"
        "历史联合评阅派生；没有新请求或重新评分。最高分不等于确诊。</p><ul>" + links + "</ul>"
    )
    with (out / "comparison.html").open("x") as stream:
        stream.write(page)
    save(documents, index, page, key="proposition-comparison:" + digest(index)[7:39])
    return dict(output=str(out), trees=len(entries), new_api_requests=0, runtime_readback=True)
