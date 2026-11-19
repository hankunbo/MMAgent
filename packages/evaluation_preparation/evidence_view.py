"""Read-only, disease-independent presentation adapters. Never recalculate scores."""

import html
import json
import re
from pathlib import Path

from packages.object_store import checksum

TYPES = {
    "goal": ("目标与组织", "#64748b"),
    "requirement": ("检查需求", "#2563eb"),
    "observation": ("病例事实与观察", "#0891b2"),
    "interpretation": ("判读与贡献", "#7c3aed"),
    "context": ("背景与质量", "#ca8a04"),
    "execution": ("执行与依赖", "#475569"),
}


def load_report(path):
    raw = Path(path).read_bytes()
    if len(raw) > 32 * 1024 * 1024:
        raise ValueError("view_input_too_large")
    if Path(path).suffix.lower() == ".html":
        match = re.search(rb'<script\b[^>]*\bid=[\'"]snapshot[\'"][^>]*>(.*?)</script>', raw, re.S)
        if not match:
            raise ValueError("view_snapshot_not_found")
        value = json.loads(match.group(1))
    else:
        value = json.loads(raw)
    return value, checksum(raw)


def disposition(row):
    if row is None:
        return "未取得或未评阅"
    e = row.get("judgment", {}).get("value")
    if e is None:
        return "已有结果，方向未评定"
    if e == 0:
        return "已判读，中性"
    if row.get("support", 0) or row.get("refutation", 0):
        return "支持" if e > 0 else "反驳"
    return "已判读，当前权重无贡献"


def normalize(report, *, input_checksum):
    version = report.get("schema_version", report.get("schema", ""))
    old = version.startswith("evidence-graph-html-snapshot/")
    if not old and version != "dynamic-case-report/1.0.0":
        raise ValueError("view_unsupported_report_version")
    scores = report.get("current_scoring" if old else "scores") or {}
    reviews = scores.get("reviews", [])
    if len({r["id"] for r in reviews}) != len(reviews):
        raise ValueError("view_duplicate_assessment")
    detail = report.get("original_review") or {}
    if detail and detail.get("scores") != scores:
        raise ValueError("view_score_binding")
    requirements = report.get("requirements", [])
    if len({r["id"] for r in requirements}) != len(requirements):
        raise ValueError("view_duplicate_requirement")
    by_requirement = {r["id"]: r for r in requirements}
    by_review = {r["id"]: r for r in reviews}
    records = detail.get("records", {})
    projection = records.get("review_tasks", {})
    tasks = {r["assessment_id"]: r for r in projection.get("tasks", [])}
    observations = {
        r["reference"]["code"]: r for r in detail.get("material", {}).get("observations", [])
    }
    sources = {r["reference"]["code"]: r for r in detail.get("material", {}).get("sources", [])}
    processes = {r["id"]: r for r in report.get("process_evaluations", [])}
    historical_facts = {
        r["assessment_id"]: r for r in report.get("evidence_records", {}).get("observations", [])
    }
    nodes, edges = [], []

    def add(identifier, kind, title, payload, parent=None, state="记录"):
        nodes.append(dict(id=identifier, type=kind, title=title, payload=payload, state=state))
        if parent:
            edges.append(dict(source=parent, target=identifier, relation="contains"))

    add("root", "goal", "病例目标与综合结果", scores.get("root", {}))
    ids = list(dict.fromkeys([*by_requirement, *by_review]))
    for identifier in ids:
        req, row = by_requirement.get(identifier, {}), by_review.get(identifier)
        task = tasks.get(identifier, {})
        title = req.get("examination") or (row or {}).get("title") or identifier
        # Store the exact assertion once. All subordinate nodes refer to this check.
        key = "check:" + identifier
        definition = task.get("definition", {})
        refs = definition.get("source_refs", []) or req.get("sources", [])
        add(
            key,
            "requirement",
            title,
            dict(
                id=identifier,
                modalities=req.get("modality", []),
                definition=definition,
                sources=[sources.get(r["code"], r) for r in refs],
                historical_only=old,
            ),
            "root",
            req.get("status", disposition(row)),
        )
        obs = req.get("observation") or task.get("observation") or historical_facts.get(identifier)
        if isinstance(obs, dict) and "code" in obs:
            obs = observations.get(obs["code"], obs)
        if obs is not None or row is not None:
            add(
                key + ":fact",
                "observation",
                "4 · 本例工具与观察结果",
                obs
                or {
                    "observation_available": row.get("observation_available"),
                    "note": "历史报告未在此导出完整观察；不从分数反推测量。",
                },
                key,
                disposition(row),
            )
        if row is not None:
            add(
                key + ":review",
                "interpretation",
                "5 · 目前如何判读与贡献",
                row,
                key,
                disposition(row),
            )
            for path in row.get("dependency_paths", []):
                for dependency in path:
                    edges.append(
                        dict(source=dependency, target=key + ":review", relation="depends_on")
                    )
        if identifier in processes or (row and row.get("role") in {"risk", "context"}):
            add(
                key + ":quality",
                "context",
                "6 · 质量与判读背景",
                processes.get(
                    identifier,
                    {
                        "role": (row or {}).get("role"),
                        "counts_again": False,
                    },
                ),
                key,
            )
        if row and row.get("dependency_paths"):
            add(
                key + ":dependencies",
                "execution",
                "6 · 计算依赖",
                {
                    "dependency_paths": row["dependency_paths"],
                    "correlation_group": row.get("correlation_group"),
                    "dependence_block": row.get("dependence_block"),
                },
                key,
            )
    # Old structural nodes are retained as context, not remapped into new score slots.
    if old:
        add(
            "historical-structure",
            "execution",
            "原树结构（原字节事实的展示）",
            report.get("nodes", []),
            "root",
        )
    add(
        "execution",
        "execution",
        "执行、工具与运行状态",
        {
            "result": report.get("result"),
            "completion": report.get("completion"),
            "native_tools": report.get("native_tools"),
        },
        "root",
    )
    add(
        "quality",
        "context",
        "输入、质量与需求去向",
        {
            "input_context": report.get("input_context"),
            "segmentation_qc": report.get("segmentation_qc"),
            "inventory": report.get("requirement_inventory"),
            "missing_is_negative": False,
        },
        "root",
    )
    summary = scores.get("root", {}).get("summary", {})
    confidence = summary.get("confidence", {}).get("value")
    return dict(
        schema_version="evidence-view/1.0.0",
        source_checksum=input_checksum,
        source_version=version,
        scoring_version=scores.get("schema_version"),
        scope="read_only_historical_presentation_not_current_authorization",
        sample=report.get("result", {}).get("sample", "历史病例"),
        counts=dict(
            requirements=len(requirements) if not old else None,
            reviewed=len(reviews),
            numeric=sum(r.get("judgment", {}).get("value") is not None for r in reviews),
            nonzero_contributors=sum(
                bool(r.get("support") or r.get("refutation")) for r in reviews
            ),
            neutral=sum(r.get("judgment", {}).get("value") == 0 for r in reviews),
            unassessed=sum(r.get("judgment", {}).get("value") is None for r in reviews),
            view_nodes=len(nodes),
        ),
        confidence=confidence,
        nodes=nodes,
        edges=edges,
        scores=scores,
        removals=scores.get("removals", []),
    )


def esc(value):
    return html.escape(str(value), quote=True)


def raw_details(value, label="精确字段与溯源"):
    return (
        "<details><summary>"
        + esc(label)
        + "</summary><pre>"
        + esc(json.dumps(value, ensure_ascii=False, indent=2))
        + "</pre></details>"
    )


def render(view):
    by_parent = {}
    for edge in view["edges"]:
        if edge["relation"] == "contains":
            by_parent.setdefault(edge["source"], []).append(edge["target"])
    nodes = {r["id"]: r for r in view["nodes"]}

    def node(identifier):
        row = nodes[identifier]
        label, color = TYPES[row["type"]]
        body = ""
        payload = row["payload"]
        if row["type"] == "interpretation":
            values = [("连续判读 e", payload.get("judgment", {}).get("value"))]
            values += [
                (label, payload.get(key, {}).get("value"))
                for key, label in (
                    ("observation", "可靠性 O"),
                    ("relation", "关系 R"),
                    ("applicability", "适用性 A"),
                )
            ]
            values += [("支持 S", payload.get("support")), ("反驳", payload.get("refutation"))]
            body += (
                "<div class='metrics'>"
                + "".join(
                    f"<span>{esc(k)} <b>{esc('未评定' if v is None else round(v, 5))}</b></span>"
                    for k, v in values
                )
                + "</div><p>"
                + esc(payload.get("judgment", {}).get("reason", ""))
                + "</p>"
            )
        if row["type"] == "requirement":
            body += "<h4>1 · 来源事实</h4>"
            for source in payload["sources"]:
                body += raw_details(source, source.get("locator") or source.get("code", "来源记录"))
            definition = payload["definition"]
            body += (
                "<h4>2 · 提炼理由与疾病关系</h4><p>"
                + esc(
                    definition.get("interpretation")
                    or "沿用原记录；本导出未携带独立判读定义，不在转换时补写。"
                )
                + "</p><h4>3 · 本节点的取证需求</h4>"
            )
            body += "<p>输入模态：" + esc("、".join(payload["modalities"]) or "见原记录") + "</p>"
        if row["type"] == "observation" and isinstance(payload, dict):
            if payload.get("summary"):
                body += "<p>" + esc(payload["summary"]) + "</p>"
            value = payload.get("value")
            if isinstance(value, str):
                try:
                    value = json.loads(value)
                except (ValueError, TypeError):
                    pass
            if isinstance(value, dict):
                visible = {
                    k: v
                    for k, v in value.items()
                    if k
                    in {
                        "value",
                        "age",
                        "signal",
                        "finding",
                        "coverage",
                        "clarity",
                        "visible_regions",
                        "unseen_regions",
                        "unit",
                    }
                }
                body += raw_details(visible, "本例观测摘要")
            elif value is not None:
                body += "<p>" + esc(value) + "</p>"
        if row["type"] == "context" and isinstance(payload, dict) and payload.get("inventory"):
            body += "<table><tr><th>候选</th><th>处理</th><th>去向/后续</th></tr>"
            labels = {
                "scheduled": "已纳入检查",
                "merged": "合并同类问题",
                "source_binding_pending": "来源待核验绑定",
            }
            for entry in payload["inventory"]["rows"]:
                body += (
                    "<tr>"
                    + "".join(
                        "<td>" + esc(x) + "</td>"
                        for x in (
                            entry["title"],
                            labels.get(entry["disposition"], entry["disposition"]),
                            entry["target"] or entry["followup"],
                        )
                    )
                    + "</tr>"
                )
            body += "</table>"
        body += raw_details(payload)
        children = "".join(node(v) for v in by_parent.get(identifier, []))
        return (
            f"<details class='node' style='--type:{color}' data-kind='{esc(row['type'])}'"
            + (" open" if row["type"] == "requirement" else "")
            + f"><summary><small>{label}</small> {esc(row['title'])}"
            + f" <em>{esc(row['state'])}</em></summary>{body}{children}</details>"
        )

    c = view["counts"]
    legend = " ".join(
        f"<span style='border-color:{color}'>{esc(label)}</span>" for label, color in TYPES.values()
    )
    score = "未评定" if view["confidence"] is None else f"{view['confidence'] * 100:.4f} / 100"
    summary = view["scores"].get("root", {}).get("summary", {})
    combined = summary.get("combined", {})
    totals = " · ".join(
        label + " " + (f"{combined[key] * 100:.4f}" if combined.get(key) is not None else "未评定")
        for key, label in (("support", "支持度"), ("refutation", "反驳度"), ("conflict", "冲突度"))
    )
    return (
        "<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<title>病例证据图 · 统一视图</title><style>"
        "body{font:15px/1.65 system-ui,sans-serif;background:#f4f6fa;color:#182233;margin:0}"
        "main{max-width:1250px;margin:auto;padding:28px}h1{margin:0}h2{margin-top:32px}"
        "header,.node{background:white;padding:16px;border-radius:9px;margin:10px 0}"
        ".node{border-left:4px solid var(--type);padding:12px 16px}.node .node{margin-left:12px}"
        "summary{cursor:pointer}small{color:var(--type);font-weight:700}em{font-style:normal;color:#64748b;font-size:13px}"
        "pre{white-space:pre-wrap;overflow-wrap:anywhere;max-height:450px;overflow:auto;background:#f5f7fb;padding:12px}"
        ".legend span{display:inline-block;border-left:4px solid;padding:0 12px;margin:4px}"
        ".metrics{display:flex;gap:18px;flex-wrap:wrap;margin:10px 0}.muted{color:#64748b}"
        "</style></head><body><main><header><h1>" + esc(view["sample"]) + " · 病例证据图</h1>"
        "<h2>1 · 病例结论概览</h2><p>综合研究分 <strong>" + score + "</strong></p>"
        "<p>" + esc(totals) + "（各项 / 100，不是证据数量）</p>"
        f"<p>评阅 {c['reviewed']} 项 · 数值判读 {c['numeric']} 项"
        f" · 非零贡献 {c['nonzero_contributors']} 项"
        f" · 中性 {c['neutral']} 项 · 方向未评定 {c['unassessed']} 项</p>"
        "<p class='muted'>这是原评分的只读展示，不是患病概率；不以 50 分作为确诊阈值。"
        "节点数不是独立证据数。缺项不作阴性，背景、工具与同源依赖不重复计票。</p>"
        "<div class='legend'>" + legend + "</div></header>"
        "<h2>2 · 检查需求 → 本例结果 → 判读与贡献</h2>"
        + "".join(node(k) for k in by_parent.get("root", []) if k.startswith("check:"))
        + "<h2>3 · 输入、质量与执行依赖</h2>"
        + "".join(node(k) for k in by_parent.get("root", []) if not k.startswith("check:"))
        + "<h2>4 · 证据移除与敏感性</h2>"
        + raw_details(view["removals"], "查看原反事实结果（不改变正常推理）")
        + "<h2>5 · 来源版本与完整性</h2>"
        + raw_details(
            {
                k: view[k]
                for k in ("source_checksum", "source_version", "scoring_version", "scope", "counts")
            }
        )
        + "</main></body></html>"
    )


def convert(source, destination, *, layout="cards-v1", run_key=None, media_path=None):
    if layout == "tree-v3":
        from .evidence_tree_v3 import convert_saved

        return convert_saved(source, destination, run_key=run_key, media_path=media_path)
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve():
        raise ValueError("view_must_not_overwrite_source")
    report, checksum_value = load_report(source)
    view = normalize(report, input_checksum=checksum_value)
    if (run_key or media_path) and layout != "tree-v2":
        raise ValueError("view_media_requires_tree_v2")
    if run_key and media_path:
        raise ValueError("view_choose_one_media_source")
    if layout == "cards-v1":
        rendered = render(view)
    elif layout == "tree-v1":
        from .evidence_tree import render as render_tree

        rendered = render_tree(report, input_checksum=checksum_value)
    elif layout == "tree-v2":
        from .evidence_media import collect
        from .evidence_tree_v2 import render as render_tree

        media = None
        if run_key:
            media = collect(report, source_checksum=checksum_value, run_key=run_key)
        elif media_path:
            if Path(media_path).stat().st_size > 32 * 1024 * 1024:
                raise ValueError("view_media_package_size")
            media = json.loads(Path(media_path).read_bytes())
        rendered = render_tree(report, input_checksum=checksum_value, media=media)
    else:
        raise ValueError("view_unsupported_layout")
    package = None
    if layout == "tree-v2" and run_key:
        package = destination.with_suffix(".media.json")
        value = json.dumps(media, ensure_ascii=False, indent=2) + "\n"
        if package.exists() and package.read_text() != value:
            raise ValueError("view_media_output_exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.read_text() != rendered:
            raise ValueError("view_output_exists_use_new_path")
    else:
        with destination.open("x") as stream:
            stream.write(rendered)
    if package is not None:
        if not package.exists():
            with package.open("x") as stream:
                stream.write(value)
    return dict(
        html=str(destination),
        source_checksum=checksum_value,
        counts=view["counts"],
        new_requests=0,
        rescored=False,
        layout=layout,
        **({"image_count": len((media or {}).get("images", {}))} if layout == "tree-v2" else {}),
    )
