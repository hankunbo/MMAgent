"""Script-independent saved-result reader, shared by every tree-v3 producer."""

import html
import json

from .evidence_assets import validate_asset

SECTIONS = (
    "来源事实",
    "我们的提炼理由",
    "本节点的取证需求",
    "本例工具与观察结果",
    "目前如何判读",
    "计算依赖与判读背景",
)


def esc(value):
    if value is None:
        value = "未单列／未知"
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False, indent=2)
    return html.escape(str(value), quote=True)


def number(value):
    return str(round(value, 6)) if type(value) in (int, float) else "未评定"


def points(value):
    return f"{value * 100:.4f} / 100" if type(value) in (int, float) else "未评定"


def line(name, value):
    return f'<div class="reading-line"><strong>{esc(name)}</strong><span>{esc(value)}</span></div>'


def raw(name, value):
    return f'<details class="fold"><summary>{esc(name)}</summary><pre>{esc(value)}</pre></details>'


def table(headers, rows, attribute=""):
    head = "".join(f"<th>{esc(v)}</th>" for v in headers)
    body = "".join("<tr>" + "".join(f"<td>{esc(v)}</td>" for v in row) + "</tr>" for row in rows)
    return (
        f'<div class="table-wrap"><table {attribute}><thead><tr>{head}</tr></thead>'
        f"<tbody>{body}</tbody></table></div>"
    )


def badge(packet, node):
    # Only the closed palette contributes CSS; unknown roles remain readable.
    kind = node.get("type", "")
    color = {
        "goal": "#64748b",
        "requirement": "#2563eb",
        "observation": "#0891b2",
        "interpretation": "#7c3aed",
        "context": "#d97706",
        "dependency": "#475569",
    }
    label = packet["types"].get(kind, ["扩展类型"])[0]
    return (
        f'<span class="badge" style="border-left:4px solid {color.get(kind, "#64748b")}">'
        f"{esc(label)}</span>"
    )


def media_html(asset):
    label = esc(asset.get("title") or asset["id"])
    body = f"<figcaption>{label}</figcaption>"
    try:
        status = asset.get("display_status")
        if status == "shared_bytes_see_linked_asset":
            return (
                body
                + line("同字节视图展示位置", asset.get("shared_display_asset_id"))
                + line("说明", asset.get("display_note"))
                + raw("原独立引用（非已取得观察）", asset.get("reference"))
            )
        if status != "ready":
            return body + line("产物未内联展示", status)
        status = validate_asset(asset)
        if status:
            return body + line("产物未内联展示", status)
    except (ValueError, KeyError, TypeError):
        return body + line("产物未内联展示", "媒体校验未通过；保留原引用")
    kind = asset.get("kind")
    if kind == "image":
        body += f'<img src="{esc(asset["data_uri"])}" alt="{label}" loading="lazy">'
    elif kind == "image_sequence":
        # Native details work without JS; every saved frame remains accessible.
        for i, frame in enumerate(asset["frames"]):
            body += (
                f"<details {'open' if i == 0 else ''}><summary>保存切片 {i + 1}</summary>"
                f'<img src="{esc(frame["data_uri"])}" alt="{label} {i + 1}" loading="lazy">'
                "</details>"
            )
    elif kind == "table":
        body += table(asset["columns"], asset["rows"])
    elif kind == "series":
        body += line("坐标及单位", asset.get("axes"))
        body += table(["x", "y"], asset["points"])
    else:
        body += line("产物类型", "未支持；没有伪造预览")
    body += line(
        "模态／眼别／时点", [asset.get(k) for k in ("modality", "laterality", "timepoint")]
    )
    if asset.get("note"):
        body += line("说明", asset["note"])
    body += raw(
        "产物身份与派生来源",
        {k: v for k, v in asset.items() if k not in {"data_uri", "frames", "points", "rows"}},
    )
    return body


def render_static(packet):
    nodes = {n["id"]: n for n in packet["nodes"]}
    anchors = {key: f"saved-node-{i}" for i, key in enumerate(nodes)}
    assets = {a["id"]: a for a in packet["assets"]["items"]}
    media_anchors = {key: f"saved-asset-{i}" for i, key in enumerate(assets)}
    used_media = set()

    def link(key, title=None):
        return (
            f'<a href="#{anchors[key]}" data-node-id="{esc(key)}">'
            f"{esc(title or nodes[key]['title'])}</a>"
        )

    def gallery(node_id=None):
        selected = (
            list(assets)
            if node_id is None
            else list(
                dict.fromkeys(
                    x["asset_id"] for x in packet["assets"]["links"] if x["node_id"] == node_id
                )
            )
        )
        parts = []
        for key in selected:
            if key in used_media:
                if node_id is not None:
                    parts.append(
                        f'<p><a href="#{media_anchors[key]}">查看共享产物：'
                        f"{esc(assets[key].get('title') or key)}</a></p>"
                    )
                continue
            used_media.add(key)
            parts.append(f'<figure id="{media_anchors[key]}">{media_html(assets[key])}</figure>')
        return '<div class="pictures">' + "".join(parts) + "</div>" if parts else ""

    def relations(node_id):
        rows = []
        for edge in packet["edges"]:
            if node_id not in (edge["source"], edge["target"]):
                continue
            other = edge["target"] if edge["source"] == node_id else edge["source"]
            rows.append(f"<li>{esc(edge['relation'])} → {link(other)}</li>")
        return "<ul>" + "".join(rows) + "</ul>"

    children = {}
    for node in packet["navigation"]:
        children.setdefault(node.get("parent"), []).append(node["id"])
    tree, stack, seen = [], [("node", "root", 0)], set()
    while stack:
        mode, key, depth = stack.pop()
        if mode == "close":
            tree.append(key)
            continue
        if key in seen:
            continue
        seen.add(key)
        node = nodes[key]
        tree.append(
            f'<li>{link(key)}{badge(packet, node)}<span class="badge">'
            f"{esc(node.get('state', ''))}</span>"
        )
        kids = children.get(key, [])
        # Keep deep trees readable without unbounded nested HTML.
        if kids and depth < 60:
            tree.append("<ul>")
            stack.append(("close", "</ul></li>", depth))
            stack.extend(("node", child, depth + 1) for child in reversed(kids))
        else:
            tree.append("</li>")
            stack.extend(("node", child, depth) for child in reversed(kids))
    nav = (
        '<nav id="saved-tree" aria-label="静态证据树"><h2>证据树</h2><ul>'
        + "".join(tree)
        + "</ul></nav>"
    )
    root = nodes["root"]
    summary = packet["scores"].get("root", {}).get("summary", {})
    body = [
        f'<section id="{anchors["root"]}"><h2>病例总览</h2>',
        line("病例输入", packet["sample"]),
        line("保存目标", root.get("goal_summary") or root["title"]),
        line("综合研究分", points(packet["confidence"])),
        *(
            [raw("仅已确认观察收尾（失败请求未重发）", root["observation_closure"])]
            if root.get("observation_closure")
            else []
        ),
        '<p class="notice">研究分不是患病概率。只读取已保存结果，不请求模型、不重新评分。</p>',
        table(
            ["需求", "评阅", "数值判读", "非零贡献", "中性", "未评定"],
            [
                [
                    packet["counts"].get(k)
                    for k in (
                        "requirements",
                        "reviewed",
                        "numeric",
                        "nonzero_contributors",
                        "neutral",
                        "unassessed",
                    )
                ]
            ],
        ),
        raw("原保存根结论与覆盖", packet["scores"].get("root")),
        "</section>",
    ]
    if summary.get("hypotheses"):
        body.append(
            table(
                ["命题", "研究分"],
                [
                    [h["hypothesis"]["title"], points(h["summary"]["confidence"]["value"])]
                    for h in summary["hypotheses"]
                ],
            )
        )
    # Render every node, not just visible navigation: relation endpoints stay reachable.
    for node in packet["nodes"]:
        key = node["id"]
        if key == "root":
            continue
        check = packet["checks"].get(node.get("check_id"))
        attr = f' data-check-id="{esc(node["check_id"])}"' if check else ""
        body.append(
            f'<details class="reading-card" id="{anchors[key]}"{attr} open><summary>'
            f"{esc(node['title'])}{badge(packet, node)}</summary>"
        )
        if check:
            s, status = check["sections"], check["status"]
            content = [
                "".join(
                    line(
                        "来源", src.get("text") or src.get("locator") or src.get("reference") or src
                    )
                    + raw("精确来源", src)
                    for src in s["sources"]
                )
                or line("来源", "本导出未携带来源正文"),
                line("为何开展本检查", s["rationale"]),
                line(
                    "取证要求",
                    s["acquisition"]
                    if s["acquisition"] is not None
                    else s["legacy_definition"].get("examination"),
                )
                + line("输入模态", s["modalities"])
                + line("本轮方法", s["method"])
                + raw("任务绑定", check.get("task")),
                line("取得状态", status.get("acquisition") or status.get("requirement"))
                + line(
                    "本例观察",
                    s["observation"]
                    if s["observation"] is not None
                    else "未取得或导出未携带；不是医学阴性",
                )
                + gallery(key),
                line(
                    "检查参照",
                    s["reference"]
                    if s["reference"] is not None
                    else s["legacy_definition"].get("interpretation"),
                )
                + line("本例比较", (s["review"] or {}).get("comparison")),
                line("适用条件", s["legacy_definition"].get("applicability_conditions"))
                + line("过程评价", s["process"])
                + line("计算依赖", s["dependencies"])
                + line("状态", status)
                + relations(key),
            ]
            score = s["score"]
            if score is not None:
                content[4] += line("判读", score.get("judgment", {}).get("reason"))
                content[4] += table(
                    ["可靠性 O", "关系 R", "适用性 A", "连续判读 e", "支持贡献", "反驳贡献"],
                    [
                        [
                            number(score.get(k, {}).get("value"))
                            for k in ("observation", "relation", "applicability", "judgment")
                        ]
                        + [number(score.get("support")), number(score.get("refutation"))]
                    ],
                    "data-score-table",
                )
                content[4] += raw("原评分及理由（不重算）", score)
            else:
                content[4] += line("判读", "没有已保存评分；不填0或医学阴性")
            content[4] += line("未解决项", (s["review"] or {}).get("unresolved"))
            for i, section in enumerate(content):
                body.append(
                    f'<section data-section="{i + 1}"><h3>{i + 1} · {SECTIONS[i]}</h3>'
                    f"{section}</section>"
                )
        else:
            body.append(relations(key) + raw("原节点引用", node) + gallery(key))
        body.append("</details>")
    body.append(
        '<section id="saved-media"><h2>其他保存产物／图片</h2>'
        + (gallery() or "<p>所有已绑定产物已在各检查的第4节列出；本区没有其他图片。</p>")
        + "</section>"
    )
    body.append(raw("完成状态", packet["appendices"].get("completion")))
    body.append(
        raw("原保存移除分析", packet["removals"])
        + raw("Memory保存记录（不是当前授权）", packet["memory"])
    )
    body.append(
        raw(
            "全部附录及覆盖说明",
            {
                "appendices": packet["appendices"],
                "coverage": packet["coverage"],
                "diagnostics": packet["diagnostics"],
            },
        )
    )
    notice = (
        '<p id="static-reader-notice" class="notice">静态阅读模式：无需 JavaScript，'
        "树、六节判读和图片均已写入页面。点击左侧节点可定位，点击小三角可折叠。"
        "若支持脚本，可使用交互阅读；此页不会运行模型。</p>"
    )
    return (
        notice
        + '<main id="saved-reader">'
        + nav
        + '<article id="saved-detail">'
        + "".join(body)
        + "</article></main>"
    )
