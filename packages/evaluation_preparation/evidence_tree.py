"""Original reader shell over a generic, read-only presentation projection.

No clinical snapshot is forged to satisfy the historical renderer. The shell/CSS
is shared, while the data adapter and controller have their own explicit version.
"""

import html
import json
from pathlib import Path

from .evidence_view import TYPES, normalize

VERSION = "evidence-tree-reader/1.0.0"
TEMPLATES = Path(__file__).resolve().parents[2] / "apps/templates"


def project(report, *, input_checksum):
    view = normalize(report, input_checksum=input_checksum)
    detail = report.get("original_review") or {}
    records = detail.get("records", {})
    tasks = {t["assessment_id"]: t for t in records.get("review_tasks", {}).get("tasks", [])}
    interpretations = {
        r["review"]["assessment_id"]: r["review"]
        for r in records.get("interpretations", [])
        if r.get("review", {}).get("assessment_id")
    }
    requirements = {r["id"]: r for r in report.get("requirements", [])}
    # A missing export field stays missing; do not load today's rules or infer a
    # historical threshold from a score, label, or a different patient's report.
    view["reader"] = dict(
        version=VERSION,
        types=TYPES,
        requirements=requirements,
        tasks=tasks,
        interpretations=interpretations,
        completion=report.get("completion"),
        source_checks="export_snapshot_only_not_current_authorization",
    )
    return view


def render(report, *, input_checksum):
    view = project(report, input_checksum=input_checksum)
    template = (TEMPLATES / "readable_evidence_graph.html").read_text()
    marker = '<script id="snapshot" type="application/json">'
    if template.count(marker) != 1:
        raise ValueError("evidence_tree_shell_contract")
    # Reuse the actual old navigation/detail shell and its responsive styles,
    # not a separately maintained approximate lookalike.
    shell = template.split(marker)[0].replace(
        "__TITLE__", html.escape(view["sample"] + " · 病例证据图", quote=True)
    )
    raw = json.dumps(view, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    for char in ("&", "<", ">", "\u2028", "\u2029"):
        raw = raw.replace(char, f"\\u{ord(char):04x}")
    script = (TEMPLATES / "evidence_tree_reader.js").read_text()
    return (
        shell
        + "<noscript>请启用 JavaScript 查看树导航；本页不会发送请求或重算分数。</noscript>"
        + '<script id="evidence-view" type="application/json">'
        + raw
        + "</script><script>\n"
        + script
        + "\n</script></body></html>\n"
    )
