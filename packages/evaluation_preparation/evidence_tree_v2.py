"""Clinical-path navigation and bound image evidence; no rescoring or inference."""

import copy
import html
import json

from .evidence_tree import TEMPLATES
from .evidence_tree import project as project_v1

VERSION = "evidence-tree-reader/2.0.0"
PROFILE = TEMPLATES.parents[1] / "configs/presentation/glaucoma_tree_v1.json"


def project(report, *, input_checksum, media=None, profile=None):
    view = project_v1(report, input_checksum=input_checksum)
    if profile is None:
        candidate = json.loads(PROFILE.read_bytes())
        targets = {r.get("target") for r in view["scores"].get("reviews", [])}
        profile = candidate if targets == {candidate["target"]} else {}
    if profile and profile.get("schema_version") != "evidence-navigation-profile/1.0.0":
        raise ValueError("navigation_profile_version")
    reader = view["reader"]
    reader.update(version=VERSION, navigation_only=True, profile=profile)
    titles = {r["id"]: r["title"] for r in report.get("requirement_inventory", {}).get("rows", [])}
    nodes = [
        {
            "id": "root",
            "parent": None,
            "title": profile.get("title", "病例评估"),
            "type": "goal",
            "state": "组织",
            "payload": {},
        }
    ]
    added = {"root"}

    def group(key, seen=()):
        if key is None:
            return "root"
        if key in seen or key not in profile.get("groups", {}):
            raise ValueError("navigation_group_parent")
        node_id = "nav:" + key
        if node_id not in added:
            parent, title = profile["groups"][key]
            parent_id = group(parent, (*seen, key))
            nodes.append(
                dict(
                    id=node_id, parent=parent_id, title=title, type="goal", state="分支", payload={}
                )
            )
            added.add(node_id)
        return node_id

    roles = {}
    for original in view["nodes"]:
        if original["type"] != "requirement":
            continue
        item = copy.deepcopy(original)
        identifier = item["payload"]["id"]
        mapping = profile.get("checks", {}).get(identifier)
        item["parent"] = group(mapping[0]) if mapping else "root"
        item["title"] = mapping[1] if mapping else titles.get(identifier, item["title"])
        roles[item["id"]] = mapping[2] if mapping else []
        nodes.append(item)
    # Order the tree by its declared clinical hierarchy, not by alphabetic IDs.
    order = {k: i for i, k in enumerate(profile.get("groups", {}))}
    checks = {k: i for i, k in enumerate(profile.get("checks", {}))}
    nodes[1:] = sorted(
        nodes[1:],
        key=lambda n: (
            0 if n["id"].startswith("nav:") else 1,
            order.get(n["id"][4:], checks.get(n.get("payload", {}).get("id"), 1000)),
        ),
    )
    reader["navigation"] = nodes
    reader["media_roles"] = roles
    reader["media"] = media or dict(images={}, missing=["未装配本轮图片包；未借用其他病例图片。"])
    if media and (
        media.get("schema_version") != "evidence-media/1.0.0"
        or media.get("source_checksum") != input_checksum
    ):
        raise ValueError("evidence_media_report_binding")
    if media:
        from .evidence_media import validate_images

        case = report.get("result", {}).get("case_code")
        expected_input = (report.get("native_tools") or {}).get("image_checksum")
        if (case and media.get("case_code") != case) or (
            expected_input and media.get("input_checksum") != expected_input
        ):
            raise ValueError("evidence_media_case_input_binding")
        validate_images(media)
    return view


def render(report, *, input_checksum, media=None):
    view = project(report, input_checksum=input_checksum, media=media)
    template = (TEMPLATES / "readable_evidence_graph.html").read_text()
    shell = template.split('<script id="snapshot" type="application/json">')[0]
    shell = shell.replace("__TITLE__", html.escape(view["sample"] + " · 病例证据图", quote=True))
    raw = json.dumps(view, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    for char in ("&", "<", ">", "\u2028", "\u2029"):
        raw = raw.replace(char, f"\\u{ord(char):04x}")
    base = (TEMPLATES / "evidence_tree_reader.js").read_text()
    if not base.endswith("show('root');\n"):
        raise ValueError("evidence_reader_base_contract")
    base = base.removesuffix("show('root');\n")
    controller = (TEMPLATES / "evidence_tree_reader_v2.js").read_text()
    style = (TEMPLATES / "evidence_tree_reader_v2.css").read_text()
    shell = shell.replace("</head>", "<style>" + style + "</style></head>")
    return (
        shell
        + "<noscript>请启用JavaScript查看证据树与本例图片。</noscript>"
        + '<script id="evidence-view" type="application/json">'
        + raw
        + "</script><script>\n"
        + base
        + controller
        + "\n</script></body></html>\n"
    )
