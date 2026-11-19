"""Structure-driven offline reader; shared original shell, no disease-specific profile."""

import html
import json
from pathlib import Path

from packages.grounding.models import digest

from .evidence_assets import attach
from .evidence_presentation import project, validate_packet
from .evidence_tree import TEMPLATES
from .evidence_tree_static import points, render_static

VERSION = "evidence-tree-reader/3.1.0"
BUNDLE = "evidence-display-export/1.0.0"


def build(report, *, input_checksum, media=None):
    return attach(project(report, source_checksum=input_checksum), report, media)


def bundle(packet):
    validate_packet(packet)
    return dict(schema_version=BUNDLE, packet_checksum=digest(packet), packet=packet)


def unbundle(value):
    if value.get("schema_version") != BUNDLE or digest(value["packet"]) != value.get(
        "packet_checksum"
    ):
        raise ValueError("presentation_bundle_integrity")
    packet = validate_packet(value["packet"])
    from .evidence_assets import validate_asset

    media = packet.get("assets", {})
    items = media.get("items", [])
    ids = {item["id"] for item in items}
    nodes = {node["id"] for node in packet["nodes"]}
    if media.get("schema_version") != "evidence-assets/1.0.0" or len(ids) != len(items):
        raise ValueError("presentation_bundle_asset_identity")
    if any(
        link["asset_id"] not in ids or link["node_id"] not in nodes
        for link in media.get("links", [])
    ):
        raise ValueError("presentation_bundle_asset_endpoint")
    for item in items:
        if item.get("display_status") == "ready" and validate_asset(item):
            raise ValueError("presentation_bundle_asset")
    return packet


def render_packet(packet):
    validate_packet(packet)
    template = (TEMPLATES / "readable_evidence_graph.html").read_text()
    marker = '<script id="snapshot" type="application/json">'
    if template.count(marker) != 1:
        raise ValueError("presentation_shell_contract")
    shell = template.split(marker)[0]
    shell = shell.replace("__TITLE__", html.escape(packet["sample"] + " · 病例证据图", quote=True))
    shell = shell.replace("<body>", '<body data-saved-fallback="true">')
    scope = packet.get("appendices", {}).get("review_scope", {})
    if scope.get("kind") in {
        "definition_increment_only",
        "definition_increment_and_confirmed_component",
    }:
        shell = shell.replace(
            '<body data-saved-fallback="true">',
            '<body data-saved-fallback="true"><aside class="notice">'
            + html.escape(scope["explanation"])
            + "</aside>",
            1,
        )
    shell = shell.replace(
        '<div id="summary"></div>',
        '<div id="summary">综合研究分 ' + points(packet["confidence"]) + "</div>",
    )
    shell = shell.replace(
        '<main><nav id="tree"', '<main id="interactive-reader" hidden><nav id="tree"'
    )
    shell = shell.replace(
        "<footer>",
        '<button id="reader-mode" type="button" hidden>切换静态全文</button>'
        + render_static(packet)
        + "<footer>",
    )
    style = (TEMPLATES / "evidence_tree_reader_v2.css").read_text()
    style += (TEMPLATES / "evidence_tree_reader_v3.css").read_text()
    shell = shell.replace("</head>", "<style>" + style + "</style></head>")
    raw = json.dumps(packet, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    for char in ("&", "<", ">", "\u2028", "\u2029"):
        raw = raw.replace(char, f"\\u{ord(char):04x}")
    script = (TEMPLATES / "evidence_tree_reader_v3.js").read_text()
    return (
        shell
        + '<script id="evidence-view" type="application/json">'
        + raw
        + "</script><script>\n"
        + script
        + "\n</script></body></html>\n"
    )


def render(report, *, input_checksum, media=None):
    return render_packet(build(report, input_checksum=input_checksum, media=media))


def reproject_saved_media(report, exported, *, source_file_checksum):
    """Fresh presentation, exact same report and validated saved media; no owner writes."""
    previous = unbundle(exported)
    if previous["report_content_checksum"] != digest(report):
        raise ValueError("presentation_media_source_binding")
    packet = project(report, source_checksum=previous["source_checksum"])
    if (
        {n["id"] for n in packet["nodes"]} != {n["id"] for n in previous["nodes"]}
        or packet["scores"] != previous["scores"]
        or any(packet.get(k) != previous.get(k) for k in ("case_code", "input_checksum"))
    ):
        raise ValueError("presentation_media_reprojection_identity")
    packet["assets"] = previous["assets"]
    packet["display_reprojection"] = dict(
        schema_version="same-report-display-reprojection/1.0.0",
        previous_packet_checksum=exported["packet_checksum"],
        source_file_checksum=source_file_checksum,
        report_content_checksum=digest(report),
        medical_recomputed=False,
        current_authorization=False,
    )
    return unbundle(bundle(packet))


def convert_saved(source, destination, *, run_key=None, media_path=None):
    from .evidence_view import load_report

    source, destination = Path(source), Path(destination)
    exported = destination.with_suffix(".view.json")
    if source.resolve() in {destination.resolve(), exported.resolve()}:
        raise ValueError("view_must_not_overwrite_source")
    if run_key and media_path:
        raise ValueError("view_choose_one_media_source")
    report, checksum_value = load_report(source)
    if report.get("schema_version") == BUNDLE:
        if run_key or media_path:
            raise ValueError("presentation_bundle_already_has_media")
        packet = unbundle(report)
    else:
        media = None
        if media_path:
            path = Path(media_path)
            if path.stat().st_size > 32 * 1024 * 1024:
                raise ValueError("view_media_package_size")
            media = json.loads(path.read_bytes())
        elif run_key:
            # Existing owner-specific collector remains an explicit legacy adapter.
            from .evidence_media import collect

            media = collect(report, source_checksum=checksum_value, run_key=run_key)
        packet = (
            reproject_saved_media(report, media, source_file_checksum=checksum_value)
            if media and media.get("schema_version") == BUNDLE
            else build(report, input_checksum=checksum_value, media=media)
        )
    page = render_packet(packet)
    saved = json.dumps(bundle(packet), ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    if media_path and Path(media_path).resolve() in {destination.resolve(), exported.resolve()}:
        raise ValueError("view_must_not_overwrite_media")
    outputs = {destination: page, exported: saved}
    # Check every destination before writing either output, including same-content retries.
    for path, value in outputs.items():
        if path.exists() and path.read_text() != value:
            raise ValueError("view_output_exists_use_new_path")
    for path, value in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            with path.open("x") as stream:
                stream.write(value)
    return dict(
        html=str(destination),
        presentation=str(exported),
        layout="tree-v3",
        source_checksum=packet["source_checksum"],
        counts=packet["counts"],
        image_count=sum(
            a["kind"] == "image" and a["display_status"] == "ready"
            for a in packet["assets"]["items"]
        ),
        media_count=len(packet["assets"]["items"]),
        diagnostics=packet["diagnostics"],
        coverage_complete=packet["coverage"]["complete"],
        new_requests=0,
        rescored=False,
    )
