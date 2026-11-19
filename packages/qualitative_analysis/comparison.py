"""Explicit read-only historical integration and itemized comparison exports."""

import copy
import html
import json
from pathlib import Path

from packages.object_store import checksum
from packages.tool_execution.process import read_file

from . import runner
from .integration import field_contexts, integrate
from .report import render
from .storage import read, seal, unseal, write_bytes_once, write_once


def verified_result(path):
    path = Path(path).absolute()
    result = read(path)
    unseal(result)
    manifest = unseal(read(path.with_suffix(".manifest.json")))
    if (
        manifest["result_checksum"] != result["checksum"]
        or manifest["json_checksum"] != checksum(read_file(path, 16777216))
        or manifest["html_checksum"] != checksum(read_file(path.with_suffix(".html"), 16777216))
    ):
        raise ValueError("qualitative_result_artifact_binding")
    origin = path.parent.parent
    if (origin / "plan.json").exists():
        runner.verify(origin)
        if result["plan_checksum"] != read(origin / "plan.json")["checksum"]:
            raise ValueError("qualitative_result_plan_binding")
    else:
        derivation = unseal(read(origin / "derivation.json"))
        if derivation["result_checksum"] != result["checksum"]:
            raise ValueError("qualitative_derivation_binding")
        for relative, expected in derivation["views"].items():
            if checksum(read_file(origin / relative, 8388608)) != expected:
                raise ValueError("qualitative_derived_view_binding")
    return result


def integrate_saved(source, output):
    source, output = Path(source).absolute(), Path(output).absolute()
    value = verified_result(source)
    if value["schema_version"] != "qualitative-case-description/1.0.0":
        raise ValueError("qualitative_integration_requires_v1_saved_result")
    output.mkdir(parents=True, exist_ok=False)
    (output / "exports").mkdir()
    (output / "views").mkdir()
    result = copy.deepcopy(unseal(value))
    result["field_contexts"] = field_contexts(result["components"])
    result = integrate(result)
    result["derivation"] = dict(
        operation="read_only_saved_record_integration_no_new_visual_judgment",
        source_result_checksum=value["checksum"],
        new_api_requests=0,
        source_result=str(source),
    )
    artifact = seal(result)
    views = {}
    for row in result["components"]:
        for v in row.get("sampled_views", []):
            name = f"views/{row['execution']['job_id']}-{v['view_id']}.png"
            raw = read_file(source.parent.parent / name, 8388608)
            if checksum(raw) != v["view_checksum"]:
                raise ValueError("qualitative_integration_view_binding")
            write_bytes_once(output / name, raw)
            views[name] = checksum(raw)
    path = output / "exports" / (artifact["checksum"][7:] + ".json")
    write_once(path, artifact)
    write_bytes_once(path.with_suffix(".html"), render(artifact).encode())
    write_once(
        path.with_suffix(".manifest.json"),
        seal(
            dict(
                result_checksum=artifact["checksum"],
                plan_checksum=artifact["plan_checksum"],
                json_checksum=checksum(read_file(path, 16777216)),
                html_checksum=checksum(read_file(path.with_suffix(".html"), 16777216)),
            )
        ),
    )
    write_once(
        output / "derivation.json",
        seal(
            dict(
                source_result_checksum=value["checksum"],
                result_checksum=artifact["checksum"],
                new_api_requests=0,
                views=views,
                software=runner.software(),
            )
        ),
    )
    verified_result(path)
    return dict(json=str(path), html=str(path.with_suffix(".html")), new_api_requests=0)


def compare(before, after, output):
    old, new = verified_result(before), verified_result(after)
    if old["packet_checksum"] != new["packet_checksum"]:
        raise ValueError("qualitative_comparison_requires_same_packet")
    old_rows = {r["component_id"]: r for r in old["components"]}
    rows = []
    for r in new["components"]:
        p = old_rows.get(r["component_id"])
        if (
            p is None
            or p["component_checksum"] != r["component_checksum"]
            or p["source_declared"] != r["source_declared"]
        ):
            raise ValueError("qualitative_comparison_source_changed")
        i = r.get("integrated", {})
        rows.append(
            dict(
                component_id=r["component_id"],
                source_preserved=True,
                source_laterality=r["source_declared"]["laterality"],
                integrated_laterality=i.get("laterality"),
                old_content=p["content_regions"],
                new_content=r["content_regions"],
                old_anatomy=p.get("anatomy", []),
                new_anatomy=i.get("anatomy", []),
                old_unit_state=p["semantics"]["unit_state"],
                new_unit=i.get("units"),
                old_textual_differences=p.get("conflicts", []),
                new_terminology_comparisons=i.get("terminology_comparisons", []),
                new_conflicts=r.get("conflicts", []),
                old_unconfirmed=p.get("unconfirmed", []),
                new_uncertainties=i.get("uncertainties", []),
                old_response_checksum=p.get("response_checksum"),
                new_response_checksum=r.get("response_checksum"),
                new_visual_response=r.get("response_checksum") is not None
                and r.get("response_checksum") != p.get("response_checksum"),
            )
        )
    result = seal(
        dict(
            schema_version="qualitative-comparison/1.0.0",
            before_result=old["checksum"],
            after_result=new["checksum"],
            packet_checksum=new["packet_checksum"],
            components=rows,
            old_background=old["background"],
            new_background=new["background"],
            summary=dict(
                component_count=len(rows),
                source_facts_preserved=all(r["source_preserved"] for r in rows),
                new_visual_component_count=sum(r["new_visual_response"] for r in rows),
                equivalent_term_count=sum(
                    c["status"] == "equivalent"
                    for r in rows
                    for c in r["new_terminology_comparisons"]
                ),
                note="术语整合不算新观察；新模型描述不是独立医学真值；共享图像不算新增采集。",
            ),
        )
    )
    output = Path(output).absolute()
    output.mkdir(parents=True, exist_ok=False)
    write_once(output / "comparison.json", result)

    def show(v):
        return "<pre>" + html.escape(json.dumps(v, ensure_ascii=False, indent=2)) + "</pre>"

    parts = [
        '<!doctype html><html lang="zh"><meta charset="utf-8"><title>患者资料新旧逐项对照</title>'
        "<style>body{font:16px system-ui;max-width:1200px;margin:2em}"
        "pre{white-space:pre-wrap;overflow-wrap:anywhere}"
        "details{border:1px solid #ccd;padding:1em;margin:1em}"
        "</style><h1>患者资料新旧逐项对照</h1>",
        show(result["summary"]),
    ]
    for row in rows:
        parts += [
            "<details><summary>"
            + html.escape(row["component_id"])
            + " · "
            + html.escape(row["source_laterality"])
            + "</summary>"
            + show(row)
            + "</details>"
        ]
    write_bytes_once(output / "comparison.html", "\n".join(parts + ["</html>"]).encode())
    return dict(
        json=str(output / "comparison.json"),
        html=str(output / "comparison.html"),
        summary=result["summary"],
    )


def combine_confirmed(current, previous, output):
    """Read-only material dossier. Historical fallback never completes the new job."""
    current, previous = Path(current).absolute(), Path(previous).absolute()
    new, old = verified_result(current), verified_result(previous)
    if new["packet_checksum"] != old["packet_checksum"] or new["scope"] != old["scope"]:
        raise ValueError("qualitative_combine_scope_or_packet")
    value = copy.deepcopy(unseal(new))
    previous_rows = {r["component_id"]: r for r in old["components"]}
    selected_sources = {}
    fallback = []
    for index, row in enumerate(value["components"]):
        p = previous_rows.get(row["component_id"])
        if (
            p is None
            or p["source_declared"] != row["source_declared"]
            or p["component_checksum"] != row["component_checksum"]
        ):
            raise ValueError("qualitative_combine_source_binding")
        original_state = row["execution"]["visual"]
        source_path = current
        if original_state != "completed" and p["execution"]["visual"] == "completed":
            # This is previous-version data, explicitly not fulfillment of a new visual task.
            if not p.get("response_checksum"):
                raise ValueError("qualitative_combine_requires_confirmed_response")
            replacement = copy.deepcopy(p)
            replacement["execution"]["historical_fallback"] = dict(
                reason="new_visual_request_not_confirmed_or_incomplete",
                current_visual_state=original_state,
                current_result_checksum=new["checksum"],
                previous_result_checksum=old["checksum"],
                previous_response_checksum=p["response_checksum"],
                fulfills_new_visual_contract=False,
            )
            value["components"][index] = replacement
            source_path = previous
            fallback.append(row["component_id"])
        selected_sources[row["component_id"]] = source_path
    value = integrate(value)
    value["coverage"]["historical_fallback_components"] = fallback
    value["coverage"]["current_revision_all_planned_jobs_completed"] = new["coverage"][
        "all_planned_jobs_completed"
    ]
    # Keep current UNKNOWN counts; reference old requests separately.
    value["derivation"] = dict(
        operation="combine_confirmed_data_with_explicit_historical_fallback",
        current_result_checksum=new["checksum"],
        previous_result_checksum=old["checksum"],
        current_result=str(current),
        previous_result=str(previous),
        new_api_requests=0,
        history_fulfills_new_visual_contract=False,
    )
    output = Path(output).absolute()
    output.mkdir(parents=True, exist_ok=False)
    (output / "exports").mkdir()
    (output / "views").mkdir()
    views = {}
    for row in value["components"]:
        for view in row.get("sampled_views", []):
            name = f"views/{row['execution']['job_id']}-{view['view_id']}.png"
            raw = read_file(selected_sources[row["component_id"]].parent.parent / name, 8388608)
            if checksum(raw) != view["view_checksum"]:
                raise ValueError("qualitative_combine_view_binding")
            write_bytes_once(output / name, raw)
            views[name] = checksum(raw)
    artifact = seal(value)
    path = output / "exports" / (artifact["checksum"][7:] + ".json")
    write_once(path, artifact)
    write_bytes_once(path.with_suffix(".html"), render(artifact).encode())
    write_once(
        path.with_suffix(".manifest.json"),
        seal(
            dict(
                result_checksum=artifact["checksum"],
                plan_checksum=artifact["plan_checksum"],
                json_checksum=checksum(read_file(path, 16777216)),
                html_checksum=checksum(read_file(path.with_suffix(".html"), 16777216)),
            )
        ),
    )
    write_once(
        output / "derivation.json",
        seal(
            dict(
                current_result_checksum=new["checksum"],
                previous_result_checksum=old["checksum"],
                result_checksum=artifact["checksum"],
                new_api_requests=0,
                views=views,
                software=runner.software(),
            )
        ),
    )
    verified_result(path)
    return dict(
        json=str(path),
        html=str(path.with_suffix(".html")),
        historical_fallback_components=fallback,
        new_api_requests=0,
        current_revision_all_planned_jobs_completed=new["coverage"]["all_planned_jobs_completed"],
    )
