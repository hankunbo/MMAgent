"""Append a confirmed, same-contract subset to a saved patient description."""

import copy
from pathlib import Path

from packages.object_store import checksum
from packages.tool_execution.process import read_file

from . import runner
from .comparison import verified_result
from .integration import integrate
from .report import render
from .storage import read, seal, unseal, write_bytes_once, write_once


def supplement_confirmed(current, supplement, output):
    current, supplement = Path(current).absolute(), Path(supplement).absolute()
    base, added = verified_result(current), verified_result(supplement)
    if base["packet_checksum"] != added["packet_checksum"]:
        raise ValueError("qualitative_supplement_packet")
    if (
        not added["coverage"]["all_planned_jobs_completed"]
        or added["execution"]["simulated"] != base["execution"]["simulated"]
    ):
        raise ValueError("qualitative_supplement_not_confirmed")
    plans = [read(p.parent.parent / "plan.json") for p in (current, supplement)]
    for field in (
        "schema_version",
        "prompt_checksum",
        "views_per_component",
        "image_edge",
        "max_output_tokens",
    ):
        if plans[0]["config"][field] != plans[1]["config"][field]:
            raise ValueError("qualitative_supplement_contract_changed")
    if (
        base["schema_version"] != added["schema_version"]
        or base["schema_version"] != "qualitative-case-description/1.1.0"
    ):
        raise ValueError("qualitative_supplement_result_version")
    existing = {r["component_id"]: r for r in base["components"]}
    incoming = {r["component_id"]: r for r in added["components"]}
    if not incoming or not incoming.keys() <= existing.keys() or added["scope"]["background_ids"]:
        raise ValueError("qualitative_supplement_scope")
    for cid, row in incoming.items():
        old = existing[cid]
        if old["execution"]["visual"] == "completed":
            raise ValueError("qualitative_supplement_would_replace_confirmed")
        if (
            row["source_declared"] != old["source_declared"]
            or row["component_checksum"] != old["component_checksum"]
        ):
            raise ValueError("qualitative_supplement_source_changed")
        if row["execution"]["visual"] != "completed" or not row.get("response_checksum"):
            raise ValueError("qualitative_supplement_unconfirmed_component")
        old_job = read(current.parent.parent / "jobs" / (old["execution"]["job_id"] + ".json"))
        new_job = read(supplement.parent.parent / "jobs" / (row["execution"]["job_id"] + ".json"))
        if old_job["context"] != new_job["context"]:
            raise ValueError("qualitative_supplement_visual_input_changed")
    value = copy.deepcopy(unseal(base))
    sources = {}
    for i, row in enumerate(value["components"]):
        cid = row["component_id"]
        sources[cid] = (current, row["execution"].get("job_id"))
        if cid in incoming:
            replacement = copy.deepcopy(incoming[cid])
            original_job = replacement["execution"]["job_id"]
            replacement["execution"]["job_id"] = "S1-" + original_job
            replacement["execution"]["supplemental_confirmation"] = dict(
                previous_visual_state=row["execution"]["visual"],
                previous_result_checksum=base["checksum"],
                supplemental_result_checksum=added["checksum"],
                original_job_id=original_job,
                response_checksum=replacement["response_checksum"],
                same_visual_contract=True,
            )
            value["components"][i] = replacement
            sources[cid] = (supplement, original_job)
    value = integrate(value)
    value["coverage"]["case_visual_completed"] = not value["coverage"].get(
        "projection_failures"
    ) and all(
        r["execution"]["visual"] == "completed"
        for r in value["components"]
        if r["execution"].get("job_id")
    )
    value["coverage"]["supplemented_components"] = list(incoming)
    value["coverage"]["original_plan_state_preserved"] = True
    value["execution"]["cumulative_api_requests"] = (
        base["execution"]["confirmed_api_requests"] + added["execution"]["confirmed_api_requests"]
    )
    value["execution"]["supplement_api_requests"] = added["execution"]["confirmed_api_requests"]
    value["execution"]["historical_unknown_request_count"] = base["execution"][
        "unknown_request_count"
    ]
    value["derivation"] = dict(
        operation="supplement_confirmed_same_contract",
        base_result_checksum=base["checksum"],
        supplemental_result_checksum=added["checksum"],
        new_api_requests=0,
        note="Case materials completed across attempts; original request states are unchanged.",
    )
    output = Path(output).absolute()
    output.mkdir(parents=True, exist_ok=False)
    (output / "views").mkdir()
    (output / "exports").mkdir()
    bindings = {}
    for row in value["components"]:
        source, job = sources[row["component_id"]]
        for view in row.get("sampled_views", []):
            raw = read_file(
                source.parent.parent / "views" / f"{job}-{view['view_id']}.png", 8388608
            )
            if checksum(raw) != view["view_checksum"]:
                raise ValueError("qualitative_supplement_image_binding")
            name = f"views/{row['execution']['job_id']}-{view['view_id']}.png"
            write_bytes_once(output / name, raw)
            bindings[name] = checksum(raw)
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
                result_checksum=artifact["checksum"],
                base_result_checksum=base["checksum"],
                supplemental_result_checksum=added["checksum"],
                views=bindings,
                software=runner.software(),
                new_api_requests=0,
            )
        ),
    )
    verified_result(path)
    return dict(
        json=str(path),
        html=str(path.with_suffix(".html")),
        case_visual_completed=value["coverage"]["case_visual_completed"],
        supplemented_components=list(incoming),
        new_api_requests=0,
    )
