"""Terminal-owned qualitative analysis. Never creates requirements or medical scores."""

import base64
import importlib.util
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

from pydantic import ValidationError

from packages.dataset_adapters.patient_storage import verify as verify_packet
from packages.grounding.models import digest
from packages.llm_provider.transport import HTTPTransport
from packages.object_store import checksum
from packages.tool_execution.process import read_file

from .models import AnalysisConfig, MaterialScope, VisualResult, VisualResultV2
from .projection import deterministic, views
from .storage import file_bindings, locked, read, seal, unseal, write_bytes_once, write_once

ROOT = Path(__file__).resolve().parents[2]


def software():
    paths = sorted((ROOT / "packages/qualitative_analysis").glob("*.py"))
    paths += [
        ROOT / "apps/qualitative_analyzer.py",
        ROOT / "packages/dataset_adapters/patient_storage.py",
        ROOT / "packages/dataset_adapters/patient_models.py",
        ROOT / "packages/llm_provider/transport.py",
        ROOT / "packages/llm_provider/worker.py",
    ]
    return file_bindings(paths, ROOT)


def config_at(path):
    path = Path(path).absolute()
    if not path.is_relative_to(ROOT / "configs"):
        raise ValueError("qualitative_public_config_required")
    config = AnalysisConfig.model_validate(read(path))
    prompt_path = ROOT / config.prompt
    if not prompt_path.is_relative_to(ROOT / "configs/prompts") or ".." in prompt_path.parts:
        raise ValueError("qualitative_prompt_path")
    prompt = read_file(prompt_path, 65536).decode()
    if checksum(prompt.encode()) != config.prompt_checksum:
        raise ValueError("qualitative_prompt_drift")
    return config, prompt


def plan(packet_dir, output, config_path, scope=None):
    packet_dir, output = Path(packet_dir).absolute(), Path(output).absolute()
    packet = verify_packet(packet_dir)
    config, prompt = config_at(config_path)
    if scope is None:
        raise ValueError("qualitative_explicit_material_scope_required")
    scope = MaterialScope.model_validate(scope)
    cids, bids = {c.id for c in packet.components}, {b.id for b in packet.background}
    if not set(scope.component_ids) <= cids or not set(scope.background_ids) <= bids:
        raise ValueError("qualitative_scope_unknown_component")
    if scope.mode == "archive_description" and (
        set(scope.component_ids) != cids or set(scope.background_ids) != bids
    ):
        raise ValueError("qualitative_archive_scope_incomplete")
    if any(p.is_symlink() for p in (output, *output.parents)):
        raise ValueError("qualitative_output_symlink")
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    for folder in ("views", "jobs", "journal", "exports"):
        (output / folder).mkdir(mode=0o700)
    structural = deterministic(packet, scope, packet_dir)
    enhanced = config.schema_version == "qualitative-analysis-config/1.1.0"
    if enhanced:
        from .integration import field_contexts

        structural["field_contexts"] = field_contexts(structural["components"])
    payloads = {p.id: p for p in packet.payloads}
    jobs, reused = [], {}
    for c in packet.components:
        if c.id not in scope.component_ids:
            continue
        row = next(r for r in structural["components"] if r["component_id"] == c.id)
        if c.status in {"missing", "withheld"}:
            row["execution"]["visual"] = "material_unavailable"
            continue
        try:
            descriptions, images = views(packet_dir, c, payloads, config)
        except (ValueError, OSError) as error:
            row["execution"].update(visual="projection_failed", error_type=type(error).__name__)
            continue
        if not images:
            continue
        # Reuse only identical content + interpretation metadata within this packet.
        signature = digest(
            dict(
                payloads=[payloads[p].checksum for p in c.payload_ids],
                kind=c.kind,
                modality=c.modality,
                unit=c.unit,
                **(dict(laterality=c.laterality) if enhanced else {}),
                transform=[d["source"]["transform"] for d in descriptions],
            )
        )
        if signature in reused:
            row["execution"].update(visual="pending", job_id=reused[signature], reused=True)
            continue
        job_id = "J" + str(len(jobs) + 1)
        reused[signature] = job_id
        # Source declarations are context, never promoted to observed confirmation.
        context = dict(
            schema_version="qualitative-visual-input/1.1.0"
            if enhanced
            else "qualitative-visual-input/1.0.0",
            source_declared=dict(
                kind=c.kind,
                modality=c.modality,
                unit=c.unit,
                **(
                    dict(
                        laterality=c.laterality,
                        laterality_basis="packet_source_declaration_not_visual_confirmation",
                    )
                    if enhanced
                    else {}
                ),
            ),
            views=descriptions,
            sampling=dict(
                view_count=len(images),
                payload_count=len(c.payload_ids),
                source_shape=list(c.shape),
                scope="supplied_views_only_not_whole_volume",
            ),
            output_schema=(VisualResultV2 if enhanced else VisualResult).model_json_schema(),
        )
        paths = []
        for d, raw in zip(descriptions, images, strict=True):
            name = f"views/{job_id}-{d['view_id']}.png"
            path = output / name
            with path.open("xb") as f:
                f.write(raw)
            path.chmod(0o600)
            paths.append(dict(path=name, checksum=checksum(raw)))
        job = seal(
            dict(
                id=job_id,
                context=context,
                images=paths,
                annotation_scope="sampled_views",
                source_component_id=c.id,
            )
        )
        write_once(output / "jobs" / (job_id + ".json"), job)
        jobs.append(dict(id=job_id, checksum=job["checksum"]))
        row["execution"].update(visual="pending", job_id=job_id, reused=False)
    write_once(output / "structural.json", seal(structural))
    record = seal(
        dict(
            schema_version="qualitative-analysis-plan/1.0.0",
            packet_dir=str(packet_dir),
            packet_checksum=packet.checksum,
            scope=scope.model_dump(mode="json"),
            config=config.model_dump(mode="json"),
            prompt=prompt,
            jobs=jobs,
            structural_checksum=read(output / "structural.json")["checksum"],
            software=software(),
            clinical_acceptance=False,
            runtime_ready=False,
        )
    )
    write_once(output / "plan.json", record)
    return status(output)


def load(output, *, current=True):
    output = Path(output).absolute()
    saved = read(output / "plan.json")
    p = unseal(saved)
    if current and p["software"] != software():
        raise ValueError("qualitative_implementation_changed_use_new_plan")
    if verify_packet(p["packet_dir"]).checksum != p["packet_checksum"]:
        raise ValueError("qualitative_packet_changed")
    structure = read(output / "structural.json")
    unseal(structure)
    if structure["checksum"] != p["structural_checksum"]:
        raise ValueError("qualitative_structural_binding")
    config = AnalysisConfig.model_validate(p["config"])
    if checksum(p["prompt"].encode()) != config.prompt_checksum:
        raise ValueError("qualitative_saved_prompt_binding")
    for binding in p["jobs"]:
        j = read(output / "jobs" / (binding["id"] + ".json"))
        unseal(j)
        if j["checksum"] != binding["checksum"]:
            raise ValueError("qualitative_job_binding")
        for image in j["images"]:
            raw = read_file(output / image["path"], 8388608)
            if checksum(raw) != image["checksum"]:
                raise ValueError("qualitative_view_changed")
        journal_binding(output, saved, binding)
    return output, saved, config


def journal_binding(output, saved, binding):
    records = {}
    for suffix in ("intent", "response"):
        path = output / "journal" / f"{binding['id']}.{suffix}.json"
        if path.exists():
            value = unseal(read(path))
            if (
                value["plan_checksum"] != saved["checksum"]
                or value["job_checksum"] != binding["checksum"]
            ):
                raise ValueError("qualitative_journal_binding")
            records[suffix] = value
    if "response" in records and (
        "intent" not in records
        or records["response"]["request_checksum"] != records["intent"]["request_checksum"]
    ):
        raise ValueError("qualitative_response_request_binding")


def response_result(output, job_id):
    path = output / "journal" / (job_id + ".response.json")
    if not path.exists():
        return None
    response = read(path)
    unseal(response)
    if response["error"]:
        return dict(
            state="confirmed_error"
            if response["response_confirmed"]
            else "not_dispatched_error"
            if response["api_requests"] == 0
            else "unknown",
            error=response["error"],
        )
    try:
        job = read(output / "jobs" / (job_id + ".json"))
        model = (
            VisualResultV2
            if job["context"]["schema_version"] == "qualitative-visual-input/1.1.0"
            else VisualResult
        )
        candidate = model.model_validate_json(response["content"])
        valid = {v["view_id"] for v in job["context"]["views"]}
        refs = list(candidate.annotations)
        if isinstance(candidate, VisualResultV2):
            refs += candidate.anatomy + candidate.uncertainties + [candidate.visual_laterality]
        if any(a.view_id not in valid for a in refs):
            raise ValueError("qualitative_view_reference")
        if not {"type_modality", "content_regions", "visibility"} <= {
            a.category for a in candidate.annotations
        }:
            raise ValueError("qualitative_missing_description_categories")
        return dict(
            state="completed",
            candidate=candidate.model_dump(mode="json"),
            response_checksum=response["checksum"],
        )
    except (ValueError, ValidationError):
        return dict(state="contract_rejected", error="annotation_schema_or_binding")


def status(output):
    output, saved, _ = load(output, current=False)
    states = {}
    for j in saved["jobs"]:
        result = response_result(output, j["id"])
        states[j["id"]] = (
            result["state"]
            if result
            else (
                "unknown"
                if (output / "journal" / (j["id"] + ".intent.json")).exists()
                else "not_sent"
            )
        )
    return dict(
        output=str(output),
        plan_checksum=saved["checksum"],
        jobs=states,
        dispatch_reservations=len(list((output / "journal").glob("*.intent.json"))),
        clinical_acceptance=False,
        database_writes=0,
    )


@contextmanager
def credentials(config, local_key):
    name = config.transport.api_key_env
    previous = os.environ.get(name)
    if previous:
        yield
        return
    if not local_key:
        raise ValueError("qualitative_key_environment_missing")
    spec = importlib.util.spec_from_file_location(
        "qualitative_private_launcher", ROOT / ".local/vectorengine/run.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        os.environ[name] = module.load_secret()
        yield
    finally:
        os.environ.pop(name, None)


def wire(output, job, plan, config):
    content = [dict(type="text", text=json.dumps(job["context"], ensure_ascii=False))]
    for image in job["images"]:
        content.append(
            dict(
                type="image_url",
                image_url=dict(
                    url="data:image/png;base64,"
                    + base64.b64encode(read_file(output / image["path"], 8388608)).decode()
                ),
            )
        )
    return dict(
        model=config.transport.model,
        store=False,
        stream=False,
        max_completion_tokens=config.max_output_tokens,
        response_format=dict(type="json_object"),
        messages=[dict(role="system", content=plan["prompt"]), dict(role="user", content=content)],
    )


def run(output, *, live=False, local_key=False, transport=None):
    output = Path(output).absolute()
    with locked(output):
        output, saved, config = load(output)
        if not live:
            return export(output)
        if (output / "cancel.json").exists():
            return export(output)
        start = output / "window.json"
        if not start.exists():
            write_once(
                start,
                seal(
                    dict(
                        plan_checksum=saved["checksum"],
                        started_at=time.time(),
                        deadline=time.time() + config.duration_seconds,
                    )
                ),
            )
        window = unseal(read(start))
        if window["plan_checksum"] != saved["checksum"]:
            raise ValueError("qualitative_window_binding")
        transport = transport or HTTPTransport(
            network_route=config.network_route, http_client=config.http_client
        )
        if not getattr(transport, "simulated", False):
            from packages.llm_provider.curl_client import verify_binary
            from packages.llm_provider.network_route import route_opener

            route_opener(config.network_route, config.transport.base_url)
            if config.http_client:
                verify_binary(config.http_client)
        for binding in saved["jobs"]:
            job_id = binding["id"]
            intent_path = output / "journal" / (job_id + ".intent.json")
            if intent_path.exists():
                continue  # confirmed or UNKNOWN is never automatically dispatched again
            count = len(list((output / "journal").glob("*.intent.json")))
            remaining = window["deadline"] - time.time()
            if count >= config.max_requests or remaining < 5 or (output / "cancel.json").exists():
                break
            job = read(output / "jobs" / (job_id + ".json"))
            body = wire(output, job, saved, config)
            timeout = min(config.transport.timeout_seconds + 10, remaining)
            # Resolve key and route before recording a dispatch intent. Never persist secrets.
            with (
                credentials(config, local_key)
                if not getattr(transport, "simulated", False)
                else _no_key()
            ):
                if (output / "cancel.json").exists() or time.time() >= window["deadline"]:
                    break
                write_once(
                    intent_path,
                    seal(
                        dict(
                            plan_checksum=saved["checksum"],
                            job_checksum=job["checksum"],
                            request_checksum=digest(body),
                            reservation=count + 1,
                            dispatched_at=time.time(),
                        )
                    ),
                )
                reply = transport.send(
                    config.transport,
                    body,
                    timeout=timeout,
                    max_bytes=config.max_response_bytes,
                    effective_http_timeout=max(1, timeout - 2),
                )
            # Capture BEFORE validation/export so cold recovery never asks the model again.
            write_once(
                output / "journal" / (job_id + ".response.json"),
                seal(
                    dict(
                        plan_checksum=saved["checksum"],
                        job_checksum=job["checksum"],
                        request_checksum=digest(body),
                        error=reply.error,
                        content=reply.content.decode("utf-8", errors="replace"),
                        api_requests=reply.api_requests,
                        response_confirmed=reply.response_confirmed,
                        simulated=bool(getattr(transport, "simulated", False)),
                        usage=reply.usage.model_dump(mode="json"),
                        finished_at=time.time(),
                    )
                ),
            )
            progress = response_result(output, job_id)
            print(
                json.dumps(
                    dict(job=job_id, state=progress["state"], dispatch_reservations=count + 1)
                ),
                flush=True,
            )
        return export(output)


@contextmanager
def _no_key():
    yield


def export(output):
    output, saved, _ = load(output, current=False)
    result = unseal(read(output / "structural.json"))
    state = status(output)
    for row in result["components"]:
        jid = row["execution"].get("job_id")
        if not jid:
            continue
        answer = response_result(output, jid)
        row["execution"]["visual"] = state["jobs"][jid]
        if answer and answer["state"] == "completed":
            candidate = answer["candidate"]
            row["model_annotations"] = candidate["annotations"]
            row["unconfirmed"] = candidate["unconfirmed"]
            for field in ("anatomy", "visual_laterality", "uncertainties"):
                if field in candidate:
                    row[field] = candidate[field]
            row["interpretation_state"] = "model_annotation_not_independent_confirmation"
            row["response_checksum"] = answer["response_checksum"]
            for category in ("content_regions", "visibility"):
                row[category] = [a for a in candidate["annotations"] if a["category"] == category]
            row["type_modality"]["actual_content_confirmation"] = "model_annotation_available"
            row["semantics"]["candidate_annotations"] = [
                a for a in candidate["annotations"] if a["category"] == "semantics"
            ]
            # Free-text model labels do not overwrite source units, eye or measurements.
            row["semantics"]["source_conflict_review"] = (
                "compare_source_and_candidates_no_auto_override"
            )
            declarations = row["source_declared"]
            row["conflicts"] = []
            for annotation in candidate["annotations"]:
                field = annotation.get("attribute")
                if field in {"unit", "modality", "laterality"}:
                    source = declarations.get(field)
                    if source not in (None, "UNKNOWN") and annotation["value"] != source:
                        row["conflicts"].append(
                            dict(
                                field=field,
                                source_value=source,
                                candidate_value=annotation["value"],
                                status="textual_difference_requires_semantic_confirmation",
                                view_id=annotation["view_id"],
                            )
                        )
            job = read(output / "jobs" / (jid + ".json"))
            row["sampled_views"] = job["context"]["views"]
            row["sampling"] = job["context"]["sampling"]
            referenced = {a["view_id"] for a in candidate["annotations"]}
            row["sampling"]["model_referenced_view_ids"] = sorted(referenced)
            row["sampling"]["unreferenced_view_ids"] = [
                v["view_id"] for v in job["context"]["views"] if v["view_id"] not in referenced
            ]
    counts = {s: list(state["jobs"].values()).count(s) for s in sorted(set(state["jobs"].values()))}
    failures = [
        r["component_id"]
        for r in result["components"]
        if r["execution"]["visual"] == "projection_failed"
    ]
    complete = all(s == "completed" for s in state["jobs"].values()) and not failures
    responses = [unseal(read(p)) for p in (output / "journal").glob("*.response.json")]
    result.update(
        schema_version="qualitative-case-description/1.0.0",
        packet_checksum=saved["packet_checksum"],
        plan_checksum=saved["checksum"],
        scope=saved["scope"],
        coverage=dict(
            job_states=counts,
            projection_failures=failures,
            all_planned_jobs_completed=complete,
            whole_volume_examined=False,
            component_count=len(result["components"]),
        ),
        execution=dict(
            dispatch_reservations=state["dispatch_reservations"],
            confirmed_api_requests=sum(r["api_requests"] or 0 for r in responses),
            unknown_request_count=counts.get("unknown", 0),
            cancelled=(output / "cancel.json").exists(),
            database_writes=0,
            simulated=any(r.get("simulated", False) for r in responses),
        ),
        clinical_acceptance=False,
        runtime_ready=False,
        disease_score=None,
    )
    if saved["config"]["schema_version"] == "qualitative-analysis-config/1.1.0":
        from .integration import integrate

        result = integrate(result)
    return persist_export(output, saved, result, state, complete)


def persist_export(output, saved, result, state, complete):
    artifact = seal(result)
    path = output / "exports" / (artifact["checksum"][7:] + ".json")
    write_once(path, artifact)
    from .report import render

    html = path.with_suffix(".html")
    if not html.exists():
        write_bytes_once(html, render(artifact).encode())
    write_once(
        path.with_suffix(".manifest.json"),
        seal(
            dict(
                result_checksum=artifact["checksum"],
                plan_checksum=saved["checksum"],
                json_checksum=checksum(read_file(path, 16777216)),
                html_checksum=checksum(read_file(html, 16777216)),
            )
        ),
    )
    return dict(
        **state,
        all_planned_jobs_completed=complete,
        json=str(path),
        html=str(html),
        result_checksum=artifact["checksum"],
    )


def verify(output):
    output, saved, _ = load(output, current=False)
    for j in saved["jobs"]:
        for suffix in ("intent", "response"):
            path = output / "journal" / f"{j['id']}.{suffix}.json"
            if path.exists():
                v = unseal(read(path))
                if v["plan_checksum"] != saved["checksum"] or v["job_checksum"] != j["checksum"]:
                    raise ValueError("qualitative_journal_binding")
        response = output / "journal" / f"{j['id']}.response.json"
        intent = output / "journal" / f"{j['id']}.intent.json"
        if response.exists() and (
            not intent.exists()
            or read(response)["request_checksum"] != read(intent)["request_checksum"]
        ):
            raise ValueError("qualitative_response_request_binding")
    for path in (output / "exports").glob("*.manifest.json"):
        manifest = unseal(read(path))
        stem = path.name.removesuffix(".manifest.json")
        data = output / "exports" / (stem + ".json")
        html = output / "exports" / (stem + ".html")
        value = read(data)
        unseal(value)
        if (
            manifest["plan_checksum"] != saved["checksum"]
            or manifest["result_checksum"] != value["checksum"]
            or manifest["json_checksum"] != checksum(read_file(data, 16777216))
            or manifest["html_checksum"] != checksum(read_file(html, 16777216))
        ):
            raise ValueError("qualitative_export_binding")
    return status(output)
