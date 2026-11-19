"""Bounded generic composition, preserving original tool and research ownership."""

import json
from pathlib import Path

from apps.vectorengine_single_case import write_once
from packages.grounding.models import digest
from packages.object_store import checksum

from .generic_control import guard, start
from .generic_input import directory, load, register


def run(key, *, recover=False):
    from .generic_execution_lock import run_lock

    spec = load(key)
    out = directory(key)
    with run_lock(spec, out):
        started = out / "started.json"
        if started.exists() and not recover:
            raise ValueError("generic_run_exists_use_recover_or_export")
        if recover and not started.exists():
            raise ValueError("generic_recovery_requires_existing_run")
        if spec["schema_version"] in {
            "fairvlmed-run-input/1.7.0",
            "fairvlmed-run-input/1.8.0",
            "fairvlmed-run-input/1.9.0",
            "fairvlmed-run-input/1.10.0",
            "fairvlmed-run-input/1.11.0",
        }:
            from packages.llm_provider.network_route import route_opener

            from .generic_input import provider_config

            config = provider_config(spec)
            route_opener(config.network_route, config.transport.base_url)
        start(spec)
        guard(key)
        registered = register(spec)
        # No graph stage is revisited merely to export a terminal report.
        if (out / "export-reference.json").exists():
            return dict(status="already_recorded", exports=finish_export(key))
        from .generic_campaign import GenericCampaign, source_plan
        from .generic_tools import acquire

        plan = source_plan(spec, registered)
        acquire(
            spec,
            registered,
            recover=recover,
            requirements=tuple(r["id"] for r in plan["requirements"]),
        )

        campaign = GenericCampaign(spec, registered)
        result = campaign.execute()
        return persist_export(key, campaign, result)


def persist_export(key, campaign, result):
    out = directory(key)
    docs = campaign.documents(result["phases"][-1]["run_id"])
    key_ref = docs.key(getattr(campaign, "report_key", "dynamic-case-report"))
    artifacts = campaign.base.runtime.read_observation_bundle(key_ref)
    if not artifacts or len(artifacts) != 2:
        raise ValueError("generic_saved_report_missing")
    descriptor = dict(
        schema_version="generic-export-reference/1.0.0",
        run_key=key,
        case_code=campaign.base.binding["case_code"],
        bundle_key=key_ref,
        run_id=docs.run_id,
        artifacts=[v.model_dump(mode="json") for v in artifacts],
        complete=result["complete"],
        clinical_acceptance=False,
        root_checksum=docs.root.checksum,
        report_key_checksum=digest(getattr(campaign, "report_key", "dynamic-case-report")),
    )
    write_once(out / "export-reference.json", descriptor)
    return dict(result=result, exports=finish_export(key))


def finish_export(key):
    """Resume display-only work from saved Runtime facts even after run expiry.

    Original export is unchanged. A display failure never causes a second paid run.
    """
    result = export(key)
    report = json.loads(Path(result["json"]).read_bytes())
    production = report.get("report_production")
    if production:
        from .report_production import PROFILE

        if production.get("schema_version") != PROFILE:
            raise ValueError("generic_report_production_version")
        destination = Path(result["html"]).with_name(Path(result["html"]).stem + "_证据树.html")
        result["tree"] = export_presentation(key, destination)
    return result


def export_presentation(key, destination, *, media_path=None):
    """Owner-bound readback to a new display snapshot; no new execution or database write.

    Memory/structure come from this exact saved report, never a current global lookup.
    Existing export(key) remains byte-identical and does not change its result shape.
    """
    from .evidence_view import convert

    saved = export(key)
    return convert(
        saved["json"],
        destination,
        layout="tree-v3",
        run_key=key if media_path is None else None,
        media_path=media_path,
    )


def export(key):
    """Read exact Runtime report objects. No source dataset/model/LLM construction."""
    from .case_execution import owners

    out = directory(key)
    reference = json.loads((out / "export-reference.json").read_bytes())
    if (
        reference["schema_version"] != "generic-export-reference/1.0.0"
        or reference["run_key"] != key
    ):
        raise ValueError("generic_export_reference")
    runtime, objects, _ = owners()
    artifacts = runtime.read_observation_bundle(reference["bundle_key"])
    if not artifacts or [a.model_dump(mode="json") for a in artifacts] != reference["artifacts"]:
        raise ValueError("generic_export_original_owner_binding")
    values = []
    for index, a in enumerate(artifacts):
        if a.scope.case_code != reference["case_code"]:
            raise ValueError("generic_export_case_binding")
        raw = objects.read_bounded(a.uri, 4194304)
        if checksum(raw) != a.checksum:
            raise ValueError("generic_export_content_changed")
        envelope = json.loads(raw)
        if (
            envelope.get("schema_version") != "agent-runtime-document/1.0.0"
            or envelope.get("run_id") != reference["run_id"]
            or envelope.get("root_checksum") != reference["root_checksum"]
            or envelope.get("key_checksum") != reference["report_key_checksum"]
            or envelope.get("index") != index
        ):
            raise ValueError("generic_export_document_scope")
        values.append(envelope["value"])
    report, display = values
    if display["report_checksum"] != digest(report):
        raise ValueError("generic_export_report_binding")
    output = Path("docs/exports") / ("FairVLMed_" + key)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_once(output.with_suffix(".json"), report)
    html = output.with_suffix(".html")
    rendered = display["html"]
    if html.exists() and html.read_text() != rendered:
        raise ValueError("generic_export_no_overwrite")
    if not html.exists():
        with html.open("x") as stream:
            stream.write(rendered)
    return dict(
        json=str(output.with_suffix(".json")), html=str(html), api_requests=0, model_starts=0
    )
