"""Read existing Tool registry and locked aggregate cards; never execute models."""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from packages.evidence_details.models import digest

ROOT = Path(__file__).resolve().parents[1]
CARDS = [
    ("configs/tools/segman-slo-resources-v1.json", ["segman_slo_disc_cup_segmenter"]),
    (
        "configs/tools/sloctolyzer-resources-v1.json",
        [
            "sloctolyzer_vessel_segmenter",
            "sloctolyzer_avod_segmenter",
            "sloctolyzer_fovea_detector",
        ],
    ),
    (
        "configs/tools/disc-geometry-v1.json",
        ["calculate_slo_rim_directional_width", "calculate_slo_disc_ellipse"],
    ),
    ("configs/tools/venular-branch-geometry-v1.json", ["calculate_slo_venular_branch_angles"]),
    (
        "docs/experiments/fairclip-validation-20260914-v1/tool-evaluation.json",
        ["fairclip_slo_glaucoma_classifier"],
    ),
]


def knowledge_sources(base):
    """Resolve exact archived source -> registered Knowledge provider, read-only."""
    from apps.knowledge_source_tools import repository
    from db.knowledge.models import KnowledgeReference

    refs = {}

    def find(value):
        if isinstance(value, dict):
            if value.get("kind") == "source_version" and all(
                k in value for k in ("code", "version", "checksum")
            ):
                ref = {k: value[k] for k in ("kind", "code", "version", "checksum")}
                refs[digest(ref)] = ref
            for item in value.values():
                find(item)
        elif isinstance(value, list):
            for item in value:
                find(item)

    find(base)
    if len(refs) > 256:
        raise ValueError("detail_knowledge_reference_limit")
    # This is an offline maintenance snapshot. The normal Planner reader cannot
    # inspect Draft archives; use the existing editor's read port, never broaden
    # Planner permissions or promote these records to Published.
    repo = repository("editor")
    result = []
    for key in sorted(refs):
        snapshot = repo.get(KnowledgeReference.model_validate_json(json.dumps(refs[key])))
        r = snapshot.record
        provider = repo.get(r.provider_ref)
        result.append(
            {
                "reference": refs[key],
                "provider": r.provider_ref.model_dump(mode="json"),
                "source": r.source_ref.model_dump(mode="json"),
                "status": snapshot.status.value,
                "withdrawn": snapshot.withdrawn,
                "provider_status": provider.status.value,
                "provider_withdrawn": provider.withdrawn,
                "content_checksum": r.content_checksum,
                "object_uri": r.object_uri,
                "media_type": r.media_type,
                "fetched_at": r.fetched_at.isoformat(),
                "size_bytes": r.size_bytes,
                "discovery_refs": [x.model_dump(mode="json") for x in r.discovery_refs],
                "current_read_is_publication_authorization": False,
            }
        )
    return result


def tool_identities(base):
    found = set()

    def walk(value):
        if isinstance(value, dict):
            code = value.get("tool_code") or value.get("code")
            version = value.get("version") or value.get("tool_version")
            if isinstance(code, str) and code.isidentifier() and isinstance(version, str):
                found.add((code, version))
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    for node in base["nodes"]:
        walk(node.get("patient_evidence", {}).get("tool"))
        walk(node.get("tools", []))
    walk(base.get("branch_geometry", {}).get("tool"))
    walk(base.get("case_completion", {}).get("geometry", {}))
    if base.get("case_evidence", {}).get("fairclip", {}).get("outputs"):
        found.add(("fairclip_slo_glaucoma_classifier", "1.0.0"))
    return sorted(found)


def collect(base):
    from apps.slo_tool_registry import checksum as manifest_checksum
    from apps.slo_tool_registry import native_connection

    tools = []
    with native_connection() as c:
        c.rollback()
        c.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
        for code, version in tool_identities(base):
            rows = c.execute(
                "SELECT i.implementation_key,v.enabled,i.enabled,i.validation_status "
                "FROM medgraph.algorithm_tool t JOIN medgraph.tool_version v ON v.tool_id=t.id "
                "JOIN medgraph.tool_implementation i ON i.tool_version_id=v.id "
                "WHERE t.code=%s AND v.version=%s ORDER BY i.implementation_key LIMIT 16",
                (code, version),
            ).fetchall()
            implementations = []
            for implementation, version_enabled, enabled, status in rows:
                manifest = (
                    c.execute(
                        "SELECT medgraph.get_tool_execution_manifest(%s,%s,%s)",
                        (code, version, implementation),
                    ).fetchone()[0]
                    if enabled and version_enabled
                    else None
                )
                # No environment variable values or arbitrary deployment credentials.
                public = (
                    None
                    if manifest is None
                    else {
                        k: manifest.get(k)
                        for k in (
                            "tool_code",
                            "version",
                            "implementation_key",
                            "capability_kind",
                            "formula_spec",
                            "invocation",
                            "entrypoint_uri",
                            "validation_report",
                            "validation_status",
                            "resources",
                            "provenance",
                            "hardware_requirements",
                        )
                    }
                )
                implementations.append(
                    dict(
                        implementation=implementation,
                        version_enabled=version_enabled,
                        enabled=enabled,
                        validation_status=status,
                        manifest_checksum="sha256:" + manifest_checksum(manifest)
                        if manifest
                        else None,
                        public_manifest=public,
                    )
                )
            tools.append(
                dict(
                    code=code,
                    version=version,
                    database="medgraph",
                    observed_implementations=implementations,
                    status="observed" if rows else "not_registered",
                    historical_execution_identity="only_as_recorded_in_base_graph",
                    execution_authorization=False,
                )
            )
    evaluations = []
    for name, codes in CARDS:
        path = ROOT / name
        if path.stat().st_size > 2_000_000:
            raise ValueError("detail_profile_size_limit")
        raw = path.read_bytes()
        value = json.loads(raw)
        # Resource hashes are retained without reading weights, datasets or labels.
        evaluations.append(
            dict(
                path=name,
                file_checksum="sha256:" + hashlib.sha256(raw).hexdigest(),
                tool_codes=codes,
                document=value,
                use="archived_aggregate_or_installation_record_not_patient_confidence",
            )
        )
    # Dedupe identical source descriptions, not distinct publications/cohorts.
    # These are source appraisals awaiting review, not published medical claims.
    sources = {}

    def source_cards(value):
        if isinstance(value, dict):
            if (
                isinstance(value.get("url"), str)
                and value.get("title")
                and any(k in value for k in ("fact", "text", "summary"))
            ):
                key = digest(value)
                sources[key] = {
                    "id": key,
                    "record": value,
                    "same_url_group": value["url"],
                    "representation": value.get("representation", "not_structured"),
                    "study_appraisal": {
                        name: {"value": None, "missing_reason": "not_structurally_appraised"}
                        for name in (
                            "study_design",
                            "sample_size",
                            "statistical_unit",
                            "population",
                            "reference_standard",
                            "effect_size",
                            "confidence_interval",
                            "risk_of_bias",
                            "confounders",
                            "external_validity",
                            "retraction_status",
                        )
                    },
                    "cohort_independence": "not_verified",
                    "quality_score": None,
                    "appraisal_scope": "source_text_is_bound; structured_review_not_yet_performed",
                }
            for item in value.values():
                source_cards(item)
        elif isinstance(value, list):
            for item in value:
                source_cards(item)

    for section in ("research_criteria", "source_associations", "display_explanations"):
        source_cards(base.get(section, {}))
    return {
        "schema": "evidence-source-tool-profiles/1.0.0",
        "assembly": {
            "builder": "judgment-binding/1.0.0",
            "actor": "assistant_engineering_binding",
            "recorded_at": datetime.now(UTC).isoformat(),
            "medical_review": False,
            "model_rerun": False,
            "new_scores": False,
            "knowledge_status": "exact_historical_graph_references_not_new_publication",
            "knowledge_access": "existing_editor_archival_get_only",
        },
        "tools": tools,
        "evaluations": evaluations,
        "sources": [sources[key] for key in sorted(sources)],
        "knowledge_sources": knowledge_sources(base),
    }
