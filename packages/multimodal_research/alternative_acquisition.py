"""One qualified alternative view after a confirmed uninterpretable observation.

Uses the original campaign, input, native artifact, Runtime and Review owners.
An accepted observation of status unknown is not a transport-UNKNOWN request.
"""

import json
import tempfile
from copy import deepcopy
from pathlib import Path

from apps.vectorengine_case_materials import ref
from db.runtime.repository.case_pipeline import identity
from packages.contracts import Artifact
from packages.grounding.models import digest, validate
from packages.llm_provider.models import ImageBinding, read_request_record
from packages.object_store import checksum
from packages.research_runtime.agent_records import RuntimeAgentJournal

VERSION = "multimodal-campaign-renewal/1.59.0"
METHOD = "original_plus_registered_disc_roi_rgb_green/1.0.0"
TILED_VERSION = "multimodal-campaign-renewal/1.60.0"
TILED_METHOD = "original_plus_full_frame_quadrants_rgb_green/1.0.0"
DEFINITION_VERSION = "multimodal-campaign-renewal/1.64.0"
AGGREGATE_VERSION = "multimodal-campaign-renewal/1.68.0"
AGGREGATE_RECOVERY_VERSION = "multimodal-campaign-renewal/1.73.0"
AGGREGATE_VERSIONS = {AGGREGATE_VERSION, AGGREGATE_RECOVERY_VERSION}
VERSIONS = {VERSION, TILED_VERSION, DEFINITION_VERSION}
METHOD_VERSIONS = {METHOD: VERSION, TILED_METHOD: TILED_VERSION}


def observation_origin(parent, target):
    """Follow only the reconstructed parent chain to an identical confirmed fact."""
    current = parent.accepted_observations().get(target)
    if current is None or current.status not in {"unknown", "ungradable"}:
        raise ValueError("alternative_view_requires_uninterpretable_observation")
    ancestor, visited = parent, set()
    for _ in range(16):
        if ancestor is None or id(ancestor) in visited:
            break
        visited.add(id(ancestor))
        if (
            ancestor.base.manifest.checksum != parent.base.manifest.checksum
            or ancestor.base.task.checksum != parent.base.task.checksum
        ):
            raise ValueError("alternative_view_ancestor_input_or_task_changed")
        result = json.loads((ancestor.output / "result.json").read_bytes())
        phases = [p for p in result.get("phases", []) if p.get("target") == target]
        if phases:
            # Never skip a nearer failed/ambiguous attempt to find a convenient
            # older success. Caller verifies this exact phase and its journal.
            observed = ancestor.accepted_observations().get(target)
            if digest(observed) != digest(current):
                raise ValueError("alternative_view_ancestor_observation_changed")
            final = result["phases"][-1]
            if ancestor.documents(final["run_id"]).read("multimodal-case-result") != (result,):
                raise ValueError("alternative_view_ancestor_result_binding")
            return ancestor
        ancestor = getattr(ancestor, "parent", None)
    raise ValueError("alternative_view_confirmed_ancestor_not_found")


def aggregation_origin(parent, target):
    """Recheck current input/source projection against the original accepted aggregate.

    The original aggregation reader has already reconstructed every accepted
    origin. This check binds the proposed replacement of an unscored observation
    to that exact current aggregate, never a caller-supplied authorization flag.
    """
    from .campaign import Phase

    baseline = getattr(parent, "aggregation_baseline", None)
    if baseline is None:
        raise ValueError("alternative_view_aggregation_required")
    phase = Phase(parent.base, observation_reader=parent.accepted_observations)
    state, _ = phase.project(run_id=baseline.documents.run_id, config=parent.config)
    phase.verify(state)
    manifest = baseline.manifest
    if (
        digest(state) != digest(baseline.state)
        or digest(state) != manifest["state_checksum"]
        or digest(baseline.submission) != manifest["submission_checksum"]
        or baseline.documents.read("accepted-aggregation-manifest") != (manifest,)
        or baseline.documents.read("dynamic-phase-input") != (state.model_dump(mode="json"),)
    ):
        raise ValueError("alternative_view_aggregation_current_binding")
    if any(i.assessment_id == target for i in baseline.submission.candidate.items):
        raise ValueError("alternative_view_must_not_replace_accepted_judgment")
    result = json.loads((baseline.path / "result.json").read_bytes())
    if result.get("complete") is not True or baseline.documents.read("multimodal-case-result") != (
        result,
    ):
        raise ValueError("alternative_view_aggregation_result_binding")
    control = parent.sessions.read(baseline.documents.run_id)
    from packages.evidence_details.accepted_aggregation import GRANT_KEY

    grant = baseline.documents.read(GRANT_KEY)
    if (
        grant != (baseline.grant,)
        or control is None
        or digest(control) != baseline.grant["control_checksum"]
        or control.phase != "stopped"
        or control.cancelled
        or control.pending is not None
    ):
        raise ValueError("alternative_view_aggregation_control_binding")
    return dict(
        schema_version="alternative-accepted-aggregation-origin/1.0.0",
        path=str(baseline.path),
        manifest_checksum=digest(manifest),
        state_checksum=digest(state),
        submission_checksum=digest(baseline.submission),
        result_checksum=digest(result),
        target=target,
        original_judgments_not_replaced=True,
    )


def original_observation(parent, target, *, ancestor_allowed=False, definition_increment=False):
    """Reconstruct a single completed local observation, not an arbitrary response."""
    aggregated = getattr(parent, "aggregation_baseline", None) is not None
    if aggregated and definition_increment:
        raise ValueError("alternative_view_aggregation_definition_scope")
    aggregate = aggregation_origin(parent, target) if aggregated else None
    ancestor_allowed = ancestor_allowed or aggregated
    origin = observation_origin(parent, target) if ancestor_allowed else parent
    result = json.loads((origin.output / "result.json").read_bytes())
    parent_result = json.loads((parent.output / "result.json").read_bytes())
    if parent_result.get("complete") is not True:
        raise ValueError("alternative_view_completed_parent_required")
    phases = [p for p in result["phases"] if p["target"] == target]
    if len(phases) != 1 or not phases[0]["accepted"] or not phases[0]["acceptance_key"]:
        raise ValueError("alternative_view_confirmed_phase_required")
    phase = phases[0]
    docs = origin.documents(phase["run_id"])
    if docs.read("dynamic-phase-result") != (phase,):
        raise ValueError("alternative_view_phase_binding")
    control = origin.sessions.read(phase["run_id"])
    if control is None or control.pending is not None or control.cancelled:
        raise ValueError("alternative_view_unresolved_control")
    observations = parent.accepted_observations()
    observation = observations.get(target)
    if observation is None or observation.status not in {"unknown", "ungradable"}:
        raise ValueError("alternative_view_requires_uninterpretable_observation")
    events = RuntimeAgentJournal(docs).read()
    reserved = [e["request_id"] for e in events if e.get("phase") == "reserved"]
    completed = [read_request_record(e["record"]) for e in events if e.get("phase") == "completed"]
    confirmations = [e for e in events if e.get("phase") == "response_confirmation"]
    if (
        len(reserved) != 1
        or len(completed) != 1
        or len(confirmations) != 1
        or reserved != [completed[0].request_id]
        or confirmations[0].get("request_id") != reserved[0]
        or confirmations[0].get("confirmed") is not True
        or completed[0].role != "vision"
        or completed[0].simulated
        or completed[0].error is not None
        or completed[0].output_checksum is None
        or completed[0].api_requests != 1
        or docs.read("live-response:" + reserved[0]) is None
    ):
        raise ValueError("alternative_view_requires_confirmed_response_not_unknown")
    from .campaign_renewal import accepted_reader

    if definition_increment:
        from .definition_alternative import unreviewed

        definition_proof = unreviewed(parent, target)
    elif not aggregated:
        final = parent_result["phases"][-1]
        reader = accepted_reader(parent, final["run_id"])
        state = reader.state_reader()
        raw = state.model_dump(mode="json")
        raw["material"]["scope"]["run_id"] = identity(
            "RUN", "alternative-preflight:" + digest((str(parent.output), target))
        )
        raw["material"]["scope"].pop("material_checksum")
        raw["material"]["scope"]["material_checksum"] = digest(raw["material"])
        accepted = reader.current(validate(type(state), raw))
        if any(i.assessment_id == target for i in accepted.submission.candidate.items):
            raise ValueError("alternative_view_must_not_replace_accepted_judgment")
    value = dict(
        target=target,
        run_id=phase["run_id"],
        phase_checksum=digest(phase),
        acceptance_key=phase["acceptance_key"],
        request_id=reserved[0],
        journal_checksum=digest(events),
        observation=observation.model_dump(mode="json"),
        observation_checksum=digest(observation),
        parent_result_checksum=digest(result),
    )
    if aggregate is not None:
        value["accepted_aggregation"] = aggregate
    if definition_increment:
        value["unreviewed_definition"] = definition_proof
    if ancestor_allowed:
        value.update(
            schema_version="alternative-observation-origin/1.1.0",
            origin_path=str(origin.output),
            current_parent_result_checksum=digest(parent_result),
            ancestor_observation_reused=origin is not parent,
        )
    return value


def proof(parent, target, *, method=METHOD, definition_increment=False):
    """Original current owners qualify this input transform, not medical accuracy."""
    from .native_fundus import recheck, records

    if method not in METHOD_VERSIONS:
        raise ValueError("alternative_view_method")
    if getattr(parent, "aggregation_baseline", None) is not None:
        ancestor, visited = parent, set()
        while ancestor is not None:
            if id(ancestor) in visited or len(visited) >= 16:
                raise ValueError("alternative_view_ancestor_depth_or_cycle")
            visited.add(id(ancestor))
            previous = ancestor.protocol.get("alternative_view") or {}
            if previous.get("target") == target and previous.get("method") == method:
                raise ValueError("alternative_view_same_method_already_attempted")
            ancestor = getattr(ancestor, "parent", None)
    elif parent.protocol.get("alternative_view"):
        raise ValueError("alternative_view_same_method_already_attempted")
    parent.base.check()
    rows = [r for r in parent.base.ledger if r["id"] == target]
    if (
        len(rows) != 1
        or rows[0].get("acquisition_method") != "vision"
        or not rows[0].get("acquisition_allowed", True)
    ):
        raise ValueError("alternative_view_visual_requirement")
    row = rows[0]
    component = next(c for c in parent.base.manifest.components if c.id == row["component"])
    if (
        component.modality != "color_fundus"
        or component.kind != "image"
        or len(component.payload_ids) != 1
    ):
        raise ValueError("alternative_view_single_color_fundus")
    native = getattr(getattr(parent.base, "active_owner", None), "native", None)
    matched = [r for r in records(native) if r["plan"]["component"] == component.id]
    if len(matched) != 1:
        raise ValueError("alternative_view_unique_native_localization")
    record = matched[0]
    recheck(parent.base, record)
    prediction = record["prediction"]
    if (
        (method == METHOD and record["measurement"]["status"] != "obtained")
        or prediction["transform"] != "RGB_bilinear_512_tensor_0_1_no_normalization"
        or prediction["analysis_shape"] != [512, 512]
    ):
        raise ValueError("alternative_view_localization_transform")
    parent.base.prepared.recheck()
    raw = dict(parent.base.prepared.payload_bytes)[component.payload_ids[0]]
    if checksum(raw) != prediction["input_checksum"]:
        raise ValueError("alternative_view_source_binding")
    old = original_observation(
        parent,
        target,
        ancestor_allowed=method == TILED_METHOD,
        definition_increment=definition_increment,
    )
    return dict(
        schema_version=(
            "alternative-observation-method/1.1.0"
            if method == TILED_METHOD
            else "alternative-observation-method/1.0.0"
        ),
        method=method,
        target=target,
        component=component.id,
        case_code=parent.base.binding["case_code"],
        input_checksum=parent.base.manifest.checksum,
        task_checksum=parent.base.task.checksum,
        source_checksum=checksum(raw),
        source_shape=prediction["input_shape"],
        source_payload=component.payload_ids[0],
        native_checksum=digest(record),
        native_bundle_key=record["bundle_key"],
        native_artifact=record["artifact"],
        analysis_shape=prediction["analysis_shape"],
        disc_bbox_xyxy=record["measurement"]["values"]["disc_bbox_xyxy"],
        original=old,
        original_image=old["observation"]["image"],
        definition_checksum=digest(row["definition"]),
        qualification="source_pixel_display_transform_only_not_clinical_model_validation",
        historical_observation_unchanged=True,
    )


def check(parent, plan):
    if plan.get("schema_version") == DEFINITION_VERSION:
        from .definition_alternative import check as check_definition

        return check_definition(parent, plan)
    selected = plan.get("acquisition_targets", [])
    aggregated = plan.get("schema_version") in AGGREGATE_VERSIONS
    if aggregated != (getattr(parent, "aggregation_baseline", None) is not None):
        raise ValueError("alternative_view_aggregation_plan_binding")
    if aggregated and (
        plan.get("acquisition_increment_only") is not True
        or plan.get("exact_request_budget") is not None
        or plan.get("max_new_requests") != 3
        or getattr(parent, "aggregation_baseline", None) is None
        or (plan.get("alternative_view") or {}).get("method") not in METHOD_VERSIONS
    ):
        raise ValueError("alternative_view_aggregation_plan_binding")
    if (
        plan.get("schema_version") not in (VERSIONS | AGGREGATE_VERSIONS)
        or plan.get("mode") != "acquisition"
        or len(selected) != 1
        or plan.get("seconds") != 5400
        or not 2 <= plan.get("max_new_requests", 0) <= 3
        or plan.get("new_public_search_allowed") is not False
        or plan.get("unknown_acquisition")
        or plan.get("review_clarification")
        or (
            plan["alternative_view"]
            != (
                proof(parent, selected[0], method=plan["alternative_view"]["method"])
                if aggregated
                else proof(parent, selected[0])
                if plan["schema_version"] == VERSION
                else proof(parent, selected[0], method=TILED_METHOD)
            )
        )
    ):
        raise ValueError("alternative_view_plan_binding")
    return selected


def prepare_material(parent, spec, output):
    """Finite local transform before claiming the successor; no API or GPU."""
    from packages.evaluation_preparation.case_execution import process

    root = Path(tempfile.mkdtemp(prefix=".alternative-view-", dir=Path(output).parent))
    source = root / "source.jpg"
    raw = dict(parent.base.prepared.payload_bytes)[spec["source_payload"]]
    if checksum(raw) != spec["source_checksum"]:
        raise ValueError("alternative_view_source_binding")
    source.write_bytes(raw)
    (root / "proof.json").write_text(json.dumps(spec, ensure_ascii=False, indent=2))
    process(
        [
            "/opt/conda/bin/python",
            "-m",
            "packages.multimodal_research.alternative_view_worker",
            str(source),
            str(root / "proof.json"),
            str(root / "result"),
        ],
        directory=root / "process",
        seconds=60,
    )
    metadata = json.loads((root / "result/view.json").read_bytes())
    image = (root / "result/view.png").read_bytes()
    if (
        metadata["source_checksum"] != spec["source_checksum"]
        or checksum(image) != metadata["checksum"]
    ):
        raise ValueError("alternative_view_worker_binding")
    # Request context must fit the existing ObservationInput contract exactly.
    from packages.grounding.models import canonical

    projected = {
        k: metadata[k]
        for k in (
            "modality",
            "indices",
            "available_members",
            "inspected_fraction",
            "transform",
            "limits",
        )
    }
    if len(canonical(projected).decode()) > 1000:
        raise ValueError("alternative_view_projection_bound")
    return root, metadata


def report_binding(campaign, report):
    owner = getattr(campaign.base, "alternative_view", None)
    if owner is None:
        return
    bundle = campaign.base.runtime.read_observation_bundle(owner.key)
    # Historical facts remain in the immutable protocol and their original run.
    # They are provenance, not current ImageBindings for display deduplication.
    projected = deepcopy(owner.spec)
    original = projected["original"]
    historical = original.pop("observation")
    if digest(historical) != original["observation_checksum"]:
        raise ValueError("alternative_report_historical_observation_binding")
    original["observation_location"] = dict(
        protocol_checksum=digest(campaign.protocol),
        pointer="/alternative_view/original/observation",
    )
    value = dict(
        schema_version="alternative-method-report/1.1.0",
        proof=projected,
        proof_checksum=digest(owner.spec),
        method=owner.spec["method"],
        metadata_checksum=owner.plan["alternative_view_metadata_checksum"],
        bundle_key=owner.key,
        artifacts=[a.model_dump(mode="json") for a in bundle or ()],
        medical_accuracy_claimed=False,
        historical_observation_replaced=False,
    )
    report["alternative_method"] = value
    for row in report["requirements"]:
        if row["id"] == owner.target:
            row["execution_binding"]["alternative_method"] = value
            row["presentation_sections"]["dependencies"] = row["execution_binding"]


class AlternativeView:
    """Prepared transform with original Runtime artifact registration and readback."""

    def __init__(self, parent, plan, output):
        self.parent, self.plan = parent, plan
        self.spec = plan["alternative_view"]
        self.output = Path(output) / "alternative-view"
        self.target = self.spec["target"]
        self.key = identity("TASK", "alternative-view:" + str(Path(output).absolute()))

    def read_material(self):
        base = self.parent.base
        base.check()
        paths = [self.output / "view.json", self.output / "view.png"]
        if any(p.is_symlink() for path in paths for p in (path, *path.parents)):
            raise ValueError("alternative_view_symlink")
        metadata = json.loads(paths[0].read_bytes())
        raw = paths[1].read_bytes()
        if (
            digest(metadata) != self.plan["alternative_view_metadata_checksum"]
            or checksum(raw) != metadata["checksum"]
            or len(raw) > 4194304
            or metadata["method"] != self.spec["method"]
            or self.spec["method"] not in METHOD_VERSIONS
            or metadata["source_checksum"] != self.spec["source_checksum"]
            or metadata["native_checksum"] != self.spec["native_checksum"]
        ):
            raise ValueError("alternative_view_material_changed")
        return raw, metadata

    def ensure(self):
        """Called under original execution lease/guard before any new Vision request."""
        base = self.parent.base
        raw, metadata = self.read_material()
        saved = base.runtime.read_observation_bundle(self.key)
        if saved:
            self.view()
            return
        uri = base.objects.promote(base.objects.stage(raw), checksum(raw))
        artifact = self.artifact(uri, raw, metadata)
        base.runtime.register_observation_bundle((artifact,), self.key)
        self.view()

    def artifact(self, uri, raw, metadata):
        base = self.parent.base
        return Artifact.model_validate(
            dict(
                schema_version="1.0",
                contract_type="Artifact",
                asset_code=identity("ASSET", self.key + checksum(raw)),
                artifact_type_code="ARTIFACT:RESEARCH_ALTERNATIVE_VIEW",
                uri=uri,
                checksum=checksum(raw),
                media_type="image/png",
                qc_status="UNASSESSED",
                dimensions=[metadata["height"], metadata["width"]],
                measurements=[],
                derived_from=[base.root.asset_code, self.spec["native_artifact"]["asset_code"]],
                research_only=True,
                scope=dict(scope_type="CASE", case_code=self.spec["case_code"]),
                laterality=self.spec["native_artifact"]["laterality"],
                coordinate_space="PIXEL_2D",
                unit=None,
                extensions=dict(
                    output_port="alternative_view",
                    method=self.spec["method"],
                    proof_checksum=digest(self.spec),
                    metadata=metadata,
                ),
                provenance=dict(
                    actor="TOOL_AGENT",
                    created_at=self.plan["started_at"],
                    generator_version="alternative-view-1.0.0",
                    source_codes=[base.root.asset_code, self.spec["native_artifact"]["asset_code"]],
                    input_checksums=[base.root.checksum, self.spec["native_artifact"]["checksum"]],
                ),
            )
        )

    def view(self):
        raw, metadata = self.read_material()
        base = self.parent.base
        saved = base.runtime.read_observation_bundle(self.key)
        if not saved or len(saved) != 1:
            raise ValueError("alternative_view_registered_artifact_required")
        artifact = saved[0]
        if (
            artifact != self.artifact(artifact.uri, raw, metadata)
            or base.objects.read_bounded(artifact.uri, 4194304) != raw
        ):
            raise ValueError("alternative_view_artifact_binding")
        image = ImageBinding(
            reference=ref("image", "ASSET:VIEW:" + digest(self.target)[7:23], metadata["checksum"]),
            media_type="image/png",
            width=metadata["width"],
            height=metadata["height"],
            coordinate_system="pixel_xy_top_left",
            location=(0, 0, metadata["width"], metadata["height"]),
            preprocessing=ref("preprocessing", "POLICY:ALTERNATIVE_VIEW", digest(metadata)),
        )
        return image, raw, metadata
