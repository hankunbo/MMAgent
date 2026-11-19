"""Explicit age-only projection from the original adapter, not evaluator records.

This opt-in supplement does not change legacy visible manifests. Source labels,
notes, summaries, sex and race remain withheld. Missing/invalid age is a gap.
"""

import json
import math

from db.runtime.repository.case_pipeline import identity
from packages.contracts import Artifact
from packages.grounding.models import canonical, digest
from packages.object_store import checksum

from .prepared import capture

POLICY = "fairvlmed-public-age/1.0.0"


def age_observation(snapshot):
    """Shared original age-only rule; never reads note, CSV age, or label values."""
    if snapshot.dataset.code != "SOURCE:DATASET:FAIRVLMED" or not snapshot.runtime_profile:
        raise ValueError("public_context_dataset_not_supported")
    raw = snapshot.fields["npz"].get("age")
    value, reason = None, "age_not_provided"
    if raw is not None:
        try:
            if isinstance(raw, bool):
                raise ValueError
            parsed = float(raw)
            if not math.isfinite(parsed) or not 0 <= parsed <= 130:
                raise ValueError
            value, reason = parsed, None
        except (ValueError, TypeError):
            reason = "age_invalid_or_out_of_input_range"
    return dict(
        value=value,
        unit="year",
        status="obtained" if value is not None else "missing",
        reason=reason,
    )


def project(snapshot, visible):
    age = age_observation(snapshot)
    return dict(
        schema_version="dataset-public-context/1.0.0",
        projection_policy=POLICY,
        case_code=visible.case.case_code,
        visible_checksum=visible.checksum,
        input_fingerprint=snapshot.fingerprint(),
        image_checksum=visible.case.artifacts[0].checksum,
        age=age,
        source_field="age",
        source_files=[f.model_dump(mode="json") for f in snapshot.files],
        hidden_fields_used=False,
        interpretation=None,
    )


def register(service, sample, *, key, policy):
    if policy != POLICY:
        raise ValueError("public_context_policy_not_installed")
    # Original owner binds the sample to the existing case, including same-key conflict.
    snap = capture(sample, service.catalog)
    registered = service.register(sample, key=key)
    visible = service.read_visible(registered.visible.case.case_code)
    if visible != registered.visible:
        raise ValueError("public_context_visible_changed")
    value = project(snap, visible)
    raw = canonical(value)
    runtime = service.repository.runtime
    task_key = identity("TASK", "public-context:" + digest(value))
    asset_code = identity("ASSET", "public-context:" + digest(value))
    prior = runtime.read_case_artifact(visible.case.case_code, asset_code)
    store = service.visible_store
    if prior:
        if read(runtime, store, visible, prior) != value:
            raise ValueError("public_context_recovery_conflict")
        return prior, value
    uri = store.promote(store.stage(raw), checksum(raw))
    base = visible.case.artifacts[0].model_dump(mode="json")
    base.update(
        asset_code=asset_code,
        artifact_type_code="ARTIFACT:RESEARCH_INPUT_CONTEXT",
        uri=uri,
        checksum=checksum(raw),
        media_type="application/json",
        dimensions=[],
        unit=None,
        coordinate_space="NOT_APPLICABLE",
        derived_from=[],
        measurements=[],
        extensions=dict(projection_policy=POLICY, input_fingerprint=snap.fingerprint()),
        provenance=dict(
            actor="PLANNER",
            created_at=visible.case.provenance.created_at.isoformat(),
            generator_version="public-context-1.0.0",
            source_codes=[snap.dataset.code],
            input_checksums=[f.checksum for f in snap.files],
        ),
    )
    artifact = Artifact.model_validate_json(json.dumps(base))
    from db.runtime.repository.input_context import register as save_context

    save_context(runtime, artifact, task_key)
    if read(runtime, store, visible, artifact) != value:
        raise ValueError("public_context_readback")
    return artifact, value


def read(runtime, store, visible, artifact):
    artifact = Artifact.model_validate_json(artifact.model_dump_json())
    if runtime.read_case_artifact(visible.case.case_code, artifact.asset_code) != artifact:
        raise ValueError("public_context_artifact_binding")
    raw = store.read_bounded(artifact.uri, 65536)
    if checksum(raw) != artifact.checksum:
        raise ValueError("public_context_bytes_changed")
    value = json.loads(raw)
    if (
        value["schema_version"] != "dataset-public-context/1.0.0"
        or value["projection_policy"] != POLICY
        or value["case_code"] != visible.case.case_code
        or value["visible_checksum"] != visible.checksum
        or value["image_checksum"] != visible.case.artifacts[0].checksum
        or artifact.extensions.get("projection_policy") != POLICY
        or value["input_fingerprint"] != artifact.extensions.get("input_fingerprint")
    ):
        raise ValueError("public_context_scope")
    return value
