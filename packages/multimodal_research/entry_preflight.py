"""Command-bound local readiness; never a substitute for current owner authorization."""

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from packages.object_store import checksum

NATIVE_ALTERNATIVE = "unavailable-native-to-visible-description/1.0.0"


def native_readiness(*, allow_alternative=False):
    """Original installation check; an unavailable method is never authorized."""
    from apps import fundus_tool_registry as tool
    from packages.grounding.models import digest

    spec = json.loads(tool.LOCK.read_bytes())
    mismatches = []
    for path, expected in spec["resources"].items():
        file = Path(path)
        actual = checksum(file.read_bytes()) if file.is_file() else None
        if actual != expected:
            mismatches.append(dict(path=path, expected=expected, actual=actual))
    if mismatches:
        if not allow_alternative:
            raise LocalReadinessError("fundus_installation_resource_changed", mismatches)
        return dict(
            schema_version="native-method-readiness/1.0.0",
            method="vision",
            native_qualified=False,
            reason="fundus_installation_resource_changed",
            installation_checksum=digest(spec),
            resource_mismatches=mismatches,
            interpretation="visible_description_only_no_mask_measurement_or_numeric_cdr",
        )
    tool.load_verified()
    return dict(
        schema_version="native-method-readiness/1.0.0",
        method="native",
        native_qualified="pending_current_owner",
        reason=None,
        installation_checksum=digest(spec),
        resource_mismatches=[],
    )


class LocalReadinessError(ValueError):
    def __init__(self, code, details):
        super().__init__(code)
        self.code, self.details = code, details


def check_local_route(manifest, task, profile, consumer):
    if consumer is not None:
        if consumer.input_checksum != manifest.checksum or consumer.task_checksum != task.checksum:
            raise ValueError("component_consumer_input_or_task")
        components = {c.id: c for c in manifest.components}
        checks = [c for c in profile["checks"] if "active:" + c["id"] == consumer.target]
        component = components.get(consumer.component_id)
        if (
            len(checks) != 1
            or component is None
            or checks[0]["method"] != "disc_cup_measurement"
            or checks[0]["modality"] != component.modality
        ):
            raise ValueError("component_consumer_requirement_binding")
        return "confirmed_component_current_owner_recheck"
    modalities = {
        c.modality
        for c in manifest.components
        if c.kind == "image" and c.status in {"obtained", "partial"} and c.payload_ids
    }
    if not profile or not any(
        c["method"] == "disc_cup_measurement" and c["modality"] in modalities
        for c in profile["checks"]
    ):
        return "no_native_fundus_required"
    if profile.get("scope_selection_policy"):
        # Readiness and current Tool qualification are checked after the
        # accepted survey selects the method, before actual native dispatch.
        return "native_method_deferred_to_accepted_input_scope"
    decision = native_readiness(
        allow_alternative=profile.get("native_unavailable_policy") == NATIVE_ALTERNATIVE
    )
    return (
        "vision_alternative_native_unavailable"
        if decision["method"] == "vision"
        else "legacy_native_fundus_current_owner_recheck"
    )


def command(output, *, preflight_only=False):
    from .execution_lease import entry_session

    # Pure local checking must not acquire a database lease or require host flock.
    # Real assembly/dispatch below still holds the original exclusive owner lease.
    if preflight_only:
        return _command(output, preflight_only=True)
    if not Path(output).is_dir():
        return _command(output, preflight_only=preflight_only)
    with entry_session(output):
        return _command(output, preflight_only=preflight_only)


def _command(output, *, preflight_only=False):
    """Persist append-only assembly failures without claiming unknown DB work was absent."""
    from .entry import assemble_entry

    phase = "local_preflight"
    try:
        result = assemble_entry(output, preflight_only=True)
        if preflight_only:
            return result
        phase = "owner_assembly"
        campaign = assemble_entry(output)
    except Exception as error:
        # No exception text: drivers may include credentials or patient values.
        code = "entry_check_failed"
        if type(error) is ValueError and len(error.args) == 1 and isinstance(error.args[0], str):
            if re.fullmatch(r"[a-z][a-z0-9_]{2,127}", error.args[0]):
                code = error.args[0]
        failure = dict(
            schema_version="multimodal-entry-failure/1.0.0",
            status="entry_not_ready",
            complete=False,
            phase=phase,
            error_type=type(error).__name__,
            error_code=error.code if isinstance(error, LocalReadinessError) else code,
            resource_mismatches=error.details if isinstance(error, LocalReadinessError) else [],
            database_writes=0 if phase == "local_preflight" else None,
            api_requests=0 if phase == "local_preflight" else None,
            tool_dispatches=0 if phase == "local_preflight" else None,
            current_owner_qualification="not_confirmed",
            created_at=datetime.now(UTC).isoformat(),
        )
        root = Path(output).absolute()
        if root.is_dir() and not any(p.is_symlink() for p in (root, *root.parents)):
            path = root / ("entry-failure-" + uuid4().hex + ".json")
            with path.open("x") as stream:
                json.dump(failure, stream, ensure_ascii=False, indent=2)
            failure["failure_record"] = str(path)
        return failure
    # Campaign owns all execution, request accounting and post-score recovery.
    return campaign.execute()
