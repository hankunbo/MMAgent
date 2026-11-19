"""11 handoff values only. There is no extension dispatcher or worker here."""

from typing import Literal

from packages.grounding.models import Checksum, Value, digest, validate
from packages.planner_control.models import Reference

from .models import ToolRequirement


class ExtensionFixture(Value):
    schema_version: Literal["extension-event-fixture/1.0.0"] = "extension-event-fixture/1.0.0"
    synthetic: Literal[True] = True
    requirement_checksum: Checksum
    outcome: Literal["failed", "qualified_for_future_run"]
    future_catalog: Reference | None = None


def replay_prerequisites(requirement, event, current_binding):
    requirement = validate(ToolRequirement, requirement)
    event = validate(ExtensionFixture, event)
    if event.requirement_checksum != digest(requirement) or (
        requirement.binding_checksum != digest(current_binding)
        or requirement.origin_run_id != current_binding.run_id
        or requirement.catalog != next(r for r in current_binding.locks if r.role == "catalog")
    ):
        raise ValueError("extension_fixture_origin_binding")
    if event.outcome == "qualified_for_future_run" and (
        event.future_catalog is None
        or event.future_catalog.role != "catalog"
        or event.future_catalog == requirement.catalog
    ):
        raise ValueError("new_catalog_required")
    return dict(
        schema_version="future-replay-prerequisites/1.0.0",
        synthetic=True,
        original_catalog=requirement.catalog.model_dump(mode="json"),
        outcome=event.outcome,
        current_run_changed=False,
        created_run=None,
        future_catalog=event.future_catalog.model_dump(mode="json")
        if event.future_catalog
        else None,
        required=(
            "new_run_id",
            "new_version_locks",
            "current_sources",
            "05_grounding",
            "06_admission",
            "explicit_replay_policy",
        )
        if event.outcome == "qualified_for_future_run"
        else (),
    )


def requirement_key(requirement):
    return (
        "requirement:"
        + digest(
            dict(
                run=requirement.origin_run_id,
                binding=requirement.binding_checksum,
                catalog=requirement.catalog,
                gap=requirement.gap,
            )
        )[7:]
    )
