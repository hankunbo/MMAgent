"""Explicit, pinned original-component selection for the existing case entry.

Configuration locates owners; it never supplies an observation or grants current
qualification. Registration is read-only and producer/current-result checks stay
with the original Runtime/06 owners.
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import Field

from packages.grounding.models import Checksum, Code, Value, digest, validate
from packages.grounding.review_files import ReviewedFile
from packages.integrations.medgraph.resources import LocalResourcePolicy
from packages.object_store import checksum


class ComponentConsumer(Value):
    schema_version: Literal["component-consumer-selection/1.0.0"]
    installation: ReviewedFile
    source_selection: ReviewedFile
    resources: LocalResourcePolicy
    producer_run: Code
    request_key: str = Field(min_length=1, max_length=256)
    execution_key: str = Field(min_length=1, max_length=256)
    registration_key: str = Field(min_length=1, max_length=256)
    component_id: str = Field(min_length=1, max_length=256)
    target: str = Field(min_length=1, max_length=256)
    input_checksum: Checksum
    task_checksum: Checksum
    output_store: str = Field(min_length=1, max_length=1024)


def public_file(record, *, limit):
    record = validate(ReviewedFile, record)
    path = Path(record.path)
    if (
        not path.is_absolute()
        or any(p.is_symlink() for p in (path, *path.parents))
        or not path.is_file()
        or record.byte_size > limit
        or path.stat().st_size != record.byte_size
    ):
        raise ValueError("component_consumer_file_binding")
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    if len(raw) != record.byte_size or checksum(raw) != record.checksum:
        raise ValueError("component_consumer_file_binding")
    return raw


def load(path, *, expected=None):
    path = Path(path).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)) or path.stat().st_size > 65536:
        raise ValueError("component_consumer_selection_file")
    with path.open("rb") as stream:
        raw = stream.read(65537)
    if len(raw) > 65536:
        raise ValueError("component_consumer_selection_file")
    value = ComponentConsumer.model_validate_json(raw)
    if expected is not None and digest(value) != expected:
        raise ValueError("component_consumer_selection_changed")
    public_file(value.installation, limit=8388608)
    public_file(value.source_selection, limit=1048576)
    return value


def registration(inputs, selection, manifest, task):
    selection = validate(ComponentConsumer, selection)
    if selection.input_checksum != manifest.checksum or selection.task_checksum != task.checksum:
        raise ValueError("component_consumer_input_or_task")
    if selection.component_id not in {c.id for c in manifest.components}:
        raise ValueError("component_consumer_component")
    result = inputs.repository.read_registration(selection.registration_key, manifest.checksum)
    if result.schema_version != "dataset-registration/4.0.0":
        raise ValueError("component_consumer_registration_version")
    # Exact producer/case/visible identity is additionally checked by the adapter.
    if inputs.read_visible(result.visible.case.case_code) != result.visible:
        raise ValueError("component_consumer_registration_changed")
    return result


def assemble(base, target, selection, *, guard):
    from apps.research_single_case import credential_boundary
    from db.runtime.repository.planner import PlannerRepository
    from db.runtime.repository.research import ResearchRunRepository
    from packages.grounding.models import Ref
    from packages.integrations.medgraph import (
        AdapterConfig,
        MedgraphCatalogReader,
        ProjectionPolicy,
        SnapshotStore,
    )
    from packages.integrations.medgraph.research import ReviewedCatalog
    from packages.integrations.medgraph.scoped_source import (
        DirectToolSelection,
        ExactPostgresSource,
    )
    from packages.research_runtime.component_deployment import assemble_component_consumer
    from packages.research_runtime.deployment import (
        CombinedResearchObjects,
        InstalledProfiles,
        existing_store,
        medgraph_readonly_dsn,
        read_installed_bundle,
    )
    from packages.tool_execution.fundus_bundle import FundusBundlePolicy, FundusBundleQCRegistry

    from .component_observations import ComponentObservations

    selection = validate(ComponentConsumer, selection)
    guard()
    source_spec = json.loads(public_file(selection.source_selection, limit=1048576))
    public_file(selection.installation, limit=8388608)
    value = read_installed_bundle(selection.installation)
    config = validate(AdapterConfig, source_spec["config"])
    boundary = credential_boundary()
    source = ExactPostgresSource(
        config,
        validate(DirectToolSelection, source_spec["selection"]),
        dsn_resolver=lambda: medgraph_readonly_dsn(boundary),
    )
    directory = base.output / "component-catalog-snapshots"
    if any(p.is_symlink() for p in (directory, *directory.parents)):
        raise ValueError("component_consumer_snapshot_path")
    directory.mkdir(exist_ok=True)
    legacy = MedgraphCatalogReader(source, SnapshotStore(directory), config, ProjectionPolicy())
    legacy.capture_snapshot()
    catalogs = ReviewedCatalog(
        legacy, validate(ReviewedFile, value["research_review"]), selection.resources
    )
    if catalogs.catalog.model_dump(mode="json") != value["catalog"]:
        raise ValueError("component_consumer_current_catalog")
    profiles = InstalledProfiles(selection.installation, catalogs)
    output = existing_store(selection.output_store, "research_output")
    objects = CombinedResearchObjects(base.inputs.visible_store, output)
    qc = FundusBundleQCRegistry(
        {validate(Ref, value["qc_ref"]): validate(FundusBundlePolicy, value["qc_policy"])}, objects
    )
    owner, _ = assemble_component_consumer(
        runtime=ResearchRunRepository(target),
        inputs=base.inputs,
        evidence=base.runtime,
        sessions=PlannerRepository(target),
        catalogs=catalogs,
        profiles=profiles,
        objects=objects,
        resources=selection.resources,
        installation=selection.installation,
        producer_run=selection.producer_run,
        request_key=selection.request_key,
        component_id=selection.component_id,
        qc=qc,
        clock=lambda: datetime.now(UTC),
    )
    guard()
    return ComponentObservations(
        base=base,
        owner=owner,
        execution_key=selection.execution_key,
        target=selection.target,
        guard=guard,
    )


def install_campaign(campaign, selection, reference):
    """Install before protocol freeze; cold assembly preserves the original clock.

    This callback is a read-only qualification check, not new-dispatch authority.
    Budget/cancel/time checks stay at the original campaign/stage dispatch port;
    injecting them into base.check would reject a last reserved request or local
    report recovery after cancellation/expiry.
    """
    base = campaign.base

    def guard():
        if load(reference["path"], expected=reference["checksum"]) != selection:
            raise ValueError("component_consumer_selection_changed")
        # No request/dispatch is possible here. Original component owners check
        # current source/input/method/QC; callers still need their dispatch guard.

    adapter = assemble(base, campaign.target, selection, guard=guard)
    if (base.output / "protocol.json").exists():
        adapter.restore()
    else:
        adapter.install()
    # Preserve the original selected candidates and remove only a confirmed fact.
    campaign.targets = tuple(t for t in campaign.targets if t not in base.observations)
    campaign.initial_targets = campaign.targets
    base.component_consumer_selection = dict(reference)
