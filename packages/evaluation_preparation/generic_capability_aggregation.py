"""Full current facts folded by the original accepted-Review aggregation owner."""

from functools import partial
from pathlib import Path

from packages.evidence_details.accepted_aggregation import AcceptedOriginReader
from packages.grounding.models import digest

from .generic_capability_projection import GenericCapabilityPhase
from .generic_capability_successor import CapabilitySuccessor
from .generic_definition_aggregation import DefinitionCommitOriginReader
from .generic_definition_execution import restore
from .generic_input import directory


def restore_plan(plan):
    history = getattr(plan, "history_software", None)
    options = dict(history_software=history) if history is not None else {}
    campaign, result = restore(Path(plan.facts).absolute(), **options)
    if type(campaign) is not CapabilitySuccessor:
        raise ValueError("generic_capability_aggregate_owner_required")
    exact_origins(plan, campaign)
    return (
        campaign,
        result,
        partial(
            GenericCapabilityPhase, selection_owner=campaign.selection_owner, new_items_only=False
        ),
    )


def exact_origins(plan, campaign):
    origins = [Path(item.campaign).absolute() for item in plan.origins]
    if origins != [directory(campaign.spec["key"]), directory(campaign.review_key)] or any(
        item.local_finalization for item in plan.origins
    ):
        raise ValueError("generic_capability_aggregate_exact_origins")


def select_readers(plan, readers, campaign):
    if type(campaign) is not CapabilitySuccessor or len(readers) != 2:
        raise ValueError("generic_capability_aggregate_owner_required")
    exact_origins(plan, campaign)
    if (
        type(readers[0]) is not AcceptedOriginReader
        or type(readers[1]) is not CapabilityCommitOriginReader
        or readers[1].campaign.review_key != campaign.review_key
    ):
        raise ValueError("generic_capability_aggregate_origin_readers")
    # There is no definition reinterpretation here. Original compatibility and
    # duplicate-item checks retain all old judgments and select only new items.
    for reader in readers:
        if (
            reader.client_transition
            or reader.network_transition != bool(campaign.selection.get("route_transition"))
            or reader.reasoned_batch_resize
            or reader.confirmed_review_reader is not None
            or reader.reused_origin_new_items_only
            or reader.alias_manifests is not None
            or reader.retained_identity_aliases
        ):
            raise ValueError("generic_capability_aggregate_no_combined_transition")
    return readers


class CapabilityCommitOriginReader(DefinitionCommitOriginReader):
    """Reuse exact original committed-cancellation proof for this typed successor."""

    def __init__(self, *, campaign, **kwargs):
        if type(campaign) is not CapabilitySuccessor:
            raise ValueError("generic_capability_committed_owner_required")
        AcceptedOriginReader.__init__(self, **kwargs)
        self.campaign = campaign


def prepare_plan(key, output):
    from apps.vectorengine_single_case import write_once
    from packages.multimodal_research.accepted_aggregation import (
        GenericCapabilityAggregationPlan,
        OriginPath,
        checked_path,
    )

    campaign, _ = restore(directory(key))
    if type(campaign) is not CapabilitySuccessor:
        raise ValueError("generic_capability_aggregate_owner_required")
    original, successor = directory(campaign.spec["key"]), directory(campaign.review_key)
    destination = checked_path(output)
    if any(destination == p or p in destination.parents for p in (original, successor)):
        raise ValueError("generic_capability_plan_overlaps_origin")
    plan = GenericCapabilityAggregationPlan(
        facts=str(successor),
        origins=(OriginPath(campaign=str(original)), OriginPath(campaign=str(successor))),
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_once(destination, plan)
    return dict(plan=str(destination), plan_checksum=digest(plan), new_api_requests=0)
