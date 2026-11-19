"""Bounded demand increment over an original completed Generic campaign.

This composition reuses the original executor and accepted observation readers.
The original completion is historical fact, not a renewed dispatch permission.
The public preflight port creates no claim, request, or execution window.
"""

import json
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from functools import partial
from types import SimpleNamespace

from packages.grounding.models import digest
from packages.llm_provider.provider import ProviderError

from .comprehensive_campaign import ComprehensiveCampaign
from .generic_capability_projection import (
    GenericCapabilityOwner,
    GenericCapabilityPhase,
    _current_selection_reads,
)
from .generic_definition_successor import CLAIM, completed_history
from .generic_input import directory
from .generic_pending_review import restore_campaign
from .generic_review import GenericReview

VERSION = "generic-capability-successor/1.0.0"
ROUTED_VERSION = "generic-capability-successor/1.1.0"
CONTINUATION_VERSION = "generic-capability-successor/1.2.0"
REVIEW_CONTINUATION_VERSION = "generic-capability-successor/1.3.0"
CONTINUATION_VERSIONS = {CONTINUATION_VERSION, REVIEW_CONTINUATION_VERSION}
VERSIONS = {VERSION, ROUTED_VERSION, *CONTINUATION_VERSIONS}
CONTINUATION_CLAIM = "capability-continuation-successor.json"
COMPLETION_VERSION = "capability-acquisition-finalization/1.0.0"


def route_configuration(config, path):
    """Explicit public route replacement, preserving every other provider field."""
    if path is None:
        return config, None
    from packages.llm_provider.config import with_replacement_network_route
    from packages.llm_provider.network_route import load_route

    route = load_route(path)
    current = with_replacement_network_route(config, route)
    return current, dict(
        schema_version="generic-capability-network-transition/1.0.0",
        path=str(path),
        origin_checksum=digest(config),
        route_checksum=digest(route),
        provider_checksum=digest(current),
    )


def request_limits(prior, targets, config):
    """Count new visual work plus its Review; never erase original reservations."""
    if type(prior) is not int or not 0 <= prior <= 40:
        raise ValueError("generic_capability_original_accounting")
    if not targets or len(targets) != len(set(targets)) or config.transport.max_retries != 0:
        raise ValueError("generic_capability_targets_or_retry")
    batch_size = config.review_batching.batch_size
    review = (len(targets) + batch_size - 1) // batch_size
    cap = min(12, 40 - prior)
    if len(targets) + review > cap:
        raise ValueError("generic_capability_successor_budget")
    return dict(
        max_new_requests=cap,
        max_cumulative_requests=40,
        expected_vision_requests=len(targets),
        minimum_review_requests=review,
        seconds=5400,
        reserve_review_seconds=max(720, review * 240),
        reserve_close_seconds=300,
    )


def origin(key, *, own_claim=None, network_route=None):
    campaign, _ = restore_campaign(directory(key))
    owner = GenericCapabilityOwner(campaign, own_claim=own_claim)
    seed, _ = campaign.base.project(
        run_id="RUN:CAPABILITY_SUCCESSOR_PREVIEW", config=campaign.config
    )
    _, projection = owner.project(seed)
    history = owner.snapshot["original_history"]
    limits = request_limits(
        history["reservations"], projection["dispatch_targets"], campaign.config
    )
    execution_config, route = route_configuration(campaign.config, network_route)
    return SimpleNamespace(
        campaign=campaign,
        owner=owner,
        history=history,
        projection=projection,
        limits=limits,
        execution_config=execution_config,
        route_transition=route,
    )


def selection_for(checked, key):
    if key == checked.campaign.spec["key"]:
        raise ValueError("generic_capability_distinct_successor_identity")
    directory(key)  # Original bounded path/key validation.
    return dict(
        schema_version=(
            REVIEW_CONTINUATION_VERSION
            if getattr(checked, "continuation", {}).get("schema_version")
            == "generic-capability-continuation-binding/1.1.0"
            else CONTINUATION_VERSION
            if getattr(checked, "continuation", None)
            else ROUTED_VERSION
            if getattr(checked, "route_transition", None)
            else VERSION
        ),
        **(
            {"continuation": checked.continuation} if getattr(checked, "continuation", None) else {}
        ),
        **(
            {"route_transition": checked.route_transition}
            if getattr(checked, "route_transition", None)
            else {}
        ),
        key=key,
        parent_key=checked.campaign.spec["key"],
        original_history=checked.history,
        selection_checksum=digest(checked.owner.snapshot),
        correspondence_checksum=digest(checked.owner.policy),
        preview_projection=checked.projection,
        provider_checksum=digest(getattr(checked, "execution_config", checked.campaign.config)),
        **checked.limits,
        old_scores_modified=False,
        whole_case_aggregation_required=True,
        observation_scope="original_partial_vision_per_requirement",
        clinical_acceptance=False,
    )


def preflight(key, successor_key, network_route=None):
    from db.runtime.repository.case_pipeline import CasePipelineRepository

    from .generic_execution_lock import target

    root = directory(key)
    lock = "generic-case-execution:" + digest(str(root.resolve()))[7:]
    with CasePipelineRepository(target()).guarded_session("CASE:GENERIC_EXECUTION", lock) as guard:
        guard()
        checked = origin(key, **({"network_route": network_route} if network_route else {}))
        from packages.multimodal_research.unknown_acquisition import check_network

        check_network(getattr(checked, "execution_config", checked.campaign.config))
        selection = selection_for(checked, successor_key)
        if directory(successor_key).exists():
            raise ValueError("generic_capability_new_output_required")
        guard()
        return dict(
            status="original_case_and_increment_verified",
            selection=selection,
            api_requests=0,
            new_runtime_runs=0,
            execution_ready=True,
            execution_authorized=False,
            readiness_scope="current_input_selection_and_network_recheck_on_dispatch",
            remaining=[
                "original_exclusive_claim_and_new_frozen_window_on_run",
                "current_owner_checks_before_dispatch",
                "real_execution_and_whole_case_audit",
            ],
        )


class CapabilitySuccessor(GenericReview):
    """Same phase executor, explicit new demand selection and cumulative budget."""

    def __init__(self, checked, key):
        self.__dict__.update(vars(checked.campaign))
        self.config = getattr(checked, "execution_config", checked.campaign.config)
        self.checked, self.acquisition = checked, checked.campaign
        self.review_key = key
        self.selection = selection_for(checked, key)
        self.selection_owner = checked.owner
        self.selection_owner.own_claim = self.root_claim()
        self.output = directory(key) / "evidence"
        self.output.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.version = self.selection["schema_version"] + ":" + key
        self.export_suffix = "_能力驱动需求增量_" + key
        self.max_new_requests = self.selection["max_new_requests"]
        self.visual_targets = tuple(
            self.selection.get("continuation", {}).get(
                "dispatch_targets", checked.projection["dispatch_targets"]
            )
        )
        self.max_vision_requests = len(self.visual_targets)
        self.duration_seconds = self.selection["seconds"]
        additions = set(checked.projection["added_requirement_ids"])
        self.task_plan = list(self.task_plan) + [
            dict(
                id=row["id"],
                action="observe_partial"
                if row["id"] in self.visual_targets
                else "record_input_gap",
                reason="current_capability_selection",
                sources=row["definition"]["source_refs"],
            )
            for row in self.selection_owner.snapshot["selected_requirements"]
            if row["id"] in additions
        ]
        self.pending_review_proof = None
        self.definition_keys = ()  # This successor does not revise original definitions.
        self.structured_completion = True
        self.phase_class = partial(
            GenericCapabilityPhase, selection_owner=self.selection_owner, new_items_only=True
        )
        # The outer credential/Review guard uses the final-review window. Only
        # explicit acquisition calls temporarily select the earlier cutoff.
        self._capability_role = "review"

    def root_claim(self):
        return self.selection.get("continuation", {}).get("root_claim", self.selection)

    def prior(self):
        continuation = self.selection.get("continuation")
        if continuation is not None:
            from .generic_capability_recovery import snapshot

            # snapshot obtains the current completed-root proof through the
            # original selection owner; do not reconstruct it a second time.
            observed = snapshot(self.checked.predecessor)
            if (
                observed != continuation["inventory"]
                or self.checked.predecessor.checked.history != self.checked.history
            ):
                raise ValueError("generic_capability_continuation_original_changed")
            return dict(
                reservations=observed["cumulative_requests"],
                completed_origin=self.checked.history,
                predecessor_inventory_checksum=digest(observed),
            )
        current = completed_history(self.acquisition, own_claim=self.root_claim())
        if current != self.checked.history:
            raise ValueError("generic_capability_origin_history_changed")
        return dict(reservations=current["reservations"], completed_origin=current)

    def paths(self):
        return [self.phase_path(t) for t in self.visual_targets] + [self.phase_path(None)]

    def phase_path(self, target):
        if target is not None and target not in self.visual_targets:
            raise ValueError("generic_capability_unselected_phase")
        return self.output / ("vision-" + digest(target)[7:19] if target is not None else "review")

    def request_key(self, target):
        self.phase_path(target)
        return (
            "fv:capability:"
            + digest(dict(parent=self.spec["key"], successor=self.review_key, target=target))[7:]
        )

    def observation_projector(self):
        def project(original_phase, run_id):
            phase = GenericCapabilityPhase(
                self.base, selection_owner=self.selection_owner, target=original_phase.target
            )
            value = phase.project(run_id=run_id, config=self.config)
            phase.verify(value[0])
            return value

        return project

    def accepted_observations(self):
        from packages.evidence_details.accepted_read_scope import historical_fact_read_scope

        with historical_fact_read_scope(), _current_selection_reads():
            predecessor = getattr(self.checked, "predecessor", self.acquisition)
            previous = predecessor.accepted_observations()
            fresh = ComprehensiveCampaign.accepted_observations(self)
            if set(previous) & set(fresh):
                raise ValueError("generic_capability_duplicate_observation")
            return dict(previous, **fresh)

    def freeze_observations(self):
        from packages.multimodal_research.implementation import freeze

        value = dict(
            original=self.acquisition.freeze_observations(),
            selection=self.selection,
            current_capability=digest(self.selection_owner.snapshot),
            correspondence=digest(self.selection_owner.policy),
            capability_implementation=freeze(),
        )
        path = self.output / "protocol.json"
        previous = json.loads(path.read_bytes()) if path.is_file() else None
        completion = (previous or {}).get("observations", {}).get("completion_protocol")
        if completion not in {None, COMPLETION_VERSION}:
            raise ValueError("generic_capability_completion_version")
        if previous is None or completion is not None:
            value["completion_protocol"] = COMPLETION_VERSION
        return value

    def execution_session(self):
        from db.runtime.repository.case_pipeline import CasePipelineRepository

        return CasePipelineRepository(self.target).session(
            self.base.binding["case_code"], "generic-capability-increment:" + self.spec["key"]
        )

    def work_deadline(self):
        reserve = self.selection["reserve_close_seconds"]
        if self._capability_role != "review":
            reserve += self.selection["reserve_review_seconds"]
        return datetime.fromisoformat(self.protocol["deadline"]) - timedelta(seconds=reserve)

    def guard(self, *, dispatch=False):
        if getattr(self, "_capability_history_only", False):
            raise ValueError("generic_capability_history_cannot_execute")
        lease = getattr(self, "_parent_execution_guard", None)
        if lease is None:
            raise ValueError("generic_capability_execution_lease_required")
        lease()
        claims = [
            (directory(self.spec["key"]) / CLAIM, self.root_claim()),
            (directory(self.review_key) / "input.json", self.selection),
        ]
        continuation = self.selection.get("continuation")
        if continuation is not None:
            claims.append(
                (directory(continuation["predecessor_key"]) / CONTINUATION_CLAIM, self.selection)
            )
        for path, expected in claims:
            if not path.is_file() or json.loads(path.read_bytes()) != expected:
                raise ValueError("generic_capability_current_claim_binding")
        if not (directory(self.review_key) / "started.json").is_file():
            raise ValueError("generic_capability_start_intent_required")
        if datetime.now(UTC) >= self.work_deadline():
            raise ProviderError("deadline")
        super().guard(dispatch=dispatch)
        lease()

    @contextmanager
    def phase_window(self, target):
        previous = self._capability_role
        self._capability_role = "vision" if target is not None else "review"
        try:
            yield
        finally:
            self._capability_role = previous

    def execute_phase(self, target=None):
        if getattr(self, "_capability_history_only", False):
            raise ValueError("generic_capability_history_cannot_execute")
        self.phase_path(target)
        with self.phase_window(target):
            return super().execute_phase(target)

    def before_review(self):
        self.selection_owner.current()
        finalization = self.protocol.get("observations", {}).get("completion_protocol")
        for target in self.visual_targets:
            # The original executor reads durable phase results before any new
            # guard/dispatch. A saved failure remains a failure, never unsent.
            try:
                result = self.execute_phase(target)
            except ProviderError as exc:
                if exc.code != "deadline" or finalization != COMPLETION_VERSION:
                    raise
                self.close_acquisition_window(target)
                return
            print(
                json.dumps(
                    dict(target=target, accepted=result["accepted"], error=result.get("error")),
                    ensure_ascii=False,
                ),
                flush=True,
            )
            if result.get("error") == "deadline" and finalization == COMPLETION_VERSION:
                self.close_acquisition_window(target)
                return

    def close_acquisition_window(self, target):
        """Only the acquisition cutoff may fall through to the reserved Review window."""
        from apps.vectorengine_single_case import write_once

        from .generic_capability_recovery import phase_inventory

        with self.phase_window(target):
            if datetime.now(UTC) < self.work_deadline():
                raise ProviderError("deadline")  # A different phase deadline is not this cutoff.
        with self.phase_window(None):
            self.guard()  # Original lease, claim, cancellation and cumulative budget.
            rows = [phase_inventory(self, item) for item in self.visual_targets]
            if any(row["status"] == "unknown_requires_explicit_resolution" for row in rows):
                raise ValueError("generic_capability_cutoff_unknown_requires_recovery")
            self.guard()
        write_once(
            self.output / "acquisition-cutoff.json",
            dict(
                schema_version=COMPLETION_VERSION,
                protocol_checksum=digest(self.protocol),
                cause="acquisition_deadline_reached_review_window_retained",
                trigger_target=target,
                phases=rows,
                deferred_targets=[row["target"] for row in rows if not row.get("accepted")],
                new_acquisition_dispatch_authorized=False,
                whole_increment_complete=False,
            ),
        )

    def phase_completed(self, phase):
        path = self.output / "acquisition-cutoff.json"
        if path.is_file():
            value = json.loads(path.read_bytes())
            if value.get("schema_version") != COMPLETION_VERSION or value.get(
                "protocol_checksum"
            ) != digest(self.protocol):
                raise ValueError("generic_capability_cutoff_binding")
            return False  # Completed partial Review does not complete the deferred acquisitions.
        return super().phase_completed(phase)

    def archive_completed(self, result):
        return result  # An increment is not an independently completed case.


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key", required=True)
    parser.add_argument("--successor-key", required=True)
    parser.add_argument("--preflight", required=True, action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(preflight(args.key, args.successor_key), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
