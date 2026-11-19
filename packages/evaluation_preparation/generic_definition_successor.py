"""Explicit definition revision over a completed Generic case's current facts.

Uses GenericReview, DefinitionExpansion and the original journals. A completed
old score is never relabelled pending. This first revision requires an original
case without another successor: foreign branches require their accounting owner,
not an operator-supplied request count. It grants no clinical publication.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

from packages.grounding.models import digest
from packages.research_runtime.agent_records import RuntimeAgentJournal

from .generic_aliases import GenericPhase
from .generic_campaign import GenericCampaign
from .generic_definition_facts import ConfirmedGenericMeasurements
from .generic_definition_input import GenericDefinitionInputView
from .generic_definition_stage import GenericDefinitionStage, binding
from .generic_input import directory
from .generic_pending_review import confirmed_campaign, restore_campaign
from .generic_review import GenericReview

VERSION = "generic-definition-review-successor/1.0.0"
CLAIM = "definition-review-successor.json"


def read(path):
    return json.loads(path.read_bytes())


def completed_history(campaign, *, own_claim=None):
    """Current exact original requests, Control and journals, including failures.

    A remote UNKNOWN reservation is charged exactly like every other reservation;
    no response is resent or classified as unsent here. This reader is not a new
    permission for the original phase. A missing registered request is an error.
    """
    if type(campaign) is not GenericCampaign:
        raise ValueError("generic_definition_original_case_owner_required")
    root = directory(campaign.spec["key"])
    if (root / "cancel-request.json").exists():
        raise ValueError("generic_definition_original_cancelled")
    for claim in root.glob("*successor*.json"):
        if claim.name != CLAIM or own_claim is None or read(claim) != own_claim:
            raise ValueError("generic_definition_other_successor_requires_accounting_owner")
    _, result = confirmed_campaign(campaign)
    if not result["complete"]:
        raise ValueError("generic_definition_requires_completed_origin")
    paths = tuple(campaign.paths())
    # The original owner enumerates phases; callers cannot omit a failed phase.
    if len({p.resolve() for p in paths}) != len(paths):
        raise ValueError("generic_definition_duplicate_phase_path")
    rows, runs = [], set()
    for path in paths:
        request_path = path / "request.json"
        if not request_path.exists():
            continue
        request = read(request_path)
        run_id = request["lock"]["run_id"]
        if run_id in runs:
            raise ValueError("generic_definition_duplicate_registered_run")
        runs.add(run_id)
        context = campaign.contexts.read_research_run(run_id)
        control = campaign.sessions.read(run_id)
        docs = campaign.documents(run_id)
        if (
            context is None
            or context.request.model_dump(mode="json") != request
            or context.run_id != run_id
            or control is None
            or control.phase != "stopped"
            or control.pending
            or control.cancelled
        ):
            raise ValueError("generic_definition_history_request_or_control")
        events = RuntimeAgentJournal(docs).read()
        rows.append(
            dict(
                path=str(path),
                run_id=run_id,
                request_checksum=digest(request),
                journal_checksum=digest(events),
                control_checksum=digest(control),
                reservations=sum(e.get("phase") == "reserved" for e in events),
            )
        )
    total = sum(r["reservations"] for r in rows)
    final = result["phases"][-1]
    docs = campaign.documents(final["run_id"])
    saved = docs.read("dynamic-case-report")
    if (
        final["run_id"] not in runs
        or total != result["new_requests"]
        or result["prior_graph_requests"] != 0
        or not saved
        or saved[0]["result"] != result
        or saved[1]["report_checksum"] != digest(saved[0])
    ):
        raise ValueError("generic_definition_history_accounting")
    return dict(
        schema_version="generic-completed-definition-origin/1.0.0",
        root_key=campaign.spec["key"],
        input_checksum=digest(campaign.spec),
        result_checksum=digest(result),
        report_checksum=digest(saved),
        original_complete=True,
        reservations=total,
        journals=rows,
        original_scores_relabelled=False,
    )


def origin(key, policy, *, own_claim=None):
    """Original owner eligibility; public policy never authorizes patient facts."""
    from packages.multimodal_research.definition_expansion import (
        GenericConfirmedIndexPolicy,
        GenericPublicInterpretationPolicy,
        GenericRegisteredMeasurementPolicy,
        read_expansion_policy,
    )

    selected = read_expansion_policy(policy)
    if type(selected) not in {
        GenericConfirmedIndexPolicy,
        GenericRegisteredMeasurementPolicy,
        GenericPublicInterpretationPolicy,
    }:
        raise ValueError("generic_definition_explicit_confirmed_policy")
    campaign, _ = restore_campaign(directory(key))
    history = completed_history(campaign, own_claim=own_claim)
    facts = ConfirmedGenericMeasurements(
        campaign,
        selected.confirmed_requirements,
        registered_measurement=type(selected) is GenericRegisteredMeasurementPolicy,
    )
    view = GenericDefinitionInputView(facts)
    frozen_binding = binding(view, policy)
    minimum = selected.reserve_review_requests + 1
    if type(selected) is GenericPublicInterpretationPolicy:
        minimum += selected.max_search_requests + selected.max_fetch_requests
    cap = min(10, 40 - history["reservations"])
    if cap < minimum or campaign.config.transport.max_retries != 0:
        raise ValueError("generic_definition_successor_budget_or_retry")
    return SimpleNamespace(
        campaign=campaign,
        facts=facts,
        view=view,
        policy=selected,
        history=history,
        binding=frozen_binding,
        max_new_requests=cap,
    )


def selection_for(checked, key):
    if key == checked.campaign.spec["key"]:
        raise ValueError("generic_definition_distinct_successor_identity")
    return dict(
        schema_version=VERSION,
        key=key,
        parent_key=checked.campaign.spec["key"],
        definition_policy=checked.policy.model_dump(mode="json"),
        definition_stage=checked.binding,
        original_history=checked.history,
        max_new_requests=checked.max_new_requests,
        max_cumulative_requests=40,
        seconds=5400,
        new_vision_requests=0,
        new_tool_executions=0,
        mode="new_interpretation_old_complete_score_retained",
        whole_case_aggregation_required=True,
    )


def preflight(key, review_key, policy_path):
    """Public terminal planning port; no claim, start, credentials or API call."""
    from db.runtime.repository.case_pipeline import CasePipelineRepository

    from .generic_execution_lock import target

    path = Path(policy_path).resolve()
    root = Path(__file__).resolve().parents[2]
    if not path.is_relative_to(root) or path.suffix != ".json":
        raise ValueError("generic_definition_public_policy_path")
    original = directory(key)
    lock_key = "generic-case-execution:" + digest(str(original.resolve()))[7:]
    with CasePipelineRepository(target()).guarded_session(
        "CASE:GENERIC_EXECUTION", lock_key
    ) as guard:
        guard()
        checked = origin(key, read(path))
        selection = selection_for(checked, review_key)
        if directory(review_key).exists():
            raise ValueError("generic_definition_new_output_required")
        guard()
        return dict(
            status="definition_input_and_original_history_verified",
            selection=selection,
            api_requests=0,
            new_case_created=False,
            execution_ready=False,
            remaining=[
                "execution_and_committed_result_restore_entry",
                "whole_case_definition_revision_selection",
                "persistent_gate",
                "production_installation",
            ],
        )


class DefinitionReview(GenericReview):
    """Same executor, new explicit eligibility and Source/Review budget paths."""

    def __init__(self, checked, key, *, sources):
        # Same composition pattern as AcquisitionSuccessor: keep original owners
        # and facts rather than calling GenericReview's pending-only constructor.
        self.__dict__.update(vars(checked.campaign))
        self.checked = checked
        self.acquisition = checked.campaign
        self.review_key = key
        self.selection = selection_for(checked, key)
        self.output = directory(key) / "evidence"
        self.output.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.version = VERSION + ":" + key
        self.export_suffix = "_定义修订评阅_" + key
        self.max_new_requests = checked.max_new_requests
        self.max_vision_requests = 0
        self.visual_targets = ()
        self.duration_seconds = self.selection["seconds"]
        self.pending_review_proof = None
        self.phase_class = GenericPhase
        self.sources = sources
        self.definition_stage = None
        self.definition_keys = None
        self.structured_completion = True

    def prior(self):
        current = completed_history(self.acquisition, own_claim=self.selection)
        if current != self.checked.history:
            raise ValueError("generic_definition_origin_history_changed")
        return dict(reservations=current["reservations"], completed_definition_origin=current)

    def paths(self):
        # The original Campaign.usage counts both stages from original journals.
        return [self.output / "definition", self.output / "review"]

    def usage(self):
        from .generic_definition_public import wire_usage

        # Public search/fetch reservations live in the original definition
        # operation journal; Source/Review live in the original agent journal.
        # Both consume this campaign's cap, including unconfirmed attempts.
        return super().usage() + wire_usage(self)

    def source_config(self):
        from packages.multimodal_research.definition_expansion import definition_config

        return definition_config(
            self.config,
            prompt_version=self.checked.policy.prompt_version,
            source_output_tokens=self.checked.policy.source_output_tokens,
        )

    def freeze_observations(self):
        return dict(
            acquisition=self.acquisition.freeze_observations(),
            observations=digest(self.accepted_observations()),
            definition_stage=self.checked.binding,
            source_provider_checksum=digest(self.source_config()),
        )

    def execution_session(self):
        from db.runtime.repository.case_pipeline import CasePipelineRepository

        return CasePipelineRepository(self.target).session(
            self.base.binding["case_code"], "generic-definition-review:" + self.spec["key"]
        )

    def guard(self, *, dispatch=False):
        lease = getattr(self, "_parent_execution_guard", None)
        if lease is None:
            raise ValueError("generic_definition_execution_lease_required")
        lease()
        if self.protocol["observations"]["source_provider_checksum"] != digest(
            self.source_config()
        ):
            raise ValueError("generic_definition_source_config_changed")
        super().guard(dispatch=dispatch)
        lease()

    def work_deadline(self):
        return datetime.fromisoformat(self.protocol["deadline"]) - timedelta(
            seconds=self.checked.policy.reserve_close_seconds
        )

    def before_review(self):
        self.prior()
        self.checked.view.check()
        if self.definition_stage is None:
            self.definition_stage = GenericDefinitionStage(
                self,
                self.checked.view,
                self.checked.policy.model_dump(mode="json"),
                sources=self.sources,
            )
        self.definition_stage.advance()
        self.bind_review_phase()

    def bind_review_phase(self):
        """No new Source request; use only current owner-adopted definitions."""
        from .generic_definition_projection import phase_type

        keys = self.definition_stage.review_keys()
        self.definition_keys = keys
        if keys:
            self.phase_class = phase_type(
                self.checked.facts, self.definition_stage.expansion.owner, keys
            )
        else:
            from .dynamic_stages import rebuild

            class NoAdoptionPhase(GenericPhase):
                def project_current(inner):
                    state = super().project_current()
                    policy = state.policy.model_copy(update=dict(items=()))
                    return rebuild(state, policy, state.material.observations)

            # Existing zero-evidence local closure; no fake Review, no new score.
            self.phase_class = NoAdoptionPhase

    def execute_phase(self, target=None):
        if target is not None or self.definition_keys is None:
            raise ValueError("generic_definition_review_requires_definition_stage")
        return super().execute_phase(target)

    def archive_completed(self, result):
        # A definition increment is not a whole-case Memory success.
        return result


def report_metadata(campaign, parent, result):
    """Metadata from persisted original projection, never a display inference."""
    from packages.multimodal_research.definition_expansion import GenericPublicInterpretationPolicy

    from .generic_definition_public import accounting

    if type(campaign) is not DefinitionReview or campaign.definition_keys is None:
        raise ValueError("generic_definition_report_owner_required")
    docs = campaign.documents(parent.material.scope.run_id)
    saved = docs.read("generic-definition-review-projection")
    if campaign.definition_keys:
        if (
            not saved
            or saved[0]["projected_state_checksum"] != digest(parent)
            or sorted(r["receipt_key"] for r in saved[0]["interpretation_revisions"])
            != sorted(campaign.definition_keys)
        ):
            raise ValueError("generic_definition_report_projection_binding")
        projection = saved[0]
    else:
        if saved is not None or parent.policy.items:
            raise ValueError("generic_definition_report_zero_increment_binding")
        projection = None
    expansion = campaign.definition_stage.expansion
    stage = expansion.read_revision()
    return dict(
        schema_version="generic-definition-review-report-scope/1.0.0",
        **(
            dict(public_request_accounting=accounting(campaign))
            if type(expansion.policy) is GenericPublicInterpretationPolicy
            else {}
        ),
        original_complete=True,
        original_result_checksum=campaign.checked.history["result_checksum"],
        original_report_checksum=campaign.checked.history["report_checksum"],
        original_history=campaign.checked.history,
        definition_stage_checksum=digest(stage),
        definition_projection=projection,
        interpretation_ids=sorted(i.id for i in parent.policy.items),
        new_requirements=[],
        new_observations=int(bool(projection and projection.get("observation_transition"))),
        new_tools=0,
        actual_new_requests=result["new_requests"],
        whole_case_aggregation_required=bool(campaign.definition_keys),
        historical_scores_modified=False,
        status="new_interpretation_increment"
        if campaign.definition_keys
        else "no_adopted_definition",
    )
