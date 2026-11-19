"""DefinitionExpansion over a new Generic successor's original Runtime budget.

This is a stage adapter, not a second case runner. The successor must already
freeze its exact definition binding and include this stage in its usage paths.
No acquisition, credential loading, grant renewal or historical mutation occurs
here. The existing Source/definition owner retains UNKNOWN and idempotency rules.
"""

import json
from datetime import datetime

from apps.vectorengine_single_case import write_once
from packages.evidence_reuse.definition_sources import KnowledgeDefinitionSources
from packages.grounding.models import canonical, digest
from packages.llm_provider.provider import RequestBudget, StructuredProvider
from packages.multimodal_research.definition_expansion import (
    DefinitionExpansion,
    GenericConfirmedIndexPolicy,
    GenericPublicInterpretationPolicy,
    GenericRegisteredMeasurementPolicy,
    definition_config,
    read_expansion_policy,
)
from packages.research_runtime.agent_records import RuntimeAgentJournal

from .generic_aliases import GenericPhase
from .generic_definition_input import GenericDefinitionInputView
from .generic_review import GenericReview

VERSION = "generic-confirmed-definition-stage/1.0.0"
INPUT_KEY = "generic-definition-stage-input"


def binding(view, policy):
    """Prepare public frozen inputs; this does not authorize execution."""
    if type(view) is not GenericDefinitionInputView:
        raise ValueError("generic_definition_stage_original_view_required")
    selected = read_expansion_policy(policy)
    view.check()
    return _view_binding(view, selected)


def _view_binding(view, selected):
    """Pure frozen identity projection; never a replacement for current checks."""
    if (
        type(selected)
        not in {
            GenericConfirmedIndexPolicy,
            GenericRegisteredMeasurementPolicy,
            GenericPublicInterpretationPolicy,
        }
        or bool(getattr(view.facts, "registered_measurement", False))
        != (type(selected) is GenericRegisteredMeasurementPolicy)
        or selected.confirmed_requirements != view.facts.selected
        or selected.public_proposition_checksum != view.public_goal_checksum
        or selected.public_target != view.task.target_code
    ):
        raise ValueError("generic_definition_stage_policy_binding")
    if type(selected) is GenericPublicInterpretationPolicy:
        from .generic_definition_public import public_root

        public_root(view, selected)
    return dict(
        schema_version=VERSION,
        policy_checksum=digest(selected),
        original_facts_checksum=view.facts.snapshot_checksum,
        original_rows_checksum=view.original_rows_checksum,
        input_checksum=view.manifest.checksum,
        public_proposition_checksum=view.public_goal_checksum,
        selected_ids=list(view.facts.selected),
        acquisition_requests=0,
        shared_campaign_accounting=True,
    )


class GenericDefinitionStage:
    """Narrow view consumed by the existing DefinitionExpansion implementation."""

    def __init__(self, campaign, view, policy, *, sources):
        if not isinstance(campaign, GenericReview) or type(view) is not GenericDefinitionInputView:
            raise ValueError("generic_definition_stage_original_campaign_required")
        if (
            campaign.acquisition is not view.facts.campaign
            or campaign.base is not campaign.acquisition.base
        ):
            raise ValueError("generic_definition_stage_original_input_binding")
        if not isinstance(sources, KnowledgeDefinitionSources):
            raise ValueError("generic_definition_stage_knowledge_owner_required")
        self.parent, self.base, self.sources = campaign, view, sources
        self.policy = read_expansion_policy(policy)
        self.frozen_binding = binding(view, policy)
        self.output = campaign.output
        self.path = self.output / "definition"
        self.protocol = json.loads(canonical(campaign.protocol))
        self.sessions = campaign.sessions
        self.campaign_id = campaign.review_key
        self.logical_campaign_id = campaign.spec["key"]
        self.targets = ()
        self.expansion = None
        self.documents = None
        self.budget = None
        self.public_connector = None
        self._check_binding()
        if self.parent.prior() != self.protocol["previous"]:
            raise ValueError("generic_definition_stage_frozen_accounting_binding")

    def _check_binding(self):
        from .generic_definition_public import check_sources

        paths = tuple(self.parent.paths())
        if (
            self.parent.protocol != self.protocol
            or self.protocol["observations"].get("definition_stage") != self.frozen_binding
            or paths.count(self.path) != 1
            or self.path == self.parent.phase_path(None)
            or self.parent.output != self.output
            or _view_binding(self.base, self.policy) != self.frozen_binding
            or self.sources.material_policy != self.policy.material_policy
            or self.sources.failure_policy != self.policy.search_failure_policy
            or self.sources.passage_plans != self.policy.passage_plans
        ):
            raise ValueError("generic_definition_stage_frozen_accounting_binding")
        check_sources(self)

    def usage(self):
        # The original campaign reads request.json -> original Runtime journal.
        # It must include Source exactly once, including cold recovery.
        self._check_binding()
        return self.parent.usage()

    def previous_attempts(self):
        self._check_binding()
        return [dict(requests=self.protocol["previous"]["reservations"])]

    def guard(self, *, dispatch=False):
        self.guard_definition_materials(dispatch=dispatch)

    def guard_definition_materials(self, *, dispatch=False):
        self._check_binding()
        self.parent.guard(dispatch=dispatch)
        self.base.check()

    @staticmethod
    def select_targets(base, available):
        # Current policy only reinterprets already confirmed observations.
        if type(base) is not GenericDefinitionInputView or available:
            raise ValueError("generic_definition_stage_no_new_acquisition")
        return ()

    def initialize(self, *, provider_factory=None, restore_only=False):
        """Persist registration before constructing a provider or reserving work."""
        self._check_binding()
        if self.expansion is not None:
            if not restore_only and self.expansion.provider is None:
                raise ValueError("generic_definition_stage_restore_is_read_only")
            return self.expansion
        # Restore never constructs a provider; original current qualification
        # and control checks still apply when reading the definition revision.
        if not restore_only:
            self.guard()
        evidence = GenericPhase(
            self.parent.base,
            observation_reader=self.parent.accepted_observations,
            model_reader=self.parent.read_model,
        )
        key = "fv:definition:" + digest((self.logical_campaign_id, self.campaign_id))[7:]
        if restore_only:
            request_path = self.path / "request.json"
            if not request_path.is_file():
                raise ValueError("generic_definition_stage_no_saved_stage")
            request = json.loads(request_path.read_bytes())
            context = self.parent.contexts.read_research_run(request["lock"]["run_id"])
            if context is None or context.request.model_dump(mode="json") != request:
                raise ValueError("generic_definition_stage_original_request_changed")
            state, _ = evidence.project(run_id=context.run_id, config=self.parent.config)
            evidence.verify(state)
        else:
            context, state = self.parent.register_phase(evidence, key)
        docs = self.parent.documents(context.run_id)
        record = dict(
            schema_version=VERSION,
            binding=self.frozen_binding,
            request_checksum=digest(context.request),
            state_checksum=digest(state),
            protocol_checksum=digest(self.protocol),
        )
        saved = docs.read(INPUT_KEY)
        if saved is not None and saved != (record,):
            raise ValueError("generic_definition_stage_saved_binding_changed")
        if restore_only and saved is None:
            raise ValueError("generic_definition_stage_no_saved_stage")
        if saved is not None and docs.read("dynamic-phase-input") != (
            state.model_dump(mode="json"),
        ):
            raise ValueError("generic_definition_stage_original_material_changed")
        if saved is None:
            with docs.session():
                docs.write(INPUT_KEY, (record,))
                docs.write("dynamic-phase-input", (state.model_dump(mode="json"),))
        self.path.mkdir(mode=0o700, parents=True, exist_ok=True)
        write_once(self.path / "request.json", context.request.model_dump(mode="json"))
        write_once(self.path / "stage-binding.json", record)
        self.documents = docs
        stage = self

        class SharedBudget(RequestBudget):
            def check_control(self):
                stage.expansion.control()

            def before_reserve(self, config_checksum):
                stage.expansion.guard()

            def before_wire(self, *args):
                stage.guard_definition_materials(dispatch=True)

        self.budget = SharedBudget(
            run_id=context.run_id,
            deadline=datetime.fromisoformat(self.protocol["deadline"]),
            max_requests=self.parent.config.max_requests,
            journal=RuntimeAgentJournal(docs),
        )
        config = definition_config(
            self.parent.config,
            prompt_version=self.policy.prompt_version,
            source_output_tokens=self.policy.source_output_tokens,
        )
        provider = None
        if not restore_only:
            provider = (
                provider_factory(config)
                if provider_factory is not None
                else StructuredProvider(config, live_opt_in=True)
            )
        self.expansion = DefinitionExpansion(
            self,
            documents=docs,
            sources=self.sources,
            provider=provider,
            budget=self.budget,
            scope=state.material.scope,
            policy=self.policy.model_dump(mode="json"),
        )
        if not restore_only and type(self.policy) is GenericPublicInterpretationPolicy:
            from .generic_definition_public import GenericPublicDefinitionConnector

            self.public_connector = GenericPublicDefinitionConnector(
                self.sources.repository, self.expansion, self.policy.public_topic_sets
            )
            self.sources.search_public = self.public_connector.search
            self.sources.fetch_hit = self.public_connector.fetch
        self._check_binding()
        return self.expansion

    def advance(self, *, provider_factory=None, until=None):
        expansion = self.initialize(provider_factory=provider_factory)
        return expansion.advance(until=until)

    def restore(self, *, report_only=False):
        """Saved-source revalidation for local reporting never grants dispatch."""
        expansion = self.initialize(restore_only=True)
        revision = expansion.read_revision() if report_only else expansion.current()
        return expansion.install(revision)

    def review_keys(self):
        """Only owner-accepted, current, effective interpretations enter Review."""
        if self.expansion is None:
            raise ValueError("generic_definition_stage_not_initialized")
        self._check_binding()
        self.base.check()
        keys = []
        for target, adoption in self.base.adoptions.items():
            receipt, projection = self.expansion.owner.read_confirmed_projection(adoption["key"])
            if projection["id"] != target or digest(receipt) != adoption["receipt_checksum"]:
                raise ValueError("generic_definition_stage_adoption_binding")
            if projection["review_allowed"]:
                keys.append(adoption["key"])
        return tuple(keys)
