"""Retain unchanged judgments when an original owner adopted a new definition.

The revised item is supplied by a separate accepted Review, never by transforming
its old score. The full original receipt remains in AcceptedOrigin provenance.
Ordinary aggregation retains its existing all-items-must-match contract.
"""

from packages.evidence_details.accepted_aggregation import AcceptedOriginReader
from packages.evidence_reuse.definition_acceptance import ResearchDefinitionOwner
from packages.grounding.models import canonical, digest

from .generic_definition_facts import ConfirmedGenericMeasurements
from .generic_definition_projection import project_interpretations


class DefinitionRevisionSelection:
    def __init__(self, *, owner, keys, seed_reader):
        if (
            type(owner) is not ResearchDefinitionOwner
            or type(owner.confirmed_observations) is not ConfirmedGenericMeasurements
            or not callable(seed_reader)
        ):
            raise ValueError("definition_selection_original_owners_required")
        self.owner, self.keys, self.seed_reader = owner, tuple(keys), seed_reader

    def manifest(self, current):
        seed = self.seed_reader(current.material.scope.run_id)
        rebuilt, manifest = project_interpretations(
            seed, self.owner, self.keys, selected_only=False
        )
        if rebuilt != current or manifest["selected_only"]:
            raise ValueError("definition_selection_current_projection_changed")
        return manifest

    def retained_ids(self, old, current, candidate):
        manifest = self.manifest(current)
        facts = self.owner.confirmed_observations.current()
        if old.material.scope.run_id != facts["run_id"]:
            raise ValueError("definition_selection_original_fact_run")
        old_requirements = {r.id: r for r in old.policy.requirements}
        old_items = {i.id: i for i in old.policy.items}
        current_items = {i.id: i for i in current.policy.items}
        candidates = {i.assessment_id: i for i in candidate.items}
        removed = set()
        for revision in manifest["interpretation_revisions"]:
            target = revision["assessment_id"]
            previous, following = old_items.get(target), current_items.get(target)
            before = old_requirements.get(target)
            if (
                target not in candidates
                or previous is None
                or following is None
                or before is None
                or canonical(before.definition) != canonical(revision["previous_definition"])
                or canonical(previous.definition) != canonical(before.definition)
                or canonical(following.definition) != canonical(revision["current_definition"])
            ):
                raise ValueError("definition_selection_exact_definition_transition")
            observation = next(
                (
                    o
                    for o in old.material.observations
                    if o.reference.code == previous.observation_code
                ),
                None,
            )
            transition = revision.get("observation_transition")
            if transition is not None:
                fact = facts["facts"][target]
                if (
                    "registered_measurement" not in facts
                    or canonical(transition) != canonical(fact["observation_transition"])
                    or observation is None
                    or canonical(observation) != canonical(transition["previous_observation"])
                    or canonical(transition["observation"]) != canonical(fact["observation_value"])
                    or digest((fact["observation_value"], fact["origin"]))
                    != revision["observation_binding"]
                ):
                    raise ValueError("definition_selection_registered_observation_changed")
            else:
                if (
                    observation is None
                    or observation.reference.model_dump(mode="json") != revision["observation"]
                ):
                    raise ValueError("definition_selection_original_observation_changed")
                if (
                    digest((observation, facts["facts"][target]["origin"]))
                    != revision["observation_binding"]
                ):
                    raise ValueError("definition_selection_original_observation_changed")
            removed.add(target)
        retained = set(candidates) - removed
        if not removed or not retained:
            raise ValueError("definition_selection_requires_revision_and_retained_origin")
        return retained


class RetainedDefinitionOriginReader(AcceptedOriginReader):
    """All original receipt/request/grant checks run before subset selection."""

    def __init__(self, *, definition_selection, **kwargs):
        if type(definition_selection) is not DefinitionRevisionSelection:
            raise ValueError("definition_selection_owner_required")
        super().__init__(**kwargs)
        if (
            self.alias_manifests is not None
            or self.reused_origin_new_items_only
            or self.client_transition
            or self.reasoned_batch_resize
        ):
            raise ValueError("definition_selection_no_combined_transition")
        self.definition_selection = definition_selection

    def check_compatible_ids(self, old, current, candidate, ids):
        if set(ids) != self.definition_selection.retained_ids(old, current, candidate):
            raise ValueError("definition_selection_unexpected_changed_item")
