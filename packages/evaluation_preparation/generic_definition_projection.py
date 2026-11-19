"""Current research interpretations over original Generic facts and bound successors.

Only a new phase may consume a revision. Registered measurements explicitly add
an observation and replace one item; old visible facts and saved judgments remain.
Requirement membership and dependence groups are preserved.
The returned manifest must be persisted by the original phase owner before send.
This projection grants neither a request budget nor clinical qualification.
"""

from packages.evidence_reuse.definition_acceptance import ResearchDefinitionOwner
from packages.grounding.models import canonical, digest, validate
from packages.llm_provider.models import Material, Source

from .dynamic_stages import rebuild
from .generic_aliases import GenericPhase
from .generic_definition_facts import ConfirmedGenericMeasurements
from .generic_pending_review import selected_state

VERSION = "generic-confirmed-interpretation-projection/1.0.0"


def project_interpretations(seed, owner, keys, *, selected_only=True):
    """Read original receipts; never accept caller-supplied definitions or scores."""
    if (
        type(owner) is not ResearchDefinitionOwner
        or type(owner.confirmed_observations) is not ConfirmedGenericMeasurements
    ):
        raise ValueError("generic_interpretation_original_owner_required")
    keys = tuple(keys)
    if (
        not 1 <= len(keys) <= 3
        or any(not isinstance(key, str) or not key for key in keys)
        or len(set(keys)) != len(keys)
    ):
        raise ValueError("generic_interpretation_receipt_selection")
    facts = owner.confirmed_observations.current()
    original_seed = seed
    observation_transition = None
    if "registered_measurement" in facts:
        from .generic_registered_measurement import project_registered

        seed, observation_transition = project_registered(seed, facts["registered_measurement"])
    if (
        seed.material.scope.run_id == facts["run_id"]
        or seed.material.scope.input_checksum != facts["input_checksum"]
        or seed.material.goal_code != facts["target"]
        or owner.binding["input_checksum"] != facts["input_checksum"]
        or owner.binding["target"] != facts["target"]
    ):
        raise ValueError("generic_interpretation_new_phase_scope")
    raw = seed.policy.model_dump(mode="json")
    requirements = {r["id"]: r for r in raw["requirements"]}
    items = {r["id"]: r for r in raw["items"]}
    observations = {o.reference.code: o for o in seed.material.observations}
    sources = {s.reference.code: s for s in seed.material.sources}
    revisions = []
    selected = set()
    for key in keys:
        receipt, projection = owner.read_confirmed_projection(key)
        target = projection["id"]
        if target in selected or target not in facts["facts"]:
            raise ValueError("generic_interpretation_duplicated_or_unselected")
        fact = facts["facts"][target]
        requirement, item = requirements.get(target), items.get(target)
        if requirement is None or item is None:
            raise ValueError("generic_interpretation_original_requirement_missing")
        observed = observations.get(item["observation_code"])
        if (
            observed is None
            or observed.status != "obtained"
            or digest((observed, fact["origin"])) != fact["observation_binding"]
            or (item["method"], item["unit"]) != (fact["method"], fact["unit"])
            or canonical(requirement["definition"]) != canonical(fact["requirement"]["definition"])
            or canonical(item["definition"]) != canonical(requirement["definition"])
            or tuple(requirement["input_modalities"])
            != tuple(fact["requirement"]["input_modalities"])
            or canonical({k: receipt["confirmed_basis"].get(k) for k in fact}) != canonical(fact)
        ):
            raise ValueError("generic_interpretation_original_fact_binding")
        definition = projection["definition"]
        if not projection["review_allowed"] or definition.interpretation_status != "installed":
            raise ValueError("generic_interpretation_not_reviewable")
        if canonical(definition) == canonical(requirement["definition"]):
            raise ValueError("generic_interpretation_no_effective_revision")
        old_definition = (
            observation_transition["previous_definition"]
            if observation_transition is not None
            else requirement["definition"]
        )
        for material in receipt["sources"]:
            source = validate(Source, material["material"])
            if source.reference.code in sources and sources[source.reference.code] != source:
                raise ValueError("generic_interpretation_source_collision")
            sources[source.reference.code] = source
        if any(
            sources.get(ref.code) is None or sources[ref.code].reference != ref
            for ref in definition.source_refs
        ):
            raise ValueError("generic_interpretation_source_binding")
        # The method is fixed by the original confirmed-fact owner. An explicit
        # registered transition was projected above; this step only adopts the
        # current interpretation and its precise source binding.
        requirement.update(
            definition=definition.model_dump(mode="json"),
            basis_kind=projection["basis_kind"],
            missing_reason=None,
        )
        item.update(
            definition=definition.model_dump(mode="json"),
            source_codes=[ref.code for ref in definition.source_refs],
        )
        selected.add(target)
        revisions.append(
            dict(
                assessment_id=target,
                receipt_key=key,
                receipt_checksum=digest(receipt),
                previous_definition=old_definition,
                current_definition=definition,
                observation=fact["observation"],
                observation_binding=fact["observation_binding"],
                method=fact["method"],
                unit=fact["unit"],
                dependency_assessment=projection["dependency_assessment"],
                qualification=receipt["qualification"],
                clinical_acceptance=False,
            )
        )
        if observation_transition is not None:
            revisions[-1]["observation_transition"] = observation_transition
    policy = validate(type(seed.policy), raw)
    material = seed.material.model_dump(mode="json")
    material["sources"] = [s.model_dump(mode="json") for s in sources.values()]
    material["scope"].pop("material_checksum")
    material["scope"]["material_checksum"] = digest(material)
    with_sources = seed.model_copy(update=dict(material=validate(Material, material)))
    projected = rebuild(with_sources, policy, seed.material.observations)
    if selected_only:
        projected = selected_state(projected, selected)
    from packages.evidence_details.review_tasks import project_review_tasks

    project_review_tasks(projected)
    manifest = dict(
        schema_version=VERSION,
        original_run_id=facts["run_id"],
        phase_run_id=seed.material.scope.run_id,
        original_facts_checksum=digest(facts),
        original_state_checksum=digest(original_seed),
        projected_state_checksum=digest(projected),
        selected_ids=sorted(selected),
        selected_only=selected_only,
        requirements_before=len(seed.policy.requirements),
        requirements_after=len(projected.policy.requirements),
        added_requirements=[],
        interpretation_revisions=revisions,
        unchanged_observations_checksum=digest(seed.material.observations),
        unchanged_groups_checksum=digest(seed.policy.groups),
        historical_scores_reused=False,
        whole_case_aggregation_required=True,
    )
    if observation_transition is not None:
        manifest.update(
            schema_version="generic-registered-interpretation-projection/1.0.0",
            observation_transition=observation_transition,
        )
        manifest.pop("unchanged_observations_checksum")
        manifest["original_observations_checksum"] = digest(original_seed.material.observations)
        manifest["projected_observations_checksum"] = digest(projected.material.observations)
    # Return detached values; a display consumer cannot mutate the original data.
    import json

    return projected, json.loads(canonical(manifest))


class GenericDefinitionReviewPhase(GenericPhase):
    """Existing phase producer with an explicit accepted-definition dependency."""

    def __init__(
        self, base, *, facts, definition_owner, definition_keys, selected_only=True, **kwargs
    ):
        if type(facts) is not ConfirmedGenericMeasurements or (
            type(definition_owner) is not ResearchDefinitionOwner
            or definition_owner.confirmed_observations is not facts
        ):
            raise ValueError("generic_interpretation_phase_owner_binding")
        super().__init__(base, **kwargs)
        self.facts = facts
        self.definition_owner = definition_owner
        self.definition_keys = tuple(definition_keys)
        self.selected_only = selected_only

    def project_current(self):
        if self.base is not self.facts.campaign.base or self.target is not None:
            raise ValueError("generic_interpretation_no_new_acquisition")
        seed = super().project_current()
        state, manifest = project_interpretations(
            seed, self.definition_owner, self.definition_keys, selected_only=self.selected_only
        )
        self.definition_revision_manifest = manifest
        return state


def phase_type(facts, owner, keys, *, selected_only=True):
    """Bind original owners without inventing another phase executor."""
    from functools import partial

    if (
        type(facts) is not ConfirmedGenericMeasurements
        or type(owner) is not ResearchDefinitionOwner
        or owner.confirmed_observations is not facts
    ):
        raise ValueError("generic_interpretation_phase_owner_binding")
    return partial(
        GenericDefinitionReviewPhase,
        facts=facts,
        definition_owner=owner,
        definition_keys=tuple(keys),
        selected_only=selected_only,
    )


def projection_record(evidence, state, run_id):
    """Revalidate outside the Runtime transaction, then persist with phase input."""
    if type(evidence) is not GenericDefinitionReviewPhase:
        raise ValueError("generic_interpretation_phase_producer_required")
    evidence.verify(state)
    manifest = evidence.definition_revision_manifest
    if manifest["phase_run_id"] != run_id or manifest["projected_state_checksum"] != digest(state):
        raise ValueError("generic_interpretation_phase_record_binding")
    import json

    return json.loads(canonical(manifest))
