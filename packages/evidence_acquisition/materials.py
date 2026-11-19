"""Pure preparation adapters. Only original owners may install a prepared projection.

No write, network, model, scoring or live-permission side effects are performed here.
All existing tasks/coverage groups stay present; changes produce different identities.
"""

from packages.evidence_details.acceptance_models import DomainState
from packages.evidence_details.review_tasks import project_review_tasks
from packages.grounding.models import digest, validate
from packages.llm_provider.models import Material, Source, TaskDefinition


def _rebuild(state, *, items=None, observations=None, sources=None):
    raw = state.model_dump(mode="json")
    if items is not None:
        raw["policy"]["items"] = [i.model_dump(mode="json") for i in items]
    material = raw["material"]
    if observations is not None:
        material["observations"] = [o.model_dump(mode="json") for o in observations]
    if sources is not None:
        material["sources"] = [s.model_dump(mode="json") for s in sources]
    policy_checksum = digest(raw["policy"])
    material["policy_ref"]["checksum"] = policy_checksum
    material["scope"]["policy_checksum"] = policy_checksum
    material["scope"].pop("material_checksum")
    material["scope"]["material_checksum"] = digest(material)
    material = validate(Material, material)
    raw["material"] = material.model_dump(mode="json")
    result = validate(DomainState, raw)
    project_review_tasks(result)
    return result


def _item(state, target):
    state = validate(DomainState, state)
    project_review_tasks(state)
    matches = [p for p in state.policy.items if p.id == target and p.fixed_model_row is None]
    if len(matches) != 1:
        raise ValueError("acquisition_target_not_editable")
    return state, matches[0]


def prepare_definition(state, target, *, definition_reference, definition_owner, source_owner):
    """Read an approved definition from its owner; caller cannot submit arbitrary rule text.

    The source owner must perform current lifecycle/object verification (e.g. the
    existing RoundCaseEvidence reader). It is not a caller 'verified' boolean.
    This prepares a new material; it does not authorize installation into an old run.
    """
    state, item = _item(state, target)
    definition = validate(
        TaskDefinition, definition_owner.read_definition(target, definition_reference)
    )
    if (
        definition.reference != definition_reference
        or definition.interpretation_status != "installed"
    ):
        raise ValueError("acquisition_definition_identity")
    selected = tuple(
        validate(Source, source_owner.read_source_ref(r)) for r in definition.source_refs
    )
    if tuple(s.reference for s in selected) != definition.source_refs or any(
        s.reference.code.startswith("POLICY:") for s in selected
    ):
        raise ValueError("acquisition_medical_source_required")
    sources = {s.reference.code: s for s in state.material.sources}
    for source in selected:
        if source.reference.code in sources and sources[source.reference.code] != source:
            raise ValueError("acquisition_source_version_conflict")
        sources[source.reference.code] = source
    # All current task references remain present. Do not silently trim another
    # task's source or enlarge the provider's original bounded material contract.
    if len(sources) > 16:
        raise ValueError("acquisition_scoped_material_required")
    updated = item.model_copy(
        update={"definition": definition, "source_codes": tuple(s.reference.code for s in selected)}
    )
    prepared = _rebuild(
        state,
        items=tuple(updated if p.id == target else p for p in state.policy.items),
        sources=tuple(sources.values()),
    )
    # Return-time revalidation before handing new material to the controlled installer.
    if (
        validate(TaskDefinition, definition_owner.read_definition(target, definition_reference))
        != definition
    ):
        raise ValueError("acquisition_definition_changed")
    if (
        tuple(validate(Source, source_owner.read_source_ref(r)) for r in definition.source_refs)
        != selected
    ):
        raise ValueError("acquisition_source_changed")
    return prepared


def prepare_visual_task(state, target, *, image, instruction):
    """One explicit observation per Vision request, with existing full task coverage.

    This reuses VisionProposal 1.0 and original image/acceptance checks. No finding
    is supplied. Quality results cannot fan out into rim/RNFL/hemorrhage findings.
    """
    state, item = _item(state, target)
    if image not in [o.image for o in state.material.observations if o.image is not None]:
        raise ValueError("acquisition_image_not_in_visible_material")
    targets = [p for p in state.policy.items if p.observation_code == item.observation_code]
    if len(targets) != 1:
        raise ValueError("acquisition_ambiguous_visual_target")
    original = next(
        o for o in state.material.observations if o.reference.code == item.observation_code
    )
    if original.status == "obtained":
        raise ValueError("acquisition_existing_observation_requires_explicit_replacement")
    definition_raw = item.definition.model_dump(mode="json", exclude={"reference"})
    definition_raw["examination"] = instruction
    definition_raw["reference"] = {
        **item.definition.reference.model_dump(mode="json"),
        "version": "targeted-visual/1.0.0",
        "checksum": digest(definition_raw),
    }
    definition = validate(TaskDefinition, definition_raw)
    observations = []
    for o in state.material.observations:
        if o.reference.code == original.reference.code:
            value = dict(
                status="unknown",
                method="live_image_observation",
                unit="qualitative",
                value=None,
                image=image,
            )
            observations.append(
                o.model_copy(
                    update={
                        **value,
                        "reference": o.reference.model_copy(
                            update={"version": "targeted-visual/1.0.0", "checksum": digest(value)}
                        ),
                    }
                )
            )
        else:
            if o.image:
                detached = o.model_dump(mode="json", exclude={"reference"})
                detached["image"] = None
                observations.append(
                    o.model_copy(
                        update={
                            "image": None,
                            "reference": o.reference.model_copy(
                                update={
                                    "version": "targeted-visual/1.0.0",
                                    "checksum": digest(detached),
                                }
                            ),
                        }
                    )
                )
            else:
                observations.append(o)
    updated = item.model_copy(
        update={"definition": definition, "method": "live_image_observation", "unit": "qualitative"}
    )
    return _rebuild(
        state,
        items=tuple(updated if p.id == target else p for p in state.policy.items),
        observations=tuple(observations),
    )


def observation_alias_candidate(state, source_id, target_id):
    """Explicit reuse request, not a copy or acceptance of one visual finding as another."""
    state, source = _item(state, source_id)
    _, target = _item(state, target_id)
    obs = next(
        o for o in state.material.observations if o.reference.code == source.observation_code
    )
    return dict(
        schema_version="observation-reuse-candidate/1.0.0",
        state_checksum=digest(state),
        source_assessment_id=source.id,
        target_assessment_id=target.id,
        observation=obs.reference.model_dump(mode="json"),
        status=obs.status,
        correlation_group=source.correlation_group,
        requires_owner_semantic_mapping=True,
        accepted=False,
    )


class TaskTargetedReader:
    """Opt-in prompt extension over an already installed/current original owner.

    Does not prepare, install or authorize material itself. The owner still checks
    Control, images, source withdrawal, permission and all current run bindings.
    """

    def __init__(self, owner):
        self.owner = owner

    def __getattr__(self, name):
        return getattr(self.owner, name)

    def vision_task(self, scope):
        state = validate(DomainState, self.owner.current())
        if state.material.scope != scope:
            raise ValueError("acquisition_vision_scope")
        images = [o for o in state.material.observations if o.image is not None]
        if len(images) != 1:
            raise ValueError("acquisition_vision_single_target")
        matches = [
            t for t in project_review_tasks(state).tasks if t.observation == images[0].reference
        ]
        if len(matches) != 1:
            raise ValueError("acquisition_vision_task_ambiguous")
        return matches[0]
