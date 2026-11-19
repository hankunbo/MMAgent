"""Source-backed observation tasks and a distinct obtained-evidence projection.

An acquisition task is not a disease-score slot. Each phase uses an original
Runtime run and immutable grant; a case campaign shares their total request cap.
"""

from apps.vectorengine_case_materials import ref
from packages.evidence_details.acceptance_models import DomainState, TaskPolicyV2, TaskPolicyV3
from packages.evidence_details.review_tasks import project_review_tasks
from packages.evidence_scoring.continuous_policy import POLICY as LEGACY_POLICY
from packages.grounding.models import digest, validate
from packages.llm_provider.models import ImageBinding, Material

VISUAL_IDS = (
    "criterion:RIM_NOTCH",
    "criterion:DISC_HEMORRHAGE",
    "association:DISC_VESSEL_COURSE",
    "association:PERIPAPILLARY_APPEARANCE",
)


def rebuild(seed, policy, observations, *, resources=None):
    raw = seed.material.model_dump(mode="json")
    raw["observations"] = [
        o.model_dump(mode="json") if hasattr(o, "model_dump") else o for o in observations
    ]
    raw["scope"]["policy_checksum"] = digest(policy)
    raw["policy_ref"]["checksum"] = digest(policy)
    raw["scope"].pop("material_checksum")
    raw["scope"]["material_checksum"] = digest(raw)
    return validate(
        DomainState,
        seed.model_copy(
            update=dict(
                material=validate(Material, raw),
                policy=policy,
                nodes=tuple(sorted({seed.material.goal_code} | {i.node for i in policy.items})),
                resources=tuple(resources or seed.resources),
            )
        ),
    )


def item_for(requirement, observation, resources):
    return dict(
        id=requirement.id,
        node=requirement.id,
        kind="visual",
        role="visual",
        observation_code=observation["reference"]["code"],
        method=observation["method"],
        unit=observation["unit"],
        laterality="UNKNOWN",
        source_codes=[r.code for r in requirement.definition.source_refs],
        applicable=True,
        definition=requirement.definition,
        correlation_group="optic_nerve_slo",
        dependence_block="same_slo_image",
        dependency_paths=[[resources[0].code]],
    )


def acquisition_state(seed, target, image, *, allowed_targets=None):
    if allowed_targets is None:
        allowed_targets = VISUAL_IDS
    requirement = next(r for r in seed.policy.requirements if r.id == target)
    if (
        target not in allowed_targets
        or requirement.input_modalities != ("SLO",)
        or requirement.definition.interpretation_status != "installed"
    ):
        raise ValueError("dynamic_visual_requirement_not_selected")
    h, w = image.dimensions
    bound = ImageBinding(
        reference=ref("image", image.asset_code, image.checksum),
        media_type="image/png",
        width=w,
        height=h,
        coordinate_system="pixel_xy_top_left",
        location=(0, 0, w, h),
        preprocessing=ref(
            "preprocessing",
            "POLICY:VISIBLE_PNG_UNCHANGED",
            digest(dict(input=image.checksum, transform="none")),
        ),
    )
    observation = dict(
        status="unknown",
        method="live_image_observation",
        unit="qualitative",
        value=None,
        image=bound.model_dump(mode="json"),
    )
    observation["reference"] = ref(
        "observation", "OBS:DYNAMIC:" + target, digest(observation)
    ).model_dump(mode="json")
    policy = validate(
        TaskPolicyV2,
        dict(
            version="source-driven-acquisition-task-1.0.0",
            scale=seed.policy.scale,
            aggregation_checksum=digest(LEGACY_POLICY),
            items=[item_for(requirement, observation, seed.resources)],
            groups=[
                dict(
                    id="acquire:" + target,
                    slots=[[target]],
                    importance=dict(slo_image_research=1, comprehensive_research=1),
                )
            ],
        ),
    )
    # This single unknown is an input to a Vision operation, never a scored result.
    return rebuild(seed, policy, [observation])


def review_state(seed, observations):
    requirements = {r.id: r for r in seed.policy.requirements}
    items, values = [], []
    for target, observation in observations.items():
        if target not in VISUAL_IDS or target not in requirements:
            raise ValueError("dynamic_observation_target")
        if (
            observation.status != "obtained"
            or not observation.value
            or not observation.value.strip()
        ):
            continue
        if observation.method != "live_image_observation" or observation.unit != "qualitative":
            raise ValueError("dynamic_observation_method")
        items.append(
            item_for(requirements[target], observation.model_dump(mode="json"), seed.resources)
        )
        values.append(observation)
    raw = seed.policy.model_dump(mode="json")
    raw["items"] = items
    for requirement in raw["requirements"]:
        if requirement["id"] in {i["id"] for i in items}:
            requirement["missing_reason"] = None
        elif requirement["id"] in observations:
            requirement["missing_reason"] = "observation_" + observations[requirement["id"]].status
    result = rebuild(seed, validate(TaskPolicyV3, raw), values)
    project_review_tasks(result)
    return result


class PhaseEvidence:
    """Only composes current original owners; no model output is supplied by callers."""

    def __init__(self, base, *, target=None, observation_reader=None, model_reader=None):
        self.base, self.target, self.observation_reader = base, target, observation_reader
        self.model_reader = model_reader
        self.targeted_acquisition = False

    def __getattr__(self, key):
        return getattr(self.base, key)

    def wrap_owner(self, owner):
        return TargetReader(owner) if self.target else owner

    def project(self, *, run_id, config):
        self.seed, manifest = self.base.project(run_id=run_id, config=config)
        self.initial = self.project_current()
        return self.initial, manifest

    def project_current(self):
        state = (
            acquisition_state(self.seed, self.target, self.image_artifact)
            if self.target
            else review_state(self.seed, self.observation_reader())
        )
        if self.target is None and self.model_reader:
            from .dynamic_model import attach_model

            state = attach_model(state, self.model_reader())
        return state

    def verify(self, state):
        self.base.verify(self.seed)
        expected = self.project_current()
        if (
            expected != self.initial
            or state.policy != self.initial.policy
            or state.material.sources != self.initial.material.sources
        ):
            raise ValueError("dynamic_phase_dependency_changed")
        if (
            state.material.scope.case_code,
            state.material.scope.run_id,
            state.material.scope.input_checksum,
            state.material.scope.visible_checksum,
        ) != (
            self.initial.material.scope.case_code,
            self.initial.material.scope.run_id,
            self.initial.material.scope.input_checksum,
            self.initial.material.scope.visible_checksum,
        ):
            raise ValueError("dynamic_phase_scope")


class TargetReader:
    def __init__(self, owner):
        self.owner = owner

    def __getattr__(self, key):
        return getattr(self.owner, key)

    @property
    def role(self):
        return self.owner.role

    @role.setter
    def role(self, value):
        self.owner.role = value

    def vision_task(self, scope):
        state = self.owner.current()
        if state.material.scope != scope:
            raise ValueError("dynamic_vision_scope")
        tasks = project_review_tasks(state).tasks
        if len(tasks) != 1:
            raise ValueError("dynamic_vision_task_count")
        return tasks[0]
