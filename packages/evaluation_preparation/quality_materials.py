"""Opt-in quality-weighted materials; no change to frozen v3 campaigns."""

import copy

from packages.evidence_details.acceptance_models import TaskPolicyV3, TaskPolicyV4
from packages.evidence_scoring.quality_policy import POLICY
from packages.grounding.models import digest, validate

from .dynamic_materials import DynamicCaseEvidence
from .dynamic_stages import VISUAL_IDS, PhaseEvidence, acquisition_state, item_for, rebuild
from .quality import mask_quality


def visual_cdr_definition(definition):
    """Original acquisition variant, with unchanged medical rule/source references."""
    from apps.vectorengine_case_materials import ref

    d = copy.deepcopy(definition)
    d["examination"] = (
        "垂直杯盘比图像估计：本例分割未得到视杯。直接查看本例图像，"
        "如边界可见，描述杯与盘的垂直边界/尺度、估计比值及范围；"
        "这不是掩膜测量。不可辨认时明确ungradable，不补0，不作诊断。"
    )
    d["applicability_conditions"] += " 本次为VLM近似估计，降低观察可靠性并保留估计范围。"
    d["reference"] = ref(
        "definition",
        "POLICY:QUALITY:VISUAL_VCDR",
        digest({k: v for k, v in d.items() if k != "reference"}),
    ).model_dump(mode="json")
    return d


def compatible_acquisition_definition(identifier, before, after):
    """Only the two exact owner-defined variants are equivalent for dispatch.

    This never changes the current task sent for review or its original definition.
    """
    from apps.vectorengine_case_materials import ref

    if before == after:
        return True
    if identifier != "criterion:VCDR_GE_07":
        return False
    for base, visual in ((before, after), (after, before)):
        expected = ref(
            "definition",
            "POLICY:ACTIVE:" + identifier,
            digest({k: v for k, v in base.items() if k != "reference"}),
        )
        if base.get("reference") == expected.model_dump(mode="json"):
            if visual_cdr_definition(base) == visual:
                return True
    return False


def upgrade(state, *, factors=None):
    """Quality inputs are derived by the installed owner, not by an LLM."""
    raw = state.policy.model_dump(mode="json")
    raw.update(schema_version="research-task-policy/4.0.0", aggregation_checksum=digest(POLICY))
    for item in raw["items"]:
        item["quality"] = (factors or {}).get(
            item["id"],
            dict(
                value=1.0,
                method="no_additional_mask_penalty",
                reason="Observation uncertainty stays in O; this item does not use mask geometry.",
                source=state.resources[0].model_dump(mode="json"),
            ),
        )
    return rebuild(state, validate(TaskPolicyV4, raw), state.material.observations)


def merge_review(seed, observations):
    """Preserve measured items; append confirmed visual values without empty slots."""
    from .dynamic_stages import review_state

    # Legacy projection is deliberately left unchanged for old exact replay.
    legacy = seed.policy.model_dump(mode="json")
    from packages.evidence_scoring.dynamic_policy import POLICY as LEGACY_POLICY

    legacy.update(
        schema_version="research-task-policy/3.0.0", aggregation_checksum=digest(LEGACY_POLICY)
    )
    for item in legacy["items"]:
        item.pop("quality", None)
    plain = rebuild(seed, validate(TaskPolicyV3, legacy), seed.material.observations)
    visual = review_state(plain, {k: v for k, v in observations.items() if k in VISUAL_IDS})
    raw = seed.policy.model_dump(mode="json")
    by_id = {i["id"]: i for i in raw["items"]}
    for item in visual.policy.items:
        value = item.model_dump(mode="json")
        if item.id in by_id:
            # Keep a real measurement rather than silently replacing it with a guess.
            continue
        if isinstance(seed.policy, TaskPolicyV4):
            value["quality"] = dict(
                value=1.0,
                method="no_additional_mask_penalty",
                reason="Vision reliability is assessed by O.",
                source=seed.resources[0].model_dump(mode="json"),
            )
        by_id[item.id] = value
    for target, observation in observations.items():
        if target in VISUAL_IDS:
            continue
        if target != "criterion:VCDR_GE_07":
            raise ValueError("quality_visual_target")
        if observation.status != "obtained":
            continue
        if observation.method != "live_image_observation" or observation.unit != "qualitative":
            raise ValueError("quality_visual_method")
        if target not in by_id:
            requirement = next(r for r in seed.policy.requirements if r.id == target)
            value = item_for(requirement, observation.model_dump(mode="json"), seed.resources)
            value["quality"] = dict(
                value=1.0,
                method="no_additional_mask_penalty",
                reason="Independent image estimate; VLM uncertainty is assessed in O.",
                source=seed.resources[0].model_dump(mode="json"),
            )
            by_id[target] = value
    raw["items"] = list(by_id.values())
    existing = {o.reference.code: o for o in seed.material.observations}
    existing.update({o.reference.code: o for o in visual.material.observations})
    existing.update(
        {
            o.reference.code: o
            for k, o in observations.items()
            if k not in VISUAL_IDS and o.status == "obtained"
        }
    )
    wanted = {i["observation_code"] for i in raw["items"]}
    for r in raw["requirements"]:
        if r["id"] in by_id:
            r["missing_reason"] = None
        elif r["id"] in observations:
            r["missing_reason"] = "observation_" + observations[r["id"]].status
    return upgrade(
        rebuild(
            seed, validate(type(seed.policy), raw), [o for k, o in existing.items() if k in wanted]
        ),
        factors={i.id: i.quality.model_dump(mode="json") for i in seed.policy.items}
        if isinstance(seed.policy, TaskPolicyV4)
        else None,
    )


def quality_observation_value(value, quality):
    """Exact existing compact observation projection; Q remains applied once."""
    import json

    from packages.grounding.models import canonical

    measured = json.loads(value)
    measured["quality"] = dict(
        value=quality["quality_factor"],
        policy=quality["policy"]["version"],
        components=quality["components"],
        reasons=quality["reasons"],
        diagnostic_checksum=digest(quality),
    )
    measured["quality_usage"] = "Scorer multiplies Q once; do not repeat this penalty in O."
    return canonical(measured).decode()


class QualityCaseEvidence(DynamicCaseEvidence):
    root_version = "quality-source-case-root/1.0.0"
    root_prefix = "quality-source-case-root:"
    root_filename = "quality-root-v1.json"

    def load_measurements(self, directory, objects):
        super().load_measurements(directory, objects)
        if not self.tool_assets:
            self.soft_quality = dict(
                quality_factor=0.0,
                measurement_available=False,
                reasons=["no_segmentation_artifacts"],
                stop_case=False,
            )
            return
        masks = {a.extensions.get("output_port"): a for a in self.tool_assets}
        self.soft_quality = mask_quality(
            objects.read_bounded(masks["disc_mask"].uri, 4194304),
            objects.read_bounded(masks["cup_mask"].uri, 4194304),
        )
        # Compatibility key only determines mathematical availability in this new owner.
        self.quality = dict(
            self.soft_quality, bbox_cdr_usable=self.soft_quality["measurement_available"]
        )

    def project(self, *, run_id, config):
        state, manifest = super().project(run_id=run_id, config=config)
        if state.policy.items:
            from packages.llm_provider.models import ObservationInput

            observations = []
            for observation in state.material.observations:
                body = observation.model_dump(mode="json")
                body["value"] = quality_observation_value(observation.value, self.soft_quality)
                body["reference"]["checksum"] = digest(
                    {k: v for k, v in body.items() if k != "reference"}
                )
                observations.append(validate(ObservationInput, body))
            state = rebuild(state, state.policy, observations)
        if not self.soft_quality["measurement_available"]:
            raw = state.policy.model_dump(mode="json")
            for requirement in raw["requirements"]:
                if requirement["id"] == "criterion:VCDR_GE_07":
                    requirement["definition"] = visual_cdr_definition(requirement["definition"])
            state = rebuild(state, validate(TaskPolicyV3, raw), state.material.observations)
        q = self.soft_quality["quality_factor"]
        factors = {
            i.id: dict(
                value=q,
                method="mask_quality_weight_v1",
                reason=";".join(self.soft_quality["reasons"]) or "connected_nested_masks",
                source=state.resources[0].model_dump(mode="json"),
            )
            for i in state.policy.items
        }
        result = upgrade(state, factors=factors)
        self.policies[run_id] = digest(result.policy)
        return result, dict(manifest, quality=self.soft_quality)

    def verify(self, state):
        super().verify(state)
        # The original artifact read is repeated; changed masks cannot keep an old Q.
        self.load_measurements(self.directory, self.objects)
        for item in state.policy.items:
            if (
                item.quality.method == "mask_quality_weight_v1"
                and item.quality.value != self.soft_quality["quality_factor"]
            ):
                raise ValueError("quality_owner_value_changed")


class QualityPhaseEvidence(PhaseEvidence):
    def project_current(self):
        if self.target is not None:
            if self.base.soft_quality["measurement_available"]:
                raise ValueError("quality_fallback_not_required")
            return acquisition_state(
                self.seed,
                self.target,
                self.image_artifact,
                allowed_targets=("criterion:VCDR_GE_07",),
            )
        state = merge_review(self.seed, self.observation_reader())
        factors = {i.id: i.quality.model_dump(mode="json") for i in state.policy.items}
        if self.model_reader:
            from packages.evidence_scoring.dynamic_policy import POLICY as OLD_POLICY

            from .dynamic_model import attach_model

            raw = state.policy.model_dump(mode="json")
            raw.update(
                schema_version="research-task-policy/3.0.0", aggregation_checksum=digest(OLD_POLICY)
            )
            for item in raw["items"]:
                item.pop("quality")
            state = attach_model(
                rebuild(state, validate(TaskPolicyV3, raw), state.material.observations),
                self.model_reader(),
            )
        return upgrade(state, factors=factors)
