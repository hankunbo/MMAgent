"""Opt-in all-requirement projection; partial observations retain their scope."""

import json
from datetime import UTC, datetime

from apps.vectorengine_case_materials import ref
from apps.vectorengine_single_case import write_once
from db.runtime.repository.case_pipeline import identity
from packages.contracts import Artifact
from packages.evidence_details.acceptance_models import TaskPolicyV4
from packages.grounding.models import canonical, digest, validate
from packages.llm_provider.models import ObservationInput
from packages.object_store import checksum

from .dynamic_stages import PhaseEvidence, TargetReader, acquisition_state, item_for, rebuild
from .inspection_views import contrast_view
from .partial_observation import evaluate, schedule
from .quality_materials import QualityCaseEvidence, upgrade


def measured_projection(record, *, include_origins=False):
    """Keep actual definitions, units and comparison gaps; never relabel CRAE."""
    r = record["results"]
    result, origins = {}, {}
    av = r.get("av_metrics", {}).get("observation")
    if av:
        classes = av["classes"]
        for target, section, field, unit in (
            ("VASCULAR_SKELETON_DENSITY", "whole_image", "skeleton_density", "px_inverse"),
            ("VASCULAR_FRACTAL", "skeleton_fractal", "value", "dimensionless"),
            ("AV_TORTUOSITY", "tortuosity", "mean", "ratio"),
            ("ARTERIOLAR_CALIBER", "annulus", "width_proxy_disc_ratio", "disc_diameter_ratio"),
        ):
            values = {k: v[section][field] for k, v in classes.items()}
            if any(v is not None for v in values.values()):
                result["association:" + target] = dict(
                    values=values,
                    unit=unit,
                    method="registered_av_" + field,
                    definition=section + "." + field,
                    reference_match="population_association_only_no_matched_normal_distribution",
                    is_CRAE=False if target == "ARTERIOLAR_CALIBER" else None,
                )
                origins["association:" + target] = "av_metrics"
    for task, target in (
        ("ellipse", "association:DISC_SHAPE_CONTEXT"),
        ("branch", "association:VENULAR_BRANCH_ANGLE"),
        ("rim", "criterion:RIM_RADIAL_LT_01"),
    ):
        value = r.get(task, {}).get("observation")
        if value:
            if task == "ellipse":
                value = {
                    k: value["measurement"][k]
                    for k in (
                        "minor_over_major",
                        "major_over_minor",
                        "major_axis_angle_degrees",
                        "area_px",
                        "center_xy",
                        "method",
                    )
                }
            elif task == "branch":
                if not value["count"] or value["median_degrees"] is None:
                    continue
                value = {k: value[k] for k in ("count", "median_degrees", "mean_degrees", "status")}
            elif task == "rim":
                value = {
                    k: value["measurement"][k]
                    for k in (
                        "minimum_normalized_width",
                        "valid_rays",
                        "invalid_rays",
                        "orientation",
                        "anatomical_missing_reason",
                    )
                }
            result[target] = dict(value=value, unit="geometry", method="registered_" + task)
            origins[target] = task
    measured = record["original_measurements"]["measurements"]
    if measured.get("area_cdr") is not None:
        result["descriptive:CUP_AREA_CONTEXT"] = dict(
            value=measured["area_cdr"],
            unit="ratio",
            method="same_mask_area_cdr",
            use="same_cup_description_not_independent_disease_evidence",
        )
        origins["descriptive:CUP_AREA_CONTEXT"] = "original_measurements"
    return (result, origins) if include_origins else result


def merge_partial(seed, observations):
    raw = seed.policy.model_dump(mode="json")
    by_id = {i["id"]: i for i in raw["items"]}
    values = {o.reference.code: o for o in seed.material.observations}
    requirements = {r.id: r for r in seed.policy.requirements}
    for target, obs in observations.items():
        if target not in requirements or requirements[target].input_modalities != ("SLO",):
            raise ValueError("partial_target_not_current_requirement")
        details = evaluate(obs)
        if not details["availability_score"]:
            continue
        value = item_for(requirements[target], obs.model_dump(mode="json"), seed.resources)
        value["quality"] = dict(
            value=details["effective_quality"],
            method="partial_visibility_v1",
            reason="V=clarity; absent/indeterminate also use coverage, multiplied once",
            source=seed.resources[0].model_dump(mode="json"),
        )
        # One requirement, one active observation; original measurement remains in
        # the tool artifact, not a duplicate count of the same anatomy.
        by_id[target] = value
        values[obs.reference.code] = obs
    raw["items"] = list(by_id.values())
    for r in raw["requirements"]:
        if r["id"] in by_id:
            r["missing_reason"] = None
        elif r["id"] in observations:
            r["missing_reason"] = "observation_" + observations[r["id"]].status
    wanted = {i["observation_code"] for i in raw["items"]}
    return rebuild(seed, validate(TaskPolicyV4, raw), [o for k, o in values.items() if k in wanted])


class ComprehensiveEvidence(QualityCaseEvidence):
    root_version = "full-evidence-case-root/1.0.0"
    root_prefix = "full-evidence-case-root:"
    root_filename = "full-evidence-root-v1.json"

    def __init__(self, sample, *, registered=None, tools_optional=False):
        super().__init__(sample, registered=registered)
        directory = self.directory / "13-full-evidence-tools-v2"
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        files = [directory / "artifact.json", directory / "result.json"]
        if tools_optional and not any(p.exists() for p in files):
            self.extra_artifact = None
            self.tools = dict(
                case_code=self.binding["case_code"],
                image_checksum=self.image_artifact.checksum,
                results={},
                status="not_obtained",
                original_measurements={"measurements": {}},
            )
        else:
            self.extra_artifact = Artifact.model_validate_json(files[0].read_bytes())
            self.tools = json.loads(files[1].read_bytes())
        if (
            self.tools["case_code"] != self.binding["case_code"]
            or self.tools["image_checksum"] != self.image_artifact.checksum
        ):
            raise ValueError("full_tool_case_binding")
        raw = self.objects.read_bounded(self.image_artifact.uri, 4194304)
        self.view_bytes, self.view_policy = contrast_view(raw)
        key = identity("TASK", "full-display-v1:" + self.binding["case_code"])
        prior = self.runtime.read_observation_bundle(key)
        uri = (
            prior[0].uri
            if prior
            else self.objects.promote(
                self.objects.stage(self.view_bytes), checksum(self.view_bytes)
            )
        )
        self.vision_artifact = self.image_artifact.model_copy(
            update=dict(
                asset_code=identity(
                    "ASSET", self.image_artifact.asset_code + digest(self.view_policy)
                ),
                uri=uri,
                checksum=checksum(self.view_bytes),
                derived_from=(self.image_artifact.asset_code,),
                extensions=dict(display_transform=self.view_policy),
                provenance=self.image_artifact.provenance.model_copy(
                    update=dict(
                        created_at=prior[0].provenance.created_at if prior else datetime.now(UTC),
                        generator_version="display-linear-p995-1.0.0",
                        source_codes=(self.image_artifact.asset_code,),
                        input_checksums=(self.image_artifact.checksum,),
                    )
                ),
            )
        )
        self.runtime.register_observation_bundle((self.vision_artifact,), key)
        if self.runtime.read_observation_bundle(key) != (self.vision_artifact,):
            raise ValueError("full_display_artifact_binding")
        write_once(directory / "display-transform.json", self.view_policy)

    def image(self, reference):
        if hasattr(self, "vision_artifact") and reference.code == self.vision_artifact.asset_code:
            current = super().image(
                ref("image", self.image_artifact.asset_code, self.image_artifact.checksum)
            )
            raw, policy = contrast_view(current)
            if (
                raw != self.view_bytes
                or policy != self.view_policy
                or reference.checksum != checksum(raw)
            ):
                raise ValueError("full_display_image_changed")
            return raw
        return super().image(reference)

    def check_tools(self):
        a = self.extra_artifact
        if a is not None:
            if self.runtime.read_case_artifact(self.binding["case_code"], a.asset_code) != a:
                raise ValueError("full_tool_artifact_changed")
            raw = self.objects.read_bounded(a.uri, 4194304)
            if checksum(raw) != a.checksum or json.loads(raw) != self.tools:
                raise ValueError("full_tool_content_changed")
        v = self.vision_artifact
        if self.runtime.read_case_artifact(self.binding["case_code"], v.asset_code) != v:
            raise ValueError("full_display_artifact_changed")

    def project(self, *, run_id, config):
        seed, manifest = super().project(run_id=run_id, config=config)
        self.check_tools()
        if self.extra_artifact is None:
            return seed, dict(
                manifest, tools_status="not_obtained", display_transform=self.view_policy
            )
        raw = seed.policy.model_dump(mode="json")
        values = list(seed.material.observations)
        requirements = {r.id: r for r in seed.policy.requirements}
        resources = (
            *seed.resources,
            ref("resource", self.extra_artifact.asset_code, self.extra_artifact.checksum),
        )
        for target, measured in measured_projection(self.tools).items():
            if target not in requirements or target in {i["id"] for i in raw["items"]}:
                continue
            compact = dict(measured)
            if len(canonical(compact).decode()) > 850:
                # Keep bounded summaries from the original algorithm, full data
                # remains in the checksum-bound Runtime artifact.
                raise ValueError("full_measurement_summary_too_large")
            compact["full_record"] = self.extra_artifact.asset_code
            obs = dict(
                status="obtained",
                method=measured["method"],
                unit=measured["unit"],
                value=canonical(compact).decode(),
                image=None,
            )
            obs["reference"] = ref("observation", "OBS:FULL:" + target, digest(obs)).model_dump(
                mode="json"
            )
            item = item_for(requirements[target], obs, resources)
            item.update(
                kind="association" if target.startswith("association:") else "criterion",
                role="diagnostic",
                correlation_group="retinal_vascular_geometry"
                if target.startswith("association:VASCULAR")
                or target
                in {
                    "association:AV_TORTUOSITY",
                    "association:ARTERIOLAR_CALIBER",
                    "association:VENULAR_BRANCH_ANGLE",
                }
                else "optic_nerve_geometry",
                dependency_paths=[[r.code for r in resources]],
            )
            item["quality"] = dict(
                value=self.tools["component_views"]["disc"]["retained_fraction"]
                * (
                    self.tools["component_views"]["cup"]["retained_fraction"]
                    if target == "criterion:RIM_RADIAL_LT_01"
                    else 1.0
                ),
                method="component_view_retention_v1",
                reason="Q is component retention, not accuracy; assess remaining uncertainty in O",
                source=resources[-1],
            )
            raw["items"].append(item)
            values.append(validate(ObservationInput, obs))
        for r in raw["requirements"]:
            if r["id"] in {i["id"] for i in raw["items"]}:
                r["missing_reason"] = None
        state = rebuild(seed, validate(TaskPolicyV4, raw), values, resources=resources)
        self.policies[run_id] = digest(state.policy)
        return state, dict(
            manifest,
            full_tools_checksum=self.extra_artifact.checksum,
            display_transform=self.view_policy,
        )

    def verify(self, state):
        super().verify(state)
        self.check_tools()


class PartialReader(TargetReader):
    partial_observation_contract = "1.0.0"


class ComprehensivePhase(PhaseEvidence):
    def wrap_owner(self, owner):
        return PartialReader(owner) if self.target else owner

    def project_current(self):
        if self.target:
            state = acquisition_state(
                self.seed,
                self.target,
                self.base.vision_artifact,
                allowed_targets=tuple(r.id for r in self.seed.policy.requirements),
            )
            # Display transform is explicit in the original image binding.
            o = state.material.observations[0].model_dump(mode="json")
            o["image"]["preprocessing"] = ref(
                "preprocessing", "POLICY:DISPLAY_LINEAR_P995", digest(self.base.view_policy)
            ).model_dump(mode="json")
            o["reference"]["checksum"] = digest({k: v for k, v in o.items() if k != "reference"})
            return rebuild(state, state.policy, [o])
        state = merge_partial(self.seed, self.observation_reader())
        if self.model_reader:
            from packages.evidence_details.acceptance_models import TaskPolicyV3
            from packages.evidence_scoring.dynamic_policy import POLICY

            from .dynamic_model import attach_model
            from .dynamic_stages import rebuild as reset

            raw = state.policy.model_dump(mode="json")
            factors = {i.id: i.quality.model_dump(mode="json") for i in state.policy.items}
            raw.update(
                schema_version="research-task-policy/3.0.0", aggregation_checksum=digest(POLICY)
            )
            for i in raw["items"]:
                i.pop("quality")
            state = attach_model(
                reset(state, validate(TaskPolicyV3, raw), state.material.observations),
                self.model_reader(),
            )
            return upgrade(state, factors=factors)
        return state

    def plan(self):
        return schedule(
            self.seed.policy.requirements, measured={i.id for i in self.seed.policy.items}
        )
