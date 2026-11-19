"""Case-parameterized research projection; only patient-free structure is reused."""

import json
from pathlib import Path
from types import SimpleNamespace

from apps.active_evidence_materials import PILOTS, RULES, ActiveCaseEvidence, load_manifest
from apps.vectorengine_case_materials import BASELINE, BASELINE_SHA, ref
from apps.vectorengine_round_materials import RoundCaseEvidence, definitions, load_catalog
from packages.evidence_acquisition.materials import prepare_visual_task
from packages.evidence_acquisition.source_aliases import alias_sources, prepare_source_aliases
from packages.evidence_details.acceptance_models import DomainState, TaskPolicyV2
from packages.evidence_scoring.continuous_policy import POLICY
from packages.grounding.models import canonical, digest, validate
from packages.llm_provider.models import ImageBinding, Material, Source
from packages.object_store import checksum


def template():
    raw = BASELINE.read_bytes()
    if checksum(raw) != BASELINE_SHA:
        raise ValueError("new_case_template_origin_changed")
    old = json.loads(raw)
    # Whitelist structure only: never return old observations, judgments or model scores.
    return dict(
        nodes=[{k: n.get(k) for k in ("key", "node_type", "parent")} for n in old["nodes"]],
        items=[
            {
                k: r[k]
                for k in (
                    "id",
                    "node",
                    "title",
                    "kind",
                    "role",
                    "correlation_group",
                    "dependence_block",
                )
            }
            for r in old["current_scoring"]["reviews"]
        ],
        groups=[
            dict(
                id=g["id"],
                importance=g["importance"],
                slots=[s["assessment_ids"] for s in g["slots"]],
            )
            for g in old["current_scoring"]["groups"]
        ],
    )


def empty_model(row, observation, sources, paths):
    missing = dict(value=None, reason="本例未运行该分类模型", missing_reason="model_not_executed")
    return dict(
        **{k: v for k, v in row.items() if k != "kind"},
        kind="model",
        assessment_checksum=digest(observation),
        source_binding_checksum=digest(sources),
        observation=missing,
        relation=missing,
        applicability=missing,
        target="glaucoma_research_hypothesis",
        judgment=dict(
            value=None,
            lower=None,
            upper=None,
            reason="本例未运行该分类模型",
            missing_reason="model_not_executed",
            method="not_assessed",
        ),
        direction="unavailable",
        observation_available=False,
        criterion_checked=False,
        reason="本例未运行该分类模型",
        dependency_paths=paths,
        weight=None,
        support=0.0,
        refutation=0.0,
    )


class NewCaseEvidence(ActiveCaseEvidence):
    """Uses existing current Knowledge, Artifact, task, acquisition and scoring owners."""

    root_version = "new-case-evidence-template/1.0.0"
    root_prefix = "new-case-root:"
    root_filename = "root.json"

    def load_measurements(self, directory, objects):
        from packages.contracts import Artifact

        self.measurement = json.loads((directory / "measurements.json").read_bytes())
        self.tool_assets = tuple(
            Artifact.model_validate_json(json.dumps(a))
            for a in json.loads((directory / "tool-artifacts.json").read_bytes())
        )
        from .case_qc import geometry_quality

        mask_assets = {a.extensions["output_port"]: a for a in self.tool_assets}
        self.quality = geometry_quality(
            objects.read_bounded(mask_assets["disc_mask"].uri, 4194304),
            objects.read_bounded(mask_assets["cup_mask"].uri, 4194304),
        )
        if self.measurement["input_checksum"] != self.image_artifact.checksum:
            raise ValueError("new_case_measurement_input")

    def build_structure(self):
        return template()

    def knowledge_catalog(self):
        return load_catalog()

    def source_reader(self, original):
        return RoundCaseEvidence(original, self.knowledge_catalog())

    def __init__(self, sample_id, *, registered=None):
        from apps.knowledge_source_tools import repository
        from apps.patient_service_assembly import OBJECT_ROOTS
        from apps.vectorengine_single_case import write_once
        from db.runtime.repository.case_pipeline import identity
        from db.runtime.repository.dataset_input import DatasetInputRepository
        from packages.contracts import Artifact
        from packages.dataset_adapters.service import VisibleInputReader
        from packages.object_store import LocalObjectStore

        from .case_execution import register

        if registered is not None:
            from .generic_input import validate_registered

            registered = validate_registered(sample_id, registered)
        directory, binding, runtime, objects = registered or register(sample_id)
        self.directory, self.binding = directory, binding
        self.runtime, self.objects = runtime, objects
        self.visible = VisibleInputReader(DatasetInputRepository(runtime), objects)
        self.visible_value = self.visible.read(binding["case_code"])
        self.image_artifact = self.visible_value.case.artifacts[0]
        self.load_measurements(directory, objects)
        if self.root_version == "new-case-evidence-template/1.0.0":
            write_once(directory / "geometry-quality.json", self.quality)
        self.structure = self.build_structure()
        self.view = dict(
            nodes=[
                dict(
                    n,
                    node_code=identity(
                        "GOAL" if n["node_type"] == "GOAL" else "NODE",
                        binding["case_code"] + n["key"],
                    ),
                )
                for n in self.structure["nodes"]
            ]
        )
        root = dict(
            schema_version=self.root_version,
            case_code=binding["case_code"],
            input_checksum=self.image_artifact.checksum,
            template_checksum=digest(self.structure),
            nodes=self.view["nodes"],
            old_patient_results_reused=False,
        )
        raw = canonical(root)
        root_key = identity("TASK", self.root_prefix + binding["case_code"])
        saved = runtime.read_observation_bundle(root_key)
        if saved is not None and len(saved) != 1:
            raise ValueError("new_case_root_receipt_conflict")
        uri = saved[0].uri if saved else objects.promote(objects.stage(raw), checksum(raw))
        if saved and objects.read_bounded(uri, 4194304) != raw:
            raise ValueError("new_case_root_content_changed")
        artifact = self.image_artifact.model_dump(mode="json")
        artifact.update(
            asset_code=identity("ASSET", self.root_prefix + binding["case_code"] + checksum(raw)),
            artifact_type_code="ARTIFACT:RESEARCH_EVIDENCE_GRAPH",
            uri=uri,
            checksum=checksum(raw),
            media_type="application/json",
            dimensions=[],
            derived_from=[self.image_artifact.asset_code],
            measurements=[],
            extensions=dict(template_version=root["schema_version"]),
            provenance=dict(
                actor="TOOL_AGENT",
                created_at=self.image_artifact.provenance.created_at.isoformat(),
                generator_version="new-case-template-1.0.0",
                source_codes=[self.image_artifact.asset_code],
                input_checksums=[self.image_artifact.checksum],
            ),
        )
        self.root = Artifact.model_validate_json(json.dumps(artifact))
        self.root_bytes = raw
        runtime.register_observation_bundle((self.root,), root_key)
        write_once(directory / self.root_filename, root)
        self.knowledge = repository("editor")
        self.knowledge_objects = LocalObjectStore(OBJECT_ROOTS["knowledge"])
        self.fixed_source = Source(
            reference=ref(
                "source", "POLICY:NEW_CASE_UNEXECUTED_MODEL", digest(dict(model="not_executed"))
            ),
            locator="Project domain missing-model record, not medical literature",
            purpose="research_material",
            state="research_only",
            text=(
                "No classification model was executed on this case. The domain preserves "
                "a missing observation and null score, never another patient's result."
            ),
        )
        self.base = self.source_reader(
            SimpleNamespace(
                knowledge=self.knowledge,
                knowledge_objects=self.knowledge_objects,
                fixed_source=self.fixed_source,
            ),
        )
        self.manifest_path = Path("docs/experiments/active-acquisition-v1/manifest.json")
        self.manifest = load_manifest(self.manifest_path)
        self.alias_manifest, self.policies = None, {}

    def image(self, reference):
        image = self.image_artifact
        if (reference.code, reference.checksum) != (image.asset_code, image.checksum):
            raise ValueError("new_case_image_scope")
        raw = self.objects.read_bounded(image.uri, 4194304)
        if checksum(raw) != image.checksum:
            raise ValueError("new_case_image_changed")
        return raw

    def project(self, *, run_id, config):
        self.alias_manifest = None
        sources = self.base.read_sources() + (self.packet(), self.fixed_source)
        source_map = {}
        for chunk, source in zip(self.base.chunks, sources, strict=False):
            source_map.setdefault(chunk["source_id"], []).append(source)
        general = definitions(self.base.catalog)
        image = self.image_artifact
        h, w = image.dimensions
        image_binding = ImageBinding(
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
        values = self.measurement["measurements"]
        values = dict(values)
        if not self.quality["bbox_cdr_usable"]:
            values["vertical_cdr"] = None
            values["radial_rim_minimum_ratio"] = None
        numeric = {
            "criterion:AGE": ("age_years", float(self.binding["demographics"]["age"]), "year"),
            PILOTS[0]: ("vertical_cdr", values["vertical_cdr"], "ratio"),
            PILOTS[1]: ("radial_rim_minimum_ratio", values["radial_rim_minimum_ratio"], "ratio"),
        }
        observations, items, resources = [], [], {}
        for asset in (image, *self.tool_assets):
            resources[asset.asset_code] = ref("resource", asset.asset_code, asset.checksum)
        visible_ref = ref("resource", "RESOURCE:NEW_CASE_DEMOGRAPHICS", digest(self.visible_value))
        resources[visible_ref.code] = visible_ref
        measure_asset = next(
            a for a in self.tool_assets if a.extensions["output_port"] == "measurements"
        )
        for row in self.structure["items"]:
            identifier = row["id"]
            value, unit, method = None, "qualitative", "unobserved"
            if identifier in numeric and numeric[identifier][1] is not None:
                name, number, unit = numeric[identifier]
                value = canonical(dict(name=name, value=number, unit=unit)).decode()
                method = (
                    "provided_age" if identifier == "criterion:AGE" else "fresh_segman_geometry"
                )
            visual = image_binding if identifier == "visual:quality" else None
            if visual:
                method = "live_image_observation"
            observation = dict(
                status="obtained" if value else "unknown",
                method=method,
                unit=unit,
                value=value,
                image=visual.model_dump(mode="json") if visual else None,
            )
            observation["reference"] = ref(
                "observation",
                "OBS:NEW:" + identifier,
                digest(dict(case=self.binding["case_code"], **observation)),
            ).model_dump(mode="json")
            observations.append(observation)
            if identifier == "criterion:AGE":
                paths = [[visible_ref.code]]
            elif identifier in numeric:
                paths = [
                    [
                        image.asset_code,
                        *[
                            a.asset_code
                            for a in self.tool_assets
                            if a.extensions["output_port"] in {"cup_mask", "disc_mask"}
                        ],
                        measure_asset.asset_code,
                    ]
                ]
            elif visual or identifier == PILOTS[2]:
                paths = [[image.asset_code]]
            else:
                missing = ref(
                    "missing_resource",
                    "missing:" + identifier,
                    digest(dict(case=self.binding["case_code"], item=identifier)),
                )
                resources[missing.code] = missing
                paths = [[missing.code]]
            definition = general.get(
                identifier,
                dict(
                    sources=[],
                    examination=row["title"],
                    applicability_conditions="本例未取得该模型或检查结果，保留缺项。",
                    interpretation_rule=None,
                    interpretation_installed=False,
                ),
            )
            selected = [s for r in definition["sources"] for s in source_map.get(r["id"], ())]
            exact = bool(definition["sources"]) and all(
                r["id"] in source_map for r in definition["sources"]
            )
            if not exact:
                selected = [self.base.definition_source]
            if row["role"] == "model":
                selected = [self.fixed_source]
            body = dict(
                examination=definition.get("examination") or row["title"],
                applicability_conditions=definition["applicability_conditions"],
                interpretation=definition["interpretation_rule"]
                if exact and definition["interpretation_installed"]
                else None,
                interpretation_status="installed"
                if exact and definition["interpretation_installed"]
                else "missing",
                source_refs=[s.reference.model_dump(mode="json") for s in selected],
            )
            body["reference"] = ref(
                "definition", "POLICY:NEW:" + identifier, digest(body)
            ).model_dump(mode="json")
            if identifier in PILOTS:
                body = self.definition(identifier).model_dump(mode="json")
                selected = [self.packet()]
            items.append(
                dict(
                    id=identifier,
                    node=row["node"],
                    kind="model"
                    if row["role"] == "model"
                    else "measurement"
                    if row["kind"] == "descriptive"
                    else row["kind"],
                    role=row["role"],
                    observation_code=observation["reference"]["code"],
                    method=method,
                    unit=unit,
                    laterality="UNKNOWN",
                    source_codes=[s.reference.code for s in selected],
                    applicable=True,
                    correlation_group=row["correlation_group"],
                    dependence_block=row["dependence_block"],
                    dependency_paths=paths,
                    definition=body,
                    fixed_model_row=empty_model(row, observation, selected, paths)
                    if row["role"] == "model"
                    else None,
                    fixed_observation_ref=observation["reference"]
                    if row["role"] == "model"
                    else None,
                )
            )
        protocol = json.loads(
            Path("configs/research/evidence_review_protocol_v1.json").read_bytes()
        )
        policy = validate(
            TaskPolicyV2,
            dict(
                version="new-case-live-1.0.0",
                scale=ref("scale", "POLICY:CASE_REVIEW", digest(protocol)),
                aggregation_checksum=digest(POLICY),
                items=items,
                groups=self.structure["groups"],
            ),
        )
        scope = dict(
            case_code=self.binding["case_code"],
            run_id=run_id,
            graph=dict(
                role="graph",
                code=self.root.asset_code,
                version="new-case-evidence-template/1.0.0",
                checksum=self.root.checksum,
                kind="research_artifact",
                parent=None,
            ),
            visible_checksum=digest(self.visible_value),
            input_checksum=self.visible_value.checksum,
            policy_checksum=digest(policy),
        )
        goal = next(n["node_code"] for n in self.view["nodes"] if n["parent"] is None)
        material = dict(
            scope=scope,
            goal_code=goal,
            goal_summary="Single new SLO research; preserve missing observations and references.",
            visibility="approved_visible_projection/1.0.0",
            sources=list(sources),
            observations=observations,
            policy_ref=ref("policy", "POLICY:NEW_CASE_LIVE", digest(policy)),
            missing=["IOP", "OCT", "visual_field", "laterality", "classification_model"],
            memory=[],
        )
        scope["material_checksum"] = digest(material)
        state = DomainState(
            material=validate(Material, material),
            policy=policy,
            laterality="UNKNOWN",
            resources=tuple(resources.values()),
            prompt_checksum=next(r.prompt_checksum for r in config.roles if r.role == "review"),
            config_checksum=digest(config),
            nodes=tuple(sorted({i.node for i in policy.items} | {goal})),
        )
        state = prepare_visual_task(
            state, PILOTS[2], image=image_binding, instruction=RULES[PILOTS[2]]["examination"]
        )
        state, self.alias_manifest = prepare_source_aliases(state)
        self.policies[run_id] = digest(state.policy)
        return state, dict(
            schema_version="new-case-live-manifest/1.0.0",
            sample_id=self.binding["sample_id"],
            case_code=self.binding["case_code"],
            image_checksum=image.checksum,
            old_patient_results_reused=False,
            geometry_quality=self.quality,
            template_checksum=digest(self.structure),
            source_aliases=self.alias_manifest,
            observations_obtained=sum(o.status == "obtained" for o in state.material.observations),
            missing_model_not_replaced_with_old_score=True,
            task_count=len(items),
        )

    def verify(self, state):
        scope = state.material.scope
        if (scope.case_code, scope.input_checksum, scope.visible_checksum) != (
            self.binding["case_code"],
            self.visible_value.checksum,
            digest(self.visible_value),
        ):
            raise ValueError("new_case_scope_changed")
        if template() != self.structure or load_catalog() != self.base.catalog:
            raise ValueError("new_case_template_changed")
        if self.visible.read(self.binding["case_code"]) != self.visible_value:
            raise ValueError("new_case_visible_changed")
        if (
            self.runtime.read_case_artifact(self.binding["case_code"], self.root.asset_code)
            != self.root
        ):
            raise ValueError("new_case_root_changed")
        if self.objects.read_bounded(self.root.uri, 4194304) != self.root_bytes:
            raise ValueError("new_case_root_bytes")
        for asset in self.tool_assets:
            if (
                self.runtime.read_case_artifact(self.binding["case_code"], asset.asset_code)
                != asset
                or checksum(self.objects.read_bounded(asset.uri, 4194304)) != asset.checksum
            ):
                raise ValueError("new_case_observation_changed")
        original = self.base.read_sources() + (self.packet(),)
        sources, aliases = alias_sources(original)
        if (
            aliases != self.alias_manifest
            or sources + (self.fixed_source,) != state.material.sources
        ):
            raise ValueError("new_case_source_changed")
        if self.policies.get(state.material.scope.run_id) != digest(state.policy):
            raise ValueError("new_case_policy_changed")
        self.image(ref("image", self.image_artifact.asset_code, self.image_artifact.checksum))
