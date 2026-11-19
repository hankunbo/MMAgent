"""Opt-in source-first material: requirements are not empty result slots.

Archived Knowledge is read by its current owner. No old patient graph, ratings,
mandatory classifier, or required segmentation files are used to build items.
This preparation port does not send APIs or authorize a stopped run.
"""

from apps.active_evidence_materials import PILOTS
from apps.vectorengine_case_materials import ref
from apps.vectorengine_round_materials import definitions, load_catalog
from packages.evidence_details.acceptance_models import DomainState, TaskPolicyV3
from packages.evidence_scoring.dynamic_policy import POLICY
from packages.grounding.models import canonical, digest, validate
from packages.llm_provider.models import Material
from packages.object_store import checksum

from .case_materials import NewCaseEvidence


class DynamicCaseEvidence(NewCaseEvidence):
    root_version = "source-driven-case-root/1.0.0"
    root_prefix = "source-driven-case-root:"
    root_filename = "dynamic-root-v1.json"
    targeted_acquisition = False  # Legacy acquisition expects compulsory policy slots.

    def load_measurements(self, directory, objects):
        self.measurement, self.tool_assets = None, ()
        self.quality = dict(bbox_cdr_usable=False, reasons=["no_segmentation_artifacts"])
        files = (directory / "measurements.json", directory / "tool-artifacts.json")
        if any(p.exists() for p in files):
            if not all(p.is_file() for p in files):
                raise ValueError("dynamic_partial_tool_receipt")
            super().load_measurements(directory, objects)
            for asset in self.tool_assets:
                if (
                    self.runtime.read_case_artifact(self.binding["case_code"], asset.asset_code)
                    != asset
                ):
                    raise ValueError("dynamic_tool_owner_binding")
                if checksum(objects.read_bounded(asset.uri, 4194304)) != asset.checksum:
                    raise ValueError("dynamic_tool_object_changed")

    def build_structure(self):
        catalog = self.knowledge_catalog() if hasattr(self, "knowledge_catalog") else load_catalog()
        # The archived catalog is a Knowledge definition index. Do not open the
        # old patient's node/review list, use list positions, or copy its scores.
        ids = sorted(
            {k for k, v in catalog["tasks"].items() if v["selected_source_ids"]} | set(PILOTS)
        )
        return dict(
            definition_catalog_checksum=digest(catalog),
            nodes=[dict(key="glaucoma", node_type="GOAL", parent=None)]
            + [dict(key=k, node_type="REQUIREMENT", parent="glaucoma") for k in ids],
            requirement_ids=ids,
        )

    def project(self, *, run_id, config):
        sources = self.base.read_sources() + (self.packet(),)
        source_map = {}
        for chunk, source in zip(self.base.chunks, sources, strict=False):
            source_map.setdefault(chunk["source_id"], []).append(source)
        general = definitions(self.base.catalog)
        resources = [
            ref("resource", a.asset_code, a.checksum)
            for a in (self.image_artifact, *self.tool_assets)
        ]
        requirements, items, observations, groups = [], [], [], []
        measured = self.measurement["measurements"] if self.measurement else {}
        values = (
            {
                PILOTS[0]: measured.get("vertical_cdr"),
                PILOTS[1]: measured.get("radial_rim_minimum_ratio"),
            }
            if self.quality["bbox_cdr_usable"]
            else {}
        )
        non_slo = {
            "criterion:IOP": ("tonometry",),
            "descriptive:OCT_REFERENCE": ("OCT",),
            "association:STRUCTURE_FUNCTION_MATCH": ("SLO", "visual_field"),
            "association:INTEREYE_COMPARISON": ("bilateral_images",),
        }
        for identifier in self.structure["requirement_ids"]:
            if identifier in PILOTS:
                definition = self.definition(identifier)
            else:
                d = general[identifier]
                selected = [s for r in d["sources"] for s in source_map.get(r["id"], ())]
                if not selected or any(r["id"] not in source_map for r in d["sources"]):
                    raise ValueError("dynamic_requirement_source_unresolved")
                body = dict(
                    examination=d["examination"] or identifier,
                    applicability_conditions=d["applicability_conditions"],
                    interpretation=d["interpretation_rule"]
                    if d["interpretation_installed"]
                    else None,
                    interpretation_status="installed"
                    if d["interpretation_installed"]
                    else "missing",
                    source_refs=[s.reference.model_dump(mode="json") for s in selected],
                )
                from packages.llm_provider.models import TaskDefinition

                definition = validate(
                    TaskDefinition,
                    dict(
                        **body,
                        reference=ref("definition", "POLICY:DYNAMIC:" + identifier, digest(body)),
                    ),
                )
            declared = self.base.catalog["tasks"][identifier].get("input_modalities")
            if declared and tuple(declared) != ("SLO",):
                non_slo[identifier] = tuple(declared)
            modalities = non_slo.get(identifier, ("SLO",))
            value = values.get(identifier)
            requirements.append(
                dict(
                    id=identifier,
                    definition=definition,
                    input_modalities=modalities,
                    basis_kind="quantitative" if identifier in PILOTS[:2] else "qualitative",
                    missing_reason=None
                    if value is not None
                    else (
                        "patient_modality_not_provided"
                        if identifier in non_slo
                        else "observation_not_obtained"
                    ),
                )
            )
            groups.append(
                dict(
                    id="requirement:" + identifier,
                    slots=[[identifier]],
                    importance=dict(
                        slo_image_research=0.0 if identifier in non_slo else 1.0,
                        comprehensive_research=1.0,
                    ),
                )
            )
            if value is None:
                continue
            observation = dict(
                status="obtained",
                method="registered_segman_geometry",
                unit="ratio",
                value=canonical(
                    dict(value=value, input_checksum=self.image_artifact.checksum)
                ).decode(),
                image=None,
            )
            observation["reference"] = ref(
                "observation", "OBS:DYNAMIC:" + identifier, digest(observation)
            )
            observations.append(observation)
            items.append(
                dict(
                    id=identifier,
                    node=identifier,
                    kind="criterion",
                    role="diagnostic",
                    observation_code=observation["reference"].code,
                    method=observation["method"],
                    unit="ratio",
                    laterality="UNKNOWN",
                    applicable=True,
                    source_codes=[r.code for r in definition.source_refs],
                    definition=definition,
                    correlation_group="optic_nerve_geometry",
                    dependence_block="same_slo_image",
                    dependency_paths=[[r.code for r in resources]],
                )
            )
        import json
        from pathlib import Path

        rubric = json.loads(Path("configs/research/evidence_review_protocol_v1.json").read_bytes())
        policy = validate(
            TaskPolicyV3,
            dict(
                version="source-driven-case-1.0.0",
                aggregation_checksum=digest(POLICY),
                scale=ref("scale", "POLICY:CASE_REVIEW", digest(rubric)),
                items=items,
                requirements=requirements,
                groups=groups,
            ),
        )
        goal = self.view["nodes"][0]["node_code"]
        scope = dict(
            case_code=self.binding["case_code"],
            run_id=run_id,
            input_checksum=self.visible_value.checksum,
            visible_checksum=digest(self.visible_value),
            policy_checksum=digest(policy),
            graph=dict(
                role="graph",
                code=self.root.asset_code,
                version=self.root_version,
                checksum=self.root.checksum,
                kind="research_artifact",
                parent=None,
            ),
        )
        raw = dict(
            scope=scope,
            goal_code=goal,
            goal_summary="Glaucoma research: source-backed requirements, obtained evidence only.",
            visibility="approved_visible_projection/1.0.0",
            sources=list(sources),
            observations=observations,
            policy_ref=ref("policy", "POLICY:DYNAMIC_CASE", digest(policy)),
            missing=[r["id"] for r in requirements if r["missing_reason"]],
            memory=[],
        )
        scope["material_checksum"] = digest(raw)
        state = DomainState(
            material=validate(Material, raw),
            policy=policy,
            laterality="UNKNOWN",
            resources=tuple(resources),
            prompt_checksum=next(r.prompt_checksum for r in config.roles if r.role == "review"),
            config_checksum=digest(config),
            nodes=tuple(sorted({goal} | {i["node"] for i in items})),
        )
        self.policies[run_id] = digest(policy)
        return state, dict(
            schema_version="dynamic-case-material/1.0.0",
            requirement_count=len(requirements),
            evidence_count=len(items),
            geometry_quality=self.quality,
            old_patient_rows_used=False,
            missing_classifier_slot_created=False,
            coverage_weighting="equal_source_backed_requirement_not_clinical_importance",
            acquisition_loop_installed=False,
        )

    def verify(self, state):
        scope = state.material.scope
        if (scope.case_code, scope.input_checksum, scope.visible_checksum) != (
            self.binding["case_code"],
            self.visible_value.checksum,
            digest(self.visible_value),
        ):
            raise ValueError("dynamic_case_scope_changed")
        if self.visible.read(self.binding["case_code"]) != self.visible_value:
            raise ValueError("dynamic_visible_changed")
        if (
            self.build_structure() != self.structure
            or self.base.catalog != self.knowledge_catalog()
        ):
            raise ValueError("dynamic_requirement_origin_changed")
        if self.policies.get(scope.run_id) != digest(state.policy):
            raise ValueError("dynamic_requirement_snapshot_changed")
        if self.base.read_sources() + (self.packet(),) != state.material.sources:
            raise ValueError("dynamic_current_sources_changed")
        for asset in (self.root, *self.tool_assets):
            if (
                self.runtime.read_case_artifact(self.binding["case_code"], asset.asset_code)
                != asset
            ):
                raise ValueError("dynamic_current_artifact_changed")
            if checksum(self.objects.read_bounded(asset.uri, 4194304)) != asset.checksum:
                raise ValueError("dynamic_artifact_bytes_changed")
        self.image(ref("image", self.image_artifact.asset_code, self.image_artifact.checksum))
