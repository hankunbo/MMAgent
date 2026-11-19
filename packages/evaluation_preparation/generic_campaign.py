"""Composition of current owners; no two-case campaign or previous failed run required."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from apps.vectorengine_single_case import write_once
from db.runtime.repository.planner import PlannerRepository
from db.runtime.repository.research import ResearchRunRepository
from packages.grounding.models import digest
from packages.object_store import checksum

from .case_live import requests_for
from .comprehensive_campaign import ComprehensiveCampaign
from .comprehensive_materials import ComprehensiveEvidence
from .dynamic_model import read_model
from .generic_input import load, provider_config
from .partial_observation import schedule


class GenericEvidence(ComprehensiveEvidence):
    root_version = "generic-source-case-root/1.0.0"
    root_filename = "generic-root.json"

    def __init__(self, spec, registered):
        self.spec = spec
        self.unified = spec["schema_version"] in {
            "fairvlmed-run-input/1.2.0",
            "fairvlmed-run-input/1.3.0",
            "fairvlmed-run-input/1.4.0",
            "fairvlmed-run-input/1.5.0",
            "fairvlmed-run-input/1.6.0",
            "fairvlmed-run-input/1.7.0",
            "fairvlmed-run-input/1.8.0",
            "fairvlmed-run-input/1.9.0",
            "fairvlmed-run-input/1.10.0",
            "fairvlmed-run-input/1.11.0",
        }
        if self.unified:
            self.root_version = "generic-source-case-root/2.0.0"
            self.root_filename = "generic-root-v2.json"
        self.root_prefix = "generic-case-root:" + spec["key"] + ":"
        super().__init__(spec["sample"], registered=registered, tools_optional=True)
        if self.unified:
            from packages.contracts import Artifact
            from packages.dataset_adapters.public_context import read

            self.context_artifact = Artifact.model_validate_json(
                json.dumps(self.binding["public_context_artifact"])
            )
            self.input_context = read(
                self.runtime, self.objects, self.visible_value, self.context_artifact
            )

    def knowledge_catalog(self):
        if self.unified:
            from .unified_requirements import catalog

            value = catalog()
            if digest(value) != self.spec["evidence_catalog_checksum"]:
                raise ValueError("generic_evidence_catalog_changed")
            return value
        return super().knowledge_catalog()

    def source_reader(self, original):
        if self.unified:
            from .unified_requirements import UnifiedSourceReader

            return UnifiedSourceReader(original, self.knowledge_catalog())
        return super().source_reader(original)

    def build_structure(self):
        result = super().build_structure()
        if self.unified:
            from .evidence_inventory import inventory

            value = self.knowledge_catalog()
            result["inventory"] = inventory(
                value, result["requirement_ids"], mappings=value["merges"]
            )
        return result

    def project(self, *, run_id, config):
        state, manifest = super().project(run_id=run_id, config=config)
        if self.unified:
            from .unified_requirements import add_context

            state = add_context(self, state)
            self.policies[run_id] = digest(state.policy)
            manifest["inventory"] = self.structure["inventory"]
        return state, manifest

    def verify(self, state):
        super().verify(state)
        if self.unified:
            from packages.dataset_adapters.public_context import read

            if (
                read(self.runtime, self.objects, self.visible_value, self.context_artifact)
                != self.input_context
            ):
                raise ValueError("generic_input_context_changed")

    def load_measurements(self, directory, objects):
        from .generic_ports import read
        from .quality import mask_quality

        ports = read(directory, self.binding, self.runtime, objects)
        if ports is None:
            self.measurement, self.tool_assets = None, ()
            self.soft_quality = dict(
                quality_factor=0.0,
                measurement_available=False,
                reasons=["no_segmentation_artifacts"],
                stop_case=False,
            )
            self.quality = dict(self.soft_quality, bbox_cdr_usable=False)
            return
        assets, raw, self.measurement = ports
        self.tool_assets = tuple(assets.values())
        self.soft_quality = mask_quality(raw["disc_mask"], raw["cup_mask"])
        self.quality = dict(
            self.soft_quality, bbox_cdr_usable=self.soft_quality["measurement_available"]
        )


class GenericCampaign(ComprehensiveCampaign):
    """Reuse phase execution/acceptance, deliberately not historical constructors."""

    def __init__(self, spec, registered):
        self.spec = spec
        self.structured_completion = spec["schema_version"] != "fairvlmed-run-input/1.0.0"
        if self.structured_completion:
            from .generic_aliases import GenericPhase

            self.phase_class = GenericPhase
        self.sample = spec["sample"]
        self.version = "generic-v1:" + spec["key"]
        self.export_suffix = "_通用流程_" + spec["key"]
        self.base = GenericEvidence(spec, registered)
        self.provider_path = spec["provider_configuration"]
        self.config = provider_config(spec)
        self.requests, self.target = requests_for(self.base)
        self.contexts = ResearchRunRepository(self.target)
        self.sessions = PlannerRepository(self.target)
        self.output = self.base.directory / "evidence"
        self.output.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.max_new_requests = spec["max_requests"]
        self.duration_seconds = spec["seconds"]
        seed, _ = self.base.project(run_id="RUN:GENERIC_PLAN_ONLY", config=self.config)
        self.task_plan = schedule(
            seed.policy.requirements,
            measured={i.id for i in seed.policy.items},
            modalities=("SLO", "age")
            if self.base.unified and self.base.input_context["age"]["value"] is not None
            else ("SLO",),
        )
        self.visual_targets = tuple(
            v["id"] for v in self.task_plan if v["action"] == "observe_partial"
        )
        self.max_vision_requests = len(self.visual_targets)
        required_reviews = (len(seed.policy.requirements) + 3) // 3 if self.base.unified else 5
        if self.max_vision_requests + required_reviews > self.max_new_requests:
            raise ValueError("generic_plan_budget_exceeded")
        write_once(self.output / "task-plan.json", self.task_plan)

    def prior(self):
        # New explicitly selected run, not continuation of a two-case repair campaign.
        return dict(reservations=0, input_spec_checksum=digest(self.spec))

    def execution_session(self):
        from .generic_execution_lock import INPUT_VERSIONS, current, scoped_lock

        if self.spec["schema_version"] not in INPUT_VERSIONS:
            return super().execution_session()
        if current(self.spec["key"]) is None:
            raise ValueError("generic_campaign_original_execution_lease_required")
        return scoped_lock(self.output / "lock")

    def guard(self, *, dispatch=False):
        from .generic_control import guard

        guard(self.spec["key"])
        return super().guard(dispatch=dispatch)

    def phase_completed(self, phase):
        return phase["accepted"] or phase["evidence_limited"]

    def observation_projector(self):
        from .generic_projection import FreshAcquisitionProjection

        return FreshAcquisitionProjection(self.base, self.config)

    def request_key(self, target):
        from .generic_input import owner_key

        return owner_key(self.spec, "phase", target or "review")

    def archive_key(self, target):
        from .generic_input import owner_key

        return owner_key(self.spec, "memory", target)

    def cancellation_signal(self):
        from .generic_control import Cancellation

        return Cancellation(self.spec["key"])

    def read_model(self):
        if self.spec["fairclip_configuration"] is None:
            return None
        return read_model(
            self.base,
            self.target,
            revision="generic-v1",
            configuration=self.spec["fairclip_configuration"],
        )

    def freeze_observations(self):
        return dict(
            tools=digest(self.base.tools),
            plan=digest(self.task_plan),
            image=self.base.image_artifact.checksum,
            display=digest(self.base.view_policy),
            model=digest(self.read_model()),
        )

    def protocol_fields(self):
        """Rebuild exact current identity without creating or renewing a window."""
        if load(self.spec["key"]) != self.spec:
            raise ValueError("generic_input_spec_changed")
        files = [
            p
            for root in ("packages/evaluation_preparation", "packages/dataset_adapters")
            for p in sorted(Path(root).glob("*.py"))
        ]
        files += [
            Path(self.provider_path),
            Path("apps/fairvlmed_case.py"),
            Path("apps/vectorengine_live_assembly.py"),
            Path("apps/vectorengine_round_materials.py"),
            Path("apps/active_evidence_materials.py"),
            Path("packages/llm_provider/transport.py"),
            Path("packages/llm_provider/review_citations.py"),
            Path(
                "configs/prompts/agents/"
                + next(r.prompt_version for r in self.config.roles if r.role == "review")
                + ".txt"
            ),
            Path("configs/prompts/agents/vision-partial-1.0.0.txt"),
        ]
        if self.base.unified:
            from .unified_requirements import CONFIG, NEI

            files += [CONFIG, NEI, Path("configs/research/vectorengine_round2_materials.json")]
        if self.spec.get("upstream_profile"):
            from packages.evidence_reuse.integration import implementation

            files += [Path(p) for p in implementation()]
        if self.spec["schema_version"] in {
            "fairvlmed-run-input/1.7.0",
            "fairvlmed-run-input/1.8.0",
            "fairvlmed-run-input/1.9.0",
            "fairvlmed-run-input/1.10.0",
            "fairvlmed-run-input/1.11.0",
        }:
            files += [
                Path("packages/llm_provider/network_route.py"),
                Path("packages/llm_provider/config.py"),
                Path("packages/llm_provider/worker.py"),
                Path("packages/llm_provider/provider.py"),
                Path("packages/evidence_details/review_batches.py"),
                Path("agents/provider_agents/batched_review.py"),
            ]
        if self.spec["schema_version"] in {
            "fairvlmed-run-input/1.8.0",
            "fairvlmed-run-input/1.9.0",
        }:
            files += [
                Path("packages/llm_provider/model_selection.py"),
                Path(self.spec["model_selection"]["path"]),
                Path(self.spec["model_selection"]["selection"]["profile"]),
            ]
        expected = dict(
            schema_version="generic-evidence-campaign/1.0.0",
            input=digest(self.spec),
            sample=self.sample,
            case_code=self.base.binding["case_code"],
            provider_checksum=digest(self.config),
            previous=self.prior(),
            observations=self.freeze_observations(),
            max_new_requests=self.max_new_requests,
            max_cumulative_graph_requests=40,
            seconds=self.duration_seconds,
            implementation={str(p): checksum(p.read_bytes()) for p in files},
            clinical_acceptance=False,
        )
        return expected

    def freeze(self):
        expected = self.protocol_fields()
        path = self.output / "protocol.json"
        if path.exists():
            value = json.loads(path.read_bytes())
            if {k: v for k, v in value.items() if k not in {"started_at", "deadline"}} != expected:
                raise ValueError("generic_frozen_run_changed")
        else:
            now = datetime.now(UTC)
            value = dict(
                expected,
                started_at=now.isoformat(),
                deadline=(now + timedelta(seconds=self.duration_seconds)).isoformat(),
            )
            write_once(path, value)
        self.protocol = value
        return value

    def archive_completed(self, result):
        # Existing archive owner derives valid branch flows from original artifacts.
        # No new publication or use of historical patient values is authorized here.
        result = super().archive_completed(result)
        if (
            not self.spec.get("upstream_profile")
            or not result["complete"]
            or not any(
                p["id"] == "criterion:AGE" and p["action"] == "review_measurement"
                for p in self.task_plan
            )
        ):
            return result
        from apps.case_memory_store import assembly

        docs = self.documents(result["phases"][-1]["run_id"])
        artifact = self.base.runtime.read_observation_bundle(
            docs.key(getattr(self, "report_key", "dynamic-case-report"))
        )[0]
        writer, owners = assembly("writer")
        key = self.archive_key("criterion:AGE")
        record = owners.artifacts.prepare(
            key=key,
            case_code=self.base.binding["case_code"],
            run_id=docs.run_id,
            origin_code=artifact.asset_code,
            kind="tool_experience",
            target="criterion:AGE",
        )
        saved = writer.create(record)
        write_once(
            self.output / "upstream-memory-archive.json",
            dict(key=key, checksum=saved.checksum, target="criterion:AGE"),
        )
        return dict(result, memory_archived=result["memory_archived"] + 1)

    def process_evaluations(self):
        rows = super().process_evaluations()
        model = self.read_model()
        if model is not None and "model:FAIRCLIP" not in {r["id"] for r in rows}:
            rows.append(
                dict(
                    id="model:FAIRCLIP",
                    action="review_confirmed_model",
                    availability_score=1.0,
                    inspected_fraction=1.0,
                    effective_quality=0.0,
                    gap_score=1.0,
                    disease_direction=None,
                    raw_quality=None,
                    interpretation_status="pending_review",
                    quality_reason="rating_not_yet_obtained",
                    source=model["source"],
                )
            )
        return rows


def source_plan(spec, registered):
    """Current Knowledge -> requirements, before any native tool or paid request."""
    from .dynamic_materials import DynamicCaseEvidence

    evidence = GenericEvidence(spec, registered)
    config = provider_config(spec)
    state, _ = (
        evidence.project(run_id="RUN:GENERIC_SOURCE_PLAN", config=config)
        if evidence.unified
        else DynamicCaseEvidence.project(evidence, run_id="RUN:GENERIC_SOURCE_PLAN", config=config)
    )
    value = dict(
        schema_version="generic-source-plan/1.0.0",
        input_checksum=evidence.image_artifact.checksum,
        dataset=spec["dataset"],
        target="glaucoma",
        case_code=evidence.binding["case_code"],
        requirements=[
            dict(
                id=r.id,
                definition=r.definition.model_dump(mode="json"),
                input_modalities=list(r.input_modalities),
                basis_kind=r.basis_kind,
            )
            for r in state.policy.requirements
        ],
        sources=[s.reference.model_dump(mode="json") for s in state.material.sources],
        old_patient_answers_used=False,
    )
    if evidence.unified:
        value["requirement_inventory"] = evidence.structure["inventory"]
        value["input_context"] = evidence.input_context
    if spec.get("upstream_profile"):
        from packages.evidence_reuse.integration import memory_owner, prepare

        path = evidence.directory / "source-plan.json"
        if path.exists():
            old = json.loads(path.read_bytes())
            from packages.evidence_reuse.requirements import align_acquisition_requirements

            comparable = dict(
                value,
                requirements=align_acquisition_requirements(
                    old["requirements"], value["requirements"]
                ),
            )
            if {k: v for k, v in old.items() if k != "upstream"} != comparable:
                raise ValueError("upstream_existing_plan_changed")
            # Historical plan remains exact; actual dispatch rechecks current eligibility.
            return old
        value["upstream"] = prepare(evidence, state, value, memory=memory_owner())
    write_once(evidence.directory / "source-plan.json", value)
    return value
