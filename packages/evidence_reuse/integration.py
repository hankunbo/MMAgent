"""Opt-in glue into the original generic run. No additional run/database/executor."""

import json
from pathlib import Path

from apps.vectorengine_single_case import write_once
from packages.dataset_adapters.components import visible_components
from packages.grounding.models import digest
from packages.memory.archive import WorkflowMemory
from packages.object_store import checksum

from .cache import ShadowCache
from .requirements import align_acquisition_requirements, audit
from .skills import compile_plans, recheck_plans

PROFILE = "evidence-reuse/1.0.0"


def implementation():
    paths = sorted(Path("packages/evidence_reuse").glob("*.py"))
    paths += [
        Path("packages/dataset_adapters/components.py"),
        Path("packages/memory/rule_archive.py"),
        Path("packages/memory/runtime_graph_archive.py"),
        Path("db/memory/repository.py"),
        Path("configs/research/evidence_skill_ports_v1.json"),
        Path("configs/research/evidence_context_roles_v1.json"),
        Path("packages/evaluation_preparation/quality_materials.py"),
    ]
    return {str(p): checksum(p.read_bytes()) for p in paths}


def memory_owner():
    from apps.case_memory_store import PRIVATE, assembly

    if not PRIVATE.exists():
        return None  # No Memory remains a valid baseline; faults aren't empty Memory.
    repo, _ = assembly()
    return WorkflowMemory(repo)


def prepare(evidence, state, plan, *, memory):
    evidence.verify(state)
    components = visible_components(evidence.visible_value, evidence.input_context)
    result = dict(
        schema_version=PROFILE,
        components=components.model_dump(mode="json"),
        requirement_resolution=audit(evidence.knowledge_catalog(), plan, components),
        skills=compile_plans(plan, components, memory=memory),
        source_plan_checksum=digest(plan),
        cache_policy="shadow_full_current_checks",
        execution_authorized=False,
    )
    evidence.verify(state)
    recheck_plans(result["skills"], plan, components, memory=memory)
    return result


class DispatchCheck:
    """Current original owners before native launch; immutable plan is not authority."""

    def __init__(self, spec, registered, *, memory=None):
        from packages.evaluation_preparation.generic_campaign import GenericEvidence
        from packages.evaluation_preparation.generic_input import provider_config

        self.spec, self.registered = spec, registered
        self.memory = memory
        self.evidence = GenericEvidence(spec, registered)
        self.config = provider_config(spec)
        self.plan = json.loads((registered[0] / "source-plan.json").read_bytes())
        self.upstream = self.plan["upstream"]
        self.original_plan = {k: v for k, v in self.plan.items() if k != "upstream"}
        if self.upstream["source_plan_checksum"] != digest(self.original_plan):
            raise ValueError("upstream_source_plan_changed")
        self.cache = ShadowCache()

    def current(self):
        from packages.evaluation_preparation.generic_input import validate_registered
        from packages.evaluation_preparation.generic_repair import implementation_matches

        if not implementation_matches(self.spec, "upstream_implementation", implementation()):
            raise ValueError("upstream_implementation_changed")
        validate_registered(self.spec["sample"], self.registered)
        state, _ = self.evidence.project(run_id="RUN:GENERIC_SOURCE_PLAN", config=self.config)
        self.evidence.verify(state)
        requirements = [
            dict(
                id=r.id,
                definition=r.definition.model_dump(mode="json"),
                input_modalities=list(r.input_modalities),
                basis_kind=r.basis_kind,
            )
            for r in state.policy.requirements
        ]

        # Method-specific instructions may follow the original measured/visual
        # variant. Current medical sources/rules and all other fields stay exact.
        aligned = align_acquisition_requirements(self.original_plan["requirements"], requirements)
        components = visible_components(self.evidence.visible_value, self.evidence.input_context)
        if components.model_dump(mode="json") != self.upstream["components"]:
            raise ValueError("upstream_input_components_changed")
        recheck_plans(self.upstream["skills"], self.original_plan, components, memory=self.memory)
        return dict(
            components=digest(components),
            requirements=digest(
                [{k: v for k, v in r.items() if k != "basis_kind"} for r in aligned]
            ),
            sources=digest([s.reference for s in state.material.sources]),
        )

    def check(self):
        # Cache candidate is patient-free port topology. Full owner checks on both sides.
        return self.cache.project(
            namespace="skill_port_projection/1.0.0",
            bindings=dict(
                implementation=self.spec["upstream_implementation"],
                skills=digest(self.upstream["skills"]),
            ),
            compute=lambda: [
                dict(requirement=p["requirement_id"], steps=p["steps"])
                for p in self.upstream["skills"]["plans"]
            ],
            revalidate=self.current,
        )

    def needs_segman(self):
        self.check()
        return any(
            s["port"] == "segman"
            for p in self.upstream["skills"]["plans"]
            if not p["missing_components"]
            for s in p["steps"]
        )

    def finish(self, records):
        self.check()
        value = dict(
            schema_version="upstream-use-receipt/1.0.0",
            source_plan_checksum=digest(self.plan),
            case_code=self.registered[1]["case_code"],
            input_checksum=self.registered[1]["image"]["checksum"],
            skills=self.upstream["skills"],
            tools=records,
            cache_events=self.cache.events,
            current_verified=True,
            execution_owner="original_native_ports",
            clinical_publication=False,
        )
        path = self.registered[0] / "upstream-use.json"
        if path.exists():
            old = json.loads(path.read_bytes())
            if {k: v for k, v in old.items() if k != "cache_events"} != {
                k: v for k, v in value.items() if k != "cache_events"
            }:
                raise ValueError("upstream_recovery_receipt_conflict")
            return old
        write_once(path, value)
        return value


def saved_report_context(campaign):
    """Read exact execution receipt; report persistence remains Runtime's responsibility."""
    directory = campaign.base.directory
    plan = json.loads((directory / "source-plan.json").read_bytes())
    receipt = json.loads((directory / "upstream-use.json").read_bytes())
    if (
        receipt["source_plan_checksum"] != digest(plan)
        or receipt["case_code"] != campaign.base.binding["case_code"]
        or receipt["input_checksum"] != campaign.base.image_artifact.checksum
    ):
        raise ValueError("upstream_report_binding")
    return dict(plan=plan["upstream"], execution=receipt)
