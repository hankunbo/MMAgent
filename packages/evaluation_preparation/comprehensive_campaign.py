"""One bounded full-inventory increment, using original live/acceptance owners."""

import json
from pathlib import Path

from apps.vectorengine_single_case import write_once
from packages.evidence_details.acceptance_models import DomainState, read_receipt, read_records
from packages.grounding.models import digest, validate
from packages.llm_provider.config import read_provider_config
from packages.object_store import checksum
from packages.research_runtime.agent_records import RuntimeAgentJournal

from .comprehensive_materials import ComprehensiveEvidence, ComprehensivePhase
from .model_repair_campaign import ModelRepairCampaign
from .partial_observation import evaluate, schedule
from .quality_campaign import QualityCampaign


class ComprehensiveCampaign(ModelRepairCampaign):
    version = "13-full-evidence-v1"
    export_suffix = "_全清单与部分观察版"
    evidence_class = ComprehensiveEvidence
    phase_class = ComprehensivePhase
    max_new_requests = 12
    max_vision_requests = 7
    duration_seconds = 2700

    def __init__(self, sample):
        super().__init__(sample)
        self.quality_previous = QualityCampaign(
            sample, revision="v1" if "00002" in sample else "wire-r1"
        )
        self.config = read_provider_config(
            json.loads(Path("configs/agents/vectorengine_full_evidence_v1.json").read_bytes())
        )
        self.extra_freeze_files = tuple(
            Path(p)
            for p in (
                "packages/llm_provider/models.py",
                "agents/provider_agents/service.py",
                "packages/evidence_details/acceptance.py",
                "packages/evidence_details/acceptance_models.py",
                "configs/prompts/agents/vision-partial-1.0.0.txt",
                "configs/prompts/agents/review-batch-1.6.0.txt",
                "packages/llm_provider/review_citations.py",
                "packages/memory/archive.py",
                "packages/memory/runtime_graph_archive.py",
            )
        )
        # Planning is source-derived and offline; no placeholder graph results.
        seed, _ = self.base.project(run_id="RUN:FULL_PLAN_ONLY", config=self.config)
        self.task_plan = schedule(
            seed.policy.requirements, measured={i.id for i in seed.policy.items}
        )
        self.visual_targets = tuple(
            p["id"] for p in self.task_plan if p["action"] == "observe_partial"
        )
        if len(self.visual_targets) > self.max_vision_requests:
            raise ValueError("full_plan_exceeds_bounded_vision_budget")
        write_once(self.output / "task-plan.json", self.task_plan)

    def prior(self):
        value = self.quality_previous.prior()
        used = self.quality_previous.usage()
        return dict(
            quality_prior=value, quality_requests=used, reservations=value["reservations"] + used
        )

    def paths(self):
        return [self.phase_path(t) for t in self.visual_targets] + [self.output / "review"]

    def phase_path(self, target):
        return self.output / ("vision-" + digest(target)[7:19] if target else "review")

    def freeze_observations(self):
        # Old qualitative answers are not supplied as fresh VLM observations.
        return dict(
            tools=checksum(
                (self.base.directory / "13-full-evidence-tools-v2/result.json").read_bytes()
            ),
            task_plan=digest(self.task_plan),
            display=digest(self.base.view_policy),
        )

    def before_review(self):
        for target in self.visual_targets:
            # Leave space for the selected acquired observations' review batches.
            self.guard()
            result = self.execute_phase(target)
            print(
                json.dumps(
                    dict(
                        sample=self.sample,
                        target=target,
                        accepted=result["accepted"],
                        error=result.get("error"),
                    )
                ),
                flush=True,
            )

    def accepted_observations(self):
        values = {}
        project = self.observation_projector()
        for target in self.visual_targets:
            path = self.phase_path(target) / "result.json"
            if not path.exists():
                continue
            result = json.loads(path.read_bytes())
            docs = self.documents(result["run_id"])
            if docs.read("dynamic-phase-result") != (result,):
                raise ValueError("full_phase_result_binding")
            if not result["accepted"]:
                continue
            initial = validate(DomainState, docs.read("dynamic-phase-input")[0])
            phase = ComprehensivePhase(self.base, target=target)
            expected, _ = project(phase, result["run_id"])
            if expected != initial:
                raise ValueError("full_current_source_binding")
            saved = docs.read("accepted:" + result["acceptance_key"])
            records, receipt = read_records(saved[0]), read_receipt(saved[-1])
            if (
                receipt.status != "accepted"
                or receipt.records_checksum != digest(records)
                or records.scope != initial.material.scope
                or len(records.observations) != 1
            ):
                raise ValueError("full_receipt_binding")
            obs = records.observations[0]
            if (
                obs.reference.code != "OBS:DYNAMIC:" + target
                or obs.image.reference.checksum != self.base.vision_artifact.checksum
            ):
                raise ValueError("full_image_binding")
            evaluate(obs)
            values[target] = obs
        return values

    def observation_projector(self):
        return lambda phase, run_id: phase.project(run_id=run_id, config=self.config)

    def process_evaluations(self):
        observations = self.accepted_observations()
        saved = self.output / "review/review-report.json"
        scoring = json.loads(saved.read_bytes())["scores"] if saved.exists() else {}
        scored = {r["id"]: r for r in scoring.get("reviews", [])}
        rows = []
        for planned in self.task_plan:
            target = planned["id"]
            row = dict(
                planned,
                availability_score=0.0,
                inspected_fraction=0.0,
                effective_quality=0.0,
                gap_score=1.0,
                disease_direction=None,
            )
            if target in observations:
                row.update(evaluate(observations[target]))
            elif planned["action"] == "review_measurement":
                rating = scored.get(target, {}).get("observation", {}).get("value")
                row.update(
                    availability_score=1.0,
                    inspected_fraction=1.0,
                    effective_quality=rating if rating is not None else 0.0,
                    gap_score=1.0 - rating if rating is not None else 1.0,
                    quality_reason="review_O_times_Q; coverage_of_defined_output_not_whole_retina",
                )
            if target in scored:
                row["disease_direction"] = scored[target]["judgment"]["value"]
            # Only selected acquisitions have a phase in this window. Reused
            # facts, measurements and missing-input requirements are still
            # displayed, but must not acquire a fabricated execution path.
            path = (
                self.phase_path(target) / "result.json" if target in self.visual_targets else None
            )
            if path is not None and path.exists():
                phase = json.loads(path.read_bytes())
                row["execution"] = phase
                if not phase["accepted"]:
                    journal = RuntimeAgentJournal(self.documents(phase["run_id"])).read()
                    row["technical_errors"] = [
                        e["record"]["error"]
                        for e in journal
                        if "record" in e and e["record"].get("error")
                    ]
                    row["observation_status"] = "technical_failure_not_image_ungradable"
            rows.append(row)
        # The classifier is appended by its original owner only when confirmed;
        # do not turn it into a mandatory empty slot in the source plan.
        for target in sorted(scored.keys() - {r["id"] for r in rows}):
            rating = scored[target]["observation"]["value"]
            rows.append(
                dict(
                    id=target,
                    action="review_confirmed_model",
                    availability_score=1.0,
                    inspected_fraction=1.0,
                    effective_quality=rating,
                    gap_score=1.0 - rating,
                    disease_direction=scored[target]["judgment"]["value"],
                    quality_reason="model_output_reliability_not_spatial_coverage",
                )
            )
        return rows

    def execute(self):
        result = super().execute()
        return self.archive_completed(result)

    def archive_completed(self, result):
        if not result["complete"]:
            return result
        from apps.case_memory_store import assembly

        docs = self.documents(result["phases"][-1]["run_id"])
        report = self.base.runtime.read_observation_bundle(
            docs.key(getattr(self, "report_key", "dynamic-case-report"))
        )[0]
        writer, owners = assembly("writer")
        archived = []
        targets = [
            "glaucoma",
            "association:VASCULAR_SKELETON_DENSITY",
            "association:VASCULAR_FRACTAL",
            "association:AV_TORTUOSITY",
            "association:ARTERIOLAR_CALIBER",
            "association:VENULAR_BRANCH_ANGLE",
            "association:DISC_SHAPE_CONTEXT",
        ]
        measured = {p["id"] for p in self.task_plan if p["action"] == "review_measurement"}
        for target in targets:
            if target != "glaucoma" and target not in measured:
                continue
            key = self.archive_key(target)
            record = owners.artifacts.prepare(
                key=key,
                case_code=self.base.binding["case_code"],
                run_id=docs.run_id,
                origin_code=report.asset_code,
                target=target,
                kind="graph_snapshot" if target == "glaucoma" else "tool_experience",
            )
            value = writer.create(record)
            archived.append(dict(key=key, checksum=value.checksum, target=target))
        write_once(self.output / "memory-archive.json", archived)
        return dict(result, memory_archived=len(archived))

    def archive_key(self, target):
        return "memory:" + self.version + ":" + self.sample + ":" + target


if __name__ == "__main__":
    import sys

    campaign = ComprehensiveCampaign(sys.argv[1])
    if len(sys.argv) > 2 and sys.argv[2] == "--prepare":
        print(
            json.dumps(
                dict(
                    plan=campaign.task_plan,
                    visual_targets=campaign.visual_targets,
                    prior=campaign.prior(),
                    requests_limit=campaign.max_new_requests,
                ),
                ensure_ascii=False,
            )
        )
    else:
        print(json.dumps(campaign.execute(), ensure_ascii=False))
