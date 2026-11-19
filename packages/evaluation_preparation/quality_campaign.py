"""Explicit new quality-weighted review; old runs and raw model outputs unchanged."""

import json
from pathlib import Path

from packages.evidence_details.acceptance_models import DomainState, read_receipt, read_records
from packages.grounding.models import digest, validate

from .model_repair_campaign import ModelRepairCampaign
from .quality_materials import QualityCaseEvidence, QualityPhaseEvidence


class QualityCampaign(ModelRepairCampaign):
    version = "13-quality-weighted-v1"
    export_suffix = "_质量加权与测量保留版"
    evidence_class = QualityCaseEvidence
    phase_class = QualityPhaseEvidence
    max_new_requests = 3
    max_vision_requests = 1
    extra_freeze_files = tuple(
        Path(p)
        for p in (
            "packages/evidence_details/acceptance_models.py",
            "packages/evidence_scoring/accepted.py",
            "packages/evidence_scoring/api.py",
            "packages/evidence_scoring/quality_policy.py",
        )
    )

    def __init__(self, sample, *, revision="v1"):
        if revision not in {"v1", "wire-r1"}:
            raise ValueError("quality_revision")
        self.revision = revision
        if revision != "v1":
            self.version = "13-quality-weighted-" + revision
        self.model_previous = ModelRepairCampaign(sample)
        super().__init__(sample)

    def prior(self):
        value = super().prior()
        requests = self.model_previous.usage()
        result = dict(
            value, model_repair_requests=requests, reservations=value["reservations"] + requests
        )
        if self.revision == "v1":
            return result
        from packages.research_runtime.agent_records import RuntimeAgentJournal

        previous = []
        directory = self.base.directory / "13-quality-weighted-v1"
        for phase in ("vision-vcdr", "review"):
            path = directory / phase / "request.json"
            if not path.exists():
                continue
            run = json.loads(path.read_bytes())["lock"]["run_id"]
            control = self.sessions.read(run)
            if control is not None and (control.phase != "stopped" or control.pending):
                raise ValueError("quality_previous_run_unresolved")
            if self.contexts.read_research_run(run) is not None:
                events = RuntimeAgentJournal(self.documents(run)).read()
                previous.append(
                    dict(
                        run=run,
                        checksum=digest(events),
                        requests=sum(e.get("phase") == "reserved" for e in events),
                    )
                )
        return dict(
            result,
            previous_quality=previous,
            reservations=result["reservations"] + sum(p["requests"] for p in previous),
        )

    def paths(self):
        return [self.output / "vision-vcdr", self.output / "review"]

    def phase_path(self, target):
        if target not in {None, "criterion:VCDR_GE_07"}:
            raise ValueError("quality_visual_target")
        return self.paths()[0 if target else 1]

    def freeze_observations(self):
        return self.previous.accepted_observations()

    def before_review(self):
        if not self.base.soft_quality["measurement_available"]:
            self.execute_phase("criterion:VCDR_GE_07")

    def accepted_observations(self):
        values = self.previous.accepted_observations()
        path = self.paths()[0] / "result.json"
        if not path.exists():
            return values
        result = json.loads(path.read_bytes())
        docs = self.documents(result["run_id"])
        if docs.read("dynamic-phase-result") != (result,):
            raise ValueError("quality_fallback_result_binding")
        if not result["accepted"]:
            return values
        initial = validate(DomainState, docs.read("dynamic-phase-input")[0])
        phase = QualityPhaseEvidence(self.base, target="criterion:VCDR_GE_07")
        expected, _ = phase.project(run_id=result["run_id"], config=self.config)
        if expected != initial:
            raise ValueError("quality_fallback_current_source")
        saved = docs.read("accepted:" + result["acceptance_key"])
        records, receipt = read_records(saved[0]), read_receipt(saved[-1])
        if (
            receipt.status != "accepted"
            or receipt.records_checksum != digest(records)
            or records.scope != initial.material.scope
            or len(records.observations) != 1
        ):
            raise ValueError("quality_fallback_receipt_binding")
        obs = records.observations[0]
        if (
            obs.reference.code != "OBS:DYNAMIC:criterion:VCDR_GE_07"
            or obs.image.reference.checksum != self.base.image_artifact.checksum
        ):
            raise ValueError("quality_fallback_image_binding")
        values["criterion:VCDR_GE_07"] = obs
        return values
