"""Explicit new model-evidence increment. Reuses observations, never old run grants."""

import fcntl
import importlib.util
import json
import os
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path

from apps.vectorengine_single_case import write_once
from packages.grounding.models import canonical, digest
from packages.object_store import checksum

from .dynamic_campaign import Campaign
from .dynamic_model import read_model
from .dynamic_report import export


class ModelRepairCampaign(Campaign):
    version = "13-model-repair-v1"
    export_suffix = "_FairCLIP修复贯通版"
    model_revision = "startup-r1"
    extra_freeze_files = ()
    max_new_requests = 2
    max_vision_requests = 0
    duration_seconds = 1200

    def before_review(self):
        """Optional bounded acquisition hook; legacy review-only behavior unchanged."""

    def phase_completed(self, phase):
        return phase["accepted"]

    def freeze_observations(self):
        return self.accepted_observations()

    def __init__(self, sample):
        self.previous = Campaign(sample)
        super().__init__(sample)
        self.output = self.base.directory / self.version
        self.output.mkdir(mode=0o700, parents=True, exist_ok=True)

    def paths(self):
        return [self.output / "review"]

    def accepted_observations(self):
        return self.previous.accepted_observations()

    def read_model(self):
        value = read_model(self.base, self.target, revision=self.model_revision)
        if value is None:
            raise ValueError("model_repair_requires_confirmed_output")
        return value

    def prior(self):
        original = self.previous.prior()
        return dict(
            original=original,
            dynamic_requests=self.previous.usage(),
            reservations=original["reservations"] + self.previous.usage(),
        )

    def freeze(self):
        # A new grant explicitly binds the already confirmed model result and
        # current source-qualified observations. Old stopped sessions stay stopped.
        files = [
            *Path("packages/evaluation_preparation").glob("*.py"),
            Path("apps/vectorengine_live_assembly.py"),
            Path("agents/expert_agent/service.py"),
            Path("packages/llm_provider/transport.py"),
            Path("configs/prompts/agents/review-batch-1.5.0.txt"),
            *self.extra_freeze_files,
        ]
        expected = dict(
            schema_version="model-evidence-increment/1.0.0",
            sample=self.sample,
            case_code=self.base.binding["case_code"],
            image_checksum=self.base.image_artifact.checksum,
            previous=self.prior(),
            model_checksum=digest(self.read_model()),
            observation_checksum=digest(self.freeze_observations()),
            provider_checksum=digest(self.config),
            max_new_requests=self.max_new_requests,
            max_cumulative_graph_requests=40,
            seconds=self.duration_seconds,
            implementation={str(p): checksum(p.read_bytes()) for p in files},
            vision_requests=self.max_vision_requests,
            new_model_execution=False,
            clinical_acceptance=False,
        )
        path = self.output / "protocol.json"
        if path.exists():
            value = json.loads(path.read_bytes())
            if {k: v for k, v in value.items() if k not in {"started_at", "deadline"}} != expected:
                raise ValueError("model_increment_frozen_input_changed")
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

    def export_result(self, result):
        return export(self, result)

    @contextmanager
    def execution_session(self):
        """Legacy filesystem exclusion; explicit successors may use a Runtime lease."""
        with (self.output / "lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            yield

    def execute(self):
        with self.execution_session():
            self.freeze()
            result_file = self.output / "result.json"
            if result_file.exists():
                result = json.loads(result_file.read_bytes())
                if self.documents(result["phases"][-1]["run_id"]).read("dynamic-case-result") != (
                    result,
                ):
                    raise ValueError("model_increment_result_binding")
                self.export_result(result)
                return result
            self.guard()
            spec = importlib.util.spec_from_file_location(
                "private_launcher", ".local/vectorengine/run.py"
            )
            launcher = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(launcher)
            name = self.config.transport.api_key_env
            previous = os.environ.get(name)
            try:
                os.environ[name] = launcher.load_secret()
                self.before_review()
                phase = self.execute_phase()
            finally:
                if previous is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = previous
            result = json.loads(
                canonical(
                    dict(
                        schema_version="dynamic-case-result/1.0.0",
                        sample=self.sample,
                        case_code=self.base.binding["case_code"],
                        phases=[phase],
                        complete=self.phase_completed(phase),
                        new_requests=self.usage(),
                        prior_graph_requests=self.protocol["previous"]["reservations"],
                        reused_visual_observations=len(self.accepted_observations()),
                        clinical_acceptance=False,
                        hidden_labels_used=False,
                    )
                )
            )
            docs = self.documents(phase["run_id"])
            with docs.session():
                docs.write("dynamic-case-result", (result,))
            write_once(result_file, result)
            self.export_result(result)
            return result
