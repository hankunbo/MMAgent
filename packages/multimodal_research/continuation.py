"""Explicit new bounded experiment from confirmed same-case facts, not run resume.

One continuation only. Historical unknown requests remain unknown and counted;
new read-only acquisitions require the CLI's explicit continuation selection.
"""

import json
from pathlib import Path

from packages.grounding.models import digest, validate
from packages.research_runtime.agent_records import RuntimeAgentJournal

from .confirmed_observations import ConfirmedObservations


def load(path, *, checksum, intent):
    path = Path(path).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("continuation_symlink")
    spec = json.loads((path / "entry.json").read_bytes())
    suite = json.loads((path / "suite-result.json").read_bytes())
    if (
        spec.get("continuation_of")
        or spec.get("previous")
        or spec["input_checksum"] != checksum
        or spec["intent"] != intent
        or not spec.get("independent_propositions")
        or suite["complete"]
        or suite["trees"]
        or digest(suite) != json.loads((path / "suite-checksum.json").read_bytes())["checksum"]
    ):
        raise ValueError("continuation_parent_not_stopped_acquisition")
    return spec, suite


class Continuation(ConfirmedObservations):
    def __init__(self, campaign, path):
        self.campaign, self.path = campaign, Path(path).absolute()
        b = campaign.base
        self.spec, self.suite = load(
            self.path, checksum=b.manifest.checksum, intent=b.task.model_dump(mode="json")
        )
        if self.suite["case_code"] != b.binding["case_code"]:
            raise ValueError("continuation_case_identity")
        self.result = dict(phases=self.suite["shared_acquisition"])
        self.config = validate(type(campaign.config), self.spec["provider"])
        self.initial_attempts = self.check_attempts()
        self.values, self.origins = self.read()
        self.binding = digest((self.values, self.origins))

    def check_attempts(self):
        b, c = self.campaign.base, self.campaign
        spec, suite = load(
            self.path, checksum=b.manifest.checksum, intent=b.task.model_dump(mode="json")
        )
        if spec != self.spec or suite != self.suite:
            raise ValueError("continuation_parent_changed")
        count = 0
        for phase in self.result["phases"]:
            control = c.sessions.read(phase["run_id"])
            docs = c.documents(phase["run_id"])
            events = RuntimeAgentJournal(docs).read()
            reserved = [e["request_id"] for e in events if e.get("phase") == "reserved"]
            records = [e["record"] for e in events if e.get("phase") == "completed"]
            if (
                phase["target"] is None
                or control.phase != "stopped"
                or control.pending is not None
                or control.cancelled
                or docs.read("dynamic-phase-result") != (phase,)
                or len(reserved) != len(set(reserved))
                or len(records) != len(reserved)
                or set(reserved) != {r["request_id"] for r in records}
            ):
                raise ValueError("continuation_parent_unconfirmed_or_cancelled")
            if any(r["role"] != "vision" for r in records):
                raise ValueError("continuation_not_read_only_vision")
            count += len(reserved)
        if count != suite["new_requests"] or count >= 40:
            raise ValueError("continuation_budget")
        return [dict(directory=str(self.path), result_checksum=digest(suite), requests=count)]

    def report(self):
        return dict(
            schema_version="confirmed-acquisition-continuation/1.0.0",
            previous=self.initial_attempts,
            origins=self.origins,
            observations_checksum=digest(self.values),
            original_runs_resumed=False,
            explicit_new_read_only_attempt=True,
            max_cumulative_requests=40,
            previous_unknown_results_reclassified=False,
        )
