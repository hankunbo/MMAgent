"""One explicit, bounded continuation of unstarted propositions from saved facts.

No stopped run is resumed. Completed trees and accepted images are read through
their original owners; only the remaining propositions receive new Review runs.
"""

import copy
import json
from pathlib import Path

from packages.grounding.models import digest, validate
from packages.object_store import checksum
from packages.research_runtime.agent_records import RuntimeAgentJournal


def load(path, *, input_checksum, intent):
    path = Path(path).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("remaining_review_symlink")
    spec = json.loads((path / "entry.json").read_bytes())
    suite = json.loads((path / "suite-result.json").read_bytes())
    if (
        spec.get("review_from")
        or spec.get("previous")
        or not spec.get("independent_propositions")
        or spec["input_checksum"] != input_checksum
        or spec["intent"] != intent
        or suite["complete"]
        or suite["error"] != "deadline"
        or not 0 < len(suite["trees"]) < 4
        or not all(t["complete"] for t in suite["trees"])
        or not all(p["accepted"] for p in suite["shared_acquisition"])
        or digest(suite) != json.loads((path / "suite-checksum.json").read_bytes())["checksum"]
    ):
        raise ValueError("remaining_review_requires_closed_deadline_and_confirmed_facts")
    return spec, suite


class RemainingReview:
    def __init__(self, campaign, path):
        from packages.evidence_details.acceptance_models import DomainState

        from .campaign import Phase
        from .continuation import Continuation
        from .proposition_campaign import PropositionCampaign
        from .report_storage import unpack

        self.campaign, self.path = campaign, Path(path).absolute()
        b = campaign.base
        self.spec, self.suite = load(
            self.path,
            input_checksum=b.manifest.checksum,
            intent=b.task.model_dump(mode="json"),
        )
        if self.suite["case_code"] != b.binding["case_code"]:
            raise ValueError("remaining_review_case_binding")
        old_base = copy.copy(b)
        old_base.output = self.path
        old_base.views_path = self.path / "views/views.json"
        if checksum(old_base.views_path.read_bytes()) != b.view_checksum:
            raise ValueError("remaining_review_view_binding")
        self.source = PropositionCampaign(
            old_base, campaign.target, campaign_id=self.path.name, synthetic=campaign.synthetic
        )
        self.source.config = validate(type(campaign.config), self.spec["provider"])
        self.source.protocol = json.loads((self.path / "protocol.json").read_bytes())
        if self.spec.get("continuation_of"):
            self.source.continuation = Continuation(self.source, self.spec["continuation_of"])
            self.source.targets = tuple(
                t for t in self.source.targets if t not in self.source.continuation.values
            )
        if list(self.source.targets) != self.source.protocol["targets"]:
            raise ValueError("remaining_review_acquisition_inventory")
        for phase in self.suite["shared_acquisition"]:
            rid = phase["run_id"]
            docs = self.source.documents(rid)
            parent = validate(DomainState, docs.read("dynamic-phase-input")[0])
            old_base.contexts[rid] = self.source.contexts.read_research_run(rid)
            evidence = Phase(old_base, target=phase["target"])
            expected, _ = evidence.project(run_id=rid, config=self.source.config)
            evidence.verify(parent)
            if expected != parent:
                raise ValueError("remaining_review_current_acquisition_sources")
        self.trees = self.suite["trees"]
        self.labels = tuple(t["proposition"] for t in self.trees)
        if len(set(self.labels)) != len(self.labels) or not set(self.labels) < set(
            campaign.propositions
        ):
            raise ValueError("remaining_review_propositions")
        self.files = {}
        for tree in self.trees:
            page = Path(tree["json"])
            report = json.loads(page.read_bytes())
            docs = self.source.documents(tree["run_id"])
            result = report["result"]
            if (
                not report["completion"]["all_obtained_reviewed"]
                or result["task"]
                != campaign.propositions[tree["proposition"]].model_dump(mode="json")
                or result["case_code"] != b.binding["case_code"]
                or result["phases"][-1]["run_id"] != tree["run_id"]
                or docs.read("multimodal-case-result") != (result,)
                or unpack(docs.read("dynamic-case-report"), key="dynamic-case-report")
                != (
                    report,
                    dict(html=Path(tree["html"]).read_text(), report_checksum=digest(report)),
                )
            ):
                raise ValueError("remaining_review_saved_tree_binding")
            for filename in (tree["json"], tree["html"]):
                file = Path(filename)
                if file.is_symlink():
                    raise ValueError("remaining_review_tree_symlink")
                self.files[filename] = checksum(file.read_bytes())
        self.previous = self.check_attempts()
        self.values = self.source.accepted_observations()
        self.values_checksum = digest(self.values)

    def check_attempts(self):
        b, c = self.campaign.base, self.source
        spec, suite = load(
            self.path,
            input_checksum=b.manifest.checksum,
            intent=b.task.model_dump(mode="json"),
        )
        if (spec, suite) != (self.spec, self.suite):
            raise ValueError("remaining_review_parent_changed")
        phases = list(suite["shared_acquisition"])
        for tree in suite["trees"]:
            docs = c.documents(tree["run_id"])
            result = docs.read("multimodal-case-result")[0]
            phases.append(result["phases"][-1])
        count, seen = 0, set()
        for phase in phases:
            rid = phase["run_id"]
            if rid in seen:
                raise ValueError("remaining_review_duplicate_run")
            seen.add(rid)
            control = c.sessions.read(rid)
            docs = c.documents(rid)
            context = c.contexts.read_research_run(rid)
            expected_task = (
                b.task
                if phase["target"] is not None
                else self.campaign.propositions[
                    next(t["proposition"] for t in self.trees if t["run_id"] == rid)
                ]
            )
            events = RuntimeAgentJournal(docs).read()
            reserved = [e["request_id"] for e in events if e.get("phase") == "reserved"]
            records = [e["record"] for e in events if e.get("phase") == "completed"]
            if (
                control.phase != "stopped"
                or context.request.scope.case_code != b.binding["case_code"]
                or context.request.input_checksum != b.manifest.checksum
                or context.request.task != expected_task
                or control.pending is not None
                or control.cancelled
                or not phase["accepted"]
                or docs.read("dynamic-phase-result") != (phase,)
                or len(reserved) != len(set(reserved))
                or len(records) != len(reserved)
                or set(reserved) != {r["request_id"] for r in records}
            ):
                raise ValueError("remaining_review_unconfirmed_parent")
            count += len(reserved)
        if count != suite["new_requests"] or c.usage() != count:
            raise ValueError("remaining_review_request_inventory")
        previous = c.previous_attempts() + [
            dict(directory=str(self.path), result_checksum=digest(suite), requests=count)
        ]
        if sum(row["requests"] for row in previous) >= 40:
            raise ValueError("remaining_review_budget")
        for name, expected in self.files.items():
            if checksum(Path(name).read_bytes()) != expected:
                raise ValueError("remaining_review_tree_changed")
        return previous

    def check(self):
        self.source.base.check()
        if self.check_attempts() != self.previous:
            raise ValueError("remaining_review_previous_attempts_changed")
        values = self.source.accepted_observations()
        if digest(values) != self.values_checksum:
            raise ValueError("remaining_review_observations_changed")
        return values

    def report(self):
        return dict(
            schema_version="remaining-proposition-review/1.0.0",
            parent=str(self.path),
            parent_checksum=digest(self.suite),
            previous_attempts=self.previous,
            completed_propositions=list(self.labels),
            remaining_propositions=[p for p in self.campaign.propositions if p not in self.labels],
            observations_checksum=self.values_checksum,
            original_runs_resumed=False,
            new_vision_requests=0,
            max_cumulative_requests=40,
        )
