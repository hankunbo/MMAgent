"""Local-only recovery of an exact full review; imports no credential/API launcher."""

import json

from agents.expert_agent.reporting import read_research_report, render_research_html
from apps.vectorengine_single_case import write_once
from packages.evidence_details.acceptance import ResearchAcceptance
from packages.evidence_details.acceptance_models import DomainState
from packages.evidence_details.finalization import (
    GRANT_KEY,
    REPORT_KEY,
    RESULT_KEY,
    ConfirmedReviewPermission,
    RecordedRequests,
)
from packages.grounding.models import digest, validate
from packages.research_runtime.agent_records import RuntimeAgentJournal

from .comprehensive_materials import ComprehensivePhase
from .comprehensive_review import ComprehensiveReview
from .dynamic_report import export


class ConfirmedCaseReview(ComprehensiveReview):
    export_suffix = "_全清单与部分观察恢复闭环版"
    report_key = REPORT_KEY
    result_key = RESULT_KEY

    def __init__(self, sample):
        super().__init__(sample)
        self.previous_output = self.output
        self.version = "13-full-finalization-r1"
        self.output = self.previous_output.parent / self.version
        self.output.mkdir(exist_ok=True)
        (self.output / "review").mkdir(exist_ok=True)
        proof = self.output / "recovery.json"
        if proof.exists():
            self.recovery = json.loads(proof.read_bytes())

    def execute(self):
        import fcntl

        with (self.output / "lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            old = json.loads((self.previous_output / "result.json").read_bytes())
            docs = self.documents(old["phases"][-1]["run_id"])
            if docs.read("dynamic-case-result") != (old,):
                raise ValueError("finalization_failed_report_binding")
            before = self.sessions.read(docs.run_id)
            journal = RuntimeAgentJournal(docs)
            events = journal.read()
            parent = validate(DomainState, docs.read("dynamic-phase-input")[0])
            phase = ComprehensivePhase(
                self.base,
                observation_reader=self.accepted_observations,
                model_reader=self.read_model,
            )
            expected, _ = phase.project(run_id=docs.run_id, config=self.config)
            if expected != parent:
                raise ValueError("finalization_current_parent_changed")

            class Owner:
                def current(self):
                    phase.verify(parent)
                    return parent

            requests = RecordedRequests(journal)
            key = "confirmed-review-finalization:1"
            prior = docs.read("accepted:" + key)
            if prior is None:
                permission, submission = ConfirmedReviewPermission.prepare(
                    owner=Owner(), documents=docs, requests=requests, config=self.config
                )
                acceptance = ResearchAcceptance(
                    owner=Owner(), documents=docs, requests=requests, live_permission=permission
                )
                submitted = acceptance.submit(submission, key=key)
                if (
                    not isinstance(submitted, dict)
                    or submitted.get("status") != "submitted_not_accepted"
                ):
                    raise ValueError("finalization_submission_rejected")
                receipt = acceptance.accept(key=key)
                if receipt.status != "accepted":
                    raise ValueError("finalization_acceptance_rejected")
            else:
                # Exact known acceptance may be read after its saving window; no new work.
                phase.verify(parent)
                acceptance = ResearchAcceptance(owner=Owner(), documents=docs, requests=requests)
                if prior[-1]["status"] != "accepted":
                    raise ValueError("finalization_rejected_key")
            report = read_research_report(acceptance, key=key, parent=parent)
            grant = docs.read(GRANT_KEY)[0]
            self.recovery = dict(
                schema_version="confirmed-review-report-provenance/1.0.0",
                grant_checksum=digest(grant),
                failed_result_checksum=digest(old),
                original_candidate_checksum=digest(report["records"]["candidate"]),
                recovered_requests=len(requests.records()),
                new_requests=0,
                stopped_run_resumed=False,
                clinical_validation=False,
            )
            write_once(self.output / "recovery.json", self.recovery)
            write_once(self.output / "review/review-report.json", report)
            html_path = self.output / "review/review-report.html"
            rendered = render_research_html(report)
            if html_path.exists() and html_path.read_text() != rendered:
                raise ValueError("finalization_html_conflict")
            if not html_path.exists():
                html_path.write_text(rendered)
            result = dict(
                old,
                complete=True,
                phases=[dict(run_id=docs.run_id, accepted=True, acceptance_key=key, error=None)],
                new_requests=0,
                prior_graph_requests=old["prior_graph_requests"] + old["new_requests"],
                recovery=self.recovery,
                execution_kind="confirmed_review_finalization_not_new_run",
            )
            with docs.session():
                docs.write(RESULT_KEY, (result,))
            write_once(self.output / "result.json", result)
            export(self, result)
            if self.sessions.read(docs.run_id) != before or journal.read() != events:
                raise ValueError("finalization_changed_control_or_requests")
            if docs.read("dynamic-case-result") != (old,):
                raise ValueError("finalization_changed_original_failure")
            return self.archive_completed(result)


if __name__ == "__main__":
    import sys

    print(json.dumps(ConfirmedCaseReview(sys.argv[1]).execute(), ensure_ascii=False))
