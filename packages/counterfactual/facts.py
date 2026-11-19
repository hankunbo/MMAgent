"""Immutable facts are obtained through the installed 09 owner, never caller score blobs."""

from dataclasses import dataclass

from packages.evidence_details.acceptance_models import DomainState
from packages.evidence_scoring.api import validate as validate_score
from packages.evidence_scoring.simulation import project
from packages.grounding.models import digest, validate
from packages.llm_provider.models import ReviewProposal
from packages.path_graph_validation.service import checked_summary
from packages.planner_control.models import Binding


@dataclass(frozen=True)
class Fact:
    summary: object
    binding: Binding

    @property
    def checksum(self):
        return digest(dict(summary=self.summary, binding=self.binding))

    @property
    def state(self):
        return validate(DomainState, self.summary.binding["parent_state"])

    @property
    def candidate(self):
        return validate(ReviewProposal, self.summary.original_report["records"]["candidate"])

    @property
    def scores(self):
        return self.summary.original_report["scores"]

    def baseline(self):
        value = project(self.state, self.candidate, self.scores)
        if value["root"] != self.scores["root"]:
            raise ValueError("simulation_baseline_reconstruction")
        return value

    def references(self):
        return dict(
            identity="fact",
            checksum=self.checksum,
            case_code=self.binding.case_code,
            run_id=self.binding.run_id,
            input_checksum=self.binding.input_checksum,
            owner=dict(
                reader="ResearchPathReader",
                receipt_checksum=self.summary.binding["receipt_checksum"],
                request_id=self.summary.original_report["records"]["request_id"],
            ),
            graph=self.summary.graph,
            records_checksum=self.summary.binding["records_checksum"],
            scores_checksum=self.summary.binding["score_checksum"],
            report_checksum=self.summary.original_report_checksum,
            policy_checksum=digest(self.state.policy),
            locks=[r.model_dump(mode="json") for r in self.binding.locks],
            sources=[s.reference.model_dump(mode="json") for s in self.state.material.sources],
            resources=[r.model_dump(mode="json") for r in self.state.resources],
        )


class FactReader:
    def __init__(self, *, path_reader, planner, run_id):
        self.path_reader, self.planner, self.run_id = path_reader, planner, run_id

    def __call__(self):
        summary = checked_summary(self.path_reader())
        control = self.planner.read(self.run_id)
        if (
            summary.domain != "research"
            or summary.run_id != control.binding.run_id
            or summary.case_code != control.binding.case_code
            or summary.binding["current_scope"]["graph"] != control.graph.model_dump(mode="json")
            or self.planner.owner.current(control.binding) != control.graph
        ):
            raise ValueError("simulation_fact_scope")
        fact = Fact(summary, validate(Binding, control.binding))
        validate_score(dict(state=fact.state, candidate=fact.candidate), fact.scores)
        fact.baseline()
        return fact
