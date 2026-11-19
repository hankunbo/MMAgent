"""Offline simulations over original RuntimeDocuments and Planner control."""

import html
import itertools
import json
import sys
from functools import wraps

from packages.evidence_details.current_rejection import CurrentResearchRejected, current_rejection
from packages.grounding.models import digest, validate
from packages.planner_control.models import Phase

from .engine import authorize, compare, compute, paired, search_fidelity
from .models import Candidate, Intervention, SimulationPolicy, SimulationReport

SEMANTICS = "counterfactual-semantics/1.1.0"


def observe_cancellation(method):
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        finally:
            primary = sys.exc_info()[1]
            try:
                self._observe(kwargs.get("cancel"))
            except Exception as cancel_error:
                if primary is not None:
                    raise primary from cancel_error
                raise

    return wrapped


def render(report):
    return dict(
        schema_version="counterfactual-html/1.0.0",
        checksum=report["checksum"],
        identity="hypothetical_candidate",
        html="<!doctype html><meta charset='utf-8'>"
        "<h1>离线假设候选 / 非医学因果效应</h1><pre>"
        + html.escape(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
        + "</pre>",
    )


class CounterfactualSimulation:
    def __init__(self, *, reader, documents, policy, alternative_readers=None):
        self.reader, self.documents = reader, documents
        self.planner, self.run_id = reader.planner, reader.run_id
        self.policy = validate(SimulationPolicy, policy)
        self.alternative_readers = alternative_readers or {}

    def _read(self, key):
        with self.documents.session():
            return self.documents.read("simulation:" + key)

    def _write(self, key, values):
        try:
            with self.documents.session():
                state = self.planner.read(self.run_id)
                from types import SimpleNamespace

                self.documents.validate_scope(
                    SimpleNamespace(
                        run_id=self.run_id, case_code=state.binding.case_code, graph=state.graph
                    )
                )
                self.documents.write("simulation:" + key, values)
                if self.documents.read("simulation:" + key) != values:
                    raise ValueError("simulation_confirmation")
        except Exception as error:
            try:
                self.documents.failure(
                    "simulation:" + key, digest(values), "object_or_transaction_failure"
                )
            except Exception as audit_error:
                raise error from audit_error
            raise

    def _claim(self, key, value):
        with self.documents.session():
            prior = self.documents.read("simulation:" + key)
            if prior is not None:
                if prior != (value,):
                    raise ValueError("simulation_claim_conflict")
                return False
            state = self.planner.read(self.run_id)
            from types import SimpleNamespace

            self.documents.validate_scope(
                SimpleNamespace(
                    run_id=self.run_id, case_code=state.binding.case_code, graph=state.graph
                )
            )
            self.documents.write("simulation:" + key, (value,))
            return True

    def _observe(self, cancel):
        state = self.planner.read(self.run_id)
        return self.planner.observe_recovery_cancel(
            self.run_id, state.pending.command if state.pending else None, cancel=cancel
        )

    def _gate(self, cancel=None):
        state = self._observe(cancel)
        if state.pending is not None:
            raise ValueError("simulation_original_pending_requires_recovery")
        reason = self.planner.research_review_limit(state)
        if state.phase == Phase.STOPPED:
            reason = reason or "stopped"
        if state.branches >= state.binding.profile.max_branches:
            reason = reason or "branch_budget"
        if state.reserved >= state.binding.profile.max_actions:
            reason = reason or "action_budget"
        if reason:
            raise ValueError("simulation_control:" + reason)
        return state

    @observe_cancellation
    def freeze(self, *, cancel=None):
        self._gate(cancel)
        prior = self._read("policy")
        if prior and prior[0].get("semantics") != SEMANTICS:
            raise ValueError("simulation_legacy_semantics_requires_historical_read")
        fact = self.reader()
        authorize(self.policy, fact, ())
        raw = dict(
            semantics=SEMANTICS,
            policy=self.policy.model_dump(mode="json"),
            binding_checksum=digest(fact.binding),
            fact=fact.references(),
            baseline=fact.baseline(),
            fact_validation="original_owner_rebuilt_6.1",
        )
        self._write("policy", (raw,))
        self._gate(cancel)
        return fact

    @observe_cancellation
    def declare(self, candidates, *, cancel=None):
        fact = self.freeze(cancel=cancel)
        candidates = tuple(validate(Candidate, c) for c in candidates)
        authorize(self.policy, fact, candidates)
        raw = dict(
            policy_checksum=digest(self.policy),
            candidates=[c.model_dump(mode="json") for c in candidates],
        )
        self._write("plan", (raw,))
        return candidates

    def removal_plan(self, *, cancel=None):
        """Bounded prefix of subsets; never claims a global optimum."""
        fact = self.freeze(cancel=cancel)  # policy durable before candidate generation
        allowed = self.policy.allowed.get("remove_evidence", ())
        if not allowed:
            raise ValueError("simulation_search_not_allowed")
        candidates = []
        limit = min(self.policy.max_search, self.policy.max_candidates)
        for size in range(1, len(allowed) + 1):
            for subset in itertools.combinations(allowed, size):
                candidates.append(
                    Candidate(
                        id="search:" + str(len(candidates)),
                        interventions=(Intervention(kind="remove_evidence", targets=subset),),
                    )
                )
                if len(candidates) == limit:
                    break
            if len(candidates) == limit:
                break
        self.declare(candidates, cancel=cancel)
        return dict(
            candidates=candidates,
            total_subsets=2 ** len(allowed) - 1,
            explored=len(candidates),
            complete=len(candidates) == 2 ** len(allowed) - 1,
            claim="bounded_algorithm_fidelity_only_not_clinical_sufficiency",
            fact_checksum=fact.checksum,
        )

    def _declared(self, candidate):
        saved_policy, plan = self._read("policy"), self._read("plan")
        if saved_policy and saved_policy[0].get("semantics") != SEMANTICS:
            raise ValueError("simulation_legacy_semantics_requires_historical_read")
        if (
            not saved_policy
            or not plan
            or saved_policy[0]["policy"] != self.policy.model_dump(mode="json")
            or plan[0]["policy_checksum"] != digest(self.policy)
            or candidate.model_dump(mode="json") not in plan[0]["candidates"]
        ):
            raise ValueError("simulation_not_predeclared_or_policy_drift")

    def _alternatives(self, fact, names, *, cancel=None, execution_gate=True):
        alternatives = {}
        for name in sorted(names):
            if execution_gate:
                self._gate(cancel)
            if name not in self.alternative_readers:
                raise CurrentResearchRejected("simulation_alternative_reader_missing")
            other = self.alternative_readers[name]()
            if other.checksum != self.policy.alternatives.get(name):
                raise CurrentResearchRejected("simulation_alternative_drift")
            paired(fact, other)  # case/input/distinct run, through original owner
            alternatives[name] = other
        return alternatives

    def _current(self, report, *, cancel=None, execution_gate=True):
        """Revalidate precisely the dependencies actually consumed by this report."""
        if report["schema_version"] != "counterfactual-report/1.1.0":
            raise ValueError("simulation_legacy_semantics_requires_historical_read")
        candidate = validate(Candidate, report["candidate"])
        self._declared(candidate)
        if execution_gate:
            self._gate(cancel)
        fact = self.reader()
        if fact.checksum != report["fact_checksum"]:
            raise CurrentResearchRejected("simulation_fact_drift")
        authorize(self.policy, fact, (candidate,))
        if report["policy_checksum"] != digest(self.policy):
            raise ValueError("simulation_report_policy_drift")
        expected = {i.alternative for i in candidate.interventions if i.alternative}
        if set(report["alternatives"]) != expected:
            raise ValueError("simulation_report_dependencies")
        for name, other in self._alternatives(
            fact, expected, cancel=cancel, execution_gate=execution_gate
        ).items():
            if paired(fact, other) != report["alternatives"][name]:
                raise CurrentResearchRejected("simulation_alternative_binding_drift")
        if execution_gate:
            self._gate(cancel)
        return fact

    def read(self, *, candidate_id, historical=False):
        Candidate(
            id=candidate_id,
            interventions=(Intervention(kind="remove_evidence", targets=("check",)),),
        )
        values = self._read("result:" + candidate_id)
        if values is None:
            return None
        report = validate(SimulationReport, values[0]).model_dump(mode="json")
        state = self.planner.read(self.run_id)
        if (
            report["run_id"] != self.run_id
            or report["case_code"] != state.binding.case_code
            or report["candidate"]["id"] != candidate_id
        ):
            raise ValueError("simulation_report_scope")
        if digest({k: v for k, v in report.items() if k != "checksum"}) != report["checksum"]:
            raise ValueError("simulation_report_checksum")
        if values != (report, render(report)):
            raise ValueError("simulation_html_source")
        if not historical:
            self._current(report)
        return report

    def read_evidence(self, *, candidate_id):
        """Current-valid evidence for a stopped run; grants no execution permission.

        Unlike read(historical=True), revalidate every original fact, policy and
        alternative dependency. Unlike execution reads, never observes cancellation
        or calls the control mutation/recovery/simulation path.
        """
        state = self.planner.replay(self.run_id)
        if state.phase != Phase.STOPPED or state.pending is not None:
            raise ValueError("simulation_evidence_terminal_required")
        report = self.read(candidate_id=candidate_id, historical=True)
        if report is None:
            return None
        self._current(report, execution_gate=False)
        if self.planner.replay(self.run_id) != state:
            raise CurrentResearchRejected("simulation_fact_drift")
        return report

    @observe_cancellation
    def run(self, candidate, *, cancel=None):
        candidate = validate(Candidate, candidate)
        self._declared(candidate)
        prior = self.read(candidate_id=candidate.id, historical=True)
        if prior and prior["candidate"] != candidate.model_dump(mode="json"):
            raise ValueError("simulation_key_content_conflict")
        try:
            self._gate(cancel)
            if prior is None:
                fact = self.reader()
                authorize(self.policy, fact, (candidate,))
                names = {i.alternative for i in candidate.interventions if i.alternative}
                alternatives = self._alternatives(fact, names, cancel=cancel)
                result = compute(fact, candidate, self.policy, alternatives)
                model = SimulationReport(
                    simulation_id="SIM:"
                    + digest([self.run_id, digest(self.policy), candidate])[7:55],
                    case_code=fact.binding.case_code,
                    run_id=self.run_id,
                    policy_checksum=digest(self.policy),
                    fact_checksum=fact.checksum,
                    candidate=candidate,
                    **result,
                    checksum=digest(None),
                )
                raw = model.model_dump(mode="json", exclude={"checksum"})
                pending = {**raw, "checksum": digest(raw)}
                self._current(pending, cancel=cancel)
                self._write("result:" + candidate.id, (pending, render(pending)))
                prior = pending
            fact = self._current(prior, cancel=cancel)
            validation = dict(
                simulation_id=prior["simulation_id"],
                fact_checksum=fact.checksum,
                report_checksum=prior["checksum"],
                fact_unchanged=True,
                simulation_identity="hypothetical_candidate",
                formula="original_6.1",
                required_review=self.policy.expert_required or fact.binding.profile.expert_required,
            )
            self._write("validation:" + candidate.id, (validation,))
            saved_review = self._read("review:" + candidate.id)
            self._current(prior, cancel=cancel)
            return dict(
                status="candidate",
                report=prior,
                validation=validation,
                review=saved_review,
                control=self.planner.report(self.run_id).model_dump(mode="json"),
            )
        except ValueError as error:
            if current_rejection(error) or str(error).startswith("simulation_control:"):
                return dict(
                    status="stopped",
                    reason=str(error),
                    report=prior,
                    control=self.planner.report(self.run_id).model_dump(mode="json"),
                )
            raise

    @observe_cancellation
    def search(self, *, cancel=None):
        plan = self.removal_plan(cancel=cancel)
        retained, evaluations = [], []
        for candidate in plan.pop("candidates"):
            result = self.run(candidate, cancel=cancel)
            if result["status"] != "candidate":
                return dict(**plan, status="stopped", result=result)
            r = result["report"]
            a, b = r["baseline"], r["simulated"]
            fidelity = search_fidelity(a, b, r["comparisons"], self.policy)
            evaluations.append(dict(candidate_id=candidate.id, **fidelity))
            if fidelity["eligible"]:
                retained.append(
                    dict(
                        candidate_id=candidate.id, remaining=b["remaining"], checksum=r["checksum"]
                    )
                )
        fact = self.reader()
        authorize(self.policy, fact, ())
        baseline = fact.baseline()
        minimum = min((len(r["remaining"]) for r in retained), default=None)
        report = dict(
            **plan,
            semantics=SEMANTICS,
            status="candidate",
            baseline=dict(
                remaining=baseline["remaining"],
                fidelity=search_fidelity(
                    baseline, baseline, compare(baseline, baseline), self.policy
                ),
            ),
            outcome="bounded_minimal_sets_found" if retained else "no_feasible_removal_found",
            evaluations=evaluations,
            minimal_within_explored=[r for r in retained if len(r["remaining"]) == minimum],
            global_optimum=False,
        )
        self._gate(cancel)
        self._write("search", (report,))
        self._gate(cancel)
        if self.reader().checksum != fact.checksum:
            raise CurrentResearchRejected("simulation_fact_drift")
        return report
