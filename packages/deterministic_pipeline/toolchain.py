"""Formal 05 → 06B → 06C bridge. No fallback from formal to research authorization."""

from packages.grounding.models import digest, validate
from packages.tool_admission import AdmissionSelection
from packages.tool_workflow import FormalContext, WorkflowChoice

from .models import PipelineError


def choose_workflow(admission, root):
    """Versioned mechanical choice: 05 order, existing artifacts, then producer order."""
    nodes = {n.identity: n for n in admission.retained}
    if root not in admission.retained_roots:
        raise PipelineError("tool_admission_rejected")
    selected = {}

    def visit(identity):
        if identity in selected:
            return
        n = nodes[identity]
        inputs = []
        selected[identity] = dict(candidate_ref=identity, inputs=inputs)
        for p in n.inputs:
            sources = [dict(kind="artifact", asset_code=a.asset_code) for a in p.artifacts]
            sources += [
                dict(kind="producer", candidate_ref=i, port=nodes[i].candidate.output_port)
                for i in p.producer_refs
            ]
            sources = sources[: p.cardinality_min]
            if len(sources) < p.cardinality_min:
                raise PipelineError("and_input_blocked")
            inputs.append(dict(port=p.port, sources=sources))
            for s in sources:
                if s["kind"] == "producer":
                    visit(s["candidate_ref"])

    visit(root)
    return validate(WorkflowChoice, dict(roots=[root], nodes=list(selected.values())))


class FormalQuery:
    """Trusted assembly supplies metadata only from registered Artifact records.

    query_builder has no evaluation dependency. Its output is checked against the
    persisted case/Artifact facts here, and again by 05 and 06 input owners.
    This port has no compiler or executor dependency.
    """

    def __init__(self, *, grounding, query_builder, reference):
        self.grounding = grounding
        self.query_builder = query_builder
        self.reference = reference

    def query(self, input_value, graph, lock, artifacts, requirement, config, started_at):
        q = self.query_builder(input_value, graph, lock, artifacts, requirement, config, started_at)
        c = input_value.case
        if (
            (q.scope.case_code, q.scope.patient_hash, q.scope.encounter_code, q.scope.laterality)
            != (c.case_code, c.patient_hash, c.encounter_code, c.laterality)
            or q.lock != lock
            or q.requirement_ref != requirement
            or q.config != config.grounding
        ):
            raise PipelineError("query_input_scope_mismatch")
        if q.context != config.context:
            raise PipelineError("query_context_mismatch")
        known = {a.asset_code: a for a in artifacts}
        for a in q.artifacts:
            original = known.get(a.asset_code)
            if original is None or (
                a.checksum,
                a.case_code,
                a.run_id,
                a.laterality,
                a.coordinate_space,
                a.qc_status,
            ) != (
                original.checksum,
                original.scope.case_code,
                original.scope.run_id,
                original.laterality,
                original.coordinate_space,
                original.qc_status,
            ):
                raise PipelineError("query_artifact_fact_mismatch")
            if not input_value.synthetic and a.acquired_at is not None:
                raise PipelineError("unproven_timepoint")
        return q, self.grounding.query(q)


class FormalToolchain(FormalQuery):
    """Explicit execution composition of the read-only query port."""

    def __init__(
        self, *, grounding, query_builder, invocations, admission, compiler, executor_for, reference
    ):
        self.grounding = grounding
        self.query_builder = query_builder
        self.invocations = invocations
        self.admission = admission
        self.compiler = compiler
        self.executor_for = executor_for
        self.reference = reference

    def compile(self, q, result, candidate):
        selection = AdmissionSelection(
            candidate_refs=(candidate.identity,), invocations=self.invocations
        )
        admission = self.admission.admit(q, result, selection)
        if candidate.identity not in admission.retained_roots:
            raise PipelineError("tool_admission_rejected")
        choice = choose_workflow(admission, candidate.identity)
        plan = self.compiler.compile(
            FormalContext(request=q, query_result=result, selection=selection, admission=admission),
            choice,
        )
        self.compiler.validate_plan(plan)
        return plan

    @staticmethod
    def action_key(target, candidate):
        return digest(
            dict(target=target, candidate=candidate.candidate, dependencies=candidate.dependencies)
        )


class RegisteredExecutionRouter:
    """Resolve an exact pre-authorized plan to one service; never try another backend."""

    def __init__(self, services_by_plan_checksum):
        self.services = dict(services_by_plan_checksum)

    def __call__(self, plan):
        service = self.services.get(plan.checksum)
        if service is None:
            raise PipelineError("execution_backend_not_registered_for_plan")
        return service
