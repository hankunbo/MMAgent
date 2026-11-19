"""Public prepare/read/compile ports; installed readers own registration and inputs."""

from packages.grounding.models import digest, seal, validate
from packages.research_runtime.models import ResearchRunContext
from packages.tool_admission.models import InputRecord, ResearchRequest
from packages.tool_workflow import ResearchContext


class ResearchRuntimeService:
    def __init__(self, *, runtime, research, inputs, compiler, synthetic):
        self.runtime, self.research, self.inputs = runtime, research, inputs
        self.compiler, self.synthetic = compiler, synthetic

    def _registered(self, request):
        current = validate(
            ResearchRequest,
            self.research.read_request(request.code, request.version, request.checksum),
        )
        if current != request:
            raise ValueError("research_registration_mismatch")
        case = self.runtime.get_case_bundle(request.scope.case_code)
        if (case.case_code, case.encounter_code, case.patient_hash, case.laterality) != (
            request.scope.case_code,
            request.scope.encounter_code,
            request.scope.patient_hash,
            request.scope.laterality,
        ):
            raise ValueError("research_case_scope_mismatch")
        artifacts = {
            a.asset_code: a for a in self.runtime.artifacts(case.case_code, request.lock.run_id)
        }
        for meta in request.artifacts:
            source = validate(
                InputRecord,
                self.inputs.read_input(request.scope, request.lock.run_id, meta.asset_code),
            )
            if source.metadata != meta or artifacts.get(meta.asset_code) != source.artifact:
                raise ValueError("research_input_fact_mismatch")
        return current

    def create(self, code, version, checksum, key):
        request = validate(ResearchRequest, self.research.read_request(code, version, checksum))
        if (request.code, request.version, request.checksum) != (code, version, checksum):
            raise ValueError("research_registration_identity")
        self._registered(request)
        context = seal(
            ResearchRunContext,
            request=request,
            synthetic=self.synthetic,
            scope_checksum=digest(request.scope),
        )
        return self.runtime.create_research_run(context, key)

    def read(self, run_id):
        context = self.runtime.read_research_run(run_id)
        if context.synthetic != self.synthetic:
            raise ValueError("research_synthetic_mismatch")
        self._registered(context.request)
        return context

    def compile(self, run_id, choice):
        context = self.read(run_id)
        admission = self.compiler.admission.admit_research(context.request)
        plan = self.compiler.compile(
            ResearchContext(request=context.request, admission=admission), choice
        )
        context.validate_plan(plan)
        return plan
