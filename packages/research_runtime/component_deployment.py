"""Compose the original component owners, without a dataset-specific case runner.

This is an installed Python assembly port, not a request-side approval API.
Construction is deliberately not current admission: confirmed cold recovery
must remain possible when a previously used input is no longer readable.
Registration, admission, compile and dispatch retain their original checks.
"""

from dataclasses import dataclass
from pathlib import Path

from db.runtime.repository.case_pipeline import identity
from db.runtime.repository.research import ResearchRuntimeExecutions
from packages.grounding.models import validate
from packages.tool_admission import ToolAdmissionService
from packages.tool_admission.models import AdmissionPolicy
from packages.tool_execution.trusted_models import TrustedLocalPolicy
from packages.tool_execution.trusted_service import TrustedLocalGPUResearchService
from packages.tool_workflow import ToolWorkflowCompiler
from packages.tool_workflow.models import compiler_config

from .component_inputs import ComponentResearchRequests
from .readiness import UnboundSingleCasePolicy
from .registration import single_choice
from .service import ResearchRuntimeService


@dataclass(frozen=True)
class ComponentServices:
    requests: ComponentResearchRequests
    admission: ToolAdmissionService
    compiler: ToolWorkflowCompiler
    research: ResearchRuntimeService

    def choice(self, request):
        # No cached admission, scope inference, fallback candidate or patient ID.
        return single_choice(self.admission.admit_research(request))


def component_executor_factory(*, services, catalogs, objects, qc, policy, workspace):
    """Bind original 06D execution to one exact plan and the original DB owner.

    Installation is not dispatch or current medical authorization. In particular,
    constructing a cold-recovery executor must not reread missing patient inputs.
    The original execute/reserve path retains compiler and current-owner checks.
    The unbound installation policy is copied, never mutated or renewed.
    """
    if type(services) is not ComponentServices:
        raise ValueError("component_original_services_required")
    if services.research.runtime is not services.requests.runtime:
        raise ValueError("component_execution_runtime_binding")
    if (
        services.research.compiler is not services.compiler
        or services.compiler.admission is not services.admission
        or services.research.research is not services.requests
        or services.research.inputs is not services.requests
        or services.admission.research is not services.requests
        or services.admission.inputs is not services.requests
        or services.admission.catalogs is not catalogs
        or services.admission.objects is not objects
    ):
        raise ValueError("component_execution_owner_binding")
    installed = validate(UnboundSingleCasePolicy, policy)
    # Obtain the target only from the original runtime owner, not a second DSN.
    target = services.requests.runtime._target
    workspace = Path(workspace)

    def executor_for(plan):
        from packages.tool_workflow import ExecutionPlan

        if type(plan) is not ExecutionPlan or len(plan.document.steps) != 1:
            raise ValueError("component_one_step_plan_required")
        if (
            plan.document.config != services.compiler.config
            or plan.document.catalog_ref != services.requests.catalog_ref
        ):
            raise ValueError("component_execution_installation_binding")
        requests = services.requests
        code = requests.request_code(requests.request_key)
        child_run = identity("RUN", "research:" + code)
        if plan.document.run_id != child_run:
            raise ValueError("component_execution_child_binding")
        # Original immutable context, not ResearchRuntimeService.read/current
        # admission: a saved confirmation may outlive its patient input bytes.
        context = requests.runtime.read_research_run(child_run)
        context.validate_plan(plan)
        request = context.request
        if (
            context.synthetic != services.research.synthetic
            or request.code != code
            or request.scope != requests.parent.request.scope
            or request.candidate != requests.candidate
            or request.invocation != requests.invocation
            or request.compatibility_policy != requests.policy
            or request.satisfied_terms != requests.scope_terms
            or request.purpose != installed.purpose
        ):
            raise ValueError("component_execution_request_binding")
        bound = validate(
            TrustedLocalPolicy,
            {**installed.model_dump(mode="json"), "allowed_plan_checksums": [plan.checksum]},
        )
        repository = ResearchRuntimeExecutions(
            target, compiler=services.compiler, plans=(plan.to_bundle_json(),)
        )
        return TrustedLocalGPUResearchService(
            compiler=services.compiler,
            catalogs=catalogs,
            objects=objects,
            qc=qc,
            repository=repository,
            policy=bound,
            workspace=workspace,
        )

    return executor_for


def assemble_component_consumer(
    *,
    runtime,
    inputs,
    evidence,
    sessions,
    catalogs,
    profiles,
    objects,
    resources,
    installation,
    producer_run,
    request_key,
    component_id,
    qc,
    clock,
):
    """Public cold assembly over an installed original component producer.

    No dataset ID, case registration, local result JSON, secret, new deadline or
    dispatch is accepted here. Installed profiles and the frozen owner bundle
    must agree. Current consumption is performed by the returned original owner;
    the caller must additionally bind the consumer's task/input and budget.
    """
    from .agent_records import RuntimeDocuments
    from .component_control import restore_component_consumer
    from .deployment import InstalledProfiles, read_installed_bundle

    if type(profiles) is not InstalledProfiles or profiles.record != installation:
        raise ValueError("component_consumer_installation_binding")
    value = read_installed_bundle(installation)
    if value != profiles.current():
        raise ValueError("component_consumer_installation_changed")
    parent = runtime.read_research_run(producer_run)
    case = runtime.get_case_bundle(parent.request.scope.case_code)
    if not case.artifacts:
        raise ValueError("component_consumer_root_missing")
    documents = RuntimeDocuments(
        repo=evidence,
        contexts=runtime,
        sessions=sessions,
        store=inputs.visible_store,
        run_id=parent.run_id,
        root=case.artifacts[0],
    )
    requests = ComponentResearchRequests(
        runtime=runtime,
        inputs=inputs,
        evidence=evidence,
        documents=documents,
        parent_run_id=parent.run_id,
        request_key=request_key,
        component_id=component_id,
        component_policy=value["component_policy"],
        candidate=value["candidate"],
        invocation=value["invocation"],
        catalog_ref=catalogs.catalog.reference(),
        policy=value["compatibility_policy"],
    )
    services = assemble_component_services(
        requests=requests,
        catalogs=catalogs,
        manifests=profiles,
        profiles=profiles,
        probes=profiles,
        allowlists=profiles,
        objects=objects,
        resources=resources,
        config=value["compiler"],
        admission_policy=value["admission_policy"],
        clock=clock,
        synthetic=parent.synthetic,
    )
    executor_for = component_executor_factory(
        services=services,
        catalogs=catalogs,
        objects=objects,
        qc=qc,
        policy=value["execution_policy"],
        workspace=value["execution_workspace"],
    )
    return restore_component_consumer(
        services=services, sessions=sessions, executor_for=executor_for
    )


def assemble_component_services(
    *,
    requests,
    catalogs,
    manifests,
    profiles,
    probes,
    allowlists,
    objects,
    resources,
    config,
    clock,
    synthetic,
    admission_policy=None,
):
    """Original 06B/C and child Runtime owners, under the original parent lock.

    The caller installs reviewed catalog/profile/allowlist readers and object
    ports. No credentials, SQL, current input read or registration happens here.
    `synthetic` is an explicit context identity, not a validation waiver: the
    original compiler/allowlist/runtime must still agree on that identity.
    The parent ComponentPlannerSteps remains the only control/budget bridge.
    """
    if type(requests) is not ComponentResearchRequests or type(synthetic) is not bool:
        raise ValueError("component_original_assembly_required")
    config = compiler_config(config)
    admission = ToolAdmissionService(
        policy=admission_policy or AdmissionPolicy(),
        grounding=None,
        catalogs=catalogs,
        manifests=manifests,
        profiles=profiles,
        probes=probes,
        inputs=requests,
        research=requests,
        objects=objects,
        resources=resources,
        clock=clock,
    )
    compiler = ToolWorkflowCompiler(admission, allowlists, config)
    research = ResearchRuntimeService(
        runtime=requests.runtime,
        research=requests,
        inputs=requests,
        compiler=compiler,
        synthetic=synthetic,
    )
    return ComponentServices(requests, admission, compiler, research)
