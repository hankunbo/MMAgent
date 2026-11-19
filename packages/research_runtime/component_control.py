"""Explicit original Planner binding for an installed component acquisition.

The graph is the saved visible input root, not a fabricated medical evidence
tree. No Provider/search or medical publication is installed by this assembly.
The parent origin and supplied frozen profile own the only action/time budget.
"""

from datetime import datetime, timedelta

from agents.planner.service import Planner
from packages.grounding.models import digest, validate
from packages.planner_control.models import Binding, Graph, Outcome, Profile, Reference

from .component_steps import ComponentPlannerSteps
from .task_context import check_current_inputs


def restore_component_consumer(*, services, sessions, executor_for):
    """Reassemble a confirmed producer from original saved Control/prepare.

    This does not begin/resume a producer, register a patient, or authorize a
    consumer. The consumer separately checks its case/task and budget, and uses
    read_current_result before consuming any artifact. No local evidence file
    supplies the producer's profile or acquisition deadline.
    """
    from .component_deployment import ComponentServices

    if type(services) is not ComponentServices or not callable(executor_for):
        raise ValueError("component_original_consumer_assembly_required")
    requests = services.requests
    state = sessions.read(requests.parent.run_id)
    if state is None or state.pending is not None or state.confirmed != 1:
        raise ValueError("component_confirmed_control_required")
    key = (
        "component-plan:"
        + digest(dict(request_key=requests.request_key, selected=state.selected))[7:]
    )
    receipt = requests.documents.read(key)
    if receipt is None or len(receipt) != 1:
        raise ValueError("component_prepare_receipt_required")
    work_deadline = datetime.fromisoformat(receipt[0]["work_deadline"])
    binding, owner, _ = assemble_component_planner(
        services=services,
        sessions=sessions,
        executor_for=executor_for,
        profile=state.binding.profile,
        work_deadline=work_deadline,
    )
    # Validate every persisted prepare field and the original child, not merely
    # the deadline extracted for construction. Current admission is a separate
    # explicit read_current_result operation, not historical reconstruction.
    owner._saved(state)
    if binding != state.binding or sessions.read(requests.parent.run_id) != state:
        raise ValueError("component_consumer_control_changed")
    return owner, state


class ComponentSnapshots:
    def __init__(self, requests, binding):
        self.requests, self.binding = requests, binding

    def current(self, binding):
        if binding != self.binding:
            raise ValueError("component_control_binding_changed")
        owner = self.requests
        if owner.runtime.read_research_run(owner.parent.run_id) != owner.parent:
            raise ValueError("component_parent_context_changed")
        check_current_inputs(owner.parent, owner.inputs)
        owner.documents._check_root()
        root = owner.documents.root
        if (binding.graph.code, binding.graph.checksum) != (root.asset_code, root.checksum):
            raise ValueError("component_control_root_changed")
        return binding.graph

    def step(self, binding, action):
        graph = self.current(binding)
        if action not in {"load", "retrieve", "instantiate", "verify"}:
            raise ValueError("component_snapshot_action")
        return Outcome(
            reason="installed_component_input_and_definition_refs_only",
            refs=(
                Reference(
                    role="input_snapshot",
                    code=graph.code,
                    version=graph.version,
                    checksum=graph.checksum,
                ),
            ),
        )


def assemble_component_planner(*, services, sessions, executor_for, profile, work_deadline):
    """Create a binding, not a run or a new deadline; caller explicitly begins it.

    Reassembly uses the original persisted run origin. Existing Control with
    different locks/profile is rejected, never amended in place.
    """
    profile = validate(Profile, profile)
    owner = services.requests
    context = owner.parent
    started, context_checksum = sessions.origin(context.run_id)
    if context_checksum != context.checksum:
        raise ValueError("component_control_context")
    deadline = started + timedelta(seconds=profile.seconds)
    if work_deadline.tzinfo is None or not started < work_deadline <= deadline:
        raise ValueError("component_control_work_deadline")
    root = owner.documents.root
    req = context.request

    def ref(role, code, value, version="1.0.0"):
        return Reference(role=role, code=code, version=version, checksum=digest(value))

    locks = [
        ref("acquisition", "POLICY:COMPONENT_CUTOFF", {"work_deadline": work_deadline.isoformat()}),
        ref("admission", "POLICY:COMPONENT_ADMISSION", services.admission.policy),
        ref("knowledge", "POLICY:COMPONENT_DEFINITION_REFS", req.task.definition_refs),
        ref("template", req.task.code, req.task),
        ref("rule", req.policy_ref.code, req.policy_ref),
        Reference(
            role="catalog",
            code=req.lock.catalog_ref.code,
            version=req.lock.catalog_ref.version,
            checksum=req.lock.catalog_ref.checksum,
        ),
        ref("mapping", "POLICY:COMPONENT_REGISTRATION", owner.registration_binding),
        Reference(
            role="execution",
            code="POLICY:COMPONENT_EXECUTION",
            version=services.compiler.config.version,
            checksum=services.compiler.config.checksum,
        ),
        ref("ranking", "POLICY:REGISTERED_ACTIONS", {"mode": "deterministic_no_agent"}),
        ref("prompt", "POLICY:NO_PROVIDER", {"provider_installed": False}),
        ref("model", "POLICY:ADMITTED_COMPONENT", owner.candidate),
        ref("sources", "POLICY:COMPONENT_DEFINITION_REFS", req.task.definition_refs),
    ]
    binding = Binding(
        case_code=req.scope.case_code,
        run_id=context.run_id,
        run_context_checksum=context_checksum,
        input_checksum=req.input_checksum,
        visible_checksum=req.visible_checksum,
        graph=Graph(
            role="graph",
            code=root.asset_code,
            version="multimodal-input-object/1.0.0",
            checksum=root.checksum,
            kind="research_artifact",
        ),
        locks=tuple(locks),
        profile=profile,
        started_at=started,
        deadline=deadline,
    )
    existing = sessions.read(context.run_id)
    if existing is not None and existing.binding != binding:
        raise ValueError("component_existing_control_binding")
    steps = ComponentPlannerSteps(
        snapshots=ComponentSnapshots(owner, binding),
        requests=owner,
        research=services.research,
        choice=services.choice,
        executor_for=executor_for,
        sessions=sessions,
        work_deadline=work_deadline,
    )
    return binding, steps, Planner(runtime=sessions, owner=steps)
