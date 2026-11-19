from types import SimpleNamespace

from packages.grounding.models import digest
from packages.task_runtime.models import DispatchRequest, Profile, Receipt
from packages.task_runtime.owner import PlannerTaskOwner
from tests.planner.support import setup


def configured():
    planner, runtime, domain = setup()
    binding = runtime.state.binding
    context = SimpleNamespace(
        run_id=binding.run_id,
        checksum=binding.run_context_checksum,
        synthetic=True,
        request=SimpleNamespace(scope=SimpleNamespace(case_code=binding.case_code)),
    )
    reader = SimpleNamespace(read_run=lambda _: SimpleNamespace(context=context))
    owner = PlannerTaskOwner(planners={binding.run_id: planner}, runtime=reader)
    request = DispatchRequest(
        run_id=binding.run_id,
        case_code=binding.case_code,
        context_checksum=binding.run_context_checksum,
        idempotency_key="submit",
        expected_revision=0,
    )
    receipt = Receipt(
        task_id="TASK:DISPATCH",
        request=request,
        profile=Profile(),
        binding_checksum=digest(binding),
        deadline=binding.deadline,
        state="queued",
        control_checksum=digest(runtime.state),
    )
    return owner, planner, runtime, domain, receipt, context
