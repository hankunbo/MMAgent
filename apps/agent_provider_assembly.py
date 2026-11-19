"""08C opt-in assembly; original 07/08A factories and 08B selection stay unchanged."""

from dataclasses import dataclass

from agents.provider_agents import ProviderTraceAgent, StructuredRoleAgent, provider_locks
from packages.grounding.models import validate
from packages.llm_provider import DeterministicFakeProvider, RequestBudget, StructuredProvider
from packages.llm_provider.config import load_provider_config
from packages.planner_control.models import Binding


@dataclass(frozen=True)
class AgentPorts:
    trace: ProviderTraceAgent
    expert: StructuredRoleAgent
    source: StructuredRoleAgent
    vision: StructuredRoleAgent
    review: StructuredRoleAgent
    budget: RequestBudget


def assemble_provider_roles(
    *, binding, materials, journal, config=None, live_opt_in=False, fake_replies=(), budget=None
):
    """Pass ports.trace to the ORIGINAL RankedActions(...,agent=...).

    Standard config is Fake. New runs must pin provider_locks(config); old run locks
    cannot be silently upgraded. Domain/Expert workflow orchestration belongs to 08D.
    """
    config = config or load_provider_config()
    binding = validate(Binding, binding)
    locks = {r.role: r for r in binding.locks}
    if any(locks.get(r.role) != r for r in provider_locks(config)):
        raise ValueError("agent_binding_configuration")
    provider = (
        DeterministicFakeProvider(config, fake_replies)
        if config.mode == "fake"
        else StructuredProvider(config, live_opt_in=live_opt_in)
    )
    if budget is not None and (
        budget.run_id != binding.run_id
        or budget.deadline != binding.deadline
        or budget.journal is not journal
        or budget.max_requests != config.max_requests
    ):
        raise ValueError("shared_budget_binding")
    budget = budget or RequestBudget(
        run_id=binding.run_id,
        deadline=binding.deadline,
        max_requests=config.max_requests,
        journal=journal,
    )
    options = dict(provider=provider, materials=materials, budget=budget)
    return AgentPorts(
        trace=ProviderTraceAgent(**options),
        expert=StructuredRoleAgent("expert", **options),
        source=StructuredRoleAgent("source", **options),
        vision=StructuredRoleAgent("vision", **options),
        review=StructuredRoleAgent("review", **options),
        budget=budget,
    )
