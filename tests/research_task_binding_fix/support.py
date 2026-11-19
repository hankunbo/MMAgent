"""Deterministic synthetic inputs and fresh public service assembly."""

from db.runtime.repository.research_lifecycle import ResearchLifecycleRepository
from packages.research_runtime.service import ResearchRuntimeService
from packages.research_runtime.single_case import ResearchSingleCaseService
from tests.research_task_binding_r01.diagnostics import png


def fixed_input(identity):
    # Both independent supports remain; distinct bytes and explicit zero/nonzero.
    # Random artifact suffixes never select the first pixel or the score.
    stable = identity.split(":ASSET:", 1)[0]
    if stable == "independent-1" or stable.startswith("nonzero"):
        return png(bytes([0, 6, 5, 4, 3, 2]))
    return png(bytes([1, 2, 3, 4, 5, 6]))


def reassemble(driver, services):
    runtime = ResearchLifecycleRepository(driver.runtime._target)
    research = ResearchRuntimeService(
        runtime=runtime,
        research=driver.research.research,
        inputs=driver.research.inputs,
        compiler=driver.research.compiler,
        synthetic=True,
    )
    services.clear()
    original = driver.bind_plan

    def bind(plan, key):
        executor, scores = original(plan, key)
        scores.runtime = runtime
        return executor, scores

    return ResearchSingleCaseService(
        runtime=runtime,
        research=research,
        choice=driver.choice,
        bind_plan=bind,
        config_checksum=driver.config_checksum,
        budget_seconds=driver.budget_seconds,
        completion_goal=driver.task_completion.goal if driver.task_completion else None,
    )
