"""Installer contracts on actual Query/Retrieval/Planner, with an explicitly empty test port."""

from types import SimpleNamespace

import pytest

from packages.grounding.models import digest
from packages.memory_review.planner import MemoryRankedActions
from packages.memory_review.task_models import TaskQuery
from packages.planner_control.actions import RankedActions
from tests.memory_review.support import QueryOwner
from tests.memory_review.test_planner import Documents
from tests.planner.support import command
from tests.planner.test_actions import setup_ranked
from tests.research_task_completion.protocol import GROUPS, install


@pytest.mark.parametrize("group", ("A", "B", "C", "D"))
def test_installer_drives_actual_query_and_original_planner(group):
    planner, runtime, owner, source = setup_ranked()
    state = runtime.state
    rid = state.binding.run_id
    query = QueryOwner(rid, state.binding.case_code)
    query.value = TaskQuery(
        **query.value.model_dump(exclude={"schema_version"}),
        task_profile=digest("contract-only-not-evaluation"),
    )
    original_query = query.value
    documents = Documents(rid)
    reads = []

    def context(run_id):
        reads.append(run_id)
        return SimpleNamespace(
            synthetic=True,
            request=SimpleNamespace(scope=SimpleNamespace(case_code=state.binding.case_code)),
        )

    repo = SimpleNamespace(
        contexts=SimpleNamespace(read_research_run=context), publication_keys=lambda: ()
    )
    planner.ranking = install(
        group, source, query_reader=query, repository=repo, documents=documents
    )
    cmd = command(planner, "select")
    event = planner.step(rid, cmd)
    assert event.outcome.decision.selected.branch == "A"
    assert runtime.state.memory == () and runtime.state.reserved == 0
    if group == "A":
        assert type(planner.ranking) is RankedActions
        assert not reads and not query.calls and documents.writes == 0
    else:
        assert isinstance(planner.ranking, MemoryRankedActions)
        receipt = planner.ranking.read_influence(runtime.state, event.outcome.decision)
        assert tuple(receipt["retrieval"]["query"]["groups"]) == GROUPS[group]
        assert receipt["retrieval"]["status"] == "ok"
        assert receipt["schema_version"] == "memory-selection/1.1.0"
        assert not any(receipt["retrieval"]["groups"].values())
        assert receipt["baseline_order"] == receipt["final_order"]
        assert documents.writes == 1
        assert query.calls and reads
    before = (query.calls, len(reads), documents.writes, list(owner.calls))
    assert planner.step(rid, cmd) == event
    assert (query.calls, len(reads), documents.writes, owner.calls) == before
    assert query.value == original_query
    assert "execute" not in owner.calls
