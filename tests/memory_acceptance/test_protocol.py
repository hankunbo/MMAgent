"""Installer contracts on actual Query/Retrieval/Planner, with an explicitly empty test port."""

import json
from types import SimpleNamespace

import pytest

from packages.memory_review.planner import MemoryRankedActions
from packages.planner_control.actions import RankedActions
from tests.memory_acceptance.protocol import GROUPS, install, load_protocol
from tests.memory_review.support import QueryOwner
from tests.memory_review.test_planner import Documents
from tests.planner.support import command
from tests.planner.test_actions import setup_ranked


@pytest.mark.parametrize("group", ("A", "B", "C", "D"))
def test_installer_drives_actual_query_and_original_planner(group):
    planner, runtime, owner, source = setup_ranked()
    state = runtime.state
    rid = state.binding.run_id
    query = QueryOwner(rid, state.binding.case_code)
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
        assert not any(receipt["retrieval"]["groups"].values())
        assert receipt["baseline_order"] == receipt["final_order"]
        assert documents.writes == 1
        assert query.calls and reads
    before = (query.calls, len(reads), documents.writes, list(owner.calls))
    assert planner.step(rid, cmd) == event
    assert (query.calls, len(reads), documents.writes, owner.calls) == before
    assert query.value == original_query
    assert "execute" not in owner.calls


@pytest.mark.parametrize("drift", ("old_config", "D_includes_recovery", "A_installs_memory"))
def test_rejects_historical_or_mislabelled_group_sets(tmp_path, drift):
    if drift == "old_config":
        from pathlib import Path

        with pytest.raises(ValueError, match="10e_protocol_mismatch"):
            load_protocol(Path("configs/memory/10e-offline-groups.json"))
        return
    value = load_protocol()
    if drift == "D_includes_recovery":
        value["groups"]["D"]["query_groups"].append("recovery")
    else:
        value["groups"]["A"]["install_memory_ranked_actions"] = True
    path = tmp_path / "drift.json"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="10e_protocol_mismatch"):
        load_protocol(path)
