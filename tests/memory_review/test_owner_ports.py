"""Use original 03/05 legality and 06 readmission, with synthetic registered assets."""

import pytest

from agents.planner.service import Planner
from packages.memory_review.models import Query
from packages.memory_review.planner import MemoryRankedActions
from packages.planner_control.models import Command
from packages.retrieval.memory.ports import BoundQueryReader
from tests.memory_review.support import QueryOwner, conditions
from tests.memory_review.test_planner import Documents, RetrievalPort
from tests.planner.support import FakeOwner, MemoryRuntime
from tests.planner.test_grounding_actions import formal


def test_memory_high_vote_cannot_restore_candidate_rejected_by_03_05():
    state, policy, source, box, reader, _, mapping, _ = formal()
    initial = source.read(state, policy)
    forbidden = next(
        a for a in initial.actions if a.candidate and a.branch == mapping.candidates[0].code
    )
    reader.disable(mapping.candidates[0].implementation_ref)
    # Original Planner executes the same select path over the original official 05 adapter.
    runtime = MemoryRuntime()
    runtime.state = state
    docs = Documents(state.binding.run_id)
    retrieval = RetrievalPort(QueryOwner(state.binding.run_id, state.binding.case_code).value)
    retrieval.groups["success"] = [dict(cases=["CASE:PRIOR"], branches=[forbidden.branch])]
    planner = Planner(
        runtime=runtime,
        owner=FakeOwner(state.binding),
        ranking=MemoryRankedActions(source, policy=policy, retrieval=retrieval, documents=docs),
    )
    cmd = Command(key="10d-05-select", action="select", expected_revision=state.revision)
    event = planner.step(state.binding.run_id, cmd)
    decision = event.outcome.decision
    assert forbidden.id not in decision.ranking.ranked_ids
    assert forbidden.id not in {a.id for a in decision.action_set.actions}
    assert decision.selected.branch != forbidden.branch and box.validations > 1


def test_06_current_use_and_catalog_revocation_are_rechecked_without_execution(tmp_path):
    from tests.tool_admission.support import fixture
    from tests.tool_admission.test_research import research

    f = fixture(tmp_path)
    request = research(f, tmp_path)
    versions = {
        request.candidate.implementation_ref.code: request.candidate.implementation_ref.version
    }
    raw = conditions().model_dump()
    raw["tool_versions"] = versions
    q = Query(
        **raw,
        run_id=request.lock.run_id,
        case_code=request.scope.case_code,
        test_context="10d-owned-fixture",
    )
    owner = BoundQueryReader(
        read_query=lambda _: q,
        read_rules=lambda _: q.rule_versions,
        tool_requests=lambda _: (request,),
        admission=f.service,
    )
    assert owner.current(q.run_id) == q
    owner.check_versions(q)
    f.service.catalogs.disable(request.candidate.implementation_ref)
    with pytest.raises(ValueError, match="memory_tool_not_currently_admitted"):
        owner.check_versions(q)
    with pytest.raises(ValueError, match="memory_tool_version_changed"):
        owner.check_versions(q.model_copy(update={"tool_versions": {}}))


def test_old_contract_bytes_are_unchanged():
    from pathlib import Path

    from packages.memory.models import read_formal

    for kind in ("Episode", "TrajectoryStep", "CounterfactualPair"):
        raw = Path("packages/contracts/examples/v1/" + kind + ".valid.json").read_bytes()
        _, returned = read_formal(raw)
        assert returned is raw
