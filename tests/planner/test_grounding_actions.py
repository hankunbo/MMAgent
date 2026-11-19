"""Actual 03 propagation + official 05 service; synthetic read ports, no database/process."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from db.runtime.repository.grounding_query import GroundingQueryService
from packages.contracts import DataAvailability, GECEGInstance, GraphNodeStatus
from packages.graph_engine import canonical_checksum
from packages.grounding.models import digest, validate
from packages.grounding.query import normalized_request
from packages.planner_control.actions import RankedActions
from packages.planner_control.grounding_actions import GroundingActions, candidate_for
from packages.planner_control.models import Control, Graph, Phase, RankingPolicy, Reference
from tests.graph.test_rule_engine_03c import _context, _gate_graph, _package
from tests.grounding.query_support import paths, ready, updated
from tests.grounding.test_models import reseal
from tests.planner.test_actions import forbid_process as forbid_process  # noqa: F401
from tests.planner.test_actions import ranked_binding


def formal(*, missing=False, unknown=False, truncated=False, chain=False):
    q, m, p, _, knowledge, reader = ready(chain=chain)
    graph, gate = _gate_graph(
        "OR", (GraphNodeStatus.PENDING,), availabilities=(DataAvailability.DERIVABLE,)
    )
    data = graph.model_dump(mode="json")
    # Keep the original graph contract and public 03 gate semantics.
    import json

    data = json.loads(json.dumps(data).replace("EVIDENCE:SYNTHETIC_1", q.target_node_code))
    data["case_code"] = q.scope.case_code
    data["scope"].update(
        case_code=q.scope.case_code, run_id=q.lock.run_id, graph_code=q.lock.graph_code
    )
    data["graph_code"] = q.lock.graph_code
    data["graph_version"] = 1
    graph = GECEGInstance.model_validate(data)
    gate = replace(gate, child_node_codes=(q.target_node_code,))
    package = _package(graph, (gate,))
    context = _context(package, snapshot_graph=graph)
    lock = reseal(
        type(q.lock),
        {**q.lock.model_dump(mode="json"), "graph_checksum": canonical_checksum(graph)},
    )
    q = updated(q, lock=lock)
    if missing:
        q = updated(q, artifacts=q.artifacts[:1], **({"patient_data_types": []} if unknown else {}))
    if truncated:
        q = updated(
            q,
            config=reseal(
                type(q.config), {**q.config.model_dump(mode="json"), "max_evaluations": 1}
            ),
        )
    q = normalized_request(q)
    box = SimpleNamespace(request=q, graph=graph, package=package, context=context, validations=0)

    class Locks:
        def read(self, run_id):
            return box.request.lock

        def validate_query_scope(self, request):
            if request != box.request:
                raise ValueError("current_scope_mismatch")

        def query_paths(self, run_id, mappings, context):
            return paths(m, p, knowledge, reader)

    service = GroundingQueryService(Locks(), SimpleNamespace(knowledge=knowledge, catalogs=reader))
    original = service.validate_result

    def checked(q, r):
        box.validations += 1
        return original(q, r)

    service.validate_result = checked
    policy = RankingPolicy()
    b = ranked_binding(
        policy,
        case_code=q.scope.case_code,
        run_id=q.lock.run_id,
        graph=Graph(
            role="graph",
            code=q.lock.graph_code,
            version="1",
            checksum=canonical_checksum(graph),
            kind="formal_geceg",
        ),
    )
    locked = dict(
        grounding_request=digest(q),
        grounding_lock=q.lock.checksum,
        catalog=q.lock.catalog_ref.checksum,
        rule=package.package_checksum,
    )
    locks = [
        r.model_copy(update={"checksum": locked[r.role]}) if r.role in locked else r
        for r in b.locks
    ]
    locks.extend(
        Reference(role=r, code="POLICY:" + r.upper(), version="1.0.0", checksum=locked[r])
        for r in ("grounding_request", "grounding_lock")
    )
    b = b.model_copy(update={"locks": tuple(locks)})
    query = service.query(q)
    state = Control(
        binding=b,
        graph=b.graph,
        phase=Phase.SELECT,
        candidates=tuple(candidate_for(c) for c in query.candidates),
    )
    source = GroundingActions(
        grounding=service,
        catalogs=reader,
        read_graph=lambda _: (box.graph, box.package, box.context),
        read_request=lambda _: box.request,
    )
    return state, policy, source, box, reader, knowledge, m, query


def test_formal_03_05_complete_and_independent_or_and_revalidation():
    state, policy, source, box, reader, _, m, result = formal()
    actions = source.read(state, policy)
    work = [a for a in actions.actions if a.kind == "select_candidate"]
    assert len(work) == 2 and box.validations == 1
    assert all(len(a.dependencies) == 3 and len(a.required_inputs) == 2 for a in work)
    assert {a.candidate_ref for a in work} == {c.identity for c in result.candidates}
    reader.disable(m.candidates[0].implementation_ref)
    fresh = source.read(state, policy)
    work = [a for a in fresh.actions if a.kind == "select_candidate"]
    assert len(work) == 1 and work[0].branch.endswith("PATH1")
    assert "safety" in {r.category for r in fresh.rejections}


@pytest.mark.parametrize("unknown", [False, True])
def test_missing_and_is_not_executable_and_unknown_producer_not_capability(unknown):
    state, policy, source, *_ = formal(missing=True, unknown=unknown)
    s = source.read(state, policy)
    assert not any(a.candidate for a in s.actions)
    assert ("unknown_producer" if unknown else "missing_data") in {r.category for r in s.rejections}


def test_search_truncation_preserved_separately():
    state, policy, source, *_ = formal(truncated=True)
    s = source.read(state, policy)
    assert s.enumeration == "search_truncated"
    assert "search_truncated" in {r.category for r in s.rejections}


@pytest.mark.parametrize("field", ["target", "eye", "case", "graph_version", "input", "policy"])
def test_wrong_query_binding_rejected(field):
    state, policy, source, box, *_ = formal()
    q = box.request
    if field == "target":
        q = q.model_copy(update={"target_node_code": "EVIDENCE:OTHER"})
    elif field in {"eye", "case"}:
        q = q.model_copy(
            update={
                "scope": q.scope.model_copy(
                    update={
                        "laterality" if field == "eye" else "case_code": "LEFT"
                        if field == "eye"
                        else "CASE:OID_OTHER0001"
                    }
                )
            }
        )
    elif field == "graph_version":
        q = q.model_copy(
            update={
                "lock": reseal(type(q.lock), {**q.lock.model_dump(mode="json"), "graph_version": 2})
            }
        )
    elif field == "input":
        q = q.model_copy(update={"artifacts": q.artifacts[:1]})
    else:
        q = q.model_copy(
            update={
                "config": reseal(type(q.config), {**q.config.model_dump(mode="json"), "top_k": 2})
            }
        )
    box.request = q
    with pytest.raises(ValueError):
        source.read(state, policy)


def test_withdrawn_during_trace_cannot_use_old_selection():
    state, policy, source, _, _, knowledge, _, _ = formal()

    class Agent:
        def rank(self, s, *, deadline):
            from packages.planner_control.actions import baseline

            knowledge.withdrawn = True
            return baseline(s)

    with pytest.raises(ValueError):
        RankedActions(source, policy=policy, agent=Agent()).select(
            state, lambda: state.binding.started_at
        )


def test_official_result_forgery_and_model_construct_rejected():
    state, policy, source, _, _, _, _, result = formal()
    bad = result.model_copy(update={"safety_checksum": digest("forged")})
    source.grounding.query = lambda _: bad
    with pytest.raises(ValueError):
        source.read(state, policy)
    c = result.candidates[0]
    bad_c = type(c).model_construct(**{**c.model_dump(), "dependencies": c.dependencies[:1]})
    bad = result.model_copy(update={"candidates": (bad_c,)})
    with pytest.raises(ValueError):
        validate(type(result), bad)


def test_03_target_no_longer_legal_cannot_be_ranked():
    state, policy, source, box, *_ = formal()
    box.context = replace(box.context, budget=replace(box.context.budget, remaining_actions=0))
    s = source.read(state, policy)
    assert not any(a.candidate for a in s.actions)
    assert "graph_not_legal" in {r.category for r in s.rejections}


def test_recursive_and_and_or_are_kept_as_05_values():
    state, policy, source, _, _, _, _, result = formal(chain=True)
    s = source.read(state, policy)
    typed = {c.identity: c for c in result.candidates}
    actions = [a for a in s.actions if a.candidate]
    assert actions and any(d.producer_options for a in actions for d in a.dependencies)
    assert all(a.dependencies == typed[a.candidate_ref].dependencies for a in actions)
