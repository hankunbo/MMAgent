"""Public one-step research consumption, version separation and original v5 calculation."""

from types import SimpleNamespace

import pytest

from apps.evidence_graph_scores import attach
from apps.planner_assembly import ResearchSnapshotSteps, assemble_research_planner
from packages.grounding.models import digest
from packages.planner_control.models import Command, Reference
from tests.planner.support import MemoryRuntime, binding
from tests.test_evidence_records import parent  # noqa: F401
from tests.test_evidence_scoring_current import current  # noqa: F401
from tests.test_evidence_scoring_interpretation import interpreted  # noqa: F401
from tests.test_evidence_scoring_participation import graded  # noqa: F401


@pytest.mark.parametrize("ranked", [False, True])
def test_saved_records_v5_steps_no_new_interpretation(current, monkeypatch, ranked):  # noqa: F811
    import json

    import apps.planner_assembly as module
    from packages.object_store import checksum

    base, score = current
    view = attach(base, score)
    parent_ref = Reference(
        role="graph",
        code="ASSET:PARENT",
        version="case-evidence-graph/1.18.0",
        checksum=digest("parent"),
    )
    raw = json.dumps(
        dict(
            schema="case-evidence-graph/1.19.0",
            base=dict(code=parent_ref.code, checksum=parent_ref.checksum),
        )
    ).encode()
    b = binding()
    if ranked:
        from packages.planner_control.models import RankingPolicy
        from tests.planner.test_actions import ranked_binding

        b = ranked_binding(RankingPolicy())
    graph = b.graph.model_copy(update={"checksum": checksum(raw), "parent": parent_ref})
    visible = SimpleNamespace(checksum=b.input_checksum)
    visible_value = dict(checksum=b.input_checksum)

    class Visible:
        def read(self, case):
            return visible

    artifact = SimpleNamespace(
        asset_code=graph.code,
        checksum=graph.checksum,
        artifact_type_code="ARTIFACT:CASE_EVIDENCE_GRAPH",
        uri="object://synthetic",
    )
    artifacts = [artifact]
    repo = SimpleNamespace(get_case_bundle=lambda c: SimpleNamespace(artifacts=artifacts))
    objects = SimpleNamespace(read_bounded=lambda *args: raw)
    original_digest = module.digest
    monkeypatch.setattr(
        module, "digest", lambda v: digest(visible_value) if v is visible else original_digest(v)
    )
    reads = []

    def read(case, code, **kwargs):
        reads.append((case, code))
        return view

    monkeypatch.setattr(module, "read_view", read)
    b = b.model_copy(update=dict(graph=graph, visible_checksum=digest(visible_value)))
    snapshots = ResearchSnapshotSteps(
        visible=Visible(),
        repository=repo,
        objects=objects,
        inventory_checksum=digest([(graph.code, graph.checksum)]),
        locks=b.locks,
    )
    runtime = MemoryRuntime()
    factory = module.assemble_ranked_research_planner if ranked else assemble_research_planner
    planner = factory(runtime=runtime, snapshots=snapshots)
    planner.begin(b)
    for action in (
        "load",
        "retrieve",
        "instantiate",
        "identify",
        "select",
        "prepare",
        "execute",
        "verify",
        "review",
    ):
        s = runtime.state
        c = Command(key=action, expected_revision=s.revision, action=action)
        e = planner.step(b.run_id, c)
        assert e.outcome.status == "ok", e
    assert runtime.state.started == 0 and runtime.state.confirmed == 0
    assert runtime.state.result_refs[-1].version == "case-research-scoring/5.0.0"
    assert runtime.state.gaps[0].kind == "unavailable_port"
    assert view == attach(base, score) and len(reads) > 5
    artifacts.append(
        SimpleNamespace(
            asset_code="ASSET:EXTERNAL",
            checksum=digest("external"),
            artifact_type_code="ARTIFACT:CASE_EVIDENCE_GRAPH",
        )
    )
    with pytest.raises(ValueError, match="inventory_changed"):
        snapshots.current(b)
    with pytest.raises(ValueError, match="formal_graph"):
        snapshots.current(
            b.model_copy(
                update={
                    "graph": graph.model_copy(update={"kind": "formal_geceg", "code": "GRAPH:ANON"})
                }
            )
        )
