"""Actual OR admission plus bounded pure-graph adversarial unit tests."""

import pytest

from packages.grounding.models import digest, validate
from packages.tool_workflow.compiler import topology
from packages.tool_workflow.models import WorkflowChoice
from tests.tool_admission.test_output_targets import alternative_fixture
from tests.tool_workflow.support import choose, prepared


def test_or_pool_is_not_an_and_execution_plan(tmp_path):
    base, _ = alternative_fixture(tmp_path)
    f = prepared(tmp_path, base=base)
    pools = [p for n in f.context.admission.retained for p in n.inputs if p.producer_refs]
    assert pools and all(len(p.producer_refs) == 2 and p.cardinality_max == 1 for p in pools)
    f.choice = choose(f.context.admission, all_roots=True)
    plan = f.compiler.compile(f.context, f.choice)
    assert len(plan.document.steps) == 3  # two roots share one of two producers
    unused = set(n.identity for n in f.context.admission.retained) - set(
        s.candidate_ref for s in plan.document.steps
    )
    assert len(unused) == 1
    expanded = f.choice.model_dump(mode="json")
    root = next(n for n in expanded["nodes"] if n["candidate_ref"] in expanded["roots"])
    selected = next(p for p in root["inputs"] if any(s["kind"] == "producer" for s in p["sources"]))
    selected["sources"] = []
    with pytest.raises(ValueError, match="selected_input_cardinality"):
        f.compiler.compile(f.context, expanded)
    pool = next(p for p in pools if p.port.port_name == selected["port"]["port_name"])
    nodes = {n.identity: n for n in f.context.admission.retained}
    selected["sources"] = [
        dict(
            kind="producer",
            candidate_ref=i,
            port=nodes[i].candidate.output_port.model_dump(mode="json"),
        )
        for i in pool.producer_refs
    ]
    with pytest.raises(ValueError, match="selected_input_cardinality"):
        f.compiler.compile(f.context, expanded)


def test_unused_selected_worksheet_nodes_are_pruned(tmp_path):
    f = prepared(tmp_path, chain=True)
    worksheet = choose(f.context.admission, all_roots=True).model_dump(mode="json")
    worksheet["roots"] = list(f.choice.roots)
    assert f.compiler.compile(f.context, worksheet) == f.compiler.compile(f.context, f.choice)


@pytest.mark.parametrize(
    "change,reason",
    [
        ("cycle", "dependency_cycle"),
        ("port", "producer_port_mismatch"),
        ("unit", "conversion_required"),
        ("laterality", "conversion_required"),
        ("type", "conversion_required"),
        ("dangling", "producer_outside_admission"),
    ],
)
def test_pure_topology_rejects_corrupted_edges(tmp_path, change, reason):
    f = prepared(tmp_path, chain=True)
    admission = f.context.admission
    data = f.choice.model_dump(mode="json")
    root = next(n for n in data["nodes"] if n["candidate_ref"] == data["roots"][0])
    input_ = next(p for p in root["inputs"] if any(s["kind"] == "producer" for s in p["sources"]))
    source = input_["sources"][0]
    producer = next(n for n in admission.retained if n.identity == source["candidate_ref"])
    if change == "port":
        source["port"]["port_name"] = "wrong"
    elif change == "dangling":
        source["candidate_ref"] = digest("absent")
    else:
        c = producer.candidate
        if change == "cycle":
            parent = next(n for n in admission.retained if n.identity == data["roots"][0])
            c = c.model_copy(update={"output": parent.candidate.inputs[0].constraint})
            # Self-loop on the producer with exact type; pure validator must not recurse forever.
            port = producer.inputs[0]
            c = c.model_copy(
                update={"inputs": (c.inputs[0].model_copy(update={"constraint": c.output}),)}
            )
            producer = producer.model_copy(
                update={
                    "candidate": c,
                    "inputs": (port.model_copy(update={"producer_refs": (producer.identity,)}),),
                }
            )
            selected = next(n for n in data["nodes"] if n["candidate_ref"] == producer.identity)
            selected["inputs"][0]["sources"] = [
                dict(
                    kind="producer",
                    candidate_ref=producer.identity,
                    port=c.output_port.model_dump(mode="json"),
                )
            ]
        else:
            constraint = c.output.model_dump(mode="json")
            if change == "unit":
                constraint["unit"] = dict(state="known", code="CONCEPT:UNIT_MM")
            elif change == "laterality":
                constraint["laterality"] = "LEFT"
            else:
                constraint["artifact_ref"]["version"] = "2.0.0"
            c = c.model_copy(update={"output": validate(type(c.output), constraint)})
            producer = producer.model_copy(update={"candidate": c})
        admission = admission.model_copy(
            update={
                "retained": tuple(
                    producer if n.identity == producer.identity else n for n in admission.retained
                )
            }
        )
    with pytest.raises(ValueError, match=reason):
        topology(admission, validate(WorkflowChoice, data), f.config.max_steps)


def test_repeated_implementation_nodes_explicitly_unsupported(tmp_path):
    f = prepared(tmp_path, chain=True)
    choice = choose(f.context.admission, all_roots=True)
    nodes = list(f.context.admission.retained)
    roots = [n for n in nodes if n.identity in choice.roots]
    altered = roots[1].model_copy(
        update={
            "candidate": roots[1].candidate.model_copy(
                update={"implementation_ref": roots[0].candidate.implementation_ref}
            )
        }
    )
    admission = f.context.admission.model_copy(
        update={"retained": tuple(altered if n.identity == altered.identity else n for n in nodes)}
    )
    with pytest.raises(ValueError, match="repeated_implementation_nodes_unsupported"):
        topology(admission, choice, f.config.max_steps)


def test_step_budget(tmp_path):
    f = prepared(tmp_path, chain=True)
    with pytest.raises(ValueError, match="step_budget_exceeded"):
        topology(f.context.admission, f.choice, 1)


def test_artifact_candidate_pool_exceeding_max_selects_only_one(tmp_path):
    from tests.grounding.query_support import updated
    from tests.tool_admission.support import fixture, register_inputs

    base = fixture(tmp_path)
    extra = base.q.artifacts[0].model_copy(update={"asset_code": "ASSET:OID_EXTRA0001"})
    base.q = register_inputs(
        updated(base.q, artifacts=(*base.q.artifacts, extra)), base.owners, base.objects
    )
    base.locks.q = base.q
    base.result = base.grounding.query(base.q)
    base.selection = base.selection.model_copy(
        update={"candidate_refs": tuple(c.identity for c in base.result.candidates)}
    )
    f = prepared(tmp_path, base=base)
    assert any(
        len(p.artifacts) > p.cardinality_max for n in f.context.admission.retained for p in n.inputs
    )
    plan = f.compiler.compile(f.context, f.choice)
    assert all(
        len(i.sources)
        <= next(
            p.cardinality_max
            for p in f.context.admission.retained[0].inputs
            if p.port.port_name == i.port.port_name
        )
        for s in plan.document.steps
        for i in s.inputs
    )
    assert len(plan.document.steps) == 1
