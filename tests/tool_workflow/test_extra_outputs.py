"""Every declared output needs request compatibility, including unconsumed side outputs."""

import pytest

from packages.grounding.models import validate
from packages.grounding.synthetic import SyntheticCatalogReader
from tests.grounding.query_support import request_for
from tests.tool_admission.support import fixture, register_deployments, register_inputs, reseal
from tests.tool_workflow.support import prepared


def with_extra_output(root, change):
    base = fixture(root)
    tools = []
    for tool in base.catalog.tools:
        port = tool.output_ports[0].model_copy(update={"port_name": "secondary"})
        constraint = tool.outputs[0].model_dump(mode="json")
        if change == "laterality":
            constraint["laterality"] = "LEFT"
        elif change == "unit":
            constraint["unit"] = dict(state="unknown", code=None)
        elif change == "coordinate":
            constraint["coordinate_space"] = "UNKNOWN"
        elif change == "scope":
            constraint["scope_terms"].append("CONCEPT:UNSATISFIED_CONDITION")
        tools.append(
            validate(
                type(tool),
                {
                    **tool.model_dump(mode="json"),
                    "output_ports": [*tool.output_ports, port],
                    "outputs": [*tool.outputs, constraint],
                },
            )
        )
    catalog = reseal(base.catalog, tools=tools)
    mapping = reseal(base.mapping, catalog_ref=catalog.reference())
    request = register_inputs(
        request_for(mapping, base.policy, artifacts=base.q.artifacts), base.owners, base.objects
    )
    reader = SyntheticCatalogReader([catalog])
    base.catalog, base.mapping, base.q = catalog, mapping, request
    base.locks.q, base.locks.mapping, base.locks.catalogs = request, mapping, reader
    base.grounding.mappings.catalogs = reader
    base.service.catalogs = reader
    invocations = register_deployments(request, catalog, base.owners, root)
    base.result = base.grounding.query(request)
    base.selection = base.selection.model_copy(
        update={
            "candidate_refs": tuple(c.identity for c in base.result.candidates),
            "invocations": invocations,
        }
    )
    return prepared(root, base=base)


@pytest.mark.parametrize("change", ["laterality", "unit", "coordinate", "scope"])
def test_extra_output_cannot_bypass_request_conditions(tmp_path, change):
    f = with_extra_output(tmp_path, change)
    assert f.context.admission.status == "preparation_only"
    with pytest.raises(ValueError, match="output_constraint_"):
        f.compiler.compile(f.context, f.choice)


def test_compatible_multiport_outputs_have_distinct_addresses(tmp_path):
    f = with_extra_output(tmp_path, "compatible")
    plan = f.compiler.compile(f.context, f.choice)
    assert len(plan.document.steps[0].outputs) == 2
    assert len({o.logical_uri for o in plan.document.steps[0].outputs}) == 2
    assert all(not o.exists and o.bytes_checksum is None for o in plan.document.steps[0].outputs)
    assert f.compiler.validate_plan(plan) == plan
