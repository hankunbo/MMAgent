"""Pre-checkpoint claims authorize historical closure, never new execution."""

from copy import deepcopy
from types import SimpleNamespace as NS

import pytest

from db.runtime.repository.case_pipeline import identity
from packages.grounding.models import digest
from packages.multimodal_research.stage_failure import check_definition_binding, safe_error


def fixture():
    task = NS(target_code="GOAL:TEST")
    c = NS(
        campaign_id="interrupted",
        protocol={"schema_version": "multimodal-campaign-renewal/1.56.0"},
        base=NS(
            manifest=NS(checksum=digest("input")),
            visible_value=NS(checksum=digest("visible")),
            task=task,
            binding={"case_code": "CASE:TEST"},
            ledger=[{"id": "check"}],
            definition_expansion_policy={"index_keys": [], "public_queries": ["q"]},
        ),
        config={"config": "unchanged"},
        request_key=lambda target: "campaign:" + target,
    )
    from packages.multimodal_research.definition_expansion import read_expansion_policy

    expected = dict(
        campaign_id=c.campaign_id,
        input_checksum=c.base.manifest.checksum,
        target=task.target_code,
        protocol_checksum=digest(c.protocol),
        parent_requirements_checksum=digest(c.base.ledger),
        policy_checksum=digest(read_expansion_policy(c.base.definition_expansion_policy)),
    )
    rid = identity("RUN", "task-research:" + c.request_key("definition-expansion"))
    request = NS(
        input_checksum=c.base.manifest.checksum,
        visible_checksum=c.base.visible_value.checksum,
        task=task,
        scope=NS(case_code="CASE:TEST"),
        policy_ref=NS(checksum=digest((c.base.ledger, c.config, None))),
    )
    context = NS(run_id=rid, request=request)
    c.contexts = NS(read_research_run=lambda *a, **kw: context)
    values = {
        "definition:operation:0": (
            dict(key="definition:query:0", kind="local_search", binding=expected),
        ),
        "definition:operation:1": (dict(key="definition:wire:1", kind="wire", binding=expected),),
    }
    docs = NS(
        run_id=rid, read=values.get, read_many=lambda keys: tuple(values.get(k) for k in keys)
    )
    return c, docs, values, context


def test_original_precheckpoint_claims_are_read_without_mutation():
    c, docs, values, _ = fixture()
    before = deepcopy(values)
    check_definition_binding(c, docs)
    assert values == before
    assert (
        safe_error(ValueError("execution_process_interrupted")) == "execution_process_interrupted"
    )


@pytest.mark.parametrize(
    "change",
    [
        "legacy",
        "run",
        "input",
        "visible",
        "task",
        "case",
        "policy",
        "empty",
        "hole",
        "duplicate",
        "binding",
        "overflow",
        "bad_checkpoint",
        "kind",
    ],
)
def test_unbound_history_cannot_authorize_closure(change):
    c, docs, values, context = fixture()
    if change == "legacy":
        c.protocol["schema_version"] = "multimodal-campaign-renewal/1.11.0"
    elif change == "run":
        docs.run_id = "RUN:WRONG"
    elif change in {"input", "visible"}:
        setattr(context.request, change + "_checksum", digest("wrong"))
    elif change == "task":
        context.request.task = NS(target_code="GOAL:WRONG")
    elif change == "case":
        context.request.scope.case_code = "CASE:WRONG"
    elif change == "policy":
        context.request.policy_ref.checksum = digest("wrong")
    elif change == "empty":
        values.clear()
    elif change == "hole":
        values["definition:operation:2"] = values.pop("definition:operation:1")
    elif change == "duplicate":
        values["definition:operation:1"][0]["key"] = values["definition:operation:0"][0]["key"]
    elif change == "binding":
        values["definition:operation:1"][0]["binding"] = {"caller": "claimed_valid"}
    elif change == "overflow":
        c.definition_operation_limit = 1
    elif change == "kind":
        values["definition:operation:1"][0]["kind"] = "caller_provided_success"
    else:
        values["definition:stage:sources"] = ({"binding": {"wrong": "checkpoint"}},)
    with pytest.raises(ValueError, match="^stage_phase_binding$"):
        check_definition_binding(c, docs)
