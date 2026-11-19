"""A locally closed public request is retained, never replayed or used as a source."""

import copy
import json
from pathlib import Path

import pytest

from packages.grounding.models import digest
from packages.multimodal_research.discovery_continuation import ConfirmedDefinitionContinuation
from tests.evaluation_preparation.test_routed_source_backlog import routed_parent

POLICY = "closed-public-wire-backlog/1.0.0"


def parent(tmp_path):
    c, ref, spec, plan, failure, records, writes = routed_parent(tmp_path)
    op = next(o for o in failure["operations"] if o["kind"] == "wire")
    plan["campaign_id"] = Path(ref["parent"]).name
    plan["definition_expansion"] = json.loads(
        Path(spec["active_acquisition"]["path"]).read_bytes()
    )["definition_expansion"]
    failure["protocol_checksum"] = digest(plan)
    original = dict(
        key=op["key"],
        kind="wire",
        wire_kind="fetch",
        binding=dict(
            campaign_id=plan["campaign_id"],
            input_checksum=plan["input"],
            protocol_checksum=digest(plan),
            policy_checksum=digest(plan["definition_expansion"]),
            target=plan["definition_expansion"]["public_target"],
        ),
    )
    records["definition:operation:0"] = (original,)
    records[op["key"] + ":result"] = (
        dict(provider="SOURCE:PROVIDER:EUROPEPMC", reservation_only=True),
    )
    records[op["key"] + ":outcome"] = (
        dict(
            provider_code="SOURCE:PROVIDER:EUROPEPMC",
            provider_version="2.0.0",
            adapter="europe_pmc",
            config_checksum=digest("public connector"),
            operation="fetch",
            request_checksum=digest("public request"),
            response_checksum=None,
            http_status=None,
            requested_at=plan["started_at"],
        ),
    )
    (Path(ref["parent"]) / "stage-failure.json").write_text(json.dumps(failure))
    (Path(ref["parent"]) / "protocol.json").write_text(json.dumps(plan))
    ref.update(failure=digest(failure), protocol=digest(plan), public_wire_policy=POLICY)
    return c, ref, plan, records, writes, op


def test_closed_absence_is_budgeted_history_not_a_response_or_retry(tmp_path):
    c, ref, plan, records, writes, op = parent(tmp_path)
    before = copy.deepcopy(records)
    owner = ConfirmedDefinitionContinuation(c, ref)
    assert owner.check_attempts()[0]["requests"] == 3
    assert owner.proof["closed_public_wire_absences"] == [op["key"]]
    assert owner.proof["public_wire_policy"] == POLICY
    assert len(writes) == 1
    assert all(records[k] == v for k, v in before.items())
    assert ConfirmedDefinitionContinuation(c, ref, read_only=True).claim == owner.claim
    assert ConfirmedDefinitionContinuation(c, ref).claim == owner.claim
    assert len(writes) == 1
    assert json.loads((Path(ref["parent"]) / "protocol.json").read_bytes()) == plan
    records[op["key"] + ":result"][0]["provider"] = "SOURCE:OTHER"
    with pytest.raises(ValueError, match="definition_successor_history_changed"):
        owner.check_attempts()


@pytest.mark.parametrize(
    "mutation",
    [
        "old",
        "unknown_version",
        "missing",
        "partial",
        "provider",
        "kind",
        "time",
        "receipt",
        "operation_missing",
        "wrong_binding",
    ],
)
def test_no_policy_or_incomplete_binding_cannot_claim(tmp_path, mutation):
    c, ref, _, records, writes, op = parent(tmp_path)
    receipt = records[op["key"] + ":outcome"][0]
    if mutation == "old":
        ref.pop("public_wire_policy")
    elif mutation == "unknown_version":
        ref["public_wire_policy"] = "closed-public-wire-backlog/9.0.0"
    elif mutation == "missing":
        records.pop(op["key"] + ":outcome")
    elif mutation == "partial":
        receipt["http_status"] = 200
    elif mutation == "provider":
        receipt["provider_code"] = "SOURCE:OTHER"
    elif mutation == "kind":
        receipt["operation"] = "search"
    elif mutation == "time":
        receipt["requested_at"] = "2000-01-01T00:00:00Z"
    elif mutation == "receipt":
        receipt["request_checksum"] = "not-a-digest"
    elif mutation == "operation_missing":
        records.pop("definition:operation:0")
    else:
        records["definition:operation:0"][0]["binding"]["input_checksum"] = digest("other input")
    with pytest.raises(ValueError, match="definition_successor"):
        ConfirmedDefinitionContinuation(c, ref)
    assert not writes
