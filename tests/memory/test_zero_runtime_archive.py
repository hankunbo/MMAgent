"""Zero-result facts may be archived; fabricated medical success may not."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from packages.evidence_scoring.accepted import build_accepted
from packages.grounding.models import digest
from packages.memory.runtime_zero_archive import prepare_zero
from tests.evaluation_preparation.test_dynamic_evidence import dynamic_fixture


@pytest.mark.parametrize(
    "change", ["score", "result", "phase", "kind", "review", "recovery", "items", "requirements"]
)
def test_zero_snapshot_rejects_fabricated_or_unbound_contents(change):
    state = dynamic_fixture(1, empty=True).owner.current()
    scope = state.material.scope
    scores = build_accepted(state, None)
    phase = dict(
        run_id=scope.run_id, accepted=False, acceptance_key=None, evidence_limited=True, error=None
    )
    result = dict(case_code=scope.case_code, phases=[phase], complete=True)
    report = dict(result=deepcopy(result), original_review=None, scores=deepcopy(scores))
    docs = {
        "dynamic-phase-input": [state],
        "dynamic-phase-result": [phase],
        "dynamic-case-result": [result],
        "dynamic-zero-evidence": [
            dict(
                status="evidence_limited",
                no_review_requests=True,
                scores=scores,
                requirements=state.policy.requirements,
            )
        ],
    }
    kind = "graph_snapshot"
    if change == "items":
        docs["dynamic-phase-input"] = [dynamic_fixture(1).owner.current()]
    if change == "score":
        report["scores"]["root"]["summary"]["confidence"]["value"] = 0.9
    elif change == "result":
        docs["dynamic-case-result"] = [dict(result, complete=False)]
    elif change == "phase":
        docs["dynamic-phase-result"] = [dict(phase, error="timeout")]
    elif change == "kind":
        kind = "tool_experience"
    elif change == "review":
        report["original_review"] = {}
    elif change == "recovery":
        report["recovery"] = {}
    elif change == "requirements":
        docs["dynamic-zero-evidence"][0]["requirements"] = []
    owner = SimpleNamespace(artifact=lambda *a: pytest.fail("invalid report reached resources"))
    with pytest.raises(ValueError, match="memory_runtime_zero"):
        prepare_zero(
            owner,
            key="test",
            case_code=scope.case_code,
            run_id=scope.run_id,
            origin=object(),
            envelope=dict(value=report, key_checksum=digest("dynamic-case-report")),
            kind=kind,
            target="glaucoma",
            refs={},
            read_document=docs.__getitem__,
        )
