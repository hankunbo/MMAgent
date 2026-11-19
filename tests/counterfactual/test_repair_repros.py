"""Independent synthetic repros through accepted facts; no actual patients/models/PG."""

import pytest

from packages.counterfactual.review import review
from packages.evidence_details.current_rejection import current_rejection
from tests.counterfactual.support import candidate, fixture, protection, run
from tests.counterfactual.test_review import reply
from tests.counterfactual.test_types import replacing


@pytest.mark.parametrize("boundary", ["run", "read", "review"])
def test_withdrawn_alternative_cannot_authorize_current_use(boundary):
    x, y = replacing("replace_tool", directions=(-0.7,))
    c = candidate("replace_tool", alternative="alt")
    original = run(x, c)["report"]
    x_before = protection(x)
    requests_before = x.f.budget.used
    y.f.reader.withdrawn = True
    with pytest.raises(ValueError) as caught:
        y.reader()
    assert current_rejection(caught.value) is not None
    x.f.provider.transport.replies.append(reply())
    result, error = None, None
    try:
        if boundary == "run":
            result = x.service.run(c)
        elif boundary == "read":
            result = x.service.read(candidate_id=c.id)
        else:
            result = review(x.service, c, ports=x.f.ports)
    except ValueError as exc:
        error = current_rejection(exc)
        if error is None:
            raise
    print(
        dict(
            boundary=boundary,
            alternate_owner_rejects=True,
            status=result.get("status", "current_report_returned") if result else None,
            error=error,
            additional_requests=x.f.budget.used - requests_before,
        )
    )
    assert error or (result and result.get("status") in {"stopped", "safe_stop"}), (
        "Withdrawn alternative was used as a current candidate/report/advisory input"
    )
    assert x.f.budget.used == requests_before
    assert x.service.read(candidate_id=c.id, historical=True) == original
    assert protection(x) == x_before


def test_minimal_fidelity_set_obeys_declared_support_refutation_conflict_metrics():
    x = fixture(
        statuses=("obtained",) * 4,
        directions=(0.8, -0.8, 0.2, -0.2),
        policy_updates=dict(
            metrics=("support", "refutation", "conflict"),
            tolerance=0.001,
            coverage_retention=0.25,
        ),
    )
    before = protection(x)
    value = x.service.search()
    invalid = []
    for selected in value["minimal_within_explored"]:
        report = x.service.read(candidate_id=selected["candidate_id"])
        deltas = {m: report["comparisons"][m]["delta"] for m in x.policy.metrics}
        if any(d is None or abs(d) > x.policy.tolerance for d in deltas.values()):
            invalid.append(
                dict(
                    candidate=selected["candidate_id"],
                    score_delta=report["comparisons"]["score"]["delta"],
                    deltas=deltas,
                    classification=report["classification"],
                )
            )
    print(
        dict(
            explored=value["explored"],
            selected=len(value["minimal_within_explored"]),
            metrics=x.policy.metrics,
            tolerance=x.policy.tolerance,
            invalid=invalid,
        )
    )
    assert not invalid, "Net-score cancellation is not preservation of the declared metrics"
    assert protection(x) == before


def test_main_source_withdrawal_is_still_rejected():
    x = fixture()
    c = candidate()
    run(x, c)
    x.f.reader.withdrawn = True
    value = x.service.run(c)
    assert value["status"] == "stopped"
    assert x.f.budget.used == 1
