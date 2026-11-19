from pathlib import Path
from types import SimpleNamespace

import pytest

from packages.evaluation_preparation.comprehensive_review import ComprehensiveReview


def test_review_successor_never_reacquires_and_retains_observation_owner():
    c = object.__new__(ComprehensiveReview)
    c.output = Path("new")
    observed = {"item": object()}
    c.acquisition = SimpleNamespace(
        phase_path=lambda t: Path("old") / t,
        accepted_observations=lambda: observed,
    )
    assert c.before_review() is None
    assert c.paths() == [Path("new/review")]
    assert c.phase_path(None) == Path("new/review")
    assert c.phase_path("item") == Path("old/item")
    assert c.accepted_observations() is observed


@pytest.mark.parametrize("phase,pending", [("select", None), ("stopped", object())])
def test_old_unknown_or_unclosed_control_does_not_authorize_successor(tmp_path, phase, pending):
    from apps.vectorengine_single_case import write_once

    write_once(tmp_path / "request.json", {"lock": {"run_id": "RUN:OLD"}})
    c = object.__new__(ComprehensiveReview)
    c.acquisition = SimpleNamespace(prior=lambda: {"reservations": 25}, paths=lambda: [tmp_path])
    c.sessions = SimpleNamespace(read=lambda _: SimpleNamespace(phase=phase, pending=pending))
    with pytest.raises(ValueError, match="predecessor_requires_control_confirmation"):
        c.prior()


def test_previous_real_attempts_including_errors_still_count():
    c = object.__new__(ComprehensiveReview)
    c.acquisition = SimpleNamespace(
        prior=lambda: {"reservations": 25}, paths=lambda: [], usage=lambda: 7
    )
    assert c.prior()["reservations"] == 32
