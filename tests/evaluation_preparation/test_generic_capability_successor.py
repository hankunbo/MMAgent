from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from packages.evaluation_preparation.generic_capability_successor import (
    CapabilitySuccessor,
    request_limits,
)
from packages.evaluation_preparation.generic_definition_successor import CLAIM
from packages.evaluation_preparation.generic_review import GenericReview


def config(*, batch=3, retries=0):
    return SimpleNamespace(
        review_batching=SimpleNamespace(batch_size=batch),
        transport=SimpleNamespace(max_retries=retries),
    )


def test_budget_counts_all_prior_reservations_and_preserves_review_capacity():
    plan = request_limits(10, ["one", "two", "three"], config())
    assert plan["max_new_requests"] == 12
    assert plan["expected_vision_requests"] == 3
    assert plan["minimum_review_requests"] == 1
    assert request_limits(36, ["one", "two", "three"], config())["max_new_requests"] == 4
    with pytest.raises(ValueError, match="successor_budget"):
        request_limits(37, ["one", "two", "three"], config())
    with pytest.raises(ValueError, match="successor_budget"):
        request_limits(36, ["one", "two", "three"], config(batch=1))


@pytest.mark.parametrize("prior", [-1, 41, True, 1.5])
def test_invalid_original_accounting_cannot_be_reset(prior):
    with pytest.raises(ValueError, match="original_accounting"):
        request_limits(prior, ["one"], config())


@pytest.mark.parametrize("targets,retries", [([], 0), (["one", "one"], 0), (["one"], 1)])
def test_no_empty_duplicate_or_implicit_retry_plan(targets, retries):
    with pytest.raises(ValueError, match="targets_or_retry"):
        request_limits(10, targets, config(retries=retries))


def campaign(tmp_path):
    value = object.__new__(CapabilitySuccessor)
    value.selection = dict(reserve_close_seconds=300, reserve_review_seconds=720)
    value.protocol = dict(deadline=(datetime.now(UTC) + timedelta(seconds=600)).isoformat())
    value.visual_targets = ("new-one", "new-two")
    value.output = tmp_path / "evidence"
    value.spec = {"key": "origin"}
    value.review_key = "new-successor"
    value._capability_role = "review"
    return value


def test_review_reserve_is_usable_and_role_restores_on_nested_error(tmp_path):
    value = campaign(tmp_path)
    assert value.work_deadline() > datetime.now(UTC)
    with pytest.raises(RuntimeError, match="stage_fault"):
        with value.phase_window("new-one"):
            assert value.work_deadline() < datetime.now(UTC)
            with value.phase_window(None):
                assert value.work_deadline() > datetime.now(UTC)
            assert value._capability_role == "vision"
            raise RuntimeError("stage_fault")
    assert value._capability_role == "review"
    assert value.work_deadline() > datetime.now(UTC)


def test_no_lease_or_claim_means_no_dispatch(tmp_path, monkeypatch):
    from packages.evaluation_preparation import generic_capability_successor as module

    value = campaign(tmp_path)
    called = []
    monkeypatch.setattr(GenericReview, "guard", lambda *a, **k: called.append(True))
    monkeypatch.setattr(module, "directory", lambda key: tmp_path / key)
    with pytest.raises(ValueError, match="execution_lease_required"):
        value.guard()
    value._parent_execution_guard = lambda: None
    with pytest.raises(ValueError, match="current_claim_binding"):
        value.guard()
    assert called == []


def test_expired_acquisition_can_read_confirmed_phase_without_new_guard(tmp_path):
    value = campaign(tmp_path)
    confirmed = {"run_id": "RUN:CONFIRMED", "target": "new-one", "accepted": True}
    value.phase_class = lambda *args, **kwargs: object()
    value.base = object()
    value.accepted_observations = lambda: {}
    value.read_model = lambda: None
    value.register_phase = lambda evidence, key: (
        SimpleNamespace(request=object(), run_id="RUN:CONFIRMED"),
        object(),
    )
    value.documents = lambda run: SimpleNamespace(read=lambda key: (confirmed,))

    def fail():
        pytest.fail("confirmed local result must not enter dispatch guard")

    value.guard = fail
    assert value.execute_phase("new-one") == confirmed
    assert value._capability_role == "review"
    with pytest.raises(ValueError, match="unselected_phase"):
        value.execute_phase("unselected")


def test_started_intent_and_exact_claim_required_before_original_budget_guard(
    tmp_path, monkeypatch
):
    import json

    from packages.evaluation_preparation import generic_capability_successor as module

    value = campaign(tmp_path)
    value._parent_execution_guard = lambda: None
    monkeypatch.setattr(module, "directory", lambda key: tmp_path / key)
    for name in ("origin", "new-successor"):
        (tmp_path / name).mkdir()
    (tmp_path / "origin" / CLAIM).write_text(json.dumps(value.selection))
    (tmp_path / "new-successor" / "input.json").write_text(json.dumps(value.selection))
    calls = []
    monkeypatch.setattr(GenericReview, "guard", lambda *a, **k: calls.append(True))
    with pytest.raises(ValueError, match="start_intent_required"):
        value.guard()
    (tmp_path / "new-successor" / "started.json").write_text("{}")
    value.guard()
    assert calls == [True]
    (tmp_path / "origin" / CLAIM).write_text("{}")
    with pytest.raises(ValueError, match="current_claim_binding"):
        value.guard()
    assert calls == [True]


def test_preflight_reports_input_readiness_without_creating_dispatch_authority(
    tmp_path, monkeypatch
):
    from contextlib import contextmanager

    from db.runtime.repository.case_pipeline import CasePipelineRepository
    from packages.evaluation_preparation import generic_capability_successor as module
    from packages.evaluation_preparation import generic_execution_lock
    from packages.multimodal_research import unknown_acquisition

    calls = []

    @contextmanager
    def lease(*args):
        yield lambda: calls.append("lease_checked")

    monkeypatch.setattr(CasePipelineRepository, "__init__", lambda *args: None)
    monkeypatch.setattr(CasePipelineRepository, "guarded_session", lease)
    monkeypatch.setattr(generic_execution_lock, "target", lambda: object())
    monkeypatch.setattr(module, "directory", lambda key: tmp_path / key)
    checked = SimpleNamespace(campaign=SimpleNamespace(config=object()))
    monkeypatch.setattr(module, "origin", lambda key: checked)
    monkeypatch.setattr(
        unknown_acquisition, "check_network", lambda config: calls.append("network_checked")
    )
    monkeypatch.setattr(module, "selection_for", lambda checked, key: {"key": key})
    value = module.preflight("old", "new")
    assert value["execution_ready"] and not value["execution_authorized"]
    assert value["readiness_scope"] == "current_input_selection_and_network_recheck_on_dispatch"
    assert value["api_requests"] == value["new_runtime_runs"] == 0
    assert not (tmp_path / "new").exists() and calls == [
        "lease_checked",
        "network_checked",
        "lease_checked",
    ]
