"""Explicit UNKNOWN analysis is not implicit retry or a legacy policy change."""

from types import SimpleNamespace

import pytest

from packages.multimodal_research import campaign_renewal as renewal
from packages.multimodal_research import unknown_acquisition as owner
from tests.evaluation_preparation.test_exact_acquisition_budget import configuration, valid_plan


@pytest.mark.parametrize(
    "change",
    [
        dict(mode="review"),
        dict(exact_request_budget=False),
        dict(acquisition_target=[]),
        dict(clarify_review=["old"]),
        dict(pending_review=True),
        dict(acquisition_increment_only=True),
        dict(acknowledge_unknown_request="old"),
    ],
)
def test_unsupported_mixed_modes_rejected_before_owner_or_write(change):
    args = dict(
        facts="missing",
        previous="missing",
        output="never-created",
        mode="acquisition",
        authorization="explicit synthetic test",
        exact_request_budget=True,
        acquisition_target=["target"],
        acknowledge_unknown_acquisition_request=["request:x"],
    )
    args.update(change)
    with pytest.raises(ValueError, match="unknown_acquisition_explicit_scope"):
        renewal.prepare(**args)


def test_new_version_retains_original_exact_cumulative_budget():
    plan = dict(
        valid_plan(),
        schema_version=owner.VERSION,
        review_context_scope=renewal.RETAINED_CONTEXT_SCOPE,
    )
    renewal.validate_exact_acquisition_budget(plan, configuration())
    plan["previous_attempts"] = [dict(requests=38)]
    with pytest.raises(ValueError, match="renewal_exact_request_budget_binding"):
        renewal.validate_exact_acquisition_budget(plan, configuration())


@pytest.mark.parametrize("ids", [[], ["one", "one"], ["a", "b", "c", "d"]])
def test_selection_is_small_and_explicit_before_owner_access(ids):
    with pytest.raises(ValueError, match="unknown_acquisition_bounded_request_ids"):
        owner.selection(None, [], ids, {})


def test_plan_rejects_unversioned_ack_and_transport_retry(monkeypatch):
    proof = dict(selected=["target"], attempts=[dict(request_id="old")])
    parent = SimpleNamespace(config=configuration(), accepted_observations=lambda: {})
    parent.config.transport.base_url = "https://example.invalid/v1"
    monkeypatch.setattr(owner, "selection", lambda *_: proof)
    plan = dict(
        schema_version=owner.VERSION,
        unknown_acquisition=proof,
        mode="acquisition",
        seconds=2700,
        new_public_search_allowed=False,
        acquisition_targets=["target"],
        previous_attempts=[],
        started_at="2026-01-01T00:00:00+00:00",
        deadline="2026-01-01T00:45:00+00:00",
    )
    assert owner.check(parent, plan) == ["target"]
    with pytest.raises(ValueError, match="unknown_acquisition_plan_binding"):
        owner.check(parent, dict(plan, schema_version=renewal.RETAINED_ACQUISITION_VERSION))
    parent.config.transport.max_retries = 1
    with pytest.raises(ValueError, match="unknown_acquisition_plan_binding"):
        owner.check(parent, plan)


def test_new_version_preserves_retained_report_and_readonly_progress(tmp_path):
    import json

    from packages.evaluation_preparation import cohort_progress
    from packages.multimodal_research.report import attach_retained_context
    from tests.evaluation_preparation.test_cohort_progress import retained_fixture
    from tests.evaluation_preparation.test_retained_context_report import fixture

    out, _, _, protocol = retained_fixture(tmp_path)
    protocol["schema_version"] = owner.VERSION
    (out / "protocol.json").write_text(json.dumps(protocol))
    progress = cohort_progress.multimodal(out)
    assert progress["saved_accepted_visual_phases"] == 1
    assert not progress["authorization"] and not progress["automatic_replay"]
    c, report, state, _ = fixture(tmp_path)
    c.protocol["schema_version"] = owner.VERSION
    attach_retained_context(c, report, state)
    assert report["retained_review_context"]["extra_scoring_votes"] == 0


def test_new_version_declares_original_runtime_lease():
    from packages.multimodal_research.execution_lease import PROFILE, declare_selected

    plan = dict(schema_version=owner.VERSION, mode="acquisition")
    declare_selected(plan)
    assert plan["execution_lease_profile"] == PROFILE


def test_original_route_preflight_rejects_environment_drift_without_network(monkeypatch):
    from packages.llm_provider.network_route import prepare_route

    route = prepare_route(environ={"HTTPS_PROXY": "http://127.0.0.1:7890"})
    config = SimpleNamespace(
        network_route=route, transport=SimpleNamespace(base_url="https://example.invalid/v1")
    )
    for key in ("NO_PROXY", "no_proxy"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:7891")
    with pytest.raises(ValueError, match="network_proxy_endpoint_changed"):
        owner.check_network(config)
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:7890")
    owner.check_network(config)


def test_local_failure_only_one_specific_version_can_be_predecessor():
    with pytest.raises(ValueError, match="unknown_local_configuration_scope"):
        owner.local_configuration_tip(None, "absent", dict(schema_version=owner.LOCAL_VERSION), {})


def test_local_version_keeps_lease_budget_and_progress_dispatch_boundaries():
    from packages.multimodal_research.execution_lease import PROFILE, declare_selected

    plan = dict(
        valid_plan(),
        schema_version=owner.LOCAL_VERSION,
        review_context_scope=renewal.RETAINED_CONTEXT_SCOPE,
    )
    renewal.validate_exact_acquisition_budget(plan, configuration())
    declare_selected(plan)
    assert plan["execution_lease_profile"] == PROFILE
    plan["previous_attempts"] = [dict(requests=38)]
    with pytest.raises(ValueError, match="renewal_exact_request_budget_binding"):
        renewal.validate_exact_acquisition_budget(plan, configuration())


def test_prepare_distinct_local_predecessor_requires_original_owner_proof(monkeypatch):
    parent = SimpleNamespace(protocol={})
    monkeypatch.setattr(renewal, "completed_parent", lambda *_: (parent, dict(complete=True)))
    args = dict(
        facts="accepted-origin",
        previous="failed-local-predecessor",
        output="not-created",
        mode="acquisition",
        authorization="synthetic bounded recovery",
        acquisition_target=["target"],
        exact_request_budget=True,
    )
    with pytest.raises(ValueError, match="renewal_acquisition_requires_completed_fact_origin"):
        renewal.prepare(**args)
    args["acknowledge_unknown_acquisition_request"] = ["request:old"]

    class OwnerReached(Exception):
        pass

    def original_owner(*_, **__):
        raise OwnerReached()

    monkeypatch.setattr(renewal, "stopped_tip", original_owner)
    with pytest.raises(OwnerReached):
        renewal.prepare(**args)
    monkeypatch.setattr(renewal, "stopped_tip", lambda *_, **__: {})
    with pytest.raises(ValueError, match="unknown_local_configuration_predecessor_required"):
        renewal.prepare(**args)


def test_fact_review_origin_is_not_the_failed_configuration_predecessor(tmp_path):
    import json

    from packages.grounding.models import digest

    phase = dict(run_id="accepted", target=None, accepted=True)
    result = dict(complete=True, phases=[phase])
    (tmp_path / "result.json").write_text(json.dumps(result))
    parent = SimpleNamespace(
        documents=lambda rid: SimpleNamespace(
            read=lambda key: (
                (phase,) if rid == "accepted" and key == "dynamic-phase-result" else None
            )
        )
    )
    plan = dict(
        schema_version=owner.LOCAL_VERSION,
        facts=str(tmp_path),
        original_fact_result=digest(result),
        accepted_fact_run_id="accepted",
        tip=dict(run_id="failed"),
    )
    assert owner.fact_review_run(parent, plan) == "accepted"
    with pytest.raises(ValueError, match="unknown_accepted_fact_origin_changed"):
        owner.fact_review_run(parent, dict(plan, accepted_fact_run_id="failed"))
    with pytest.raises(ValueError, match="unknown_fact_origin_version"):
        owner.fact_review_run(parent, dict(plan, schema_version=owner.VERSION))


def test_renewal_run_route_mismatch_never_reaches_executor(monkeypatch):
    from packages.llm_provider.network_route import prepare_route

    calls = []
    config = SimpleNamespace(
        network_route=prepare_route(environ={"HTTPS_PROXY": "http://127.0.0.1:7890"}),
        transport=SimpleNamespace(base_url="https://example.invalid/v1"),
    )
    campaign = SimpleNamespace(config=config, execute=lambda: calls.append("execute"))
    monkeypatch.setattr(renewal, "assemble", lambda _: campaign)
    for key in ("NO_PROXY", "no_proxy"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:7897")
    with pytest.raises(ValueError, match="network_proxy_endpoint_changed"):
        renewal.run("existing-prepared-successor")
    assert calls == []
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:7890")
    renewal.run("existing-prepared-successor")
    assert calls == ["execute"]


def test_configuration_successor_is_explicit_bounded_and_report_readable(tmp_path):
    import json

    from packages.evaluation_preparation import cohort_progress
    from packages.multimodal_research.execution_lease import PROFILE, declare_selected
    from packages.multimodal_research.report import attach_retained_context
    from tests.evaluation_preparation.test_cohort_progress import retained_fixture
    from tests.evaluation_preparation.test_retained_context_report import fixture

    plan = dict(
        valid_plan(),
        schema_version=owner.CONFIGURATION_VERSION,
        review_context_scope=renewal.RETAINED_CONTEXT_SCOPE,
    )
    renewal.validate_exact_acquisition_budget(plan, configuration())
    declare_selected(plan)
    assert plan["execution_lease_profile"] == PROFILE
    plan["previous_attempts"] = [dict(requests=40)]
    with pytest.raises(ValueError, match="renewal_exact_request_budget_binding"):
        renewal.validate_exact_acquisition_budget(plan, configuration())
    out, _, _, protocol = retained_fixture(tmp_path)
    protocol["schema_version"] = owner.CONFIGURATION_VERSION
    (out / "protocol.json").write_text(json.dumps(protocol))
    assert not cohort_progress.multimodal(out)["authorization"]
    c, report, state, _ = fixture(tmp_path)
    c.protocol["schema_version"] = owner.CONFIGURATION_VERSION
    attach_retained_context(c, report, state)
    assert report["retained_review_context"]["extra_scoring_votes"] == 0


@pytest.mark.parametrize(
    "version", [owner.VERSION, owner.LOCAL_VERSION, owner.CONFIGURATION_VERSION]
)
def test_ordinary_configuration_recovery_cannot_recur(version):
    with pytest.raises(ValueError, match="unknown_local_configuration_scope"):
        owner.local_configuration_tip(
            None, "absent", dict(schema_version=version), {}, ordinary=True
        )


def test_local_request_ack_cannot_mix_unknown_requests():
    with pytest.raises(ValueError, match="configuration_attempt_single_request_required"):
        owner.selection(None, [], ["local", "unknown"], {}, local_request_ids=["local"])


def test_configuration_fact_origin_can_use_exact_confirmed_local_closure(tmp_path, monkeypatch):
    import json

    from packages.grounding.models import digest

    result = dict(
        complete=False, phases=[dict(run_id="original-review", target=None, accepted=False)]
    )
    (tmp_path / "result.json").write_text(json.dumps(result))
    proof = dict(schema_version="renewal-recorded-origin/1.1.0", checksum="exact-owner-proof")
    plan = dict(
        schema_version=owner.CONFIGURATION_VERSION,
        facts=str(tmp_path),
        original_fact_result=digest(result),
        accepted_fact_run_id="original-review",
        recorded_review_origin=proof,
    )
    monkeypatch.setattr(renewal, "recorded_acquisition_origin", lambda parent, rid: proof)
    assert owner.fact_review_run(None, plan) == "original-review"
    with pytest.raises(ValueError, match="unknown_accepted_fact_origin_changed"):
        owner.fact_review_run(
            None, dict(plan, recorded_review_origin=dict(proof, checksum="wrong"))
        )
