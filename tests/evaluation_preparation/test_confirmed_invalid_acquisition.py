"""Confirmed invalid replies remain failures; one explicit attempt has new identity."""

from types import SimpleNamespace

import pytest

from packages.grounding.models import digest
from packages.multimodal_research import campaign_renewal as renewal
from packages.multimodal_research import unknown_acquisition as owner
from packages.object_store import checksum
from tests.evaluation_preparation.test_exact_acquisition_budget import configuration, valid_plan


def response_fixture():
    text = "Image not available; no observation."
    response = dict(
        schema_version="bounded-live-response/1.0.0",
        request_id="request:old",
        raw_model_text=text,
        raw_checksum=checksum(text.encode()),
        accepted_medical_evidence=False,
        untrusted_model_output=True,
        validation_error="schema",
    )
    record = SimpleNamespace(
        request_id="request:old", error="schema", output_checksum=digest("raw reply")
    )
    confirmations = [dict(request_id="request:old", confirmed=True)]
    return response, record, confirmations


def test_saved_confirmed_failure_exact_binding():
    response, record, confirmations = response_fixture()
    docs = SimpleNamespace(
        read=lambda key: (response,) if key == "live-response:request:old" else None
    )
    assert owner.confirmed_invalid_response(docs, record, confirmations) == dict(
        response_checksum=digest(response), output_checksum=record.output_checksum
    )
    confirmations[0]["confirmed"] = None
    with pytest.raises(ValueError, match="confirmed_invalid_acquisition_response_required"):
        owner.confirmed_invalid_response(docs, record, confirmations)


@pytest.mark.parametrize(
    "changed",
    [
        dict(request_id="request:another"),
        dict(validation_error=None),
        dict(raw_model_text="changed"),
        dict(accepted_medical_evidence=True),
        dict(untrusted_model_output=False),
        dict(schema_version="future"),
        dict(raw_model_text=""),
    ],
)
def test_saved_response_cannot_be_used_after_changed_binding(changed):
    response, record, confirmations = response_fixture()
    response.update(changed)
    with pytest.raises(ValueError, match="confirmed_invalid_acquisition_response_binding"):
        owner.confirmed_invalid_response(
            SimpleNamespace(read=lambda _: (response,)), record, confirmations
        )


@pytest.mark.parametrize(
    "change",
    [
        dict(mode="review"),
        dict(exact_request_budget=False),
        dict(acquisition_target=[]),
        dict(previous="different"),
        dict(request_limit=3),
        dict(recovery_http_client="changed"),
        dict(acknowledge_unknown_acquisition_request=["old"]),
        dict(alternative_view_method="changed"),
        dict(pending_review=True),
        dict(acknowledge_confirmed_invalid_acquisition_request=["a", "b"]),
    ],
)
def test_cli_scope_rejects_mixed_modes_before_owner_or_write(change):
    args = dict(
        facts="missing",
        previous="missing",
        output="never-written",
        mode="acquisition",
        authorization="test",
        exact_request_budget=True,
        acquisition_target=["target"],
        acknowledge_confirmed_invalid_acquisition_request=["request:old"],
        request_limit=2,
    )
    args.update(change)
    with pytest.raises(ValueError, match="confirmed_invalid_acquisition_explicit_scope"):
        renewal.prepare(**args)


def test_new_version_budget_lease_report_and_progress(tmp_path):
    import json

    from packages.evaluation_preparation import cohort_progress
    from packages.multimodal_research.execution_lease import PROFILE, declare_selected
    from packages.multimodal_research.report import attach_retained_context
    from tests.evaluation_preparation.test_cohort_progress import retained_fixture
    from tests.evaluation_preparation.test_retained_context_report import fixture

    plan = dict(
        valid_plan(),
        schema_version=owner.CONFIRMED_INVALID_VERSION,
        review_context_scope=renewal.RETAINED_CONTEXT_SCOPE,
    )
    renewal.validate_exact_acquisition_budget(plan, configuration())
    declare_selected(plan)
    assert plan["execution_lease_profile"] == PROFILE
    plan["previous_attempts"] = [dict(requests=40)]
    with pytest.raises(ValueError, match="renewal_exact_request_budget_binding"):
        renewal.validate_exact_acquisition_budget(plan, configuration())
    out, _, _, protocol = retained_fixture(tmp_path)
    protocol["schema_version"] = owner.CONFIRMED_INVALID_VERSION
    (out / "protocol.json").write_text(json.dumps(protocol))
    assert cohort_progress.multimodal(out)["saved_accepted_visual_phases"] == 1
    c, report, state, _ = fixture(tmp_path)
    c.protocol["schema_version"] = owner.CONFIRMED_INVALID_VERSION
    attach_retained_context(c, report, state)
    assert report["retained_review_context"]["extra_scoring_votes"] == 0


def test_confirmed_proof_cannot_be_relabelled_unknown_or_expand_budget(monkeypatch):
    proof = dict(selected=["target"], attempts=[dict(request_id="old")])
    parent = SimpleNamespace(config=configuration(), accepted_observations=lambda: {})
    calls = []

    def select(*args, **kw):
        calls.append(kw)
        return proof

    monkeypatch.setattr(owner, "selection", select)
    plan = dict(
        schema_version=owner.CONFIRMED_INVALID_VERSION,
        confirmed_invalid_acquisition=proof,
        mode="acquisition",
        seconds=2700,
        new_public_search_allowed=False,
        acquisition_targets=["target"],
        previous_attempts=[],
        max_new_requests=2,
        max_model_dispatches=2,
        started_at="2026-01-01T00:00:00+00:00",
        deadline="2026-01-01T00:45:00+00:00",
    )
    assert owner.check(parent, plan) == ["target"] and calls == [dict(confirmed_invalid=True)]
    for change in [
        dict(schema_version=owner.VERSION),
        dict(unknown_acquisition=proof),
        dict(max_new_requests=3),
        dict(max_model_dispatches=3),
    ]:
        with pytest.raises(ValueError, match="unknown_acquisition_plan_binding"):
            owner.check(parent, dict(plan, **change))
