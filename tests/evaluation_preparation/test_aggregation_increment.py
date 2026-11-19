"""Only new observations are scheduled/reviewed; explicit full fold remains separate."""

import json
from types import SimpleNamespace

import pytest

from packages.grounding.models import digest
from packages.multimodal_research.accepted_aggregation import full_scope, read_plan
from packages.multimodal_research.campaign import review_observations
from packages.multimodal_research.campaign_renewal import check_local_history, unattempted_targets
from packages.multimodal_research.memory import archive


def test_full_projection_is_explicit_and_does_not_mutate_old_selector():
    parent = SimpleNamespace(review_target_ids=("new",), confirmed_review_reader=object())
    campaign = SimpleNamespace(base=parent)
    old = dict(
        schema_version="accepted-review-aggregation-plan/1.0.0",
        facts="facts",
        origins=[dict(campaign="previous")],
    )
    full_scope(campaign, read_plan(old))
    assert campaign.base is parent and review_observations(parent, {"old": 1, "new": 2}) == {
        "new": 2
    }
    new = {
        **old,
        "schema_version": "accepted-review-aggregation-plan/1.1.0",
        "scope": "all_current_obtained",
    }
    full_scope(campaign, read_plan(new))
    assert campaign.base is not parent and parent.review_target_ids == ("new",)
    assert review_observations(campaign.base, {"old": 1, "new": 2}) == {"old": 1, "new": 2}


def test_selection_never_reexecutes_old_unknown_or_confirmed_and_requires_input():
    parent = SimpleNamespace(
        initial_targets=("old", "unknown", "new1", "new2", "missing", "numeric"),
        base=SimpleNamespace(
            ledger=[
                dict(
                    id=i,
                    component=None if i == "missing" else "volume",
                    acquisition_method="source_measurement" if i == "numeric" else "vision",
                )
                for i in ("old", "unknown", "new1", "new2", "missing", "numeric")
            ]
        ),
    )
    assert unattempted_targets(parent, {"unknown"}, {"old": object()}) == ["new1", "new2"]


def test_aggregation_tip_uses_exact_manifest_not_invented_legacy_protocol(tmp_path):
    result, manifest = {"local_result": True}, {"original_owner": True}
    (tmp_path / "result.json").write_text(json.dumps(result))
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    plan = dict(
        tip=dict(
            path=str(tmp_path),
            result=digest(result),
            protocol=digest(manifest),
            protocol_kind="accepted_aggregation_manifest",
        )
    )
    check_local_history(plan)
    (tmp_path / "manifest.json").write_text(json.dumps({"original_owner": False}))
    with pytest.raises(ValueError, match="renewal_history_changed"):
        check_local_history(plan)


def test_acquisition_increment_cannot_archive_itself_as_complete_case():
    campaign = SimpleNamespace(protocol={"acquisition_increment_only": True})
    assert archive(campaign, {"complete": True}) == dict(
        status="not_archived_whole_case_acquisition_increment", eligible_for_planning=False
    )


def accounting_fixture(tmp_path):
    from packages.multimodal_research import accepted_aggregation as a

    root = dict(directory=str(tmp_path / "root"), requests=16, checksum=digest("root"))
    old = dict(
        new_requests=3, previous_attempts=[root], complete=False, phases=[dict(error="transport")]
    )
    plan = a.read_plan(
        dict(
            schema_version="accepted-review-aggregation-plan/1.3.0",
            facts=str(tmp_path / "failed"),
            origins=[dict(campaign=root["directory"])],
        )
    )
    campaign = SimpleNamespace(protocol=dict(previous_attempts=[root]))
    readers = (SimpleNamespace(accounting=[root]),)
    return a, plan, campaign, old, readers


def test_failed_facts_requests_cannot_disappear_when_only_old_reviews_are_reused(tmp_path):
    a, plan, campaign, old, readers = accounting_fixture(tmp_path)
    attempts = a.request_history(plan, campaign, old, readers)
    assert sum(row["requests"] for row in attempts) == 19
    assert len(attempts) == 2
    assert attempts[-1] == dict(directory=plan.facts, requests=3, checksum=digest(old))
    assert old["complete"] is False and old["phases"][0]["error"] == "transport"


def test_typed_source_failure_is_owner_checked_once_and_preserved(monkeypatch, tmp_path):
    from packages.multimodal_research import campaign_renewal

    a, plan, campaign, old, readers = accounting_fixture(tmp_path)
    original = dict(old["previous_attempts"][0], kind="routed_source_failure_deferred")
    old["previous_attempts"] = [original]
    readers[0].accounting = [dict(original)]
    checked = []

    def verify(owner, entries):
        assert owner is campaign
        checked.extend(entries)
        return set()

    monkeypatch.setattr(campaign_renewal, "attempted_acquisition_targets", verify)
    attempts = a.request_history(plan, campaign, old, readers)
    assert checked == [original]
    assert attempts[0] == original and len(attempts) == 2
    assert sum(row["requests"] for row in attempts) == 19
    a.verify_request_history(
        plan,
        campaign,
        old,
        readers,
        dict(request_history_checksum=digest(attempts)),
        dict(previous_attempts=attempts, cumulative_requests=19, new_requests=0),
    )
    assert old["previous_attempts"] == [original]


def test_alias_plan_retains_confirmed_definition_failure_via_original_owner(monkeypatch, tmp_path):
    from packages.multimodal_research import campaign_renewal
    from packages.multimodal_research.confirmed_definition_failure import HISTORY_KIND

    a, plan, campaign, old, readers = accounting_fixture(tmp_path)
    failure = dict(old["previous_attempts"][0], kind=HISTORY_KIND)
    old["previous_attempts"] = [failure]
    readers[0].accounting = [dict(failure)]
    with pytest.raises(ValueError, match="request_history_invalid"):
        a.request_history(plan, campaign, old, readers)
    plan = a.read_plan(
        dict(plan.model_dump(mode="json"), schema_version="accepted-review-aggregation-plan/1.6.0")
    )
    checked = []

    def verify(owner, attempts):
        assert owner is campaign
        checked.extend(attempts)
        return set()

    monkeypatch.setattr(campaign_renewal, "attempted_acquisition_targets", verify)
    attempts = a.request_history(plan, campaign, old, readers)
    assert checked == [failure] and attempts[0] == failure
    assert sum(x["requests"] for x in attempts) == 19

    def reject(*_):
        raise ValueError("confirmed_definition_failure_history_owner")

    monkeypatch.setattr(campaign_renewal, "attempted_acquisition_targets", reject)
    with pytest.raises(ValueError, match="confirmed_definition_failure_history_owner"):
        a.request_history(plan, campaign, old, readers)
    assert old["previous_attempts"] == [failure]


def test_source_failure_label_does_not_substitute_for_original_owner(monkeypatch, tmp_path):
    from packages.multimodal_research import campaign_renewal

    a, plan, campaign, old, readers = accounting_fixture(tmp_path)
    old["previous_attempts"][0]["kind"] = "routed_source_failure_deferred"

    def reject(*_):
        raise ValueError("renewal_routed_source_history_owner")

    monkeypatch.setattr(campaign_renewal, "attempted_acquisition_targets", reject)
    with pytest.raises(ValueError, match="renewal_routed_source_history_owner"):
        a.request_history(plan, campaign, old, readers)


@pytest.mark.parametrize("kind", [None, "unknown", "caller_validated"])
def test_unrecognized_typed_history_is_not_accepted(tmp_path, kind):
    a, plan, campaign, old, readers = accounting_fixture(tmp_path)
    old["previous_attempts"][0]["kind"] = kind
    with pytest.raises(ValueError, match="accepted_aggregation_request_history_invalid"):
        a.request_history(plan, campaign, old, readers)


@pytest.mark.parametrize("count", [-1, True, 1.5, "3", 25])
def test_full_history_rejects_invalid_or_over_budget_failed_requests(tmp_path, count):
    a, plan, campaign, old, readers = accounting_fixture(tmp_path)
    old["new_requests"] = count
    with pytest.raises(ValueError, match="accepted_aggregation_request_history"):
        a.request_history(plan, campaign, old, readers)


def test_full_history_cannot_merge_same_directory_different_receipt(tmp_path):
    a, plan, campaign, old, readers = accounting_fixture(tmp_path)
    readers[0].accounting[0] = dict(readers[0].accounting[0], checksum=digest("changed"))
    with pytest.raises(ValueError, match="request_history_conflict"):
        a.request_history(plan, campaign, old, readers)


def test_full_history_restore_rejects_omitted_failed_window(tmp_path):
    a, plan, campaign, old, readers = accounting_fixture(tmp_path)
    attempts = a.request_history(plan, campaign, old, readers)
    manifest = dict(request_history_checksum=digest(attempts))
    result = dict(previous_attempts=attempts, cumulative_requests=19, new_requests=0)
    a.verify_request_history(plan, campaign, old, readers, manifest, result)
    result.update(previous_attempts=attempts[:1], cumulative_requests=16)
    with pytest.raises(ValueError, match="request_history_binding"):
        a.verify_request_history(plan, campaign, old, readers, manifest, result)


def test_aggregation_as_facts_retains_its_verified_lineage_not_older_campaign_protocol(tmp_path):
    a, plan, campaign, old, readers = accounting_fixture(tmp_path)
    prior = a.request_history(plan, campaign, old, readers)
    new_plan = a.read_plan(dict(plan.model_dump(mode="json"), facts=str(tmp_path / "fold")))
    folded = dict(previous_attempts=prior, new_requests=0, cumulative_requests=19)
    attempts = a.request_history(new_plan, campaign, folded, readers)
    assert sum(row["requests"] for row in attempts) == 19
    assert attempts[:2] == prior and attempts[-1]["requests"] == 0
    assert len(attempts) == 3


def test_full_accounted_projection_clears_increment_only_context_without_mutating_parent(tmp_path):
    from packages.multimodal_research.review_context import retain_observations

    a, plan, campaign, _, _ = accounting_fixture(tmp_path)
    original = SimpleNamespace(
        review_target_ids=("new",),
        confirmed_review_reader=object(),
        review_context_scope="current-owner-prior-visible/1.0.0",
    )
    campaign.base = original
    a.full_scope(campaign, plan)
    assert campaign.base is not original
    assert original.confirmed_review_reader is not None
    assert original.review_context_scope == "current-owner-prior-visible/1.0.0"
    assert campaign.base.review_target_ids is None
    assert campaign.base.confirmed_review_reader is None
    assert retain_observations(campaign.base, {}, {}, ("current-owner-context",)) == (
        "current-owner-context",
    )


@pytest.mark.parametrize("version", ["1.0.0", "1.1.0", "1.2.0"])
def test_historical_aggregation_accounting_keeps_original_interpretation(tmp_path, version):
    a, plan, campaign, old, readers = accounting_fixture(tmp_path)
    raw = dict(
        schema_version="accepted-review-aggregation-plan/" + version,
        facts=plan.facts,
        origins=[dict(campaign=plan.origins[0].campaign)],
    )
    historical = a.read_plan(raw)
    assert a.request_history(historical, campaign, old, readers) == readers[0].accounting
