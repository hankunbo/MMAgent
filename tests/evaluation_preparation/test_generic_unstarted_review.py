"""Fair unaccepted Review: original-owner unsent partition, never UNKNOWN replay."""

import json
from types import SimpleNamespace

import pytest

from packages.evaluation_preparation import generic_pending_review as p
from packages.evaluation_preparation import generic_review as r
from packages.evaluation_preparation.generic_review import review_configuration
from packages.grounding.models import digest
from tests.evaluation_preparation.test_generic_current_pending import bound_config
from tests.expert_agent.test_pending_failed_attempt import failed

PROFILE = "bound-unstarted-batches-v1"


def test_unstarted_profile_preserves_bound_provider_exactly():
    origin = bound_config()
    assert review_configuration(origin, PROFILE) == origin


@pytest.mark.parametrize("ack", ["unknown", "truncated"])
def test_unstarted_never_authorizes_claimed_request_before_loading(ack, monkeypatch):
    monkeypatch.setattr(r, "load", lambda _: pytest.fail("must reject before loading"))
    with pytest.raises(ValueError, match="no_reassessment_authorization"):
        r.run_review(
            "parent",
            "child",
            review_profile=PROFILE,
            **{f"acknowledge_{ack}_request": "original-request"},
        )


def campaign_fixture(monkeypatch, tmp_path):
    f, state = failed(monkeypatch, tmp_path)
    old = dict(
        complete=False,
        phases=[dict(target=None, accepted=False, acceptance_key=None, run_id=f.binding.run_id)],
    )
    (tmp_path / "result.json").write_text(json.dumps(old))
    f.documents.write("dynamic-case-result", (old,))
    f.documents.write("dynamic-phase-input", (state,))

    class Projection:
        def __init__(self, *args, **kwargs):
            pass

        def project(self, **kwargs):
            assert kwargs == dict(run_id=f.binding.run_id, config=f.config)
            return state, None

        def verify(self, current):
            f.evidence.verify(current)

    monkeypatch.setattr(p, "GenericPhase", Projection)
    campaign = SimpleNamespace(
        output=tmp_path,
        documents=lambda _: f.documents,
        base=None,
        config=f.config,
        accepted_observations=lambda: {},
        read_model=lambda: None,
    )
    return f, state, campaign


def test_real_failed_owner_partition_repeat_without_claim_or_audit(monkeypatch, tmp_path):
    f, _, campaign = campaign_fixture(monkeypatch, tmp_path)
    before = digest(f.documents.saved), digest(f.ports.budget.journal.read())
    proof = p.current_selection(campaign, None, profile=PROFILE)
    assert proof["inventory"]["accepted_ids"] == []
    assert proof["selected_ids"]
    assert not set(proof["selected_ids"]) & {
        i for b in proof["inventory"]["quarantined_batches"] for i in b["assessment_ids"]
    }
    assert p.current_selection(campaign, None, profile=PROFILE) == proof
    assert before == (digest(f.documents.saved), digest(f.ports.budget.journal.read()))


@pytest.mark.parametrize("fault", ["accepted", "key", "input", "claim", "cancel"])
def test_changed_original_binding_or_consumed_state_rejects(monkeypatch, tmp_path, fault):
    f, state, campaign = campaign_fixture(monkeypatch, tmp_path)
    if fault in {"accepted", "key"}:
        old = json.loads((tmp_path / "result.json").read_bytes())
        old["phases"][0]["accepted" if fault == "accepted" else "acceptance_key"] = (
            True if fault == "accepted" else "wrong"
        )
        (tmp_path / "result.json").write_text(json.dumps(old))
    elif fault == "input":
        f.documents.saved["dynamic-phase-input"] = (dict(changed=True),)
    elif fault == "claim":
        key = next(
            k for k in f.documents.saved if k.startswith("review-batches:") and k.endswith(":claim")
        )
        f.documents.saved[key][0]["input_checksum"] = digest("wrong")
    else:
        control = f.planner.read(f.binding.run_id)
        monkeypatch.setattr(
            f.documents.sessions, "read", lambda _: control.model_copy(update={"cancelled": True})
        )
    with pytest.raises(ValueError, match="generic_pending_|pending_attempt_"):
        p.current_selection(campaign, None, profile=PROFILE)


@pytest.mark.parametrize("reserved,expected", [(7, 10), (36, 4), (37, None)])
def test_actual_batch_count_preserves_cumulative_budget(reserved, expected, tmp_path, monkeypatch):
    from packages.evaluation_preparation.generic_campaign import GenericCampaign

    origin = bound_config()

    def setup(self, spec, registered):
        self.config, self.provider_path = origin, "original-bound-provider"

    monkeypatch.setattr(GenericCampaign, "__init__", setup)
    monkeypatch.setattr(r, "directory", lambda key: tmp_path / key)
    monkeypatch.setattr(r.GenericReview, "prior", lambda _: dict(reservations=reserved))
    proof = dict(selected_ids=[str(i) for i in range(12)])
    calls = []

    def current(*args, **kwargs):
        calls.append(kwargs)
        return dict(proof)

    monkeypatch.setattr(p, "current_selection", current)

    def construct():
        return r.GenericReview(
            dict(schema_version="fairvlmed-run-input/1.8.0"), None, "new", review_profile=PROFILE
        )

    if expected is None:
        with pytest.raises(ValueError, match="selection_exceeds_request_budget"):
            construct()
    else:
        c = construct()
        assert c.config == origin and c.acquisition.config == origin
        assert (c.max_new_requests, c.max_vision_requests, c.duration_seconds) == (
            expected,
            0,
            2700,
        )
        c.before_review()
        assert calls == [dict(profile=PROFILE), dict(profile=PROFILE)]
        proof["selected_ids"] = ["changed"]
        with pytest.raises(ValueError, match="changed_before_dispatch"):
            c.before_review()


def test_cli_forwards_unstarted_profile_without_ack(monkeypatch):
    calls = []
    monkeypatch.setattr(r, "run_review", lambda **kwargs: calls.append(kwargs) or {})
    assert r.main(["--key", "parent", "--review-key", "child", "--review-profile", PROFILE]) == 0
    assert calls[0]["review_profile"] == PROFILE
    assert not any("acknowledge" in k for k in calls[0])


@pytest.mark.parametrize("preflight_error", [False, True])
def test_entry_claims_exact_selection_only_after_owner_preflight(
    tmp_path, monkeypatch, preflight_error
):
    from packages.evaluation_preparation import generic_control, generic_run

    origin = bound_config()
    spec = dict(key="parent")
    parent = tmp_path / "parent"
    child = tmp_path / "child"
    parent.mkdir()
    (parent / "binding.json").write_text("{}")
    calls = []
    proof = dict(selected_ids=["unsent"], inventory=dict(accepted_ids=[]))

    def construct(*args, **kwargs):
        calls.append("preflight")
        if preflight_error:
            raise ValueError("original_source_changed")
        assert kwargs["review_profile"] == PROFILE
        return SimpleNamespace(execute=lambda: calls.append("execute") or dict(complete=True))

    def prepared(*args, **kwargs):
        result = construct(*args, **kwargs)
        result.pending_review_proof = proof
        return result

    def persist(key, campaign, result):
        (child / "export-reference.json").write_text("{}")
        return dict(result=result)

    monkeypatch.setattr(r, "directory", lambda key: tmp_path / key)
    monkeypatch.setattr(r, "load", lambda _: spec)
    monkeypatch.setattr(r, "provider_config", lambda _: origin)
    monkeypatch.setattr(r, "register", lambda _: None)
    monkeypatch.setattr(r, "GenericReview", prepared)
    monkeypatch.setattr(generic_control, "start", lambda _: calls.append("start"))
    monkeypatch.setattr(generic_control, "guard", lambda _: calls.append("guard"))
    monkeypatch.setattr(generic_run, "persist_export", persist)
    monkeypatch.setattr(generic_run, "export", lambda _: dict(saved=True))
    if preflight_error:
        with pytest.raises(ValueError, match="original_source_changed"):
            r.run_review("parent", "child", review_profile=PROFILE)
        assert not (parent / "review-successor.json").exists()
        assert not (child / "input.json").exists()
        assert calls == ["preflight"]
        return
    assert r.run_review("parent", "child", review_profile=PROFILE)["result"]["complete"]
    selection = json.loads((parent / "review-successor.json").read_bytes())
    assert selection["schema_version"] == "generic-review-selection/1.4.0"
    assert selection["origin_provider_checksum"] == selection["provider_checksum"] == digest(origin)
    assert selection["eligibility_checksum"] == digest(proof)
    assert selection["selected_ids"] == ["unsent"]
    assert (selection["max_new_requests"], selection["max_cumulative_requests"]) == (10, 40)
    assert r.run_review("parent", "child", review_profile=PROFILE)["status"] == "already_recorded"
    assert calls == ["preflight", "start", "guard", "execute", "preflight"]
    with pytest.raises(ValueError):
        r.run_review("parent", "different-child", review_profile=PROFILE)
    assert calls.count("execute") == 1


@pytest.mark.parametrize("fault", [None, "origin", "derivation", "profile", "eligibility", "ack"])
def test_v14_cold_restore_revalidates_original_selection(tmp_path, monkeypatch, fault):
    from packages.evaluation_preparation import case_execution, generic_input

    config = bound_config()
    spec = dict(key="parent", schema_version="fairvlmed-run-input/1.8.0")
    proof = dict(selected_ids=["unsent"], inventory=dict(accepted_ids=[]))
    selection = dict(
        schema_version="generic-review-selection/1.4.0",
        parent_key="parent",
        review_key="child",
        review_profile=PROFILE,
        input_checksum=digest(spec),
        provider_derivation="bound-provider-unchanged/1.0.0",
        origin_provider_checksum=digest(config),
        provider_checksum=digest(config),
        selected_ids=["unsent"],
        eligibility_checksum=digest(proof),
    )
    if fault:
        field = dict(
            origin="origin_provider_checksum",
            derivation="provider_derivation",
            profile="review_profile",
            eligibility="eligibility_checksum",
            ack="acknowledge_unknown_request",
        )[fault]
        selection[field] = "unexpected"
    parent, child = tmp_path / "parent", tmp_path / "child"
    parent.mkdir()
    child.mkdir()
    (parent / "review-successor.json").write_text(json.dumps(selection))
    (child / "input.json").write_text(json.dumps(dict(selection, key="child")))
    for path in (parent, child):
        (path / "binding.json").write_text("{}")
    output = child / "evidence"
    output.mkdir()
    result = dict(phases=[dict(run_id="RUN:TEST", accepted=True)])
    protocol = dict(provider_checksum=digest(config))
    (output / "result.json").write_text(json.dumps(result))
    (output / "protocol.json").write_text(json.dumps(protocol))
    saved = {
        "dynamic-case-result": (result,),
        "dynamic-case-report": (dict(result=result),),
        "dynamic-campaign": (protocol,),
    }
    campaign = SimpleNamespace(
        config=config,
        pending_review_proof=proof,
        output=output,
        documents=lambda _: SimpleNamespace(read=lambda key: saved[key]),
        paths=lambda: [],
    )
    calls = []

    def constructor(*args, **kwargs):
        calls.append(kwargs)
        return campaign

    monkeypatch.setattr(generic_input, "directory", lambda key: tmp_path / key)
    monkeypatch.setattr(generic_input, "load", lambda _: spec)
    monkeypatch.setattr(generic_input, "provider_config", lambda _: config)
    monkeypatch.setattr(case_execution, "owners", lambda: (None, None, None))
    monkeypatch.setattr(r, "GenericReview", constructor)
    if fault:
        with pytest.raises(ValueError, match="generic_aggregation_"):
            p.restore_campaign(child)
    else:
        assert p.restore_campaign(child) == (campaign, result)
        assert calls == [dict(review_profile=PROFILE)]
