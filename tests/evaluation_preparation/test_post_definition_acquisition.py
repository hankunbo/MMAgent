import json
from types import SimpleNamespace

import pytest

from packages.grounding.models import digest
from packages.multimodal_research import campaign_renewal as renewal
from packages.multimodal_research import unknown_acquisition as owner
from tests.evaluation_preparation.test_definition_terminal_backlog import saved


def fixture(tmp_path, monkeypatch):
    latest = tmp_path / "latest"
    latest.mkdir()
    facts = tmp_path / "facts"
    facts.mkdir()
    s = saved(latest, monkeypatch)
    # These unit cases exercise lineage/Control. Exact raw Source validation is
    # reused from the original owner and checked by the actual read-only probe.
    monkeypatch.setattr(
        "packages.multimodal_research.definition_alternative.confirmed_sources",
        lambda parent, docs, events, protocol: digest(events),
    )
    old_phase = dict(run_id="fact", target=None, accepted=True, acceptance_key="old")
    old = dict(complete=True, phases=[old_phase])
    (facts / "result.json").write_text(json.dumps(old))
    s.protocol["facts"] = str(facts)
    prior = [dict(directory=str(facts), requests=23, checksum=digest(old))]
    s.protocol["previous_attempts"] = s.result["previous_attempts"] = prior
    s.fact_records = {"dynamic-phase-result": (old_phase,), "multimodal-case-result": (old,)}
    s.fact_records["authorized-campaign-renewal"] = (
        dict(output=str(latest), plan_checksum=digest(s.protocol)),
    )
    original_documents = s.parent.documents
    fact_docs = SimpleNamespace(run_id="fact", read=lambda key: s.fact_records.get(key))
    s.parent.documents = lambda rid: fact_docs if rid == "fact" else original_documents(rid)
    s.parent.sessions.read = lambda rid: SimpleNamespace(
        phase="stopped", pending=None, cancelled=False
    )
    s.facts = facts
    (latest / "protocol.json").write_text(json.dumps(s.protocol))
    (latest / "result.json").write_text(json.dumps(s.result))
    return s


def test_zero_increment_tip_keeps_old_facts_exact_history_and_budget(tmp_path, monkeypatch):
    s = fixture(tmp_path, monkeypatch)
    tip, proof = owner.post_definition_tip(s.parent, s.path, s.facts)
    assert sum(a["requests"] for a in tip["previous_attempts"]) == 25
    assert proof["accepted_fact_run_id"] == "fact"
    assert proof["no_effective_definition_increment"]
    assert not proof["source_replay_allowed"]
    assert owner.post_definition_tip(s.parent, s.path, s.facts) == (tip, proof)
    assert s.fact_records["dynamic-phase-result"][0]["accepted"] is True


@pytest.mark.parametrize(
    "fault",
    [
        "changed_claim",
        "missing_claim",
        "old_result",
        "missing_revision",
        "added",
        "replaced",
        "current_cancelled",
        "pending",
        "marker",
        "unknown_source",
        "wrong_facts",
        "nonterminal",
    ],
)
def test_changed_owner_or_history_does_not_authorize_successor(tmp_path, monkeypatch, fault):
    s = fixture(tmp_path, monkeypatch)
    if fault == "changed_claim":
        s.fact_records["authorized-campaign-renewal"] = (
            {"output": "other", "plan_checksum": "other"},
        )
    elif fault == "missing_claim":
        s.fact_records.pop("authorized-campaign-renewal")
    elif fault == "old_result":
        s.fact_records["multimodal-case-result"] = ({"complete": False},)
    elif fault == "missing_revision":
        s.records.pop("definition:revision")
    elif fault in ["added", "replaced"]:
        s.revision[fault] = ["new"]
    elif fault == "current_cancelled":
        s.parent.sessions.read = lambda _: SimpleNamespace(
            phase="stopped", pending=None, cancelled=True
        )
    elif fault == "pending":
        s.parent.sessions.read = lambda _: SimpleNamespace(
            phase="stopped", pending="unknown", cancelled=False
        )
    elif fault == "marker":
        (s.path / "cancel").touch()
    elif fault == "unknown_source":
        s.events.pop()
    elif fault == "wrong_facts":
        s.protocol["facts"] = "other"
        (s.path / "protocol.json").write_text(json.dumps(s.protocol))
    else:
        s.result["complete"] = False
        (s.path / "result.json").write_text(json.dumps(s.result))
    with pytest.raises(ValueError):
        owner.post_definition_tip(s.parent, s.path, s.facts)


def test_new_budget_and_reader_version():
    from tests.evaluation_preparation.test_exact_acquisition_budget import configuration, valid_plan

    plan = dict(
        valid_plan(),
        schema_version=owner.POST_DEFINITION_VERSION,
        review_context_scope=renewal.RETAINED_CONTEXT_SCOPE,
    )
    renewal.validate_exact_acquisition_budget(plan, configuration())
    plan["previous_attempts"] = [dict(requests=39)]
    with pytest.raises(ValueError):
        renewal.validate_exact_acquisition_budget(plan, configuration())


def test_unversioned_extra_proof_is_rejected():
    with pytest.raises(ValueError, match="post_definition_acquisition_explicit_version"):
        owner.check(
            None,
            dict(schema_version=owner.VERSION, post_definition_acquisition={"untrusted": True}),
        )


def test_new_version_is_readable_in_report_and_progress(tmp_path):
    from packages.evaluation_preparation import cohort_progress
    from packages.multimodal_research.report import attach_retained_context
    from tests.evaluation_preparation.test_cohort_progress import retained_fixture
    from tests.evaluation_preparation.test_retained_context_report import fixture as report_fixture

    out, _, _, protocol = retained_fixture(tmp_path)
    protocol["schema_version"] = owner.POST_DEFINITION_VERSION
    (out / "protocol.json").write_text(json.dumps(protocol))
    progress = cohort_progress.multimodal(out)
    assert progress["saved_accepted_visual_phases"] == 1
    assert not progress["automatic_replay"]
    campaign, report, state, _ = report_fixture(tmp_path)
    campaign.protocol["schema_version"] = owner.POST_DEFINITION_VERSION
    attach_retained_context(campaign, report, state)
    assert report["retained_review_context"]["extra_scoring_votes"] == 0
