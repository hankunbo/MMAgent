"""Accounting boundary units; real-owner preflight is a separate evidence gate."""

import copy
import json
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from packages.evaluation_preparation import generic_definition_successor as module
from packages.evaluation_preparation.generic_campaign import GenericCampaign
from packages.grounding.models import digest


class Control(BaseModel):
    phase: str = "stopped"
    pending: str | None = None
    cancelled: bool = False


class Request(BaseModel):
    lock: dict


def fixture(monkeypatch, tmp_path):
    campaign = object.__new__(GenericCampaign)
    campaign.spec = dict(key="original")
    paths = [tmp_path / "vision", tmp_path / "review"]
    requests = {}
    for i, path in enumerate(paths):
        path.mkdir()
        value = dict(lock=dict(run_id=f"RUN:{i}"))
        (path / "request.json").write_text(json.dumps(value))
        requests[f"RUN:{i}"] = Request(**value)
    controls = {key: Control() for key in requests}
    # UNKNOWN is retained and charged; no request-execution method exists here.
    events = {
        "RUN:0": [dict(phase="reserved", request_id="unknown-vision")],
        "RUN:1": [dict(phase="reserved", request_id="review"), dict(phase="completed")],
    }
    result = dict(
        complete=True,
        new_requests=2,
        prior_graph_requests=0,
        phases=[dict(run_id="RUN:1", accepted=True, acceptance_key="accepted-review")],
    )
    report = dict(result=copy.deepcopy(result), scores={"immutable": True})
    saved = (report, dict(report_checksum=digest(report)))
    docs = {key: SimpleNamespace(run_id=key, read=lambda name: saved) for key in requests}
    campaign.paths = lambda: paths
    campaign.contexts = SimpleNamespace(
        read_research_run=lambda key: SimpleNamespace(run_id=key, request=requests[key])
    )
    campaign.sessions = SimpleNamespace(read=lambda key: controls[key])
    campaign.documents = lambda key: docs[key]
    monkeypatch.setattr(module, "directory", lambda _: tmp_path)
    monkeypatch.setattr(module, "confirmed_campaign", lambda c: (c, result))
    monkeypatch.setattr(
        module, "RuntimeAgentJournal", lambda d: SimpleNamespace(read=lambda: events[d.run_id])
    )
    return SimpleNamespace(
        campaign=campaign,
        paths=paths,
        requests=requests,
        controls=controls,
        events=events,
        result=result,
        root=tmp_path,
        saved=saved,
    )


def test_complete_origin_is_not_relabelled_and_unknown_is_charged(monkeypatch, tmp_path):
    f = fixture(monkeypatch, tmp_path)
    before = copy.deepcopy((f.result, f.events, f.saved))
    proof = module.completed_history(f.campaign)
    assert proof["reservations"] == 2
    assert proof["original_complete"] is True
    assert proof["original_scores_relabelled"] is False
    assert [r["reservations"] for r in proof["journals"]] == [1, 1]
    assert (f.result, f.events, f.saved) == before


@pytest.mark.parametrize(
    "change",
    [
        "missing_request",
        "changed_request",
        "extra_reservation",
        "old_prior",
        "duplicate_run",
        "duplicate_path",
    ],
)
def test_accounting_omissions_and_double_count_are_rejected(monkeypatch, tmp_path, change):
    f = fixture(monkeypatch, tmp_path)
    if change == "missing_request":
        (f.paths[0] / "request.json").unlink()
    elif change == "changed_request":
        f.requests["RUN:0"] = Request(lock=dict(run_id="RUN:OTHER"))
    elif change == "extra_reservation":
        f.events["RUN:0"].append(dict(phase="reserved", request_id="another"))
    elif change == "old_prior":
        f.result["prior_graph_requests"] = 1
    elif change == "duplicate_run":
        (f.paths[1] / "request.json").write_text((f.paths[0] / "request.json").read_text())
    else:
        f.paths.append(f.paths[0])
    with pytest.raises(ValueError, match="generic_definition_(history|duplicate)"):
        module.completed_history(f.campaign)


@pytest.mark.parametrize("change", ["pending", "cancelled", "running"])
def test_nonterminal_or_cancelled_old_control_cannot_be_used(monkeypatch, tmp_path, change):
    f = fixture(monkeypatch, tmp_path)
    if change == "pending":
        f.controls["RUN:0"].pending = "inflight"
    elif change == "cancelled":
        f.controls["RUN:0"].cancelled = True
    else:
        f.controls["RUN:0"].phase = "executing"
    with pytest.raises(ValueError, match="history_request_or_control"):
        module.completed_history(f.campaign)


def test_unknown_sibling_is_not_omitted_and_own_claim_is_exact(monkeypatch, tmp_path):
    f = fixture(monkeypatch, tmp_path)
    own = {"schema_version": module.VERSION, "key": "new"}
    (tmp_path / module.CLAIM).write_text(json.dumps(own))
    assert module.completed_history(f.campaign, own_claim=own)["reservations"] == 2
    with pytest.raises(ValueError, match="other_successor"):
        module.completed_history(f.campaign, own_claim=dict(own, key="different"))
    (tmp_path / "review-successor.json").write_text("{}")
    with pytest.raises(ValueError, match="other_successor"):
        module.completed_history(f.campaign, own_claim=own)


def test_late_claim_and_new_journal_change_frozen_history(monkeypatch, tmp_path):
    f = fixture(monkeypatch, tmp_path)
    checked = SimpleNamespace(history=module.completed_history(f.campaign))
    successor = object.__new__(module.DefinitionReview)
    successor.acquisition = f.campaign
    successor.checked = checked
    successor.selection = {"key": "successor"}
    assert successor.prior()["reservations"] == 2
    f.events["RUN:0"].append(dict(phase="confirmed"))
    with pytest.raises(ValueError, match="origin_history_changed"):
        successor.prior()


def test_definition_and_review_paths_share_original_budget(monkeypatch, tmp_path):
    from packages.evaluation_preparation.dynamic_campaign import Campaign

    f = fixture(monkeypatch, tmp_path)
    successor = object.__new__(module.DefinitionReview)
    successor.output = tmp_path
    (tmp_path / "definition").mkdir()
    (tmp_path / "definition/request.json").write_text((f.paths[0] / "request.json").read_text())
    successor.contexts, successor.documents = f.campaign.contexts, f.campaign.documents
    import packages.evaluation_preparation.dynamic_campaign as original

    monkeypatch.setattr(original, "RuntimeAgentJournal", module.RuntimeAgentJournal)
    assert Campaign.usage(successor) == 2
    assert successor.paths() == [tmp_path / "definition", tmp_path / "review"]


def test_definition_report_uses_saved_projection_and_keeps_old_score_scope(monkeypatch, tmp_path):
    from packages.evaluation_preparation.generic_definition_projection import (
        project_interpretations,
    )
    from tests.evaluation_preparation.test_generic_definition_stage import setup

    f = setup(monkeypatch, tmp_path)
    f.stage.advance(provider_factory=f.factory)
    keys = f.stage.review_keys()
    state, manifest = project_interpretations(f.state, f.stage.expansion.owner, keys)
    f.documents.write("generic-definition-review-projection", (manifest,))
    campaign = object.__new__(module.DefinitionReview)
    campaign.definition_keys = keys
    campaign.definition_stage = f.stage
    campaign.documents = lambda _: f.documents
    campaign.checked = SimpleNamespace(
        history=dict(result_checksum=digest("old-result"), report_checksum=digest("old-report"))
    )
    monkeypatch.setattr(
        f.stage.expansion,
        "current",
        lambda: pytest.fail("report cannot require execution authority"),
    )
    metadata = module.report_metadata(campaign, state, dict(new_requests=1))
    assert metadata["status"] == "new_interpretation_increment"
    assert metadata["whole_case_aggregation_required"]
    assert metadata["historical_scores_modified"] is False
    assert metadata["new_observations"] == 0 and metadata["new_requirements"] == []
    assert metadata["definition_projection"] == manifest
    campaign.definition_keys = ("another-key",)
    with pytest.raises(ValueError, match="report_projection_binding"):
        module.report_metadata(campaign, state, dict(new_requests=1))


def test_cli_preflight_is_distinct_from_run_or_pending_review(monkeypatch):
    import sys

    from apps.fairvlmed_case import main

    calls = []

    def plan(*args):
        calls.append(args)
        return dict(status="test_preflight", api_requests=0)

    monkeypatch.setattr(module, "preflight", plan)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fairvlmed_case",
            "definition-plan",
            "--key",
            "old",
            "--review-key",
            "new",
            "--definition-policy",
            "public.json",
        ],
    )
    assert main() == 0
    assert calls == [("old", "new", "public.json")]


def test_zero_adoption_keeps_zero_increment_without_fabricated_review(monkeypatch, tmp_path):
    from packages.evaluation_preparation.dynamic_stages import rebuild
    from packages.llm_provider.provider import DeterministicFakeProvider
    from tests.evaluation_preparation.test_generic_definition_stage import setup

    f = setup(monkeypatch, tmp_path)
    response = dict(
        schema_version="research-definition-proposal/6.0.0",
        definitions=[],
        no_addition_reason="No source-grounded interpretation fits this measured method",
    )
    f.stage.advance(provider_factory=lambda config: DeterministicFakeProvider(config, [response]))
    assert f.stage.review_keys() == ()
    campaign = object.__new__(module.DefinitionReview)
    campaign.definition_stage = f.stage
    campaign.checked = SimpleNamespace(
        history=dict(result_checksum=digest("old-result"), report_checksum=digest("old-report"))
    )
    campaign.documents = lambda _: f.documents
    campaign.bind_review_phase()
    state = rebuild(
        f.state, f.state.policy.model_copy(update=dict(items=())), f.state.material.observations
    )
    monkeypatch.setattr(
        f.stage.expansion,
        "current",
        lambda: pytest.fail("report cannot require execution authority"),
    )
    metadata = module.report_metadata(campaign, state, dict(new_requests=1))
    assert metadata["status"] == "no_adopted_definition"
    assert metadata["definition_projection"] is None
    assert metadata["interpretation_ids"] == []
    assert metadata["whole_case_aggregation_required"] is False
    assert f.stage.budget.used == 1
    assert not (tmp_path / "review/request.json").exists()
