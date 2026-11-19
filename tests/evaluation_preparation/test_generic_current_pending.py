"""Bound-provider explicit UNKNOWN reassessment; old requests never replayed."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from packages.evaluation_preparation import generic_pending_review as p
from packages.evaluation_preparation import generic_review as r
from packages.evidence_details.pending_review import read_pending_review
from packages.evidence_details.review_batches import collect_partial
from packages.grounding.models import digest, validate
from packages.llm_provider.config import read_provider_config
from packages.llm_provider.provider import ProviderError
from packages.llm_provider.transport import Reply
from packages.planner_control.models import Control
from tests.agent_provider.batch_support import batch_reply, generate
from tests.evaluation_preparation.test_dynamic_evidence import dynamic_fixture


def bound_config():
    raw = json.loads(Path("configs/agents/fairvlmed_relay_review_shape_proxy_v1.json").read_bytes())
    raw["transport"]["model"] = "gpt-6-sol"
    return read_provider_config(raw)


def test_current_profile_preserves_entire_bound_provider_except_batch_size():
    origin = bound_config()
    selected = r.review_configuration(origin, p.CURRENT_PROFILE)
    raw = selected.model_dump(mode="json")
    assert raw["review_batching"]["batch_size"] == 1
    raw["review_batching"]["batch_size"] = 3
    assert raw == origin.model_dump(mode="json")
    with pytest.raises(ValueError, match="profile_origin|profile_delta"):
        p.configuration(origin)  # Historical fixed-provider profile stays strict.


@pytest.mark.parametrize(
    "profile,truncated,unknown",
    [
        (p.CURRENT_PROFILE, None, None),
        (p.CURRENT_PROFILE, "request:truncated", "request:unknown"),
        (p.PROFILE, None, "request:unknown"),
        (None, None, "request:unknown"),
    ],
)
def test_invalid_authorization_rejected_before_input_or_successor(
    profile, truncated, unknown, monkeypatch
):
    monkeypatch.setattr(r, "load", lambda _: pytest.fail("must reject before reading/registering"))
    with pytest.raises(ValueError, match="generic_pending_"):
        r.run_review(
            "old",
            "new",
            review_profile=profile,
            acknowledge_truncated_request=truncated,
            acknowledge_unknown_request=unknown,
        )


def unknown_fixture(monkeypatch, pg=None):
    from datetime import timedelta

    from agents.provider_agents import StructuredRoleAgent
    from packages.evidence_details.acceptance import ResearchAcceptance
    from packages.evidence_details.live_permission import ResearchLiveGrant, ResearchLivePermission
    from packages.llm_provider.bounded_live import BoundedLiveBudget
    from packages.llm_provider.provider import StructuredProvider
    from packages.planner_control.models import Profile
    from tests.agent_provider.test_bounded_live import SyntheticJournal, SyntheticWire
    from tests.expert_agent.test_batch_assembly import LiveDocuments
    from tests.planner.support import binding

    f = dynamic_fixture(
        17,
        replies=[batch_reply, Reply(error="transport", api_requests=1, response_confirmed=False)],
        config_path=p.PROVIDER.replace("single", "shape"),
    )
    profile = Profile(max_steps=20, max_no_progress=2, max_retries=1, seconds=2700)
    f.binding = (
        pg.binding.model_copy(
            update={"profile": profile, "deadline": pg.binding.started_at + timedelta(seconds=2700)}
        )
        if pg
        else binding(profile=profile)
    )
    f.config = f.config.model_copy(update={"mode": "live"})
    raw = f.owner.state.model_dump(mode="json")
    raw["config_checksum"] = digest(f.config)
    raw["material"]["scope"].update(
        case_code=f.binding.case_code,
        run_id=f.binding.run_id,
        graph=f.binding.graph.model_dump(mode="json"),
        input_checksum=f.binding.input_checksum,
        visible_checksum=f.binding.visible_checksum,
    )
    raw["material"]["scope"].pop("material_checksum")
    if pg:
        from apps.vectorengine_case_materials import ref

        raw["resources"] = [
            ref("resource", pg.root.asset_code, pg.root.checksum).model_dump(mode="json")
        ]
        for item in raw["policy"]["items"]:
            item["dependency_paths"] = [[pg.root.asset_code]]
            if item.get("quality"):
                item["quality"]["source"] = raw["resources"][0]
        raw["material"]["scope"]["policy_checksum"] = digest(raw["policy"])
        raw["material"]["policy_ref"]["checksum"] = digest(raw["policy"])
    raw["material"]["scope"]["material_checksum"] = digest(raw["material"])
    f.owner.state = validate(type(f.owner.state), raw)
    if pg:
        from agents.planner.service import Planner
        from packages.research_runtime.agent_records import RuntimeAgentJournal
        from tests.planner.support import FakeOwner

        f.documents = pg.documents
        planner = Planner(runtime=f.documents.sessions, owner=FakeOwner(f.binding))
        planner.begin(f.binding)
        journal = RuntimeAgentJournal(f.documents)
    else:
        f.documents = LiveDocuments()
        f.documents.run_id = f.binding.run_id
        f.documents.root = SimpleNamespace(checksum=f.binding.graph.checksum)
        control = validate(Control, dict(binding=f.binding, graph=f.binding.graph, phase="select"))
        f.documents.sessions = SimpleNamespace(read=lambda _: control)
        journal = SyntheticJournal()
        journal.documents = f.documents
    f.budget = BoundedLiveBudget(binding=f.binding, config=f.config, journal=journal)
    wire = SyntheticWire(
        [batch_reply, Reply(error="transport", api_requests=1, response_confirmed=False)]
    )
    f.provider = StructuredProvider(
        f.config, live_opt_in=True, transport=wire, sleep=lambda _: None
    )
    f.agent = StructuredRoleAgent("review", provider=f.provider, materials=f.owner, budget=f.budget)
    grant = ResearchLiveGrant(
        enabled=True,
        case_code=f.binding.case_code,
        run_id=f.binding.run_id,
        input_checksum=f.binding.input_checksum,
        visible_checksum=f.binding.visible_checksum,
        policy_checksum=f.owner.current().material.scope.policy_checksum,
        config_checksum=digest(f.config),
        root_checksum=f.documents.root.checksum,
        purpose="bounded_single_case_research",
        deadline=f.binding.deadline,
        roles=("review",),
    )
    permission = ResearchLivePermission(
        grant=grant, config=f.config, documents=f.documents, budget=f.budget
    )
    f.acceptance = ResearchAcceptance(
        owner=f.owner, documents=f.documents, requests=f.budget, live_permission=permission
    )
    with pytest.raises(ProviderError, match="transport"):
        generate(f)
    sub = collect_partial(f.owner.current(), f.documents, "synthetic-batches")
    f.acceptance.submit(sub, key="partial")
    assert f.acceptance.accept(key="partial").status == "accepted"
    if pg:
        from packages.planner_control.models import Command

        planner.step(
            f.binding.run_id,
            Command(
                key="close-pending",
                action="stop",
                expected_revision=planner.read(f.binding.run_id).revision,
            ),
        )
    else:
        control = validate(Control, dict(binding=f.binding, graph=f.binding.graph, phase="stopped"))
        f.documents.sessions = SimpleNamespace(read=lambda _: control)
        monkeypatch.setattr(
            "packages.evidence_details.pending_review.RuntimeAgentJournal",
            lambda _: f.budget.journal,
        )
    return f, sub


def test_original_owner_selects_exact_unknown_not_accepted_or_other_unsent(monkeypatch):
    f, sub = unknown_fixture(monkeypatch)
    ack = f.budget.records()[-1].request_id
    before = digest(f.documents.saved), digest(f.budget.journal.read())
    kwargs = dict(
        documents=f.documents,
        state=f.owner.current(),
        acceptance_key="partial",
        allow_empty=True,
        allow_legacy_partial=True,
        inspect_truncated_failures=True,
    )
    unacknowledged = read_pending_review(**kwargs)
    with pytest.raises(ValueError, match="unknown_ack_mismatch"):
        p.select_unknown(unacknowledged, ack)
    with pytest.raises(ValueError, match="unknown_ack_mismatch"):
        read_pending_review(**kwargs, acknowledge_unknown_request="request:wrong")
    inventory = read_pending_review(**kwargs, acknowledge_unknown_request=ack)
    proof = p.select_unknown(inventory, ack)
    assert proof["selected_ids"] == sorted(sub.plan.batches[1].assessment_ids)
    assert not set(proof["selected_ids"]) & set(inventory["accepted_ids"])
    assert not set(proof["selected_ids"]) & set(unacknowledged["selected_ids"])
    monkeypatch.setattr(
        p,
        "original_phase",
        lambda _: (f.documents, f.owner.current(), None, {"acceptance_key": "partial"}),
    )
    assert (
        p.current_selection(
            object(), None, profile=p.CURRENT_PROFILE, acknowledge_unknown_request=ack
        )
        == proof
    )
    assert before == (digest(f.documents.saved), digest(f.budget.journal.read()))


def test_constructor_recheck_keeps_provider_and_budget_and_no_vision(tmp_path, monkeypatch):
    from packages.evaluation_preparation.generic_campaign import GenericCampaign

    original = bound_config()

    def setup(self, spec, registered):
        self.config, self.provider_path = original, "bound-original-provider.json"

    monkeypatch.setattr(GenericCampaign, "__init__", setup)
    monkeypatch.setattr(r, "directory", lambda key: tmp_path / key)
    monkeypatch.setattr(r.GenericReview, "prior", lambda self: {"reservations": 14})
    calls = []
    proof = dict(selected_ids=["a", "b", "c"])

    def current(campaign, truncated, **kwargs):
        calls.append((truncated, kwargs))
        return dict(proof)

    monkeypatch.setattr(p, "current_selection", current)
    c = r.GenericReview(
        {"schema_version": "fairvlmed-run-input/1.8.0"},
        None,
        "new",
        review_profile=p.CURRENT_PROFILE,
        acknowledge_unknown_request="request:exact",
    )
    assert c.provider_path == "bound-original-provider.json"
    assert c.acquisition.config == original
    assert c.config == r.review_configuration(original, p.CURRENT_PROFILE)
    assert c.max_new_requests == 10 and c.max_vision_requests == 0 and c.duration_seconds == 2700
    c.before_review()
    assert len(calls) == 2 and calls[0] == calls[1]
    proof["selected_ids"] = ["changed"]
    with pytest.raises(ValueError, match="changed_before_dispatch"):
        c.before_review()


def test_original_cli_forwards_exact_profile_and_ack(monkeypatch):
    calls = []
    monkeypatch.setattr(r, "run_review", lambda *args, **kwargs: calls.append((args, kwargs)) or {})
    assert (
        r.main(
            [
                "--key",
                "old",
                "--review-key",
                "new",
                "--review-profile",
                p.CURRENT_PROFILE,
                "--acknowledge-unknown-request",
                "request:exact",
            ],
        )
        == 0
    )
    assert calls == [
        (
            (),
            dict(
                key="old",
                review_key="new",
                review_profile=p.CURRENT_PROFILE,
                acknowledge_unknown_request="request:exact",
            ),
        )
    ]


@pytest.mark.parametrize(
    "change", [None, "origin", "derivation", "profile", "eligibility", "old_ack"]
)
def test_new_selection_cold_restore_checks_exact_version_and_bound_provider(
    tmp_path, monkeypatch, change
):
    from packages.evaluation_preparation import case_execution, generic_input

    origin = bound_config()
    spec = dict(key="parent", schema_version="fairvlmed-run-input/1.8.0")
    binding = dict(synthetic="binding")
    proof = dict(selected_ids=["x"], acknowledge_unknown_request="request:exact")
    config = r.review_configuration(origin, p.CURRENT_PROFILE)
    selection = dict(
        schema_version="generic-review-selection/1.3.0",
        parent_key="parent",
        review_key="child",
        review_profile=p.CURRENT_PROFILE,
        input_checksum=digest(spec),
        provider_derivation="bound-provider-batch-size-only/1.0.0",
        origin_provider_checksum=digest(origin),
        provider_checksum=digest(config),
        selected_ids=["x"],
        eligibility_checksum=digest(proof),
        acknowledge_unknown_request="request:exact",
    )
    if change == "origin":
        selection["origin_provider_checksum"] = digest("changed")
    elif change == "derivation":
        selection["provider_derivation"] = "unbound"
    elif change == "profile":
        selection["review_profile"] = p.PROFILE
    elif change == "eligibility":
        selection["eligibility_checksum"] = digest("changed")
    elif change == "old_ack":
        selection["acknowledge_truncated_request"] = None
    parent, child = tmp_path / "parent", tmp_path / "child"
    parent.mkdir()
    child.mkdir()
    (parent / "review-successor.json").write_text(json.dumps(selection))
    (child / "input.json").write_text(json.dumps(dict(selection, key="child")))
    for path in (parent, child):
        (path / "binding.json").write_text(json.dumps(binding))
    output = child / "evidence"
    output.mkdir()
    result = dict(phases=[dict(run_id="RUN:SYNTHETIC", accepted=True)])
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

    def construct(*args, **kwargs):
        calls.append(kwargs)
        return campaign

    monkeypatch.setattr(generic_input, "directory", lambda key: tmp_path / key)
    monkeypatch.setattr(generic_input, "load", lambda _: spec)
    monkeypatch.setattr(generic_input, "provider_config", lambda _: origin)
    monkeypatch.setattr(case_execution, "owners", lambda: (None, None, None))
    monkeypatch.setattr(r, "GenericReview", construct)
    if change:
        with pytest.raises(ValueError, match="generic_aggregation_"):
            p.restore_campaign(child)
    else:
        assert p.restore_campaign(child) == (campaign, result)
        assert calls == [
            dict(review_profile=p.CURRENT_PROFILE, acknowledge_unknown_request="request:exact")
        ]


def test_wrong_exact_ack_rejects_before_claim_and_time_window(tmp_path, monkeypatch):
    (tmp_path / "parent").mkdir()
    monkeypatch.setattr(r, "directory", lambda key: tmp_path / key)
    monkeypatch.setattr(r, "load", lambda _: {"schema_version": "fairvlmed-run-input/1.8.0"})
    monkeypatch.setattr(r, "provider_config", lambda _: bound_config())
    monkeypatch.setattr(r, "register", lambda _: None)

    def reject(*args, **kwargs):
        assert kwargs["acknowledge_unknown_request"] == "request:wrong"
        raise ValueError("pending_review_unknown_ack_mismatch")

    monkeypatch.setattr(r, "GenericReview", reject)
    with pytest.raises(ValueError, match="unknown_ack_mismatch"):
        r.run_review(
            "parent",
            "child",
            review_profile=p.CURRENT_PROFILE,
            acknowledge_unknown_request="request:wrong",
        )
    assert not (tmp_path / "parent/review-successor.json").exists()
    assert not (tmp_path / "child/started.json").exists()
