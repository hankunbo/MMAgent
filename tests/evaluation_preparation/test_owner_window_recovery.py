"""Narrow real-owner contracts, staged without changing live frozen implementation."""

import copy
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from packages.evaluation_preparation.dynamic_campaign import bounded_phase_binding
from packages.evidence_reuse.definition_acceptance import ResearchDefinitionOwner
from packages.llm_provider.provider import ProviderError
from packages.multimodal_research.campaign import MultimodalCampaign
from tests.evaluation_preparation.test_definition_expansion import Sources, draft
from tests.evaluation_preparation.test_owner_dependency_definition import raw
from tests.evaluation_preparation.test_stage_recovery_14a import quoted


@pytest.mark.parametrize("quoted_value", [False, True])
def test_reconstruction_brackets_one_source_then_rereads_on_next_call(quoted_value):
    sources = Sources()
    reads = []
    original = sources.read

    def tracked(code):
        reads.append(code)
        return original(code)

    sources.read = tracked
    owner = ResearchDefinitionOwner(
        documents=None,
        sources=sources,
        campaign_id="synthetic",
        target="GOAL:TEST",
        input_checksum="synthetic",
        guard=lambda: None,
    )
    value = quoted() if quoted_value else draft("GOAL:TEST")
    before = copy.deepcopy(value)
    saved = owner.build(value)
    assert value == before and len(reads) == 2
    assert owner.build(value) == saved and len(reads) == 4
    sources.value["state"] = "withdrawn"
    with pytest.raises(ValueError, match="definition_draft_exact_excerpt"):
        owner.build(value)


def test_change_during_owner_reconstruction_is_rejected():
    sources = Sources()
    read = sources.read
    calls = []

    def changed(code):
        calls.append(code)
        if len(calls) == 2:
            sources.value["state"] = "withdrawn"
        return read(code)

    sources.read = changed
    owner = ResearchDefinitionOwner(
        documents=None,
        sources=sources,
        campaign_id="synthetic",
        target="GOAL:TEST",
        input_checksum="synthetic",
        guard=lambda: None,
    )
    with pytest.raises(ValueError, match="definition_draft_source_changed"):
        owner.build(quoted())


def test_input_change_on_final_source_read_still_rejected():
    owner, manifest, sources, docs, _, value = raw()
    original = sources.read
    calls = []

    def changed(code):
        calls.append(code)
        result = original(code)
        if len(calls) == 2:
            manifest.components = (manifest.components[0].model_copy(update={"id": "changed"}),)
        return result

    sources.read = changed
    with pytest.raises(ValueError, match="definition_dependency_input_changed"):
        owner.accept("changed", value)
    assert docs.read("definition:changed") is None


@pytest.mark.parametrize("method", ["guard", "guard_definition_materials", "guard_control"])
@pytest.mark.parametrize("stop", ["deadline", "cancelled"])
def test_expired_or_cancelled_fast_stop_before_expensive_reads(tmp_path, method, stop):
    campaign = object.__new__(MultimodalCampaign)
    campaign.output = tmp_path
    campaign.protocol = {"deadline": (datetime.now(UTC) + timedelta(seconds=60)).isoformat()}
    if stop == "deadline":
        campaign.protocol["deadline"] = "2000-01-01T00:00:00+00:00"
    else:
        (tmp_path / "cancel").touch()
    campaign.base = SimpleNamespace(
        active_owner=True,
        detailed=False,
        check=lambda: pytest.fail("expensive read before fast stop"),
        check_definition_materials=lambda: pytest.fail("expensive read before fast stop"),
    )
    campaign.previous_attempts = lambda: pytest.fail("history read before fast stop")
    with pytest.raises(ProviderError, match=stop):
        getattr(campaign, method)()


def test_binding_narrows_profile_and_preserves_original_exact_contract():
    from tests.planner.support import binding

    original = binding()
    value = bounded_phase_binding(original, original.started_at + timedelta(seconds=12.9))
    assert value.deadline == original.started_at + timedelta(seconds=12)
    assert value.profile.seconds == 12 and value.started_at == original.started_at
    assert bounded_phase_binding(original, value.deadline, value) == value
    # Historical bindings stay byte-exact; outer campaign fast stop still applies.
    assert bounded_phase_binding(original, value.deadline, original) == original
    assert bounded_phase_binding(original, original.deadline + timedelta(seconds=100)) == original
    with pytest.raises(ProviderError, match="deadline"):
        bounded_phase_binding(original, original.started_at)
    with pytest.raises(ValueError, match="existing_binding_changed"):
        bounded_phase_binding(
            original.model_copy(update={"run_id": "RUN:OTHER"}), value.deadline, value
        )


@pytest.mark.parametrize("boundary", [1, 2, 3, 4, 5, 6])
def test_original_planner_stops_before_next_step_or_request(monkeypatch, tmp_path, boundary):
    from threading import Event

    from apps.vectorengine_live_assembly import run_case_stages
    from tests.expert_agent.batch_case_support import vision_reply
    from tests.expert_agent.test_batch_assembly import assembly

    f = assembly(monkeypatch, replies=[vision_reply])
    calls = []

    def guard():
        calls.append(1)
        if len(calls) == boundary:
            raise ProviderError("deadline")

    with pytest.raises(ProviderError, match="deadline"):
        run_case_stages(
            planner=f.planner,
            owner=f.owner,
            ports=f.ports,
            review=f.review,
            documents=f.documents,
            binding=f.binding,
            config=f.config,
            output=tmp_path,
            signal_cancel=Event(),
            selected_roles=("vision",),
            before_step=guard,
        )
    assert len(calls) == boundary
    assert len(f.wire.calls) == f.ports.budget.used == 0


def test_memory_capacity_is_explicit_and_other_errors_still_fail(monkeypatch):
    from packages.memory.multimodal_archive import ArchiveSourceCapacityError
    from packages.multimodal_research import memory, report_storage

    report = {"requirements": [], "scores": {"unchanged": True}}
    monkeypatch.setattr(report_storage, "unpack", lambda *a, **kw: (report, {}))
    docs = SimpleNamespace(
        run_id="RUN:TEST",
        key=lambda key: key,
        read=lambda key: (),
        repo=SimpleNamespace(
            read_observation_bundle=lambda key: [SimpleNamespace(asset_code="ASSET:REPORT")]
        ),
    )
    original = copy.deepcopy(report)
    result = dict(case_code="CASE:TEST", task=dict(target_code="GOAL:TEST"))
    refs = tuple({"code": str(i)} for i in range(257))

    def prepare(**kw):
        raise ArchiveSourceCapacityError(refs)

    repo = SimpleNamespace(create=lambda record: pytest.fail("partial write"))
    outcome = memory.persist(repo, SimpleNamespace(prepare=prepare), docs, result)
    assert outcome["status"] == "not_archived_source_capacity"
    assert outcome["deferred"]["required_sources"] == 257
    assert outcome["deferred"]["maximum_sources"] == 256
    assert not outcome["source_references_truncated"] and not outcome["items"]
    assert report == original

    def invalid(**kw):
        raise ValueError("source_changed")

    with pytest.raises(ValueError, match="source_changed"):
        memory.persist(repo, SimpleNamespace(prepare=invalid), docs, result)


@pytest.mark.parametrize(
    "mutation",
    [
        None,
        "wire",
        "unknown",
        "accepted",
        "new_fact",
        "empty",
        "source_missing",
        "duplicate",
        "other_error",
        "wrong_mode",
    ],
)
def test_definition_review_successor_only_after_zero_wire_stopped_timeout(mutation):
    from packages.multimodal_research.campaign_renewal import definition_review_selection

    protocol = dict(
        schema_version="multimodal-campaign-renewal/1.11.0",
        mode="definitions",
        definition_increment_only=True,
    )
    result = dict(complete=False, phases=[dict(target=None, accepted=False, error="deadline")])
    revision = dict(
        added=[], accepted={"d": {}}, replaced=[dict(id="check:original", definition="d")]
    )
    events = [dict(phase="rejected")]
    if mutation == "wire":
        events.append(dict(phase="completed"))
    elif mutation == "unknown":
        events.append(dict(phase="reserved"))
    elif mutation == "accepted":
        result["phases"][0]["accepted"] = True
    elif mutation == "new_fact":
        revision["added"] = ["check:new"]
    elif mutation == "empty":
        revision["replaced"] = []
    elif mutation == "source_missing":
        revision["accepted"] = {}
    elif mutation == "duplicate":
        revision["replaced"] *= 2
    elif mutation == "other_error":
        result["phases"][0]["error"] = "scope"
    elif mutation == "wrong_mode":
        protocol["mode"] = "acquisition"
    if mutation is None:
        assert definition_review_selection(protocol, result, revision, events) == (
            "check:original",
        )
    else:
        with pytest.raises(ValueError, match="zero_wire_timeout"):
            definition_review_selection(protocol, result, revision, events)


def test_existing_cli_passes_explicit_definition_review_flag(monkeypatch, capsys):
    import sys

    from apps import multimodal_case
    from packages.multimodal_research import campaign_renewal

    seen = []
    monkeypatch.setattr(
        campaign_renewal, "prepare", lambda **kw: seen.append(kw) or {"prepared": True}
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "multimodal_case",
            "renewal-prepare",
            "--facts",
            "facts",
            "--previous",
            "previous",
            "--output",
            "new",
            "--mode",
            "review",
            "--review-definition-increment",
            "--request-limit",
            "2",
            "--authorization",
            "synthetic-test-only",
        ],
    )
    assert multimodal_case.main() == 0
    assert seen[0]["review_definition_increment"] and seen[0]["request_limit"] == 2
    assert seen[0]["mode"] == "review" and seen[0]["expansion_policy"] is None
    assert "prepared" in capsys.readouterr().out
