from threading import Event

import pytest

from packages.grounding.models import digest
from packages.planner_control.models import Gap
from packages.validation_recovery import Failure
from packages.validation_recovery.requirements import ExtensionFixture, replay_prerequisites
from tests.agent_provider.support import ref
from tests.expert_agent.support import expert_reply
from tests.validation_recovery.support import fixture


def test_or_fallback_preserves_facts_gap_budget_and_extension_intent():
    h = fixture(category="capability_gap")
    before = h.planner.read(h.run_id)
    report = h.service.resolve(h.run_id, key="or")
    state = h.planner.read(h.run_id)
    assert state.phase == "select" and state.attempted == ("A",) and state.branches == 1
    assert state.result_refs == before.result_refs and state.reserved == 1 and state.started == 0
    assert report["control"]["unresolved"] == [g.model_dump(mode="json") for g in before.gaps]
    audit, requests = tuple(h.f.documents.audit), h.f.budget.used
    assert h.service.resolve(h.run_id, key="or") == report
    assert tuple(h.f.documents.audit) == audit and h.f.budget.used == requests == 0
    for name in ("select", "prepare", "execute", "verify"):
        h.planner.step(h.run_id, h.command(name))
    state = h.planner.read(h.run_id)
    assert state.selected.branch == "B" and state.reserved == 2 and state.started == 0
    assert before.result_refs[0] in state.result_refs and state.gaps == before.gaps
    assert state.binding.deadline == before.binding.deadline
    assert h.planner.replay(h.run_id) == state


@pytest.mark.parametrize("required", [False, True])
@pytest.mark.parametrize("reply", ["valid", "invalid", "timeout"])
def test_exhausted_or_and_blocked_stop_with_original_expert(required, reply):
    from packages.llm_provider.provider import ProviderError

    response = (
        expert_reply(action="safe_stop")
        if reply == "valid"
        else (
            expert_reply(action="alternative", action_id="invented")
            if reply == "invalid"
            else ProviderError("synthetic_timeout")
        )
    )
    h = fixture(replies=(response,), required=required, blocked=("B",))
    before = h.planner.read(h.run_id)
    report = h.service.resolve(h.run_id, key="exhausted")
    state = h.planner.read(h.run_id)
    assert state.phase == "stopped" and state.pending is None
    assert report["report"]["incident"]["stop_reason"] == "legal_paths_exhausted"
    assert report["report"]["retained_evidence"] == [
        r.model_dump(mode="json") for r in before.result_refs
    ]
    assert state.graph == before.graph and state.binding == before.binding
    assert state.started == 0 and h.f.budget.used <= h.f.budget.max_requests
    assert h.service.resolve(h.run_id, key="exhausted") == report


@pytest.mark.parametrize(
    "category,kind",
    [
        ("missing_input", "missing_data"),
        ("product_demand", "product_demand"),
        ("mapping_unresolved", "mapping"),
        ("unknown_producer", "unknown_producer"),
        ("admission_rejected", "safety"),
        ("execution_failed", "technical"),
        ("qc_failed", "technical"),
        ("medical_negative", None),
        ("valid_refutation", None),
        ("version_conflict", None),
    ],
)
def test_failure_categories_do_not_fabricate_capability_or_refutation(category, kind):
    gap = Gap(kind=kind, target="GOAL:SYNTHETIC", reason="synthetic") if kind else None
    failure = Failure(
        category=category,
        gap=gap,
        target="GOAL:SYNTHETIC",
        reason="synthetic",
        sources=(ref("source", "SOURCE:SYNTHETIC"),),
    )
    assert failure.category == category
    with pytest.raises(ValueError, match="failure_category_gap_mismatch"):
        Failure(
            **{
                **failure.model_dump(),
                "gap": Gap(
                    kind="capability",
                    target=failure.target,
                    reason="fake",
                    capability_code="CAPABILITY:FAKE",
                    requirement_ref=ref("requirement", "REQUIREMENT:FAKE"),
                ),
            }
        )


def test_capability_identity_required_and_synthetic_extension_cannot_mutate_run():
    with pytest.raises(ValueError, match="resolved_capability_required"):
        Failure(
            category="capability_gap",
            target="GOAL:X",
            reason="fake",
            sources=(ref("source", "SOURCE:X"),),
        )
    h = fixture(category="capability_gap")
    value = h.service.resolve(h.run_id, key="gap")
    from packages.grounding.models import validate
    from packages.validation_recovery import ToolRequirement

    requirement = validate(ToolRequirement, value["incident"]["requirements"][0])
    original = h.planner.read(h.run_id)
    for result in ("failed", "qualified_for_future_run"):
        event = ExtensionFixture(
            requirement_checksum=digest(requirement),
            outcome=result,
            future_catalog=ref("catalog", "CATALOG:NEW") if result != "failed" else None,
        )
        prerequisites = replay_prerequisites(requirement, event, original.binding)
        assert prerequisites["created_run"] is None and not prerequisites["current_run_changed"]
        assert h.planner.read(h.run_id) == original


@pytest.mark.parametrize("limit", ["cancelled", "deadline", "no_progress", "branches", "requests"])
def test_limits_do_not_refresh_deadline_or_start_work(limit):
    from packages.planner_control.models import Profile

    limits = {"no_progress": {"max_no_progress": 1}, "branches": {"max_branches": 0}}
    h = fixture(blocked=("B",), profile=Profile(**limits.get(limit, {})))
    signal = Event()
    if limit == "cancelled":
        signal.set()
    elif limit == "deadline":
        h.planner.clock = lambda: h.f.binding.deadline
    elif limit == "requests":
        for _ in range(h.f.budget.max_requests):
            h.f.budget.reserve(
                h.f.reader.state.material.scope, h.f.binding.deadline, digest(h.f.provider.config)
            )
    requests_before = h.f.budget.used
    before = h.planner.read(h.run_id)
    result = h.service.resolve(h.run_id, key="limit", cancel=signal)
    after = h.planner.read(h.run_id)
    assert after.phase == "stopped" and after.binding.deadline == before.binding.deadline
    assert after.reserved == before.reserved and after.started == before.started
    if limit in {"cancelled", "deadline", "no_progress", "requests"}:
        assert h.f.budget.used == requests_before
    assert result["report"]["conclusion_status"] == "bounded_research_stop"
