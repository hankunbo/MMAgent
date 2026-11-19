"""Versioned bounded observation projection; no truncation of scope findings."""

import json
from pathlib import Path

import pytest

from packages.grounding.models import digest, validate
from packages.llm_provider.input_scope import ScopeVisionProposal, ScopeVisionProposalV2


def real_candidate():
    path = Path(
        "docs/acceptance/capability-scope-real-diagnostic-20261107/schema-reproduction.json"
    )
    return json.loads(path.read_bytes())["raw_model_response"]


def test_retained_actual_response_exposes_old_capacity_and_new_lossless_storage():
    original = real_candidate()
    before = digest(original)
    with pytest.raises(ValueError, match="input_scope_original_observation_bound"):
        validate(ScopeVisionProposal, original)
    # Explicit new candidate, not a silent reinterpretation of a failed request.
    new = dict(original, schema_version="visual-input-scope-candidate/1.1.0")
    candidate = validate(ScopeVisionProposalV2, new)
    assert candidate.model_dump(mode="json") == new
    value = json.loads(candidate.observation_value())
    assert value["candidate_checksum"] == digest(candidate)
    assert value["details_pointer"] == "/candidate"
    assert value["disease_direction"] == "not_assessed"
    assert len(candidate.observation_value()) <= 1000
    assert digest(original) == before


def test_maximum_scope_payload_does_not_depend_on_model_counting_json_characters():
    raw = real_candidate()
    region = dict(raw["regions"][0], finding="观察" * 50, limitation="限制" * 50)
    raw.update(
        schema_version="visual-input-scope-candidate/1.1.0",
        finding="a" * 240,
        regions=[dict(region, region="r" + str(n)) for n in range(4)],
    )
    candidate = validate(ScopeVisionProposalV2, raw)
    assert len(candidate.observation_value()) <= 1000
    assert candidate.model_dump(mode="json") == raw


def test_storage_version_preserves_visibility_and_region_uniqueness_checks():
    raw = dict(real_candidate(), schema_version="visual-input-scope-candidate/1.1.0")
    raw["regions"][0].update(visibility="not_identified", coverage=0.9)
    with pytest.raises(ValueError, match="input_scope_unidentified_coverage"):
        validate(ScopeVisionProposalV2, raw)


def storage_fixture(monkeypatch):
    from packages.evidence_details.input_scope import ScopeSubmissionV2
    from packages.llm_provider.input_scope import STORAGE_PROMPT
    from tests.agent_provider import test_input_scope as legacy

    monkeypatch.setattr(legacy, "PROMPT", STORAGE_PROMPT)

    def change(raw):
        raw.update(schema_version="visual-input-scope-candidate/1.1.0", finding="x" * 240)

    f = legacy.fixture(response_change=change)
    candidate = f.agent.call(f.scope, deadline=f.binding.deadline)
    record = f.budget.records()[-1]
    state = f.reader.state
    sub = ScopeSubmissionV2(
        scope=f.scope,
        actor="PLANNER",
        correlation="SOURCE:AGENT_RESEARCH",
        request_id=record.request_id,
        prompt_checksum=record.prompt_checksum,
        config_checksum=record.config_checksum,
        observation_checksums=tuple(o.reference for o in state.material.observations),
        source_checksums=tuple(s.reference for s in state.material.sources),
        candidate=candidate,
    )
    return f, sub


def test_new_wire_through_original_provider_acceptance_and_exact_readback(monkeypatch):
    from packages.evidence_details.acceptance_models import read_records
    from packages.evidence_details.review_batches import read_submission

    f, sub = storage_fixture(monkeypatch)
    assert read_submission(sub) == sub
    f.acceptor.submit(sub, key="scope-storage")
    receipt = f.acceptor.accept(key="scope-storage")
    assert receipt.status == "accepted" and receipt.scores_checksum is None
    saved = f.acceptor.read(key="scope-storage")
    records = read_records(saved[0])
    assert records.schema_version == "case-input-scope-observation/2.0.0"
    assert records.candidate == sub.candidate
    assert records.candidate.finding == "x" * 240
    assert records.observations[0].value == sub.candidate.observation_value()
    assert saved[2]["scores"] is None and not records.interpretations
    before = len(f.docs.audit)
    assert f.acceptor.accept(key="scope-storage") == receipt
    assert len(f.docs.audit) == before and len(f.provider.transport.calls) == 1
    bad = records.model_dump(mode="json")
    bad["observations"][0]["value"] = "unbound summary"
    with pytest.raises(ValueError, match="input_scope_saved_observation_binding"):
        read_records(bad)


@pytest.mark.parametrize("change", ["input", "component", "source", "image", "contract"])
def test_storage_contract_does_not_bypass_current_owner_or_original_request(monkeypatch, change):
    from packages.evidence_details.input_scope import check_submission

    f, sub = storage_fixture(monkeypatch)
    if change in {"input", "component"}:
        field = "input_checksum" if change == "input" else "component_checksum"
        f.owner.input_scope_task = lambda _: f.task.model_copy(update={field: digest("changed")})
    elif change == "image":
        f.reader.image_bytes = b"changed"
    elif change == "source":
        f.reader.withdrawn = True
    else:
        record = f.budget.records()[-1].model_copy(update={"prompt_checksum": digest("old")})
        with pytest.raises(ValueError, match="input_scope_request_contract_binding"):
            check_submission(sub, f.reader.state, f.owner, record)
        return
    if change == "source":
        with pytest.raises(ValueError, match="source_withdrawn"):
            f.acceptor.submit(sub, key="changed-storage")
    else:
        assert f.acceptor.submit(sub, key="changed-storage").status == "rejected"
    assert not f.docs.read("accepted:changed-storage")
    assert len(f.provider.transport.calls) == 1


def test_scope_policy_explicitly_selects_new_prompt_without_changing_old_default(tmp_path):
    from packages.llm_provider.input_scope import PROMPT, STORAGE_PROMPT
    from packages.multimodal_research.input_scope import project_plan
    from tests.evaluation_preparation.test_input_scope_phase import fixture

    base, campaign, _ = fixture(tmp_path)
    old = base.input_scope_plan
    assert old["prompt_version"] == PROMPT
    raw = json.loads(Path(old["policy"]["path"]).read_bytes())
    raw.update(
        schema_version="input-scope-phase-policy/1.1.0",
        observation_storage="bound_candidate_reference",
    )
    policy = tmp_path / "scope-v2.json"
    policy.write_text(json.dumps(raw))
    plan = project_plan(base, dict(path=str(policy), checksum=digest(raw)))
    assert plan["prompt_version"] == STORAGE_PROMPT
    assert plan["entries"] == old["entries"]
    assert project_plan(base, old["policy"]) == old


def test_original_expert_orchestrator_accepts_full_new_candidate_and_report(monkeypatch):
    from agents.expert_agent.reporting import read_research_report, render_research_html
    from tests.expert_agent.support import expert_reply, run, setup

    f, _ = storage_fixture(monkeypatch)
    f.reader.input_scope_task = f.owner.input_scope_task
    f.reader.vision_task = f.owner.vision_task
    g = setup(
        [f.reply, expert_reply()],
        reader=f.reader,
        binding=f.binding,
        config=f.config,
        role="vision",
        facts={"required": True},
    )
    parent = g.reader.current()
    result = run(g, role="vision")
    assert result["status"] == "accepted"
    report = read_research_report(g.acceptance, key=result["acceptance_key"], parent=parent)
    assert report["records"]["schema_version"] == "case-input-scope-observation/2.0.0"
    assert report["records"]["candidate"]["finding"] == "x" * 240
    assert "optic_disc" in render_research_html(report)
    assert report["scores"] is None
    assert run(g, role="vision") == result and g.budget.used == 2
    payload = json.loads(g.provider.transport.calls[-1]["messages"][1]["content"][0]["text"])
    assert payload["schema_version"] == "input-scope-expert-input/1.1.0"
    assert payload["expert_review"]["candidate"]["finding"] == "x" * 240


def test_new_policy_first_install_passes_original_frozen_owner(tmp_path):
    from packages.llm_provider.input_scope import STORAGE_PROMPT
    from packages.multimodal_research.active_requirements import load
    from tests.evaluation_preparation.test_input_scope_phase import PROFILE, fixture

    profile = load(PROFILE)
    policy_path = Path("configs/research/input_scope_policy_v2.json")
    profile["input_scope_policy"] = dict(
        path=str(policy_path), checksum=digest(json.loads(policy_path.read_bytes()))
    )
    base, _, _ = fixture(tmp_path, profile=profile)
    base.active_owner.check()
    assert base.input_scope_plan["prompt_version"] == STORAGE_PROMPT
