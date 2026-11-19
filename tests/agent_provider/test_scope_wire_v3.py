import json
from pathlib import Path

import pytest

from packages.evidence_details.acceptance_models import read_records
from packages.evidence_details.input_scope import ScopeSubmissionV3
from packages.evidence_details.review_batches import read_submission
from packages.grounding.models import digest
from packages.llm_provider.input_scope import TEXT_PROMPT, ScopeVisionProposalV3
from tests.agent_provider import test_input_scope as original


def test_new_wire_through_original_provider_acceptance_and_exact_cold_read(monkeypatch):
    monkeypatch.setattr(original, "PROMPT", TEXT_PROMPT)
    monkeypatch.setattr(original, "ScopeSubmission", ScopeSubmissionV3)

    def response(value):
        value["schema_version"] = "visual-input-scope-candidate/1.2.0"
        value["regions"][0]["limitation"] = "L" * 106
        value["regions"][0]["finding"] = "F" * 230

    f = original.fixture(response_change=response)
    sub = original.generate(f)
    assert type(sub.candidate) is ScopeVisionProposalV3
    assert read_submission(sub) == sub
    assert f.budget.records()[0].prompt_version == TEXT_PROMPT
    assert f.budget.records()[0].prompt_checksum == digest(
        Path("configs/prompts/agents", TEXT_PROMPT + ".txt").read_text()
    )
    f.acceptor.submit(sub, key="new-text-wire")
    receipt = f.acceptor.accept(key="new-text-wire")
    assert receipt.status == "accepted" and receipt.scores_checksum is None
    records = read_records(f.acceptor.read(key="new-text-wire")[0])
    assert records.schema_version == "case-input-scope-observation/3.0.0"
    assert records.candidate.regions[0].limitation == "L" * 106
    assert records.candidate.regions[0].finding == "F" * 230
    assert not records.interpretations
    count = len(f.docs.audit)
    assert f.acceptor.accept(key="new-text-wire") == receipt
    assert len(f.docs.audit) == count and len(f.budget.records()) == 1


def test_policy_drives_actual_phase_prompt_and_schema(tmp_path):
    from packages.multimodal_research.active_requirements import load
    from packages.multimodal_research.input_scope import configuration, project_plan, read_policy
    from tests.evaluation_preparation.test_input_scope_phase import PROFILE, fixture

    policy_path = Path("configs/research/input_scope_policy_v3.json")
    ref = dict(path=str(policy_path), checksum=digest(json.loads(policy_path.read_text())))
    assert read_policy(ref).schema_version == "input-scope-phase-policy/1.2.0"
    profile = load(PROFILE)
    profile["input_scope_policy"] = ref
    base, campaign, _ = fixture(tmp_path, profile=profile)
    plan = project_plan(base, ref)
    assert plan["prompt_version"] == TEXT_PROMPT
    selected = configuration(campaign.config, plan)
    role = next(r for r in selected.roles if r.role == "vision")
    assert role.prompt_version == TEXT_PROMPT
    assert role.prompt_checksum == plan["prompt_checksum"]
    assert selected.transport == campaign.config.transport


def test_old_prompt_cannot_accept_new_wire_version():
    from packages.llm_provider.provider import ProviderError

    def response(value):
        value["schema_version"] = "visual-input-scope-candidate/1.2.0"

    f = original.fixture(response_change=response)
    with pytest.raises(ProviderError):
        original.generate(f)
    assert f.budget.records()[0].error == "schema"
    assert not f.docs.audit
