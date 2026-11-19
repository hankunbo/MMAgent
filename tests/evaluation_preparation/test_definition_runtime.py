"""New opt-in definition window reuses the original Runtime exclusion owner."""

import json
from contextlib import nullcontext
from pathlib import Path

import pytest

from packages.multimodal_research import campaign_renewal as renewal
from packages.multimodal_research import execution_lease as lease
from packages.multimodal_research.definition_expansion import read_expansion_policy

VERSION = "multimodal-campaign-renewal/1.56.0"
POLICY = "campaign-definition-expansion/1.30.0"


def policy():
    raw = json.loads(
        Path("configs/research/multimodal_definition_raw_revision_v1.json").read_text()
    )
    raw.update(
        schema_version=POLICY,
        prompt_version="definition-draft-6.4.0",
        unresolved_relation_retrieval="unseen-first-for-owner-gap/1.0.0",
    )
    return raw


def test_candidate_contract_and_new_public_material_are_both_explicit():
    old = json.loads(
        Path("configs/research/multimodal_definition_raw_revision_v1.json").read_text()
    )
    new = read_expansion_policy(policy())
    assert new.prompt_version == "definition-draft-6.4.0"
    assert new.unresolved_relation_retrieval == "unseen-first-for-owner-gap/1.0.0"
    assert read_expansion_policy(old).unresolved_relation_retrieval is None
    assert POLICY in renewal.CURRENT_INCREMENT_POLICIES


def test_new_definition_window_uses_runtime_lease_not_filesystem(tmp_path, monkeypatch):
    c = object.__new__(renewal.RenewedCampaign)
    c.output = tmp_path
    c.protocol = dict(
        schema_version=VERSION,
        mode="definitions",
        definition_increment_only=True,
        expansion_policy=policy(),
        execution_lease_profile=lease.PROFILE,
    )
    seen = []
    monkeypatch.setattr(lease, "session", lambda value: seen.append(value) or nullcontext())
    assert renewal.current_increment(c.protocol)
    with c.execution_session():
        pass
    assert seen == [c]
    c.protocol["execution_lease_profile"] = "invalid"
    with pytest.raises(ValueError, match="execution_lease_profile"):
        c.execution_session()


@pytest.mark.parametrize("fault", ["mode", "policy", "increment", "profile", "legacy"])
def test_declaration_cannot_upgrade_old_or_unrelated_windows(fault):
    plan = dict(
        schema_version=VERSION,
        mode="definitions",
        definition_increment_only=True,
        expansion_policy=policy(),
        execution_lease_profile=lease.PROFILE,
    )
    if fault == "mode":
        plan["mode"] = "acquisition"
    elif fault == "policy":
        plan["expansion_policy"]["schema_version"] = "campaign-definition-expansion/1.28.0"
    elif fault == "increment":
        plan["definition_increment_only"] = False
    elif fault == "profile":
        plan["execution_lease_profile"] = "invalid"
    else:
        plan["schema_version"] = "multimodal-campaign-renewal/1.11.0"
    with pytest.raises(ValueError, match="definition_runtime"):
        renewal.validate_definition_runtime(plan)


def test_valid_declaration_and_original_budget_and_deadline_rules():
    p = dict(
        schema_version=VERSION,
        mode="definitions",
        definition_increment_only=True,
        expansion_policy=policy(),
        execution_lease_profile=lease.PROFILE,
    )
    assert renewal.validate_definition_runtime(p) is None
    assert renewal.current_increment(dict(schema_version="multimodal-campaign-renewal/1.11.0"))
    with pytest.raises(ValueError, match="proposal_capacity"):
        renewal.validate_increment_window(p["expansion_policy"], seconds=1200, cap=3)
