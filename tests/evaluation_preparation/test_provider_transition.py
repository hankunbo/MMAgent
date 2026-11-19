"""New provider window cannot relabel old execution or reset cumulative budgets."""

import copy
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from packages.evidence_details.live_permission import validate_live_config
from packages.grounding.models import digest
from packages.llm_provider.config import qualified_citation_upgrade
from packages.multimodal_research.campaign import configuration as original_configuration
from packages.multimodal_research.provider_transition import (
    PROFILE,
    VERSION,
    configuration,
    plan_for,
    read_ancestry,
    verify_unexecuted,
)


def configs():
    old = qualified_citation_upgrade(
        original_configuration(
            resilient_vision=True,
            directional=True,
            exact_primary_status=True,
            qualified_observations=True,
        )
    )
    return old, configuration(old, PROFILE)


def origin():
    now = datetime.now(UTC)
    return now, dict(
        schema_version="multimodal-campaign-renewal/1.20.0",
        mode="review",
        review_acquired_only=True,
        repair_acquired_contract=True,
        max_model_dispatches=1,
        acquisition_targets=[],
        review_target_ids=["active:new"],
        deadline=(now - timedelta(days=1)).isoformat(),
        max_new_requests=2,
        previous_attempts=[dict(requests=19)],
        provider=digest("old"),
    )


def test_new_profile_keeps_old_prompts_limits_and_default():
    old, new = configs()
    assert old.transport.base_url == "https://api.vectorengine.cn/v1"
    assert new.transport.base_url == "https://api.relayrouter.ai/v1"
    assert new.transport.model == "gpt-6-sol"
    assert old.roles == new.roles
    assert old.max_requests == new.max_requests == 40
    assert new.transport.max_retries == 0
    assert validate_live_config(old) == old
    assert validate_live_config(new) == new


def test_plan_does_not_mutate_old_and_keeps_all_charges():
    now, old = origin()
    before = copy.deepcopy(old)
    _, config = configs()
    new = plan_for(
        old,
        previous="old",
        output="new",
        authorization="operator instruction",
        config=config,
        started=now,
        implementation={},
    )
    assert old == before
    assert new["schema_version"] == VERSION
    assert new["previous_attempts"] == old["previous_attempts"]
    assert new["provider_transition"]["previous_protocol"] == digest(old)
    assert new["max_new_requests"] == 2 and new["max_model_dispatches"] == 1
    assert new["prior_deadline"] == old["deadline"]
    assert datetime.fromisoformat(new["deadline"]) == now + timedelta(seconds=1800)


@pytest.mark.parametrize(
    "change",
    [
        {"schema_version": VERSION},
        {"mode": "acquisition"},
        {"repair_acquired_contract": False},
        {"max_model_dispatches": 2},
        {"review_target_ids": ["one", "two"]},
        {"max_new_requests": 1},
        {"previous_attempts": [dict(requests=39)]},
        {"deadline": (datetime.now(UTC) + timedelta(days=1)).isoformat()},
    ],
)
def test_invalid_transition_refused(change):
    now, old = origin()
    old.update(change)
    _, config = configs()
    with pytest.raises(ValueError, match="provider_transition_scope_or_budget"):
        plan_for(
            old,
            previous="old",
            output="new",
            authorization="test",
            config=config,
            started=now,
            implementation={},
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("base_url", "https://unapproved.invalid/v1"),
        ("model", "another-model"),
        ("api_key_env", "ANOTHER_KEY"),
    ],
)
def test_unknown_transport_not_authorized(field, value):
    _, config = configs()
    config = config.model_copy(
        update=dict(transport=config.transport.model_copy(update={field: value}))
    )
    with pytest.raises(ValueError, match="bounded_live_configuration_required"):
        validate_live_config(config)


@pytest.mark.parametrize("activity", ["result.json", "request.json", "cancel", "review"])
def test_any_predecessor_activity_rejected(tmp_path, activity):
    (tmp_path / "protocol.json").write_text("{}")
    (tmp_path / activity).write_text("{}")
    with pytest.raises(ValueError, match="predecessor_has_activity"):
        verify_unexecuted(None, tmp_path, {})


def test_runtime_presence_rejected_even_without_local_result(tmp_path):
    (tmp_path / "protocol.json").write_text("{}")
    owner = SimpleNamespace(
        request_key=lambda _: "old",
        contexts=SimpleNamespace(read_research_run=lambda *a, **k: object()),
    )
    with pytest.raises(ValueError, match="predecessor_has_runtime"):
        verify_unexecuted(owner, tmp_path, {})


def test_owner_readback_current_node_is_not_a_cycle(tmp_path):
    assert read_ancestry(tmp_path, 1, (tmp_path,)) == (tmp_path,)
    assert read_ancestry(tmp_path, 0, ()) == (tmp_path,)
    with pytest.raises(ValueError, match="version_or_cycle"):
        read_ancestry(tmp_path, 2, (tmp_path, tmp_path / "other"))
    with pytest.raises(ValueError, match="version_or_cycle"):
        read_ancestry(tmp_path, 17, ())
