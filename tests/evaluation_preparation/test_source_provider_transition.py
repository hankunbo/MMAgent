"""Explicit Source-fallback provider derivation plus the original claim owner."""

import copy
import json
from pathlib import Path

import pytest

from packages.llm_provider.config import read_provider_config
from packages.llm_provider.network_route import prepare_route
from packages.llm_provider.review_rating import apply
from packages.multimodal_research.discovery_continuation import (
    ConfirmedDefinitionContinuation,
)
from packages.multimodal_research.discovery_continuation import (
    definition_provider_transition as transition,
)
from tests.evaluation_preparation.test_routed_source_backlog import routed_parent


def pair():
    old = json.loads(Path("configs/agents/multimodal_relay_local_gpt6_review_v1.json").read_bytes())
    previous = dict(
        schema_version="multimodal-live-entry/1.29.0",
        execution_profile="relay-root-question-v1",
        network_route=old["network_route"],
        provider=old,
    )
    following = json.loads(
        Path("configs/agents/multimodal_relay_local_gpt6_review_v4.json").read_bytes()
    )
    return previous, following


def test_exact_legacy_contract_keeps_original_reference():
    previous, _ = pair()
    assert transition(previous, previous["provider"], None) is None


def test_explicit_derivation_is_not_replay_permission():
    previous, new = pair()
    proof = transition(previous, new, "rating-presence/1.1.0")
    assert proof["original_source_replay_authorized"] is False
    assert proof["requires_original_owner_no_acquisition_or_review_started"]


@pytest.mark.parametrize(
    "field", ["model", "retries", "batch", "budget", "source", "vision", "review_tokens"]
)
def test_unrelated_provider_changes_rejected(field):
    previous, new = pair()
    if field == "model":
        new["transport"]["model"] = "gpt-4o-2024-08-06"
    elif field == "retries":
        new["transport"]["max_retries"] = 1
    elif field == "batch":
        new["review_batching"]["batch_size"] = 1
    elif field == "budget":
        new["max_requests"] = 39
    else:
        role = next(
            r
            for r in new["roles"]
            if r["role"] == ("review" if field == "review_tokens" else field)
        )
        role["max_output_tokens"] += 1
    with pytest.raises(ValueError):
        transition(previous, new, "rating-presence/1.1.0")


@pytest.mark.parametrize("change", ["version", "profile", "child", "route_binding", "policy"])
def test_parent_scope_is_explicit(change):
    previous, new = pair()
    policy = "rating-presence/1.1.0"
    if change == "version":
        previous["schema_version"] = "multimodal-live-entry/1.30.0"
    elif change == "profile":
        previous["execution_profile"] = "relay-context-v1"
    elif change == "child":
        previous["definition_failure_of"] = {"parent": "old"}
    elif change == "route_binding":
        previous["network_route"] = None
    else:
        policy = None
    with pytest.raises(ValueError):
        transition(previous, new, policy)


def selected(tmp_path):
    c, ref, spec, _, _, records, writes = routed_parent(tmp_path)
    raw = apply(c.config, "rating-presence/1.1.0").model_dump(mode="json")
    raw["network_route"] = prepare_route(
        environ={"HTTPS_PROXY": "http://localhost:8890"}
    ).model_dump(mode="json")
    config = read_provider_config(raw)
    ref = dict(ref, provider_transition=transition(spec, config, "rating-presence/1.1.0"))
    c.config, c.base.review_contract = config, "rating-presence/1.1.0"
    return c, ref, records, writes


def test_claim_cold_readback_and_repeat_preserve_original_source(tmp_path):
    c, ref, records, writes = selected(tmp_path)
    before = copy.deepcopy(records)
    files = {p: p.read_bytes() for p in Path(ref["parent"]).glob("*.json")}
    owner = ConfirmedDefinitionContinuation(c, ref)
    assert owner.attempts[0]["requests"] == 3
    assert owner.claim["reference"]["provider_transition"] == ref["provider_transition"]
    assert ConfirmedDefinitionContinuation(c, ref, read_only=True).claim == owner.claim
    assert ConfirmedDefinitionContinuation(c, ref).claim == owner.claim
    assert len(writes) == 1
    assert all(records[k] == v for k, v in before.items())
    assert all(p.read_bytes() == content for p, content in files.items())


@pytest.mark.parametrize(
    "change", ["acquisition_started", "proof", "missing_proof", "source_response"]
)
def test_transition_does_not_bypass_original_owner(tmp_path, change):
    c, ref, records, writes = selected(tmp_path)
    if change == "acquisition_started":
        c.contexts.read_research_run = lambda run_id: {"exists": True}
        error = "definition_successor_acquisition_started"
    elif change == "source_response":
        records["definition:response:r0"][0]["raw_text"] = "{}"
        error = "definition_successor_response_binding"
    else:
        if change == "proof":
            ref["provider_transition"]["original_source_replay_authorized"] = True
        else:
            ref.pop("provider_transition")
        error = "definition_successor_reference_changed"
    with pytest.raises(ValueError, match="^" + error + "$"):
        ConfirmedDefinitionContinuation(c, ref)
    assert not writes
