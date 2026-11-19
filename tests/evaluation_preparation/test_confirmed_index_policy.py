"""Current indexes use original owners, without opening another public search."""

import pytest

from packages.llm_provider.provider import ProviderError
from packages.multimodal_research import campaign_renewal as renewal
from packages.multimodal_research import execution_lease as lease
from packages.multimodal_research.definition_expansion import (
    ConfirmedIndexExpansionPolicy,
    DefinitionExpansion,
    read_expansion_policy,
)
from tests.evaluation_preparation.test_definition_runtime import policy


def indexed_policy():
    value = policy()
    value.update(
        schema_version="campaign-definition-expansion/1.31.0",
        index_keys=["sha256:" + "a" * 64],
        unresolved_relation_retrieval=None,
        retrieval_scope="confirmed-index-only/1.0.0",
    )
    return value


def plan():
    return dict(
        schema_version="multimodal-campaign-renewal/1.56.0",
        mode="definitions",
        definition_increment_only=True,
        expansion_policy=indexed_policy(),
        execution_lease_profile=lease.PROFILE,
        new_public_search_allowed=False,
    )


def test_index_only_scope_keeps_candidate_contract_and_runtime_owner():
    parsed = read_expansion_policy(indexed_policy())
    assert isinstance(parsed, ConfirmedIndexExpansionPolicy)
    assert parsed.prompt_version == "definition-draft-6.4.0"
    assert parsed.unresolved_relation_retrieval is None
    renewal.validate_definition_runtime(plan())
    assert renewal.current_increment(plan())


@pytest.mark.parametrize("indexes", [[], ["key"] * 9])
def test_index_count_is_explicit_and_bounded(indexes):
    with pytest.raises(ValueError):
        read_expansion_policy(dict(indexed_policy(), index_keys=indexes))


def test_new_policy_cannot_request_unseen_first_public_search():
    with pytest.raises(ValueError):
        read_expansion_policy(
            dict(indexed_policy(), unresolved_relation_retrieval="unseen-first-for-owner-gap/1.0.0")
        )


@pytest.mark.parametrize("value", [True, None, "false"])
def test_runtime_window_cannot_relax_no_network_scope(value):
    with pytest.raises(ValueError, match="definition_index_only_no_public_dispatch"):
        renewal.validate_definition_runtime(dict(plan(), new_public_search_allowed=value))


@pytest.mark.parametrize("kind", ["search", "fetch", None])
def test_no_wire_reservation_even_if_connector_is_incorrectly_installed(kind):
    expansion = object.__new__(DefinitionExpansion)
    expansion.policy = read_expansion_policy(indexed_policy())
    # No documents or budget installed: rejection must precede any owner write.
    with pytest.raises(ProviderError, match="definition_index_only_no_public_dispatch"):
        expansion.reserve_wire("public", kind=kind)


def test_old_public_policy_still_enters_original_guard():
    expansion = object.__new__(DefinitionExpansion)
    expansion.policy = read_expansion_policy(policy())

    def guard():
        raise ValueError("original_guard_reached")

    expansion.guard = guard
    with pytest.raises(ValueError, match="original_guard_reached"):
        expansion.reserve_wire("public", kind="search")
