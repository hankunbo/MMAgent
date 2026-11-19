"""Selection retains current original tools without inferring catalog equivalence."""

from types import SimpleNamespace

import pytest

from packages.evaluation_preparation.generic_capability_selection import confirmed_methods
from packages.grounding.models import digest
from tests.evaluation_preparation.test_generic_definition_facts import fixture


def prepared():
    f = fixture()
    o = f.state.material.observations[0]
    f.state.policy.items = (
        SimpleNamespace(
            id=f.target, observation_code=o.reference.code, method=o.method, unit=o.unit
        ),
    )
    calls = []
    f.base.check_tools = lambda: calls.append(True)
    return f, calls


def test_selection_publishes_exact_method_fact_without_new_score_or_equivalence():
    f, calls = prepared()
    before = digest(f.state.material.observations)
    value = confirmed_methods(f.state, f.base)
    assert list(value["facts"]) == [f.target]
    assert value["facts"][f.target] == f.current["facts"][f.target]
    assert not value["score_reuse_authorized"] and not value["new_execution_authorized"]
    assert len(calls) == 2 and digest(f.state.material.observations) == before


def test_no_method_alias_from_similar_title_or_caller_dataset():
    f, _ = prepared()
    f.base.dataset = "anything"
    f.state.policy.items[0].method = "different_model_same_task_name"
    assert confirmed_methods(f.state, f.base)["facts"] == {}
    assert len(f.state.material.observations) == 1


def test_changed_observation_is_not_hidden_by_existing_method_name():
    f, _ = prepared()
    f.state.material.observations = (
        f.state.material.observations[0].model_copy(update={"value": "0.9"}),
    )
    with pytest.raises(ValueError, match="generic_definition_measurement_identity"):
        confirmed_methods(f.state, f.base)


def test_current_tool_owner_revocation_propagates():
    f, _ = prepared()

    def fail():
        raise ValueError("full_tool_artifact_changed")

    f.base.check_tools = fail
    with pytest.raises(ValueError, match="full_tool_artifact_changed"):
        confirmed_methods(f.state, f.base)
