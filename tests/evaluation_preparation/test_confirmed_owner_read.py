"""Single owner read/projection; never a cache of current qualification."""

import copy

import pytest

from tests.evaluation_preparation.test_confirmed_interpretation_definition import confirmed_fixture


def test_combined_read_rebuilds_once_and_checks_fact_twice(tmp_path, monkeypatch):
    owner, candidate, item, _, _ = confirmed_fixture(tmp_path)
    receipt = owner.accept("current", candidate)
    builds, facts = [], []
    build, check = owner._build, item.check_fact

    def tracked_build(value):
        builds.append(True)
        return build(value)

    def tracked_fact():
        facts.append(True)
        return check()

    monkeypatch.setattr(owner, "_build", tracked_build)
    monkeypatch.setattr(item, "check_fact", tracked_fact)
    legacy = owner.project_confirmed(owner.read_for_report("current"))
    assert (len(builds), len(facts)) == (2, 4)
    builds.clear()
    facts.clear()
    current, projected = owner.read_confirmed_projection("current")
    assert current == receipt and projected == legacy
    assert (len(builds), len(facts)) == (1, 2)
    owner.read_confirmed_projection("current")
    assert (len(builds), len(facts)) == (2, 4), "each call revalidates; no retained grant"
    owner.sources.value["state"] = "withdrawn"
    with pytest.raises(ValueError, match="definition_draft_exact_excerpt"):
        owner.read_confirmed_projection("current")


def test_combined_read_exact_receipt_and_old_version_rejection(tmp_path):
    owner, candidate, _, _, documents = confirmed_fixture(tmp_path)
    with pytest.raises(ValueError, match="definition_confirmed_receipt_version"):
        owner.read_confirmed_projection("previous")
    receipt = copy.deepcopy(owner.accept("current", candidate))
    receipt["dependency_assessment"]["review_allowed"] = 1
    # Corrupt readback is a negative fixture, never a valid receipt.
    original = documents.read
    documents.read = lambda key: (receipt,) if key == "definition:current" else original(key)
    with pytest.raises(ValueError, match="definition_confirmed_receipt_binding"):
        owner.read_confirmed_projection("current")


def test_combined_read_rechecks_fact_after_source_reconstruction(tmp_path, monkeypatch):
    owner, candidate, item, _, _ = confirmed_fixture(tmp_path)
    owner.accept("current", candidate)
    original, calls = item.check_fact, []

    def change_between_checks():
        calls.append(True)
        if len(calls) == 2:
            item.value["changed"] = True
        original()

    monkeypatch.setattr(item, "check_fact", change_between_checks)
    with pytest.raises(ValueError, match="component_observation_changed"):
        owner.read_confirmed_projection("current")
    assert len(calls) == 2
