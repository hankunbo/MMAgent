"""One original source owner per historical reconstruction, with fresh exit proof."""

from contextvars import Context

import pytest
from test_source_reads import sources

from packages.evidence_details import accepted_read_scope as scoped
from packages.grounding.models import digest


def factory_fixture(monkeypatch):
    made = []

    def factory():
        owner, records, values, calls = sources(monkeypatch)
        owner.initial = owner.read()
        made.append((owner, records, values, calls))
        return owner

    return factory, made


def test_repeated_ancestor_construction_shares_owner_and_returns_fresh_proof(monkeypatch):
    factory, made = factory_fixture(monkeypatch)
    with scoped.historical_fact_read_scope():
        first = scoped._historical_medical_sources(factory)
        for _ in range(76):
            current = scoped._historical_medical_sources(factory)
            assert current is first
            assert current.read() == first.initial
        assert len(made) == 1 and len(made[0][3]) == 3
    assert len(made[0][3]) == 6  # Actual original source reader repeats at return.
    assert scoped._facts.get() is None


def test_outside_scope_and_later_call_never_reuse_owner(monkeypatch):
    factory, made = factory_fixture(monkeypatch)
    one = scoped._historical_medical_sources(factory)
    two = scoped._historical_medical_sources(factory)
    with scoped.historical_fact_read_scope():
        three = scoped._historical_medical_sources(factory)
    with scoped.historical_fact_read_scope():
        four = scoped._historical_medical_sources(factory)
    assert len({id(v) for v in (one, two, three, four)}) == 4
    assert [len(row[3]) for row in made] == [3, 3, 6, 6]


def test_nested_scope_and_independent_context_have_distinct_owners(monkeypatch):
    factory, made = factory_fixture(monkeypatch)

    def independent():
        with scoped.historical_fact_read_scope():
            return scoped._historical_medical_sources(factory)

    with scoped.historical_fact_read_scope():
        one = scoped._historical_medical_sources(factory)
        with scoped.historical_fact_read_scope():
            assert scoped._historical_medical_sources(factory) is one
        assert Context().run(independent) is not one
    assert len(made) == 2 and all(len(row[3]) == 6 for row in made)


@pytest.mark.parametrize("mutation", ["initial", "reader"])
def test_owner_or_initial_mutation_rejected_even_without_another_read(monkeypatch, mutation):
    factory, _ = factory_fixture(monkeypatch)
    with pytest.raises(ValueError, match="historical_medical_source_factory_owner_changed"):
        with scoped.historical_fact_read_scope():
            one = scoped._historical_medical_sources(factory)
            if mutation == "initial":
                one.initial.clear()
            else:
                one.reader.chunks[0]["locator"] = "changed"
    assert scoped._facts.get() is None


def test_second_factory_cannot_replace_original_owner(monkeypatch):
    factory, _ = factory_fixture(monkeypatch)
    with pytest.raises(ValueError, match="historical_medical_source_factory_changed"):
        with scoped.historical_fact_read_scope():
            scoped._historical_medical_sources(factory)
            scoped._historical_medical_sources(lambda: factory())
    assert scoped._facts.get() is None


def test_withdrawal_still_checked_after_shared_ancestors(monkeypatch):
    factory, made = factory_fixture(monkeypatch)
    with pytest.raises(ValueError, match="source_withdrawn"):
        with scoped.historical_fact_read_scope():
            one = scoped._historical_medical_sources(factory)
            _, records, values, _ = made[0]
            values[digest(records["source"].reference())].withdrawn = True
            assert scoped._historical_medical_sources(factory) is one
    assert scoped._facts.get() is None


def test_recursive_factory_and_technical_error_do_not_leak_scope(monkeypatch):
    def recursive():
        return scoped._historical_medical_sources(recursive)

    with pytest.raises(ValueError, match="historical_medical_source_factory_cycle"):
        with scoped.historical_fact_read_scope():
            scoped._historical_medical_sources(recursive)
    assert scoped._facts.get() is None

    def fail():
        raise OSError("source_constructor_failed")

    with pytest.raises(OSError, match="source_constructor_failed"):
        with scoped.historical_fact_read_scope():
            scoped._historical_medical_sources(fail)
    factory, made = factory_fixture(monkeypatch)
    with scoped.historical_fact_read_scope():
        scoped._historical_medical_sources(factory)
    assert scoped._facts.get() is None and len(made) == 1
