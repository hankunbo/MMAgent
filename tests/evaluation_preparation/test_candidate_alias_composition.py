"""Actual definition discovery consumes the batch port without changing selection."""

import pytest

from tests.evaluation_preparation.test_definition_complete_pool import make_sources


def test_actual_source_owner_batches_queries_but_preserves_legacy_selection():
    owner, _ = make_sources("definition-passage-plan/3.0.0")
    expected = owner.reconcile_pool(queries=["q"], limit=1)
    owner, _ = make_sources("definition-passage-plan/3.0.0")
    calls = []
    old = owner.repository.query_candidate_index

    def batch(key, queries, filters, *, before_read):
        before_read()
        calls.append((key, queries))
        return tuple(old(key, q, filters) for q in queries)

    owner.repository.query_candidate_aliases = batch
    actual = owner.reconcile_pool(queries=["q"], limit=1)
    assert calls == [("original", ("public target",))]
    execution = actual["retrieval"].pop("local_query_execution")
    assert execution["current_reads_per_index"] == 1
    assert execution["consumption_revalidation_required"] is True
    assert actual == expected


def test_batch_failure_is_not_empty_material_or_silent_legacy_fallback():
    owner, _ = make_sources("definition-passage-plan/3.0.0")

    def failed(*args, **kwargs):
        raise RuntimeError("source_backend_failure")

    owner.repository.query_candidate_aliases = failed
    owner.repository.query_candidate_index = lambda *a, **kw: pytest.fail("no fallback")
    with pytest.raises(RuntimeError, match="source_backend_failure"):
        owner.reconcile_pool(queries=["q"], limit=1)
