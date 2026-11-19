"""Actual public query terms and bounded incremental (not whole-case) review."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from packages.multimodal_research.definition_expansion import read_expansion_policy


def test_targeted_policy_preserves_old_templates_and_uses_real_controlled_terms():
    from packages.knowledge_search.broad import broad_plan

    p = read_expansion_policy(
        json.loads(Path("configs/research/multimodal_definition_gate_v2.json").read_bytes())
    )
    assert p.max_search_requests == p.max_fetch_requests == 2
    assert p.max_definitions == p.max_new_observations == 1
    assert p.max_operations == 24
    assert set(p.public_queries) == set(p.public_topic_sets) == set(p.query_requirements)
    for topics in p.public_topic_sets.values():
        plan = broad_plan(("SOURCE:EUROPEPMC",), topics=topics, limit=2)
        assert len(plan.concepts) == len(topics) >= 2
        assert plan.query_plan_version == "5.0.0"
    old = read_expansion_policy(
        json.loads(Path("configs/research/multimodal_definition_gate_v1.json").read_bytes())
    )
    assert old.schema_version == "campaign-definition-expansion/1.2.0"
    assert old.public_topics["lamina cribrosa"] == "oct"  # Historical mismatch retained.


def test_connector_passes_all_topics_to_original_search_owner(monkeypatch):
    from apps import knowledge_literature_tools as app
    from packages.evidence_reuse.definition_sources import LiteratureDefinitionConnector

    seen = []

    def search(*args, **kwargs):
        seen.append((args, kwargs))
        return dict(status="empty", results=[])

    monkeypatch.setattr(app, "search", search)
    c = LiteratureDefinitionConnector(None, None, {"public": ("oct", "progression")})
    assert c.search(query="public", key="new-query", limit=2) == []
    assert seen[0][0][2] == "glaucoma"
    assert seen[0][1]["topics"] == ("oct", "progression")


def test_query_arbitrary_text_cannot_bypass_original_public_catalog():
    value = json.loads(Path("configs/research/multimodal_definition_gate_v2.json").read_bytes())
    value["public_topic_sets"][value["public_queries"][0]] = ["oct", "patient-private-value"]
    with pytest.raises(ValueError, match="topic"):
        read_expansion_policy(value)


def test_gate_selection_uses_actual_revision_not_all_old_observations():
    from packages.multimodal_research.campaign_renewal import gate_selection

    revision = dict(added=["new"], replaced=[dict(id="changed")], rows=[dict(id="old")])
    c = SimpleNamespace(
        protocol=dict(definition_increment_only=True),
        targets=("old", "new"),
        base=SimpleNamespace(),
        accepted_observations=lambda: {"old": object(), "changed": object()},
    )
    selected = gate_selection(c, revision)
    assert selected == ("changed", "new")
    assert c.targets == ("new",)
    assert c.base.review_target_ids == selected
    assert revision["rows"] == [dict(id="old")]


def test_gate_projection_filter_keeps_old_facts_but_never_relabels_them():
    from packages.multimodal_research.campaign import review_observations

    values = {"old": object(), "new": object()}
    base = SimpleNamespace(review_target_ids=("new",))
    assert review_observations(base, values) == {"new": values["new"]}
    assert set(values) == {"old", "new"}
    assert review_observations(SimpleNamespace(), values) == values
    base.review_target_ids = ()
    assert review_observations(base, values) == {}


def targeted(e):
    e.policy = read_expansion_policy(
        json.loads(Path("configs/research/multimodal_definition_gate_v2.json").read_bytes())
    )
    return e


def test_public_attempt_caps_include_unknown_and_refuse_before_extra_send(tmp_path):
    from packages.llm_provider.provider import ProviderError
    from tests.evaluation_preparation.test_definition_expansion import fixture

    e = targeted(fixture(tmp_path))
    for kind in ("search", "fetch"):
        e.reserve_wire("SOURCE:PUBLIC", kind=kind)
        e.reserve_wire("SOURCE:PUBLIC", kind=kind)
        before = list(e.operations())
        with pytest.raises(ProviderError, match="public_request_budget"):
            e.reserve_wire("SOURCE:PUBLIC", kind=kind)
        assert e.operations() == before
    assert len(e.operations()) == 4  # Unknown attempts are never refunded.


def test_new_xml_policy_explicit_old_default_unchanged():
    from packages.evidence_reuse.definition_sources import KnowledgeDefinitionSources

    old = KnowledgeDefinitionSources(repository=None, store=None)
    assert old.index_options() == {}
    new = KnowledgeDefinitionSources(
        repository=None, store=None, material_policy="xml-scripts-whole-text/1.0.0"
    )
    assert new.index_options()["parse_policy"].version == "xml-scripts/1.0.0"
    assert new.index_options()["chunk_policy"].max_chars == 3000


def test_increment_scope_survives_saved_second_conversion_without_rescoring():
    from packages.evaluation_preparation.evidence_tree_v3 import (
        build,
        bundle,
        render_packet,
        unbundle,
    )
    from packages.grounding.models import digest
    from tests.evaluation_preparation.test_unified_view import report

    value = report()
    value["review_scope"] = dict(
        kind="definition_increment_only", explanation="增量分不是整例重新汇总 <safe>"
    )
    p = build(value, input_checksum=digest(value))
    page = render_packet(unbundle(bundle(p)))
    assert "增量分不是整例重新汇总 &lt;safe&gt;" in page
    assert p["scores"] == value["scores"]
