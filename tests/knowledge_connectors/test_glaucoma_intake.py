"""Curated discovery is not HTTP retrieval, archival or publication."""

import json

from apps.glaucoma_knowledge_intake import CONFIG, discover
from packages.knowledge_connectors import load_configuration
from packages.knowledge_search.privacy import safe_url


def test_ng81_catalog_uses_existing_strict_configuration():
    config = load_configuration(CONFIG)
    assert len(config.providers) == 1
    provider = config.providers[0]
    assert provider.adapter == "guideline"
    assert all(safe_url(x.url) == x.url for x in provider.catalog)
    assert provider.additional_institution is False


def test_real_catalog_search_is_explicitly_not_archival():
    result = discover()
    assert len(result["page"]["records"]) == 2
    assert result["page"]["responses"] == []
    assert not result["database_written"] and not result["published"]
    assert result["fetch"] == []
    assert all(x["source_type"] == "GUIDELINE" for x in result["page"]["records"])


def test_twelve_topics_keep_original_node_identities_and_no_fake_source_refs():
    work = json.loads(CONFIG.with_name("knowledge_glaucoma_topics.json").read_text())
    original = json.loads(
        (
            CONFIG.parent.parent / "docs/acceptance/07-k-layered-needs-v1/knowledge-draft.json"
        ).read_text()
    )
    actual_codes = {r["code"] for r in original["records"] if r["kind"] == "node"}
    assert len(work["topics"]) == len({t["node"] for t in work["topics"]}) == 12
    assert all(work["node_namespace"] + t["node"] in actual_codes for t in work["topics"])
    assert all(set(t["source_leads"]) <= work["sources"].keys() for t in work["topics"])
    assert work["source_refs"] == []
    assert work["sources"]["NEI"]["type"] == "WEB"
    assert work["sources"]["NEI"]["connector"] is None
