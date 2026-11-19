import pytest

from apps.glaucoma_egs_content import CONFIG
from packages.knowledge_connectors import (
    ConnectorError,
    FakeClock,
    SafeHTTPClient,
    StructuredQuery,
    WebSearchFallbackConnector,
    load_configuration,
)
from packages.knowledge_connectors.http import WireResponse
from tests.knowledge_connectors.test_http_04b1 import ScriptedTransport


def connector(status=200):
    config = load_configuration(CONFIG).providers[0]
    transport = ScriptedTransport(
        WireResponse(status, (("content-type", "text/html"),), b"<p>Synthetic listing</p>")
    )
    client = SafeHTTPClient(
        config, transport, clock=FakeClock(), resolver=lambda _: ("93.184.216.34",)
    )
    return WebSearchFallbackConnector(config, client), transport


def test_education_catalog_not_guideline_or_remote_ranking():
    c, transport = connector()
    page = c.search(StructuredQuery(text='("glaucoma") ("research cohort")', page_size=10))
    assert len(page.records) == 4
    assert all(r.source_type == "WEB" for r in page.records)
    assert c.capabilities().discovery_mode == "curated_catalog"
    assert len(page.responses) == len(transport.requests) == 1
    assert page.responses[0].metadata.operation == "search"
    assert transport.requests[0][0].url == "https://www.eugs.org/"


def test_unmatched_query_empty_not_medical_negative():
    c, _ = connector()
    assert c.search(StructuredQuery(text="unmatchedxyz", page_size=10)).records == ()


def test_remote_listing_failure_not_successful_empty():
    c, _ = connector(500)
    with pytest.raises(ConnectorError):
        c.search(StructuredQuery(text="glaucoma", page_size=10))
