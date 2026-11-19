"""Offline wire-level checks of explicit adapter identity policies."""

from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import pytest
from pydantic import ValidationError

from packages.knowledge_connectors import (
    ConnectorError,
    ErrorCategory,
    FakeClock,
    PinnedHTTPSTransport,
    ProviderConfig,
    SafeHTTPClient,
    SourceLocator,
    StructuredQuery,
    load_fixture,
)
from packages.knowledge_connectors.connectors import CONNECTOR_TYPES
from packages.knowledge_connectors.http import WireResponse
from packages.knowledge_connectors.ports import digest
from tests.knowledge_connectors.test_connectors_04b1 import FIXTURES, fake
from tests.knowledge_connectors.test_http_04b1 import ScriptedTransport

EMAIL = "fixture@example.invalid"
KEY = "SYNTHETIC_REPAIR_KEY_NOT_REAL"
QUERY = StructuredQuery(text="synthetic fixture", page_size=1)


def setup(configurations, kind, changes, environment):
    config = fake(configurations, kind)._connector.config
    config = ProviderConfig.model_validate(config.model_copy(update=changes))
    tape = load_fixture(FIXTURES / (kind + ".json"))
    transport = ScriptedTransport(
        *(WireResponse(e.status, e.headers, e.body_text.encode()) for e in tape.exchanges),
        WireResponse(200, (("content-type", "application/json"),), b"{}"),
    )
    client = SafeHTTPClient(
        config,
        transport,
        resolver=lambda host: ("93.184.216.34",),
        clock=FakeClock(),
        environment=environment,
    )
    return CONNECTOR_TYPES[kind](config, client), transport


def exercise(connector):
    page = connector.search(QUERY)
    kind = connector.adapter
    locator = (
        SourceLocator(record_id="PMC999990001")
        if kind == "europe_pmc"
        else SourceLocator(url=page.records[0].url)
        if kind == "web_fallback"
        else SourceLocator(record_id=page.records[0].record_id)
    )
    document = connector.fetch(locator)
    health = connector.healthcheck()
    assert health.available
    return (*page.responses, document.response, health.response)


def test_pubmed_all_four_operations_have_tool_email_and_send_only_key(configurations):
    reads = []

    def environment(name):
        reads.append(name)
        return {"MMAGENT_NCBI_EMAIL": EMAIL, "MMAGENT_TEST_KEY": KEY}.get(name)

    connector, transport = setup(
        configurations,
        "pubmed",
        {"auth_mode": "query", "api_key_env": "MMAGENT_TEST_KEY"},
        environment,
    )
    assert reads == []  # No credential resolution during offline assembly.
    outputs = exercise(connector)
    assert tuple(r.metadata.operation for r in outputs) == (
        "search",
        "summary",
        "fetch",
        "healthcheck",
    )
    assert reads.count("MMAGENT_TEST_KEY") == 4
    assert connector.config.http.rate_limit_per_minute == 120
    for (request, _, _), output in zip(transport.requests, outputs, strict=True):
        parsed = urlsplit(request.url)
        params = dict(parse_qsl(parsed.query))
        assert params["tool"] == connector.config.pubmed_tool == "MMAgent-Knowledge-04B1"
        assert params["email"] == EMAIL and params["api_key"] == KEY
        assert "mailto" not in params
        public = [
            (k, v) for k, v in parse_qsl(parsed.query) if k not in ("tool", "email", "api_key")
        ]
        assert output.metadata.request_checksum == digest(
            urlunsplit(parsed._replace(query=urlencode(public))).encode()
        )
        assert output.metadata.config_checksum == connector.config.checksum()
        for value in (
            repr(request),
            repr(output),
            output.model_dump_json(),
            connector.config.model_dump_json(),
        ):
            assert KEY not in value and EMAIL not in value
    assert outputs[1].metadata.search_request_checksum == outputs[0].metadata.request_checksum


@pytest.mark.parametrize("configured", [False, True])
def test_crossref_contact_is_only_opt_in_mailto_on_search_fetch_health(configurations, configured):
    connector, transport = setup(
        configurations,
        "crossref",
        {"contact_email_env": "CROSSREF_CONTACT" if configured else None},
        lambda name: EMAIL,
    )
    exercise(connector)
    for request, _, _ in transport.requests:
        params = dict(parse_qsl(urlsplit(request.url).query))
        assert params.get("mailto") == (EMAIL if configured else None)
        assert not {"email", "tool", "api_key"} & params.keys()


@pytest.mark.parametrize("kind", ["europe_pmc", "guideline", "web_fallback"])
@pytest.mark.parametrize("source", ["absent", "environment"])
def test_other_adapters_never_infer_contact_parameters(configurations, kind, source):
    def forbidden(name):
        raise AssertionError("unrelated adapter must not resolve contact environment")

    changes = {} if source == "absent" else {"contact_email_env": "UNUSED_CONTACT"}
    connector, transport = setup(configurations, kind, changes, forbidden)
    exercise(connector)
    assert transport.requests
    for request, _, _ in transport.requests:
        assert not {"mailto", "email", "tool"} & dict(parse_qsl(urlsplit(request.url).query)).keys()


@pytest.mark.parametrize("email", [None, "", "bad", "bad\r\n@invalid.test", EMAIL])
def test_real_pubmed_send_fails_before_network_without_deployment_identity(configurations, email):
    with pytest.raises(ConnectorError) as caught:
        client = SafeHTTPClient(
            configurations["pubmed"], PinnedHTTPSTransport(), environment=lambda name: email
        )
        client._authenticate(client.config.health_endpoint)
    assert caught.value.category is ErrorCategory.CONFIGURATION
    if email:
        assert email not in str(caught.value) and email not in repr(caught.value)


@pytest.mark.parametrize("operation", ["search", "summary", "fetch", "healthcheck"])
def test_missing_contact_is_configuration_failure_before_send(configurations, operation):
    config = configurations["pubmed"]
    transport = ScriptedTransport()
    client = SafeHTTPClient(
        config,
        transport,
        resolver=lambda host: ("93.184.216.34",),
        clock=FakeClock(),
        environment=lambda name: None,
    )
    with pytest.raises(ConnectorError) as caught:
        client.get(
            config.health_endpoint,
            operation=operation,
            expected_mime=("application/json",),
            search_request_checksum=digest(b"search") if operation == "summary" else None,
        )
    assert caught.value.category is ErrorCategory.CONFIGURATION
    assert transport.requests == []


@pytest.mark.parametrize("tool", [None, "", "white space", "line\nbreak", 42])
def test_pubmed_tool_is_strict_and_required(configurations, tool):
    with pytest.raises(ValidationError):
        ProviderConfig.model_validate(
            configurations["pubmed"].model_copy(update={"pubmed_tool": tool})
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"auth_mode": "bearer", "api_key_env": "TEST_KEY"},
        {"auth_mode": "query", "api_key_env": "TEST_KEY", "api_key_parameter": "token"},
        {"contact_email_env": "bad env"},
        {"contact_email": EMAIL},
    ],
)
def test_pubmed_invalid_identity_configuration_rejected(configurations, changes):
    with pytest.raises(ValidationError):
        ProviderConfig.model_validate(configurations["pubmed"].model_copy(update=changes))


def test_pubmed_rate_tiers_and_identity_change_configuration_checksum(configurations):
    config = configurations["pubmed"]
    assert config.http.rate_limit_per_minute <= 180
    revised = ProviderConfig.model_validate(
        config.model_copy(update={"pubmed_tool": "MMAgent-Knowledge-04B1-v2"})
    )
    assert revised.checksum() != config.checksum()
    policy = config.http.model_copy(update={"rate_limit_per_minute": 181})
    with pytest.raises(ValidationError):
        ProviderConfig.model_validate(config.model_copy(update={"http": policy}))
    keyed = ProviderConfig.model_validate(
        config.model_copy(update={"auth_mode": "query", "api_key_env": "TEST_KEY", "http": policy})
    )
    with pytest.raises(ValidationError):
        ProviderConfig.model_validate(
            keyed.model_copy(
                update={"http": policy.model_copy(update={"rate_limit_per_minute": 601})}
            )
        )
