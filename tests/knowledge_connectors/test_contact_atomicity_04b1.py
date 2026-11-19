"""Second repair: production policy tested with synthetic, never delivered identities."""

import json
import traceback
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from uuid import uuid4

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
    load_configuration,
)
from packages.knowledge_connectors.connectors import CONNECTOR_TYPES
from packages.knowledge_connectors.http import WireResponse
from packages.knowledge_connectors.ports import digest
from tests.knowledge_connectors.test_provider_identity_repair_04b1 import exercise, setup

RESERVED = "fixture@example.invalid"
KEY = "SYNTHETIC_ATOMIC_KEY_NOT_REAL"


@pytest.fixture
def email():
    # Synthetic non-reserved syntax exercises production validation, not a real mailbox.
    return "synthetic@deployment-" + uuid4().hex + ".org"


class OutboundTransport:
    """An undeclared custom transport must receive the production policy by default."""

    def __init__(self):
        self.requests = []

    def send(self, request, ip, max_bytes):
        self.requests.append(request)
        return WireResponse(200, (("content-type", "application/json"),), b"{}")


class WrappedOutboundTransport:
    def __init__(self):
        self.inner = OutboundTransport()

    @property
    def requests(self):
        return self.inner.requests

    def send(self, *args):
        return self.inner.send(*args)


def configured(configurations, kind="pubmed", **changes):
    return ProviderConfig.model_validate(
        configurations[kind].model_copy(update={"contact_email_env": "CONTACT_EMAIL", **changes})
    )


def client(config, transport, environment):
    return SafeHTTPClient(
        config,
        transport,
        environment=environment,
        resolver=lambda host: ("93.184.216.34",),
        clock=FakeClock(),
    )


def get(http):
    return http.get(
        http.config.health_endpoint, operation="healthcheck", expected_mime=("application/json",)
    )


@pytest.mark.parametrize("kind", ["pubmed", "crossref"])
@pytest.mark.parametrize("entry", ["dict", "json", "copy", "toml"])
def test_direct_contact_entry_is_strictly_rejected(configurations, email, kind, entry, tmp_path):
    config = configurations[kind]
    values = config.model_dump(mode="json") | {"contact_email_env": None, "contact_email": email}
    with pytest.raises(ValidationError) as caught:
        if entry == "json":
            ProviderConfig.model_validate_json(json.dumps(values))
        elif entry == "copy":
            ProviderConfig.model_validate(
                config.model_copy(update={"contact_email_env": None, "contact_email": email})
            )
        elif entry == "toml":
            from pathlib import Path

            text = Path("configs/knowledge_sources.toml").read_text()
            text = text.replace(
                'adapter = "' + kind + '"',
                'adapter = "' + kind + '"\ncontact_email = "' + email + '"',
            )
            path = tmp_path / "bad.toml"
            path.write_text(text)
            load_configuration(path)
        else:
            values = config.model_dump() | {"contact_email_env": None, "contact_email": email}
            ProviderConfig.model_validate(values)
    assert email not in str(caught.value) and email not in repr(caught.value)


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("kind", ["pubmed", "crossref"])
def test_dynamic_identity_is_read_once_and_exact_validated_value_is_sent(
    configurations, email, reverse, kind
):
    reads = []
    values = iter([RESERVED, email] if reverse else [email, RESERVED])

    def environment(name):
        reads.append(name)
        return next(values)

    transport = OutboundTransport()
    http = client(configured(configurations, kind), transport, environment)
    assert not reads
    if reverse:
        with pytest.raises(ConnectorError) as caught:
            get(http)
        assert caught.value.category is ErrorCategory.CONFIGURATION
        assert not transport.requests
    else:
        get(http)
        param = "email" if kind == "pubmed" else "mailto"
        assert dict(parse_qsl(urlsplit(transport.requests[0].url).query))[param] == email
    assert reads == ["CONTACT_EMAIL"]


def test_pinned_constructor_does_not_consume_dynamic_identity(configurations, email):
    reads = []
    values = iter([email, email, RESERVED])

    def environment(name):
        reads.append(name)
        return next(values)

    http = client(configured(configurations), PinnedHTTPSTransport(), environment)
    assert reads == []
    request = http._authenticate(http.config.health_endpoint)
    assert reads == ["CONTACT_EMAIL"]
    assert dict(parse_qsl(urlsplit(request.url).query))["email"] == email


@pytest.mark.parametrize("transport_type", [OutboundTransport, WrappedOutboundTransport])
@pytest.mark.parametrize(
    "value",
    [None, "broken", RESERVED, "fixture@sub.example.org"],
    ids=["missing", "malformed", "reserved", "example"],
)
@pytest.mark.parametrize("kind", ["pubmed", "crossref"])
def test_custom_outbound_cannot_bypass_identity_validation(
    configurations, transport_type, value, kind, monkeypatch
):
    import packages.knowledge_connectors.http as module

    transport = transport_type()
    http = client(configured(configurations, kind), transport, lambda name: value)
    created = []
    original = module.WireRequest

    def wire(*args):
        created.append(True)
        return original(*args)

    monkeypatch.setattr(module, "WireRequest", wire)
    with pytest.raises(ConnectorError) as caught:
        get(http)
    assert caught.value.category is ErrorCategory.CONFIGURATION
    assert not transport.requests and not created


@pytest.mark.parametrize("kind", ["pubmed", "crossref"])
def test_real_policy_all_adapter_operations_are_atomic_and_private(
    configurations, email, kind, caplog, monkeypatch
):
    reads = []

    def environment(name):
        reads.append(name)
        return KEY if name == "TEST_KEY" else email

    connector, transport = setup(
        configurations,
        kind,
        {"contact_email_env": "CONTACT_EMAIL", "api_key_env": "TEST_KEY", "auth_mode": "query"},
        environment,
    )
    transport.contact_mode = "production"
    assert not reads
    materials = []
    import packages.knowledge_connectors.ports as ports

    original = ports.digest

    def capture(data):
        materials.append(data)
        return original(data)

    monkeypatch.setattr(ports, "digest", capture)
    outputs = exercise(connector)
    assert reads == [name for _ in outputs for name in ("CONTACT_EMAIL", "TEST_KEY")]
    for (request, _, _), output in zip(transport.requests, outputs, strict=True):
        parsed = urlsplit(request.url)
        params = dict(parse_qsl(parsed.query))
        assert params["email" if kind == "pubmed" else "mailto"] == email
        assert params["api_key"] == KEY
        public = [
            (k, v)
            for k, v in parse_qsl(parsed.query)
            if k not in ("tool", "email", "mailto", "api_key")
        ]
        assert output.metadata.request_checksum == digest(
            urlunsplit(parsed._replace(query=urlencode(public))).encode()
        )
        for value in (
            repr(request),
            repr(output),
            str(output.model_dump()),
            output.model_dump_json(),
            str(connector.config.model_dump()),
            connector.config.model_dump_json(),
            repr(connector.config),
            caplog.text,
        ):
            assert email not in value and KEY not in value
    assert materials
    assert all(email.encode() not in data and KEY.encode() not in data for data in materials)


@pytest.mark.parametrize("operation", ["search", "fetch", "healthcheck"])
@pytest.mark.parametrize("fault", ["contact", "key"])
def test_environment_exceptions_are_sanitized_before_wire_and_not_health_unavailability(
    configurations, email, operation, fault, caplog, monkeypatch
):
    import packages.knowledge_connectors.http as module

    def environment(name):
        if (name == "CONTACT_EMAIL") == (fault == "contact"):
            raise OSError(email + KEY)
        return email

    config = configured(configurations, api_key_env="TEST_KEY", auth_mode="query")
    transport = OutboundTransport()
    http = client(config, transport, environment)
    connector = CONNECTOR_TYPES["pubmed"](config, http)
    created = []
    original = module.WireRequest

    def wire(*args):
        created.append(True)
        return original(*args)

    monkeypatch.setattr(module, "WireRequest", wire)
    with pytest.raises(ConnectorError) as caught:
        if operation == "search":
            connector.search(StructuredQuery(text="synthetic"))
        elif operation == "fetch":
            connector.fetch(SourceLocator(record_id="999990001"))
        else:
            connector.healthcheck()
    assert caught.value.category is (
        ErrorCategory.CONFIGURATION if fault == "contact" else ErrorCategory.AUTHENTICATION
    )
    assert not transport.requests and not created
    public = "".join(traceback.format_exception(caught.value)) + caplog.text
    assert email not in public and KEY not in public


def test_validated_contact_is_immutable_and_has_no_public_serialization(configurations, email):
    import dataclasses
    import pickle

    http = client(configured(configurations), OutboundTransport(), lambda name: email)
    contact = http._validated_contact(http.config)
    assert email not in repr(contact)
    assert not hasattr(contact, "model_dump") and not hasattr(contact, "model_dump_json")
    for serialize in (json.dumps, pickle.dumps, vars, dataclasses.asdict):
        with pytest.raises(TypeError):
            serialize(contact)
    with pytest.raises(AttributeError):
        contact._ValidatedContact__parameters = ()
    with pytest.raises(AttributeError):
        del contact._ValidatedContact__parameters


@pytest.mark.parametrize("kind", ["pubmed", "crossref"])
def test_missing_environment_reference_is_adapter_specific(configurations, kind):
    def forbidden(name):
        raise AssertionError("no reference means no environment read")

    transport = OutboundTransport()
    config = configured(configurations, kind, contact_email_env=None)
    http = client(config, transport, forbidden)
    if kind == "pubmed":
        with pytest.raises(ConnectorError) as caught:
            get(http)
        assert caught.value.category is ErrorCategory.CONFIGURATION
        assert not transport.requests
    else:
        get(http)
        assert (
            not {"email", "mailto", "tool"}
            & dict(parse_qsl(urlsplit(transport.requests[0].url).query)).keys()
        )


@pytest.mark.parametrize("mode", ["synthetic", "production", "typo", None])
def test_synthetic_exception_requires_explicit_transport_declaration(configurations, mode):
    transport = OutboundTransport()
    transport.contact_mode = mode
    http = client(configured(configurations), transport, lambda name: RESERVED)
    if mode == "synthetic":
        get(http)
        assert len(transport.requests) == 1
    else:
        with pytest.raises(ConnectorError) as caught:
            get(http)
        assert caught.value.category is ErrorCategory.CONFIGURATION
        assert not transport.requests


@pytest.mark.parametrize("mechanism", ["retry", "redirect"])
def test_every_retry_or_redirect_validates_its_own_single_read(configurations, email, mechanism):
    values = iter([email, RESERVED])
    reads = []

    def environment(name):
        reads.append(name)
        return next(values)

    class Transport(OutboundTransport):
        def send(self, request, ip, max_bytes):
            self.requests.append(request)
            return WireResponse(
                503 if mechanism == "retry" else 302,
                (("content-type", "application/json"), ("location", "/next")),
                b"{}",
            )

    transport = Transport()
    http = client(configured(configurations), transport, environment)
    with pytest.raises(ConnectorError) as caught:
        get(http)
    assert caught.value.category is ErrorCategory.CONFIGURATION
    assert reads == ["CONTACT_EMAIL", "CONTACT_EMAIL"] and len(transport.requests) == 1
    assert dict(parse_qsl(urlsplit(transport.requests[0].url).query))["email"] == email


@pytest.mark.parametrize(
    "bad", ["a..b@domain.org", "a@bad..org", "a@-bad.org", "a.@domain.org", "a@bad-.org", 42]
)
def test_invalid_contact_syntax_fails_in_both_modes(configurations, bad):
    for mode in ("production", "synthetic"):
        transport = OutboundTransport()
        transport.contact_mode = mode
        http = client(configured(configurations), transport, lambda name: bad)
        with pytest.raises(ConnectorError) as caught:
            get(http)
        assert caught.value.category is ErrorCategory.CONFIGURATION
        assert not transport.requests


@pytest.mark.parametrize("parameter", ["email", "tool", "mailto"])
def test_api_key_cannot_be_used_as_a_contact_parameter(configurations, parameter):
    with pytest.raises(ValidationError):
        configured(
            configurations,
            "crossref",
            api_key_env="KEY",
            auth_mode="query",
            api_key_parameter=parameter,
        )


def test_request_encoding_failure_is_private_and_precedes_wire(
    configurations, email, monkeypatch, caplog
):
    import packages.knowledge_connectors.http as module

    transport = OutboundTransport()
    http = client(configured(configurations), transport, lambda name: email)
    created = []

    def fail(*args, **kwargs):
        raise ValueError(email + KEY)

    monkeypatch.setattr(module, "urlencode", fail)
    monkeypatch.setattr(module, "WireRequest", lambda *args: created.append(True))
    with pytest.raises(ConnectorError) as caught:
        get(http)
    assert caught.value.category is ErrorCategory.SECURITY
    assert not created and not transport.requests
    public = "".join(traceback.format_exception(caught.value)) + caplog.text
    assert email not in public and KEY not in public
