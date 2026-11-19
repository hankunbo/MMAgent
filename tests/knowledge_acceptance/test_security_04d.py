"""Combined transport/parser/storage attacks, with independent finite expectations."""

import itertools
from pathlib import Path

import pytest

from packages.knowledge_connectors import ConnectorError, ErrorCategory, load_configuration
from packages.knowledge_index.parsers import parse
from packages.object_store import LocalObjectStore, checksum
from tests.knowledge_connectors.test_http_04b1 import client, get, response


def configurations():
    return {
        p.adapter: p
        for p in reversed(load_configuration(Path("configs/knowledge_sources.toml")).providers)
    }


@pytest.mark.parametrize("private", ["169.254.169.254", "::ffff:10.0.0.1", "100.64.0.1"])
def test_dns_answer_permutation_after_authenticated_redirect(private):
    secret = "synthetic-04d-secret-never-log"
    for addresses in itertools.permutations(("93.184.216.34", private)):
        answers = iter((("93.184.216.34",), addresses))
        http, transport = client(
            configurations(),
            response(307, headers=(("location", "/works/next"),)),
            response(),
            resolver=lambda host, answers=answers: next(answers),
            changes={"auth_mode": "bearer", "api_key_env": "MMAGENT_04D_SYNTHETIC_KEY"},
            environment=lambda name: secret,
        )
        with pytest.raises(ConnectorError) as caught:
            get(http)
        assert caught.value.category is ErrorCategory.SECURITY
        assert len(transport.requests) == 1
        assert secret not in repr(transport.requests)
        assert secret not in str(caught.value)
        assert secret not in caught.value.metadata.model_dump_json()


@pytest.mark.parametrize(
    "location",
    [
        "//169.254.169.254/latest/meta-data/",
        "https://api.crossref.org@127.0.0.1/",
        "https://api.crossref.org.evil.invalid/",
        "https://api.crossref.org/next?api_key=synthetic-injected",
        "https://api.crossref.org/next?email=synthetic%40example.invalid",
    ],
)
def test_authenticated_redirect_never_forwards_to_an_unapproved_target(location):
    secret = "synthetic-04d-auth-secret"
    http, transport = client(
        configurations(),
        response(308, headers=(("location", location),)),
        response(),
        changes={"auth_mode": "bearer", "api_key_env": "MMAGENT_04D_SYNTHETIC_KEY"},
        environment=lambda name: secret,
    )
    with pytest.raises(ConnectorError) as caught:
        get(http)
    assert caught.value.category is ErrorCategory.SECURITY
    assert len(transport.requests) == 1
    assert secret not in repr(transport.requests) + str(caught.value)


@pytest.mark.parametrize("mime", ["application/xml", "application/pdf", "text/html"])
def test_oversized_document_rejected_before_parser(mime):
    result = parse(b"x" * 2_000_001, mime)
    assert (result.status, result.reason, result.document) == ("failed", "resource_limit", None)


@pytest.mark.parametrize(
    "body,mime,reason",
    [
        (b"<html><p>synthetic</p></html>", "application/pdf", "wrong_mime"),
        (b"%PDF-1.7 synthetic", "application/xml", "wrong_mime"),
        (b"<article><p>synthetic</p></article>", "text/html", "wrong_mime"),
        (b"PK\x03\x04synthetic archive", "application/xml", "unsafe_document"),
    ],
)
def test_mime_and_container_attacks_have_no_medical_output(body, mime, reason):
    result = parse(body, mime)
    assert (result.status, result.reason, result.document) == ("failed", reason, None)


@pytest.mark.parametrize("parameter_entity", [False, True])
def test_xxe_cannot_read_a_local_sentinel(tmp_path, parameter_entity):
    sentinel = tmp_path / "synthetic-secret.txt"
    sentinel.write_text("synthetic-04d-private-sentinel")
    if parameter_entity:
        declaration = f'<!ENTITY % remote SYSTEM "{sentinel.as_uri()}">%remote;'
        text = "synthetic"
    else:
        declaration = f'<!ENTITY remote SYSTEM "{sentinel.as_uri()}">'
        text = "&remote;"
    result = parse(
        f"<!DOCTYPE article [{declaration}]><article><p>{text}</p></article>".encode(),
        "application/xml",
    )
    assert result.status == "failed" and result.reason == "unsafe_document"
    assert result.document is None
    assert sentinel.read_text() not in result.model_dump_json()


@pytest.mark.parametrize(
    "uri",
    [
        "object://knowledge/objects/../sentinel",
        "object://knowledge/objects/%2e%2e/sentinel",
        "object://knowledge/objects/" + "a" * 32 + "/../../sentinel",
        "file:///synthetic/sentinel",
    ],
)
def test_path_traversal_cannot_read_or_delete_objects(tmp_path, uri):
    store = LocalObjectStore(tmp_path / "store")
    owned = store.promote(store.stage(b"synthetic intact"), checksum(b"synthetic intact"))
    for operation in (store.read, store.discard, lambda value: store.read_bounded(value, 128)):
        with pytest.raises(ValueError, match="invalid managed object URI"):
            operation(uri)
    assert store.read(owned) == b"synthetic intact"


def test_one_byte_staging_corruption_cannot_be_promoted(tmp_path):
    store = LocalObjectStore(tmp_path / "store")
    original = b"synthetic immutable evidence"
    uri = store.stage(original)
    path = store.root / uri.removeprefix("object://knowledge/")
    path.write_bytes(bytes([original[0] ^ 1]) + original[1:])
    with pytest.raises(ValueError, match="checksum mismatch"):
        store.promote(uri, checksum(original))
    assert not tuple((store.root / "objects").iterdir())
    store.discard(uri)
