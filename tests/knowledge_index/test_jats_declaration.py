"""Inert JATS PUBLIC declaration does not grant DTD or external entity access."""

import pytest

from packages.knowledge_index.models import ParsePolicy
from packages.knowledge_index.parsers import parse

DECL = b'<!DOCTYPE article PUBLIC "-//NLM//DTD JATS v1.4//EN" "JATS-archivearticle1-4.dtd">'


def new(raw):
    return parse(raw, "application/xml", policy=ParsePolicy(version="xml-jats/1.0.0"))


def test_jats_declaration_explicit_version_old_bytes_still_rejected():
    raw = DECL + b"<article><p>Area 2 mm<sup>2</sup>.</p></article>"
    assert parse(raw, "application/xml").reason == "unsafe_document"
    result = new(raw)
    assert result.status == "parsed" and result.document.parser_version == "2.2.0"
    assert result.document.blocks[0].text == "Area 2 mm^{2}."


@pytest.mark.parametrize(
    "raw",
    [
        b'<!DOCTYPE article [<!ENTITY x "boom">]><article><p>&x;</p></article>',
        b'<!DOCTYPE article [<!ATTLIST p a CDATA "changed">]><article><p>x</p></article>',
        b'<!DOCTYPE article SYSTEM "file:///etc/passwd"><article><p>x</p></article>',
        DECL.replace(b"JATS-archivearticle1-4.dtd", b"https://example.org/evil.dtd")
        + b"<article/>",
        DECL + b"<article><p><script>x</script></p></article>",
    ],
)
def test_entities_internal_defaults_external_urls_and_scripts_rejected(raw):
    assert new(raw).reason == "unsafe_document"
