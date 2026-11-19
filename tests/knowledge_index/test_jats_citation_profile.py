"""Narrow JATS bibliography superscripts, never inferred numeric exponents."""

import pytest

from packages.knowledge_index.models import ParsePolicy
from packages.knowledge_index.parsers import parse, parser_spec

POLICY = ParsePolicy(version="xml-jats-citations/1.0.0")
BODY = (
    b'<article><body><p>Rim loss<sup><xref rid="R1" ref-type="bibr">1</xref>, '
    b'<xref rid="R2" ref-type="bibr">2</xref></sup>; area mm<sup>2</sup>.</p></body>'
    b'<back><ref-list><ref id="R1"><label>1</label><mixed-citation>First source.'
    b'</mixed-citation></ref><ref id="R2"><label>2</label><mixed-citation>Second source.'
    b"</mixed-citation></ref></ref-list></back></article>"
)


def test_explicit_citation_profile_preserves_order_and_numeric_script():
    result = parse(BODY, "application/xml", policy=POLICY)
    assert result.status == "parsed"
    assert result.document.parser_version == "2.3.0"
    assert parser_spec("application/xml", POLICY).version == "2.3.0"
    assert result.document.blocks[0].text == "Rim loss^{cite:1, 2}; area mm^{2}."
    assert (
        parse(BODY, "application/xml", policy=ParsePolicy(version="xml-jats/1.0.0")).reason
        == "numeric_markup_unsupported"
    )


@pytest.mark.parametrize(
    "raw",
    [
        BODY.replace(b'rid="R1"', b'rid="MISSING"'),
        BODY.replace(b'id="R2"', b'id="R1"'),
        BODY.replace(b'<ref id="R1">', b'<notref id="R1">').replace(b"</ref>", b"</notref>", 1),
        BODY.replace(b'ref-type="bibr"', b'ref-type="fig"', 1),
        BODY.replace(b">1</xref>", b"><italic>1</italic></xref>", 1),
        BODY.replace(b">1</xref>", b"> </xref>", 1),
        BODY.replace(
            b'<xref rid="R1"', b'<foreign:xref xmlns:foreign="urn:not-jats" rid="R1"', 1
        ).replace(b"</xref>", b"</foreign:xref>", 1),
        BODY.replace(
            b'<ref id="R1"', b'<foreign:ref xmlns:foreign="urn:not-jats" id="R1"', 1
        ).replace(b"</ref>", b"</foreign:ref>", 1),
        BODY.replace(b"<sup>2</sup>", b"<sup><sub>2</sub></sup>"),
        BODY.replace(b"<sup>2</sup>", b"<inline-formula>1+1</inline-formula>"),
    ],
)
def test_unsupported_nested_semantics_not_flattened(raw):
    result = parse(raw, "application/xml", policy=POLICY)
    assert result.document is None and result.reason == "numeric_markup_unsupported"


@pytest.mark.parametrize(
    "raw",
    [
        BODY.replace(b">1</xref>", b"><script>1</script></xref>", 1),
        b'<!DOCTYPE article [<!ENTITY x SYSTEM "file:///etc/passwd">]><article><p>&x;</p></article>',
    ],
)
def test_security_not_relaxed(raw):
    result = parse(raw, "application/xml", policy=POLICY)
    assert result.document is None and result.reason == "unsafe_document"


def test_xml_profile_does_not_accept_html():
    assert parse(b"<html><p>x</p></html>", "text/html", policy=POLICY).reason == "xml_scripts_mime"
