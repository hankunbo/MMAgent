"""Explicit XML inline-script fidelity, not suppression of unsupported maths."""

import pytest

from packages.knowledge_index.models import ParsePolicy
from packages.knowledge_index.parsers import parse


def parse_new(body):
    return parse(body, "application/xml", policy=ParsePolicy(version="xml-scripts/1.0.0"))


def test_opt_in_retains_exponents_subscripts_and_default_rejection():
    from packages.knowledge_index.parsers import parser_spec

    raw = b"<article><p>Area 2 mm<sup>2</sup>; R<sub>n</sub>; p &lt; 0.05</p></article>"
    old = parse(raw, "application/xml")
    assert old.reason == "numeric_markup_unsupported"
    result = parse_new(raw)
    assert result.status == "parsed"
    assert result.document.parser_version == "2.1.0"
    assert (
        parser_spec("application/xml", ParsePolicy(version="xml-scripts/1.0.0")).version == "2.1.0"
    )
    assert result.document.fidelity_policy == "text-fidelity/2.1.0"
    assert result.document.blocks[0].text == "Area 2 mm^{2}; R_{n}; p < 0.05"
    assert raw == b"<article><p>Area 2 mm<sup>2</sup>; R<sub>n</sub>; p &lt; 0.05</p></article>"


def test_grid_preserves_script_and_exact_offsets():
    r = parse_new(b"<article><table><tr><th>mm<sup>2</sup></th><td>0.5</td></tr></table></article>")
    assert r.status == "parsed"
    b = r.document.blocks[0]
    assert b.table.cells[0].text == "mm^{2}"
    for c in b.table.cells:
        assert b.text[c.span_start : c.span_end] == c.text


@pytest.mark.parametrize(
    "body,reason",
    [
        (b"<article><p>x<sup><sub>2</sub></sup></p></article>", "numeric_markup_unsupported"),
        (
            b"<article><p>x<inline-formula>1+1</inline-formula></p></article>",
            "numeric_markup_unsupported",
        ),
        (
            b'<!DOCTYPE article [<!ENTITY x SYSTEM "file:///etc/passwd">]><article><p>&x;</p></article>',
            "unsafe_document",
        ),
        (b"<article><p>x<sup><script>2</script></sup></p></article>", "unsafe_document"),
    ],
)
def test_no_formula_flattening_or_external_entities(body, reason):
    result = parse_new(body)
    assert result.document is None and result.reason == reason
