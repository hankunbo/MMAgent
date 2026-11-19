"""Traceable research quote alignment; no rewriting of sources or model output.

Only layout whitespace may differ. Negation, punctuation, numbers, units, case,
word order and all other characters must match a unique contiguous source span.
This does not authorize a source, proposal or clinical assertion.
"""

import re

from packages.object_store import checksum

LAYOUT = re.compile(r"[ \t\r\n\u00a0]+")
LITERAL_POLICY = "research-quote-policy/literal/1.0.0"
LAYOUT_POLICY = "research-quote-policy/layout-whitespace/1.0.0"


def quote_policy(owner):
    """Installed owner policy, never a field supplied by the generated proposal."""
    return check_quote_policy(getattr(owner, "source_quote_policy", LITERAL_POLICY))


def check_quote_policy(value):
    if value not in (LITERAL_POLICY, LAYOUT_POLICY):
        raise ValueError("research_quote_policy_not_supported")
    return value


def _layout_projection(text):
    chars, starts, ends = [], [], []
    cursor = 0
    for match in LAYOUT.finditer(text):
        for index in range(cursor, match.start()):
            chars.append(text[index])
            starts.append(index)
            ends.append(index + 1)
        chars.append(" ")
        starts.append(match.start())
        ends.append(match.end())
        cursor = match.end()
    for index in range(cursor, len(text)):
        chars.append(text[index])
        starts.append(index)
        ends.append(index + 1)
    return "".join(chars), starts, ends


def align_research_quote(quote, source_text, *, allow_layout=False):
    """Return a reproducible span or None; default retains literal-only behavior."""
    if not isinstance(quote, str) or not quote or not isinstance(source_text, str):
        return None
    index = source_text.find(quote)
    if index >= 0:
        start, end, mode = index, index + len(quote), "literal"
    elif allow_layout:
        projected, starts, ends = _layout_projection(source_text)
        quoted, _, _ = _layout_projection(quote)
        index = projected.find(quoted)
        if not quoted.strip() or index < 0 or projected.find(quoted, index + 1) >= 0:
            return None
        start, end, mode = starts[index], ends[index + len(quoted) - 1], "layout_whitespace"
        if _layout_projection(source_text[start:end])[0] != quoted:
            return None
    else:
        return None
    original = source_text[start:end]
    return dict(
        schema_version="research-quote-alignment/1.0.0",
        mode=mode,
        source_text_checksum=checksum(source_text.encode()),
        returned_quote_checksum=checksum(quote.encode()),
        offset_unit="unicode_codepoint",
        start=start,
        end=end,
        original_text=original,
        original_text_checksum=checksum(original.encode()),
        returned_quote_unchanged=True,
        medical_validity_not_implied=True,
    )
