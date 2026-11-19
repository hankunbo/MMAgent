"""Owner-material literal quote choices; no text normalization or medical scoring."""

import re
from copy import deepcopy

from packages.grounding.models import canonical, digest


def choices(material):
    rows = []
    for source in material.sources:
        if source.reference.code.startswith("POLICY:") or source.reference.role != "source":
            continue
        text = source.text
        for match in re.finditer(r"[^.!?]+[.!?](?:\d+(?:[-, ]\d+)*)?|[^.!?]+$", text):
            raw = match.group().strip()
            # Intraword PDF line breaks are not silently repaired or offered for transcription.
            if len(raw) < 16 or re.search(r"[A-Za-z]\n[A-Za-z]", raw):
                continue
            if len(raw) > 600:
                raw = raw[:600].rsplit(" ", 1)[0]
            start = text.find(raw, match.start(), match.end())
            if start < 0:
                continue
            rows.append(
                dict(
                    source_code=source.reference.code,
                    quote=raw,
                    start=start,
                    end=start + len(raw),
                    source_checksum=source.reference.checksum,
                )
            )
            break
    return rows


def project_wire(body, config, role, projected):
    if projected is None and next(r for r in config.roles if r.role == role).prompt_version in {
        "definition-draft-2.0.0",
        "definition-draft-2.1.0",
        "definition-draft-2.2.0",
        "definition-draft-2.3.0",
        "definition-draft-3.0.0",
        "definition-draft-4.0.0",
        "definition-draft-4.1.0",
        "definition-draft-5.0.0",
        "definition-draft-6.0.0",
        "definition-draft-6.1.0",
        "definition-draft-6.2.0",
        "definition-draft-6.3.0",
        "definition-draft-6.4.0",
        "definition-draft-6.5.0",
        "definition-draft-6.6.0",
        "definition-draft-6.7.0",
        "definition-draft-6.8.0",
    }:
        return body  # Dedicated definition output schema; never SourceProposal quote selection.
    if (
        role != "source"
        or not getattr(config, "source_quote_selection", None)
        or projected.material.scope.case_code == "CASE:PUBLIC_PROBE"
    ):
        return body
    rows = choices(projected.material)
    if not rows:
        raise ValueError("no_literal_source_choices")
    result = deepcopy(body)
    result["messages"][1]["content"].append(
        dict(
            type="text",
            text=canonical(
                dict(
                    SOURCE_LITERAL_CHOICES=rows,
                    instruction=(
                        "Select one source_code/quote pair from this table. "
                        "Copy it verbatim, including whitespace and punctuation. "
                        "Do not fix PDF spelling, merge words, combine passages "
                        "or append citation numbers. Offsets are audit data; "
                        "return the original SourceProposal schema only, without extra fields."
                    ),
                    choices_checksum=digest(rows),
                )
            ).decode(),
        )
    )
    return result
