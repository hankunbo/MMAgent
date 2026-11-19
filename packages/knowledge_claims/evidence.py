"""Validate exact current derivations, canonical slices and bounded table grids."""

import json
import re
from dataclasses import dataclass

from packages.knowledge_index.chunking import chunk as rechunk
from packages.knowledge_index.models import (
    FIDELITY_VERSION,
    PARSER_VERSION,
    ParsedDocument,
    ParsePolicy,
    canonical,
    digest,
    parse_chunk_policy,
)
from packages.knowledge_index.parsers import parser_spec
from packages.object_store import checksum

from .models import DraftOutput
from .policy import PolicyRejected, occurrences
from .policy import polarity as supported_polarity


class Rejected(ValueError):
    """Safe finite reason only; no provider/source text in exception messages."""


@dataclass(frozen=True)
class Evidence:
    chunk: object
    document: object
    source: object
    parsed: ParsedDocument
    block: object
    text: str
    receipt: dict
    original: bytes = b""
    source_scope: object = None


def require(ok, reason):
    if not ok:
        raise Rejected(reason)


class _DerivationRead:
    """Pure computation reuse for one bounded repository read, never qualification.

    Keys include exact bytes/policy, not merely caller-supplied checksums. Original
    objects and current SQL records are still read and checked for each member.
    The repository constructs this locally; it must not survive a batch call.
    """

    def __init__(self):
        self.documents = {}
        self.chunks = {}

    def parse(self, body):
        if body not in self.documents:
            parsed = ParsedDocument.model_validate_json(body)
            require(canonical(parsed) == body, "parse_canonical")
            cursor = 0
            for ordinal, block in enumerate(parsed.blocks, 1):
                require(
                    block.position.ordinal == ordinal
                    and block.position.span_start == cursor
                    and block.position.span_end == cursor + len(block.text),
                    "canonical_span",
                )
                cursor = block.position.span_end + 1
            self.documents[body] = parsed
        return self.documents[body]

    def derive(self, body, parsed, policy):
        key = (body, canonical(policy))
        if key not in self.chunks:
            self.chunks[key] = rechunk(parsed, policy)
        return self.chunks[key]


def load_evidence(
    hit,
    receipt,
    source,
    document,
    chunk,
    store,
    *,
    research_scripts=False,
    _derivations=None,
    material_profile="claim-evidence/1.0.0",
):
    """Records and receipt are read by the trusted repository, never supplied by a model."""
    require(
        material_profile in {"claim-evidence/1.0.0", "research-definition-material/1.1.0"},
        "evidence_material_profile",
    )
    d, c = document.derivation, chunk.derivation
    require(d is not None and c is not None, "legacy_fidelity")
    cfg = receipt.get("derivation_config", {})
    policy_version = cfg.get("parse_policy", {}).get("version")
    scripts = policy_version in {"xml-scripts/1.0.0", "xml-jats/1.0.0", "xml-jats-citations/1.0.0"}
    require(not scripts or research_scripts, "legacy_fidelity")
    fidelity = "text-fidelity/2.1.0" if scripts else FIDELITY_VERSION
    version = (
        "2.3.0"
        if policy_version == "xml-jats-citations/1.0.0"
        else "2.2.0"
        if policy_version == "xml-jats/1.0.0"
        else "2.1.0"
        if scripts
        else PARSER_VERSION
    )
    require(not scripts or source.media_type in {"application/xml", "text/xml"}, "parser_version")
    require(
        d.fidelity_policy == c.fidelity_policy == cfg.get("fidelity_policy") == fidelity,
        "legacy_fidelity",
    )
    require(
        document.parser_version
        == chunk.parser_version
        == d.parser_version
        == c.parser_version
        == version,
        "parser_version",
    )
    try:
        pp = ParsePolicy.model_validate_json(json.dumps(cfg["parse_policy"]))
        cp = parse_chunk_policy(cfg["chunk_policy"])
        spec = parser_spec(source.media_type, pp)
    except (KeyError, ValueError, StopIteration):
        raise Rejected("derivation_config") from None
    require(re.fullmatch(r"2\.[0-9]+\.[0-9]+", cp.version), "chunk_policy_version")
    require(spec is not None and spec.name == d.parser, "derivation_config")
    require(cfg["parser"] == json.loads(canonical(vars(spec))), "derivation_config")
    require(
        d.parser == c.parser
        and d.parse_policy_checksum == c.parse_policy_checksum == digest(pp)
        and d.chunk_policy_checksum == c.chunk_policy_checksum == digest(cp)
        and chunk.chunking_policy_version == cp.version
        and d.original_checksum == c.original_checksum == source.content_checksum
        and d.parse_uri == c.parse_uri == document.object_uri
        and d.parse_checksum == c.parse_checksum == document.content_checksum
        and d.language == c.language
        and d.targets == c.targets
        and d.table_evidence == c.table_evidence
        and d.position is None,
        "derivation_binding",
    )

    def wire(r):
        return r.model_dump(mode="json")

    require(
        wire(chunk.document_ref) == wire(document.reference()) == receipt["document_ref"]
        and wire(chunk.source_version_ref)
        == wire(document.source_version_ref)
        == wire(source.reference())
        == receipt["source_version_ref"]
        and wire(source.source_ref) == receipt["source_ref"]
        and wire(chunk.reference()) in receipt["chunk_refs"],
        "source_binding",
    )
    require(
        all(
            hit[k] == expected
            for k, expected in (
                ("chunk_ref", wire(chunk.reference())),
                ("document_ref", wire(document.reference())),
                ("source_version_ref", wire(source.reference())),
                ("source_ref", wire(source.source_ref)),
                ("fidelity_policy", fidelity),
                ("position", wire(c.position)),
                (
                    "parse_ref",
                    {
                        "uri": d.parse_uri,
                        "checksum": d.parse_checksum,
                        "block_ordinal": c.position.ordinal,
                    },
                ),
            )
        ),
        "hit_binding",
    )
    require(receipt["config_checksum"] == digest(receipt["config"]), "derivation_config")
    require(
        receipt["input_checksum"]
        == digest(
            {
                "source_checksum": source.content_checksum,
                "parse_checksum": d.parse_checksum,
                "chunks": receipt["chunk_refs"],
            }
        ),
        "derivation_binding",
    )
    require(
        receipt["key"]
        == hit["index_key"]
        == digest(
            {
                "source": wire(source.reference()),
                "parser": cfg["parser"],
                "parse_policy": wire(pp),
                "chunk_policy": wire(cp),
                "config": receipt["config"],
                "applicability": [wire(r) for r in document.applicability_refs],
                "targets": d.targets,
            }
        ),
        "derivation_binding",
    )
    original = store.read_bounded(source.object_uri, pp.max_bytes)
    require(
        len(original) == source.size_bytes and checksum(original) == d.original_checksum,
        "original_checksum",
    )
    body = store.read_bounded(hit["parse_ref"]["uri"], 4_000_000)
    require(
        len(body) == document.size_bytes and checksum(body) == d.parse_checksum, "parse_checksum"
    )
    computations = _derivations if _derivations is not None else _DerivationRead()
    parsed = computations.parse(body)
    require(
        parsed.fidelity_policy == fidelity
        and parsed.parser_version == version
        and parsed.parser == d.parser
        and parsed.policy_checksum == digest(pp)
        and parsed.language == d.language,
        "parse_binding",
    )
    matches = [b for b in parsed.blocks if b.position.ordinal == c.position.ordinal]
    require(len(matches) == 1, "block_binding")
    block = matches[0]
    derived = computations.derive(body, parsed, cp)
    index = receipt["chunk_refs"].index(wire(chunk.reference()))
    require(len(derived) == len(receipt["chunk_refs"]), "chunk_binding")
    require(
        c.previous_code == (receipt["chunk_refs"][index - 1]["code"] if index else None)
        and c.next_code
        == (receipt["chunk_refs"][index + 1]["code"] if index + 1 < len(derived) else None),
        "chunk_binding",
    )
    expected = derived[index]
    require(expected.position == c.position, "chunk_binding")
    require(
        chunk.location.span_start == c.position.span_start
        and chunk.location.span_end == c.position.span_end
        and chunk.location.page == c.position.page
        and chunk.location.paragraph == c.position.ordinal
        and chunk.location.section == " / ".join(c.position.section_path)[:128],
        "chunk_binding",
    )
    raw = store.read_bounded(chunk.object_uri, 16384)
    require(
        len(raw) == chunk.size_bytes
        and checksum(raw) == chunk.content_checksum
        and raw == expected.text.encode(),
        "chunk_checksum",
    )
    text = raw.decode()
    ss = hit["snippet_span"]
    require(
        ss["coordinate_system"] == "canonical_unicode"
        and type(ss["start"]) is int
        and type(ss["end"]) is int
        and type(ss["truncated"]) is bool
        and ss["start"] == c.position.span_start
        and ss["start"] < ss["end"] <= c.position.span_end
        and ss["truncated"] == (ss["end"] < c.position.span_end)
        and hit["snippet"] == text[: ss["end"] - ss["start"]],
        "snippet_binding",
    )
    if block.position.table_status == "unavailable":
        cap = block.position.table_capability
        require(cap is not None and bool(cap.reason), "table_reason_missing")
        raise Rejected("table_unavailable")
    if block.position.kind == "table":
        validate_grid(block)
        require(c.position == block.position and text == block.text, "table_chunk_limit")
    else:
        require(
            block.table is None and block.position.table_status == "not_applicable",
            "table_structure",
        )
        # A truncated paragraph could omit negation/conditions before or after the chunk.
        require(c.position == block.position and text == block.text, "incomplete_context")
    if material_profile == "claim-evidence/1.0.0":
        require(len(text) <= 2000, "claim_text_limit")
    else:
        # Complete research context, never a longer formal claim or truncated table.
        require(len(text) <= 4000, "definition_source_requires_bounded_chunk")
    return Evidence(chunk, document, source, parsed, block, text, receipt, original)


def validate_grid(block):
    p, grid = block.position, block.table
    cap = p.table_capability
    require(
        p.table_status == "structured" and cap is not None and grid is not None, "table_structure"
    )
    require(
        cap.version == grid.version == "table-grid/1.0.0"
        and cap.numeric_evidence is True
        and cap.status == "structured"
        and cap.reason is None
        and cap.rows == grid.rows
        and cap.columns == grid.columns
        and cap.checksum == digest(grid),
        "grid_checksum",
    )
    require(len(grid.cells) == grid.rows * grid.columns, "grid_coordinates")
    cells = {(c.row, c.column): c for c in grid.cells}
    require(
        list(cells) == [(r, c) for r in range(grid.rows) for c in range(grid.columns)],
        "grid_coordinates",
    )
    cursor = 0
    for cell in grid.cells:
        require(
            cell.span_start == cursor
            and cell.span_end == cursor + len(cell.text)
            and block.text[cell.span_start : cell.span_end] == cell.text,
            "cell_span",
        )
        require(len(set(cell.headers)) == len(cell.headers), "header_coordinates")
        for coord in cell.headers:
            header = cells.get(coord)
            require(
                header is not None
                and header.header
                and coord != (cell.row, cell.column)
                and (header.row == cell.row or header.column == cell.column),
                "header_coordinates",
            )
        cursor = cell.span_end + 1  # Single tab/newline in canonical table text.
    rows = ["\t".join(cells[r, c].text for c in range(grid.columns)) for r in range(grid.rows)]
    require("\n".join(rows) == block.text, "cell_span")


UNCERTAIN = re.compile(
    r"\b(?:may|might|possibly|uncertain|insufficient evidence|insufficient)\b|可能|证据不足|不确定",
    re.I,
)
NEGATIVE = re.compile(
    r"\b(?:not|no|never|without|exclude|avoid)\b|不推荐|不得|不能|排除|阴性", re.I
)
STRENGTH = re.compile(r"\b(?:recommend(?:ed)?|should|must|consider)\b|推荐|应当|必须|建议", re.I)
NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def validate_output(
    value: DraftOutput, evidence: Evidence, *, policy_version="claim-validation/2.0.0"
):
    if policy_version == "claim-validation/3.0.0":
        from .n124 import validate_n124_output

        return validate_n124_output(value, evidence)
    require(policy_version == "claim-validation/2.0.0", "validation_policy_binding")
    value = DraftOutput.model_validate(dict(vars(value)))
    text, p = evidence.text, evidence.chunk.derivation.position
    require(value.span.start == p.span_start and value.span.end == p.span_end, "quote_span")
    require(value.quote == text, "quote_missing")
    require(value.statement == value.target_proposition == text, "unsupported_fact")
    # Version 1 declines semantic rewriting rather than treating a second LLM as proof.
    fields = (
        "uncertainty",
        "recommendation_strength",
        "population",
        "disease_stage",
        "condition",
        "intervention",
        "exam",
        "modality",
        "outcome",
        "time",
        "laterality",
        "device",
        "applicability",
    )
    require(value.subject in text, "unsupported_fact")
    for field in fields:
        require(all(term in text for term in getattr(value, field)), "unsupported_fact")
    uncertain = tuple(m.group() for m in UNCERTAIN.finditer(text))
    require(all(t in value.uncertainty for t in uncertain), "uncertainty_lost")
    require(
        all(m.group() in value.recommendation_strength for m in STRENGTH.finditer(text)),
        "strength_lost",
    )
    # Keep explicit population/condition tokens in dedicated fields as well as the full claim.
    for pattern, fields_for_cue in (
        (r"\b(?:adults?|children|patients|elderly)\b|成人|儿童|患者|老年", ("population",)),
        (r"\b(?:if|when|unless|only)\b|如果|仅当|除非", ("condition", "applicability")),
    ):
        for match in re.finditer(pattern, text, re.I):
            require(
                any(match.group() in t for f in fields_for_cue for t in getattr(value, f)),
                "scope_lost",
            )
    # Validate declared slices first; then bind each numeric subspan to its own occurrence.
    numeric_subspans = []
    for n in value.numbers:
        start, end = n.span.start - p.span_start, n.span.end - p.span_start
        require(0 <= start < end <= len(text) and text[start:end] == n.quote, "numeric_span")
        pieces = []
        if n.comparator is not None:
            pieces.append(re.escape(n.comparator))
        pieces.append("(?P<number>" + re.escape(n.value) + ")")
        if n.unit is not None:
            pieces.append(re.escape(n.unit))
        matched = re.fullmatch(r"\s*".join(pieces), n.quote)
        require(matched is not None, "numeric_fragment")
        require("\n" not in n.quote and "\t" not in n.quote, "cross_fragment_number")
        if evidence.block.table:
            require(
                any(c.span_start <= start < end <= c.span_end for c in evidence.block.table.cells),
                "cross_cell_number",
            )
        numeric_subspans.append((start + matched.start("number"), start + matched.end("number")))
    try:
        expected = occurrences(text)
        require(len(expected) == len(value.numbers), "numeric_omission")
        previous_end = -1
        for source, n, subspan in zip(expected, value.numbers, numeric_subspans, strict=True):
            start, end = n.span.start - p.span_start, n.span.end - p.span_start
            require(
                subspan == (source.number_start, source.number_end) and start >= previous_end,
                "numeric_occurrence_mismatch",
            )
            require(source.unit is None or n.unit is not None, "numeric_unit_lost")
            require(
                source.comparator is None or n.comparator is not None, "numeric_comparator_lost"
            )
            require(
                (n.value, n.unit, n.comparator, start, end)
                == (source.value, source.unit, source.comparator, source.start, source.end),
                "numeric_binding",
            )
            previous_end = end
        resolved = supported_polarity(text, evidence.block.table)
    except PolicyRejected as exc:
        raise Rejected(str(exc)) from None
    require(value.polarity == resolved, "polarity_changed")
    return value
