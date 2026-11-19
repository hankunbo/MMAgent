"""Versioned bounded parsing and retrieval configuration; no clinical inference."""

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_serializer, model_validator

FIDELITY_VERSION = "text-fidelity/2.0.0"
PARSER_VERSION = "2.0.0"


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    @model_serializer(mode="wrap")
    def legacy_extensions(self, handler):
        result = handler(self)
        # Absent extensions must not change canonical bytes/checksums of v1 artifacts.
        for name in ("fidelity_policy", "table", "table_capability", "table_evidence"):
            if result.get(name) is None:
                result.pop(name, None)
        return result


def canonical(value) -> bytes:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def digest(value) -> str:
    return "sha256:" + hashlib.sha256(canonical(value)).hexdigest()


class ParsePolicy(Model):
    version: str = "parse-limits/1.0.0"
    max_bytes: int = Field(default=2_000_000, ge=1, le=4_000_000)
    max_chars: int = Field(default=250_000, ge=1, le=500_000)
    max_nodes: int = Field(default=8000, ge=1, le=20000)
    max_depth: int = Field(default=48, ge=1, le=64)
    max_pages: int = Field(default=100, ge=1, le=200)
    timeout_seconds: int = Field(default=8, ge=1, le=30)
    memory_mb: int = Field(default=512, ge=256, le=1024)
    ocr: Literal["unsupported"] = "unsupported"


class TableCapability(Model):
    version: Literal["table-grid/1.0.0"] = "table-grid/1.0.0"
    status: Literal["structured", "unavailable"]
    reason: str | None = Field(default=None, max_length=128)
    rows: int | None = Field(default=None, ge=1, le=64)
    columns: int | None = Field(default=None, ge=1, le=32)
    checksum: str | None = None
    numeric_evidence: bool


class TableCell(Model):
    row: int = Field(ge=0, lt=64)
    column: int = Field(ge=0, lt=32)
    text: str = Field(max_length=4096)
    header: bool
    headers: tuple[tuple[int, int], ...] = Field(default=(), max_length=96)
    # Unicode offsets relative to the containing canonical table block.
    span_start: int = Field(ge=0)
    span_end: int = Field(ge=0)


class TableGrid(Model):
    version: Literal["table-grid/1.0.0"] = "table-grid/1.0.0"
    rows: int = Field(ge=1, le=64)
    columns: int = Field(ge=1, le=32)
    cells: tuple[TableCell, ...] = Field(max_length=512)


class Position(Model):
    section_path: tuple[str, ...] = Field(max_length=64)
    kind: Literal["paragraph", "heading", "list", "table", "caption"]
    ordinal: int = Field(ge=1)
    page: int | None = Field(default=None, ge=1)
    span_start: int = Field(ge=0)
    span_end: int = Field(ge=1)
    table_status: Literal["structured", "unavailable", "not_applicable"] = "not_applicable"
    table_capability: TableCapability | None = None


class Block(Model):
    text: str = Field(min_length=1, max_length=500000)
    position: Position
    table: TableGrid | None = None


class ParsedDocument(Model):
    parser: str
    parser_version: str
    policy_checksum: str
    language: str = Field(max_length=32)
    title: str = Field(max_length=512)
    blocks: tuple[Block, ...]
    fidelity_policy: Literal["text-fidelity/2.0.0", "text-fidelity/2.1.0"] | None = None


class ParseResult(Model):
    status: Literal["parsed", "failed", "no_text", "unsupported"]
    reason: str | None = None
    document: ParsedDocument | None = None


class ChunkPolicy(Model):
    version: str = "2.0.0"
    max_chars: int = Field(default=1200, ge=64, le=4096)
    overlap_chars: int = Field(default=100, ge=0, le=512)
    max_chunks: int = Field(default=192, ge=1, le=250)
    projection_version: Literal["bounded-projection/1.0.0"] = "bounded-projection/1.0.0"

    @model_validator(mode="after")
    def overlap_bound(self):
        if self.overlap_chars >= self.max_chars // 2:
            raise ValueError("overlap must be less than half the chunk bound")
        return self


class N124ChunkPolicy(ChunkPolicy):
    """Bounded selection from a complete parse; never merges or renumbers blocks."""

    version: Literal["2.1.0"] = "2.1.0"
    selection: Literal["nice-ng81-n124-paragraph/1.0.0"]
    source_checksum: Literal[
        "sha256:2bcf6cc2147ce8f986d58b99b86a164fd3cf4bd53f4e99771b45270f8abcdb26"
    ] = "sha256:2bcf6cc2147ce8f986d58b99b86a164fd3cf4bd53f4e99771b45270f8abcdb26"


class SectionChunkPolicy(ChunkPolicy):
    """Explicit source-bound partial index; preserve complete selected blocks/positions."""

    version: Literal["2.2.0"] = "2.2.0"
    selection: Literal["exact-section-paths/1.0.0"] = "exact-section-paths/1.0.0"
    source_checksum: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    section_paths: tuple[tuple[str, ...], ...] = Field(min_length=1, max_length=32)

    @model_validator(mode="after")
    def exact_bounded_sections(self):
        if any(
            not p or len(p) > 64 or any(not s or len(s) > 512 for s in p)
            for p in self.section_paths
        ):
            raise ValueError("section_path_bounds")
        if len(set(self.section_paths)) != len(self.section_paths):
            raise ValueError("duplicate_section_path")
        return self


class ProseChunkPolicy(ChunkPolicy):
    """Source-bound partial index: keep non-table blocks, retain the entire parse."""

    version: Literal["2.3.0"] = "2.3.0"
    selection: Literal["non-table-blocks/1.0.0"] = "non-table-blocks/1.0.0"
    source_checksum: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")


class WindowChunkPolicy(ChunkPolicy):
    """An explicit complete-block range from an immutable full source parse."""

    version: Literal["2.4.0"] = "2.4.0"
    selection: Literal["exact-ordinal-window/1.0.0"] = "exact-ordinal-window/1.0.0"
    source_checksum: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    first_ordinal: int = Field(ge=1)
    last_ordinal: int = Field(ge=1)
    tables: Literal["included", "excluded"] = "included"

    @model_validator(mode="after")
    def ordered(self):
        if self.last_ordinal < self.first_ordinal:
            raise ValueError("window_ordinal_order")
        return self


def parse_chunk_policy(value):
    data = dict(vars(value)) if isinstance(value, ChunkPolicy) else value
    model = {
        "2.1.0": N124ChunkPolicy,
        "2.2.0": SectionChunkPolicy,
        "2.3.0": ProseChunkPolicy,
        "2.4.0": WindowChunkPolicy,
    }.get(data.get("version"), ChunkPolicy)
    # Persisted JSON represents tuple paths as arrays. Validate its JSON contract
    # rather than rejecting the owner's own immutable section-selection receipt.
    return model.model_validate_json(canonical(data), strict=True)


class Derivation(Model):
    version: Literal["derivation/1.0.0"] = "derivation/1.0.0"
    original_checksum: str
    parse_uri: str
    parse_checksum: str
    parser: str
    parser_version: str
    parse_policy_checksum: str
    chunk_policy_checksum: str
    language: str = Field(max_length=32)
    targets: tuple[str, ...] = Field(default=(), max_length=32)
    position: Position | None = None
    previous_code: str | None = None
    next_code: str | None = None
    fidelity_policy: Literal["text-fidelity/2.0.0", "text-fidelity/2.1.0"] | None = None
    table_evidence: Literal["structured_or_absent", "contains_unavailable"] | None = None


class ProviderSpec(Model):
    provider: str = Field(min_length=1, max_length=128)
    config_version: str = Field(min_length=1, max_length=64)
    model: str = Field(min_length=1, max_length=128)
    model_version: str = Field(min_length=1, max_length=64)
    dimension: int = Field(ge=1, le=2000)
    preprocessing: Literal["unicode-space/1.0.0"] = "unicode-space/1.0.0"


class IndexConfig(Model):
    version: str = "index/1.0.0"
    mode: Literal["hybrid", "fts_only"]
    vector_unavailable_reason: str | None = Field(default=None, max_length=256)
    embedder: ProviderSpec | None = None
    reranker: ProviderSpec | None = None
    reranker_mode: Literal["enabled", "disabled"] = "disabled"
    fts_config: Literal["simple"] = "simple"
    fts_preprocessing: Literal["unicode-cjk/1.0.0"] = "unicode-cjk/1.0.0"
    fusion: Literal["weighted-rrf/1.0.0"] = "weighted-rrf/1.0.0"
    fts_weight: float = Field(default=0.5, ge=0, le=1)
    vector_weight: float = Field(default=0.5, ge=0, le=1)
    rrf_k: int = Field(default=60, ge=1, le=1000)
    recall_k: int = Field(default=40, ge=1, le=200)
    top_k: int = Field(default=5, ge=1, le=50)
    snippet_chars: int = Field(default=600, ge=1, le=1200)

    @model_validator(mode="after")
    def explicit_capabilities(self):
        if self.mode == "hybrid":
            if self.embedder is None or self.vector_unavailable_reason is not None:
                raise ValueError(
                    "hybrid requires a versioned embedder and no unavailability reason"
                )
            if self.fts_weight <= 0 or self.vector_weight <= 0:
                raise ValueError("hybrid requires both positive fusion weights")
        elif (
            not self.vector_unavailable_reason
            or self.embedder is not None
            or self.vector_weight != 0
        ):
            raise ValueError("FTS-only requires reason, no embedder and zero vector weight")
        if (self.reranker_mode == "enabled") != (self.reranker is not None):
            raise ValueError("reranker mode/spec mismatch")
        if self.top_k > self.recall_k:
            raise ValueError("Top-K exceeds recall bound")
        return self


class Filters(Model):
    languages: tuple[str, ...] = Field(default=(), max_length=32)
    source_types: tuple[str, ...] = Field(default=(), max_length=16)
    date_from: str | None = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    date_to: str | None = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    population: tuple[str, ...] = Field(default=(), max_length=32)
    conditions: tuple[str, ...] = Field(default=(), max_length=32)
    modalities: tuple[str, ...] = Field(default=(), max_length=32)
    targets: tuple[str, ...] = Field(default=(), max_length=32)
    anatomy: tuple[str, ...] = Field(default=(), max_length=32)

    @model_validator(mode="after")
    def ordered_dates(self):
        from datetime import date

        for value in (self.date_from, self.date_to):
            if value:
                date.fromisoformat(value)
        if self.date_from and self.date_to and self.date_from > self.date_to:
            raise ValueError("date range inverted")
        return self
