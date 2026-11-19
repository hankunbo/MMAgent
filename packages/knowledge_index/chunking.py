"""Stable semantic blocks with bounded sentence-aware splits and overlap."""

import re

from .models import (
    Block,
    ChunkPolicy,
    N124ChunkPolicy,
    ParsedDocument,
    ProseChunkPolicy,
    SectionChunkPolicy,
    WindowChunkPolicy,
    parse_chunk_policy,
)


def chunk(document: ParsedDocument, policy: ChunkPolicy) -> tuple[Block, ...]:
    policy = parse_chunk_policy(policy)
    if isinstance(policy, N124ChunkPolicy):
        selected = [
            b
            for b in document.blocks
            if b.position.section_path
            == ("Glaucoma: diagnosis and management", "Recommendations", "1.2 Diagnosis", "1.2.4")
        ]
        if (
            document.parser != "structured-html"
            or len(selected) != 2
            or selected[0].position.kind != "heading"
            or selected[0].text != "1.2.4"
            or selected[1].position.kind != "paragraph"
            or selected[1].position.ordinal != selected[0].position.ordinal + 1
            or len(selected[1].text) > policy.max_chars
        ):
            raise ValueError("n124_chunk_selection")
        return (selected[1],)
    if document.fidelity_policy and not re.fullmatch(
        r"(?:[2-9]|[1-9][0-9]+)\.[0-9]+\.[0-9]+", policy.version
    ):
        raise ValueError("legacy_chunk_policy_requires_replay")
    blocks = document.blocks
    if isinstance(policy, WindowChunkPolicy):
        _ordered_blocks(blocks)
        blocks = tuple(
            b for b in blocks if policy.first_ordinal <= b.position.ordinal <= policy.last_ordinal
        )
        if (
            not blocks
            or blocks[0].position.ordinal != policy.first_ordinal
            or blocks[-1].position.ordinal != policy.last_ordinal
        ):
            raise ValueError("window_endpoint_missing")
        if policy.tables == "excluded":
            blocks = tuple(b for b in blocks if b.position.kind != "table")
        if not blocks:
            raise ValueError("source_window_empty")
    if isinstance(policy, ProseChunkPolicy):
        blocks = tuple(b for b in blocks if b.position.kind != "table")
        if not any(b.position.kind != "heading" for b in blocks):
            raise ValueError("prose_selection_empty")
    if isinstance(policy, SectionChunkPolicy):
        requested = set(policy.section_paths)
        present = {b.position.section_path for b in blocks if b.position.kind != "heading"}
        if not requested <= present:
            raise ValueError("selected_section_missing_or_heading_only")
        blocks = tuple(b for b in blocks if b.position.section_path in requested)
    result = []
    for block in blocks:
        if block.position.kind == "table" and len(block.text) > policy.max_chars:
            raise ValueError("table_chunk_limit")
        start = 0
        while start < len(block.text):
            end = min(start + policy.max_chars, len(block.text))
            if end < len(block.text):
                breaks = list(re.finditer(r"[.!?。！？](?:\s|$)|\n", block.text[start:end]))
                suitable = [m.end() for m in breaks if m.end() >= policy.max_chars // 2]
                if suitable:
                    end = start + suitable[-1]
            text = block.text[start:end]
            result.append(
                Block(
                    text=text,
                    table=block.table,
                    position=block.position.model_copy(
                        update={
                            "span_start": block.position.span_start + start,
                            "span_end": block.position.span_start + end,
                        }
                    ),
                )
            )
            if len(result) > policy.max_chunks:
                raise ValueError("chunk_count_limit")
            if end == len(block.text):
                break
            start = end - policy.overlap_chars
    return tuple(result)


def _ordered_blocks(blocks):
    ordinals = [b.position.ordinal for b in blocks]
    if ordinals != sorted(set(ordinals)):
        raise ValueError("window_ordinal_order")


def plan_windows(document, *, source_checksum, policy, max_windows=4):
    """Preflight the complete finite plan before any window is persisted.

    Never truncate blocks, renumber positions, or enlarge the per-index limits.
    Medical relevance does not determine which blocks are retained.
    """
    policy = parse_chunk_policy(policy)
    if type(policy) not in {ChunkPolicy, ProseChunkPolicy}:
        raise ValueError("window_requires_whole_or_prose_policy")
    if type(max_windows) is not int or not 1 <= max_windows <= 4:
        raise ValueError("source_window_limit")
    if isinstance(policy, ProseChunkPolicy) and policy.source_checksum != source_checksum:
        raise ValueError("window_source_binding")
    _ordered_blocks(document.blocks)
    excluded = isinstance(policy, ProseChunkPolicy)
    blocks = tuple(b for b in document.blocks if not excluded or b.position.kind != "table")
    if not blocks:
        raise ValueError("source_window_empty")
    base = ChunkPolicy(
        max_chars=policy.max_chars,
        overlap_chars=policy.overlap_chars,
        max_chunks=policy.max_chunks,
        projection_version=policy.projection_version,
    )
    windows, first, last, count = [], None, None, 0

    def window(start, end):
        return WindowChunkPolicy(
            source_checksum=source_checksum,
            first_ordinal=start,
            last_ordinal=end,
            tables="excluded" if excluded else "included",
            max_chars=policy.max_chars,
            overlap_chars=policy.overlap_chars,
            max_chunks=policy.max_chunks,
            projection_version=policy.projection_version,
        )

    for block in blocks:
        size = len(chunk(document.model_copy(update={"blocks": (block,)}), base))
        if first is not None and count + size > policy.max_chunks:
            windows.append(window(first, last))
            if len(windows) >= max_windows:
                raise ValueError("source_window_limit")
            first, count = None, 0
        if first is None:
            first = block.position.ordinal
        last, count = block.position.ordinal, count + size
    windows.append(window(first, last))
    return tuple(windows)
