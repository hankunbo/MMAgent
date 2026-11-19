"""Complete original parse retained; selection preserves one existing whole block."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from packages.knowledge_index.chunking import chunk
from packages.knowledge_index.models import (
    ChunkPolicy,
    N124ChunkPolicy,
    canonical,
    digest,
    parse_chunk_policy,
)
from tests.knowledge_claims.n124_support import unit_evidence


def test_exact_selection_preserves_canonical_coordinates_and_old_limit():
    document = unit_evidence().parsed
    with pytest.raises(ValueError, match="chunk_count_limit"):
        chunk(document, ChunkPolicy())
    policy = N124ChunkPolicy(selection="nice-ng81-n124-paragraph/1.0.0")
    result = chunk(document, policy)
    assert len(result) == 1 and result[0] == document.blocks[77]
    assert result[0].position.span_start == 4893
    assert len(document.blocks) == 365
    assert digest(policy) != digest(ChunkPolicy())
    assert b"selection" not in canonical(ChunkPolicy())
    assert parse_chunk_policy(policy) == policy


@pytest.mark.parametrize("mutation", ["missing_selection", "version", "source", "list"])
def test_selection_is_not_a_general_fallback(mutation):
    policy = N124ChunkPolicy(selection="nice-ng81-n124-paragraph/1.0.0")
    data = dict(vars(policy))
    if mutation == "missing_selection":
        data.pop("selection")
    elif mutation == "version":
        data["version"] = "2.0.0"
    elif mutation == "source":
        data["source_checksum"] = "sha256:" + "a" * 64
    else:
        data["selection"] = "nice-ng81-n111-list/1.0.0"
    with pytest.raises(ValidationError):
        parse_chunk_policy(data)


def test_index_schema_snapshots():
    import json

    from packages.knowledge_index.models import Derivation, ParsedDocument

    for model in (N124ChunkPolicy, Derivation, ParsedDocument):
        assert (
            json.loads(
                Path(
                    "packages/knowledge_index/schemas", model.__name__ + ".schema.json"
                ).read_text()
            )
            == model.model_json_schema()
        )
