"""Deterministic values and object-store failure boundaries, without network IO."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from db.knowledge.models import SourceVersion
from packages.knowledge_archive import Bibliography
from packages.knowledge_archive.models import identity_text
from packages.object_store import LocalObjectStore, checksum


@pytest.mark.parametrize(
    "raw,expected", [("eng", "en"), ("EN_us", "en-us"), ("zho", "zh"), ("english", "en")]
)
def test_language(raw, expected):
    assert Bibliography(language=raw).language == expected


@pytest.mark.parametrize("raw", ["2024", "2024-02", "2024-02-29"])
def test_date_precision(raw):
    assert Bibliography(publication_date=raw).publication_date == raw


@pytest.mark.parametrize("raw", ["2023-02-29", "2024-13", "24", "January 2024", "2024-00-00"])
def test_invalid_date(raw):
    with pytest.raises(ValidationError):
        Bibliography(publication_date=raw)


def test_names_and_sets():
    a = Bibliography(
        title="Ａ Synthetic  Study",
        authors=("First Author", "Second Author", "First Author"),
        institutions=("Z Institute", "A Institute", "Z Institute"),
    )
    b = Bibliography(
        title="A Synthetic Study",
        authors=("First Author", "Second Author"),
        institutions=("A Institute", "Z Institute"),
    )
    assert a == b
    assert identity_text("Ａ Study: Result!") == "a study result"
    assert Bibliography(authors=tuple(reversed(a.authors))).authors[0] != a.authors[0]


def test_local_object_promotion_checksum_and_scope(tmp_path):
    store = LocalObjectStore(tmp_path)
    uri = store.stage(b"synthetic object")
    with pytest.raises(ValueError):
        store.promote(uri, checksum(b"wrong"))
    promoted = store.promote(uri, checksum(b"synthetic object"))
    assert store.read(promoted) == b"synthetic object"
    with pytest.raises(ValueError):
        store.read("object://knowledge/objects/../../outside")
    store.discard(promoted)
    assert not list(tmp_path.glob("*/*"))


def test_source_version_schema_snapshot():
    assert (
        json.loads(Path("packages/knowledge_archive/schemas/SourceVersion.schema.json").read_text())
        == SourceVersion.model_json_schema()
    )


def test_archive_metadata_cannot_claim_a_different_material_checksum():
    from packages.knowledge_archive import ArchiveMetadata

    with pytest.raises(ValidationError):
        ArchiveMetadata(
            bibliography=Bibliography(title="Synthetic metadata"),
            metadata_checksum="sha256:" + "0" * 64,
            fetcher_version="1.0.0",
            pipeline_version="1.0.0",
        )
