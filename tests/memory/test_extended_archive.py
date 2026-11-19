"""Versioned capacity is not a source omission or publication exemption."""

import pytest

from db.runtime.repository.case_pipeline import identity
from packages.grounding.models import digest, validate
from packages.memory.archive_models import (
    ArchiveRecord,
    ArchiveSourceCapacityError,
    ArtifactSource,
    ExtendedArchiveRecord,
    archive_record_class,
)
from packages.memory.writer_models import parse_record


def record(count):
    sources = tuple(
        ArtifactSource(
            case_code=identity("CASE", "test"),
            run_id=identity("RUN", "test"),
            key=f"s{i}",
            object_code=f"ASSET:S{i}",
            checksum=digest(i),
            object_checksum=digest(i),
            dependencies=(digest(i),),
            applicability=("USE:RESEARCH_WORKFLOW",),
            nature="source_fact",
        )
        for i in range(count)
    )
    return archive_record_class(sources)(
        key="snapshot",
        kind="graph_snapshot",
        case_code=identity("CASE", "test"),
        run_id=identity("RUN", "test"),
        sources=sources,
        applicability=("USE:RESEARCH_WORKFLOW",),
        nature="source_fact",
        detail=dict(
            type="case_snapshot",
            target="GOAL:TEST",
            modality="SLO",
            origin_code="ASSET:S0",
            modality_source="synthetic_boundary",
            steps=(),
            outcome="recorded",
        ),
    )


@pytest.mark.parametrize("count", [1, 64, 65, 71, 128, 129, 256])
def test_versioned_source_boundaries_and_exact_readback(count):
    value = record(count)
    assert len(value.sources) == count
    assert value.schema_version == f"research-memory/1.{3 if count <= 64 else 4}.0"
    assert parse_record(value.model_dump(mode="json")) == value
    if count > 64:
        with pytest.raises(ValueError):
            validate(ArchiveRecord, value)


def test_no_truncation_and_no_silent_legacy_upgrade():
    with pytest.raises(ArchiveSourceCapacityError) as failure:
        record(257)
    assert failure.value.required_sources == 257
    assert failure.value.maximum_sources == 256
    raw = record(65).model_dump(mode="json")
    raw["schema_version"] = "research-memory/1.3.0"
    with pytest.raises(ValueError):
        parse_record(raw)


def test_extended_revalidates_scope_duplicate_and_payload_bound():
    raw = record(65).model_dump(mode="json")
    raw["sources"][1] = raw["sources"][0]
    with pytest.raises(ValueError, match="memory_duplicate_source"):
        parse_record(raw)
    raw = record(65).model_dump(mode="json")
    raw["sources"][0]["case_code"] = identity("CASE", "other")
    with pytest.raises(ValueError, match="memory_scope"):
        parse_record(raw)
    raw = record(65).model_dump(mode="json")
    raw["detail"]["execution_snapshot"] = {"oversize": "x" * 4194304}
    with pytest.raises(ValueError, match="memory_archive_payload_capacity"):
        validate(ExtendedArchiveRecord, raw)
