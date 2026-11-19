"""Deterministic additive Memory schemas; does not regenerate frozen contracts v1."""

import argparse
import json
from pathlib import Path

from .archive_models import (
    ArchiveReadResult,
    ArchiveRecord,
    ArtifactSource,
    ExtendedArchiveReadResult,
    ExtendedArchiveRecord,
)
from .models import EmbeddingIndex, ReadResult, Record, Review, Source


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).parent / "schemas"
    for cls in (
        Source,
        Record,
        Review,
        EmbeddingIndex,
        ReadResult,
        ArchiveReadResult,
        ArchiveRecord,
        ExtendedArchiveReadResult,
        ExtendedArchiveRecord,
        ArtifactSource,
    ):
        expected = (
            json.dumps(cls.model_json_schema(), ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        )
        path = root / (cls.__name__ + ".json")
        if args.write:
            root.mkdir(exist_ok=True)
            path.write_text(expected)
        elif not path.is_file() or path.read_text() != expected:
            raise SystemExit("memory_schema_drift:" + cls.__name__)


if __name__ == "__main__":
    main()
