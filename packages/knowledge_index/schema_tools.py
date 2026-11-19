"""04B4 schemas for the bounded N124 selection and unchanged derivation models."""

import argparse
import json
from pathlib import Path

from .models import (
    Derivation,
    N124ChunkPolicy,
    ParsedDocument,
    ProseChunkPolicy,
    SectionChunkPolicy,
    WindowChunkPolicy,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent / "schemas"
    if args.write:
        root.mkdir(exist_ok=True)
    for model in (
        N124ChunkPolicy,
        SectionChunkPolicy,
        ProseChunkPolicy,
        WindowChunkPolicy,
        Derivation,
        ParsedDocument,
    ):
        path = root / (model.__name__ + ".schema.json")
        expected = model.model_json_schema()
        if args.write:
            path.write_text(json.dumps(expected, ensure_ascii=False, indent=2) + "\n")
        elif json.loads(path.read_text()) != expected:
            raise SystemExit("schema mismatch: " + str(path))
    print("6 index-owned schemas verified")


if __name__ == "__main__":
    main()
