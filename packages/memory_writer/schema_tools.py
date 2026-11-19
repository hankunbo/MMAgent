"""Additive Writer schemas, independent of frozen contracts v1 and Memory 1.0."""

import argparse
import json
from pathlib import Path

from packages.memory.writer_models import ExecutionSource, WriterReadResult, WriterRecord


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).parent / "schemas"
    for cls in (ExecutionSource, WriterRecord, WriterReadResult):
        content = json.dumps(cls.model_json_schema(), indent=2, sort_keys=True) + "\n"
        path = root / (cls.__name__ + ".json")
        if args.write:
            root.mkdir(exist_ok=True)
            path.write_text(content)
        elif not path.is_file() or path.read_text() != content:
            raise SystemExit("writer_schema_drift:" + cls.__name__)


if __name__ == "__main__":
    main()
