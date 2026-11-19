"""Validate dispatch schemas and checked-in baseline without any database/network access."""

import argparse
import json
from pathlib import Path

from packages.task_runtime.models import DispatchRequest, Lease, Profile, Receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--write", action="store_true")
    group.add_argument("--check", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    directory = root / "packages/task_runtime/schemas/v1"
    if args.write:
        directory.mkdir(parents=True, exist_ok=True)
    for model in (DispatchRequest, Lease, Profile, Receipt):
        path = directory / (model.__name__ + ".schema.json")
        expected = (
            json.dumps(model.model_json_schema(), ensure_ascii=False, indent=2, sort_keys=True)
            + "\n"
        )
        if args.write:
            path.write_text(expected)
        elif not path.is_file() or path.read_text() != expected:
            raise SystemExit("dispatch schema snapshot mismatch: " + path.name)
    path = root / "configs/runtime/research_dispatch_v1.json"
    assert Profile.model_validate(json.loads(path.read_text())) == Profile()
    print("12B four deterministic dispatch schema snapshots and synthetic serial baseline verified")


if __name__ == "__main__":
    main()
