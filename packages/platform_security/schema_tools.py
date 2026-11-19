"""Deterministic security contracts; contains no deployment credentials."""

import argparse
import json
from pathlib import Path

from .egress import Approval, EgressRequest, Policy


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--write", action="store_true")
    group.add_argument("--check", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).parent / "schemas"
    for model in (Policy, Approval, EgressRequest):
        expected = (
            json.dumps(model.model_json_schema(), ensure_ascii=False, indent=2, sort_keys=True)
            + "\n"
        )
        path = root / (model.__name__ + ".schema.json")
        if args.write:
            root.mkdir(exist_ok=True)
            path.write_text(expected)
        elif not path.exists() or path.read_text() != expected:
            raise SystemExit("platform security schema drift: " + model.__name__)
    print("3 platform security schemas verified")


if __name__ == "__main__":
    main()
