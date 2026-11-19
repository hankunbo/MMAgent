"""Versioned platform profile, observations and extended OpenAPI snapshots."""

import argparse
import json
from pathlib import Path

from apps.api.assembly import Dependencies
from apps.api.auth import LocalIdentities

from .runtime import Installation, MetricBlock, Profile


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--write", action="store_true")
    group.add_argument("--check", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2] / "docs/api/platform"
    docs = {c.__name__ + ".schema.json": c.model_json_schema() for c in (MetricBlock, Profile)}
    docs["openapi.json"] = Installation(Dependencies(LocalIdentities({}))).app().app.openapi()
    for name, value in docs.items():
        expected = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        path = root / name
        if args.write:
            root.mkdir(parents=True, exist_ok=True)
            path.write_text(expected)
        elif not path.exists() or path.read_text() != expected:
            raise SystemExit("platform schema drift: " + name)
    print("platform profile, metric block and extended OpenAPI verified")


if __name__ == "__main__":
    main()
