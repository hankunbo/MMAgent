"""Read-only 13A inventory; environment roots, unique output, no labels on stdout."""

import argparse
import json
import os
from pathlib import Path

from packages.evaluation_preparation.service import build
from packages.grounding.models import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    registry, layouts = build(os.environ)
    args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    document = registry.model_dump(mode="json")
    for name, value in {
        "registry.json": document,
        "inventory.json": layouts,
        "binding.json": dict(
            registry_checksum=digest(document), inventory_checksum=digest(layouts)
        ),
    }.items():
        (args.output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                status="preparation_saved",
                datasets=len(registry.entries),
                formal_test_unlocked=False,
                model_executions=0,
            )
        )
    )


if __name__ == "__main__":
    main()
