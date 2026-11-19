"""Versioned G2 schemas; shared contracts and their generation remain unchanged."""

import argparse
import json
from pathlib import Path

from .models import (
    BenchmarkManifest,
    EvaluationFreeze,
    EvaluationPolicy,
    LabelDefinition,
    ModelInput,
)


def schemas():
    return {
        f"{model.__name__}.schema.json": json.dumps(
            model.model_json_schema(), ensure_ascii=False, indent=2, sort_keys=True
        )
        + "\n"
        for model in (
            BenchmarkManifest,
            EvaluationFreeze,
            EvaluationPolicy,
            LabelDefinition,
            ModelInput,
        )
    }


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--write", action="store_true")
    group.add_argument("--check", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).parent / "schemas"
    for name, content in schemas().items():
        if args.write:
            root.mkdir(exist_ok=True)
            (root / name).write_text(content)
        else:
            if (root / name).read_text() != content:
                raise ValueError("benchmark_schema_drift:" + name)
    print("5 benchmark schemas", "written" if args.write else "verified")


if __name__ == "__main__":
    main()
