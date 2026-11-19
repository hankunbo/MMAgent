"""Deterministic 06C schemas; no changes to contracts v1 generation or Makefile."""

import argparse
import json
from pathlib import Path

from .models import CommandAllowlist, CompilerConfig, PlanBundle, PlanDocument, WorkflowChoice


def schemas():
    return {
        f"{model.__name__}.schema.json": json.dumps(
            model.model_json_schema(), ensure_ascii=False, indent=2, sort_keys=True
        )
        + "\n"
        for model in (CommandAllowlist, CompilerConfig, PlanBundle, PlanDocument, WorkflowChoice)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).parent / "schemas"
    for name, content in schemas().items():
        if args.write:
            root.mkdir(exist_ok=True)
            (root / name).write_text(content)
        else:
            assert (root / name).read_text() == content, name
    print(
        "5 deterministic 06C schema artifacts verified" if not args.write else "5 schemas written"
    )


if __name__ == "__main__":
    main()
