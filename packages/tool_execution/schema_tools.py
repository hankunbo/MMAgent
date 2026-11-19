"""Deterministic 06D schemas; no changes to contracts v1 generation or Makefile."""

import argparse
import json
from pathlib import Path

from .fundus_bundle import FundusBundle, FundusBundlePolicy
from .models import ExecutionDecision, ExecutionPolicy, QCRecord, StepResult, WorkflowResult
from .qc import PNGPolicy


def schemas():
    return {
        f"{model.__name__}.schema.json": json.dumps(
            model.model_json_schema(), ensure_ascii=False, indent=2, sort_keys=True
        )
        + "\n"
        for model in (
            ExecutionPolicy,
            ExecutionDecision,
            QCRecord,
            StepResult,
            WorkflowResult,
            PNGPolicy,
            FundusBundle,
            FundusBundlePolicy,
        )
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
        f"{len(schemas())} deterministic 06D schema artifacts verified"
        if not args.write
        else f"{len(schemas())} schemas written"
    )


if __name__ == "__main__":
    main()
