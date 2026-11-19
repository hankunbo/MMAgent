"""Actual owner schemas for H additions; old owner snapshots are unchanged."""

import argparse
import json
from pathlib import Path

from packages.knowledge_review.engineering_stop import (
    EngineeringBasis,
    EngineeringReview,
    StopBasisBinding,
)
from packages.tool_execution.scalar_output import GeometryOutput, ScalarOutput

from .models import ScoreDefinition, ScoreEntry, ScoreEnvelope

MODELS = (
    ScoreDefinition,
    ScoreEntry,
    ScoreEnvelope,
    ScalarOutput,
    GeometryOutput,
    EngineeringBasis,
    EngineeringReview,
    StopBasisBinding,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).parent / "schemas"
    for model in MODELS:
        path = root / (model.__name__ + ".schema.json")
        content = (
            json.dumps(model.model_json_schema(), ensure_ascii=False, indent=2, sort_keys=True)
            + "\n"
        )
        if args.write:
            root.mkdir(exist_ok=True)
            path.write_text(content)
        elif not path.exists() or path.read_text() != content:
            raise SystemExit("H owner schema drift: " + path.name)


if __name__ == "__main__":
    main()
