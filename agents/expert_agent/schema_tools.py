"""Deterministic research schemas alongside, never replacing frozen contracts v1."""

import argparse
import json
from pathlib import Path

from packages.llm_provider.models import ExpertReviewInput

from .models import (
    ExpertDecision,
    ReviewFacts,
    ReviewFrame,
    ReviewPolicy,
    ReviewResult,
    ReviewResultV2,
    ReviewResultV3,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).parent / "schemas"
    models = (
        ReviewPolicy,
        ReviewFacts,
        ReviewFrame,
        ExpertDecision,
        ExpertReviewInput,
        ReviewResult,
        ReviewResultV2,
        ReviewResultV3,
    )
    if args.write:
        root.mkdir(exist_ok=True)
    for model in models:
        path = root / (model.__name__ + ".schema.json")
        data = (
            json.dumps(model.model_json_schema(), sort_keys=True, ensure_ascii=False, indent=2)
            + "\n"
        )
        if args.write:
            path.write_text(data)
        elif not path.exists() or path.read_text() != data:
            raise SystemExit("Expert schema drift: " + path.name)
    print("verified", len(models), "research expert schemas")


if __name__ == "__main__":
    main()
