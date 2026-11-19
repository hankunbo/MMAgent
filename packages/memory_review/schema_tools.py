"""Deterministic snapshots for new envelopes; old snapshots are not regenerated."""

import argparse
import json
from pathlib import Path

from packages.research_runtime.task_models import TaskCompletion, TaskDefinition

from .models import (
    Advice,
    Binding,
    Conditions,
    JointKnowledgeBinding,
    Opinion,
    Policy,
    Publication,
    Query,
    ReviewInput,
)
from .task_models import TASK_MODELS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).with_name("schemas")
    if args.write:
        root.mkdir(exist_ok=True)
    for model in (
        Advice,
        Binding,
        Conditions,
        JointKnowledgeBinding,
        Opinion,
        Policy,
        Publication,
        Query,
        ReviewInput,
        *TASK_MODELS.values(),
        TaskDefinition,
        TaskCompletion,
    ):
        path = root / (model.__name__ + ".json")
        expected = (
            json.dumps(model.model_json_schema(), indent=2, ensure_ascii=False, sort_keys=True)
            + "\n"
        )
        if args.write:
            path.write_text(expected)
        else:
            assert path.read_text() == expected, path


if __name__ == "__main__":
    main()
