"""Additive task schema generation; the original research v1 schemas are untouched."""

import argparse
import json
from pathlib import Path

from .component_inputs import ComponentInputPolicy
from .task_context import TaskInput, TaskIntent, TaskRunContext, TaskRunRequest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("write", "check"))
    args = parser.parse_args()
    from packages.dataset_adapters.multimodal.tasks import Registry, TaskProjection

    models = (
        TaskIntent,
        TaskInput,
        TaskRunRequest,
        TaskRunContext,
        Registry,
        TaskProjection,
        ComponentInputPolicy,
    )
    for model in models:
        path = Path(__file__).parent / "schemas" / (model.__name__ + ".schema.json")
        content = (
            json.dumps(model.model_json_schema(), ensure_ascii=False, indent=2, sort_keys=True)
            + "\n"
        )
        if args.command == "write":
            path.write_text(content)
        elif not path.exists() or path.read_text() != content:
            raise SystemExit("task schema drift: " + model.__name__)
    print(f"{len(models)} task research schemas verified")


if __name__ == "__main__":
    main()
