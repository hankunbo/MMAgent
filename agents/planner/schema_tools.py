"""Use the project's existing deterministic schema write/check convention."""

import argparse
import json
from pathlib import Path

from packages.planner_control.models import (
    ActionDecision,
    Binding,
    Command,
    Control,
    ControlReport,
    Event,
    LegalActionSet,
    Outcome,
    RankingPolicy,
    SelectionOutcome,
    SelectionPending,
    TraceRanking,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).parent / "schemas"
    for model in (
        Binding,
        Command,
        Control,
        ControlReport,
        Event,
        Outcome,
        LegalActionSet,
        RankingPolicy,
        TraceRanking,
        ActionDecision,
        SelectionOutcome,
        SelectionPending,
    ):
        path = root / (model.__name__ + ".schema.json")
        data = (
            json.dumps(model.model_json_schema(), sort_keys=True, ensure_ascii=False, indent=2)
            + "\n"
        )
        if args.write:
            path.write_text(data)
        elif not path.exists() or path.read_text() != data:
            raise SystemExit("Planner schema drift: " + path.name)


if __name__ == "__main__":
    main()
