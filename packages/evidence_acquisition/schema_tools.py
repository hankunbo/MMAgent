"""Generate/check only the new opt-in acquisition schemas; old snapshots unchanged."""

import argparse
import json
from pathlib import Path

from .models import (
    AcquisitionPlan,
    ActionHistory,
    GapCard,
    Requirement,
    SearchProgress,
    SearchSuggestion,
    TargetedVisionPayload,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).parent / "schemas"
    if args.write:
        root.mkdir(exist_ok=True)
    models = (
        AcquisitionPlan,
        ActionHistory,
        GapCard,
        Requirement,
        SearchProgress,
        SearchSuggestion,
        TargetedVisionPayload,
    )
    for model in models:
        path = root / (model.__name__ + ".schema.json")
        text = (
            json.dumps(model.model_json_schema(), ensure_ascii=False, indent=2, sort_keys=True)
            + "\n"
        )
        if args.write:
            path.write_text(text)
        elif not path.exists() or path.read_text() != text:
            raise SystemExit("acquisition schema drift: " + path.name)
    print("verified", len(models), "acquisition schemas")


if __name__ == "__main__":
    main()
