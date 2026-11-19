"""Generate/check only additive multimodal schemas, never contracts v1 snapshots."""

import argparse
import json
from pathlib import Path

from .analysis import InputPlan
from .capabilities import InputCapabilities
from .models import CaseComponents
from .storage import MultimodalRegistration, MultimodalVisible, StoredInput


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    directory = Path(__file__).parent / "schemas"
    if args.write:
        directory.mkdir(exist_ok=True)
    models = (
        CaseComponents,
        MultimodalVisible,
        MultimodalRegistration,
        StoredInput,
        InputPlan,
        InputCapabilities,
    )
    for model in models:
        path = directory / (model.__name__ + ".json")
        expected = json.dumps(model.model_json_schema(), sort_keys=True, indent=2) + "\n"
        if args.write:
            path.write_text(expected)
        elif not path.is_file() or path.read_text() != expected:
            raise SystemExit("multimodal_schema_drift:" + model.__name__)
    print(f"{len(models)} multimodal schemas verified")


if __name__ == "__main__":
    main()
