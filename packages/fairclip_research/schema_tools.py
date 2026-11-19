"""Generate/check the new output schema, leaving contracts v1 unchanged."""

import argparse
import json
from pathlib import Path

from .models import FairCLIPOutput
from .qc import FairCLIPQCPolicy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).parent / "schemas"
    for model in (FairCLIPOutput, FairCLIPQCPolicy):
        path = root / (model.__name__ + ".schema.json")
        data = json.dumps(model.model_json_schema(), indent=2, sort_keys=True) + "\n"
        if args.write:
            root.mkdir(exist_ok=True)
            path.write_text(data)
        elif not path.exists() or path.read_text() != data:
            raise SystemExit("fairclip_schema_drift")


if __name__ == "__main__":
    main()
