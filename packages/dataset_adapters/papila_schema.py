import argparse
import json
from pathlib import Path

from .papila_models import PapilaClinicalInput


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    path = Path(__file__).parent / "schemas/PapilaClinicalInput.schema.json"
    expected = json.dumps(PapilaClinicalInput.model_json_schema(), indent=2, sort_keys=True) + "\n"
    if args.write:
        path.write_text(expected)
    elif path.read_text() != expected:
        raise SystemExit("papila_schema_drift")
    print("1 PAPILA clinical preparation schema verified")


if __name__ == "__main__":
    main()
