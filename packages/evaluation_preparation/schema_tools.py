import argparse
import json
from pathlib import Path

from .models import EvaluationPreparation, Registry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    for model in (EvaluationPreparation, Registry):
        path = Path(__file__).parent / "schemas" / (model.__name__ + ".json")
        expected = json.dumps(model.model_json_schema(), indent=2, sort_keys=True) + "\n"
        if args.write:
            path.parent.mkdir(exist_ok=True)
            path.write_text(expected)
        elif not path.exists() or path.read_text() != expected:
            raise SystemExit("evaluation_schema_drift")
    print("2 evaluation preparation schemas verified")


if __name__ == "__main__":
    main()
