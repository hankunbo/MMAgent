"""Exact additive Schema snapshots for the actual simulation models."""

import argparse
import json
from pathlib import Path

from .models import Candidate, Intervention, SimulationPolicy, SimulationReport


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    for model in (Candidate, Intervention, SimulationPolicy, SimulationReport):
        path = Path(__file__).parent / "schemas" / (model.__name__ + ".schema.json")
        expected = json.dumps(model.model_json_schema(), ensure_ascii=False, indent=2) + "\n"
        if args.write:
            path.parent.mkdir(exist_ok=True)
            path.write_text(expected)
        elif path.read_text() != expected:
            raise SystemExit(model.__name__ + " schema drift")
        print(model.__name__ + " schema verified")


if __name__ == "__main__":
    main()
