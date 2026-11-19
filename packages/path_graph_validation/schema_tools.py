"""Exact additive schema snapshot, independent of frozen contracts v1."""

import argparse
import json
from pathlib import Path

from .models import PathGraphReview, PathGraphSummary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    for model in (PathGraphSummary, PathGraphReview):
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
