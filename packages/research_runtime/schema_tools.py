"""Check the J domain schema without changing frozen contracts or H schemas."""

import argparse
import json
from pathlib import Path

from .models import ResearchRunContext
from .readiness import UnboundSingleCasePolicy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    for model in (ResearchRunContext, UnboundSingleCasePolicy):
        path = Path(__file__).parent / "schemas" / (model.__name__ + ".schema.json")
        content = (
            json.dumps(model.model_json_schema(), ensure_ascii=False, indent=2, sort_keys=True)
            + "\n"
        )
        if args.write:
            path.parent.mkdir(exist_ok=True)
            path.write_text(content)
        elif not path.exists() or path.read_text() != content:
            raise SystemExit("J owner schema drift")


if __name__ == "__main__":
    main()
