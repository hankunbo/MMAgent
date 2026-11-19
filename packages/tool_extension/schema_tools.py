"""Additive synthetic extension schema snapshots."""

import argparse
import json
from pathlib import Path

from .models import CatalogFixture, ExtensionGrant, ExtensionPlan, ExtensionPolicy, ExtensionResult


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    for model in (CatalogFixture, ExtensionGrant, ExtensionPlan, ExtensionPolicy, ExtensionResult):
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
